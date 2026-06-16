#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=Path("database"))
    p.add_argument("--runs-dir", type=Path, default=Path("outputs/task2_runs"))
    p.add_argument("--out-dir", type=Path, default=Path("outputs/task2_gallery_aggregation_sweep"))
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--alphas", nargs="*", type=float, default=[0.0, 0.002, 0.005, 0.01, 0.02, 0.03])
    p.add_argument("--softmax-temperatures", nargs="*", type=float, default=[0.01, 0.03, 0.05, 0.1, 0.2])
    p.add_argument("--topks", nargs="*", type=int, default=[1, 3, 5, 10, 20])
    p.add_argument("--stable-weight", action="store_true", help="Also evaluate Fisher-style stable-dimension weighted embeddings.")
    return p.parse_args()


def l2(x: np.ndarray) -> np.ndarray:
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-8, None)


def metric(y_true: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    r1 = float(np.mean([y in row[:1] for y, row in zip(y_true, pred)]))
    r3 = float(np.mean([y in row[:3] for y, row in zip(y_true, pred)]))
    r5 = float(np.mean([y in row[:5] for y, row in zip(y_true, pred)]))
    return {"R@1": r1, "R@3": r3, "R@5": r5, "Score": 0.5 * r1 + 0.3 * r3 + 0.2 * r5}


def top5(scores: np.ndarray, ships: np.ndarray) -> np.ndarray:
    return ships[np.argsort(-scores, axis=1)[:, :5]]


def stable_dimension_weights(ref_emb: np.ndarray, ref_ship_ids: np.ndarray) -> np.ndarray:
    ship_means = []
    intra_vars = []
    for sid in sorted(set(ref_ship_ids.tolist())):
        x = ref_emb[ref_ship_ids == sid]
        ship_means.append(x.mean(axis=0))
        if len(x) > 1:
            intra_vars.append(x.var(axis=0))
    ship_means = np.vstack(ship_means)
    inter = ship_means.var(axis=0)
    intra = np.vstack(intra_vars).mean(axis=0) if intra_vars else np.ones_like(inter)
    weights = inter / (intra + 1e-6)
    weights = np.clip(weights, np.percentile(weights, 1), np.percentile(weights, 99))
    weights = weights / (weights.mean() + 1e-8)
    return weights.astype(np.float32)


def apply_weights(ref_emb: np.ndarray, query_emb: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scale = np.sqrt(weights).reshape(1, -1)
    return l2(ref_emb * scale), l2(query_emb * scale)


def ship_index(ref_ship_ids: np.ndarray) -> tuple[np.ndarray, list[np.ndarray], np.ndarray]:
    ships = np.array(sorted(set(ref_ship_ids.tolist())), dtype=int)
    groups = [np.where(ref_ship_ids == sid)[0] for sid in ships]
    counts = np.array([len(g) for g in groups], dtype=np.float32)
    return ships, groups, counts


def prototype_scores(query: np.ndarray, ref: np.ndarray, groups: list[np.ndarray]) -> np.ndarray:
    protos = np.vstack([l2(ref[idx].mean(axis=0, keepdims=True))[0] for idx in groups]).astype(np.float32)
    return query @ protos.T


def medoid_scores(query: np.ndarray, ref: np.ndarray, groups: list[np.ndarray]) -> np.ndarray:
    medoids = []
    for idx in groups:
        x = ref[idx]
        proto = l2(x.mean(axis=0, keepdims=True))
        local = int(np.argmax((x @ proto.T).reshape(-1)))
        medoids.append(x[local])
    medoids = np.vstack(medoids).astype(np.float32)
    return query @ medoids.T


def aggregate_from_clip_scores(sim: np.ndarray, groups: list[np.ndarray], method: str, value: float | int | None) -> np.ndarray:
    out = np.zeros((sim.shape[0], len(groups)), dtype=np.float32)
    for j, idx in enumerate(groups):
        vals = sim[:, idx]
        if method == "clip_max":
            out[:, j] = vals.max(axis=1)
        elif method == "clip_mean":
            out[:, j] = vals.mean(axis=1)
        elif method == "clip_median":
            out[:, j] = np.median(vals, axis=1)
        elif method == "topk_mean":
            k = min(int(value), vals.shape[1])
            part = np.partition(vals, kth=vals.shape[1] - k, axis=1)[:, -k:]
            out[:, j] = part.mean(axis=1)
        elif method == "softmax_pool":
            temperature = float(value)
            z = vals / max(temperature, 1e-6)
            z = z - z.max(axis=1, keepdims=True)
            w = np.exp(z)
            w = w / np.clip(w.sum(axis=1, keepdims=True), 1e-8, None)
            out[:, j] = (w * vals).sum(axis=1)
        elif method == "trimmed_tophalf_mean":
            k = max(1, vals.shape[1] // 2)
            part = np.partition(vals, kth=vals.shape[1] - k, axis=1)[:, -k:]
            out[:, j] = part.mean(axis=1)
        else:
            raise ValueError(method)
    return out


def score_variant(
    query: np.ndarray,
    ref: np.ndarray,
    groups: list[np.ndarray],
    counts: np.ndarray,
    method: str,
    value: float | int | None,
    alpha: float,
) -> np.ndarray:
    if method == "prototype":
        scores = prototype_scores(query, ref, groups)
    elif method == "medoid":
        scores = medoid_scores(query, ref, groups)
    else:
        sim = query @ ref.T
        scores = aggregate_from_clip_scores(sim, groups, method, value)
    if alpha:
        scores = scores - alpha * np.log1p(counts.reshape(1, -1))
    return scores.astype(np.float32)


def candidate_specs(topks: list[int], temperatures: list[float]) -> list[tuple[str, float | int | None]]:
    specs: list[tuple[str, float | int | None]] = [
        ("prototype", None),
        ("medoid", None),
        ("clip_max", None),
        ("clip_mean", None),
        ("clip_median", None),
        ("trimmed_tophalf_mean", None),
    ]
    specs.extend(("topk_mean", k) for k in topks)
    specs.extend(("softmax_pool", t) for t in temperatures)
    return specs


def prediction_stats(pred: np.ndarray) -> dict[str, float]:
    top1 = pred[:, 0]
    counts = pd.Series(top1).value_counts()
    return {
        "top1_unique_ships": int(counts.size),
        "top1_max_ship_fraction": float(counts.iloc[0] / len(top1)),
        "top1_entropy": float(-(counts / len(top1) * np.log((counts / len(top1)) + 1e-12)).sum()),
    }


def evaluate_run(
    run_name: str,
    ref: np.ndarray,
    query: np.ndarray,
    ref_ship_ids: np.ndarray,
    y: np.ndarray,
    alphas: list[float],
    specs: list[tuple[str, float | int | None]],
    feature_mode: str,
) -> list[dict[str, object]]:
    ships, groups, counts = ship_index(ref_ship_ids)
    rows = []
    for method, value in specs:
        for alpha in alphas:
            scores = score_variant(query, ref, groups, counts, method, value, alpha)
            pred = top5(scores, ships)
            row = {
                "run": run_name,
                "feature_mode": feature_mode,
                "method": method,
                "value": value,
                "alpha": alpha,
                **metric(y, pred),
                **prediction_stats(pred),
            }
            rows.append(row)
    return rows


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    gallery = pd.read_csv(args.data_dir / "task2_test/gallery.csv")
    val = pd.read_csv(args.data_dir / "task2_test/val.csv")
    ref_ship_ids = gallery["ship_id"].astype(int).to_numpy()
    y = val["ship_id"].astype(int).to_numpy()
    specs = candidate_specs(args.topks, args.softmax_temperatures)

    all_rows = []
    for run in args.runs:
        run_dir = args.runs_dir / run
        ref = l2(np.load(run_dir / "reference_embeddings.npy").astype(np.float32))
        query = l2(np.load(run_dir / "val_embeddings.npy").astype(np.float32))
        if len(ref) != len(gallery) or len(query) != len(val):
            raise ValueError(f"{run}: embedding shape mismatch")
        print(json.dumps({"run": run, "feature_mode": "base", "ref": ref.shape, "query": query.shape}, ensure_ascii=False), flush=True)
        rows = evaluate_run(run, ref, query, ref_ship_ids, y, args.alphas, specs, "base")
        all_rows.extend(rows)
        print(json.dumps(max(rows, key=lambda r: r["Score"]), ensure_ascii=False), flush=True)

        if args.stable_weight:
            weights = stable_dimension_weights(ref, ref_ship_ids)
            weighted_ref, weighted_query = apply_weights(ref, query, weights)
            np.save(args.out_dir / f"{run}__stable_dimension_weights.npy", weights)
            print(json.dumps({"run": run, "feature_mode": "stable_weighted", "weight_min": float(weights.min()), "weight_max": float(weights.max())}, ensure_ascii=False), flush=True)
            rows = evaluate_run(run, weighted_ref, weighted_query, ref_ship_ids, y, args.alphas, specs, "stable_weighted")
            all_rows.extend(rows)
            print(json.dumps(max(rows, key=lambda r: r["Score"]), ensure_ascii=False), flush=True)

    results = pd.DataFrame(all_rows).sort_values(["Score", "R@1", "R@3"], ascending=False)
    results.to_csv(args.out_dir / "aggregation_sweep_scores.csv", index=False)
    best_by_run = (
        results.sort_values(["run", "feature_mode", "Score"], ascending=[True, True, False])
        .groupby(["run", "feature_mode"], as_index=False)
        .head(1)
        .sort_values("Score", ascending=False)
    )
    best_by_run.to_csv(args.out_dir / "aggregation_sweep_best_by_run.csv", index=False)
    best_by_method = (
        results.sort_values(["method", "value", "Score"], ascending=[True, True, False])
        .groupby(["method", "value"], dropna=False, as_index=False)
        .head(1)
        .sort_values("Score", ascending=False)
    )
    best_by_method.to_csv(args.out_dir / "aggregation_sweep_best_by_method.csv", index=False)
    summary = {
        "runs": args.runs,
        "num_candidates": int(len(results)),
        "best_overall": results.iloc[0].to_dict(),
        "outputs": {
            "scores": str(args.out_dir / "aggregation_sweep_scores.csv"),
            "best_by_run": str(args.out_dir / "aggregation_sweep_best_by_run.csv"),
            "best_by_method": str(args.out_dir / "aggregation_sweep_best_by_method.csv"),
        },
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"best_overall": summary["best_overall"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
