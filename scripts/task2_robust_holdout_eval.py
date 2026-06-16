#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import sys

sys.path.append(str(Path(__file__).resolve().parent))
import task2_subprototype_eval as spe  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=Path("database"))
    p.add_argument("--runs-dir", type=Path, default=Path("outputs/task2_runs"))
    p.add_argument("--out-dir", type=Path, default=Path("outputs/task2_robust_holdout"))
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--holdout-frac", type=float, default=0.2)
    p.add_argument("--methods", nargs="*", default=None)
    return p.parse_args()


def per_ship_holdout_indices(
    gallery: pd.DataFrame,
    mode: str,
    frac: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    ref_parts = []
    query_parts = []
    df = gallery.copy()
    ts = pd.to_datetime(df["ais_timestamp"], utc=True, format="mixed")
    df["_timestamp"] = ts
    df["_month_key"] = ts.dt.strftime("%Y-%m")
    for _, group in df.groupby("ship_id", sort=True):
        idx = group.index.to_numpy()
        n = len(idx)
        if n < 2:
            ref_parts.append(idx)
            continue
        q = max(1, int(round(n * frac)))
        q = min(q, n - 1)
        if mode == "random20":
            q_idx = rng.choice(idx, size=q, replace=False)
        elif mode == "latest20":
            q_idx = group.sort_values("_timestamp").tail(q).index.to_numpy()
        elif mode == "earliest20":
            q_idx = group.sort_values("_timestamp").head(q).index.to_numpy()
        elif mode == "month_latest":
            months = group.sort_values("_timestamp")["_month_key"].drop_duplicates().tolist()
            q_idx = np.array([], dtype=int)
            for month in reversed(months):
                candidate = group[group["_month_key"] == month].index.to_numpy()
                if len(candidate) < n:
                    q_idx = candidate
                    break
            if len(q_idx) == 0:
                q_idx = group.sort_values("_timestamp").tail(q).index.to_numpy()
        elif mode == "ais_group_random20":
            group_keys = group["ais_timestamp"].astype(str).drop_duplicates().to_numpy()
            rng.shuffle(group_keys)
            chosen = []
            for key in group_keys:
                candidate = group[group["ais_timestamp"].astype(str) == key].index.to_numpy()
                if len(np.concatenate(chosen + [candidate])) >= q and n - len(np.concatenate(chosen + [candidate])) >= 1:
                    chosen.append(candidate)
                    break
                if n - len(np.concatenate(chosen + [candidate])) >= 1:
                    chosen.append(candidate)
            q_idx = np.concatenate(chosen) if chosen else group.sort_values("_timestamp").tail(q).index.to_numpy()
        else:
            raise ValueError(mode)
        q_idx = np.unique(q_idx)
        if len(q_idx) >= n:
            q_idx = q_idx[: n - 1]
        r_idx = np.setdiff1d(idx, q_idx, assume_unique=False)
        ref_parts.append(r_idx)
        query_parts.append(q_idx)
    ref_idx = np.concatenate(ref_parts).astype(int)
    query_idx = np.concatenate(query_parts).astype(int)
    return ref_idx, query_idx


def metric(y_true: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    r1 = float(np.mean([y in row[:1] for y, row in zip(y_true, pred)]))
    r3 = float(np.mean([y in row[:3] for y, row in zip(y_true, pred)]))
    r5 = float(np.mean([y in row[:5] for y, row in zip(y_true, pred)]))
    return {"R@1": r1, "R@3": r3, "R@5": r5, "Score": 0.5 * r1 + 0.3 * r3 + 0.2 * r5}


def evaluate_one(
    run_name: str,
    split_name: str,
    method: str,
    ref_df: pd.DataFrame,
    query_df: pd.DataFrame,
    ref_emb: np.ndarray,
    query_emb: np.ndarray,
    seed: int,
) -> list[dict[str, object]]:
    proto_emb, proto_ship_ids, proto_meta = spe.make_prototypes(method, ref_emb, ref_df, seed)
    proto_source_clips = proto_meta["source_clips"].astype(float).to_numpy()
    y = query_df["ship_id"].astype(int).to_numpy()
    rows = []
    for alpha in [0.0, 0.005, 0.01, 0.02, 0.03]:
        scores, ships = spe.score_by_ship(query_emb, proto_emb, proto_ship_ids, proto_source_clips, alpha)
        pred = spe.top5_from_scores(scores, ships)
        rows.append({
            "run": run_name,
            "split": split_name,
            "method": method,
            "alpha": alpha,
            "num_ref": int(len(ref_df)),
            "num_query": int(len(query_df)),
            "num_prototypes": int(len(proto_ship_ids)),
            **metric(y, pred),
        })
    return rows


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    methods = args.methods or ["ship_centroid", "clip_max", "kmeans_k5", "kmeans_adaptive", "time_hour4", "ais_moving", "ais_speed_bin"]
    split_modes = ["random20", "latest20", "earliest20", "month_latest", "ais_group_random20"]
    gallery = spe.add_bins(pd.read_csv(args.data_dir / "task2_test/gallery.csv")).reset_index(drop=True)

    rows = []
    split_rows = []
    for mode in split_modes:
        ref_idx, query_idx = per_ship_holdout_indices(gallery, mode, args.holdout_frac, args.seed)
        split_rows.append({
            "split": mode,
            "num_ref": int(len(ref_idx)),
            "num_query": int(len(query_idx)),
            "query_ships": int(gallery.iloc[query_idx]["ship_id"].nunique()),
            "ref_ships": int(gallery.iloc[ref_idx]["ship_id"].nunique()),
        })
        for run in args.runs:
            run_dir = args.runs_dir / run
            emb = spe.l2(np.load(run_dir / "reference_embeddings.npy").astype(np.float32))
            if len(emb) != len(gallery):
                raise ValueError(f"{run}: gallery embedding row mismatch")
            ref_df = gallery.iloc[ref_idx].copy().reset_index(drop=True)
            query_df = gallery.iloc[query_idx].copy().reset_index(drop=True)
            ref_emb = emb[ref_idx]
            query_emb = emb[query_idx]
            print(json.dumps({"split": mode, "run": run, "ref": len(ref_df), "query": len(query_df)}, ensure_ascii=False), flush=True)
            for method in methods:
                rows.extend(evaluate_one(run, mode, method, ref_df, query_df, ref_emb, query_emb, args.seed))

    results = pd.DataFrame(rows)
    results.to_csv(args.out_dir / "robust_holdout_scores.csv", index=False)
    pd.DataFrame(split_rows).to_csv(args.out_dir / "robust_holdout_splits.csv", index=False)
    best_by = (
        results.sort_values(["run", "split", "method", "Score"], ascending=[True, True, True, False])
        .groupby(["run", "split", "method"], as_index=False)
        .head(1)
    )
    best_by.to_csv(args.out_dir / "robust_holdout_best_by_method.csv", index=False)
    agg = (
        best_by.groupby(["run", "method"], as_index=False)
        .agg(
            robust_mean_score=("Score", "mean"),
            robust_min_score=("Score", "min"),
            robust_std_score=("Score", "std"),
            mean_r1=("R@1", "mean"),
            mean_r3=("R@3", "mean"),
            mean_r5=("R@5", "mean"),
        )
        .sort_values("robust_mean_score", ascending=False)
    )
    agg.to_csv(args.out_dir / "robust_holdout_summary.csv", index=False)
    summary = {
        "runs": args.runs,
        "methods": methods,
        "splits": split_rows,
        "best_by_robust_mean": agg.iloc[0].to_dict(),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"best_by_robust_mean": summary["best_by_robust_mean"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
