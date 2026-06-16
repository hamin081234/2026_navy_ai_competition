#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import sys

sys.path.append(str(Path(__file__).resolve().parent))
import task2_gallery_aggregation_sweep as agg  # noqa: E402
from task2_robust_holdout_eval import per_ship_holdout_indices  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=Path("database"))
    p.add_argument("--runs-dir", type=Path, default=Path("outputs/task2_runs"))
    p.add_argument("--out-dir", type=Path, default=Path("outputs/task2_balanced_ensemble"))
    p.add_argument("--step", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--holdout-frac", type=float, default=0.2)
    return p.parse_args()


def add_time_columns(gallery: pd.DataFrame) -> pd.DataFrame:
    out = gallery.copy().reset_index(drop=True)
    ts = pd.to_datetime(out["ais_timestamp"], utc=True, format="mixed")
    out["_timestamp"] = ts
    out["_month_key"] = ts.dt.strftime("%Y-%m")
    return out


def normalize(scores: np.ndarray) -> np.ndarray:
    return (scores - scores.mean(axis=1, keepdims=True)) / (scores.std(axis=1, keepdims=True) + 1e-6)


def simplex(n: int, step: float):
    units = int(round(1.0 / step))

    def rec(prefix, remaining, slots):
        if slots == 1:
            yield tuple(prefix + [remaining / units])
            return
        for value in range(remaining + 1):
            yield from rec(prefix + [value / units], remaining - value, slots - 1)

    yield from rec([], units, n)


def candidate_specs() -> list[dict[str, object]]:
    return [
        {
            "name": "val_trainonly_softmax001_a003",
            "run": "quick_target100_trainonly_mel96_f16000",
            "feature_mode": "stable_weighted",
            "method": "softmax_pool",
            "value": 0.01,
            "alpha": 0.03,
        },
        {
            "name": "val_quick_clipmax_a003",
            "run": "quick_target100_mel96_f16000",
            "feature_mode": "stable_weighted",
            "method": "clip_max",
            "value": None,
            "alpha": 0.03,
        },
        {
            "name": "robust_topk20_a0",
            "run": "robust_rows100_seed123_ep2_aug",
            "feature_mode": "stable_weighted",
            "method": "topk_mean",
            "value": 20,
            "alpha": 0.0,
        },
        {
            "name": "projection_proto_a003",
            "run": "projection_clip_trainonly",
            "feature_mode": "stable_weighted",
            "method": "prototype",
            "value": None,
            "alpha": 0.03,
        },
        {
            "name": "frozen_clip_proto_a002",
            "run": "frozen_clip_vit_b32_logmel96_f16000",
            "feature_mode": "base",
            "method": "prototype",
            "value": None,
            "alpha": 0.02,
        },
    ]


def load_weighted_embeddings(spec: dict[str, object], split: str, gallery: pd.DataFrame, runs_dir: Path, weights_dir: Path):
    run = str(spec["run"])
    run_dir = runs_dir / run
    ref_ship_ids = gallery["ship_id"].astype(int).to_numpy()
    ref = agg.l2(np.load(run_dir / "reference_embeddings.npy").astype(np.float32))
    query = agg.l2(np.load(run_dir / f"{split}_embeddings.npy").astype(np.float32))
    if spec["feature_mode"] == "stable_weighted":
        weights_path = weights_dir / f"{run}__stable_dimension_weights.npy"
        weights = np.load(weights_path) if weights_path.exists() else agg.stable_dimension_weights(ref, ref_ship_ids)
        ref, query = agg.apply_weights(ref, query, weights)
    return ref, query


def score_full_split(spec: dict[str, object], split: str, gallery: pd.DataFrame, runs_dir: Path, weights_dir: Path):
    ref_ship_ids = gallery["ship_id"].astype(int).to_numpy()
    ships, groups, counts = agg.ship_index(ref_ship_ids)
    ref, query = load_weighted_embeddings(spec, split, gallery, runs_dir, weights_dir)
    scores = agg.score_variant(query, ref, groups, counts, str(spec["method"]), spec.get("value"), float(spec["alpha"]))
    return scores.astype(np.float32), ships


def score_gallery_holdout(spec: dict[str, object], emb: np.ndarray, gallery: pd.DataFrame, ref_idx: np.ndarray, query_idx: np.ndarray):
    ref_df = gallery.iloc[ref_idx].reset_index(drop=True)
    ref_ship_ids = ref_df["ship_id"].astype(int).to_numpy()
    ships, groups, counts = agg.ship_index(ref_ship_ids)
    ref = agg.l2(emb[ref_idx].astype(np.float32))
    query = agg.l2(emb[query_idx].astype(np.float32))
    if spec["feature_mode"] == "stable_weighted":
        weights = agg.stable_dimension_weights(ref, ref_ship_ids)
        ref, query = agg.apply_weights(ref, query, weights)
    scores = agg.score_variant(query, ref, groups, counts, str(spec["method"]), spec.get("value"), float(spec["alpha"]))
    return scores.astype(np.float32), ships


def metric_from_scores(scores: np.ndarray, ships: np.ndarray, y: np.ndarray) -> dict[str, float]:
    pred = agg.top5(scores, ships)
    return {**agg.metric(y, pred), **agg.prediction_stats(pred)}


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    weights_dir = Path("outputs/task2_gallery_aggregation_sweep/core5")
    specs = candidate_specs()
    gallery_raw = pd.read_csv(args.data_dir / "task2_test/gallery.csv")
    gallery = add_time_columns(gallery_raw)
    val = pd.read_csv(args.data_dir / "task2_test/val.csv")
    test = pd.read_csv(args.data_dir / "task2_test/test.csv")
    sample = pd.read_csv(args.data_dir / "sample_submission_task2.csv")
    y_val = val["ship_id"].astype(int).to_numpy()

    # Full official validation/test score matrices.
    val_scores = []
    test_scores = []
    ships_ref = None
    input_rows = []
    for spec in specs:
        vs, ships = score_full_split(spec, "val", gallery_raw, args.runs_dir, weights_dir)
        ts, ships_t = score_full_split(spec, "test", gallery_raw, args.runs_dir, weights_dir)
        if ships_ref is None:
            ships_ref = ships
        elif not np.array_equal(ships_ref, ships) or not np.array_equal(ships_ref, ships_t):
            raise RuntimeError("ship order mismatch")
        val_scores.append(normalize(vs))
        test_scores.append(normalize(ts))
        input_rows.append({"candidate": spec["name"], **{k: v for k, v in spec.items() if k != "name"}, **metric_from_scores(vs, ships, y_val)})

    pd.DataFrame(input_rows).to_csv(args.out_dir / "balanced_ensemble_inputs.csv", index=False)

    # Robust holdout score matrices per split.
    split_modes = ["random20", "latest20", "earliest20", "month_latest", "ais_group_random20"]
    run_embeddings: dict[str, np.ndarray] = {
        str(spec["run"]): agg.l2(np.load(args.runs_dir / str(spec["run"]) / "reference_embeddings.npy").astype(np.float32))
        for spec in specs
    }
    split_score_bank: dict[str, tuple[list[np.ndarray], np.ndarray, np.ndarray]] = {}
    split_meta = []
    for split in split_modes:
        ref_idx, query_idx = per_ship_holdout_indices(gallery, split, args.holdout_frac, args.seed)
        y = gallery.iloc[query_idx]["ship_id"].astype(int).to_numpy()
        split_scores = []
        split_ships = None
        for spec in specs:
            scores, ships = score_gallery_holdout(spec, run_embeddings[str(spec["run"])], gallery, ref_idx, query_idx)
            if split_ships is None:
                split_ships = ships
            elif not np.array_equal(split_ships, ships):
                raise RuntimeError(f"ship order mismatch in {split}")
            split_scores.append(normalize(scores))
        split_score_bank[split] = (split_scores, split_ships, y)
        split_meta.append({
            "split": split,
            "num_ref": int(len(ref_idx)),
            "num_query": int(len(query_idx)),
            "ref_ships": int(gallery.iloc[ref_idx]["ship_id"].nunique()),
            "query_ships": int(gallery.iloc[query_idx]["ship_id"].nunique()),
        })
    pd.DataFrame(split_meta).to_csv(args.out_dir / "balanced_ensemble_splits.csv", index=False)

    rows = []
    split_rows = []
    for weights in simplex(len(specs), args.step):
        val_combo = sum(w * s for w, s in zip(weights, val_scores))
        val_metrics = metric_from_scores(val_combo, ships_ref, y_val)
        robust_scores = []
        robust_min = None
        robust_top1_max = []
        for split, (score_list, split_ships, y_split) in split_score_bank.items():
            combo = sum(w * s for w, s in zip(weights, score_list))
            m = metric_from_scores(combo, split_ships, y_split)
            robust_scores.append(float(m["Score"]))
            robust_top1_max.append(float(m["top1_max_ship_fraction"]))
            split_rows.append({"weights": str(weights), "split": split, **m})
        row = {
            "weights": str(weights),
            "official_R@1": val_metrics["R@1"],
            "official_R@3": val_metrics["R@3"],
            "official_R@5": val_metrics["R@5"],
            "official_Score": val_metrics["Score"],
            "official_top1_unique": val_metrics["top1_unique_ships"],
            "official_top1_max_fraction": val_metrics["top1_max_ship_fraction"],
            "official_top1_entropy": val_metrics["top1_entropy"],
            "robust_mean_score": float(np.mean(robust_scores)),
            "robust_min_score": float(np.min(robust_scores)),
            "robust_std_score": float(np.std(robust_scores)),
            "robust_mean_top1_max_fraction": float(np.mean(robust_top1_max)),
        }
        row["balanced_score"] = (
            row["official_Score"]
            + row["robust_mean_score"]
            + 0.5 * row["robust_min_score"]
            - 0.15 * row["official_top1_max_fraction"]
        )
        row["conservative_score"] = min(row["official_Score"], row["robust_mean_score"]) + 0.5 * row["robust_min_score"]
        rows.append(row)

    leaderboard = pd.DataFrame(rows).sort_values(["balanced_score", "official_Score"], ascending=False)
    leaderboard.to_csv(args.out_dir / "balanced_ensemble_leaderboard.csv", index=False)
    pd.DataFrame(split_rows).to_csv(args.out_dir / "balanced_ensemble_split_scores.csv", index=False)

    # Export three useful submissions: balanced objective, robust objective, official objective.
    selected = {
        "balanced": leaderboard.iloc[0],
        "official": leaderboard.sort_values(["official_Score", "robust_mean_score"], ascending=False).iloc[0],
        "robust": leaderboard.sort_values(["robust_mean_score", "official_Score"], ascending=False).iloc[0],
        "conservative": leaderboard.sort_values(["conservative_score", "official_Score"], ascending=False).iloc[0],
    }
    submission_rows = []
    for label, row in selected.items():
        weights = tuple(float(x.strip()) for x in row["weights"].strip("()").split(","))
        test_combo = sum(w * s for w, s in zip(weights, test_scores))
        pred = agg.top5(test_combo, ships_ref)
        sub = sample.copy()
        pred_map = dict(zip(test["filename"], [",".join(map(str, p)) for p in pred]))
        sub["top5_ship_ids"] = sub["filename"].map(pred_map)
        path = args.out_dir / f"submission_task2_{label}_balanced_ensemble.csv"
        sub.to_csv(path, index=False)
        submission_rows.append({"label": label, "submission": str(path), **row.to_dict()})
    pd.DataFrame(submission_rows).to_csv(args.out_dir / "balanced_ensemble_selected_submissions.csv", index=False)

    summary = {
        "candidates": specs,
        "best_balanced": selected["balanced"].to_dict(),
        "best_official": selected["official"].to_dict(),
        "best_robust": selected["robust"].to_dict(),
        "best_conservative": selected["conservative"].to_dict(),
        "outputs": {
            "leaderboard": str(args.out_dir / "balanced_ensemble_leaderboard.csv"),
            "split_scores": str(args.out_dir / "balanced_ensemble_split_scores.csv"),
            "selected_submissions": str(args.out_dir / "balanced_ensemble_selected_submissions.csv"),
        },
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
