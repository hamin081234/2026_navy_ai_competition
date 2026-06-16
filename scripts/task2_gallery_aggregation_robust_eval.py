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
    p.add_argument("--out-dir", type=Path, default=Path("outputs/task2_gallery_aggregation_sweep/robust_eval"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--holdout-frac", type=float, default=0.2)
    return p.parse_args()


def add_time_columns(gallery: pd.DataFrame) -> pd.DataFrame:
    out = gallery.copy().reset_index(drop=True)
    ts = pd.to_datetime(out["ais_timestamp"], utc=True, format="mixed")
    out["_timestamp"] = ts
    out["_month_key"] = ts.dt.strftime("%Y-%m")
    return out


def normalize_scores(scores: np.ndarray) -> np.ndarray:
    return (scores - scores.mean(axis=1, keepdims=True)) / (scores.std(axis=1, keepdims=True) + 1e-6)


def official_val_score(spec: dict, gallery: pd.DataFrame, val: pd.DataFrame, runs_dir: Path, weights_dir: Path):
    ref_ship_ids = gallery["ship_id"].astype(int).to_numpy()
    ships, groups, counts = agg.ship_index(ref_ship_ids)
    run_dir = runs_dir / spec["run"]
    ref = agg.l2(np.load(run_dir / "reference_embeddings.npy").astype(np.float32))
    query = agg.l2(np.load(run_dir / "val_embeddings.npy").astype(np.float32))
    if spec["feature_mode"] == "stable_weighted":
        weights_path = weights_dir / f"{spec['run']}__stable_dimension_weights.npy"
        weights = np.load(weights_path) if weights_path.exists() else agg.stable_dimension_weights(ref, ref_ship_ids)
        ref, query = agg.apply_weights(ref, query, weights)
    scores = agg.score_variant(query, ref, groups, counts, spec["method"], spec.get("value"), float(spec["alpha"]))
    pred = agg.top5(scores, ships)
    y = val["ship_id"].astype(int).to_numpy()
    return {**agg.metric(y, pred), **agg.prediction_stats(pred)}


def score_split(spec: dict, emb: np.ndarray, gallery: pd.DataFrame, ref_idx: np.ndarray, query_idx: np.ndarray):
    ref_df = gallery.iloc[ref_idx].reset_index(drop=True)
    query_df = gallery.iloc[query_idx].reset_index(drop=True)
    ref_ship_ids = ref_df["ship_id"].astype(int).to_numpy()
    ships, groups, counts = agg.ship_index(ref_ship_ids)
    ref = agg.l2(emb[ref_idx].astype(np.float32))
    query = agg.l2(emb[query_idx].astype(np.float32))
    if spec["feature_mode"] == "stable_weighted":
        weights = agg.stable_dimension_weights(ref, ref_ship_ids)
        ref, query = agg.apply_weights(ref, query, weights)
    scores = agg.score_variant(query, ref, groups, counts, spec["method"], spec.get("value"), float(spec["alpha"]))
    pred = agg.top5(scores, ships)
    y = query_df["ship_id"].astype(int).to_numpy()
    return scores, ships, {**agg.metric(y, pred), **agg.prediction_stats(pred)}


def candidate_specs() -> list[dict[str, object]]:
    # Include the validation-best neighborhood and stronger penalties to test diversity/robustness.
    specs: list[dict[str, object]] = []
    for alpha in [0.02, 0.03, 0.04, 0.05, 0.07, 0.10]:
        specs.append({
            "name": f"trainonly_stable_softmax001_a{alpha:g}",
            "run": "quick_target100_trainonly_mel96_f16000",
            "feature_mode": "stable_weighted",
            "method": "softmax_pool",
            "value": 0.01,
            "alpha": alpha,
        })
        specs.append({
            "name": f"quick_stable_clipmax_a{alpha:g}",
            "run": "quick_target100_mel96_f16000",
            "feature_mode": "stable_weighted",
            "method": "clip_max",
            "value": None,
            "alpha": alpha,
        })
    specs.extend([
        {
            "name": "projection_stable_proto_a003",
            "run": "projection_clip_trainonly",
            "feature_mode": "stable_weighted",
            "method": "prototype",
            "value": None,
            "alpha": 0.03,
        },
        {
            "name": "robust_stable_topk20_a0",
            "run": "robust_rows100_seed123_ep2_aug",
            "feature_mode": "stable_weighted",
            "method": "topk_mean",
            "value": 20,
            "alpha": 0.0,
        },
    ])
    return specs


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    gallery = add_time_columns(pd.read_csv(args.data_dir / "task2_test/gallery.csv"))
    val = pd.read_csv(args.data_dir / "task2_test/val.csv")
    split_modes = ["random20", "latest20", "earliest20", "month_latest", "ais_group_random20"]
    specs = candidate_specs()
    weights_dir = Path("outputs/task2_gallery_aggregation_sweep/core5")

    embeddings: dict[str, np.ndarray] = {}
    for spec in specs:
        run = str(spec["run"])
        if run not in embeddings:
            embeddings[run] = agg.l2(np.load(args.runs_dir / run / "reference_embeddings.npy").astype(np.float32))

    rows = []
    split_rows = []
    for split in split_modes:
        ref_idx, query_idx = per_ship_holdout_indices(gallery, split, args.holdout_frac, args.seed)
        split_rows.append({
            "split": split,
            "num_ref": int(len(ref_idx)),
            "num_query": int(len(query_idx)),
            "ref_ships": int(gallery.iloc[ref_idx]["ship_id"].nunique()),
            "query_ships": int(gallery.iloc[query_idx]["ship_id"].nunique()),
        })
        for spec in specs:
            _, _, m = score_split(spec, embeddings[str(spec["run"])], gallery, ref_idx, query_idx)
            row = {
                "candidate": spec["name"],
                "split": split,
                **{k: v for k, v in spec.items() if k != "name"},
                "num_ref": int(len(ref_idx)),
                "num_query": int(len(query_idx)),
                **m,
            }
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)

    scores = pd.DataFrame(rows)
    scores.to_csv(args.out_dir / "aggregation_robust_split_scores.csv", index=False)
    pd.DataFrame(split_rows).to_csv(args.out_dir / "aggregation_robust_splits.csv", index=False)

    official_rows = []
    for spec in specs:
        official = official_val_score(spec, gallery, val, args.runs_dir, weights_dir)
        official_rows.append({"candidate": spec["name"], **{k: v for k, v in spec.items() if k != "name"}, **official})
    official_df = pd.DataFrame(official_rows)
    official_df.to_csv(args.out_dir / "aggregation_official_val_scores.csv", index=False)

    summary = (
        scores.groupby("candidate", as_index=False)
        .agg(
            robust_mean_score=("Score", "mean"),
            robust_min_score=("Score", "min"),
            robust_std_score=("Score", "std"),
            robust_mean_r1=("R@1", "mean"),
            robust_mean_r3=("R@3", "mean"),
            robust_mean_r5=("R@5", "mean"),
            robust_mean_top1_unique=("top1_unique_ships", "mean"),
            robust_mean_top1_max_fraction=("top1_max_ship_fraction", "mean"),
        )
        .merge(
            official_df[["candidate", "R@1", "R@3", "R@5", "Score", "top1_unique_ships", "top1_max_ship_fraction"]].rename(
                columns={
                    "R@1": "official_R@1",
                    "R@3": "official_R@3",
                    "R@5": "official_R@5",
                    "Score": "official_Score",
                    "top1_unique_ships": "official_top1_unique",
                    "top1_max_ship_fraction": "official_top1_max_fraction",
                }
            ),
            on="candidate",
            how="left",
        )
    )
    summary["selection_score"] = summary["official_Score"] + summary["robust_mean_score"] - 0.5 * summary["official_top1_max_fraction"]
    summary = summary.sort_values(["robust_mean_score", "official_Score"], ascending=False)
    summary.to_csv(args.out_dir / "aggregation_robust_summary.csv", index=False)

    best = summary.iloc[0].to_dict()
    out_summary = {
        "splits": split_rows,
        "best_by_robust_mean": best,
        "outputs": {
            "split_scores": str(args.out_dir / "aggregation_robust_split_scores.csv"),
            "official_scores": str(args.out_dir / "aggregation_official_val_scores.csv"),
            "summary": str(args.out_dir / "aggregation_robust_summary.csv"),
        },
    }
    (args.out_dir / "summary.json").write_text(json.dumps(out_summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"best_by_robust_mean": best}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
