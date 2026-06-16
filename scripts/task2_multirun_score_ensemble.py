#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

import sys

sys.path.append(str(Path(__file__).resolve().parent))
import task2_gallery_aggregation_sweep as agg  # noqa: E402
from task2_ais_group_pooling_sweep import add_group_columns, group_score_matrix, stable_weight  # noqa: E402
from task2_robust_holdout_eval import per_ship_holdout_indices  # noqa: E402


CANDIDATES = [
    {
        "name": "r100s123_conservative",
        "run": "robust_rows100_seed123_ep2_aug",
        "weights": {"clip20": 0.85, "ais5": 0.05, "ais10": 0.10},
    },
    {
        "name": "r80s123_balanced",
        "run": "robust_rows80_seed123_ep2_aug",
        "weights": {"clip20": 0.85, "ais5": 0.15, "ais10": 0.00},
    },
    {
        "name": "r100s777_conservative",
        "run": "robust_rows100_seed777_ep2_aug",
        "weights": {"clip20": 0.90, "ais5": 0.05, "ais10": 0.05},
    },
    {
        "name": "r100s123_val086",
        "run": "robust_rows100_seed123_ep2_aug",
        "weights": {"clip20": 0.10, "ais5": 0.40, "ais10": 0.50},
    },
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=Path("database"))
    p.add_argument("--runs-dir", type=Path, default=Path("outputs/task2_runs"))
    p.add_argument("--out-dir", type=Path, default=Path("outputs/task2_ais_group_pooling/multirun_score_ensemble"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--holdout-frac", type=float, default=0.2)
    p.add_argument("--grid-units", type=int, default=20)
    return p.parse_args()


def normalize(scores: np.ndarray) -> np.ndarray:
    return (scores - scores.mean(axis=1, keepdims=True)) / (scores.std(axis=1, keepdims=True) + 1e-6)


def topk_clip_scores(query: np.ndarray, ref: np.ndarray, gallery: pd.DataFrame, topk: int = 20):
    ships, groups, counts = agg.ship_index(gallery["ship_id"].astype(int).to_numpy())
    scores = agg.score_variant(query, ref, groups, counts, "topk_mean", topk, 0.0)
    return scores, ships


def ais_scores(query: np.ndarray, ref: np.ndarray, gallery: pd.DataFrame, k: int):
    return group_score_matrix(
        query=query,
        ref=ref,
        gallery=gallery,
        group_col="ais_exact",
        group_reduction="max",
        ship_reduction="topk_mean",
        topk=k,
        alpha=0.0,
        group_size_beta=0.0,
    )


def metric(scores: np.ndarray, ships: np.ndarray, y: np.ndarray):
    pred = agg.top5(scores, ships)
    return {**agg.metric(y, pred), **agg.prediction_stats(pred)}


def run_components(run_dir: Path, gallery: pd.DataFrame, val_emb_raw: np.ndarray | None = None):
    ref_raw = agg.l2(np.load(run_dir / "reference_embeddings.npy").astype(np.float32))
    val_raw = agg.l2(np.load(run_dir / "val_embeddings.npy").astype(np.float32)) if val_emb_raw is None else val_emb_raw
    test_raw = agg.l2(np.load(run_dir / "test_embeddings.npy").astype(np.float32))
    ship_ids = gallery["ship_id"].astype(int).to_numpy()
    weights = agg.stable_dimension_weights(ref_raw, ship_ids)
    ref = agg.l2(ref_raw * np.sqrt(weights).reshape(1, -1))
    val_emb = agg.l2(val_raw * np.sqrt(weights).reshape(1, -1))
    test_emb = agg.l2(test_raw * np.sqrt(weights).reshape(1, -1))

    val_clip, ships = topk_clip_scores(val_emb, ref, gallery, 20)
    val_ais5, ships5 = ais_scores(val_emb, ref, gallery, 5)
    val_ais10, ships10 = ais_scores(val_emb, ref, gallery, 10)
    if not np.array_equal(ships, ships5) or not np.array_equal(ships, ships10):
        raise RuntimeError(f"ship order mismatch in {run_dir}")
    test_clip, _ = topk_clip_scores(test_emb, ref, gallery, 20)
    test_ais5, _ = ais_scores(test_emb, ref, gallery, 5)
    test_ais10, _ = ais_scores(test_emb, ref, gallery, 10)
    components = {
        "clip20": (normalize(val_clip), normalize(test_clip)),
        "ais5": (normalize(val_ais5), normalize(test_ais5)),
        "ais10": (normalize(val_ais10), normalize(test_ais10)),
    }
    return ref_raw, components, ships


def candidate_score(components: dict[str, tuple[np.ndarray, np.ndarray]], weights: dict[str, float], idx: int):
    return normalize(sum(weights[name] * components[name][idx] for name in weights))


def compositions(n: int, units: int):
    if n == 1:
        yield (units,)
        return
    for i in range(units + 1):
        for rest in compositions(n - 1, units - i):
            yield (i, *rest)


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    gallery = add_group_columns(pd.read_csv(args.data_dir / "task2_test/gallery.csv"))
    val = pd.read_csv(args.data_dir / "task2_test/val.csv")
    test = pd.read_csv(args.data_dir / "task2_test/test.csv")
    sample = pd.read_csv(args.data_dir / "sample_submission_task2.csv")
    val_sample = val[["filename"]].copy()
    y_val = val["ship_id"].astype(int).to_numpy()

    run_cache = {}
    val_scores = {}
    test_scores = {}
    ships = None
    for cand in CANDIDATES:
        run = cand["run"]
        if run not in run_cache:
            run_cache[run] = run_components(args.runs_dir / run, gallery)
        _, comps, run_ships = run_cache[run]
        if ships is None:
            ships = run_ships
        elif not np.array_equal(ships, run_ships):
            raise RuntimeError("global ship order mismatch")
        val_scores[cand["name"]] = candidate_score(comps, cand["weights"], 0)
        test_scores[cand["name"]] = candidate_score(comps, cand["weights"], 1)

    split_modes = ["random20", "latest20", "earliest20", "month_latest", "ais_group_random20"]
    split_bank = {}
    split_meta = []
    for split in split_modes:
        ref_idx, query_idx = per_ship_holdout_indices(gallery, split, args.holdout_frac, args.seed)
        ref_df = gallery.iloc[ref_idx].copy().reset_index(drop=True)
        y = gallery.iloc[query_idx]["ship_id"].astype(int).to_numpy()
        split_bank[split] = {"scores": {}, "ships": None, "y": y}
        for cand in CANDIDATES:
            ref_raw = run_cache[cand["run"]][0]
            ref_s = agg.l2(ref_raw[ref_idx].astype(np.float32))
            query_s = agg.l2(ref_raw[query_idx].astype(np.float32))
            ref_s, query_s = stable_weight(ref_s, query_s, ref_df["ship_id"].astype(int).to_numpy())
            c, split_ships = topk_clip_scores(query_s, ref_s, ref_df, 20)
            a5, s5 = ais_scores(query_s, ref_s, ref_df, 5)
            a10, s10 = ais_scores(query_s, ref_s, ref_df, 10)
            if not np.array_equal(split_ships, s5) or not np.array_equal(split_ships, s10):
                raise RuntimeError(f"ship order mismatch {split} {cand['name']}")
            comps = {"clip20": normalize(c), "ais5": normalize(a5), "ais10": normalize(a10)}
            score = normalize(sum(cand["weights"][name] * comps[name] for name in cand["weights"]))
            split_bank[split]["scores"][cand["name"]] = score
            if split_bank[split]["ships"] is None:
                split_bank[split]["ships"] = split_ships
            elif not np.array_equal(split_bank[split]["ships"], split_ships):
                raise RuntimeError(f"split ship order mismatch {split}")
        split_meta.append({"split": split, "num_ref": len(ref_idx), "num_query": len(query_idx)})
    pd.DataFrame(split_meta).to_csv(args.out_dir / "ensemble_splits.csv", index=False)

    names = [c["name"] for c in CANDIDATES]
    rows = []
    split_rows = []
    for comp in compositions(len(names), args.grid_units):
        w = {name: units / args.grid_units for name, units in zip(names, comp)}
        val_score = sum(w[name] * val_scores[name] for name in names)
        vm = metric(val_score, ships, y_val)
        robust_scores = []
        robust_top1 = []
        weight_str = str(tuple(w[name] for name in names))
        for split, bank in split_bank.items():
            s = sum(w[name] * bank["scores"][name] for name in names)
            m = metric(s, bank["ships"], bank["y"])
            robust_scores.append(m["Score"])
            robust_top1.append(m["top1_max_ship_fraction"])
            split_rows.append({"weights": weight_str, "split": split, **m})
        row = {
            "weights": weight_str,
            **{f"w_{name}": w[name] for name in names},
            "official_R@1": vm["R@1"],
            "official_R@3": vm["R@3"],
            "official_R@5": vm["R@5"],
            "official_Score": vm["Score"],
            "official_top1_unique": vm["top1_unique_ships"],
            "official_top1_max_fraction": vm["top1_max_ship_fraction"],
            "robust_mean_score": float(np.mean(robust_scores)),
            "robust_min_score": float(np.min(robust_scores)),
            "robust_std_score": float(np.std(robust_scores)),
            "robust_mean_top1_max_fraction": float(np.mean(robust_top1)),
        }
        row["balanced_score"] = row["official_Score"] + row["robust_mean_score"] + 0.5 * row["robust_min_score"] - 0.15 * row["official_top1_max_fraction"]
        row["conservative_score"] = min(row["official_Score"], row["robust_mean_score"]) + 0.5 * row["robust_min_score"]
        rows.append(row)

    leaderboard = pd.DataFrame(rows).sort_values(["balanced_score", "official_Score"], ascending=False)
    leaderboard.to_csv(args.out_dir / "ensemble_leaderboard.csv", index=False)
    pd.DataFrame(split_rows).to_csv(args.out_dir / "ensemble_split_scores.csv", index=False)
    pd.DataFrame(CANDIDATES).to_csv(args.out_dir / "ensemble_inputs.csv", index=False)

    selected = {
        "balanced": leaderboard.iloc[0],
        "conservative": leaderboard.sort_values(["conservative_score", "official_Score"], ascending=False).iloc[0],
        "official": leaderboard.sort_values(["official_Score", "robust_mean_score"], ascending=False).iloc[0],
        "robust": leaderboard.sort_values(["robust_mean_score", "official_Score"], ascending=False).iloc[0],
        "val086_robust": leaderboard[leaderboard["official_Score"] >= 0.086].sort_values(["robust_mean_score", "robust_min_score"], ascending=False).head(1),
        "val09_robust": leaderboard[leaderboard["official_Score"] >= 0.09].sort_values(["robust_mean_score", "robust_min_score"], ascending=False).head(1),
    }
    selected_rows = []
    for label, row_obj in selected.items():
        if isinstance(row_obj, pd.DataFrame):
            if row_obj.empty:
                continue
            row = row_obj.iloc[0]
        else:
            row = row_obj
        w = {name: float(row[f"w_{name}"]) for name in names}
        val_score = sum(w[name] * val_scores[name] for name in names)
        test_score = sum(w[name] * test_scores[name] for name in names)
        pred_val = agg.top5(val_score, ships)
        pred = agg.top5(test_score, ships)
        val_sub = val_sample.copy()
        val_pred_map = dict(zip(val["filename"], [",".join(map(str, p)) for p in pred_val]))
        val_sub["top5_ship_ids"] = val_sub["filename"].map(val_pred_map)
        sub = sample.copy()
        pred_map = dict(zip(test["filename"], [",".join(map(str, p)) for p in pred]))
        sub["top5_ship_ids"] = sub["filename"].map(pred_map)
        val_path = args.out_dir / f"validation_task2_multirun_score_ensemble_{label}.csv"
        path = args.out_dir / f"submission_task2_multirun_score_ensemble_{label}.csv"
        val_sub.to_csv(val_path, index=False)
        sub.to_csv(path, index=False)
        selected_rows.append({"label": label, "validation_submission": str(val_path), "submission": str(path), **row.to_dict()})
    pd.DataFrame(selected_rows).to_csv(args.out_dir / "ensemble_selected_submissions.csv", index=False)

    summary = {
        "candidates": CANDIDATES,
        "best_balanced": selected_rows[0] if selected_rows else None,
        "outputs": {
            "leaderboard": str(args.out_dir / "ensemble_leaderboard.csv"),
            "split_scores": str(args.out_dir / "ensemble_split_scores.csv"),
            "selected_submissions": str(args.out_dir / "ensemble_selected_submissions.csv"),
        },
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
