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
    p.add_argument("--out-dir", type=Path, default=Path("outputs/task2_ais_group_pooling"))
    p.add_argument("--run", default="robust_rows100_seed123_ep2_aug")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--holdout-frac", type=float, default=0.2)
    p.add_argument("--fast", action="store_true")
    return p.parse_args()


def add_group_columns(gallery: pd.DataFrame) -> pd.DataFrame:
    out = gallery.copy().reset_index(drop=True)
    ts = pd.to_datetime(out["ais_timestamp"], utc=True, format="mixed")
    out["_timestamp"] = ts
    out["date"] = ts.dt.strftime("%Y-%m-%d")
    out["month_key"] = ts.dt.strftime("%Y-%m")
    out["hour"] = ts.dt.strftime("%Y-%m-%d %H")
    out["hour4"] = ts.dt.strftime("%Y-%m-%d") + "_" + pd.cut(
        ts.dt.hour,
        bins=[-1, 5, 11, 17, 23],
        labels=["night", "morning", "afternoon", "evening"],
    ).astype(str)
    out["ais_exact"] = out["ais_timestamp"].astype(str)
    out["sog_bin"] = pd.cut(
        out["sog"].astype(float),
        bins=[-0.01, 0.5, 5.0, 10.0, 100.0],
        labels=["stopped", "slow", "medium", "fast"],
    ).astype(str)
    heading = out["true_heading"].astype(float) % 360
    out["heading4"] = pd.cut(
        heading,
        bins=[-0.01, 90, 180, 270, 360],
        labels=["h000_090", "h090_180", "h180_270", "h270_360"],
        include_lowest=True,
    ).astype(str)
    out["date_sog"] = out["date"] + "__" + out["sog_bin"]
    out["date_heading"] = out["date"] + "__" + out["heading4"]
    out["date_sog_heading"] = out["date"] + "__" + out["sog_bin"] + "__" + out["heading4"]
    return out


def normalize(scores: np.ndarray) -> np.ndarray:
    return (scores - scores.mean(axis=1, keepdims=True)) / (scores.std(axis=1, keepdims=True) + 1e-6)


def stable_weight(ref: np.ndarray, query: np.ndarray, ship_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    weights = agg.stable_dimension_weights(ref, ship_ids)
    return agg.apply_weights(ref, query, weights)


def group_score_matrix(
    query: np.ndarray,
    ref: np.ndarray,
    gallery: pd.DataFrame,
    group_col: str,
    group_reduction: str,
    ship_reduction: str,
    topk: int,
    alpha: float,
    group_size_beta: float,
) -> tuple[np.ndarray, np.ndarray]:
    sim = query @ ref.T
    ships = np.array(sorted(gallery["ship_id"].astype(int).unique().tolist()), dtype=int)
    scores = np.full((len(query), len(ships)), -1e9, dtype=np.float32)
    frame = gallery[["ship_id", group_col]].copy()
    frame["_idx"] = np.arange(len(frame))
    for j, sid in enumerate(ships):
        ship_frame = frame[frame["ship_id"].astype(int) == int(sid)]
        group_scores = []
        group_sizes = []
        for _, group in ship_frame.groupby(group_col, sort=False):
            idx = group["_idx"].to_numpy(dtype=int)
            vals = sim[:, idx]
            if group_reduction == "max":
                gs = vals.max(axis=1)
            elif group_reduction == "mean":
                gs = vals.mean(axis=1)
            elif group_reduction == "softmax":
                z = vals / 0.03
                z = z - z.max(axis=1, keepdims=True)
                w = np.exp(z)
                w = w / np.clip(w.sum(axis=1, keepdims=True), 1e-8, None)
                gs = (w * vals).sum(axis=1)
            else:
                raise ValueError(group_reduction)
            if group_size_beta:
                gs = gs - group_size_beta * np.log1p(len(idx))
            group_scores.append(gs)
            group_sizes.append(len(idx))
        mat = np.vstack(group_scores).T
        if ship_reduction == "max":
            ss = mat.max(axis=1)
        elif ship_reduction == "topk_mean":
            k = min(int(topk), mat.shape[1])
            part = np.partition(mat, kth=mat.shape[1] - k, axis=1)[:, -k:]
            ss = part.mean(axis=1)
        elif ship_reduction == "softmax":
            z = mat / 0.05
            z = z - z.max(axis=1, keepdims=True)
            w = np.exp(z)
            w = w / np.clip(w.sum(axis=1, keepdims=True), 1e-8, None)
            ss = (w * mat).sum(axis=1)
        else:
            raise ValueError(ship_reduction)
        if alpha:
            # Penalize number of independent groups, not raw clip count.
            ss = ss - alpha * np.log1p(mat.shape[1])
        scores[:, j] = ss
    return scores.astype(np.float32), ships


def evaluate_scores(scores: np.ndarray, ships: np.ndarray, y: np.ndarray) -> dict[str, float]:
    pred = agg.top5(scores, ships)
    return {**agg.metric(y, pred), **agg.prediction_stats(pred)}


def candidate_specs(fast: bool = False) -> list[dict[str, object]]:
    if fast:
        specs = []
        for group_col in ["ais_exact", "date", "date_sog"]:
            for topk in [5, 10, 20]:
                for alpha in [0.0, 0.01]:
                    specs.append({
                        "group_col": group_col,
                        "group_reduction": "max",
                        "ship_reduction": "topk_mean",
                        "topk": topk,
                        "alpha": alpha,
                        "group_size_beta": 0.0,
                    })
            for beta in [0.005, 0.01]:
                specs.append({
                    "group_col": group_col,
                    "group_reduction": "max",
                    "ship_reduction": "topk_mean",
                    "topk": 20,
                    "alpha": 0.0,
                    "group_size_beta": beta,
                })
        return specs
    specs = []
    for group_col in ["ais_exact", "date", "hour", "hour4", "month_key", "date_sog", "date_heading", "date_sog_heading"]:
        for group_reduction in ["max", "mean"]:
            for ship_reduction in ["topk_mean", "max"]:
                for topk in ([3, 5, 10, 20] if ship_reduction == "topk_mean" else [1]):
                    for alpha in [0.0, 0.01, 0.02]:
                        specs.append({
                            "group_col": group_col,
                            "group_reduction": group_reduction,
                            "ship_reduction": ship_reduction,
                            "topk": topk,
                            "alpha": alpha,
                            "group_size_beta": 0.0,
                        })
    # Targeted duplicate/group-size penalty variants for large same-date groups.
    for group_col in ["date", "ais_exact", "date_sog"]:
        for beta in [0.005, 0.01, 0.02]:
            specs.append({
                "group_col": group_col,
                "group_reduction": "max",
                "ship_reduction": "topk_mean",
                "topk": 10,
                "alpha": 0.0,
                "group_size_beta": beta,
            })
            specs.append({
                "group_col": group_col,
                "group_reduction": "max",
                "ship_reduction": "topk_mean",
                "topk": 20,
                "alpha": 0.0,
                "group_size_beta": beta,
            })
    return specs


def spec_name(spec: dict[str, object]) -> str:
    return (
        f"{spec['group_col']}__g{spec['group_reduction']}__s{spec['ship_reduction']}"
        f"__k{spec['topk']}__a{spec['alpha']}__b{spec['group_size_beta']}"
    )


def score_spec_on_split(spec, ref, query, ref_df, y):
    scores, ships = group_score_matrix(
        query=query,
        ref=ref,
        gallery=ref_df,
        group_col=str(spec["group_col"]),
        group_reduction=str(spec["group_reduction"]),
        ship_reduction=str(spec["ship_reduction"]),
        topk=int(spec["topk"]),
        alpha=float(spec["alpha"]),
        group_size_beta=float(spec["group_size_beta"]),
    )
    return scores, ships, evaluate_scores(scores, ships, y)


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    gallery_full = add_group_columns(pd.read_csv(args.data_dir / "task2_test/gallery.csv"))
    val = pd.read_csv(args.data_dir / "task2_test/val.csv")
    test = pd.read_csv(args.data_dir / "task2_test/test.csv")
    sample = pd.read_csv(args.data_dir / "sample_submission_task2.csv")
    run_dir = args.runs_dir / args.run
    ref_full = agg.l2(np.load(run_dir / "reference_embeddings.npy").astype(np.float32))
    val_emb = agg.l2(np.load(run_dir / "val_embeddings.npy").astype(np.float32))
    test_emb = agg.l2(np.load(run_dir / "test_embeddings.npy").astype(np.float32))
    ship_ids_full = gallery_full["ship_id"].astype(int).to_numpy()
    ref_full_w, val_emb_w = stable_weight(ref_full, val_emb, ship_ids_full)
    _, test_emb_w = agg.apply_weights(ref_full, test_emb, agg.stable_dimension_weights(ref_full, ship_ids_full))
    y_val = val["ship_id"].astype(int).to_numpy()
    specs = candidate_specs(args.fast)

    official_rows = []
    score_cache = {}
    for spec in specs:
        scores, ships, m = score_spec_on_split(spec, ref_full_w, val_emb_w, gallery_full, y_val)
        name = spec_name(spec)
        score_cache[name] = (scores, ships, spec)
        official_rows.append({"candidate": name, **spec, **m})
    official_df = pd.DataFrame(official_rows).sort_values(["Score", "R@1", "R@3"], ascending=False)
    official_df.to_csv(args.out_dir / "ais_group_official_scores.csv", index=False)

    split_modes = ["random20", "latest20", "earliest20", "month_latest", "ais_group_random20"]
    robust_rows = []
    split_meta = []
    for split in split_modes:
        ref_idx, query_idx = per_ship_holdout_indices(gallery_full, split, args.holdout_frac, args.seed)
        ref_df = gallery_full.iloc[ref_idx].copy().reset_index(drop=True)
        query_df = gallery_full.iloc[query_idx].copy().reset_index(drop=True)
        ref = agg.l2(ref_full[ref_idx].astype(np.float32))
        query = agg.l2(ref_full[query_idx].astype(np.float32))
        ref_w, query_w = stable_weight(ref, query, ref_df["ship_id"].astype(int).to_numpy())
        y = query_df["ship_id"].astype(int).to_numpy()
        split_meta.append({"split": split, "num_ref": len(ref_df), "num_query": len(query_df)})
        for spec in specs:
            _, _, m = score_spec_on_split(spec, ref_w, query_w, ref_df, y)
            robust_rows.append({"candidate": spec_name(spec), "split": split, **spec, **m})
    robust_df = pd.DataFrame(robust_rows)
    robust_df.to_csv(args.out_dir / "ais_group_robust_split_scores.csv", index=False)
    pd.DataFrame(split_meta).to_csv(args.out_dir / "ais_group_robust_splits.csv", index=False)

    summary = (
        robust_df.groupby("candidate", as_index=False)
        .agg(
            robust_mean_score=("Score", "mean"),
            robust_min_score=("Score", "min"),
            robust_std_score=("Score", "std"),
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
    summary["balanced_score"] = summary["official_Score"] + summary["robust_mean_score"] + 0.5 * summary["robust_min_score"] - 0.15 * summary["official_top1_max_fraction"]
    summary = summary.sort_values(["balanced_score", "official_Score"], ascending=False)
    summary.to_csv(args.out_dir / "ais_group_summary.csv", index=False)

    selected = {
        "official": summary.sort_values(["official_Score", "robust_mean_score"], ascending=False).iloc[0],
        "robust": summary.sort_values(["robust_mean_score", "official_Score"], ascending=False).iloc[0],
        "balanced": summary.iloc[0],
        "val09_robust": summary[summary["official_Score"] >= 0.09].sort_values(["robust_mean_score", "official_Score"], ascending=False).head(1),
    }
    selected_rows = []
    for label, row_obj in selected.items():
        if isinstance(row_obj, pd.DataFrame):
            if row_obj.empty:
                continue
            row = row_obj.iloc[0]
        else:
            row = row_obj
        spec = next(s for s in specs if spec_name(s) == row["candidate"])
        test_scores, ships = group_score_matrix(
            query=test_emb_w,
            ref=ref_full_w,
            gallery=gallery_full,
            group_col=str(spec["group_col"]),
            group_reduction=str(spec["group_reduction"]),
            ship_reduction=str(spec["ship_reduction"]),
            topk=int(spec["topk"]),
            alpha=float(spec["alpha"]),
            group_size_beta=float(spec["group_size_beta"]),
        )
        pred = agg.top5(test_scores, ships)
        sub = sample.copy()
        pred_map = dict(zip(test["filename"], [",".join(map(str, p)) for p in pred]))
        sub["top5_ship_ids"] = sub["filename"].map(pred_map)
        path = args.out_dir / f"submission_task2_ais_group_{label}.csv"
        sub.to_csv(path, index=False)
        selected_rows.append({"label": label, "submission": str(path), **row.to_dict()})
    selected_df = pd.DataFrame(selected_rows)
    selected_df.to_csv(args.out_dir / "ais_group_selected_submissions.csv", index=False)

    out_summary = {
        "run": args.run,
        "best_balanced": selected_df[selected_df["label"] == "balanced"].iloc[0].to_dict() if not selected_df.empty else None,
        "outputs": {
            "official": str(args.out_dir / "ais_group_official_scores.csv"),
            "robust": str(args.out_dir / "ais_group_robust_split_scores.csv"),
            "summary": str(args.out_dir / "ais_group_summary.csv"),
            "selected": str(args.out_dir / "ais_group_selected_submissions.csv"),
        },
    }
    (args.out_dir / "summary.json").write_text(json.dumps(out_summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out_summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
