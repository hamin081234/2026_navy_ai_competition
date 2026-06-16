#!/usr/bin/env python3
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


DATA_DIR = Path("database")
RUNS_DIR = Path("outputs/task2_runs")
OUT_DIR = Path("analysis/monthly_daily_distribution")


def l2(x: np.ndarray) -> np.ndarray:
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-8, None)


def score_matrix(query_emb, ref_emb, ref_ship_ids, method, topk=None, alpha=0.0):
    query_emb = l2(query_emb.astype(np.float32))
    ref_emb = l2(ref_emb.astype(np.float32))
    ships = np.array(sorted(set(ref_ship_ids.tolist())), dtype=int)
    sim = query_emb @ ref_emb.T
    scores = np.zeros((len(query_emb), len(ships)), dtype=np.float32)
    for j, sid in enumerate(ships):
        cols = np.where(ref_ship_ids == sid)[0]
        vals = sim[:, cols]
        if method == "prototype":
            proto = l2(ref_emb[cols].mean(axis=0, keepdims=True))
            scores[:, j] = (query_emb @ proto.T).reshape(-1)
        elif method == "max":
            scores[:, j] = vals.max(axis=1)
        elif method == "topk":
            k = min(int(topk), vals.shape[1])
            part = np.partition(vals, kth=vals.shape[1] - k, axis=1)[:, -k:]
            scores[:, j] = part.mean(axis=1)
        else:
            raise ValueError(method)
        if alpha:
            scores[:, j] -= alpha * math.log1p(len(cols))
    return scores.astype(np.float32), ships


def hybrid_score(query_emb, ref_emb, ref_ship_ids, weights, topk, alpha):
    proto, ships = score_matrix(query_emb, ref_emb, ref_ship_ids, "prototype", alpha=0.0)
    top, _ = score_matrix(query_emb, ref_emb, ref_ship_ids, "topk", topk=topk, alpha=0.0)
    mx, _ = score_matrix(query_emb, ref_emb, ref_ship_ids, "max", alpha=0.0)
    scores = weights[0] * proto + weights[1] * top + weights[2] * mx
    if alpha:
        for j, sid in enumerate(ships):
            n = int(np.sum(ref_ship_ids == sid))
            scores[:, j] -= alpha * math.log1p(n)
    return scores.astype(np.float32), ships


def parse_weights(value):
    if pd.isna(value) or value in ("", "nan", None):
        return None
    return tuple(float(x.strip()) for x in str(value).strip("()").split(","))


def top5(scores, ships):
    return ships[np.argsort(-scores, axis=1)[:, :5]]


def metrics(y, pred):
    r1 = float(np.mean([yy in row[:1] for yy, row in zip(y, pred)]))
    r3 = float(np.mean([yy in row[:3] for yy, row in zip(y, pred)]))
    r5 = float(np.mean([yy in row[:5] for yy, row in zip(y, pred)]))
    return {"R@1": r1, "R@3": r3, "R@5": r5, "Score": 0.5 * r1 + 0.3 * r3 + 0.2 * r5}


def normalize_scores(scores):
    return (scores - scores.mean(axis=1, keepdims=True)) / (scores.std(axis=1, keepdims=True) + 1e-6)


def run_normal_score(run_dir: Path, ref_ship_ids):
    query = np.load(run_dir / "val_embeddings.npy")
    ref = np.load(run_dir / "reference_embeddings.npy")
    best = pd.read_csv(run_dir / "retrieval_leaderboard.csv").iloc[0].to_dict()
    method = best["retrieval"]
    alpha = float(best["alpha"])
    if method == "hybrid":
        scores, ships = hybrid_score(query, ref, ref_ship_ids, parse_weights(best["weights"]), int(best["topk"]), alpha)
    elif method == "topk":
        scores, ships = score_matrix(query, ref, ref_ship_ids, "topk", topk=int(best["topk"]), alpha=alpha)
    else:
        scores, ships = score_matrix(query, ref, ref_ship_ids, method, alpha=alpha)
    return scores, ships, best


def run_ensemble_from_summary(summary_path: Path, ref_ship_ids):
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    weights = parse_weights(summary["best_validation"]["weights"])
    scores_list = []
    ships_ref = None
    rows = []
    for run_name in summary["normal_runs"]:
        scores, ships, best = run_normal_score(RUNS_DIR / run_name, ref_ship_ids)
        if ships_ref is None:
            ships_ref = ships
        elif not np.array_equal(ships_ref, ships):
            raise RuntimeError(f"ship order mismatch: {run_name}")
        scores_list.append(normalize_scores(scores))
        rows.append({"run": run_name, **best})
    if summary.get("subprototype_specs"):
        raise ValueError("This helper handles normal-run ensembles only.")
    final = sum(w * s for w, s in zip(weights, scores_list))
    return final.astype(np.float32), ships_ref, {"weights": weights, "inputs": rows}


def build_ship_concentration(gallery: pd.DataFrame) -> pd.DataFrame:
    g = gallery.copy()
    g["date"] = pd.to_datetime(g["ais_timestamp"]).dt.date.astype(str)
    g["month"] = pd.to_datetime(g["ais_timestamp"]).dt.strftime("%Y-%m")

    daily = g.groupby(["month", "date"]).size().reset_index(name="global_day_count")
    monthly = g.groupby("month").size().reset_index(name="global_month_count")
    daily = daily.merge(monthly, on="month", how="left")
    daily["global_day_fraction_in_month"] = daily["global_day_count"] / daily["global_month_count"]

    rows = []
    for sid, group in g.groupby("ship_id"):
        by_date = group.groupby(["month", "date"]).size().reset_index(name="ship_day_count")
        by_date = by_date.sort_values("ship_day_count", ascending=False).reset_index(drop=True)
        total = int(len(group))
        top = by_date.iloc[0]
        top3 = int(by_date.head(3)["ship_day_count"].sum())
        global_row = daily[(daily["month"] == top["month"]) & (daily["date"] == top["date"])].iloc[0]
        rows.append(
            {
                "ship_id": int(sid),
                "gallery_count": total,
                "ship_active_days": int(len(by_date)),
                "ship_majority_month": str(top["month"]),
                "ship_majority_date": str(top["date"]),
                "ship_max_day_count": int(top["ship_day_count"]),
                "ship_max_day_fraction": float(top["ship_day_count"] / total),
                "ship_top3_day_fraction": float(top3 / total),
                "majority_date_global_count": int(global_row["global_day_count"]),
                "majority_date_global_month_count": int(global_row["global_month_count"]),
                "majority_date_global_fraction": float(global_row["global_day_fraction_in_month"]),
            }
        )
    out = pd.DataFrame(rows)
    out["ship_concentration_bin"] = pd.cut(
        out["ship_max_day_fraction"],
        bins=[-0.001, 0.25, 0.50, 0.75, 1.001],
        labels=["<=25%", "25-50%", "50-75%", ">75%"],
    )
    out["global_spike_bin"] = pd.cut(
        out["majority_date_global_fraction"],
        bins=[-0.001, 0.10, 0.20, 0.50, 1.001],
        labels=["<=10%", "10-20%", "20-50%", ">50%"],
    )
    out["gallery_count_bin"] = pd.qcut(
        out["gallery_count"].rank(method="first"),
        q=4,
        labels=["Q1_low_count", "Q2", "Q3", "Q4_high_count"],
    )
    return out


def attach_hits(val: pd.DataFrame, pred: np.ndarray, model_name: str, ship_conc: pd.DataFrame) -> pd.DataFrame:
    y = val["ship_id"].astype(int).to_numpy()
    rows = val[["filename", "ship_id", "ship_type"]].copy()
    rows["model"] = model_name
    rows["pred_top1"] = pred[:, 0]
    rows["hit1"] = [int(t in row[:1]) for t, row in zip(y, pred)]
    rows["hit3"] = [int(t in row[:3]) for t, row in zip(y, pred)]
    rows["hit5"] = [int(t in row[:5]) for t, row in zip(y, pred)]
    rows = rows.merge(ship_conc, on="ship_id", how="left")
    return rows


def summarize_group(rows: pd.DataFrame, group_col: str) -> pd.DataFrame:
    out = (
        rows.groupby(["model", group_col], observed=False)
        .agg(
            n=("filename", "count"),
            ships=("ship_id", "nunique"),
            gallery_count_mean=("gallery_count", "mean"),
            ship_max_day_fraction_mean=("ship_max_day_fraction", "mean"),
            majority_global_fraction_mean=("majority_date_global_fraction", "mean"),
            R1=("hit1", "mean"),
            R3=("hit3", "mean"),
            R5=("hit5", "mean"),
        )
        .reset_index()
    )
    out["Score"] = 0.5 * out["R1"] + 0.3 * out["R3"] + 0.2 * out["R5"]
    out.insert(1, "grouping", group_col)
    out = out.rename(columns={group_col: "group"})
    return out


def correlation_rows(rows: pd.DataFrame) -> pd.DataFrame:
    metrics_cols = ["gallery_count", "ship_max_day_fraction", "ship_top3_day_fraction", "majority_date_global_fraction"]
    hit_cols = ["hit1", "hit3", "hit5"]
    out = []
    for model, g in rows.groupby("model"):
        for x in metrics_cols:
            for y in hit_cols:
                out.append(
                    {
                        "model": model,
                        "x": x,
                        "y": y,
                        "pearson": float(g[x].corr(g[y], method="pearson")),
                        "spearman": float(g[x].corr(g[y], method="spearman")),
                    }
                )
    return pd.DataFrame(out)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    gallery = pd.read_csv(DATA_DIR / "task2_test/gallery.csv")
    val = pd.read_csv(DATA_DIR / "task2_test/val.csv")
    ref_ship_ids = gallery["ship_id"].astype(int).to_numpy()
    y = val["ship_id"].astype(int).to_numpy()
    ship_conc = build_ship_concentration(gallery)
    ship_conc.to_csv(OUT_DIR / "ship_gallery_day_concentration.csv", index=False)

    candidates = {}
    for name in [
        "quick_target100_mel96_f16000",
        "quick_target100_trainonly_mel96_f16000",
        "projection_clip_trainonly",
        "robust_rows100_seed123_ep2_aug",
    ]:
        scores, ships, _ = run_normal_score(RUNS_DIR / name, ref_ship_ids)
        candidates[name] = (scores, ships)
    scores, ships, _ = run_ensemble_from_summary(RUNS_DIR / "debug_ensemble_5normal" / "summary.json", ref_ship_ids)
    candidates["debug_ensemble_5normal"] = (scores, ships)

    all_rows = []
    model_summary = []
    for name, (scores, ships) in candidates.items():
        pred = top5(scores, ships)
        all_rows.append(attach_hits(val, pred, name, ship_conc))
        model_summary.append({"model": name, **metrics(y, pred)})
    row_hits = pd.concat(all_rows, ignore_index=True)
    row_hits.to_csv(OUT_DIR / "concentration_accuracy_row_hits.csv", index=False)
    pd.DataFrame(model_summary).to_csv(OUT_DIR / "concentration_accuracy_model_summary.csv", index=False)

    grouped = pd.concat(
        [
            summarize_group(row_hits, "ship_concentration_bin"),
            summarize_group(row_hits, "global_spike_bin"),
            summarize_group(row_hits, "gallery_count_bin"),
        ],
        ignore_index=True,
    )
    grouped.to_csv(OUT_DIR / "concentration_accuracy_grouped.csv", index=False)

    corr = correlation_rows(row_hits)
    corr.to_csv(OUT_DIR / "concentration_accuracy_correlations.csv", index=False)

    # Ship-level view, useful for checking whether a few ships dominate row-level conclusions.
    ship_level = (
        row_hits.groupby(["model", "ship_id"])
        .agg(
            val_rows=("filename", "count"),
            R1=("hit1", "mean"),
            R3=("hit3", "mean"),
            R5=("hit5", "mean"),
            gallery_count=("gallery_count", "first"),
            ship_max_day_fraction=("ship_max_day_fraction", "first"),
            ship_top3_day_fraction=("ship_top3_day_fraction", "first"),
            majority_date_global_fraction=("majority_date_global_fraction", "first"),
            ship_concentration_bin=("ship_concentration_bin", "first"),
            global_spike_bin=("global_spike_bin", "first"),
            gallery_count_bin=("gallery_count_bin", "first"),
        )
        .reset_index()
    )
    ship_level["Score"] = 0.5 * ship_level["R1"] + 0.3 * ship_level["R3"] + 0.2 * ship_level["R5"]
    ship_level.to_csv(OUT_DIR / "concentration_accuracy_ship_level.csv", index=False)

    print("model summary")
    print(pd.DataFrame(model_summary).to_string(index=False))
    print("\\nship concentration bins")
    print(grouped[grouped["grouping"] == "ship_concentration_bin"].to_string(index=False))
    print("\\nglobal spike bins")
    print(grouped[grouped["grouping"] == "global_spike_bin"].to_string(index=False))
    print("\\ncorrelations")
    print(corr.to_string(index=False))


if __name__ == "__main__":
    main()
