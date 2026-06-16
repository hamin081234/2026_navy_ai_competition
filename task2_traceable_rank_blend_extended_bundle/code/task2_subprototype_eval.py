#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=Path("database"))
    p.add_argument("--runs-dir", type=Path, default=Path("outputs/task2_runs"))
    p.add_argument("--out-dir", type=Path, default=Path("outputs/task2_subprototype"))
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--methods", nargs="*", default=None)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def l2(x: np.ndarray) -> np.ndarray:
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-8, None)


def metric(y_true: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    r1 = float(np.mean([y in row[:1] for y, row in zip(y_true, pred)]))
    r3 = float(np.mean([y in row[:3] for y, row in zip(y_true, pred)]))
    r5 = float(np.mean([y in row[:5] for y, row in zip(y_true, pred)]))
    return {"R@1": r1, "R@3": r3, "R@5": r5, "Score": 0.5 * r1 + 0.3 * r3 + 0.2 * r5}


def top5_from_scores(scores: np.ndarray, ships: np.ndarray) -> np.ndarray:
    return ships[np.argsort(-scores, axis=1)[:, :5]]


def add_bins(gallery: pd.DataFrame) -> pd.DataFrame:
    out = gallery.copy()
    ts = pd.to_datetime(out["ais_timestamp"], utc=True, format="mixed")
    out["hour"] = ts.dt.hour
    out["month"] = ts.dt.month
    out["hour4"] = pd.cut(
        out["hour"],
        bins=[-1, 5, 11, 17, 23],
        labels=["night", "morning", "afternoon", "evening"],
    ).astype(str)
    out["season"] = out["month"].map({
        12: "winter", 1: "winter", 2: "winter",
        3: "spring", 4: "spring", 5: "spring",
        6: "summer", 7: "summer", 8: "summer",
        9: "fall", 10: "fall", 11: "fall",
    })
    out["speed_bin"] = pd.cut(
        out["sog"].astype(float),
        bins=[-0.01, 0.5, 5.0, 10.0, 100.0],
        labels=["stopped", "slow", "medium", "fast"],
    ).astype(str)
    out["moving"] = np.where(out["sog"].astype(float) < 0.5, "stopped", "moving")
    heading = out["true_heading"].astype(float) % 360
    out["heading4"] = pd.cut(
        heading,
        bins=[-0.01, 90, 180, 270, 360],
        labels=["h000_090", "h090_180", "h180_270", "h270_360"],
        include_lowest=True,
    ).astype(str)
    out["speed_hour4"] = out["speed_bin"] + "__" + out["hour4"]
    out["speed_heading4"] = out["speed_bin"] + "__" + out["heading4"]
    return out


def adaptive_k(n: int) -> int:
    if n < 10:
        return 1
    if n < 50:
        return 2
    if n < 100:
        return 3
    return 5


def prototypes_from_group_labels(
    ref_emb: np.ndarray,
    gallery: pd.DataFrame,
    labels: pd.Series,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    rows = []
    ship_ids = []
    meta_rows = []
    frame = gallery[["filename", "ship_id"]].copy()
    frame["group_label"] = labels.astype(str).values
    for (ship_id, group_label), idx in frame.groupby(["ship_id", "group_label"], sort=True).groups.items():
        idx_arr = np.fromiter(idx, dtype=int)
        proto = l2(ref_emb[idx_arr].mean(axis=0, keepdims=True))[0]
        rows.append(proto)
        ship_ids.append(int(ship_id))
        meta_rows.append({
            "ship_id": int(ship_id),
            "prototype_label": str(group_label),
            "source_clips": int(len(idx_arr)),
        })
    return np.vstack(rows).astype(np.float32), np.array(ship_ids, dtype=int), pd.DataFrame(meta_rows)


def prototypes_centroid(ref_emb: np.ndarray, gallery: pd.DataFrame):
    return prototypes_from_group_labels(ref_emb, gallery, gallery["ship_id"].astype(str))


def prototypes_clip(ref_emb: np.ndarray, gallery: pd.DataFrame):
    meta = pd.DataFrame({
        "ship_id": gallery["ship_id"].astype(int).to_numpy(),
        "prototype_label": gallery["filename"].astype(str).to_numpy(),
        "source_clips": 1,
    })
    return ref_emb.astype(np.float32), gallery["ship_id"].astype(int).to_numpy(), meta


def prototypes_kmeans(ref_emb: np.ndarray, gallery: pd.DataFrame, k_mode: str, seed: int):
    rows = []
    ship_ids = []
    meta_rows = []
    for ship_id, idx in gallery.groupby("ship_id", sort=True).groups.items():
        idx_arr = np.fromiter(idx, dtype=int)
        n = len(idx_arr)
        if k_mode == "adaptive":
            k = adaptive_k(n)
        else:
            k = int(k_mode)
        k = max(1, min(k, n))
        x = ref_emb[idx_arr]
        if k == 1:
            labels = np.zeros(n, dtype=int)
            centers = x.mean(axis=0, keepdims=True)
        else:
            km = KMeans(n_clusters=k, random_state=seed, n_init=10)
            labels = km.fit_predict(x)
            centers = km.cluster_centers_
        centers = l2(centers)
        for c in range(k):
            rows.append(centers[c])
            ship_ids.append(int(ship_id))
            meta_rows.append({
                "ship_id": int(ship_id),
                "prototype_label": f"kmeans_{k_mode}_{c}",
                "source_clips": int(np.sum(labels == c)),
            })
    return np.vstack(rows).astype(np.float32), np.array(ship_ids, dtype=int), pd.DataFrame(meta_rows)


def make_prototypes(method: str, ref_emb: np.ndarray, gallery: pd.DataFrame, seed: int):
    if method == "ship_centroid":
        return prototypes_centroid(ref_emb, gallery)
    if method == "clip_max":
        return prototypes_clip(ref_emb, gallery)
    if method.startswith("kmeans_k"):
        return prototypes_kmeans(ref_emb, gallery, method.replace("kmeans_k", ""), seed)
    if method == "kmeans_adaptive":
        return prototypes_kmeans(ref_emb, gallery, "adaptive", seed)
    group_methods = {
        "time_hour4": "hour4",
        "time_month": "month",
        "time_season": "season",
        "ais_speed_bin": "speed_bin",
        "ais_moving": "moving",
        "ais_heading4": "heading4",
        "ais_speed_hour4": "speed_hour4",
        "ais_speed_heading4": "speed_heading4",
    }
    if method in group_methods:
        return prototypes_from_group_labels(ref_emb, gallery, gallery[group_methods[method]])
    raise ValueError(f"Unknown method: {method}")


def score_by_ship(
    query_emb: np.ndarray,
    proto_emb: np.ndarray,
    proto_ship_ids: np.ndarray,
    proto_source_clips: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    q = l2(query_emb)
    p = l2(proto_emb)
    sim = q @ p.T
    ships = np.array(sorted(set(proto_ship_ids.tolist())), dtype=int)
    scores = np.full((len(q), len(ships)), -1e9, dtype=np.float32)
    for j, sid in enumerate(ships):
        cols = np.where(proto_ship_ids == sid)[0]
        scores[:, j] = sim[:, cols].max(axis=1)
        if alpha:
            scores[:, j] -= alpha * math.log1p(float(proto_source_clips[cols].sum()))
    return scores, ships


def evaluate_method(
    run_name: str,
    method: str,
    ref_emb: np.ndarray,
    val_emb: np.ndarray,
    gallery: pd.DataFrame,
    y_val: np.ndarray,
    seed: int,
    proto_out_dir: Path,
) -> list[dict[str, object]]:
    started = time.time()
    proto_emb, proto_ship_ids, proto_meta = make_prototypes(method, ref_emb, gallery, seed)
    proto_meta.to_csv(proto_out_dir / f"{run_name}__{method}__prototype_meta.csv", index=False)
    proto_source_clips = proto_meta["source_clips"].astype(float).to_numpy()
    rows = []
    for alpha in [0.0, 0.002, 0.005, 0.01, 0.02, 0.03]:
        scores, ships = score_by_ship(val_emb, proto_emb, proto_ship_ids, proto_source_clips, alpha)
        pred = top5_from_scores(scores, ships)
        rows.append({
            "run": run_name,
            "method": method,
            "alpha": alpha,
            "num_prototypes": int(len(proto_ship_ids)),
            "min_prototypes_per_ship": int(pd.Series(proto_ship_ids).value_counts().min()),
            "max_prototypes_per_ship": int(pd.Series(proto_ship_ids).value_counts().max()),
            "runtime_seconds": round(time.time() - started, 3),
            **metric(y_val, pred),
        })
    return rows


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    proto_out_dir = args.out_dir / "prototype_meta"
    proto_out_dir.mkdir(parents=True, exist_ok=True)

    methods = args.methods or [
        "ship_centroid",
        "clip_max",
        "kmeans_k2",
        "kmeans_k3",
        "kmeans_k5",
        "kmeans_adaptive",
        "time_hour4",
        "time_month",
        "time_season",
        "ais_speed_bin",
        "ais_moving",
        "ais_heading4",
        "ais_speed_hour4",
        "ais_speed_heading4",
    ]

    gallery = add_bins(pd.read_csv(args.data_dir / "task2_test/gallery.csv"))
    val = pd.read_csv(args.data_dir / "task2_test/val.csv")
    y_val = val["ship_id"].astype(int).to_numpy()

    all_rows = []
    for run in args.runs:
        run_dir = args.runs_dir / run
        ref_path = run_dir / "reference_embeddings.npy"
        val_path = run_dir / "val_embeddings.npy"
        if not ref_path.exists() or not val_path.exists():
            raise FileNotFoundError(f"Missing embeddings for {run}: {ref_path}, {val_path}")
        ref_emb = l2(np.load(ref_path).astype(np.float32))
        val_emb = l2(np.load(val_path).astype(np.float32))
        if len(ref_emb) != len(gallery):
            raise ValueError(f"{run}: reference row mismatch {len(ref_emb)} != {len(gallery)}")
        if len(val_emb) != len(val):
            raise ValueError(f"{run}: val row mismatch {len(val_emb)} != {len(val)}")

        print(json.dumps({"run": run, "methods": methods, "ref_shape": ref_emb.shape, "val_shape": val_emb.shape}, ensure_ascii=False), flush=True)
        for method in methods:
            rows = evaluate_method(run, method, ref_emb, val_emb, gallery, y_val, args.seed, proto_out_dir)
            all_rows.extend(rows)
            best = max(rows, key=lambda r: r["Score"])
            print(json.dumps(best, ensure_ascii=False), flush=True)

    results = pd.DataFrame(all_rows).sort_values(["Score", "R@1", "R@3"], ascending=False)
    results.to_csv(args.out_dir / "subprototype_scores.csv", index=False)

    best_by_run_method = (
        results.sort_values(["run", "method", "Score"], ascending=[True, True, False])
        .groupby(["run", "method"], as_index=False)
        .head(1)
        .sort_values(["run", "Score"], ascending=[True, False])
    )
    best_by_run_method.to_csv(args.out_dir / "subprototype_best_by_method.csv", index=False)

    summary = {
        "runs": args.runs,
        "methods": methods,
        "best_overall": results.iloc[0].to_dict(),
        "output": {
            "scores": str(args.out_dir / "subprototype_scores.csv"),
            "best_by_method": str(args.out_dir / "subprototype_best_by_method.csv"),
            "prototype_meta_dir": str(proto_out_dir),
        },
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"best_overall": summary["best_overall"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
