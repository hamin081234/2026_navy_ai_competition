#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import task2_gallery_aggregation_sweep as agg


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=Path("database"))
    p.add_argument("--cache-dir", type=Path, default=Path("outputs/task2_cache/logmel96_f16000"))
    p.add_argument("--out-dir", type=Path, default=Path("outputs/task2_handcrafted_acoustic"))
    return p.parse_args()


def l2(x):
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-8, None)


def normalize(scores):
    return (scores - scores.mean(axis=1, keepdims=True)) / (scores.std(axis=1, keepdims=True) + 1e-6)


def extract_features(spec: np.ndarray) -> np.ndarray:
    # spec: [N, mel, time], normalized log-mel.
    x = spec.astype(np.float32)
    mean = x.mean(axis=2)
    std = x.std(axis=2)
    p10 = np.percentile(x, 10, axis=2)
    p90 = np.percentile(x, 90, axis=2)
    delta = np.diff(x, axis=2)
    d_mean = delta.mean(axis=2)
    d_std = delta.std(axis=2)

    low = x[:, :32, :].mean(axis=1)
    mid = x[:, 32:64, :].mean(axis=1)
    high = x[:, 64:, :].mean(axis=1)
    bands = []
    for band in [low, mid, high]:
        bands.append(band.mean(axis=1, keepdims=True))
        bands.append(band.std(axis=1, keepdims=True))
        bands.append(np.percentile(band, 90, axis=1, keepdims=True))
        env = band - band.mean(axis=1, keepdims=True)
        fft = np.abs(np.fft.rfft(env, axis=1))[:, 1:40]
        fft = np.log1p(fft)
        bands.append(fft)

    # Spectral centroid-like statistics on mel index.
    weights = np.exp(x - x.max(axis=1, keepdims=True))
    weights = weights / np.clip(weights.sum(axis=1, keepdims=True), 1e-8, None)
    mel_idx = np.arange(x.shape[1], dtype=np.float32).reshape(1, -1, 1)
    centroid = (weights * mel_idx).sum(axis=1)
    centroid_stats = np.stack(
        [centroid.mean(axis=1), centroid.std(axis=1), np.percentile(centroid, 10, axis=1), np.percentile(centroid, 90, axis=1)],
        axis=1,
    )

    feat = np.concatenate([mean, std, p10, p90, d_mean, d_std, *bands, centroid_stats], axis=1)
    feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
    feat = (feat - feat.mean(axis=0, keepdims=True)) / (feat.std(axis=0, keepdims=True) + 1e-6)
    return l2(feat.astype(np.float32))


def load_split_features(meta, cache, split):
    df = meta[meta["split"] == split].copy().reset_index(drop=True)
    idx = df["cache_index"].astype(int).to_numpy()
    feats = extract_features(np.asarray(cache[idx], dtype=np.float32))
    return df, feats


def score_variant(query, ref, gallery, method, topk):
    ships, groups, counts = agg.ship_index(gallery["ship_id"].astype(int).to_numpy())
    if method == "prototype":
        scores = agg.score_variant(query, ref, groups, counts, "prototype", None, 0.0)
    elif method == "clip_max":
        scores = agg.score_variant(query, ref, groups, counts, "clip_max", None, 0.0)
    else:
        scores = agg.score_variant(query, ref, groups, counts, "topk_mean", topk, 0.0)
    return normalize(scores), ships


def make_submission(path, sample, names, pred):
    sub = sample.copy()
    pred_map = dict(zip(names, [",".join(map(str, row)) for row in pred]))
    sub["top5_ship_ids"] = sub["filename"].map(pred_map)
    sub.to_csv(path, index=False)


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(args.cache_dir / "metadata.csv")
    cache = np.load(args.cache_dir / "logmel.npy", mmap_mode="r")
    gallery_raw = pd.read_csv(args.data_dir / "task2_test/gallery.csv")
    sample = pd.read_csv(args.data_dir / "sample_submission_task2.csv")
    gallery, ref = load_split_features(meta, cache, "gallery")
    val, val_feat = load_split_features(meta, cache, "val")
    test, test_feat = load_split_features(meta, cache, "test")
    # Restore labels/ship ids from official tables to avoid relying on cache float formatting.
    gallery["ship_id"] = gallery_raw["ship_id"].astype(int).to_numpy()
    y = pd.read_csv(args.data_dir / "task2_test/val.csv")["ship_id"].astype(int).to_numpy()

    rows = []
    selected = []
    for method, topk in [("prototype", 0), ("clip_max", 0), ("topk_mean", 1), ("topk_mean", 3), ("topk_mean", 5), ("topk_mean", 10), ("topk_mean", 20)]:
        val_scores, ships = score_variant(val_feat, ref, gallery, method, topk)
        pred = agg.top5(val_scores, ships)
        m = {**agg.metric(y, pred), **agg.prediction_stats(pred)}
        name = f"{method}_k{topk}"
        rows.append({"method": method, "topk": topk, **m})
        test_scores, _ = score_variant(test_feat, ref, gallery, method, topk)
        sub_path = args.out_dir / f"submission_handcrafted_{name}.csv"
        make_submission(sub_path, sample, test["filename"].astype(str).tolist(), agg.top5(test_scores, ships))
        selected.append({"submission": str(sub_path), "method": method, "topk": topk, **m})
        print(json.dumps({"method": method, "topk": topk, **m}, ensure_ascii=False), flush=True)
    leaderboard = pd.DataFrame(rows).sort_values(["Score", "R@1"], ascending=False)
    selected_df = pd.DataFrame(selected).sort_values(["Score", "R@1"], ascending=False)
    leaderboard.to_csv(args.out_dir / "handcrafted_leaderboard.csv", index=False)
    selected_df.to_csv(args.out_dir / "handcrafted_selected.csv", index=False)
    summary = {"best": leaderboard.iloc[0].to_dict(), "selected": str(args.out_dir / "handcrafted_selected.csv")}
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
