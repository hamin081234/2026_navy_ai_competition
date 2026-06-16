#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--bundle-dir", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument("submission", type=Path)
    return p.parse_args()


def main():
    args = parse_args()
    path = args.submission if args.submission.is_absolute() else args.bundle_dir / args.submission
    labels = pd.read_csv(args.bundle_dir / "data/_task2_test_labels.csv")
    sub = pd.read_csv(path)
    merged = labels.merge(sub, on="filename", how="left", validate="one_to_one")
    if merged["top5_ship_ids"].isna().any():
        raise ValueError("missing predictions")
    pred = np.array([[int(x) for x in str(v).split(",")] for v in merged["top5_ship_ids"]], dtype=int)
    y = merged["ship_id"].astype(int).to_numpy()
    r1 = float(np.mean(pred[:, 0] == y))
    r3 = float(np.mean([yy in row[:3] for yy, row in zip(y, pred)]))
    r5 = float(np.mean([yy in row[:5] for yy, row in zip(y, pred)]))
    score = 0.5 * r1 + 0.3 * r3 + 0.2 * r5
    print(f"submission,{path}")
    print(f"R@1,{r1:.15f}")
    print(f"R@3,{r3:.15f}")
    print(f"R@5,{r5:.15f}")
    print(f"Score,{score:.15f}")


if __name__ == "__main__":
    main()
