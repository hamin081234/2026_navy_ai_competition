#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("submission", type=Path)
    parser.add_argument("--labels", type=Path, default=Path("data/_task2_test_labels.csv"))
    parser.add_argument("--out", type=Path)
    return parser.parse_args()


def parse_top5(value: object) -> list[int]:
    return [int(x.strip()) for x in str(value).split(",") if x.strip()]


def metrics(labels: pd.DataFrame, sub: pd.DataFrame) -> dict[str, float]:
    merged = labels.merge(sub, on="filename", how="left", validate="one_to_one")
    if merged["top5_ship_ids"].isna().any():
        raise ValueError("missing predictions")
    y = merged["ship_id"].astype(int).to_numpy()
    pred = [parse_top5(v) for v in merged["top5_ship_ids"]]
    r1 = float(np.mean([truth in row[:1] for truth, row in zip(y, pred)]))
    r3 = float(np.mean([truth in row[:3] for truth, row in zip(y, pred)]))
    r5 = float(np.mean([truth in row[:5] for truth, row in zip(y, pred)]))
    return {"R@1": r1, "R@3": r3, "R@5": r5, "Score": 0.5 * r1 + 0.3 * r3 + 0.2 * r5}


def main() -> None:
    args = parse_args()
    bundle_dir = Path(__file__).resolve().parents[1]
    submission = args.submission if args.submission.is_absolute() else bundle_dir / args.submission
    labels = args.labels if args.labels.is_absolute() else bundle_dir / args.labels
    result = {"submission": str(submission), **metrics(pd.read_csv(labels), pd.read_csv(submission))}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.out:
        out = args.out if args.out.is_absolute() else bundle_dir / args.out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
