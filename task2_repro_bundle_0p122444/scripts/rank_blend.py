#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


SCHEMES = {
    "flat": np.array([1.0, 0.9, 0.8, 0.7, 0.6], dtype=np.float32),
    "linear": np.array([1.0, 0.8, 0.6, 0.4, 0.2], dtype=np.float32),
    "reciprocal": np.array([1.0, 0.5, 1.0 / 3.0, 0.25, 0.2], dtype=np.float32),
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--bundle-dir", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--scheme", choices=sorted(SCHEMES), default="flat")
    p.add_argument("--path", action="append", required=True)
    p.add_argument("--weight", action="append", type=float, required=True)
    return p.parse_args()


def top5_idx(score: np.ndarray) -> np.ndarray:
    part = np.argpartition(-score, kth=4, axis=1)[:, :5]
    vals = np.take_along_axis(score, part, axis=1)
    order = np.argsort(-vals, axis=1)
    return np.take_along_axis(part, order, axis=1)


def main():
    args = parse_args()
    if len(args.path) != len(args.weight):
        raise ValueError("--path and --weight counts must match")

    sample = pd.read_csv(args.bundle_dir / "data/sample_submission_task2.csv")
    ships = np.array(sorted(pd.read_csv(args.bundle_dir / "data/task2_target_ships.csv")["ship_id"].astype(int)), dtype=np.int32)
    ship_to_idx = {int(s): i for i, s in enumerate(ships)}
    scores = np.zeros((len(sample), len(ships)), dtype=np.float32)
    rank_values = SCHEMES[args.scheme]
    sources = []

    for path_text, weight in zip(args.path, args.weight):
        if weight <= 0:
            continue
        path = Path(path_text)
        if not path.is_absolute():
            path = args.bundle_dir / path
        sub = pd.read_csv(path)
        if len(sub) != len(sample) or not sub["filename"].equals(sample["filename"]):
            raise ValueError(f"submission shape/order mismatch: {path}")
        for row_idx, value in enumerate(sub["top5_ship_ids"].astype(str)):
            for rank, sid_text in enumerate(value.split(",")):
                scores[row_idx, ship_to_idx[int(sid_text)]] += float(weight) * float(rank_values[rank])
        sources.append({"path": str(path), "weight": weight})

    pred_idx = top5_idx(scores)
    pred_ship = ships[pred_idx]
    out = sample.copy()
    out["top5_ship_ids"] = [",".join(map(str, row)) for row in pred_ship]
    out_path = args.out if args.out.is_absolute() else args.bundle_dir / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(json.dumps({"output": str(out_path), "scheme": args.scheme, "sources": sources}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
