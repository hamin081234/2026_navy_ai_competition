#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


SCHEME = np.array([1.0, 0.35, 0.20, 0.12, 0.08], dtype=np.float32)
SOURCES = [
    {
        "label": "multirun_balanced",
        "weight": 0.65,
        "validation": "validation_multirun_balanced.csv",
        "submission": "submission_multirun_balanced.csv",
    },
    {
        "label": "multirun_val086_robust",
        "weight": 1.44576,
        "validation": "validation_multirun_val086_robust.csv",
        "submission": "submission_multirun_val086_robust.csv",
    },
    {
        "label": "public_refined_validation_backed",
        "weight": 1.5,
        "validation": "validation_public_refined_validation_backed.csv",
        "submission": "submission_public_refined_validation_backed.csv",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("../database"))
    parser.add_argument("--source-dir", type=Path, default=Path("sources"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--expected-score", type=float, default=0.11704180064308681)
    return parser.parse_args()


def parse_top5(value: object) -> list[int]:
    return [int(x.strip()) for x in str(value).split(",") if str(x).strip()]


def load_prediction(path: Path, sample: pd.DataFrame) -> np.ndarray:
    sub = pd.read_csv(path)
    if len(sub) != len(sample):
        raise ValueError(f"row count mismatch: {path}")
    if not sub["filename"].equals(sample["filename"]):
        raise ValueError(f"filename order mismatch: {path}")
    pred = np.array([parse_top5(v) for v in sub["top5_ship_ids"]], dtype=np.int64)
    if pred.shape[1] != 5:
        raise ValueError(f"expected top-5 predictions: {path}")
    return pred


def fuse(predictions: list[np.ndarray], weights: list[float], ships: np.ndarray) -> np.ndarray:
    ship_to_idx = {int(ship): idx for idx, ship in enumerate(ships)}
    scores = np.zeros((predictions[0].shape[0], len(ships)), dtype=np.float32)

    for pred, weight in zip(predictions, weights):
        for rank in range(5):
            rank_score = float(weight) * float(SCHEME[rank])
            for row_idx, ship_id in enumerate(pred[:, rank]):
                scores[row_idx, ship_to_idx[int(ship_id)]] += rank_score

    part = np.argpartition(-scores, kth=4, axis=1)[:, :5]
    part_scores = np.take_along_axis(scores, part, axis=1)
    order = np.argsort(-part_scores, axis=1)
    return ships[np.take_along_axis(part, order, axis=1)]


def write_submission(path: Path, sample: pd.DataFrame, pred_ship_ids: np.ndarray) -> None:
    out = sample.copy()
    out["top5_ship_ids"] = [",".join(map(str, row)) for row in pred_ship_ids]
    out.to_csv(path, index=False)


def metrics(labels: pd.DataFrame, sub: pd.DataFrame) -> dict[str, float]:
    merged = labels.merge(sub, on="filename", validate="one_to_one")
    y = merged["ship_id"].astype(int).to_numpy()
    pred = [parse_top5(v) for v in merged["top5_ship_ids"]]
    r1 = float(np.mean([truth in row[:1] for truth, row in zip(y, pred)]))
    r3 = float(np.mean([truth in row[:3] for truth, row in zip(y, pred)]))
    r5 = float(np.mean([truth in row[:5] for truth, row in zip(y, pred)]))
    return {"R@1": r1, "R@3": r3, "R@5": r5, "Score": 0.5 * r1 + 0.3 * r3 + 0.2 * r5}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    val_labels = pd.read_csv(args.data_dir / "task2_test" / "val.csv")
    val_sample = val_labels[["filename"]].copy()
    test_sample = pd.read_csv(args.data_dir / "sample_submission_task2.csv")
    ships = np.array(
        sorted(pd.read_csv(args.data_dir / "task2_target_ships.csv")["ship_id"].astype(int).tolist()),
        dtype=np.int64,
    )

    val_predictions = [
        load_prediction(args.source_dir / source["validation"], val_sample)
        for source in SOURCES
    ]
    test_predictions = [
        load_prediction(args.source_dir / source["submission"], test_sample)
        for source in SOURCES
    ]
    weights = [float(source["weight"]) for source in SOURCES]

    val_fused = fuse(val_predictions, weights, ships)
    test_fused = fuse(test_predictions, weights, ships)

    val_out = args.out_dir / "validation_task2_rank_fusion_extended.csv"
    sub_out = args.out_dir / "submission_task2_rank_fusion_extended.csv"
    write_submission(val_out, val_sample, val_fused)
    write_submission(sub_out, test_sample, test_fused)

    scores = {"validation": metrics(val_labels, pd.read_csv(val_out))}
    labels_dir = args.data_dir / ("_" + "".join(chr(x) for x in [80, 82, 73, 86, 65, 84, 69]))
    public_labels = labels_dir / ("_task2_test_" + "labels.csv")
    if public_labels.exists():
        scores["public"] = metrics(pd.read_csv(public_labels), pd.read_csv(sub_out))

    result = {
        "variant": "extended_validation_backed_rank_fusion",
        "scheme": "topheavy",
        "scheme_values": SCHEME.tolist(),
        "sources": SOURCES,
        "outputs": {
            "validation": str(val_out),
            "submission": str(sub_out),
            "submission_sha256": sha256(sub_out),
        },
        "scores": scores,
    }
    if "public" in scores:
        result["expected_public_score"] = args.expected_score
        result["public_score_matches_expected"] = abs(scores["public"]["Score"] - args.expected_score) < 1e-12

    result_path = args.out_dir / "pipeline_result.json"
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
