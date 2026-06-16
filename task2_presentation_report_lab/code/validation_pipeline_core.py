#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
VAL_LABEL_PATH = ROOT / "database/task2_test/val.csv"

RANK_POSITION_WEIGHTS = np.array([1.0, 0.62, 0.40, 0.25, 0.16], dtype=np.float32)


@dataclass(frozen=True)
class ValidationSource:
    name: str
    path: Path
    weight: float


VALIDATION_SOURCES = [
    ValidationSource(
        name="tree_reranker_random_forest",
        path=ROOT
        / "task2_validation_first_fusion_lab/sources/validation_first_manual_fusion_reproduced/"
        / "step00__validation__validation_random_forest.csv",
        weight=1.00,
    ),
    ValidationSource(
        name="panns_type_aware_projection",
        path=ROOT
        / "task2_validation_first_fusion_lab/sources/validation_first_manual_fusion_reproduced/"
        / "step01__validation__validation_epoch1_config37.csv",
        weight=0.45,
    ),
    ValidationSource(
        name="extended_hgb_wide_reranker",
        path=ROOT
        / "task2_validation_first_fusion_lab/sources/validation_first_manual_fusion_reproduced/"
        / "step02__validation__validation_hgb_wide.csv",
        weight=0.10,
    ),
    ValidationSource(
        name="reciprocal_rank_refined_source",
        path=ROOT
        / "task2_validation_first_fusion_lab/sources/validation_first_manual_fusion_reproduced/"
        / "step03__validation__validation_refined_reciprocal.csv",
        weight=0.20,
    ),
    ValidationSource(
        name="type_gated_retrieval",
        path=ROOT
        / "task2_validation_first_fusion_lab/sources/validation_first_manual_fusion_reproduced/"
        / "step04__validation__validation_soft_topk_15_topk_20_b0p5.csv",
        weight=0.08,
    ),
]


def parse_top5(value: str) -> list[int]:
    ids = [int(part.strip()) for part in str(value).split(",")]
    if len(ids) != 5:
        raise ValueError(f"top5 must contain exactly 5 ids: {value!r}")
    if len(set(ids)) != 5:
        raise ValueError(f"top5 contains duplicated ids: {value!r}")
    return ids


def load_prediction(path: Path, expected_filenames: pd.Series) -> list[list[int]]:
    df = pd.read_csv(path)
    required = {"filename", "top5_ship_ids"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    if len(df) != len(expected_filenames):
        raise ValueError(f"{path} row count mismatch: {len(df)} != {len(expected_filenames)}")
    if not df["filename"].astype(str).reset_index(drop=True).equals(expected_filenames.astype(str).reset_index(drop=True)):
        raise ValueError(f"{path} filename order mismatch")
    return [parse_top5(value) for value in df["top5_ship_ids"]]


def add_rank_scores(
    score_matrix: np.ndarray,
    ship_to_col: dict[int, int],
    predictions: list[list[int]],
    source_weight: float,
) -> None:
    for row_idx, top5 in enumerate(predictions):
        for rank_idx, ship_id in enumerate(top5):
            score_matrix[row_idx, ship_to_col[ship_id]] += source_weight * RANK_POSITION_WEIGHTS[rank_idx]


def fuse_sources(sources: list[ValidationSource], labels: pd.DataFrame) -> np.ndarray:
    target_ships = sorted(labels["ship_id"].astype(int).unique().tolist())
    ship_to_col = {ship_id: idx for idx, ship_id in enumerate(target_ships)}
    score_matrix = np.zeros((len(labels), len(target_ships)), dtype=np.float32)

    expected_filenames = labels["filename"].astype(str)
    for source in sources:
        predictions = load_prediction(source.path, expected_filenames)
        add_rank_scores(score_matrix, ship_to_col, predictions, source.weight)

    order = np.argsort(-score_matrix, axis=1)[:, :5]
    ships = np.array(target_ships, dtype=int)
    return ships[order]


def recall_at_k(y_true: np.ndarray, y_pred_top5: np.ndarray, k: int) -> float:
    return float(np.mean([truth in row[:k] for truth, row in zip(y_true, y_pred_top5)]))


def evaluate(labels: pd.DataFrame, prediction: np.ndarray) -> dict[str, float]:
    y_true = labels["ship_id"].astype(int).to_numpy()
    r1 = recall_at_k(y_true, prediction, 1)
    r3 = recall_at_k(y_true, prediction, 3)
    r5 = recall_at_k(y_true, prediction, 5)
    return {
        "R@1": r1,
        "R@3": r3,
        "R@5": r5,
        "Score": 0.5 * r1 + 0.3 * r3 + 0.2 * r5,
    }


def main() -> None:
    labels = pd.read_csv(VAL_LABEL_PATH)
    prediction = fuse_sources(VALIDATION_SOURCES, labels)
    metrics = evaluate(labels, prediction)

    print("Validation sources:")
    for source in VALIDATION_SOURCES:
        print(f"- {source.name}: weight={source.weight}, path={source.path.relative_to(ROOT)}")
    print("\nMetrics:")
    for key, value in metrics.items():
        print(f"{key}: {value:.15f}")


if __name__ == "__main__":
    main()

