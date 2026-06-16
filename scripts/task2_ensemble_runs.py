#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=Path("database"))
    p.add_argument("--out-dir", type=Path, default=Path("outputs/task2_runs/ensemble"))
    p.add_argument("--runs", nargs="+", required=True)
    return p.parse_args()


def l2(x):
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-8, None)


def score_matrix(query_emb, ref_emb, ref_ship_ids, method, topk=None, alpha=0.0):
    query_emb = l2(query_emb)
    ref_emb = l2(ref_emb)
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
        elif method == "hybrid":
            raise ValueError("hybrid is handled separately")
        else:
            raise ValueError(method)
        if alpha:
            scores[:, j] -= alpha * math.log1p(len(cols))
    return scores, ships


def hybrid_score(query_emb, ref_emb, ref_ship_ids, weights, topk, alpha):
    proto, ships = score_matrix(query_emb, ref_emb, ref_ship_ids, "prototype", alpha=0.0)
    top, _ = score_matrix(query_emb, ref_emb, ref_ship_ids, "topk", topk=topk, alpha=0.0)
    mx, _ = score_matrix(query_emb, ref_emb, ref_ship_ids, "max", alpha=0.0)
    scores = weights[0] * proto + weights[1] * top + weights[2] * mx
    if alpha:
        for j, sid in enumerate(ships):
            n = int(np.sum(ref_ship_ids == sid))
            scores[:, j] -= alpha * math.log1p(n)
    return scores, ships


def top5(scores, ships):
    return ships[np.argsort(-scores, axis=1)[:, :5]]


def metrics(y, pred):
    r1 = float(np.mean([yy in row[:1] for yy, row in zip(y, pred)]))
    r3 = float(np.mean([yy in row[:3] for yy, row in zip(y, pred)]))
    r5 = float(np.mean([yy in row[:5] for yy, row in zip(y, pred)]))
    return {"R@1": r1, "R@3": r3, "R@5": r5, "Score": 0.5 * r1 + 0.3 * r3 + 0.2 * r5}


def parse_weights(value):
    if pd.isna(value) or value in ("", "nan", None):
        return None
    return tuple(float(x.strip()) for x in str(value).strip("()").split(","))


def run_score(run_dir: Path, ref_ship_ids, split: str):
    query = np.load(run_dir / f"{split}_embeddings.npy")
    ref = np.load(run_dir / "reference_embeddings.npy")
    leaderboard = pd.read_csv(run_dir / "retrieval_leaderboard.csv")
    best = leaderboard.iloc[0].to_dict()
    method = best["retrieval"]
    alpha = float(best["alpha"])
    if method == "hybrid":
        scores, ships = hybrid_score(query, ref, ref_ship_ids, parse_weights(best["weights"]), int(best["topk"]), alpha)
    elif method == "topk":
        scores, ships = score_matrix(query, ref, ref_ship_ids, "topk", topk=int(best["topk"]), alpha=alpha)
    else:
        scores, ships = score_matrix(query, ref, ref_ship_ids, method, alpha=alpha)
    return scores.astype(np.float32), ships, best


def normalize_scores(scores):
    mean = scores.mean(axis=1, keepdims=True)
    std = scores.std(axis=1, keepdims=True) + 1e-6
    return (scores - mean) / std


def candidate_weights(n):
    if n == 1:
        return [(1.0,)]
    values = [0.0, 0.25, 0.5, 0.75, 1.0]
    out = []
    for raw in itertools.product(values, repeat=n):
        s = sum(raw)
        if s <= 0:
            continue
        w = tuple(x / s for x in raw)
        if w not in out:
            out.append(w)
    return out


def validate_submission(sub, sample, target_ids):
    errors = []
    if len(sub) != len(sample):
        errors.append("row_count mismatch")
    if not sub["filename"].equals(sample["filename"]):
        errors.append("filename order mismatch")
    for i, value in enumerate(sub["top5_ship_ids"].astype(str)):
        parts = value.split(",")
        if len(parts) != 5:
            errors.append(f"row {i} wrong top5 length")
            continue
        ids = [int(x) for x in parts]
        if len(set(ids)) != 5:
            errors.append(f"row {i} duplicate ids")
        if any(x not in target_ids for x in ids):
            errors.append(f"row {i} non-target id")
    return errors[:20]


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    gallery = pd.read_csv(args.data_dir / "task2_test/gallery.csv")
    val = pd.read_csv(args.data_dir / "task2_test/val.csv")
    test = pd.read_csv(args.data_dir / "task2_test/test.csv")
    sample = pd.read_csv(args.data_dir / "sample_submission_task2.csv")
    targets = set(pd.read_csv(args.data_dir / "task2_target_ships.csv")["ship_id"].astype(int))
    ref_ship_ids = gallery["ship_id"].astype(int).to_numpy()
    y = val["ship_id"].astype(int).to_numpy()

    val_scores = []
    test_scores = []
    run_rows = []
    ships_ref = None
    for run in args.runs:
        run_dir = Path(run)
        vs, ships, best = run_score(run_dir, ref_ship_ids, "val")
        ts, ships_t, _ = run_score(run_dir, ref_ship_ids, "test")
        if ships_ref is None:
            ships_ref = ships
        if not np.array_equal(ships_ref, ships) or not np.array_equal(ships_ref, ships_t):
            raise RuntimeError("Ship order mismatch across runs")
        val_scores.append(normalize_scores(vs))
        test_scores.append(normalize_scores(ts))
        pred = top5(vs, ships)
        run_rows.append({"run": str(run_dir), **best, **{f"single_{k}": v for k, v in metrics(y, pred).items()}})

    rows = []
    best = None
    for weights in candidate_weights(len(val_scores)):
        vs = sum(w * s for w, s in zip(weights, val_scores))
        pred = top5(vs, ships_ref)
        row = {"weights": str(weights), **metrics(y, pred)}
        rows.append(row)
        if best is None or row["Score"] > best["Score"]:
            best = row

    leaderboard = pd.DataFrame(rows).sort_values("Score", ascending=False)
    leaderboard.to_csv(args.out_dir / "ensemble_leaderboard.csv", index=False)
    pd.DataFrame(run_rows).to_csv(args.out_dir / "ensemble_input_runs.csv", index=False)

    weights = tuple(float(x.strip()) for x in best["weights"].strip("()").split(","))
    final_test_scores = sum(w * s for w, s in zip(weights, test_scores))
    test_top5 = top5(final_test_scores, ships_ref)
    pred_map = dict(zip(test["filename"], [",".join(map(str, row)) for row in test_top5]))
    sub = sample.copy()
    sub["top5_ship_ids"] = sub["filename"].map(pred_map)
    errors = validate_submission(sub, sample, targets)
    sub.to_csv(args.out_dir / "submission_task2_ensemble.csv", index=False)
    summary = {
        "runs": args.runs,
        "best_validation": best,
        "submission_validation_errors": errors,
        "submission": str(args.out_dir / "submission_task2_ensemble.csv"),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    with Path("analysis/task2_execution_log.md").open("a", encoding="utf-8") as f:
        f.write(
            f"""
## Ensemble Run

저장된 run embedding을 이용해 score matrix ensemble을 수행했다.

입력 run:

{chr(10).join(f'- `{r}`' for r in args.runs)}

Best ensemble validation:

- weights: `{best['weights']}`
- R@1: `{best['R@1']:.6f}`
- R@3: `{best['R@3']:.6f}`
- R@5: `{best['R@5']:.6f}`
- Score: `{best['Score']:.6f}`

submission validation errors: `{errors}`

"""
        )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
