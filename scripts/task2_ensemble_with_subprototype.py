#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))
import task2_subprototype_eval as spe  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=Path("database"))
    p.add_argument("--runs-dir", type=Path, default=Path("outputs/task2_runs"))
    p.add_argument("--out-dir", type=Path, default=Path("outputs/task2_runs/ensemble_with_subprototype"))
    p.add_argument("--normal-runs", nargs="*", default=[])
    p.add_argument(
        "--subproto",
        nargs="*",
        default=[],
        help="Specs like run_name:method:alpha, e.g. quick_target100_mel96_f16000:clip_max:0.03",
    )
    p.add_argument("--weight-step", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
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


def run_normal_score(run_dir: Path, ref_ship_ids, split: str):
    query = np.load(run_dir / f"{split}_embeddings.npy").astype(np.float32)
    ref = np.load(run_dir / "reference_embeddings.npy").astype(np.float32)
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
    return scores, ships, best


def run_subproto_score(run_dir: Path, gallery: pd.DataFrame, method: str, alpha: float, split: str, seed: int):
    ref = spe.l2(np.load(run_dir / "reference_embeddings.npy").astype(np.float32))
    query = spe.l2(np.load(run_dir / f"{split}_embeddings.npy").astype(np.float32))
    proto_emb, proto_ship_ids, proto_meta = spe.make_prototypes(method, ref, gallery, seed)
    proto_source_clips = proto_meta["source_clips"].astype(float).to_numpy()
    scores, ships = spe.score_by_ship(query, proto_emb, proto_ship_ids, proto_source_clips, alpha)
    best = {"retrieval": f"subproto:{method}", "topk": None, "weights": None, "alpha": alpha}
    return scores.astype(np.float32), ships, best


def normalize_scores(scores):
    mean = scores.mean(axis=1, keepdims=True)
    std = scores.std(axis=1, keepdims=True) + 1e-6
    return (scores - mean) / std


def simplex_weights(n: int, step: float):
    units = int(round(1.0 / step))

    def rec(prefix, remaining_units, remaining_slots):
        if remaining_slots == 1:
            yield tuple(prefix + [remaining_units / units])
            return
        for value in range(remaining_units + 1):
            yield from rec(prefix + [value / units], remaining_units - value, remaining_slots - 1)

    if n == 1:
        yield (1.0,)
        return
    yield from rec([], units, n)


def top5(scores, ships):
    return ships[np.argsort(-scores, axis=1)[:, :5]]


def metrics(y, pred):
    r1 = float(np.mean([yy in row[:1] for yy, row in zip(y, pred)]))
    r3 = float(np.mean([yy in row[:3] for yy, row in zip(y, pred)]))
    r5 = float(np.mean([yy in row[:5] for yy, row in zip(y, pred)]))
    return {"R@1": r1, "R@3": r3, "R@5": r5, "Score": 0.5 * r1 + 0.3 * r3 + 0.2 * r5}


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
        if len(errors) >= 20:
            break
    return errors


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    gallery_raw = pd.read_csv(args.data_dir / "task2_test/gallery.csv")
    gallery = spe.add_bins(gallery_raw)
    val = pd.read_csv(args.data_dir / "task2_test/val.csv")
    test = pd.read_csv(args.data_dir / "task2_test/test.csv")
    sample = pd.read_csv(args.data_dir / "sample_submission_task2.csv")
    targets = set(pd.read_csv(args.data_dir / "task2_target_ships.csv")["ship_id"].astype(int))
    ref_ship_ids = gallery_raw["ship_id"].astype(int).to_numpy()
    y = val["ship_id"].astype(int).to_numpy()

    val_scores = []
    test_scores = []
    candidate_rows = []
    ships_ref = None

    for run in args.normal_runs:
        run_dir = args.runs_dir / run
        vs, ships, best = run_normal_score(run_dir, ref_ship_ids, "val")
        ts, ships_t, _ = run_normal_score(run_dir, ref_ship_ids, "test")
        if ships_ref is None:
            ships_ref = ships
        if not np.array_equal(ships_ref, ships) or not np.array_equal(ships_ref, ships_t):
            raise RuntimeError(f"Ship order mismatch for {run}")
        val_scores.append(normalize_scores(vs))
        test_scores.append(normalize_scores(ts))
        candidate_rows.append({
            "candidate": run,
            "kind": "normal",
            **best,
            **{f"single_{k}": v for k, v in metrics(y, top5(vs, ships)).items()},
        })

    for spec in args.subproto:
        run, method, alpha_text = spec.split(":")
        alpha = float(alpha_text)
        run_dir = args.runs_dir / run
        vs, ships, best = run_subproto_score(run_dir, gallery, method, alpha, "val", args.seed)
        ts, ships_t, _ = run_subproto_score(run_dir, gallery, method, alpha, "test", args.seed)
        if ships_ref is None:
            ships_ref = ships
        if not np.array_equal(ships_ref, ships) or not np.array_equal(ships_ref, ships_t):
            raise RuntimeError(f"Ship order mismatch for {spec}")
        val_scores.append(normalize_scores(vs))
        test_scores.append(normalize_scores(ts))
        candidate_rows.append({
            "candidate": spec,
            "kind": "subprototype",
            **best,
            **{f"single_{k}": v for k, v in metrics(y, top5(vs, ships)).items()},
        })

    rows = []
    best = None
    for weights in simplex_weights(len(val_scores), args.weight_step):
        vs = sum(w * s for w, s in zip(weights, val_scores))
        pred = top5(vs, ships_ref)
        row = {"weights": str(weights), **metrics(y, pred)}
        rows.append(row)
        if best is None or row["Score"] > best["Score"]:
            best = row

    leaderboard = pd.DataFrame(rows).sort_values(["Score", "R@1", "R@3"], ascending=False)
    leaderboard.to_csv(args.out_dir / "ensemble_leaderboard.csv", index=False)
    pd.DataFrame(candidate_rows).to_csv(args.out_dir / "ensemble_input_candidates.csv", index=False)

    weights = tuple(float(x.strip()) for x in best["weights"].strip("()").split(","))
    final_test_scores = sum(w * s for w, s in zip(weights, test_scores))
    test_top5 = top5(final_test_scores, ships_ref)
    pred_map = dict(zip(test["filename"], [",".join(map(str, row)) for row in test_top5]))
    sub = sample.copy()
    sub["top5_ship_ids"] = sub["filename"].map(pred_map)
    errors = validate_submission(sub, sample, targets)
    submission_path = args.out_dir / "submission_task2_ensemble_subprototype.csv"
    sub.to_csv(submission_path, index=False)

    summary = {
        "normal_runs": args.normal_runs,
        "subprototype_specs": args.subproto,
        "weight_step": args.weight_step,
        "best_validation": best,
        "submission_validation_errors": errors,
        "submission": str(submission_path),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
