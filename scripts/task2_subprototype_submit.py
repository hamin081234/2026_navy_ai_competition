#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
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
    p.add_argument("--out-dir", type=Path, default=Path("outputs/task2_subprototype_clipcount_penalty"))
    p.add_argument("--run", default="quick_target100_mel96_f16000")
    p.add_argument("--method", default="clip_max")
    p.add_argument("--alpha", type=float, default=0.03)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--submission-name", default="submission_task2_best_subprototype.csv")
    return p.parse_args()


def validate_submission(sub: pd.DataFrame, sample: pd.DataFrame, target_ids: set[int]) -> list[str]:
    errors = []
    if len(sub) != len(sample):
        errors.append("row count mismatch")
    if not sub["filename"].equals(sample["filename"]):
        errors.append("filename order mismatch")
    for idx, value in enumerate(sub["top5_ship_ids"].astype(str)):
        parts = value.split(",")
        if len(parts) != 5:
            errors.append(f"row {idx}: expected 5 ids")
            continue
        ids = [int(x) for x in parts]
        if len(set(ids)) != 5:
            errors.append(f"row {idx}: duplicate ids")
        if any(x not in target_ids for x in ids):
            errors.append(f"row {idx}: non-target id")
        if len(errors) >= 20:
            break
    return errors


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    run_dir = args.runs_dir / args.run
    gallery = spe.add_bins(pd.read_csv(args.data_dir / "task2_test/gallery.csv"))
    test = pd.read_csv(args.data_dir / "task2_test/test.csv")
    sample = pd.read_csv(args.data_dir / "sample_submission_task2.csv")
    target_ids = set(pd.read_csv(args.data_dir / "task2_target_ships.csv")["ship_id"].astype(int).tolist())

    ref_emb = spe.l2(np.load(run_dir / "reference_embeddings.npy").astype(np.float32))
    test_emb = spe.l2(np.load(run_dir / "test_embeddings.npy").astype(np.float32))
    proto_emb, proto_ship_ids, proto_meta = spe.make_prototypes(args.method, ref_emb, gallery, args.seed)
    proto_source_clips = proto_meta["source_clips"].astype(float).to_numpy()
    scores, ships = spe.score_by_ship(test_emb, proto_emb, proto_ship_ids, proto_source_clips, args.alpha)
    pred = spe.top5_from_scores(scores, ships)

    sub = sample.copy()
    pred_map = dict(zip(test["filename"].astype(str), [",".join(map(str, row)) for row in pred]))
    sub["top5_ship_ids"] = sub["filename"].map(pred_map)
    errors = validate_submission(sub, sample, target_ids)
    out_path = args.out_dir / args.submission_name
    sub.to_csv(out_path, index=False)

    summary = {
        "run": args.run,
        "method": args.method,
        "alpha": args.alpha,
        "num_prototypes": int(len(proto_ship_ids)),
        "submission": str(out_path),
        "validation_errors": errors,
    }
    (args.out_dir / "submission_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
