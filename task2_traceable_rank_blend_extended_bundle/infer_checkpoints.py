#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parent
CODE_DIR = ROOT / "code"
sys.path.insert(0, str(CODE_DIR))

import task2_train_cached_retrieval as base  # noqa: E402


RUNS = [
    "robust_rows100_seed123_ep2_aug",
    "robust_rows80_seed123_ep2_aug",
    "robust_rows100_seed777_ep2_aug",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("../database"))
    parser.add_argument("--cache-dir", type=Path, default=Path("generated/cache/logmel96_f16000"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--out-runs-dir", type=Path, default=Path("generated/runs"))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def load_run_args(run_dir: Path, data_dir: Path, cache_dir: Path, batch_size: int, num_workers: int) -> SimpleNamespace:
    meta = json.loads((run_dir / "run_meta.json").read_text(encoding="utf-8"))
    saved = meta.get("args", {})
    return SimpleNamespace(
        data_dir=data_dir,
        cache_dir=cache_dir,
        max_target_rows=saved.get("max_target_rows"),
        max_train_rows=saved.get("max_train_rows"),
        no_gallery_train=bool(saved.get("no_gallery_train", False)),
        seed=int(saved.get("seed", 42)),
        emb_dim=int(saved.get("emb_dim", 256)),
        batch_size=batch_size,
        num_workers=num_workers,
    )


@torch.no_grad()
def infer_one(run_name: str, args: argparse.Namespace, device: torch.device) -> dict[str, object]:
    src_run_dir = args.checkpoint_dir / run_name
    if not (src_run_dir / "model_last.pt").exists():
        raise FileNotFoundError(src_run_dir / "model_last.pt")

    run_args = load_run_args(src_run_dir, args.data_dir, args.cache_dir, args.batch_size, args.num_workers)
    _, gallery, val, test, label_map = base.load_cached_tables(run_args)

    checkpoint = torch.load(src_run_dir / "model_last.pt", map_location=device)
    ckpt_label_map = checkpoint.get("label_map", label_map)
    emb_dim = int(checkpoint.get("args", {}).get("emb_dim", run_args.emb_dim))
    model = base.CachedSpecCNN(len(ckpt_label_map), emb_dim).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    out_dir = args.out_runs_dir / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    ref_emb, ref_names = base.embed_df(model, gallery, args.cache_dir / "logmel.npy", run_args, device)
    val_emb, val_names = base.embed_df(model, val, args.cache_dir / "logmel.npy", run_args, device)
    test_emb, test_names = base.embed_df(model, test, args.cache_dir / "logmel.npy", run_args, device)

    np.save(out_dir / "reference_embeddings.npy", ref_emb.astype(np.float32))
    np.save(out_dir / "val_embeddings.npy", val_emb.astype(np.float32))
    np.save(out_dir / "test_embeddings.npy", test_emb.astype(np.float32))
    pd.DataFrame({"filename": ref_names}).to_csv(out_dir / "reference_filenames.csv", index=False)
    pd.DataFrame({"filename": val_names}).to_csv(out_dir / "val_filenames.csv", index=False)
    pd.DataFrame({"filename": test_names}).to_csv(out_dir / "test_filenames.csv", index=False)

    for name in ["run_meta.json", "summary.json", "train_history.csv", "retrieval_leaderboard.csv"]:
        src = src_run_dir / name
        if src.exists():
            shutil.copy2(src, out_dir / name)
    shutil.copy2(src_run_dir / "model_last.pt", out_dir / "model_last.pt")

    return {
        "run": run_name,
        "reference_embeddings": list(ref_emb.shape),
        "val_embeddings": list(val_emb.shape),
        "test_embeddings": list(test_emb.shape),
        "output": str(out_dir),
    }


def main() -> None:
    args = parse_args()
    if args.device == "cuda":
        device = torch.device("cuda")
    elif args.device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args.out_runs_dir.mkdir(parents=True, exist_ok=True)
    results = [infer_one(run, args, device) for run in RUNS]
    result = {"device": str(device), "runs": results}
    out_path = args.out_runs_dir / "inference_result.json"
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
