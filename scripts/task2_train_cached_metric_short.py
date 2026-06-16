#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import task2_train_cached_retrieval as base


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=Path("database"))
    p.add_argument("--cache-dir", type=Path, default=Path("outputs/task2_cache/logmel96_f16000"))
    p.add_argument("--out-dir", type=Path, default=Path("outputs/task2_runs"))
    p.add_argument("--run-name", required=True)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--max-train-minutes", type=float, default=None)
    p.add_argument("--max-target-rows", type=int, default=100)
    p.add_argument("--max-train-rows", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--emb-dim", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=321)
    p.add_argument("--no-gallery-train", action="store_true")
    p.add_argument("--augment", choices=["base", "strong", "none"], default="strong")
    p.add_argument("--supcon-weight", type=float, default=0.0)
    p.add_argument("--temperature", type=float, default=0.07)
    return p.parse_args()


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def strong_augment_spec(x: torch.Tensor):
    # Cache spectrograms are already normalized log-mel tensors. Keep transforms light enough
    # that ship identity is not destroyed, but stronger than the original masking-only setup.
    x = x + torch.randn_like(x) * float(torch.empty(1).uniform_(0.008, 0.03))
    x = x * float(torch.empty(1).uniform_(0.85, 1.15))
    if torch.rand(()) < 0.35:
        shift = int(torch.randint(-12, 13, (1,)).item())
        x = torch.roll(x, shifts=shift, dims=1)
    for _ in range(int(torch.randint(1, 4, (1,)).item())):
        if torch.rand(()) < 0.75:
            width = int(torch.randint(4, 24, (1,)).item())
            start = int(torch.randint(0, max(1, x.shape[1] - width), (1,)).item())
            x[:, start:start + width] = 0
    for _ in range(int(torch.randint(1, 3, (1,)).item())):
        if torch.rand(()) < 0.65:
            width = int(torch.randint(3, 16, (1,)).item())
            start = int(torch.randint(0, max(1, x.shape[0] - width), (1,)).item())
            x[start:start + width, :] = 0
    return x


class MetricSpecDataset(Dataset):
    def __init__(self, df: pd.DataFrame, cache_path: Path, with_label: bool, augment: str):
        self.df = df.reset_index(drop=True)
        self.cache = np.load(cache_path, mmap_mode="r")
        self.indices = self.df["cache_index"].astype(int).to_numpy()
        self.with_label = with_label
        self.augment = augment
        self.labels = self.df["label"].astype(int).to_numpy() if with_label and "label" in self.df else None
        self.filenames = self.df["filename"].astype(str).tolist()

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        x = torch.from_numpy(np.asarray(self.cache[self.indices[idx]], dtype=np.float32))
        if self.augment == "base":
            x = base.augment_spec(x)
        elif self.augment == "strong":
            x = strong_augment_spec(x)
        if self.with_label:
            return x, int(self.labels[idx])
        return x, self.filenames[idx]


def supervised_contrastive_loss(emb: torch.Tensor, labels: torch.Tensor, temperature: float):
    labels = labels.view(-1, 1)
    mask = torch.eq(labels, labels.T).float().to(emb.device)
    logits = emb @ emb.T / temperature
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    logits_mask = torch.ones_like(mask) - torch.eye(mask.shape[0], device=emb.device)
    mask = mask * logits_mask
    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12))
    positives = mask.sum(dim=1)
    valid = positives > 0
    if not torch.any(valid):
        return emb.new_tensor(0.0)
    mean_log_prob_pos = (mask * log_prob).sum(dim=1)[valid] / positives[valid]
    return -mean_log_prob_pos.mean()


def train_model(model, train_df, cache_path, args, device, run_dir):
    ds = MetricSpecDataset(train_df, cache_path, with_label=True, augment=args.augment)
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        sampler=base.make_sampler(train_df),
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = max(1, args.epochs * math.ceil(len(train_df) / args.batch_size))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps)
    rows = []
    start = time.time()
    steps = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses, ce_losses, con_losses, correct, total = [], [], [], 0, 0
        for spec, y in dl:
            if args.max_train_minutes and (time.time() - start) / 60 >= args.max_train_minutes:
                break
            spec = spec.to(device, non_blocking=True)
            y = y.long().to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            logits, emb = model(spec)
            ce = F.cross_entropy(logits, y, label_smoothing=0.05)
            con = supervised_contrastive_loss(emb, y, args.temperature) if args.supcon_weight > 0 else emb.new_tensor(0.0)
            loss = ce + args.supcon_weight * con
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            steps += 1
            losses.append(float(loss.detach().cpu()))
            ce_losses.append(float(ce.detach().cpu()))
            con_losses.append(float(con.detach().cpu()))
            correct += int((logits.argmax(1) == y).sum().detach().cpu())
            total += int(y.numel())
        row = {
            "epoch": epoch,
            "steps": steps,
            "train_loss": float(np.mean(losses)) if losses else None,
            "ce_loss": float(np.mean(ce_losses)) if ce_losses else None,
            "supcon_loss": float(np.mean(con_losses)) if con_losses else None,
            "train_accuracy": float(correct / total) if total else None,
            "elapsed_min": float((time.time() - start) / 60),
        }
        rows.append(row)
        print(json.dumps({"train": row}, ensure_ascii=False), flush=True)
        if args.max_train_minutes and (time.time() - start) / 60 >= args.max_train_minutes:
            break
    pd.DataFrame(rows).to_csv(run_dir / "train_history.csv", index=False)


@torch.no_grad()
def embed_df(model, df, cache_path, args, device):
    ds = MetricSpecDataset(df, cache_path, with_label=False, augment="none")
    dl = DataLoader(
        ds,
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    model.eval()
    embs, names = [], []
    for spec, filename in dl:
        spec = spec.to(device, non_blocking=True)
        embs.append(model.features(spec).detach().cpu().numpy().astype(np.float32))
        names.extend(filename)
    return np.vstack(embs), names


def main():
    args = parse_args()
    seed_all(args.seed)
    run_dir = args.out_dir / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    cache_path = args.cache_dir / "logmel.npy"
    train, gallery, val, test, label_map = base.load_cached_tables(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    model = base.CachedSpecCNN(len(label_map), args.emb_dim).to(device)
    meta = {
        "run_name": args.run_name,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "rows": {"train": len(train), "gallery": len(gallery), "reference": len(gallery), "val": len(val), "test": len(test)},
    }
    (run_dir / "run_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False), flush=True)
    train_model(model, train, cache_path, args, device, run_dir)
    torch.save({"model_state": model.state_dict(), "label_map": label_map, "args": meta["args"]}, run_dir / "model_last.pt")
    ref_emb, ref_names = embed_df(model, gallery, cache_path, args, device)
    val_emb, val_names = embed_df(model, val, cache_path, args, device)
    test_emb, test_names = embed_df(model, test, cache_path, args, device)
    np.save(run_dir / "reference_embeddings.npy", ref_emb)
    np.save(run_dir / "val_embeddings.npy", val_emb)
    np.save(run_dir / "test_embeddings.npy", test_emb)

    ref_ship_ids = gallery["ship_id"].astype(int).to_numpy()
    y = val["ship_id"].astype(int).to_numpy()
    rows, _, _ = base.evaluate(val_emb, ref_emb, ref_ship_ids, y)
    pd.DataFrame(rows).to_csv(run_dir / "retrieval_leaderboard.csv", index=False)
    best = rows[0]
    if best["retrieval"] == "hybrid":
        weights = tuple(float(x.strip()) for x in best["weights"].strip("()").split(","))
        test_scores, ships = base.hybrid(test_emb, ref_emb, ref_ship_ids, weights, int(best["topk"]), float(best["alpha"]))
    elif best["retrieval"] == "topk":
        test_scores, ships = base.score_matrix(test_emb, ref_emb, ref_ship_ids, "topk", topk=int(best["topk"]), alpha=float(best["alpha"]))
    else:
        test_scores, ships = base.score_matrix(test_emb, ref_emb, ref_ship_ids, best["retrieval"], alpha=float(best["alpha"]))
    test_pred = base.top5(test_scores, ships)
    sample = pd.read_csv(args.data_dir / "sample_submission_task2.csv")
    pred_map = dict(zip(test_names, [",".join(map(str, row)) for row in test_pred]))
    sub = sample.copy()
    sub["top5_ship_ids"] = sub["filename"].map(pred_map)
    errors = base.validate_submission(sub, sample, set(label_map.keys()))
    sub.to_csv(run_dir / "submission_task2.csv", index=False)
    summary = {
        "run_meta": meta,
        "best_validation": best,
        "submission_validation_errors": errors,
        "artifacts": {
            "submission": str(run_dir / "submission_task2.csv"),
            "leaderboard": str(run_dir / "retrieval_leaderboard.csv"),
            "model": str(run_dir / "model_last.pt"),
        },
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
