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
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=Path("database"))
    p.add_argument("--cache-dir", type=Path, default=Path("outputs/task2_cache/logmel96_f16000"))
    p.add_argument("--out-dir", type=Path, default=Path("outputs/task2_runs"))
    p.add_argument("--run-name", default="cached_cnn_logmel96_f16000")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--max-train-minutes", type=float, default=None)
    p.add_argument("--max-train-rows", type=int, default=None)
    p.add_argument("--max-target-rows", type=int, default=None, help="Limit train_target rows before adding gallery.")
    p.add_argument("--no-augment", action="store_true")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--emb-dim", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-gallery-train", action="store_true")
    return p.parse_args()


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class CachedSpecDataset(Dataset):
    def __init__(self, df: pd.DataFrame, cache_path: Path, with_label: bool, augment: bool):
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
        if self.augment:
            x = augment_spec(x)
        if self.with_label:
            return x, int(self.labels[idx])
        return x, self.filenames[idx]


def augment_spec(x: torch.Tensor):
    x = x + torch.randn_like(x) * 0.015
    if torch.rand(()) < 0.5:
        width = int(torch.randint(4, 18, (1,)).item())
        start = int(torch.randint(0, max(1, x.shape[1] - width), (1,)).item())
        x[:, start:start + width] = 0
    if torch.rand(()) < 0.5:
        width = int(torch.randint(3, 12, (1,)).item())
        start = int(torch.randint(0, max(1, x.shape[0] - width), (1,)).item())
        x[start:start + width, :] = 0
    return x


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
            nn.MaxPool2d(2),
        )

    def forward(self, x):
        return self.net(x)


class CachedSpecCNN(nn.Module):
    def __init__(self, n_classes: int, emb_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            ConvBlock(1, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 192),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.embedding = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(192, emb_dim, bias=False),
            nn.BatchNorm1d(emb_dim),
        )
        self.classifier = nn.Linear(emb_dim, n_classes)

    def features(self, spec):
        emb = self.embedding(self.encoder(spec.unsqueeze(1)))
        return F.normalize(emb, dim=1)

    def forward(self, spec):
        emb = self.features(spec)
        return self.classifier(emb), emb


def load_cached_tables(args):
    meta = pd.read_csv(args.cache_dir / "metadata.csv")
    targets = sorted(pd.read_csv(args.data_dir / "task2_target_ships.csv")["ship_id"].astype(int).tolist())
    label_map = {sid: i for i, sid in enumerate(targets)}
    for split in ["train_target", "gallery", "val"]:
        mask = meta["split"] == split
        meta.loc[mask, "label"] = meta.loc[mask, "ship_id"].astype(int).map(label_map)
    train = meta[meta["split"] == "train_target"].copy()
    gallery = meta[meta["split"] == "gallery"].copy()
    if args.max_target_rows is not None and args.max_target_rows < len(train):
        original_train = train.copy()
        per_ship = max(1, args.max_target_rows // train["ship_id"].nunique())
        parts = []
        for _, group in train.groupby("ship_id"):
            parts.append(group.sample(n=min(per_ship, len(group)), random_state=args.seed))
        train = pd.concat(parts, ignore_index=False)
        if len(train) < args.max_target_rows:
            remaining = original_train.drop(index=train.index, errors="ignore")
            if len(remaining) > 0:
                extra = remaining.sample(n=min(args.max_target_rows - len(train), len(remaining)), random_state=args.seed)
                train = pd.concat([train, extra], ignore_index=False)
        if len(train) > args.max_target_rows:
            train = train.sample(n=args.max_target_rows, random_state=args.seed)
        train = train.reset_index(drop=True)
    val = meta[meta["split"] == "val"].copy()
    test = meta[meta["split"] == "test"].copy()
    if not args.no_gallery_train:
        train = pd.concat([train, gallery], ignore_index=True, sort=False)
    if args.max_train_rows is not None and args.max_train_rows < len(train):
        per_ship = max(1, args.max_train_rows // train["ship_id"].nunique())
        parts = []
        for _, group in train.groupby("ship_id"):
            parts.append(group.sample(n=min(per_ship, len(group)), random_state=args.seed))
        train = pd.concat(parts, ignore_index=True)
        if len(train) > args.max_train_rows:
            train = train.sample(n=args.max_train_rows, random_state=args.seed)
    return train.reset_index(drop=True), gallery.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True), label_map


def make_sampler(df):
    counts = df["label"].astype(int).value_counts().to_dict()
    weights = df["label"].astype(int).map(lambda x: 1.0 / counts[int(x)]).to_numpy(dtype=np.float64)
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


def train_model(model, train_df, cache_path, args, device, run_dir):
    ds = CachedSpecDataset(train_df, cache_path, with_label=True, augment=not args.no_augment)
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        sampler=make_sampler(train_df),
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
        losses, correct, total = [], 0, 0
        for spec, y in dl:
            if args.max_train_minutes and (time.time() - start) / 60 >= args.max_train_minutes:
                break
            spec = spec.to(device, non_blocking=True)
            y = y.long().to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            logits, _ = model(spec)
            loss = F.cross_entropy(logits, y, label_smoothing=0.05)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            steps += 1
            losses.append(float(loss.detach().cpu()))
            correct += int((logits.argmax(1) == y).sum().detach().cpu())
            total += int(y.numel())
        row = {
            "epoch": epoch,
            "steps": steps,
            "train_loss": float(np.mean(losses)) if losses else None,
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
    ds = CachedSpecDataset(df, cache_path, with_label=False, augment=False)
    dl = DataLoader(ds, batch_size=args.batch_size * 2, shuffle=False, num_workers=args.num_workers, pin_memory=torch.cuda.is_available())
    model.eval()
    embs, names = [], []
    for spec, filename in dl:
        spec = spec.to(device, non_blocking=True)
        embs.append(model.features(spec).detach().cpu().numpy().astype(np.float32))
        names.extend(filename)
    return np.vstack(embs), names


def l2(x):
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-8, None)


def score_matrix(query_emb, ref_emb, ref_ship_ids, method, topk=5, alpha=0.0):
    query_emb, ref_emb = l2(query_emb), l2(ref_emb)
    ships = np.array(sorted(set(ref_ship_ids.tolist())), dtype=int)
    sim = query_emb @ ref_emb.T
    out = np.zeros((len(query_emb), len(ships)), dtype=np.float32)
    for j, sid in enumerate(ships):
        cols = np.where(ref_ship_ids == sid)[0]
        vals = sim[:, cols]
        if method == "prototype":
            proto = l2(ref_emb[cols].mean(axis=0, keepdims=True))
            out[:, j] = (query_emb @ proto.T).reshape(-1)
        elif method == "max":
            out[:, j] = vals.max(axis=1)
        elif method == "topk":
            k = min(topk, vals.shape[1])
            part = np.partition(vals, kth=vals.shape[1] - k, axis=1)[:, -k:]
            out[:, j] = part.mean(axis=1)
        if alpha:
            out[:, j] -= alpha * math.log1p(len(cols))
    return out, ships


def hybrid(query_emb, ref_emb, ref_ship_ids, weights, topk, alpha):
    p, ships = score_matrix(query_emb, ref_emb, ref_ship_ids, "prototype", alpha=0)
    t, _ = score_matrix(query_emb, ref_emb, ref_ship_ids, "topk", topk=topk, alpha=0)
    m, _ = score_matrix(query_emb, ref_emb, ref_ship_ids, "max", alpha=0)
    s = weights[0] * p + weights[1] * t + weights[2] * m
    if alpha:
        for j, sid in enumerate(ships):
            s[:, j] -= alpha * math.log1p(int(np.sum(ref_ship_ids == sid)))
    return s, ships


def top5(scores, ships):
    return ships[np.argsort(-scores, axis=1)[:, :5]]


def metric(y, pred):
    r1 = float(np.mean([yy in row[:1] for yy, row in zip(y, pred)]))
    r3 = float(np.mean([yy in row[:3] for yy, row in zip(y, pred)]))
    r5 = float(np.mean([yy in row[:5] for yy, row in zip(y, pred)]))
    return {"R@1": r1, "R@3": r3, "R@5": r5, "Score": 0.5 * r1 + 0.3 * r3 + 0.2 * r5}


def evaluate(val_emb, ref_emb, ref_ship_ids, y):
    rows, candidates = [], []
    for method in ["prototype", "max"]:
        for alpha in [0, 0.005, 0.01, 0.02, 0.03]:
            scores, ships = score_matrix(val_emb, ref_emb, ref_ship_ids, method, alpha=alpha)
            row = {"retrieval": method, "topk": None, "weights": None, "alpha": alpha, **metric(y, top5(scores, ships))}
            rows.append(row); candidates.append((row, scores, ships))
    for topk in [3, 5, 10]:
        for alpha in [0, 0.005, 0.01, 0.02, 0.03]:
            scores, ships = score_matrix(val_emb, ref_emb, ref_ship_ids, "topk", topk=topk, alpha=alpha)
            row = {"retrieval": "topk", "topk": topk, "weights": None, "alpha": alpha, **metric(y, top5(scores, ships))}
            rows.append(row); candidates.append((row, scores, ships))
    for topk in [3, 5, 10]:
        for weights in [(0.5, 0.3, 0.2), (0.4, 0.4, 0.2), (0.3, 0.5, 0.2), (0.6, 0.3, 0.1)]:
            for alpha in [0, 0.005, 0.01, 0.02, 0.03]:
                scores, ships = hybrid(val_emb, ref_emb, ref_ship_ids, weights, topk, alpha)
                row = {"retrieval": "hybrid", "topk": topk, "weights": str(weights), "alpha": alpha, **metric(y, top5(scores, ships))}
                rows.append(row); candidates.append((row, scores, ships))
    rows = sorted(rows, key=lambda x: x["Score"], reverse=True)
    best = rows[0]
    for row, scores, ships in candidates:
        if row == best:
            return rows, scores, ships
    raise RuntimeError("best candidate not found")


def validate_submission(sub, sample, target_ids):
    errors = []
    if len(sub) != len(sample):
        errors.append("row_count mismatch")
    if not sub["filename"].equals(sample["filename"]):
        errors.append("filename order mismatch")
    for i, value in enumerate(sub["top5_ship_ids"].astype(str)):
        parts = value.split(",")
        if len(parts) != 5:
            errors.append(f"row {i} length")
            continue
        ids = [int(x) for x in parts]
        if len(set(ids)) != 5:
            errors.append(f"row {i} duplicate")
        if any(x not in target_ids for x in ids):
            errors.append(f"row {i} target")
    return errors[:20]


def main():
    args = parse_args()
    seed_all(args.seed)
    run_dir = args.out_dir / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    cache_path = args.cache_dir / "logmel.npy"
    train, gallery, val, test, label_map = load_cached_tables(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CachedSpecCNN(len(label_map), args.emb_dim).to(device)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
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
    rows, _, _ = evaluate(val_emb, ref_emb, ref_ship_ids, y)
    pd.DataFrame(rows).to_csv(run_dir / "retrieval_leaderboard.csv", index=False)
    best = rows[0]
    if best["retrieval"] == "hybrid":
        weights = tuple(float(x.strip()) for x in best["weights"].strip("()").split(","))
        test_scores, ships = hybrid(test_emb, ref_emb, ref_ship_ids, weights, int(best["topk"]), float(best["alpha"]))
    elif best["retrieval"] == "topk":
        test_scores, ships = score_matrix(test_emb, ref_emb, ref_ship_ids, "topk", topk=int(best["topk"]), alpha=float(best["alpha"]))
    else:
        test_scores, ships = score_matrix(test_emb, ref_emb, ref_ship_ids, best["retrieval"], alpha=float(best["alpha"]))
    test_pred = top5(test_scores, ships)
    sample = pd.read_csv(args.data_dir / "sample_submission_task2.csv")
    pred_map = dict(zip(test_names, [",".join(map(str, row)) for row in test_pred]))
    sub = sample.copy()
    sub["top5_ship_ids"] = sub["filename"].map(pred_map)
    errors = validate_submission(sub, sample, set(label_map.keys()))
    sub.to_csv(run_dir / "submission_task2.csv", index=False)
    summary = {"run_meta": meta, "best_validation": best, "submission_validation_errors": errors, "artifacts": {"submission": str(run_dir / "submission_task2.csv"), "leaderboard": str(run_dir / "retrieval_leaderboard.csv"), "model": str(run_dir / "model_last.pt")}}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    with Path("analysis/task2_execution_log.md").open("a", encoding="utf-8") as f:
        f.write(f"""
## Cached Run: {args.run_name}

log-mel memmap cache를 사용해 학습/검증/제출 생성을 실행했다.

- cache: `{args.cache_dir}`
- output: `{run_dir}`
- train rows: `{len(train)}`
- reference rows: `{len(gallery)}`
- validation rows: `{len(val)}`
- best retrieval: `{best['retrieval']}`
- alpha: `{best['alpha']}`
- R@1: `{best['R@1']:.6f}`
- R@3: `{best['R@3']:.6f}`
- R@5: `{best['R@5']:.6f}`
- Score: `{best['Score']:.6f}`
- submission validation errors: `{errors}`

해석:

- 이 run은 WAV를 반복해서 읽지 않고 cache에서 spectrogram을 직접 읽는다.
- direct WAV run보다 반복 속도를 높이기 위한 기반 실험이다.

""")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
