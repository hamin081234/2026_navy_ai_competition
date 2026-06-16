#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


@dataclass(frozen=True)
class RunConfig:
    run_name: str
    n_mels: int = 128
    f_max: int = 8000
    hop_length: int = 320
    batch_size: int = 48
    emb_dim: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-4
    label_smoothing: float = 0.05
    use_gallery_in_train: bool = True
    reference_mode: str = "gallery"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("database"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/task2_runs"))
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-train-minutes", type=float, default=None)
    parser.add_argument("--max-train-rows", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--n-mels", type=int, default=128)
    parser.add_argument("--f-max", type=int, default=8000)
    parser.add_argument("--hop-length", type=int, default=320)
    parser.add_argument("--emb-dim", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-gallery-train", action="store_true")
    parser.add_argument("--reference-mode", choices=["gallery", "gallery_train"], default="gallery")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_config(args: argparse.Namespace) -> RunConfig:
    if args.run_name:
        run_name = args.run_name
    else:
        run_name = f"scratch_cnn_mel{args.n_mels}_f{args.f_max}_seed{args.seed}"
    return RunConfig(
        run_name=run_name,
        n_mels=args.n_mels,
        f_max=args.f_max,
        hop_length=args.hop_length,
        batch_size=args.batch_size,
        emb_dim=args.emb_dim,
        lr=args.lr,
        weight_decay=args.weight_decay,
        use_gallery_in_train=not args.no_gallery_train,
        reference_mode=args.reference_mode,
    )


def load_tables(data_dir: Path, cfg: RunConfig, max_train_rows: int | None):
    target_ids = pd.read_csv(data_dir / "task2_target_ships.csv")["ship_id"].astype(int).tolist()
    target_set = set(target_ids)
    sorted_targets = sorted(target_ids)
    label_map = {ship_id: idx for idx, ship_id in enumerate(sorted_targets)}

    train = pd.read_csv(data_dir / "train/train.csv")
    train = train[train["ship_id"].isin(target_set)].copy()
    train["audio_path"] = train["filename"].map(lambda x: data_dir / "train/audio" / x)
    train["label"] = train["ship_id"].map(label_map)
    train["source"] = "train"

    gallery = pd.read_csv(data_dir / "task2_test/gallery.csv")
    gallery["audio_path"] = gallery["filename"].map(lambda x: data_dir / "task2_test/audio" / x)
    gallery["label"] = gallery["ship_id"].map(label_map)
    gallery["source"] = "gallery"

    val = pd.read_csv(data_dir / "task2_test/val.csv")
    val["audio_path"] = val["filename"].map(lambda x: data_dir / "task2_test/audio" / x)
    val["label"] = val["ship_id"].map(label_map)
    val["source"] = "val"

    test = pd.read_csv(data_dir / "task2_test/test.csv")
    test["audio_path"] = test["filename"].map(lambda x: data_dir / "task2_test/audio" / x)
    test["source"] = "test"

    train_parts = [train[["filename", "ship_id", "ship_type", "audio_path", "label", "source"]]]
    if cfg.use_gallery_in_train:
        train_parts.append(gallery[["filename", "ship_id", "ship_type", "audio_path", "label", "source"]])
    train_df = pd.concat(train_parts, ignore_index=True, sort=False)
    if max_train_rows is not None:
        train_df = balanced_sample(train_df, max_train_rows, seed=42)

    if cfg.reference_mode == "gallery_train":
        ref_df = pd.concat(
            [
                gallery[["filename", "ship_id", "ship_type", "audio_path", "label", "source"]],
                train[["filename", "ship_id", "ship_type", "audio_path", "label", "source"]],
            ],
            ignore_index=True,
            sort=False,
        )
    else:
        ref_df = gallery[["filename", "ship_id", "ship_type", "audio_path", "label", "source"]].copy()

    return {
        "train": train_df.reset_index(drop=True),
        "gallery": gallery.reset_index(drop=True),
        "reference": ref_df.reset_index(drop=True),
        "val": val.reset_index(drop=True),
        "test": test.reset_index(drop=True),
        "label_map": label_map,
        "target_ids": sorted_targets,
    }


def balanced_sample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if n >= len(df):
        return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    per_ship = max(1, n // df["ship_id"].nunique())
    parts = []
    for _, group in df.groupby("ship_id"):
        parts.append(group.sample(n=min(per_ship, len(group)), random_state=seed))
    out = pd.concat(parts, ignore_index=True)
    if len(out) < n:
        remain = df.drop(index=out.index, errors="ignore")
        extra = df.sample(n=n - len(out), random_state=seed)
        out = pd.concat([out, extra], ignore_index=True)
    return out.sample(n=min(n, len(out)), random_state=seed).reset_index(drop=True)


class ShipWaveDataset(Dataset):
    def __init__(self, df: pd.DataFrame, with_label: bool, augment: bool = False):
        self.paths = df["audio_path"].astype(str).tolist()
        self.filenames = df["filename"].astype(str).tolist()
        self.with_label = with_label
        self.augment = augment
        self.labels = df["label"].astype(int).tolist() if with_label and "label" in df.columns else None

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        wav, sr = sf.read(self.paths[idx], dtype="float32", always_2d=False)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        wav = np.asarray(wav, dtype=np.float32)
        if len(wav) < 160000:
            wav = np.pad(wav, (0, 160000 - len(wav)))
        elif len(wav) > 160000:
            wav = wav[:160000]
        wav = wav - np.mean(wav, dtype=np.float32)
        peak = np.max(np.abs(wav)) + 1e-6
        wav = wav / peak
        x = torch.from_numpy(wav.astype(np.float32))
        if self.augment:
            x = augment_wave(x)
        if self.with_label:
            return x, int(self.labels[idx])
        return x, self.filenames[idx]


def augment_wave(x: torch.Tensor) -> torch.Tensor:
    gain = 10 ** (float(torch.empty(1).uniform_(-4.0, 4.0)) / 20.0)
    x = x * gain
    shift = int(torch.randint(-8000, 8001, (1,)).item())
    if shift != 0:
        x = torch.roll(x, shifts=shift)
    if torch.rand(()) < 0.25:
        noise = torch.randn_like(x) * float(torch.empty(1).uniform_(0.001, 0.006))
        x = x + noise
    return x.clamp(-1.0, 1.0)


def make_sampler(df: pd.DataFrame) -> WeightedRandomSampler:
    counts = df["label"].value_counts().to_dict()
    weights = df["label"].map(lambda x: 1.0 / counts[int(x)]).to_numpy(dtype=np.float64)
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LogMelShipCNN(nn.Module):
    def __init__(self, n_classes: int, cfg: RunConfig):
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=32000,
            n_fft=1024,
            win_length=1024,
            hop_length=cfg.hop_length,
            n_mels=cfg.n_mels,
            f_min=20,
            f_max=cfg.f_max,
            power=2.0,
        )
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
            nn.Linear(192, cfg.emb_dim, bias=False),
            nn.BatchNorm1d(cfg.emb_dim),
        )
        self.classifier = nn.Linear(cfg.emb_dim, n_classes)

    def features(self, wav: torch.Tensor) -> torch.Tensor:
        spec = torch.log(self.mel(wav) + 1e-6)
        mean = spec.mean(dim=(1, 2), keepdim=True)
        std = spec.std(dim=(1, 2), keepdim=True).clamp_min(1e-5)
        spec = (spec - mean) / std
        x = spec.unsqueeze(1)
        emb = self.embedding(self.encoder(x))
        return F.normalize(emb, dim=1)

    def forward(self, wav: torch.Tensor):
        emb = self.features(wav)
        return self.classifier(emb), emb


def train_one(model, train_df, cfg, args, device, run_dir):
    ds = ShipWaveDataset(train_df, with_label=True, augment=True)
    dl = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        sampler=make_sampler(train_df),
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    total_steps = max(1, args.epochs * math.ceil(len(train_df) / cfg.batch_size))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps)
    start = time.time()
    history = []
    global_step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        correct = 0
        total = 0
        for wav, y in dl:
            if args.max_train_minutes and (time.time() - start) / 60.0 >= args.max_train_minutes:
                break
            wav = wav.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            logits, _ = model(wav)
            loss = F.cross_entropy(logits, y, label_smoothing=cfg.label_smoothing)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            global_step += 1
            losses.append(float(loss.detach().cpu()))
            correct += int((logits.argmax(dim=1) == y).sum().detach().cpu())
            total += int(y.numel())
        row = {
            "epoch": epoch,
            "steps": global_step,
            "train_loss": float(np.mean(losses)) if losses else None,
            "train_accuracy": float(correct / total) if total else None,
            "elapsed_min": float((time.time() - start) / 60.0),
        }
        history.append(row)
        print(json.dumps({"train": row}, ensure_ascii=False), flush=True)
        if args.max_train_minutes and (time.time() - start) / 60.0 >= args.max_train_minutes:
            break
    pd.DataFrame(history).to_csv(run_dir / "train_history.csv", index=False)
    return history


@torch.no_grad()
def embed_df(model, df: pd.DataFrame, cfg: RunConfig, args, device):
    ds = ShipWaveDataset(df, with_label=False, augment=False)
    dl = DataLoader(
        ds,
        batch_size=cfg.batch_size * 2,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    model.eval()
    embs = []
    names = []
    for wav, filename in dl:
        wav = wav.to(device, non_blocking=True)
        emb = model.features(wav).detach().cpu().numpy().astype(np.float32)
        embs.append(emb)
        names.extend(filename)
    return np.vstack(embs), names


def l2_normalize(x: np.ndarray) -> np.ndarray:
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-8, None)


def score_matrix(query_emb: np.ndarray, ref_emb: np.ndarray, ref_ship_ids: np.ndarray, method: str, topk: int = 5, alpha: float = 0.0):
    query_emb = l2_normalize(query_emb)
    ref_emb = l2_normalize(ref_emb)
    ships = np.array(sorted(set(ref_ship_ids.tolist())), dtype=int)
    sim = query_emb @ ref_emb.T
    out = np.zeros((len(query_emb), len(ships)), dtype=np.float32)
    for j, sid in enumerate(ships):
        cols = np.where(ref_ship_ids == sid)[0]
        vals = sim[:, cols]
        if method == "prototype":
            proto = l2_normalize(ref_emb[cols].mean(axis=0, keepdims=True))
            out[:, j] = (query_emb @ proto.T).reshape(-1)
        elif method == "max":
            out[:, j] = vals.max(axis=1)
        elif method == "topk":
            k = min(topk, vals.shape[1])
            part = np.partition(vals, kth=vals.shape[1] - k, axis=1)[:, -k:]
            out[:, j] = part.mean(axis=1)
        else:
            raise ValueError(method)
        if alpha:
            out[:, j] -= alpha * math.log1p(len(cols))
    return out, ships


def hybrid_score_matrix(query_emb, ref_emb, ref_ship_ids, weights, topk, alpha):
    proto, ships = score_matrix(query_emb, ref_emb, ref_ship_ids, "prototype", topk=topk, alpha=0.0)
    top, _ = score_matrix(query_emb, ref_emb, ref_ship_ids, "topk", topk=topk, alpha=0.0)
    mx, _ = score_matrix(query_emb, ref_emb, ref_ship_ids, "max", topk=topk, alpha=0.0)
    score = weights[0] * proto + weights[1] * top + weights[2] * mx
    if alpha:
        for j, sid in enumerate(ships):
            n = int(np.sum(ref_ship_ids == sid))
            score[:, j] -= alpha * math.log1p(n)
    return score, ships


def top5_from_scores(scores: np.ndarray, ships: np.ndarray) -> np.ndarray:
    order = np.argsort(-scores, axis=1)[:, :5]
    return ships[order]


def metrics(y_true: np.ndarray, top5: np.ndarray) -> dict[str, float]:
    r1 = float(np.mean([truth in row[:1] for truth, row in zip(y_true, top5)]))
    r3 = float(np.mean([truth in row[:3] for truth, row in zip(y_true, top5)]))
    r5 = float(np.mean([truth in row[:5] for truth, row in zip(y_true, top5)]))
    return {"R@1": r1, "R@3": r3, "R@5": r5, "Score": 0.5 * r1 + 0.3 * r3 + 0.2 * r5}


def evaluate_all(query_emb, ref_emb, ref_ship_ids, y_true):
    rows = []
    candidates = []
    for method in ["prototype", "max"]:
        for alpha in [0.0, 0.005, 0.01, 0.02]:
            scores, ships = score_matrix(query_emb, ref_emb, ref_ship_ids, method, alpha=alpha)
            top5 = top5_from_scores(scores, ships)
            rows.append({"retrieval": method, "topk": None, "weights": None, "alpha": alpha, **metrics(y_true, top5)})
            candidates.append((rows[-1], scores, ships))
    for topk in [3, 5, 10]:
        for alpha in [0.0, 0.005, 0.01, 0.02]:
            scores, ships = score_matrix(query_emb, ref_emb, ref_ship_ids, "topk", topk=topk, alpha=alpha)
            top5 = top5_from_scores(scores, ships)
            rows.append({"retrieval": "topk", "topk": topk, "weights": None, "alpha": alpha, **metrics(y_true, top5)})
            candidates.append((rows[-1], scores, ships))
    for topk in [3, 5, 10]:
        for weights in [(0.5, 0.3, 0.2), (0.4, 0.4, 0.2), (0.3, 0.5, 0.2), (0.6, 0.3, 0.1)]:
            for alpha in [0.0, 0.005, 0.01, 0.02]:
                scores, ships = hybrid_score_matrix(query_emb, ref_emb, ref_ship_ids, weights, topk, alpha)
                top5 = top5_from_scores(scores, ships)
                rows.append({"retrieval": "hybrid", "topk": topk, "weights": str(weights), "alpha": alpha, **metrics(y_true, top5)})
                candidates.append((rows[-1], scores, ships))
    rows = sorted(rows, key=lambda r: r["Score"], reverse=True)
    best_row = rows[0]
    for row, scores, ships in candidates:
        if row == best_row:
            return rows, scores, ships
    raise RuntimeError("Best candidate not found")


def validate_submission(sub: pd.DataFrame, sample: pd.DataFrame, target_ids: set[int]) -> list[str]:
    errors = []
    if len(sub) != len(sample):
        errors.append(f"row_count mismatch: {len(sub)} != {len(sample)}")
    if not sub["filename"].equals(sample["filename"]):
        errors.append("filename order mismatch")
    for i, value in enumerate(sub["top5_ship_ids"].astype(str)):
        parts = value.split(",")
        if len(parts) != 5:
            errors.append(f"row {i} top5 length {len(parts)}")
            continue
        try:
            ids = [int(x) for x in parts]
        except ValueError:
            errors.append(f"row {i} non-integer ids: {value}")
            continue
        if len(set(ids)) != 5:
            errors.append(f"row {i} duplicate ids: {value}")
        bad = [x for x in ids if x not in target_ids]
        if bad:
            errors.append(f"row {i} ids outside target set: {bad}")
    return errors[:20]


def jsonable_args(args: argparse.Namespace) -> dict:
    out = {}
    for key, value in vars(args).items():
        out[key] = str(value) if isinstance(value, Path) else value
    return out


def append_execution_log(path: Path, run_dir: Path, run_meta: dict, best: dict):
    lines = [
        "",
        f"## Run: {run_meta['run_name']}",
        "",
        "새 scratch 파이프라인으로 학습, 검증, 제출 생성을 실행했다.",
        "",
        "설정:",
        "",
        f"- output: `{run_dir}`",
        f"- device: `{run_meta['device']}`",
        f"- train rows: `{run_meta['rows']['train']}`",
        f"- reference rows: `{run_meta['rows']['reference']}`",
        f"- validation rows: `{run_meta['rows']['val']}`",
        f"- feature: log-mel `{run_meta['config']['n_mels']}` mels, fmax `{run_meta['config']['f_max']}`",
        f"- reference mode: `{run_meta['config']['reference_mode']}`",
        "",
        "Best validation retrieval:",
        "",
        f"- method: `{best['retrieval']}`",
        f"- topk: `{best.get('topk')}`",
        f"- weights: `{best.get('weights')}`",
        f"- alpha: `{best['alpha']}`",
        f"- R@1: `{best['R@1']:.6f}`",
        f"- R@3: `{best['R@3']:.6f}`",
        f"- R@5: `{best['R@5']:.6f}`",
        f"- Score: `{best['Score']:.6f}`",
        "",
        "해석:",
        "",
        "- 이 점수는 `task2_gallery`를 reference, `task2_val`을 query로 둔 official validation 점수다.",
        "- 현재 run은 새 파이프라인 검증과 후보 모델 선별을 위한 실행이다.",
        "- 다음 run은 이 결과보다 높은 Score 또는 다른 오답 패턴을 만드는지 기준으로 판단한다.",
        "",
    ]
    with path.open("a", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.max_train_rows = args.max_train_rows or 400
        args.epochs = min(args.epochs, 1)
        args.max_train_minutes = args.max_train_minutes or 3
        args.run_name = args.run_name or "smoke_scratch_cnn"
    cfg = build_config(args)
    seed_all(args.seed)
    run_dir = args.out_dir / cfg.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    tables = load_tables(args.data_dir, cfg, args.max_train_rows)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    model = LogMelShipCNN(n_classes=len(tables["label_map"]), cfg=cfg).to(device)

    run_meta = {
        "run_name": cfg.run_name,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "config": asdict(cfg),
        "args": jsonable_args(args),
        "rows": {k: int(len(v)) for k, v in tables.items() if isinstance(v, pd.DataFrame)},
    }
    (run_dir / "run_meta.json").write_text(json.dumps(run_meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(run_meta, ensure_ascii=False), flush=True)

    train_one(model, tables["train"], cfg, args, device, run_dir)
    torch.save({"model_state": model.state_dict(), "config": asdict(cfg), "label_map": tables["label_map"]}, run_dir / "model_last.pt")

    ref_emb, ref_names = embed_df(model, tables["reference"], cfg, args, device)
    val_emb, val_names = embed_df(model, tables["val"], cfg, args, device)
    test_emb, test_names = embed_df(model, tables["test"], cfg, args, device)
    np.save(run_dir / "reference_embeddings.npy", ref_emb)
    np.save(run_dir / "val_embeddings.npy", val_emb)
    np.save(run_dir / "test_embeddings.npy", test_emb)

    ref_ship_ids = tables["reference"]["ship_id"].astype(int).to_numpy()
    y_true = tables["val"]["ship_id"].astype(int).to_numpy()
    eval_rows, best_val_scores, best_ships = evaluate_all(val_emb, ref_emb, ref_ship_ids, y_true)
    pd.DataFrame(eval_rows).to_csv(run_dir / "retrieval_leaderboard.csv", index=False)
    best = eval_rows[0]

    test_scores, test_ships = None, None
    if best["retrieval"] == "hybrid":
        weights = tuple(float(x.strip()) for x in best["weights"].strip("()").split(","))
        test_scores, test_ships = hybrid_score_matrix(test_emb, ref_emb, ref_ship_ids, weights, int(best["topk"]), float(best["alpha"]))
    elif best["retrieval"] == "topk":
        test_scores, test_ships = score_matrix(test_emb, ref_emb, ref_ship_ids, "topk", topk=int(best["topk"]), alpha=float(best["alpha"]))
    else:
        test_scores, test_ships = score_matrix(test_emb, ref_emb, ref_ship_ids, best["retrieval"], alpha=float(best["alpha"]))
    test_top5 = top5_from_scores(test_scores, test_ships)

    sample = pd.read_csv(args.data_dir / "sample_submission_task2.csv")
    pred_map = dict(zip(test_names, [",".join(map(str, row)) for row in test_top5]))
    sub = sample.copy()
    sub["top5_ship_ids"] = sub["filename"].map(pred_map)
    errors = validate_submission(sub, sample, set(tables["target_ids"]))
    sub.to_csv(run_dir / "submission_task2.csv", index=False)

    summary = {
        "run_meta": run_meta,
        "best_validation": best,
        "submission_validation_errors": errors,
        "artifacts": {
            "submission": str(run_dir / "submission_task2.csv"),
            "leaderboard": str(run_dir / "retrieval_leaderboard.csv"),
            "model": str(run_dir / "model_last.pt"),
        },
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    append_execution_log(Path("analysis/task2_execution_log.md"), run_dir, run_meta, best)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
