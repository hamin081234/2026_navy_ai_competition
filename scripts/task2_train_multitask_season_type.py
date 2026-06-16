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

import task2_train_cached_retrieval as base
from task2_run_robust_local_sweep import score_submission


TYPE_ORDER = ["A_SmallWorking", "B_MotorBoat", "C_Passenger", "D_LargeShip"]
SEASON_ORDER = ["fall", "summer", "winter"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=Path("database"))
    p.add_argument("--cache-dir", type=Path, default=Path("outputs/task2_cache/logmel96_f16000"))
    p.add_argument("--out-dir", type=Path, default=Path("outputs/task2_runs"))
    p.add_argument("--run-name", default="multitask_rows100_seed777_ep8")
    p.add_argument("--report", type=Path, default=Path("analysis/task2_multitask_season_type.md"))
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--max-target-rows", type=int, default=100)
    p.add_argument("--max-train-rows", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--emb-dim", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--type-weight", type=float, default=0.3)
    p.add_argument("--season-weight", type=float, default=0.2)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=777)
    p.add_argument("--no-augment", action="store_true")
    p.add_argument("--no-gallery-train", action="store_true")
    return p.parse_args()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def season_from_month(month):
    if month in (12, 1, 2):
        return "winter"
    if month in (3, 4, 5):
        return "spring"
    if month in (6, 7, 8):
        return "summer"
    return "fall"


def add_timestamp_season(df, data_dir, split):
    out = df.copy()
    if split == "train_target":
        raw = pd.read_csv(data_dir / "train/train.csv")[["filename", "ais_timestamp"]]
    elif split == "gallery":
        raw = pd.read_csv(data_dir / "task2_test/gallery.csv")[["filename", "ais_timestamp"]]
    else:
        return out
    out = out.merge(raw, on="filename", how="left", validate="many_to_one")
    ts = pd.to_datetime(out["ais_timestamp"], utc=True, format="mixed")
    out["season"] = ts.dt.month.map(season_from_month)
    return out


def load_tables(args):
    train, gallery, val, test, ship_label_map = base.load_cached_tables(args)
    train = add_timestamp_season(train, args.data_dir, "train_target")
    gallery = add_timestamp_season(gallery, args.data_dir, "gallery")

    # If gallery has already been appended to train by base.load_cached_tables, it needs timestamp too.
    if "season" not in train.columns or train["season"].isna().any():
        train_raw = pd.read_csv(args.data_dir / "train/train.csv")[["filename", "ais_timestamp"]]
        gallery_raw = pd.read_csv(args.data_dir / "task2_test/gallery.csv")[["filename", "ais_timestamp"]]
        ts_raw = pd.concat([train_raw, gallery_raw], ignore_index=True)
        train = train.drop(columns=[c for c in ["ais_timestamp", "season"] if c in train.columns], errors="ignore")
        train = train.merge(ts_raw, on="filename", how="left", validate="many_to_one")
        ts = pd.to_datetime(train["ais_timestamp"], utc=True, format="mixed")
        train["season"] = ts.dt.month.map(season_from_month)

    type_map = {name: i for i, name in enumerate(TYPE_ORDER)}
    season_values = [s for s in SEASON_ORDER if s in set(train["season"].dropna().astype(str)) | set(gallery["season"].dropna().astype(str))]
    season_map = {name: i for i, name in enumerate(season_values)}

    for df in [train, gallery, val]:
        df["type_label"] = df["ship_type"].astype(str).map(type_map)
    for df in [train, gallery]:
        df["season_label"] = df["season"].astype(str).map(season_map)

    return train, gallery, val, test, ship_label_map, type_map, season_map


class MultiTaskDataset(Dataset):
    def __init__(self, df, cache_path, with_label, augment):
        self.df = df.reset_index(drop=True)
        self.cache = np.load(cache_path, mmap_mode="r")
        self.indices = self.df["cache_index"].astype(int).to_numpy()
        self.with_label = with_label
        self.augment = augment
        self.filenames = self.df["filename"].astype(str).tolist()
        if with_label:
            self.ship = self.df["label"].astype(int).to_numpy()
            self.typ = self.df["type_label"].astype(int).to_numpy()
            self.season = self.df["season_label"].astype(int).to_numpy()

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        x = torch.from_numpy(np.asarray(self.cache[self.indices[idx]], dtype=np.float32))
        if self.augment:
            x = base.augment_spec(x)
        if self.with_label:
            return x, int(self.ship[idx]), int(self.typ[idx]), int(self.season[idx])
        return x, self.filenames[idx]


class MultiTaskCNN(nn.Module):
    def __init__(self, n_ship, n_type, n_season, emb_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            base.ConvBlock(1, 32),
            base.ConvBlock(32, 64),
            base.ConvBlock(64, 128),
            base.ConvBlock(128, 192),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.embedding = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(192, emb_dim, bias=False),
            nn.BatchNorm1d(emb_dim),
        )
        self.ship_head = nn.Linear(emb_dim, n_ship)
        self.type_head = nn.Linear(emb_dim, n_type)
        self.season_head = nn.Linear(emb_dim, n_season)

    def features(self, spec):
        emb = self.embedding(self.encoder(spec.unsqueeze(1)))
        return F.normalize(emb, dim=1)

    def forward(self, spec):
        emb = self.features(spec)
        return self.ship_head(emb), self.type_head(emb), self.season_head(emb), emb


def make_sampler(df):
    counts = df["label"].astype(int).value_counts().to_dict()
    weights = df["label"].astype(int).map(lambda x: 1.0 / counts[int(x)]).to_numpy(dtype=np.float64)
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


@torch.no_grad()
def infer_df(model, df, cache_path, args, device):
    ds = MultiTaskDataset(df, cache_path, with_label=False, augment=False)
    dl = DataLoader(ds, batch_size=args.batch_size * 2, shuffle=False, num_workers=args.num_workers, pin_memory=torch.cuda.is_available())
    model.eval()
    embs, ship_logits, type_logits, season_logits, names = [], [], [], [], []
    for spec, filename in dl:
        spec = spec.to(device, non_blocking=True)
        s, t, z, emb = model(spec)
        embs.append(emb.detach().cpu().numpy().astype(np.float32))
        ship_logits.append(s.detach().cpu().numpy().astype(np.float32))
        type_logits.append(t.detach().cpu().numpy().astype(np.float32))
        season_logits.append(z.detach().cpu().numpy().astype(np.float32))
        names.extend(filename)
    return {
        "emb": np.vstack(embs),
        "ship_logits": np.vstack(ship_logits),
        "type_logits": np.vstack(type_logits),
        "season_logits": np.vstack(season_logits),
        "names": names,
    }


def l2(x):
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-8, None)


def metric(y, pred):
    r1 = float(np.mean([yy in row[:1] for yy, row in zip(y, pred)]))
    r3 = float(np.mean([yy in row[:3] for yy, row in zip(y, pred)]))
    r5 = float(np.mean([yy in row[:5] for yy, row in zip(y, pred)]))
    return {"R@1": r1, "R@3": r3, "R@5": r5, "Score": 0.5 * r1 + 0.3 * r3 + 0.2 * r5}


def top5(scores, ships):
    return ships[np.argsort(-scores, axis=1)[:, :5]]


def season_filtered_ship_scores(query_emb, ref_emb, ref_ship_ids, ref_season_labels, season_prob, keep, ship_topk):
    query_emb, ref_emb = l2(query_emb), l2(ref_emb)
    ships = np.array(sorted(set(ref_ship_ids.tolist())), dtype=int)
    sim = query_emb @ ref_emb.T
    out = np.zeros((len(query_emb), len(ships)), dtype=np.float32)
    season_rank = np.argsort(-season_prob, axis=1)[:, :keep]
    for i in range(len(query_emb)):
        allowed = set(season_rank[i].tolist())
        season_mask = np.array([int(s) in allowed for s in ref_season_labels], dtype=bool)
        for j, sid in enumerate(ships):
            cols = np.where((ref_ship_ids == sid) & season_mask)[0]
            if len(cols) == 0:
                cols = np.where(ref_ship_ids == sid)[0]
            vals = sim[i, cols]
            k = min(ship_topk, len(vals))
            out[i, j] = np.sort(vals)[-k:].mean()
    return out, ships


def type_gate(scores, ships, ship_to_type_label, type_prob, keep):
    out = scores.copy()
    type_rank = np.argsort(-type_prob, axis=1)[:, :keep]
    for i in range(out.shape[0]):
        allowed = set(type_rank[i].tolist())
        for j, sid in enumerate(ships):
            if int(ship_to_type_label[int(sid)]) not in allowed:
                out[i, j] = -1e9
    return out


def make_submission(path, sample, test_names, pred):
    sub = sample.copy()
    pred_map = dict(zip(test_names, [",".join(map(str, row)) for row in pred]))
    sub["top5_ship_ids"] = sub["filename"].map(pred_map)
    sub.to_csv(path, index=False)
    return path


def eval_epoch(model, tables, maps, args, run_dir, epoch, device):
    train, gallery, val, test = tables
    ship_label_map, type_map, season_map = maps
    cache_path = args.cache_dir / "logmel.npy"
    ref = infer_df(model, gallery, cache_path, args, device)
    val_out = infer_df(model, val, cache_path, args, device)
    test_out = infer_df(model, test, cache_path, args, device)

    ref_ship_ids = gallery["ship_id"].astype(int).to_numpy()
    ref_season = gallery["season_label"].astype(int).to_numpy()
    y = val["ship_id"].astype(int).to_numpy()
    y_type = val["type_label"].astype(int).to_numpy()
    ship_to_type_label = gallery.groupby("ship_id")["type_label"].first().astype(int).to_dict()
    sample = pd.read_csv(args.data_dir / "sample_submission_task2.csv")

    type_prob_val = torch.softmax(torch.from_numpy(val_out["type_logits"]), dim=1).numpy()
    type_prob_test = torch.softmax(torch.from_numpy(test_out["type_logits"]), dim=1).numpy()
    season_prob_val = torch.softmax(torch.from_numpy(val_out["season_logits"]), dim=1).numpy()
    season_prob_test = torch.softmax(torch.from_numpy(test_out["season_logits"]), dim=1).numpy()
    type_top1_acc = float(np.mean(type_prob_val.argmax(axis=1) == y_type))
    type_top2_acc = float(np.mean([yt in row for yt, row in zip(y_type, np.argsort(-type_prob_val, axis=1)[:, :2])]))

    rows = []
    for ship_topk in [15, 20]:
        for season_keep in [1, 2, len(season_map)]:
            val_scores, ships = season_filtered_ship_scores(val_out["emb"], ref["emb"], ref_ship_ids, ref_season, season_prob_val, season_keep, ship_topk)
            test_scores, _ = season_filtered_ship_scores(test_out["emb"], ref["emb"], ref_ship_ids, ref_season, season_prob_test, season_keep, ship_topk)
            for type_keep in [1, 2, len(type_map)]:
                name = f"ep{epoch}_shiptopk{ship_topk}_season{season_keep}_type{type_keep}"
                gated_val = type_gate(val_scores, ships, ship_to_type_label, type_prob_val, type_keep)
                gated_test = type_gate(test_scores, ships, ship_to_type_label, type_prob_test, type_keep)
                pred_val = top5(gated_val, ships)
                pred_test = top5(gated_test, ships)
                sub_path = run_dir / f"submission_{name}.csv"
                make_submission(sub_path, sample, test_out["names"], pred_test)
                diag = score_submission(sub_path, args.data_dir)
                row = {
                    "epoch": epoch,
                    "ship_topk": ship_topk,
                    "season_keep": season_keep,
                    "type_keep": type_keep,
                    "type_top1_acc": type_top1_acc,
                    "type_top2_acc": type_top2_acc,
                    **metric(y, pred_val),
                    "diagnostic_Score": diag["test_Score"],
                    "submission": str(sub_path),
                }
                rows.append(row)
    df = pd.DataFrame(rows).sort_values(["Score", "R@1", "diagnostic_Score"], ascending=False)
    df.to_csv(run_dir / f"retrieval_epoch{epoch}.csv", index=False)
    return df


def md_table(df, cols):
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, r in df.iterrows():
        vals = []
        for c in cols:
            v = r[c]
            if c in {"R@1", "R@3", "R@5", "Score", "diagnostic_Score", "type_top1_acc", "type_top2_acc"}:
                vals.append(f"{float(v):.6f}")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def main():
    args = parse_args()
    seed_all(args.seed)
    run_dir = args.out_dir / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    cache_path = args.cache_dir / "logmel.npy"
    train, gallery, val, test, ship_label_map, type_map, season_map = load_tables(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    model = MultiTaskCNN(len(ship_label_map), len(type_map), len(season_map), args.emb_dim).to(device)

    meta = {
        "run_name": args.run_name,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "rows": {"train": len(train), "gallery": len(gallery), "val": len(val), "test": len(test)},
        "type_map": type_map,
        "season_map": season_map,
    }
    (run_dir / "run_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"run_start": meta}, ensure_ascii=False), flush=True)

    ds = MultiTaskDataset(train, cache_path, with_label=True, augment=not args.no_augment)
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        sampler=make_sampler(train),
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = max(1, args.epochs * math.ceil(len(train) / args.batch_size))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps)

    train_rows, all_eval = [], []
    start = time.time()
    steps = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses, ship_correct, type_correct, season_correct, total = [], 0, 0, 0, 0
        for spec, y_ship, y_type, y_season in dl:
            spec = spec.to(device, non_blocking=True)
            y_ship = y_ship.long().to(device, non_blocking=True)
            y_type = y_type.long().to(device, non_blocking=True)
            y_season = y_season.long().to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            ship_logits, type_logits, season_logits, _ = model(spec)
            loss_ship = F.cross_entropy(ship_logits, y_ship, label_smoothing=0.05)
            loss_type = F.cross_entropy(type_logits, y_type, label_smoothing=0.02)
            loss_season = F.cross_entropy(season_logits, y_season, label_smoothing=0.02)
            loss = loss_ship + args.type_weight * loss_type + args.season_weight * loss_season
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            steps += 1
            losses.append(float(loss.detach().cpu()))
            ship_correct += int((ship_logits.argmax(1) == y_ship).sum().detach().cpu())
            type_correct += int((type_logits.argmax(1) == y_type).sum().detach().cpu())
            season_correct += int((season_logits.argmax(1) == y_season).sum().detach().cpu())
            total += int(y_ship.numel())

        train_row = {
            "epoch": epoch,
            "steps": steps,
            "train_loss": float(np.mean(losses)) if losses else None,
            "train_ship_acc": float(ship_correct / total) if total else None,
            "train_type_acc": float(type_correct / total) if total else None,
            "train_season_acc": float(season_correct / total) if total else None,
            "elapsed_min": float((time.time() - start) / 60),
        }
        train_rows.append(train_row)
        print(json.dumps({"train": train_row}, ensure_ascii=False), flush=True)

        torch.save(
            {
                "model_state": model.state_dict(),
                "ship_label_map": ship_label_map,
                "type_map": type_map,
                "season_map": season_map,
                "args": meta["args"],
                "epoch": epoch,
            },
            run_dir / f"model_epoch{epoch}.pt",
        )
        eval_df = eval_epoch(model, (train, gallery, val, test), (ship_label_map, type_map, season_map), args, run_dir, epoch, device)
        all_eval.append(eval_df)
        print(json.dumps({"epoch_eval_best": eval_df.iloc[0].to_dict()}, ensure_ascii=False), flush=True)

    pd.DataFrame(train_rows).to_csv(run_dir / "train_history.csv", index=False)
    leaderboard = pd.concat(all_eval, ignore_index=True).sort_values(["Score", "R@1", "diagnostic_Score"], ascending=False)
    leaderboard_path = run_dir / "multitask_leaderboard.csv"
    leaderboard.to_csv(leaderboard_path, index=False)
    best = leaderboard.iloc[0]
    best_alias = run_dir / "submission_task2_best_multitask.csv"
    best_alias.write_bytes(Path(best.submission).read_bytes())

    lines = ["# Task 2 Multi-Task Season + Ship Type Training", ""]
    lines.append("ship_id, ship_type, season을 함께 학습하고, 추론 시 season/type head로 후보를 제한한 실험이다.")
    lines.append("")
    lines.append("## Best By Validation")
    lines.append("")
    for k in ["epoch", "ship_topk", "season_keep", "type_keep", "type_top1_acc", "type_top2_acc", "R@1", "R@3", "R@5", "Score", "diagnostic_Score", "submission"]:
        v = best[k]
        if isinstance(v, float):
            lines.append(f"- {k}: `{v:.6f}`")
        else:
            lines.append(f"- {k}: `{v}`")
    lines.append(f"- alias: `{best_alias}`")
    lines.append("")
    lines.append("## Top Results")
    lines.append("")
    cols = ["epoch", "ship_topk", "season_keep", "type_keep", "type_top1_acc", "type_top2_acc", "R@1", "R@3", "R@5", "Score", "diagnostic_Score"]
    lines.append(md_table(leaderboard.head(25), cols))
    lines.append("")
    lines.append("## 해석")
    lines.append("")
    lines.append("- 이 실험은 후처리 prototype이 아니라 모델 head가 직접 season/type을 예측하게 만든다.")
    lines.append("- validation 기준으로 기존 season+type 후처리 best를 넘는지 확인한다.")
    lines.append("- 넘지 못하면 season/type은 head 학습보다 후처리 gate로 쓰는 편이 낫다.")
    args.report.write_text("\n".join(lines), encoding="utf-8")

    with Path("analysis/task2_execution_log.md").open("a", encoding="utf-8") as f:
        f.write(f"""
## Multi-Task Season + Ship Type Training 완료

ship_id, ship_type, season head를 함께 학습하고, season/type head 예측으로 ship_id 후보를 제한했다.

- report: `{args.report}`
- leaderboard: `{leaderboard_path}`
- best alias: `{best_alias}`

Best by validation:

- epoch: `{int(best.epoch)}`
- ship_topk: `{int(best.ship_topk)}`
- season_keep: `{int(best.season_keep)}`
- type_keep: `{int(best.type_keep)}`
- validation Score: `{best.Score:.6f}`
- diagnostic label Score: `{best.diagnostic_Score:.6f}`
- submission: `{best.submission}`

""")
    print(leaderboard.head(25).to_string(index=False))
    print("\nBEST", int(best.epoch), best.Score, best.submission)


if __name__ == "__main__":
    main()
