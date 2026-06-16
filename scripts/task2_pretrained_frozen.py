#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=Path("database"))
    p.add_argument("--cache-dir", type=Path, default=Path("outputs/task2_cache/logmel96_f16000"))
    p.add_argument("--out-dir", type=Path, default=Path("outputs/task2_runs"))
    p.add_argument("--model", choices=["resnet18", "efficientnet_b0", "clip_vit_b32"], default="resnet18")
    p.add_argument("--run-name", default=None)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=0)
    return p.parse_args()


class CachedSplitDataset(Dataset):
    def __init__(self, meta: pd.DataFrame, cache_path: Path):
        self.meta = meta.reset_index(drop=True)
        self.cache = np.load(cache_path, mmap_mode="r")
        self.indices = self.meta["cache_index"].astype(int).to_numpy()
        self.filenames = self.meta["filename"].astype(str).tolist()

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        spec = torch.from_numpy(np.asarray(self.cache[self.indices[idx]], dtype=np.float32))
        return spec, self.filenames[idx]


def load_splits(cache_dir: Path):
    meta = pd.read_csv(cache_dir / "metadata.csv")
    return (
        meta[meta["split"] == "gallery"].copy().reset_index(drop=True),
        meta[meta["split"] == "val"].copy().reset_index(drop=True),
        meta[meta["split"] == "test"].copy().reset_index(drop=True),
    )


class VisionExtractor:
    def __init__(self, name: str, device: torch.device):
        self.name = name
        self.device = device
        if name == "resnet18":
            from torchvision.models import ResNet18_Weights, resnet18
            weights = ResNet18_Weights.DEFAULT
            model = resnet18(weights=weights)
            model.fc = torch.nn.Identity()
            self.model = model.eval().to(device)
            self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
            self.std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        elif name == "efficientnet_b0":
            from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0
            weights = EfficientNet_B0_Weights.DEFAULT
            model = efficientnet_b0(weights=weights)
            model.classifier = torch.nn.Identity()
            self.model = model.eval().to(device)
            self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
            self.std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        elif name == "clip_vit_b32":
            from transformers import CLIPVisionModel
            self.model = CLIPVisionModel.from_pretrained(
                "openai/clip-vit-base-patch32",
                local_files_only=True,
            ).eval().to(device)
            self.mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device).view(1, 3, 1, 1)
            self.std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(1, 3, 1, 1)
        else:
            raise ValueError(name)

    @torch.no_grad()
    def __call__(self, spec: torch.Tensor):
        spec = spec.to(self.device, non_blocking=True)
        img = F.interpolate(spec.unsqueeze(1), size=(224, 224), mode="bilinear", align_corners=False)
        img = img.repeat(1, 3, 1, 1)
        img = (img - self.mean) / self.std
        if self.name == "clip_vit_b32":
            emb = self.model(pixel_values=img).pooler_output
        else:
            emb = self.model(img)
        return emb.detach().cpu().numpy().astype(np.float32)


@torch.no_grad()
def embed_split(extractor, df, cache_path, args, device):
    ds = CachedSplitDataset(df, cache_path)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=torch.cuda.is_available())
    embs, names = [], []
    for spec, filename in dl:
        embs.append(extractor(spec))
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
    run_name = args.run_name or f"frozen_{args.model}_logmel96_f16000"
    run_dir = args.out_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    cache_path = args.cache_dir / "logmel.npy"
    gallery, val, test = load_splits(args.cache_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extractor = VisionExtractor(args.model, device)
    meta = {
        "run_name": run_name,
        "model": args.model,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "rows": {"gallery": len(gallery), "reference": len(gallery), "val": len(val), "test": len(test)},
    }
    (run_dir / "run_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False), flush=True)
    ref_emb, ref_names = embed_split(extractor, gallery, cache_path, args, device)
    val_emb, val_names = embed_split(extractor, val, cache_path, args, device)
    test_emb, test_names = embed_split(extractor, test, cache_path, args, device)
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
    pred = top5(test_scores, ships)
    sample = pd.read_csv(args.data_dir / "sample_submission_task2.csv")
    pred_map = dict(zip(test_names, [",".join(map(str, row)) for row in pred]))
    sub = sample.copy()
    sub["top5_ship_ids"] = sub["filename"].map(pred_map)
    targets = set(pd.read_csv(args.data_dir / "task2_target_ships.csv")["ship_id"].astype(int))
    errors = validate_submission(sub, sample, targets)
    sub.to_csv(run_dir / "submission_task2.csv", index=False)
    summary = {
        "run_meta": meta,
        "best_validation": best,
        "submission_validation_errors": errors,
        "artifacts": {"submission": str(run_dir / "submission_task2.csv"), "leaderboard": str(run_dir / "retrieval_leaderboard.csv")},
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    with Path("analysis/task2_execution_log.md").open("a", encoding="utf-8") as f:
        f.write(f"""
## Frozen Pretrained Run: {run_name}

log-mel cache를 이미지 입력처럼 사용해 frozen pretrained vision encoder embedding을 평가했다.

- model: `{args.model}`
- cache: `{args.cache_dir}`
- output: `{run_dir}`
- best retrieval: `{best['retrieval']}`
- alpha: `{best['alpha']}`
- R@1: `{best['R@1']:.6f}`
- R@3: `{best['R@3']:.6f}`
- R@5: `{best['R@5']:.6f}`
- Score: `{best['Score']:.6f}`
- submission validation errors: `{errors}`

해석:

- 이 run은 fine-tuning 없이 사전학습 encoder가 만든 embedding만으로 retrieval한 결과다.
- scratch CNN보다 높으면 pretrained embedding을 앙상블 후보로 사용하고, 낮으면 fine-tuning 전에는 단독 사용하지 않는다.

""")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
