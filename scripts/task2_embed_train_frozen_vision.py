#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


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
        elif name == "clip_vit_b32":
            from transformers import CLIPVisionModel
            self.model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32", local_files_only=True).eval().to(device)
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
def embed(extractor, df, cache_path, batch_size, num_workers):
    ds = CachedSplitDataset(df, cache_path)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available())
    rows, names = [], []
    for spec, filename in dl:
        rows.append(extractor(spec))
        names.extend(filename)
        done = len(names)
        if done % (batch_size * 20) == 0 or done == len(df):
            print(json.dumps({"embedded": done, "total": len(df)}, ensure_ascii=False), flush=True)
    return np.vstack(rows), names


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cache-dir", type=Path, default=Path("outputs/task2_cache/logmel96_f16000"))
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--model", choices=["resnet18", "clip_vit_b32"], required=True)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    meta = pd.read_csv(args.cache_dir / "metadata.csv")
    train = meta[meta["split"] == "train_target"].copy().reset_index(drop=True)
    cache_path = args.cache_dir / "logmel.npy"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extractor = VisionExtractor(args.model, device)
    emb, names = embed(extractor, train, cache_path, args.batch_size, args.num_workers)
    np.save(args.run_dir / "train_embeddings.npy", emb)
    train.to_csv(args.run_dir / "train_metadata.csv", index=False)
    summary = {
        "run_dir": str(args.run_dir),
        "model": args.model,
        "train_rows": len(train),
        "embedding_shape": list(emb.shape),
        "artifact": str(args.run_dir / "train_embeddings.npy"),
    }
    (args.run_dir / "train_embedding_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
