#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=Path("database"))
    p.add_argument("--out-dir", type=Path, default=Path("outputs/task2_cache/logmel96_f16000"))
    p.add_argument("--n-mels", type=int, default=96)
    p.add_argument("--f-max", type=int, default=16000)
    p.add_argument("--hop-length", type=int, default=320)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--max-train-rows", type=int, default=None)
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def read_wave(path: Path) -> np.ndarray:
    wav, sr = sf.read(path, dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    wav = np.asarray(wav, dtype=np.float32)
    if len(wav) < 160000:
        wav = np.pad(wav, (0, 160000 - len(wav)))
    elif len(wav) > 160000:
        wav = wav[:160000]
    wav = wav - np.mean(wav, dtype=np.float32)
    peak = np.max(np.abs(wav)) + 1e-6
    return (wav / peak).astype(np.float32)


def load_tables(data_dir: Path, max_train_rows: int | None):
    target_ids = set(pd.read_csv(data_dir / "task2_target_ships.csv")["ship_id"].astype(int))
    train = pd.read_csv(data_dir / "train/train.csv")
    train = train[train["ship_id"].isin(target_ids)].copy()
    train["split"] = "train_target"
    train["audio_path"] = train["filename"].map(lambda x: data_dir / "train/audio" / x)
    if max_train_rows is not None and max_train_rows < len(train):
        train = (
            train.groupby("ship_id", group_keys=False)
            .apply(lambda x: x.sample(n=max(1, min(len(x), max_train_rows // len(target_ids))), random_state=42))
            .reset_index(drop=True)
        )
        if len(train) > max_train_rows:
            train = train.sample(n=max_train_rows, random_state=42).reset_index(drop=True)

    gallery = pd.read_csv(data_dir / "task2_test/gallery.csv")
    gallery["split"] = "gallery"
    gallery["audio_path"] = gallery["filename"].map(lambda x: data_dir / "task2_test/audio" / x)

    val = pd.read_csv(data_dir / "task2_test/val.csv")
    val["split"] = "val"
    val["audio_path"] = val["filename"].map(lambda x: data_dir / "task2_test/audio" / x)

    test = pd.read_csv(data_dir / "task2_test/test.csv")
    test["split"] = "test"
    test["audio_path"] = test["filename"].map(lambda x: data_dir / "task2_test/audio" / x)

    cols = ["split", "filename", "audio_path"]
    optional = ["ship_id", "ship_type"]
    frames = []
    for df in [train, gallery, val, test]:
        keep = cols + [c for c in optional if c in df.columns]
        frames.append(df[keep].copy())
    meta = pd.concat(frames, ignore_index=True, sort=False)
    meta["cache_index"] = np.arange(len(meta), dtype=np.int64)
    return meta


@torch.no_grad()
def build_cache(args):
    args.out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = args.out_dir / "metadata.csv"
    cache_path = args.out_dir / "logmel.npy"
    stats_path = args.out_dir / "stats.json"
    if cache_path.exists() and meta_path.exists() and not args.force:
        print(json.dumps({"status": "exists", "cache": str(cache_path), "metadata": str(meta_path)}, ensure_ascii=False))
        return

    meta = load_tables(args.data_dir, args.max_train_rows)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=32000,
        n_fft=1024,
        win_length=1024,
        hop_length=args.hop_length,
        n_mels=args.n_mels,
        f_min=20,
        f_max=args.f_max,
        power=2.0,
    ).to(device)

    # Determine frame count from one item.
    wav0 = torch.from_numpy(read_wave(Path(meta.audio_path.iloc[0]))).unsqueeze(0).to(device)
    spec0 = torch.log(mel(wav0) + 1e-6)
    n_frames = int(spec0.shape[-1])
    arr = np.lib.format.open_memmap(
        cache_path,
        mode="w+",
        dtype=np.float16,
        shape=(len(meta), args.n_mels, n_frames),
    )

    for start in range(0, len(meta), args.batch_size):
        end = min(start + args.batch_size, len(meta))
        wavs = [read_wave(Path(p)) for p in meta.audio_path.iloc[start:end]]
        x = torch.from_numpy(np.stack(wavs)).to(device)
        spec = torch.log(mel(x) + 1e-6)
        mean = spec.mean(dim=(1, 2), keepdim=True)
        std = spec.std(dim=(1, 2), keepdim=True).clamp_min(1e-5)
        spec = ((spec - mean) / std).detach().cpu().numpy().astype(np.float16)
        arr[start:end] = spec
        if end % (args.batch_size * 20) == 0 or end == len(meta):
            print(json.dumps({"cached": end, "total": len(meta)}, ensure_ascii=False), flush=True)

    arr.flush()
    meta.to_csv(meta_path, index=False)
    stats = {
        "cache": str(cache_path),
        "metadata": str(meta_path),
        "rows": int(len(meta)),
        "shape": [int(x) for x in arr.shape],
        "dtype": "float16",
        "n_mels": args.n_mels,
        "f_max": args.f_max,
        "hop_length": args.hop_length,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "split_counts": meta["split"].value_counts().to_dict(),
    }
    stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"status": "done", **stats}, ensure_ascii=False))


def main():
    args = parse_args()
    build_cache(args)


if __name__ == "__main__":
    main()
