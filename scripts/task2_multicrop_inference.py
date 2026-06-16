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

import task2_gallery_aggregation_sweep as agg
import task2_train_cached_retrieval as train_base


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=Path("database"))
    p.add_argument("--runs-dir", type=Path, default=Path("outputs/task2_runs"))
    p.add_argument("--out-dir", type=Path, default=Path("outputs/task2_multicrop_inference"))
    p.add_argument("--models", nargs="*", default=[
        "robust_rows100_seed123_ep2_aug:model_last.pt",
        "checkpoint_rows100_seed777_ep8_aug:model_epoch3.pt",
    ])
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--crop-seconds", nargs="*", type=float, default=[5.0, 4.0, 3.0])
    p.add_argument("--n-mels", type=int, default=96)
    p.add_argument("--f-max", type=int, default=16000)
    p.add_argument("--hop-length", type=int, default=320)
    return p.parse_args()


def parse_model_entry(entry: str):
    if ":" in entry:
        return entry.split(":", 1)
    return entry, "model_last.pt"


def normalize(scores: np.ndarray) -> np.ndarray:
    return (scores - scores.mean(axis=1, keepdims=True)) / (scores.std(axis=1, keepdims=True) + 1e-6)


def read_wave(path: Path) -> np.ndarray:
    wav, sr = sf.read(path, dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    wav = np.asarray(wav, dtype=np.float32)
    target = 160000
    if len(wav) < target:
        wav = np.pad(wav, (0, target - len(wav)))
    elif len(wav) > target:
        wav = wav[:target]
    wav = wav - np.mean(wav, dtype=np.float32)
    peak = np.max(np.abs(wav)) + 1e-6
    return (wav / peak).astype(np.float32)


def make_crops(wav: np.ndarray, crop_seconds: list[float]):
    target = 160000
    out = []
    for seconds in crop_seconds:
        length = min(target, max(1, int(seconds * 32000)))
        if length >= target:
            starts = [0]
        else:
            starts = sorted(set([0, (target - length) // 2, target - length]))
        for start in starts:
            crop = wav[start:start + length]
            if len(crop) < target:
                pad_left = (target - len(crop)) // 2
                pad_right = target - len(crop) - pad_left
                crop = np.pad(crop, (pad_left, pad_right))
            crop = crop - np.mean(crop, dtype=np.float32)
            peak = np.max(np.abs(crop)) + 1e-6
            out.append((crop / peak).astype(np.float32))
    return out


def load_tables(data_dir: Path):
    gallery = pd.read_csv(data_dir / "task2_test/gallery.csv")
    gallery["audio_path"] = gallery["filename"].map(lambda x: data_dir / "task2_test/audio" / x)
    val = pd.read_csv(data_dir / "task2_test/val.csv")
    val["audio_path"] = val["filename"].map(lambda x: data_dir / "task2_test/audio" / x)
    test = pd.read_csv(data_dir / "task2_test/test.csv")
    test["audio_path"] = test["filename"].map(lambda x: data_dir / "task2_test/audio" / x)
    targets = sorted(pd.read_csv(data_dir / "task2_target_ships.csv")["ship_id"].astype(int).tolist())
    label_map = {sid: i for i, sid in enumerate(targets)}
    return gallery, val, test, label_map


def load_model(run_dir: Path, ckpt_name: str, n_classes: int, device):
    ckpt = torch.load(run_dir / ckpt_name, map_location="cpu")
    args = ckpt.get("args", {})
    emb_dim = int(args.get("emb_dim", 256)) if isinstance(args, dict) else 256
    model = train_base.CachedSpecCNN(n_classes, emb_dim)
    model.load_state_dict(ckpt["model_state"])
    return model.to(device), {int(k): int(v) for k, v in (ckpt.get("label_map") or {}).items()}


@torch.no_grad()
def infer_multicrop(model, df, mel, args, device):
    model.eval()
    all_emb, all_logits, file_index = [], [], []
    names = df["filename"].astype(str).tolist()
    batch = []
    batch_file_idx = []
    for file_idx, path in enumerate(df["audio_path"]):
        wav = read_wave(Path(path))
        for crop in make_crops(wav, args.crop_seconds):
            batch.append(crop)
            batch_file_idx.append(file_idx)
            if len(batch) >= args.batch_size:
                emb, logits = infer_batch(model, mel, batch, device)
                all_emb.append(emb)
                all_logits.append(logits)
                file_index.extend(batch_file_idx)
                batch, batch_file_idx = [], []
    if batch:
        emb, logits = infer_batch(model, mel, batch, device)
        all_emb.append(emb)
        all_logits.append(logits)
        file_index.extend(batch_file_idx)
    return {
        "emb": np.vstack(all_emb),
        "logits": np.vstack(all_logits),
        "file_index": np.asarray(file_index, dtype=np.int64),
        "names": names,
        "num_files": len(df),
    }


def infer_batch(model, mel, wavs, device):
    x = torch.from_numpy(np.stack(wavs)).to(device)
    spec = torch.log(mel(x) + 1e-6)
    mean = spec.mean(dim=(1, 2), keepdim=True)
    std = spec.std(dim=(1, 2), keepdim=True).clamp_min(1e-5)
    spec = (spec - mean) / std
    logits, emb = model(spec)
    return emb.detach().cpu().numpy().astype(np.float32), logits.detach().cpu().numpy().astype(np.float32)


def reduce_by_file(values: np.ndarray, file_index: np.ndarray, n_files: int, mode: str):
    out = np.zeros((n_files, values.shape[1]), dtype=np.float32)
    for i in range(n_files):
        rows = values[file_index == i]
        if mode == "mean":
            out[i] = rows.mean(axis=0)
        elif mode == "max":
            out[i] = rows.max(axis=0)
        else:
            raise ValueError(mode)
    return out


def ship_scores_from_views(query, ref, gallery, q_reduce: str):
    ref_emb = agg.l2(ref["emb"].astype(np.float32))
    query_emb = agg.l2(query["emb"].astype(np.float32))
    # Repeat ship ids per gallery crop.
    gallery_ship = gallery["ship_id"].astype(int).to_numpy()
    ref_ship_ids = gallery_ship[ref["file_index"]]
    weights = agg.stable_dimension_weights(ref_emb, ref_ship_ids)
    ref_w, query_w = agg.apply_weights(ref_emb, query_emb, weights)
    ships, groups, counts = agg.ship_index(ref_ship_ids)
    crop_scores = agg.score_variant(query_w, ref_w, groups, counts, "topk_mean", 20, 0.0)
    file_scores = reduce_by_file(crop_scores, query["file_index"], query["num_files"], q_reduce)
    return normalize(file_scores), ships


def class_scores(query, label_map, ships, mode: str):
    cols = [label_map[int(ship)] for ship in ships]
    logits = reduce_by_file(query["logits"], query["file_index"], query["num_files"], "mean")[:, cols]
    if mode == "logit":
        return normalize(logits.astype(np.float32))
    z = logits - logits.max(axis=1, keepdims=True)
    prob = np.exp(z)
    prob = prob / np.clip(prob.sum(axis=1, keepdims=True), 1e-8, None)
    return normalize(prob.astype(np.float32))


def metric(scores, ships, y):
    pred = agg.top5(scores, ships)
    return {**agg.metric(y, pred), **agg.prediction_stats(pred)}


def make_submission(path, sample, names, pred):
    sub = sample.copy()
    pred_map = dict(zip(names, [",".join(map(str, row)) for row in pred]))
    sub["top5_ship_ids"] = sub["filename"].map(pred_map)
    sub.to_csv(path, index=False)


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    gallery, val, test, fallback_label_map = load_tables(args.data_dir)
    sample = pd.read_csv(args.data_dir / "sample_submission_task2.csv")
    y_val = val["ship_id"].astype(int).to_numpy()
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
    rows = []
    selected = []
    for entry in args.models:
        run, ckpt_name = parse_model_entry(entry)
        safe = f"{run}__{Path(ckpt_name).stem}".replace(".", "_")
        print(json.dumps({"multicrop": safe}, ensure_ascii=False), flush=True)
        model, label_map = load_model(args.runs_dir / run, ckpt_name, len(fallback_label_map), device)
        if not label_map:
            label_map = fallback_label_map
        ref = infer_multicrop(model, gallery, mel, args, device)
        val_out = infer_multicrop(model, val, mel, args, device)
        test_out = infer_multicrop(model, test, mel, args, device)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        for q_reduce in ["mean", "max"]:
            val_ret, ships = ship_scores_from_views(val_out, ref, gallery, q_reduce)
            test_ret, _ = ship_scores_from_views(test_out, ref, gallery, q_reduce)
            val_logit = class_scores(val_out, label_map, ships, "logit")
            test_logit = class_scores(test_out, label_map, ships, "logit")
            candidates = {
                f"{safe}_ret_{q_reduce}": (val_ret, test_ret),
                f"{safe}_logit": (val_logit, test_logit),
                f"{safe}_ret_{q_reduce}_logit_95_05": (0.95 * val_ret + 0.05 * val_logit, 0.95 * test_ret + 0.05 * test_logit),
            }
            for name, (val_scores, test_scores) in candidates.items():
                m = metric(val_scores, ships, y_val)
                rows.append({"model": name, **m})
                pred = agg.top5(test_scores, ships)
                sub_path = args.out_dir / f"submission_{name}.csv"
                make_submission(sub_path, sample, test_out["names"], pred)
                selected.append({"model": name, "submission": str(sub_path), **m})
                print(json.dumps({"model": name, **m}, ensure_ascii=False), flush=True)
    pd.DataFrame(rows).sort_values(["Score", "R@1"], ascending=False).to_csv(args.out_dir / "multicrop_leaderboard.csv", index=False)
    pd.DataFrame(selected).sort_values(["Score", "R@1"], ascending=False).to_csv(args.out_dir / "multicrop_selected.csv", index=False)


if __name__ == "__main__":
    main()
