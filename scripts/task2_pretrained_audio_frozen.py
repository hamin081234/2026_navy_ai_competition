#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=Path("database"))
    p.add_argument("--out-dir", type=Path, default=Path("outputs/task2_runs"))
    p.add_argument("--model", choices=["ast", "clap", "panns"], default="ast")
    p.add_argument("--run-name", default=None)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--embed-train-targets", action="store_true")
    return p.parse_args()


def read_audio(path: Path, target_sr: int):
    wav, sr = sf.read(path, dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    wav = torch.from_numpy(np.asarray(wav, dtype=np.float32))
    wav = wav - wav.mean()
    peak = wav.abs().max().clamp_min(1e-6)
    wav = wav / peak
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    return wav.numpy()


def load_tables(data_dir: Path):
    gallery = pd.read_csv(data_dir / "task2_test/gallery.csv")
    val = pd.read_csv(data_dir / "task2_test/val.csv")
    test = pd.read_csv(data_dir / "task2_test/test.csv")
    for df in [gallery, val, test]:
        df["audio_path"] = df["filename"].map(lambda x: data_dir / "task2_test/audio" / x)
    return gallery, val, test


def load_target_train(data_dir: Path):
    train = pd.read_csv(data_dir / "train/train.csv")
    targets = set(pd.read_csv(data_dir / "task2_target_ships.csv")["ship_id"].astype(int).tolist())
    train = train[train["ship_id"].astype(int).isin(targets)].reset_index(drop=True)
    train["audio_path"] = train["filename"].map(lambda x: data_dir / "train/audio" / x)
    return train


class ASTFrozen:
    name = "ast"

    def __init__(self, device: str):
        from transformers import ASTFeatureExtractor, ASTModel

        self.device = torch.device(device)
        model_id = "MIT/ast-finetuned-audioset-10-10-0.4593"
        self.extractor = ASTFeatureExtractor.from_pretrained(model_id, local_files_only=True)
        self.model = ASTModel.from_pretrained(model_id, local_files_only=True).eval().to(self.device)
        self.target_sr = int(getattr(self.extractor, "sampling_rate", 16000) or 16000)

    @torch.no_grad()
    def embed_paths(self, paths: list[str], batch_size: int):
        rows = []
        for start in range(0, len(paths), batch_size):
            batch_paths = paths[start:start + batch_size]
            audios = [read_audio(Path(p), self.target_sr) for p in batch_paths]
            inputs = self.extractor(audios, sampling_rate=self.target_sr, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            out = self.model(**inputs).last_hidden_state[:, 0]
            rows.append(out.detach().cpu().numpy().astype(np.float32))
            if (start + len(batch_paths)) % (batch_size * 20) == 0 or start + len(batch_paths) == len(paths):
                print(json.dumps({"embedded": start + len(batch_paths), "total": len(paths)}, ensure_ascii=False), flush=True)
        return np.vstack(rows)


class CLAPFrozen:
    name = "clap"

    def __init__(self, device: str):
        from transformers import ClapModel, ClapProcessor

        self.device = torch.device(device)
        model_id = "laion/clap-htsat-unfused"
        self.processor = ClapProcessor.from_pretrained(model_id, local_files_only=True)
        self.model = ClapModel.from_pretrained(model_id, local_files_only=True).eval().to(self.device)
        self.target_sr = int(getattr(self.processor.feature_extractor, "sampling_rate", 48000) or 48000)

    @torch.no_grad()
    def embed_paths(self, paths: list[str], batch_size: int):
        rows = []
        for start in range(0, len(paths), batch_size):
            batch_paths = paths[start:start + batch_size]
            audios = [read_audio(Path(p), self.target_sr) for p in batch_paths]
            inputs = self.processor(audio=audios, sampling_rate=self.target_sr, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            out = self.model.get_audio_features(**inputs)
            if hasattr(out, "pooler_output") and out.pooler_output is not None:
                out = out.pooler_output
            elif hasattr(out, "last_hidden_state") and out.last_hidden_state is not None:
                out = out.last_hidden_state.mean(dim=1)
            rows.append(out.detach().cpu().numpy().astype(np.float32))
            if (start + len(batch_paths)) % (batch_size * 20) == 0 or start + len(batch_paths) == len(paths):
                print(json.dumps({"embedded": start + len(batch_paths), "total": len(paths)}, ensure_ascii=False), flush=True)
        return np.vstack(rows)


class PANNsFrozen:
    name = "panns"

    def __init__(self, device: str):
        from panns_inference import AudioTagging

        self.device = device
        self.target_sr = 32000
        self.wrapper = AudioTagging(device=device)

    @torch.no_grad()
    def embed_paths(self, paths: list[str], batch_size: int):
        rows = []
        for start in range(0, len(paths), batch_size):
            batch_paths = paths[start:start + batch_size]
            audios = [read_audio(Path(p), self.target_sr) for p in batch_paths]
            max_len = max(len(a) for a in audios)
            batch = np.zeros((len(audios), max_len), dtype=np.float32)
            for i, audio in enumerate(audios):
                batch[i, : len(audio)] = audio
            _, emb = self.wrapper.inference(batch)
            rows.append(emb.astype(np.float32))
            if (start + len(batch_paths)) % (batch_size * 20) == 0 or start + len(batch_paths) == len(paths):
                print(json.dumps({"embedded": start + len(batch_paths), "total": len(paths)}, ensure_ascii=False), flush=True)
        return np.vstack(rows)


def l2(x):
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-8, None)


def score_matrix(query_emb, ref_emb, ref_ship_ids, method, topk=5, alpha=0.0):
    query_emb, ref_emb = l2(query_emb), l2(ref_emb)
    ships = np.array(sorted(set(ref_ship_ids.tolist())), dtype=int)
    sim = query_emb @ ref_emb.T
    scores = np.zeros((len(query_emb), len(ships)), dtype=np.float32)
    for j, sid in enumerate(ships):
        cols = np.where(ref_ship_ids == sid)[0]
        vals = sim[:, cols]
        if method == "prototype":
            proto = l2(ref_emb[cols].mean(axis=0, keepdims=True))
            scores[:, j] = (query_emb @ proto.T).reshape(-1)
        elif method == "max":
            scores[:, j] = vals.max(axis=1)
        elif method == "topk":
            k = min(topk, vals.shape[1])
            part = np.partition(vals, kth=vals.shape[1] - k, axis=1)[:, -k:]
            scores[:, j] = part.mean(axis=1)
        else:
            raise ValueError(method)
        if alpha:
            scores[:, j] -= alpha * math.log1p(len(cols))
    return scores, ships


def hybrid(query_emb, ref_emb, ref_ship_ids, weights, topk, alpha):
    p, ships = score_matrix(query_emb, ref_emb, ref_ship_ids, "prototype", alpha=0)
    t, _ = score_matrix(query_emb, ref_emb, ref_ship_ids, "topk", topk=topk, alpha=0)
    m, _ = score_matrix(query_emb, ref_emb, ref_ship_ids, "max", alpha=0)
    scores = weights[0] * p + weights[1] * t + weights[2] * m
    if alpha:
        for j, sid in enumerate(ships):
            scores[:, j] -= alpha * math.log1p(int(np.sum(ref_ship_ids == sid)))
    return scores, ships


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
            rows.append(row)
            candidates.append((row, scores, ships))
    for topk in [3, 5, 10]:
        for alpha in [0, 0.005, 0.01, 0.02, 0.03]:
            scores, ships = score_matrix(val_emb, ref_emb, ref_ship_ids, "topk", topk=topk, alpha=alpha)
            row = {"retrieval": "topk", "topk": topk, "weights": None, "alpha": alpha, **metric(y, top5(scores, ships))}
            rows.append(row)
            candidates.append((row, scores, ships))
    for topk in [3, 5, 10]:
        for weights in [(0.5, 0.3, 0.2), (0.4, 0.4, 0.2), (0.3, 0.5, 0.2), (0.6, 0.3, 0.1)]:
            for alpha in [0, 0.005, 0.01, 0.02, 0.03]:
                scores, ships = hybrid(val_emb, ref_emb, ref_ship_ids, weights, topk, alpha)
                row = {"retrieval": "hybrid", "topk": topk, "weights": str(weights), "alpha": alpha, **metric(y, top5(scores, ships))}
                rows.append(row)
                candidates.append((row, scores, ships))
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
    run_name = args.run_name or f"frozen_{args.model}_audio"
    run_dir = args.out_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    gallery, val, test = load_tables(args.data_dir)
    if args.model == "ast":
        extractor = ASTFrozen(args.device)
    elif args.model == "clap":
        extractor = CLAPFrozen(args.device)
    elif args.model == "panns":
        extractor = PANNsFrozen(args.device)
    else:
        raise ValueError(args.model)
    meta = {
        "run_name": run_name,
        "model": args.model,
        "device": args.device,
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "rows": {"gallery": len(gallery), "reference": len(gallery), "val": len(val), "test": len(test)},
    }
    (run_dir / "run_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False), flush=True)

    if (run_dir / "reference_embeddings.npy").exists() and (run_dir / "val_embeddings.npy").exists() and (run_dir / "test_embeddings.npy").exists():
        ref_emb = np.load(run_dir / "reference_embeddings.npy")
        val_emb = np.load(run_dir / "val_embeddings.npy")
        test_emb = np.load(run_dir / "test_embeddings.npy")
        print(json.dumps({"loaded_existing_embeddings": str(run_dir)}, ensure_ascii=False), flush=True)
    else:
        ref_emb = extractor.embed_paths(gallery["audio_path"].astype(str).tolist(), args.batch_size)
        val_emb = extractor.embed_paths(val["audio_path"].astype(str).tolist(), args.batch_size)
        test_emb = extractor.embed_paths(test["audio_path"].astype(str).tolist(), args.batch_size)
        np.save(run_dir / "reference_embeddings.npy", ref_emb)
        np.save(run_dir / "val_embeddings.npy", val_emb)
        np.save(run_dir / "test_embeddings.npy", test_emb)
    if args.embed_train_targets:
        train = load_target_train(args.data_dir)
        train_emb = extractor.embed_paths(train["audio_path"].astype(str).tolist(), args.batch_size)
        np.save(run_dir / "train_embeddings.npy", train_emb)
        train.drop(columns=["audio_path"]).to_csv(run_dir / "train_metadata.csv", index=False)

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
    pred_map = dict(zip(test["filename"], [",".join(map(str, row)) for row in pred]))
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
## Frozen Audio Pretrained Run: {run_name}

raw audio를 직접 입력으로 받는 사전학습 audio encoder의 frozen embedding을 평가했다.

- model: `{args.model}`
- output: `{run_dir}`
- best retrieval: `{best['retrieval']}`
- alpha: `{best['alpha']}`
- R@1: `{best['R@1']:.6f}`
- R@3: `{best['R@3']:.6f}`
- R@5: `{best['R@5']:.6f}`
- Score: `{best['Score']:.6f}`
- submission validation errors: `{errors}`

해석:

- 이 run은 fine-tuning 없이 사전학습 audio encoder의 embedding만 사용한다.
- 기존 scratch/vision-pretrained 모델과 score ensemble을 다시 구성할 후보로 평가한다.

""")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
