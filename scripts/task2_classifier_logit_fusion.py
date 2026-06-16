#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import task2_gallery_aggregation_sweep as agg
import task2_train_cached_retrieval as train_base
from task2_robust_holdout_eval import per_ship_holdout_indices


DEFAULT_MODELS = [
    "robust_rows100_seed123_ep2_aug:model_last.pt",
    "robust_rows80_seed123_ep2_aug:model_last.pt",
    "robust_rows100_seed777_ep2_aug:model_last.pt",
    "checkpoint_rows100_seed123_ep8_aug:model_epoch3.pt",
    "checkpoint_rows100_seed777_ep8_aug:model_epoch3.pt",
    "final_rows100_seed777_ep8_aug:model_epoch3.pt",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=Path("database"))
    p.add_argument("--cache-dir", type=Path, default=Path("outputs/task2_cache/logmel96_f16000"))
    p.add_argument("--runs-dir", type=Path, default=Path("outputs/task2_runs"))
    p.add_argument("--out-dir", type=Path, default=Path("outputs/task2_classifier_logit_fusion"))
    p.add_argument("--models", nargs="*", default=DEFAULT_MODELS, help="Entries like run:model_epoch3.pt")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--grid-units", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--holdout-frac", type=float, default=0.2)
    return p.parse_args()


def parse_model_entry(entry: str):
    if ":" in entry:
        run, ckpt = entry.split(":", 1)
    else:
        run, ckpt = entry, "model_last.pt"
    return run, ckpt


def normalize(scores: np.ndarray) -> np.ndarray:
    return (scores - scores.mean(axis=1, keepdims=True)) / (scores.std(axis=1, keepdims=True) + 1e-6)


def softmax_np(logits: np.ndarray, temperature: float) -> np.ndarray:
    z = logits / max(temperature, 1e-6)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / np.clip(e.sum(axis=1, keepdims=True), 1e-8, None)


def metric(scores: np.ndarray, ships: np.ndarray, y: np.ndarray):
    pred = agg.top5(scores, ships)
    return {**agg.metric(y, pred), **agg.prediction_stats(pred)}


def make_submission(path: Path, sample: pd.DataFrame, test_names: list[str], pred: np.ndarray):
    sub = sample.copy()
    pred_map = dict(zip(test_names, [",".join(map(str, row)) for row in pred]))
    sub["top5_ship_ids"] = sub["filename"].map(pred_map)
    sub.to_csv(path, index=False)


def load_tables(args):
    shim = argparse.Namespace(
        data_dir=args.data_dir,
        cache_dir=args.cache_dir,
        max_target_rows=None,
        max_train_rows=None,
        no_gallery_train=False,
    )
    train, gallery, val, test, label_map = train_base.load_cached_tables(shim)
    return gallery, val, test, label_map


def checkpoint_label_map(ckpt, fallback):
    label_map = ckpt.get("label_map") or ckpt.get("ship_label_map") or fallback
    return {int(k): int(v) for k, v in label_map.items()}


@torch.no_grad()
def infer_model(model, df, cache_path: Path, args, device):
    ds = train_base.CachedSpecDataset(df, cache_path, with_label=False, augment=False)
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    embs, logits, names = [], [], []
    model.eval()
    for spec, filename in dl:
        spec = spec.to(device, non_blocking=True)
        out, emb = model(spec)
        embs.append(emb.detach().cpu().numpy().astype(np.float32))
        logits.append(out.detach().cpu().numpy().astype(np.float32))
        names.extend(filename)
    return {"emb": np.vstack(embs), "logits": np.vstack(logits), "names": names}


def load_model(run_dir: Path, ckpt_name: str, n_classes: int, device):
    ckpt = torch.load(run_dir / ckpt_name, map_location="cpu")
    args = ckpt.get("args", {})
    emb_dim = int(args.get("emb_dim", 256)) if isinstance(args, dict) else 256
    model = train_base.CachedSpecCNN(n_classes, emb_dim)
    state = ckpt["model_state"]
    model.load_state_dict(state)
    return model.to(device), ckpt


def retrieval_scores(ref_emb, query_emb, gallery):
    ref = agg.l2(ref_emb.astype(np.float32))
    query = agg.l2(query_emb.astype(np.float32))
    ship_ids = gallery["ship_id"].astype(int).to_numpy()
    weights = agg.stable_dimension_weights(ref, ship_ids)
    ref_w, query_w = agg.apply_weights(ref, query, weights)
    ships, groups, counts = agg.ship_index(ship_ids)
    scores = agg.score_variant(query_w, ref_w, groups, counts, "topk_mean", 20, 0.0)
    return normalize(scores), ships


def class_scores(logits: np.ndarray, label_map: dict[int, int], ships: np.ndarray, mode: str):
    inv = {label: ship for ship, label in label_map.items()}
    label_cols = [label_map[int(ship)] for ship in ships]
    raw = logits[:, label_cols].astype(np.float32)
    if mode == "logit":
        return normalize(raw)
    if mode == "prob_t1":
        return normalize(softmax_np(logits, 1.0)[:, label_cols].astype(np.float32))
    if mode == "prob_t2":
        return normalize(softmax_np(logits, 2.0)[:, label_cols].astype(np.float32))
    if mode == "prob_t4":
        return normalize(softmax_np(logits, 4.0)[:, label_cols].astype(np.float32))
    raise ValueError(mode)


def evaluate_components(name, val_components, test_components, ships, y_val, val_names, test_names, val_sample, sample, out_dir, units):
    rows = []
    selected_rows = []
    component_names = list(val_components.keys())
    if len(component_names) == 1:
        comps = [(units,)]
    else:
        comps = []
        for i in range(units + 1):
            for j in range(units - i + 1):
                if len(component_names) == 2:
                    comps.append((i, units - i))
                elif len(component_names) == 3:
                    comps.append((i, j, units - i - j))
        # Deduplicate when len == 2.
        comps = sorted(set(comps))
    for comp in comps:
        w = {n: c / units for n, c in zip(component_names, comp)}
        scores = sum(w[n] * val_components[n] for n in component_names)
        m = metric(scores, ships, y_val)
        row = {"model": name, "weights": str(tuple(w[n] for n in component_names)), **{f"w_{n}": w[n] for n in component_names}, **m}
        rows.append(row)
    df = pd.DataFrame(rows).sort_values(["Score", "R@1", "R@3"], ascending=False)
    for label, row in {
        "best_val": df.iloc[0],
        "low_concentration": df.sort_values(["top1_max_ship_fraction", "Score"], ascending=[True, False]).iloc[0],
    }.items():
        w = {n: float(row[f"w_{n}"]) for n in component_names}
        val_score = sum(w[n] * val_components[n] for n in component_names)
        test_score = sum(w[n] * test_components[n] for n in component_names)
        val_pred = agg.top5(val_score, ships)
        pred = agg.top5(test_score, ships)
        val_path = out_dir / f"validation_submission_{name}_{label}.csv"
        sub_path = out_dir / f"submission_{name}_{label}.csv"
        make_submission(val_path, val_sample, val_names, val_pred)
        make_submission(sub_path, sample, test_names, pred)
        selected_rows.append({"label": label, "validation_submission": str(val_path), "submission": str(sub_path), **row.to_dict()})
    return df, pd.DataFrame(selected_rows)


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    gallery, val, test, fallback_label_map = load_tables(args)
    sample = pd.read_csv(args.data_dir / "sample_submission_task2.csv")
    val_sample = val[["filename"]].copy()
    y_val = val["ship_id"].astype(int).to_numpy()
    cache_path = args.cache_dir / "logmel.npy"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    all_rows = []
    all_selected = []
    model_outputs = {}
    for entry in args.models:
        run, ckpt_name = parse_model_entry(entry)
        run_dir = args.runs_dir / run
        if not (run_dir / ckpt_name).exists():
            print(json.dumps({"skip_missing": entry}, ensure_ascii=False), flush=True)
            continue
        safe_name = f"{run}__{Path(ckpt_name).stem}".replace(".", "_")
        print(json.dumps({"infer": safe_name, "checkpoint": str(run_dir / ckpt_name)}, ensure_ascii=False), flush=True)
        model, ckpt = load_model(run_dir, ckpt_name, len(fallback_label_map), device)
        label_map = checkpoint_label_map(ckpt, fallback_label_map)
        ref = infer_model(model, gallery, cache_path, args, device)
        val_out = infer_model(model, val, cache_path, args, device)
        test_out = infer_model(model, test, cache_path, args, device)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        ret_val, ships = retrieval_scores(ref["emb"], val_out["emb"], gallery)
        ret_test, _ = retrieval_scores(ref["emb"], test_out["emb"], gallery)
        val_components = {"retrieval": ret_val}
        test_components = {"retrieval": ret_test}
        for mode in ["logit", "prob_t1", "prob_t2", "prob_t4"]:
            val_components[mode] = class_scores(val_out["logits"], label_map, ships, mode)
            test_components[mode] = class_scores(test_out["logits"], label_map, ships, mode)

        # Evaluate useful pairs/triples without exploding the search.
        component_sets = {
            "ret_logit": ["retrieval", "logit"],
            "ret_prob2": ["retrieval", "prob_t2"],
            "ret_logit_prob2": ["retrieval", "logit", "prob_t2"],
            "logits_only": ["logit"],
        }
        for set_name, names in component_sets.items():
            sub_val = {n: val_components[n] for n in names}
            sub_test = {n: test_components[n] for n in names}
            df, selected = evaluate_components(
                f"{safe_name}__{set_name}",
                sub_val,
                sub_test,
                ships,
                y_val,
                val_out["names"],
                test_out["names"],
                val_sample,
                sample,
                args.out_dir,
                args.grid_units,
            )
            all_rows.append(df)
            all_selected.append(selected)
        model_outputs[safe_name] = {
            "run": run,
            "checkpoint": ckpt_name,
            "num_classes": len(label_map),
            "val_shape": list(val_out["logits"].shape),
        }

    leaderboard = pd.concat(all_rows, ignore_index=True).sort_values(["Score", "R@1", "R@3"], ascending=False)
    selected = pd.concat(all_selected, ignore_index=True)
    leaderboard.to_csv(args.out_dir / "classifier_logit_fusion_leaderboard.csv", index=False)
    selected.to_csv(args.out_dir / "classifier_logit_fusion_selected.csv", index=False)

    # Global rank fusion over top selected submissions will be handled by a separate script if useful.
    summary = {
        "models": model_outputs,
        "best_validation": leaderboard.iloc[0].to_dict(),
        "leaderboard": str(args.out_dir / "classifier_logit_fusion_leaderboard.csv"),
        "selected": str(args.out_dir / "classifier_logit_fusion_selected.csv"),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
