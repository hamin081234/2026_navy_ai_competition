#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

import task2_train_cached_retrieval as tr
from task2_run_robust_local_sweep import score_submission
from task2_type_gated_retrieval import (
    hard_gate_with_fill,
    make_submission,
    metric,
    soft_boost,
    top5,
    type_accuracy,
    type_score_matrix,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=Path, default=Path("outputs/task2_runs/checkpoint_rows100_seed777_ep8_aug"))
    p.add_argument("--epoch", type=int, default=2)
    p.add_argument("--data-dir", type=Path, default=Path("database"))
    p.add_argument("--cache-dir", type=Path, default=Path("outputs/task2_cache/logmel96_f16000"))
    p.add_argument("--out-dir", type=Path, default=Path("outputs/task2_runs/subclass_type_gated_retrieval"))
    p.add_argument("--report", type=Path, default=Path("analysis/task2_subclass_type_gated_retrieval.md"))
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=0)
    return p.parse_args()


def l2(x):
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-8, None)


def timestamp_key(df, mode):
    ts = pd.to_datetime(df["ais_timestamp"], utc=True)
    if mode == "exact":
        return df["ais_timestamp"].astype(str)
    if mode == "date":
        return ts.dt.strftime("%Y-%m-%d")
    if mode == "hour":
        return ts.dt.strftime("%Y-%m-%d %H")
    if mode == "10min":
        return ts.dt.floor("10min").dt.strftime("%Y-%m-%d %H:%M")
    raise ValueError(mode)


def make_subclasses(gallery, ref_emb, mode):
    g = gallery.reset_index(drop=True).copy()
    g["_time_key"] = timestamp_key(g, mode)
    rows, protos = [], []
    for (sid, key), idx in g.groupby(["ship_id", "_time_key"], sort=True).groups.items():
        idx = np.array(list(idx), dtype=int)
        proto = ref_emb[idx].mean(axis=0)
        protos.append(proto)
        rows.append(
            {
                "ship_id": int(sid),
                "ship_type": str(g.loc[idx[0], "ship_type"]),
                "subclass_key": str(key),
                "size": int(len(idx)),
            }
        )
    sub = pd.DataFrame(rows)
    proto = l2(np.vstack(protos).astype(np.float32))
    return sub, proto


def subclass_score_matrix(query_emb, subclass_emb, sub_df, method, topk=5, size_beta=0.0):
    query_emb, subclass_emb = l2(query_emb), l2(subclass_emb)
    ships = np.array(sorted(sub_df["ship_id"].astype(int).unique().tolist()), dtype=int)
    sim = query_emb @ subclass_emb.T
    out = np.zeros((len(query_emb), len(ships)), dtype=np.float32)
    for j, sid in enumerate(ships):
        cols = np.where(sub_df["ship_id"].astype(int).to_numpy() == sid)[0]
        vals = sim[:, cols].copy()
        if size_beta:
            sizes = np.log1p(sub_df.iloc[cols]["size"].to_numpy(dtype=np.float32))
            vals = vals + size_beta * sizes[None, :]
        if method == "max":
            out[:, j] = vals.max(axis=1)
        elif method == "topk":
            k = min(topk, vals.shape[1])
            part = np.partition(vals, kth=vals.shape[1] - k, axis=1)[:, -k:]
            out[:, j] = part.mean(axis=1)
        elif method == "softmax":
            temp = float(topk)
            z = vals / max(temp, 1e-6)
            z = z - z.max(axis=1, keepdims=True)
            w = np.exp(z)
            w = w / np.clip(w.sum(axis=1, keepdims=True), 1e-8, None)
            out[:, j] = (vals * w).sum(axis=1)
        else:
            raise ValueError(method)
    return out, ships


def md_table(df, cols):
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, r in df.iterrows():
        vals = []
        for c in cols:
            v = r[c]
            if c in {
                "R@1",
                "R@3",
                "R@5",
                "Score",
                "diagnostic_R@1",
                "diagnostic_R@3",
                "diagnostic_R@5",
                "diagnostic_Score",
                "type_top1_acc",
                "type_top2_acc",
                "size_beta",
            }:
                vals.append(f"{float(v):.6f}")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)

    run_meta = json.loads((args.run_dir / "run_meta.json").read_text(encoding="utf-8"))
    seed = int(run_meta.get("seed", run_meta["args"].get("seed", 777)))
    max_target_rows = run_meta["args"].get("max_target_rows", 100)
    model_args = SimpleNamespace(
        data_dir=args.data_dir,
        cache_dir=args.cache_dir,
        max_target_rows=None if max_target_rows is None else int(max_target_rows),
        max_train_rows=run_meta["args"].get("max_train_rows"),
        no_gallery_train=bool(run_meta["args"].get("no_gallery_train", False)),
        seed=seed,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    _, gallery, val, test, label_map = tr.load_cached_tables(model_args)
    if "ais_timestamp" not in gallery.columns:
        gallery_raw = pd.read_csv(args.data_dir / "task2_test/gallery.csv")[["filename", "ais_timestamp"]]
        gallery = gallery.merge(gallery_raw, on="filename", how="left", validate="one_to_one")
    ckpt_path = args.run_dir / f"model_epoch{args.epoch}.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu")
    emb_dim = int(run_meta["args"].get("emb_dim", 256))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = tr.CachedSpecCNN(len(label_map), emb_dim).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    cache_path = args.cache_dir / "logmel.npy"
    ref_emb, _ = tr.embed_df(model, gallery, cache_path, model_args, device)
    val_emb, _ = tr.embed_df(model, val, cache_path, model_args, device)
    test_emb, test_names = tr.embed_df(model, test, cache_path, model_args, device)

    ref_ship_ids = gallery["ship_id"].astype(int).to_numpy()
    ref_types = gallery["ship_type"].astype(str).to_numpy()
    y = val["ship_id"].astype(int).to_numpy()
    y_types = val["ship_type"].astype(str).to_numpy()
    ship_to_type = gallery.groupby("ship_id")["ship_type"].first().astype(str).to_dict()
    sample = pd.read_csv(args.data_dir / "sample_submission_task2.csv")

    type_scores_val, types = type_score_matrix(val_emb, ref_emb, ref_types, "prototype", topk=20)
    type_scores_test, _ = type_score_matrix(test_emb, ref_emb, ref_types, "prototype", topk=20)
    type_top1 = type_accuracy(y_types, type_scores_val, types, k=1)
    type_top2 = type_accuracy(y_types, type_scores_val, types, k=2)

    rows = []

    # Include the current best type-gated file-level setting as baseline.
    file_scores_val, ships = tr.score_matrix(val_emb, ref_emb, ref_ship_ids, "topk", topk=15, alpha=0)
    file_scores_test, _ = tr.score_matrix(test_emb, ref_emb, ref_ship_ids, "topk", topk=15, alpha=0)
    pred_val = hard_gate_with_fill(file_scores_val, ships, ship_to_type, type_scores_val, types, 2)
    pred_test = hard_gate_with_fill(file_scores_test, ships, ship_to_type, type_scores_test, types, 2)
    sub_path = args.out_dir / "baseline_file_topk15_type_top2.csv"
    make_submission(sub_path, sample, test_names, pred_test)
    diag = score_submission(sub_path, args.data_dir)
    rows.append(
        {
            "strategy": "file_topk15_type_top2",
            "time_mode": "none",
            "subclass_method": "file_topk",
            "subclass_topk": 15,
            "size_beta": 0.0,
            "type_gate": "hard_top2",
            "type_top1_acc": type_top1,
            "type_top2_acc": type_top2,
            **metric(y, pred_val),
            "diagnostic_R@1": diag["test_R@1"],
            "diagnostic_R@3": diag["test_R@3"],
            "diagnostic_R@5": diag["test_R@5"],
            "diagnostic_Score": diag["test_Score"],
            "submission": str(sub_path),
        }
    )

    for mode in ["exact", "date", "hour", "10min"]:
        sub_df, sub_emb = make_subclasses(gallery, ref_emb, mode)
        sub_df.to_csv(args.out_dir / f"subclasses_{mode}.csv", index=False)
        for method in ["max", "topk", "softmax"]:
            topks = [1] if method == "max" else ([1, 2, 3, 5] if method == "topk" else [0.03, 0.05, 0.08, 0.12])
            for topk in topks:
                for size_beta in [0.0, 0.02, 0.05]:
                    val_scores, ships = subclass_score_matrix(val_emb, sub_emb, sub_df, method, topk=topk, size_beta=size_beta)
                    test_scores, _ = subclass_score_matrix(test_emb, sub_emb, sub_df, method, topk=topk, size_beta=size_beta)

                    variants = [
                        ("no_type", top5(val_scores, ships), top5(test_scores, ships)),
                        ("hard_top2", hard_gate_with_fill(val_scores, ships, ship_to_type, type_scores_val, types, 2), hard_gate_with_fill(test_scores, ships, ship_to_type, type_scores_test, types, 2)),
                        ("soft_b0.5", top5(soft_boost(val_scores, ships, ship_to_type, type_scores_val, types, 0.5), ships), top5(soft_boost(test_scores, ships, ship_to_type, type_scores_test, types, 0.5), ships)),
                        ("soft_b1.0", top5(soft_boost(val_scores, ships, ship_to_type, type_scores_val, types, 1.0), ships), top5(soft_boost(test_scores, ships, ship_to_type, type_scores_test, types, 1.0), ships)),
                    ]
                    for gate_name, pred_val, pred_test in variants:
                        sub_path = args.out_dir / f"{mode}_{method}_{str(topk).replace('.', 'p')}_sb{str(size_beta).replace('.', 'p')}_{gate_name}.csv"
                        make_submission(sub_path, sample, test_names, pred_test)
                        diag = score_submission(sub_path, args.data_dir)
                        rows.append(
                            {
                                "strategy": "subclass_type",
                                "time_mode": mode,
                                "subclass_method": method,
                                "subclass_topk": topk,
                                "size_beta": size_beta,
                                "type_gate": gate_name,
                                "type_top1_acc": type_top1,
                                "type_top2_acc": type_top2,
                                **metric(y, pred_val),
                                "diagnostic_R@1": diag["test_R@1"],
                                "diagnostic_R@3": diag["test_R@3"],
                                "diagnostic_R@5": diag["test_R@5"],
                                "diagnostic_Score": diag["test_Score"],
                                "submission": str(sub_path),
                            }
                        )

    df = pd.DataFrame(rows).sort_values(["Score", "R@1", "diagnostic_Score"], ascending=False)
    leaderboard = args.out_dir / "subclass_type_gated_leaderboard.csv"
    df.to_csv(leaderboard, index=False)
    best = df.iloc[0]
    best_alias = args.out_dir / "submission_task2_best_subclass_type_gated.csv"
    best_alias.write_bytes(Path(best.submission).read_bytes())

    lines = ["# Task 2 Subclass + Ship Type Gated Retrieval", ""]
    lines.append("gallery timestamp subclass와 ship_type gate를 결합한 후처리 실험이다.")
    lines.append("")
    lines.append("## 기준 모델")
    lines.append("")
    lines.append(f"- run_dir: `{args.run_dir}`")
    lines.append(f"- epoch: `{args.epoch}`")
    lines.append(f"- checkpoint: `{ckpt_path}`")
    lines.append("")
    lines.append("## Best By Validation")
    lines.append("")
    lines.append(f"- strategy: `{best.strategy}`")
    lines.append(f"- time_mode: `{best.time_mode}`")
    lines.append(f"- subclass_method: `{best.subclass_method}`")
    lines.append(f"- subclass_topk: `{best.subclass_topk}`")
    lines.append(f"- size_beta: `{best.size_beta}`")
    lines.append(f"- type_gate: `{best.type_gate}`")
    lines.append(f"- validation Score: `{best.Score:.6f}`")
    lines.append(f"- diagnostic label Score: `{best.diagnostic_Score:.6f}`")
    lines.append(f"- submission: `{best.submission}`")
    lines.append(f"- alias: `{best_alias}`")
    lines.append("")
    lines.append("## Top Results")
    lines.append("")
    cols = [
        "strategy",
        "time_mode",
        "subclass_method",
        "subclass_topk",
        "size_beta",
        "type_gate",
        "R@1",
        "R@3",
        "R@5",
        "Score",
        "diagnostic_Score",
    ]
    lines.append(md_table(df.head(30), cols))
    lines.append("")
    lines.append("## 해석")
    lines.append("")
    lines.append("- exact/date/hour/10min timestamp grouping은 이 데이터에서 같은 그룹 수를 만들었다. 즉 timestamp가 날짜 단위로 반복되는 구조다.")
    lines.append("- subclass는 같은 ship_id 내부의 gallery timestamp 그룹을 별도 prototype으로 만든다.")
    lines.append("- ship_type gate는 subclass score가 ship_id score로 합쳐진 뒤 적용한다.")
    lines.append("- validation 기준으로 baseline type-gated보다 개선되는지 보고, 개선될 때만 active submission으로 승격한다.")
    args.report.write_text("\n".join(lines), encoding="utf-8")

    with Path("analysis/task2_execution_log.md").open("a", encoding="utf-8") as f:
        f.write(f"""
## Subclass + Ship Type Gated Retrieval 실험

gallery timestamp subclass와 ship_type gate를 결합한 후처리 실험을 수행했다.

- report: `{args.report}`
- leaderboard: `{leaderboard}`
- best alias: `{best_alias}`

Best by validation:

- strategy: `{best.strategy}`
- time_mode: `{best.time_mode}`
- subclass_method: `{best.subclass_method}`
- subclass_topk: `{best.subclass_topk}`
- size_beta: `{best.size_beta}`
- type_gate: `{best.type_gate}`
- validation Score: `{best.Score:.6f}`
- diagnostic label Score: `{best.diagnostic_Score:.6f}`
- submission: `{best.submission}`

""")
    print(df.head(30).to_string(index=False))
    print("\nBEST", best.strategy, best.Score, best.submission)


if __name__ == "__main__":
    main()
