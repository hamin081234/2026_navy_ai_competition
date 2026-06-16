#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

import task2_train_cached_retrieval as tr
from task2_run_robust_local_sweep import score_submission


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=Path, default=Path("outputs/task2_runs/checkpoint_rows100_seed777_ep8_aug"))
    p.add_argument("--epoch", type=int, default=2)
    p.add_argument("--data-dir", type=Path, default=Path("database"))
    p.add_argument("--cache-dir", type=Path, default=Path("outputs/task2_cache/logmel96_f16000"))
    p.add_argument("--out-dir", type=Path, default=Path("outputs/task2_runs/type_gated_retrieval"))
    p.add_argument("--report", type=Path, default=Path("analysis/task2_type_gated_retrieval.md"))
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=0)
    return p.parse_args()


def l2(x):
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-8, None)


def metric(y, pred):
    r1 = float(np.mean([yy in row[:1] for yy, row in zip(y, pred)]))
    r3 = float(np.mean([yy in row[:3] for yy, row in zip(y, pred)]))
    r5 = float(np.mean([yy in row[:5] for yy, row in zip(y, pred)]))
    return {"R@1": r1, "R@3": r3, "R@5": r5, "Score": 0.5 * r1 + 0.3 * r3 + 0.2 * r5}


def top5(scores, ships):
    return ships[np.argsort(-scores, axis=1)[:, :5]]


def type_score_matrix(query_emb, ref_emb, ref_types, method, topk=20):
    query_emb, ref_emb = l2(query_emb), l2(ref_emb)
    types = np.array(sorted(set(ref_types.tolist())), dtype=object)
    sim = query_emb @ ref_emb.T
    out = np.zeros((len(query_emb), len(types)), dtype=np.float32)
    for j, typ in enumerate(types):
        cols = np.where(ref_types == typ)[0]
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
        else:
            raise ValueError(method)
    return out, types


def type_accuracy(true_types, type_scores, types, k=1):
    pred = types[np.argsort(-type_scores, axis=1)[:, :k]]
    return float(np.mean([tt in row for tt, row in zip(true_types, pred)]))


def hard_gate(ship_scores, ships, ship_to_type, type_scores, types, keep_top_types):
    adjusted = ship_scores.copy()
    ranked_types = types[np.argsort(-type_scores, axis=1)[:, :keep_top_types]]
    for i in range(adjusted.shape[0]):
        allowed = set(ranked_types[i].tolist())
        for j, sid in enumerate(ships):
            if ship_to_type[int(sid)] not in allowed:
                adjusted[i, j] = -1e9
    return adjusted


def hard_gate_with_fill(ship_scores, ships, ship_to_type, type_scores, types, keep_top_types):
    gated = hard_gate(ship_scores, ships, ship_to_type, type_scores, types, keep_top_types)
    pred = top5(gated, ships)
    # All four ship_type buckets have enough ships here, but keep a fallback for robustness.
    for i in range(len(pred)):
        if len(set(pred[i].tolist())) < 5:
            base = top5(ship_scores[i : i + 1], ships)[0]
            seen = set(pred[i].tolist())
            filled = list(pred[i])
            for sid in base:
                if int(sid) not in seen:
                    filled.append(int(sid))
                    seen.add(int(sid))
                if len(filled) == 5:
                    break
            pred[i] = np.array(filled[:5], dtype=pred.dtype)
    return pred


def soft_boost(ship_scores, ships, ship_to_type, type_scores, types, beta):
    type_index = {typ: j for j, typ in enumerate(types)}
    adjusted = ship_scores.copy()
    for j, sid in enumerate(ships):
        adjusted[:, j] += beta * type_scores[:, type_index[ship_to_type[int(sid)]]]
    return adjusted


def make_submission(path, sample, test_names, pred):
    sub = sample.copy()
    pred_map = dict(zip(test_names, [",".join(map(str, row)) for row in pred]))
    sub["top5_ship_ids"] = sub["filename"].map(pred_map)
    sub.to_csv(path, index=False)
    return path


def md_table(df, cols):
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, r in df.iterrows():
        vals = []
        for c in cols:
            v = r[c]
            if c in {"R@1", "R@3", "R@5", "Score", "diagnostic_R@1", "diagnostic_R@3", "diagnostic_R@5", "diagnostic_Score", "type_top1_acc", "type_top2_acc"}:
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
    train, gallery, val, test, label_map = tr.load_cached_tables(model_args)
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
    val_sample = val[["filename"]].copy()
    val_names = val["filename"].astype(str).tolist()
    rows = []
    submissions = {}

    type_settings = []
    for type_method in ["prototype", "max", "topk"]:
        for type_topk in ([20] if type_method != "topk" else [10, 20, 40]):
            type_scores_val, types = type_score_matrix(val_emb, ref_emb, ref_types, type_method, topk=type_topk)
            type_scores_test, _ = type_score_matrix(test_emb, ref_emb, ref_types, type_method, topk=type_topk)
            type_settings.append((type_method, type_topk, type_scores_val, type_scores_test, types))

    for ship_method in ["prototype", "max", "topk"]:
        ship_topks = [None] if ship_method != "topk" else [10, 15, 20]
        for ship_topk in ship_topks:
            if ship_method == "topk":
                ship_scores_val, ships = tr.score_matrix(val_emb, ref_emb, ref_ship_ids, ship_method, topk=ship_topk, alpha=0)
                ship_scores_test, _ = tr.score_matrix(test_emb, ref_emb, ref_ship_ids, ship_method, topk=ship_topk, alpha=0)
            else:
                ship_scores_val, ships = tr.score_matrix(val_emb, ref_emb, ref_ship_ids, ship_method, alpha=0)
                ship_scores_test, _ = tr.score_matrix(test_emb, ref_emb, ref_ship_ids, ship_method, alpha=0)

            pred = top5(ship_scores_val, ships)
            row = {
                "strategy": "baseline",
                "ship_method": ship_method,
                "ship_topk": ship_topk,
                "type_method": "",
                "type_topk": "",
                "gate": "",
                "beta": "",
                "type_top1_acc": np.nan,
                "type_top2_acc": np.nan,
                **metric(y, pred),
            }
            sub_path = args.out_dir / f"baseline_{ship_method}_{ship_topk or 'na'}.csv"
            val_path = args.out_dir / f"validation_baseline_{ship_method}_{ship_topk or 'na'}.csv"
            make_submission(val_path, val_sample, val_names, pred)
            make_submission(sub_path, sample, test_names, top5(ship_scores_test, ships))
            diag = score_submission(sub_path, args.data_dir)
            row.update({
                "diagnostic_R@1": diag["test_R@1"],
                "diagnostic_R@3": diag["test_R@3"],
                "diagnostic_R@5": diag["test_R@5"],
                "diagnostic_Score": diag["test_Score"],
                "validation_submission": str(val_path),
                "submission": str(sub_path),
            })
            rows.append(row)
            submissions[str(sub_path)] = sub_path

            for type_method, type_topk, type_scores_val, type_scores_test, types in type_settings:
                type_top1 = type_accuracy(y_types, type_scores_val, types, k=1)
                type_top2 = type_accuracy(y_types, type_scores_val, types, k=2)

                for keep in [1, 2]:
                    pred_val = hard_gate_with_fill(ship_scores_val, ships, ship_to_type, type_scores_val, types, keep)
                    pred_test = hard_gate_with_fill(ship_scores_test, ships, ship_to_type, type_scores_test, types, keep)
                    row = {
                        "strategy": "hard_gate",
                        "ship_method": ship_method,
                        "ship_topk": ship_topk,
                        "type_method": type_method,
                        "type_topk": type_topk,
                        "gate": f"top{keep}_type",
                        "beta": "",
                        "type_top1_acc": type_top1,
                        "type_top2_acc": type_top2,
                        **metric(y, pred_val),
                    }
                    sub_path = args.out_dir / f"hard_{ship_method}_{ship_topk or 'na'}_{type_method}_{type_topk}_top{keep}.csv"
                    val_path = args.out_dir / f"validation_hard_{ship_method}_{ship_topk or 'na'}_{type_method}_{type_topk}_top{keep}.csv"
                    make_submission(val_path, val_sample, val_names, pred_val)
                    make_submission(sub_path, sample, test_names, pred_test)
                    diag = score_submission(sub_path, args.data_dir)
                    row.update({
                        "diagnostic_R@1": diag["test_R@1"],
                        "diagnostic_R@3": diag["test_R@3"],
                        "diagnostic_R@5": diag["test_R@5"],
                        "diagnostic_Score": diag["test_Score"],
                        "validation_submission": str(val_path),
                        "submission": str(sub_path),
                    })
                    rows.append(row)

                for beta in [0.1, 0.2, 0.35, 0.5, 0.75, 1.0]:
                    pred_val = top5(soft_boost(ship_scores_val, ships, ship_to_type, type_scores_val, types, beta), ships)
                    pred_test = top5(soft_boost(ship_scores_test, ships, ship_to_type, type_scores_test, types, beta), ships)
                    row = {
                        "strategy": "soft_boost",
                        "ship_method": ship_method,
                        "ship_topk": ship_topk,
                        "type_method": type_method,
                        "type_topk": type_topk,
                        "gate": "",
                        "beta": beta,
                        "type_top1_acc": type_top1,
                        "type_top2_acc": type_top2,
                        **metric(y, pred_val),
                    }
                    sub_path = args.out_dir / f"soft_{ship_method}_{ship_topk or 'na'}_{type_method}_{type_topk}_b{str(beta).replace('.', 'p')}.csv"
                    val_path = args.out_dir / f"validation_soft_{ship_method}_{ship_topk or 'na'}_{type_method}_{type_topk}_b{str(beta).replace('.', 'p')}.csv"
                    make_submission(val_path, val_sample, val_names, pred_val)
                    make_submission(sub_path, sample, test_names, pred_test)
                    diag = score_submission(sub_path, args.data_dir)
                    row.update({
                        "diagnostic_R@1": diag["test_R@1"],
                        "diagnostic_R@3": diag["test_R@3"],
                        "diagnostic_R@5": diag["test_R@5"],
                        "diagnostic_Score": diag["test_Score"],
                        "validation_submission": str(val_path),
                        "submission": str(sub_path),
                    })
                    rows.append(row)

    df = pd.DataFrame(rows)
    df = df.sort_values(["Score", "R@1", "diagnostic_Score"], ascending=False)
    leaderboard = args.out_dir / "type_gated_leaderboard.csv"
    df.to_csv(leaderboard, index=False)
    best = df.iloc[0]
    best_alias = args.out_dir / "submission_task2_best_type_gated.csv"
    best_alias.write_bytes(Path(best.submission).read_bytes())

    lines = ["# Task 2 Ship Type Gated Retrieval", ""]
    lines.append("ship_type을 먼저 추정한 뒤 ship_id 후보를 고르는 방식이 도움이 되는지 확인한 실험이다.")
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
    lines.append(f"- ship_method: `{best.ship_method}`")
    lines.append(f"- ship_topk: `{best.ship_topk}`")
    lines.append(f"- type_method: `{best.type_method}`")
    lines.append(f"- type_topk: `{best.type_topk}`")
    lines.append(f"- gate: `{best.gate}`")
    lines.append(f"- beta: `{best.beta}`")
    lines.append(f"- validation Score: `{best.Score:.6f}`")
    lines.append(f"- diagnostic label Score: `{best.diagnostic_Score:.6f}`")
    lines.append(f"- type top1 acc: `{best.type_top1_acc:.6f}`")
    lines.append(f"- type top2 acc: `{best.type_top2_acc:.6f}`")
    lines.append(f"- submission: `{best.submission}`")
    lines.append(f"- alias: `{best_alias}`")
    lines.append("")
    lines.append("## Top Results")
    lines.append("")
    cols = ["strategy", "ship_method", "ship_topk", "type_method", "type_topk", "gate", "beta", "type_top1_acc", "type_top2_acc", "R@1", "R@3", "R@5", "Score", "diagnostic_Score"]
    lines.append(md_table(df.head(25), cols))
    lines.append("")
    lines.append("## 해석")
    lines.append("")
    lines.append("- hard gating은 ship_type 예측이 틀리면 정답 ship_id가 후보군에서 빠지므로 위험하다.")
    lines.append("- soft boost는 ship_type 정보를 보조 신호로만 쓰기 때문에, 타입 예측이 애매한 경우에도 ship_id retrieval이 복구할 여지가 있다.")
    lines.append("- 이 실험에서 validation 기준 개선이 확인되면, 다음 단계는 ship_id와 ship_type을 함께 학습하는 multi-task model이다.")
    args.report.write_text("\n".join(lines), encoding="utf-8")

    with Path("analysis/task2_execution_log.md").open("a", encoding="utf-8") as f:
        f.write(f"""
## Ship Type Gated Retrieval 실험

ship_type을 먼저 추정한 뒤 ship_id retrieval에 반영하는 후처리 실험을 수행했다.

- report: `{args.report}`
- leaderboard: `{leaderboard}`
- best alias: `{best_alias}`

Best by validation:

- strategy: `{best.strategy}`
- ship_method: `{best.ship_method}`
- ship_topk: `{best.ship_topk}`
- type_method: `{best.type_method}`
- type_topk: `{best.type_topk}`
- validation Score: `{best.Score:.6f}`
- diagnostic label Score: `{best.diagnostic_Score:.6f}`
- submission: `{best.submission}`

""")
    print(df.head(25).to_string(index=False))
    print("\nBEST", best.strategy, best.Score, best.submission)


if __name__ == "__main__":
    main()
