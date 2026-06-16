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
    p.add_argument("--out-dir", type=Path, default=Path("outputs/task2_runs/season_type_gated_retrieval"))
    p.add_argument("--report", type=Path, default=Path("analysis/task2_season_type_gated_retrieval.md"))
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=0)
    return p.parse_args()


def l2(x):
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-8, None)


def season_from_month(month):
    if month in (12, 1, 2):
        return "winter"
    if month in (3, 4, 5):
        return "spring"
    if month in (6, 7, 8):
        return "summer"
    return "fall"


def add_gallery_timestamp(gallery, data_dir):
    if "ais_timestamp" in gallery.columns:
        out = gallery.copy()
    else:
        raw = pd.read_csv(data_dir / "task2_test/gallery.csv")[["filename", "ais_timestamp"]]
        out = gallery.merge(raw, on="filename", how="left", validate="one_to_one")
    ts = pd.to_datetime(out["ais_timestamp"], utc=True, format="mixed")
    out["season"] = ts.dt.month.map(season_from_month)
    return out


def label_score_matrix(query_emb, ref_emb, ref_labels, method="prototype", topk=20):
    query_emb, ref_emb = l2(query_emb), l2(ref_emb)
    labels = np.array(sorted(set(ref_labels.tolist())), dtype=object)
    sim = query_emb @ ref_emb.T
    out = np.zeros((len(query_emb), len(labels)), dtype=np.float32)
    for j, label in enumerate(labels):
        cols = np.where(ref_labels == label)[0]
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
    return out, labels


def season_filtered_ship_scores(query_emb, ref_emb, ref_ship_ids, ref_seasons, season_scores, seasons, season_keep, ship_topk):
    query_emb, ref_emb = l2(query_emb), l2(ref_emb)
    ships = np.array(sorted(set(ref_ship_ids.tolist())), dtype=int)
    sim = query_emb @ ref_emb.T
    out = np.zeros((len(query_emb), len(ships)), dtype=np.float32)
    season_rank = seasons[np.argsort(-season_scores, axis=1)[:, :season_keep]]
    for i in range(len(query_emb)):
        allowed = set(season_rank[i].tolist())
        season_mask = np.array([s in allowed for s in ref_seasons], dtype=bool)
        for j, sid in enumerate(ships):
            cols = np.where((ref_ship_ids == sid) & season_mask)[0]
            if len(cols) == 0:
                cols = np.where(ref_ship_ids == sid)[0]
            vals = sim[i, cols]
            k = min(ship_topk, len(vals))
            out[i, j] = np.sort(vals)[-k:].mean()
    return out, ships


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
    gallery = add_gallery_timestamp(gallery, args.data_dir)

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
    ref_seasons = gallery["season"].astype(str).to_numpy()
    y = val["ship_id"].astype(int).to_numpy()
    y_types = val["ship_type"].astype(str).to_numpy()
    ship_to_type = gallery.groupby("ship_id")["ship_type"].first().astype(str).to_dict()
    sample = pd.read_csv(args.data_dir / "sample_submission_task2.csv")
    val_sample = val[["filename"]].copy()
    val_names = val["filename"].astype(str).tolist()

    type_scores_val, types = type_score_matrix(val_emb, ref_emb, ref_types, "prototype", topk=20)
    type_scores_test, _ = type_score_matrix(test_emb, ref_emb, ref_types, "prototype", topk=20)
    type_top1 = type_accuracy(y_types, type_scores_val, types, k=1)
    type_top2 = type_accuracy(y_types, type_scores_val, types, k=2)

    season_settings = []
    for method in ["prototype", "max", "topk"]:
        topks = [20] if method != "topk" else [10, 20, 40]
        for topk in topks:
            s_val, seasons = label_score_matrix(val_emb, ref_emb, ref_seasons, method, topk=topk)
            s_test, _ = label_score_matrix(test_emb, ref_emb, ref_seasons, method, topk=topk)
            season_settings.append((method, topk, s_val, s_test, seasons))

    rows = []
    for ship_topk in [10, 15, 20]:
        base_val, ships = tr.score_matrix(val_emb, ref_emb, ref_ship_ids, "topk", topk=ship_topk, alpha=0)
        base_test, _ = tr.score_matrix(test_emb, ref_emb, ref_ship_ids, "topk", topk=ship_topk, alpha=0)
        for type_gate in ["none", "type_top2"]:
            pred_val = top5(base_val, ships) if type_gate == "none" else hard_gate_with_fill(base_val, ships, ship_to_type, type_scores_val, types, 2)
            pred_test = top5(base_test, ships) if type_gate == "none" else hard_gate_with_fill(base_test, ships, ship_to_type, type_scores_test, types, 2)
            sub_path = args.out_dir / f"baseline_shiptopk{ship_topk}_{type_gate}.csv"
            val_path = args.out_dir / f"validation_baseline_shiptopk{ship_topk}_{type_gate}.csv"
            make_submission(val_path, val_sample, val_names, pred_val)
            make_submission(sub_path, sample, test_names, pred_test)
            diag = score_submission(sub_path, args.data_dir)
            rows.append(
                {
                    "strategy": "baseline",
                    "ship_topk": ship_topk,
                    "season_method": "",
                    "season_topk": "",
                    "season_keep": "",
                    "type_gate": type_gate,
                    "type_top1_acc": type_top1,
                    "type_top2_acc": type_top2,
                    **metric(y, pred_val),
                    "diagnostic_Score": diag["test_Score"],
                    "validation_submission": str(val_path),
                    "submission": str(sub_path),
                }
            )

        for season_method, season_topk, s_val, s_test, seasons in season_settings:
            for season_keep in [1, 2, 3]:
                filt_val, ships = season_filtered_ship_scores(val_emb, ref_emb, ref_ship_ids, ref_seasons, s_val, seasons, season_keep, ship_topk)
                filt_test, _ = season_filtered_ship_scores(test_emb, ref_emb, ref_ship_ids, ref_seasons, s_test, seasons, season_keep, ship_topk)
                for type_gate in ["none", "type_top2"]:
                    pred_val = top5(filt_val, ships) if type_gate == "none" else hard_gate_with_fill(filt_val, ships, ship_to_type, type_scores_val, types, 2)
                    pred_test = top5(filt_test, ships) if type_gate == "none" else hard_gate_with_fill(filt_test, ships, ship_to_type, type_scores_test, types, 2)
                    sub_path = args.out_dir / f"season_{season_method}{season_topk}_keep{season_keep}_shiptopk{ship_topk}_{type_gate}.csv"
                    val_path = args.out_dir / f"validation_season_{season_method}{season_topk}_keep{season_keep}_shiptopk{ship_topk}_{type_gate}.csv"
                    make_submission(val_path, val_sample, val_names, pred_val)
                    make_submission(sub_path, sample, test_names, pred_test)
                    diag = score_submission(sub_path, args.data_dir)
                    rows.append(
                        {
                            "strategy": "season_filter",
                            "ship_topk": ship_topk,
                            "season_method": season_method,
                            "season_topk": season_topk,
                            "season_keep": season_keep,
                            "type_gate": type_gate,
                            "type_top1_acc": type_top1,
                            "type_top2_acc": type_top2,
                            **metric(y, pred_val),
                            "diagnostic_Score": diag["test_Score"],
                            "validation_submission": str(val_path),
                            "submission": str(sub_path),
                        }
                    )

    df = pd.DataFrame(rows).sort_values(["Score", "R@1", "diagnostic_Score"], ascending=False)
    leaderboard = args.out_dir / "season_type_gated_leaderboard.csv"
    df.to_csv(leaderboard, index=False)
    best = df.iloc[0]
    best_alias = args.out_dir / "submission_task2_best_season_type_gated.csv"
    best_val_alias = args.out_dir / "validation_task2_best_season_type_gated.csv"
    best_alias.write_bytes(Path(best.submission).read_bytes())
    best_val_alias.write_bytes(Path(best.validation_submission).read_bytes())

    lines = ["# Task 2 Season + Ship Type Gated Retrieval", ""]
    lines.append("gallery timestamp를 season label로 바꿔, season 기반 후보 제한이 ship_type gate와 함께 도움이 되는지 확인했다.")
    lines.append("")
    lines.append("## Best By Validation")
    lines.append("")
    lines.append(f"- strategy: `{best.strategy}`")
    lines.append(f"- ship_topk: `{best.ship_topk}`")
    lines.append(f"- season_method: `{best.season_method}`")
    lines.append(f"- season_topk: `{best.season_topk}`")
    lines.append(f"- season_keep: `{best.season_keep}`")
    lines.append(f"- type_gate: `{best.type_gate}`")
    lines.append(f"- validation Score: `{best.Score:.6f}`")
    lines.append(f"- diagnostic label Score: `{best.diagnostic_Score:.6f}`")
    lines.append(f"- validation submission: `{best.validation_submission}`")
    lines.append(f"- submission: `{best.submission}`")
    lines.append(f"- alias: `{best_alias}`")
    lines.append(f"- validation alias: `{best_val_alias}`")
    lines.append("")
    lines.append("## Top Results")
    lines.append("")
    cols = ["strategy", "ship_topk", "season_method", "season_topk", "season_keep", "type_gate", "R@1", "R@3", "R@5", "Score", "diagnostic_Score"]
    lines.append(md_table(df.head(25), cols))
    lines.append("")
    lines.append("## 해석")
    lines.append("")
    lines.append("- query에는 timestamp가 없으므로 season은 입력값이 아니라 오디오 기반 예측/유사도 신호로만 쓸 수 있다.")
    lines.append("- 이 후처리에서 baseline을 넘으면 season auxiliary training을 진행할 근거가 생긴다.")
    lines.append("- baseline을 넘지 못하면 season은 ship_id retrieval에 직접 gate로 쓰기보다 학습 regularizer로만 쓰는 편이 낫다.")
    args.report.write_text("\n".join(lines), encoding="utf-8")

    with Path("analysis/task2_execution_log.md").open("a", encoding="utf-8") as f:
        f.write(f"""
## Season + Ship Type Gated Retrieval 실험

gallery timestamp를 season label로 변환해서 season 기반 후보 제한을 실험했다.

- report: `{args.report}`
- leaderboard: `{leaderboard}`
- best alias: `{best_alias}`

Best by validation:

- strategy: `{best.strategy}`
- ship_topk: `{best.ship_topk}`
- season_method: `{best.season_method}`
- season_topk: `{best.season_topk}`
- season_keep: `{best.season_keep}`
- type_gate: `{best.type_gate}`
- validation Score: `{best.Score:.6f}`
- diagnostic label Score: `{best.diagnostic_Score:.6f}`
- submission: `{best.submission}`

""")
    print(df.head(25).to_string(index=False))
    print("\nBEST", best.strategy, best.Score, best.submission)


if __name__ == "__main__":
    main()
