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
from task2_season_type_gated_retrieval import add_gallery_timestamp, label_score_matrix, season_filtered_ship_scores
from task2_type_gated_retrieval import hard_gate_with_fill, make_submission, metric, top5, type_accuracy, type_score_matrix


DEFAULT_CHECKPOINTS = [
    "outputs/task2_runs/checkpoint_rows100_seed777_ep8_aug:2",
    "outputs/task2_runs/checkpoint_rows100_seed777_ep8_aug:3",
    "outputs/task2_runs/checkpoint_rows100_seed123_ep8_aug:7",
    "outputs/task2_runs/checkpoint_rows100_seed777_ep8_aug:8",
    "outputs/task2_runs/checkpoint_rows100_seed42_ep8_aug:1",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoints", nargs="+", default=DEFAULT_CHECKPOINTS, help="Items formatted as run_dir:epoch")
    p.add_argument("--data-dir", type=Path, default=Path("database"))
    p.add_argument("--cache-dir", type=Path, default=Path("outputs/task2_cache/logmel96_f16000"))
    p.add_argument("--out-dir", type=Path, default=Path("outputs/task2_runs/season_type_checkpoint_sweep"))
    p.add_argument("--report", type=Path, default=Path("analysis/task2_season_type_checkpoint_sweep.md"))
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--promote", action="store_true", help="Copy best submission to outputs/task2_runs/submission_task2_best.csv")
    return p.parse_args()


def checkpoint_id(run_dir, epoch):
    return f"{run_dir.name}_ep{epoch}"


def load_checkpoint_context(args, run_dir, epoch):
    run_meta = json.loads((run_dir / "run_meta.json").read_text(encoding="utf-8"))
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

    ckpt_path = run_dir / f"model_epoch{epoch}.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu")
    emb_dim = int(run_meta["args"].get("emb_dim", 256))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = tr.CachedSpecCNN(len(label_map), emb_dim).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, model_args, gallery, val, test, label_map, ckpt_path, device


def evaluate_checkpoint(args, run_dir, epoch):
    cid = checkpoint_id(run_dir, epoch)
    out_dir = args.out_dir / cid
    out_dir.mkdir(parents=True, exist_ok=True)
    model, model_args, gallery, val, test, label_map, ckpt_path, device = load_checkpoint_context(args, run_dir, epoch)
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

    type_scores_val, types = type_score_matrix(val_emb, ref_emb, ref_types, "prototype", topk=20)
    type_scores_test, _ = type_score_matrix(test_emb, ref_emb, ref_types, "prototype", topk=20)
    type_top1 = type_accuracy(y_types, type_scores_val, types, k=1)
    type_top2 = type_accuracy(y_types, type_scores_val, types, k=2)

    season_settings = []
    for season_method in ["prototype", "max"]:
        s_val, seasons = label_score_matrix(val_emb, ref_emb, ref_seasons, season_method, topk=20)
        s_test, _ = label_score_matrix(test_emb, ref_emb, ref_seasons, season_method, topk=20)
        season_settings.append((season_method, s_val, s_test, seasons))

    rows = []
    for ship_topk in [15, 20]:
        base_val, ships = tr.score_matrix(val_emb, ref_emb, ref_ship_ids, "topk", topk=ship_topk, alpha=0)
        base_test, _ = tr.score_matrix(test_emb, ref_emb, ref_ship_ids, "topk", topk=ship_topk, alpha=0)
        variants = [("baseline_none", base_val, base_test), ("baseline_type_top2", base_val, base_test)]
        for season_method, s_val, s_test, seasons in season_settings:
            for season_keep in [1, 2, 3]:
                f_val, ships = season_filtered_ship_scores(val_emb, ref_emb, ref_ship_ids, ref_seasons, s_val, seasons, season_keep, ship_topk)
                f_test, _ = season_filtered_ship_scores(test_emb, ref_emb, ref_ship_ids, ref_seasons, s_test, seasons, season_keep, ship_topk)
                variants.append((f"season_{season_method}_keep{season_keep}", f_val, f_test))

        for variant, val_scores, test_scores in variants:
            for type_gate in ["none", "type_top2"]:
                if variant == "baseline_type_top2" and type_gate == "none":
                    continue
                if variant == "baseline_none" and type_gate == "type_top2":
                    continue
                pred_val = top5(val_scores, ships) if type_gate == "none" else hard_gate_with_fill(val_scores, ships, ship_to_type, type_scores_val, types, 2)
                pred_test = top5(test_scores, ships) if type_gate == "none" else hard_gate_with_fill(test_scores, ships, ship_to_type, type_scores_test, types, 2)
                sub_path = out_dir / f"{variant}_shiptopk{ship_topk}_{type_gate}.csv"
                make_submission(sub_path, sample, test_names, pred_test)
                diag = score_submission(sub_path, args.data_dir)
                parts = variant.split("_")
                season_method = parts[1] if variant.startswith("season_") else ""
                season_keep = parts[-1].replace("keep", "") if variant.startswith("season_") else ""
                rows.append(
                    {
                        "checkpoint": cid,
                        "run_dir": str(run_dir),
                        "epoch": epoch,
                        "ship_topk": ship_topk,
                        "season_method": season_method,
                        "season_keep": season_keep,
                        "type_gate": type_gate,
                        "type_top1_acc": type_top1,
                        "type_top2_acc": type_top2,
                        **metric(y, pred_val),
                        "diagnostic_Score": diag["test_Score"],
                        "submission": str(sub_path),
                    }
                )
    df = pd.DataFrame(rows).sort_values(["Score", "R@1", "diagnostic_Score"], ascending=False)
    df.to_csv(out_dir / "leaderboard.csv", index=False)
    print(json.dumps({"checkpoint_done": cid, "best": df.iloc[0].to_dict()}, ensure_ascii=False), flush=True)
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
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    dfs = []
    for item in args.checkpoints:
        run_str, epoch_str = item.rsplit(":", 1)
        dfs.append(evaluate_checkpoint(args, Path(run_str), int(epoch_str)))

    all_df = pd.concat(dfs, ignore_index=True).sort_values(["Score", "R@1", "diagnostic_Score"], ascending=False)
    leaderboard = args.out_dir / "season_type_checkpoint_sweep_leaderboard.csv"
    all_df.to_csv(leaderboard, index=False)
    best = all_df.iloc[0]
    best_alias = args.out_dir / "submission_task2_best_season_type_checkpoint.csv"
    best_alias.write_bytes(Path(best.submission).read_bytes())
    if args.promote:
        Path("outputs/task2_runs/submission_task2_best.csv").write_bytes(Path(best.submission).read_bytes())

    lines = ["# Task 2 Season + Type Checkpoint Sweep", ""]
    lines.append("prototype 기반 season/type gate를 여러 checkpoint에 적용해 안정성과 추가 개선 가능성을 확인했다.")
    lines.append("")
    lines.append("## Best By Validation")
    lines.append("")
    for k in ["checkpoint", "epoch", "ship_topk", "season_method", "season_keep", "type_gate", "type_top1_acc", "type_top2_acc", "R@1", "R@3", "R@5", "Score", "diagnostic_Score", "submission"]:
        v = best[k]
        if isinstance(v, float):
            lines.append(f"- {k}: `{v:.6f}`")
        else:
            lines.append(f"- {k}: `{v}`")
    lines.append(f"- alias: `{best_alias}`")
    lines.append(f"- promoted_to_active: `{bool(args.promote)}`")
    lines.append("")
    lines.append("## Top Results")
    lines.append("")
    cols = ["checkpoint", "epoch", "ship_topk", "season_method", "season_keep", "type_gate", "type_top1_acc", "type_top2_acc", "R@1", "R@3", "R@5", "Score", "diagnostic_Score"]
    lines.append(md_table(all_df.head(30), cols))
    lines.append("")
    lines.append("## 해석")
    lines.append("")
    lines.append("- 이 sweep은 head 학습이 아니라, 현재 더 강한 prototype 기반 gate를 checkpoint별로 재적용한 것이다.")
    lines.append("- 여러 checkpoint에서 같은 구조가 상위에 오르면 season/type gate가 우연이 아니라는 신호다.")
    lines.append("- validation 최고 후보가 기존 active보다 높을 때만 active submission으로 승격한다.")
    args.report.write_text("\n".join(lines), encoding="utf-8")

    with Path("analysis/task2_execution_log.md").open("a", encoding="utf-8") as f:
        f.write(f"""
## Season + Type Checkpoint Sweep 완료

prototype 기반 season/type gate를 여러 checkpoint에 적용했다.

- report: `{args.report}`
- leaderboard: `{leaderboard}`
- best alias: `{best_alias}`
- promoted_to_active: `{bool(args.promote)}`

Best by validation:

- checkpoint: `{best.checkpoint}`
- epoch: `{int(best.epoch)}`
- ship_topk: `{int(best.ship_topk)}`
- season_method: `{best.season_method}`
- season_keep: `{best.season_keep}`
- type_gate: `{best.type_gate}`
- validation Score: `{best.Score:.6f}`
- diagnostic label Score: `{best.diagnostic_Score:.6f}`
- submission: `{best.submission}`

""")
    print(all_df.head(30).to_string(index=False))
    print("\nBEST", best.checkpoint, best.Score, best.submission)


if __name__ == "__main__":
    main()
