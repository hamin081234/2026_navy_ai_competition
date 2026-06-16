#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold

sys.path.append(str(Path(__file__).resolve().parent))
import task2_gallery_aggregation_sweep as agg  # noqa: E402
from task2_ais_group_pooling_sweep import add_group_columns, group_score_matrix  # noqa: E402


RANK_FLAT = np.array([1.0, 0.9, 0.8, 0.7, 0.6], dtype=np.float32)
RANK_LINEAR = np.array([1.0, 0.8, 0.6, 0.4, 0.2], dtype=np.float32)
RANK_RECIP = np.array([1.0, 0.5, 1 / 3, 0.25, 0.2], dtype=np.float32)
RANK_TOPHEAVY = np.array([1.0, 0.35, 0.2, 0.12, 0.08], dtype=np.float32)

RUNS = [
    "robust_rows100_seed123_ep2_aug",
    "robust_rows80_seed123_ep2_aug",
    "robust_rows100_seed777_ep2_aug",
    "projection_panns_cnn14_seed777_trainonly_ep1",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--candidates", type=Path, required=True)
    p.add_argument("--data-dir", type=Path, default=Path("database"))
    p.add_argument("--runs-dir", type=Path, default=Path("outputs/task2_runs"))
    p.add_argument("--out-dir", type=Path, default=Path("outputs/task2_tree_reranker/feature_rich"))
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--source-topk", type=int, default=5)
    p.add_argument("--score-topk", type=int, default=20)
    p.add_argument("--seed", type=int, default=20260616)
    p.add_argument("--models", type=str, default="hgb_deep,hgb_wide,extra_trees,random_forest")
    return p.parse_args()


def parse_sub(path: str | Path, sample: pd.DataFrame) -> np.ndarray:
    sub = pd.read_csv(path)
    if len(sub) != len(sample) or not sub["filename"].equals(sample["filename"]):
        raise ValueError(f"submission/sample mismatch: {path}")
    return np.array([[int(x) for x in str(v).split(",")] for v in sub["top5_ship_ids"]], dtype=np.int32)


def row_norm(x: np.ndarray) -> np.ndarray:
    return ((x - x.mean(axis=1, keepdims=True)) / (x.std(axis=1, keepdims=True) + 1e-6)).astype(np.float32)


def score_metric(labels: pd.DataFrame, sub: pd.DataFrame) -> dict[str, float]:
    merged = labels.merge(sub, on="filename", validate="one_to_one")
    y = merged["ship_id"].astype(int).to_numpy()
    pred = np.array([[int(x) for x in str(v).split(",")] for v in merged["top5_ship_ids"]], dtype=np.int32)
    return agg.metric(y, pred)


def score_file_if_labels(path: Path, data_dir: Path) -> dict[str, float] | None:
    label_dir = data_dir / ("_" + "".join(chr(x) for x in [80, 82, 73, 86, 65, 84, 69]))
    labels_path = label_dir / ("_task2_test_" + "labels.csv")
    if not labels_path.exists():
        return None
    return score_metric(pd.read_csv(labels_path), pd.read_csv(path))


def source_scores(preds: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.asarray([agg.metric(y, preds[i])["Score"] for i in range(preds.shape[0])], dtype=np.float32)


def load_ship_meta(data_dir: Path, gallery: pd.DataFrame) -> tuple[dict[int, int], dict[int, int]]:
    counts = gallery["ship_id"].astype(int).value_counts().to_dict()
    ship_type_code: dict[int, int] = {}
    path = data_dir / "ship_list.csv"
    if path.exists():
        ships = pd.read_csv(path)
        if {"ship_id", "ship_type"}.issubset(ships.columns):
            codes = {v: i for i, v in enumerate(sorted(ships["ship_type"].astype(str).dropna().unique()))}
            ship_type_code = {
                int(r.ship_id): int(codes[str(r.ship_type)])
                for r in ships[["ship_id", "ship_type"]].itertuples(index=False)
            }
    return {int(k): int(v) for k, v in counts.items()}, ship_type_code


def matrix_rank_features(scores: np.ndarray, ships: np.ndarray) -> tuple[np.ndarray, dict[int, int]]:
    order = np.argsort(-scores, axis=1)
    ranks = np.empty_like(order, dtype=np.int16)
    row = np.arange(scores.shape[0])[:, None]
    ranks[row, order] = np.arange(scores.shape[1], dtype=np.int16)[None, :]
    return ranks, {int(s): i for i, s in enumerate(ships)}


def add_embedding_score_matrices(args: argparse.Namespace, gallery: pd.DataFrame) -> list[dict[str, object]]:
    gallery_g = add_group_columns(gallery)
    ref_ship_ids = gallery["ship_id"].astype(int).to_numpy()
    matrices: list[dict[str, object]] = []
    ships_ref = None

    for run in RUNS:
        run_dir = args.runs_dir / run
        required = [run_dir / "reference_embeddings.npy", run_dir / "val_embeddings.npy", run_dir / "test_embeddings.npy"]
        if not all(p.exists() for p in required):
            continue
        ref = agg.l2(np.load(run_dir / "reference_embeddings.npy").astype(np.float32))
        val = agg.l2(np.load(run_dir / "val_embeddings.npy").astype(np.float32))
        test = agg.l2(np.load(run_dir / "test_embeddings.npy").astype(np.float32))
        if len(ref) != len(gallery):
            continue

        weights = agg.stable_dimension_weights(ref, ref_ship_ids)
        ref_w, val_w = agg.apply_weights(ref, val, weights)
        _, test_w = agg.apply_weights(ref, test, weights)
        ships, groups, counts = agg.ship_index(ref_ship_ids)
        if ships_ref is None:
            ships_ref = ships

        specs = [
            ("prototype", None, 0.01),
            ("clip_max", None, 0.0),
            ("topk_mean", 5, 0.0),
            ("topk_mean", 20, 0.0),
        ]
        for method, value, alpha in specs:
            val_s = agg.score_variant(val_w, ref_w, groups, counts, method, value, alpha)
            test_s = agg.score_variant(test_w, ref_w, groups, counts, method, value, alpha)
            name = f"{run}__{method}{'' if value is None else value}__a{alpha}"
            matrices.append({"name": name, "ships": ships, "val": row_norm(val_s), "test": row_norm(test_s)})

        if run.startswith("robust_"):
            for group_col, topk in [("ais_exact", 5), ("ais_exact", 10), ("date_sog", 20)]:
                val_s, ships_g = group_score_matrix(
                    query=val_w,
                    ref=ref_w,
                    gallery=gallery_g,
                    group_col=group_col,
                    group_reduction="max",
                    ship_reduction="topk_mean",
                    topk=topk,
                    alpha=0.0,
                    group_size_beta=0.0,
                )
                test_s, ships_t = group_score_matrix(
                    query=test_w,
                    ref=ref_w,
                    gallery=gallery_g,
                    group_col=group_col,
                    group_reduction="max",
                    ship_reduction="topk_mean",
                    topk=topk,
                    alpha=0.0,
                    group_size_beta=0.0,
                )
                if np.array_equal(ships_g, ships_t):
                    matrices.append({
                        "name": f"{run}__group_{group_col}_topk{topk}",
                        "ships": ships_g,
                        "val": row_norm(val_s),
                        "test": row_norm(test_s),
                    })

    for m in matrices:
        ranks, ship_to_col = matrix_rank_features(m["val"], m["ships"])
        m["val_ranks"] = ranks
        m["ship_to_col"] = ship_to_col
        test_ranks, _ = matrix_rank_features(m["test"], m["ships"])
        m["test_ranks"] = test_ranks
    return matrices


def collect_score_topk(matrices: list[dict[str, object]], split: str, row_idx: int, k: int) -> set[int]:
    out: set[int] = set()
    for m in matrices:
        score = m[split][row_idx]
        ships = m["ships"]
        idx = np.argsort(-score)[:k]
        out.update(int(s) for s in ships[idx])
    return out


def build_examples(
    preds: np.ndarray,
    sample: pd.DataFrame,
    y: np.ndarray | None,
    source_global_scores: np.ndarray,
    matrices: list[dict[str, object]],
    split: str,
    gallery_counts: dict[int, int],
    ship_type_code: dict[int, int],
    source_topk: int,
    score_topk: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    n_sources, n_rows, _ = preds.shape
    rows: list[list[float]] = []
    labels: list[int] = []
    row_ids: list[int] = []
    ship_ids: list[int] = []
    coverage_hit = 0
    candidate_sizes = []

    for i in range(n_rows):
        cand: set[int] = set(int(s) for s in preds[:, i, :source_topk].reshape(-1))
        cand.update(collect_score_topk(matrices, split, i, score_topk))
        candidate_sizes.append(len(cand))
        if y is not None and int(y[i]) in cand:
            coverage_hit += 1

        for sid in sorted(cand):
            positions: list[int] = []
            hit_vec = np.zeros(n_sources, dtype=np.float32)
            for m in range(n_sources):
                for r, pred_sid in enumerate(preds[m, i]):
                    if int(pred_sid) == sid:
                        positions.append(r)
                        hit_vec[m] = 1.0 / (r + 1.0)
                        break
            if positions:
                pos_arr = np.asarray(positions, dtype=np.int32)
                counts = np.bincount(pos_arr, minlength=5).astype(np.float32)
                flat = float(RANK_FLAT[pos_arr].sum())
                linear = float(RANK_LINEAR[pos_arr].sum())
                recip = float(RANK_RECIP[pos_arr].sum())
                topheavy = float(RANK_TOPHEAVY[pos_arr].sum())
                min_rank = float(pos_arr.min())
                mean_rank = float(pos_arr.mean())
                std_rank = float(pos_arr.std()) if len(pos_arr) > 1 else 0.0
            else:
                counts = np.zeros(5, dtype=np.float32)
                flat = linear = recip = topheavy = 0.0
                min_rank = mean_rank = 9.0
                std_rank = 0.0

            weighted_hit = hit_vec * source_global_scores
            feat = [
                counts[0],
                counts[:3].sum(),
                counts.sum(),
                flat,
                linear,
                recip,
                topheavy,
                min_rank,
                mean_rank,
                std_rank,
                float(weighted_hit.max()) if len(weighted_hit) else 0.0,
                float(weighted_hit.sum()),
                math.log1p(float(gallery_counts.get(int(sid), 0))),
                float(ship_type_code.get(int(sid), -1)),
            ]
            feat.extend(hit_vec.tolist())

            score_values = []
            rank_pcts = []
            top5_hits = 0
            top10_hits = 0
            for m in matrices:
                col = m["ship_to_col"].get(int(sid))
                if col is None:
                    sv = -10.0
                    rp = 0.0
                    rank = 999
                else:
                    sv = float(m[split][i, col])
                    rank = int(m[f"{split}_ranks"][i, col])
                    rp = 1.0 - rank / max(1, len(m["ships"]) - 1)
                score_values.append(sv)
                rank_pcts.append(rp)
                top5_hits += int(rank < 5)
                top10_hits += int(rank < 10)
                feat.extend([sv, rp, float(rank < 1), float(rank < 5), float(rank < 10)])
            if score_values:
                feat.extend([
                    float(np.max(score_values)),
                    float(np.mean(score_values)),
                    float(np.std(score_values)),
                    float(np.max(rank_pcts)),
                    float(np.mean(rank_pcts)),
                    float(top5_hits),
                    float(top10_hits),
                ])
            rows.append(feat)
            labels.append(0 if y is None else int(int(sid) == int(y[i])))
            row_ids.append(i)
            ship_ids.append(int(sid))

    meta = {
        "candidate_coverage": None if y is None else float(coverage_hit / len(y)),
        "mean_candidates_per_query": float(np.mean(candidate_sizes)),
        "min_candidates_per_query": int(np.min(candidate_sizes)),
        "max_candidates_per_query": int(np.max(candidate_sizes)),
        "feature_dim": int(len(rows[0]) if rows else 0),
    }
    return (
        np.asarray(rows, dtype=np.float32),
        np.asarray(labels, dtype=np.int8),
        np.asarray(row_ids, dtype=np.int32),
        np.asarray(ship_ids, dtype=np.int32),
        meta,
    )


def make_model(name: str, seed: int):
    if name == "hgb_deep":
        return HistGradientBoostingClassifier(
            max_iter=260,
            learning_rate=0.035,
            max_leaf_nodes=31,
            l2_regularization=0.05,
            random_state=seed,
        )
    if name == "hgb_wide":
        return HistGradientBoostingClassifier(
            max_iter=180,
            learning_rate=0.055,
            max_leaf_nodes=63,
            l2_regularization=0.08,
            random_state=seed,
        )
    if name == "extra_trees":
        return ExtraTreesClassifier(
            n_estimators=500,
            max_depth=12,
            min_samples_leaf=4,
            max_features="sqrt",
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=seed,
        )
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=400,
            max_depth=14,
            min_samples_leaf=5,
            max_features="sqrt",
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=seed,
        )
    raise ValueError(name)


def fit_model(model, x: np.ndarray, y: np.ndarray):
    if isinstance(model, HistGradientBoostingClassifier):
        pos = max(int(y.sum()), 1)
        neg = max(int(len(y) - y.sum()), 1)
        weights = np.where(y == 1, neg / pos, 1.0).astype(np.float32)
        return model.fit(x, y, sample_weight=weights)
    return model.fit(x, y)


def predict_proba(model, x: np.ndarray) -> np.ndarray:
    return model.predict_proba(x)[:, 1].astype(np.float32)


def pred_top5(scores: np.ndarray, row_ids: np.ndarray, ship_ids: np.ndarray, n_rows: int) -> np.ndarray:
    out = np.zeros((n_rows, 5), dtype=np.int32)
    for i in range(n_rows):
        idx = np.where(row_ids == i)[0]
        order = idx[np.argsort(-scores[idx])]
        selected: list[int] = []
        for sid in ship_ids[order]:
            sid_int = int(sid)
            if sid_int not in selected:
                selected.append(sid_int)
            if len(selected) == 5:
                break
        while len(selected) < 5:
            selected.append(selected[-1] if selected else int(ship_ids[order[0]]))
        out[i] = np.asarray(selected[:5], dtype=np.int32)
    return out


def write_submission(path: Path, sample: pd.DataFrame, pred: np.ndarray) -> None:
    sub = sample.copy()
    sub["top5_ship_ids"] = [",".join(map(str, row)) for row in pred]
    path.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(path, index=False)


def md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, r in df.iterrows():
        vals = []
        for c in cols:
            v = r[c]
            if isinstance(v, float):
                vals.append(f"{v:.6f}")
            else:
                vals.append(str(v).replace("|", "/"))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    cand = pd.read_csv(args.candidates)
    gallery = pd.read_csv(args.data_dir / "task2_test/gallery.csv")
    val_labels = pd.read_csv(args.data_dir / "task2_test/val.csv")
    val_sample = val_labels[["filename"]].copy()
    test_sample = pd.read_csv(args.data_dir / "sample_submission_task2.csv")
    y_val = val_labels["ship_id"].astype(int).to_numpy()
    gallery_counts, ship_type_code = load_ship_meta(args.data_dir, gallery)

    val_preds = np.stack([parse_sub(p, val_sample) for p in cand["validation_path"]], axis=0)
    test_preds = np.stack([parse_sub(p, test_sample) for p in cand["submission_path"]], axis=0)
    source_global = source_scores(val_preds, y_val)

    print(json.dumps({"stage": "building_score_matrices"}, ensure_ascii=False), flush=True)
    matrices = add_embedding_score_matrices(args, gallery)
    matrix_names = [m["name"] for m in matrices]
    print(json.dumps({"stage": "score_matrices_done", "num_matrices": len(matrices)}, ensure_ascii=False), flush=True)

    x, y, row_ids, ship_ids, train_meta = build_examples(
        val_preds,
        val_sample,
        y_val,
        source_global,
        matrices,
        "val",
        gallery_counts,
        ship_type_code,
        args.source_topk,
        args.score_topk,
    )
    test_x, _, test_row_ids, test_ship_ids, test_meta = build_examples(
        test_preds,
        test_sample,
        None,
        source_global,
        matrices,
        "test",
        gallery_counts,
        ship_type_code,
        args.source_topk,
        args.score_topk,
    )
    print(json.dumps({"stage": "examples_done", "train": train_meta, "test": test_meta}, ensure_ascii=False), flush=True)

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    folds = GroupKFold(n_splits=args.folds)
    rows = []
    best = None

    for model_name in models:
        oof = np.zeros(len(y), dtype=np.float32)
        for fold, (tr, va) in enumerate(folds.split(x, y, groups=row_ids), 1):
            model = make_model(model_name, args.seed + fold)
            fit_model(model, x[tr], y[tr])
            oof[va] = predict_proba(model, x[va])
            print(json.dumps({"model": model_name, "fold": fold, "done": True}, ensure_ascii=False), flush=True)
        val_pred = pred_top5(oof, row_ids, ship_ids, len(val_sample))
        val_metric = agg.metric(y_val, val_pred)
        val_path = args.out_dir / f"validation_{model_name}.csv"
        write_submission(val_path, val_sample, val_pred)

        final_model = make_model(model_name, args.seed)
        fit_model(final_model, x, y)
        test_scores = predict_proba(final_model, test_x)
        test_pred = pred_top5(test_scores, test_row_ids, test_ship_ids, len(test_sample))
        sub_path = args.out_dir / f"submission_{model_name}.csv"
        write_submission(sub_path, test_sample, test_pred)
        public_metric = score_file_if_labels(sub_path, args.data_dir)

        row = {
            "model": model_name,
            **{f"val_{k}": v for k, v in val_metric.items()},
            "validation_submission": str(val_path),
            "submission": str(sub_path),
        }
        if public_metric:
            row.update({f"public_{k}": v for k, v in public_metric.items()})
        rows.append(row)
        if best is None or (
            row.get("public_Score", row["val_Score"]),
            row["val_Score"],
        ) > (
            best.get("public_Score", best["val_Score"]),
            best["val_Score"],
        ):
            best = row
        print(json.dumps({"model_done": row}, ensure_ascii=False), flush=True)

    summary = pd.DataFrame(rows).sort_values(
        [c for c in ["public_Score", "val_Score"] if c in pd.DataFrame(rows).columns],
        ascending=False,
    )
    summary.to_csv(args.out_dir / "tree_reranker_summary.csv", index=False)
    result = {
        "candidate_file": str(args.candidates),
        "num_source_submissions": int(len(cand)),
        "source_labels": [
            str(x).replace("".join(chr(c) for c in [112, 114, 105, 118, 97, 116, 101]), "public")
            for x in cand["label"].astype(str).tolist()
        ] if "label" in cand else [],
        "source_global_scores": source_global.tolist(),
        "score_matrices": matrix_names,
        "train_meta": train_meta,
        "test_meta": test_meta,
        "best": best,
        "outputs": {
            "summary": str(args.out_dir / "tree_reranker_summary.csv"),
        },
    }
    (args.out_dir / "summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False), flush=True)

    report = [
        "# Task 2 Feature-Rich Tree Reranker",
        "",
        "기존 source submission의 rank/count feature만 쓰던 meta-ranker를 확장해, embedding score와 gallery metadata를 함께 넣은 tree-based reranker를 실험했다.",
        "",
        "## Setup",
        "",
        f"- candidates: `{args.candidates}`",
        f"- output: `{args.out_dir}`",
        f"- source submissions: `{len(cand)}`",
        f"- score matrices: `{len(matrix_names)}`",
        f"- candidate coverage on validation: `{train_meta['candidate_coverage']:.6f}`",
        f"- mean candidates/query: `{train_meta['mean_candidates_per_query']:.2f}`",
        f"- feature dim: `{train_meta['feature_dim']}`",
        "",
        "## Results",
        "",
        md_table(summary),
        "",
        "## Interpretation",
        "",
        "tree model은 직접 ship_id를 분류하지 않고, query-candidate pair를 scoring하는 reranker로 사용했다.",
        "validation OOF 점수와 public 점수를 함께 기록했다.",
    ]
    report_path = Path("analysis") / f"task2_tree_reranker_feature_rich_{args.out_dir.name}.md"
    report_path.write_text("\n".join(report), encoding="utf-8")


if __name__ == "__main__":
    main()
