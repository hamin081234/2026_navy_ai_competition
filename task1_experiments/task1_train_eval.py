#!/usr/bin/env python3
"""Task 1 ship-type classification experiments.

This script trains measured baselines for the uploaded competition data:
AIS-only, audio-only handcrafted features, and fused audio+AIS models.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import soundfile as sf
from scipy import signal, stats
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


CLASSES = ["A_SmallWorking", "B_MotorBoat", "C_Passenger", "D_LargeShip"]
BANDS = np.array([0, 20, 40, 80, 160, 315, 630, 1250, 2500, 5000, 10000, 16000], dtype=float)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="database", type=Path)
    parser.add_argument("--out-dir", default="task1_experiments", type=Path)
    parser.add_argument("--audio-workers", default=-1, type=int)
    parser.add_argument("--skip-audio", action="store_true")
    return parser.parse_args()


def load_tables(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(data_dir / "train/train.csv")
    val = pd.read_csv(data_dir / "task1_test/val.csv")
    test = pd.read_csv(data_dir / "task1_test/test.csv")
    train["split"] = "train"
    val["split"] = "val"
    test["split"] = "test"
    train["audio_path"] = train["filename"].map(lambda x: data_dir / "train/audio" / x)
    val["audio_path"] = val["filename"].map(lambda x: data_dir / "task1_test/audio" / x)
    test["audio_path"] = test["filename"].map(lambda x: data_dir / "task1_test/audio" / x)
    return train, val, test


def add_ais_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    ts = pd.to_datetime(out["ais_timestamp"], utc=True, format="mixed")
    out["hour_sin"] = np.sin(2 * np.pi * ts.dt.hour / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * ts.dt.hour / 24.0)
    out["month_sin"] = np.sin(2 * np.pi * ts.dt.month / 12.0)
    out["month_cos"] = np.cos(2 * np.pi * ts.dt.month / 12.0)
    out["dow_sin"] = np.sin(2 * np.pi * ts.dt.dayofweek / 7.0)
    out["dow_cos"] = np.cos(2 * np.pi * ts.dt.dayofweek / 7.0)
    for col in ["cog", "true_heading"]:
        radians = np.deg2rad(out[col].astype(float) % 360)
        out[f"{col}_sin"] = np.sin(radians)
        out[f"{col}_cos"] = np.cos(radians)
    diff = ((out["cog"].astype(float) - out["true_heading"].astype(float) + 180) % 360) - 180
    out["cog_heading_absdiff"] = np.abs(diff)
    out["is_stopped"] = (out["sog"].astype(float) < 0.5).astype(int)
    out["sog_log1p"] = np.log1p(out["sog"].astype(float).clip(lower=0))
    return out


def audio_features_one(path: Path) -> dict[str, float | str]:
    y, sr = sf.read(path, always_2d=False, dtype="float32")
    if y.ndim > 1:
        y = y.mean(axis=1)
    y = np.nan_to_num(y.astype(np.float32), copy=False)
    if y.size == 0:
        return {"filename": path.name}

    abs_y = np.abs(y)
    rms = float(np.sqrt(np.mean(y * y) + 1e-12))
    peak = float(np.max(abs_y) + 1e-12)
    q = np.quantile(y, [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99])
    zcr = float(np.mean(y[:-1] * y[1:] < 0)) if y.size > 1 else 0.0

    f, pxx = signal.welch(y, fs=sr, window="hann", nperseg=2048, noverlap=1024, detrend=False)
    pxx = pxx.astype(np.float64) + 1e-18
    total_power = float(np.sum(pxx))
    centroid = float(np.sum(f * pxx) / total_power)
    cumulative = np.cumsum(pxx)
    rolloff = float(f[np.searchsorted(cumulative, 0.85 * cumulative[-1])])
    flatness = float(stats.gmean(pxx) / np.mean(pxx))

    feats: dict[str, float | str] = {
        "filename": path.name,
        "audio_mean": float(np.mean(y)),
        "audio_std": float(np.std(y)),
        "audio_abs_mean": float(np.mean(abs_y)),
        "audio_rms": rms,
        "audio_log_rms": float(np.log(rms + 1e-12)),
        "audio_peak": peak,
        "audio_crest": float(peak / (rms + 1e-12)),
        "audio_zcr": zcr,
        "audio_skew": float(stats.skew(y, bias=False)),
        "audio_kurtosis": float(stats.kurtosis(y, bias=False)),
        "audio_q01": float(q[0]),
        "audio_q05": float(q[1]),
        "audio_q25": float(q[2]),
        "audio_q50": float(q[3]),
        "audio_q75": float(q[4]),
        "audio_q95": float(q[5]),
        "audio_q99": float(q[6]),
        "audio_centroid": centroid,
        "audio_rolloff85": rolloff,
        "audio_flatness": flatness,
    }

    segments = np.array_split(y, 10)
    seg_rms = np.array([np.sqrt(np.mean(s * s) + 1e-12) for s in segments])
    for i, value in enumerate(seg_rms):
        feats[f"seg{i}_log_rms"] = float(np.log(value + 1e-12))
    feats["seg_rms_std"] = float(np.std(seg_rms))
    feats["seg_rms_max_min"] = float(np.max(seg_rms) - np.min(seg_rms))

    for lo, hi in zip(BANDS[:-1], BANDS[1:]):
        mask = (f >= lo) & (f < hi)
        band_power = float(np.sum(pxx[mask]))
        feats[f"band_{int(lo)}_{int(hi)}_log"] = float(np.log(band_power + 1e-18))
        feats[f"band_{int(lo)}_{int(hi)}_frac"] = float(band_power / total_power)
    return feats


def build_audio_features(all_df: pd.DataFrame, out_dir: Path, workers: int) -> pd.DataFrame:
    cache = out_dir / "audio_features_task1.joblib"
    if cache.exists():
        return joblib.load(cache)
    paths = list(all_df["audio_path"])
    rows = joblib.Parallel(n_jobs=workers, verbose=10, batch_size=32)(
        joblib.delayed(audio_features_one)(Path(p)) for p in paths
    )
    feats = pd.DataFrame(rows)
    joblib.dump(feats, cache)
    feats.to_csv(out_dir / "audio_features_task1.csv", index=False)
    return feats


def evaluate(name: str, model, x_train, y_train, x_val, y_val) -> dict[str, object]:
    model.fit(x_train, y_train)
    pred = model.predict(x_val)
    return {
        "model": name,
        "macro_f1": float(f1_score(y_val, pred, average="macro")),
        "accuracy": float(accuracy_score(y_val, pred)),
        "classification_report": classification_report(y_val, pred, labels=CLASSES, output_dict=True, zero_division=0),
    }


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    train, val, test = load_tables(args.data_dir)
    all_df = pd.concat([train, val, test], ignore_index=True, sort=False)
    all_df = add_ais_features(all_df)

    if not args.skip_audio:
        audio = build_audio_features(all_df, args.out_dir, args.audio_workers)
        all_df = all_df.merge(audio, on="filename", how="left")

    ais_cols = [
        "sog",
        "sog_log1p",
        "is_stopped",
        "cog_sin",
        "cog_cos",
        "true_heading_sin",
        "true_heading_cos",
        "cog_heading_absdiff",
        "hour_sin",
        "hour_cos",
        "month_sin",
        "month_cos",
        "dow_sin",
        "dow_cos",
    ]
    audio_cols = [c for c in all_df.columns if c.startswith(("audio_", "seg", "band_"))]

    train_df = all_df[all_df["split"] == "train"].copy()
    val_df = all_df[all_df["split"] == "val"].copy()
    test_df = all_df[all_df["split"] == "test"].copy()

    y_train = train_df["ship_type"]
    y_val = val_df["ship_type"]
    results = []

    numeric_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    tree_pipe = Pipeline([("imputer", SimpleImputer(strategy="median"))])

    experiments = [
        ("majority_train_prior", [], DummyClassifier(strategy="most_frequent")),
        ("ais_logistic_balanced", ais_cols, Pipeline([
            ("prep", ColumnTransformer([("num", numeric_pipe, ais_cols)], remainder="drop")),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0, n_jobs=-1)),
        ])),
        ("ais_random_forest_balanced", ais_cols, Pipeline([
            ("prep", ColumnTransformer([("num", tree_pipe, ais_cols)], remainder="drop")),
            ("clf", RandomForestClassifier(
                n_estimators=500,
                min_samples_leaf=5,
                class_weight="balanced_subsample",
                random_state=42,
                n_jobs=-1,
            )),
        ])),
        ("ais_hist_gradient_boosting", ais_cols, Pipeline([
            ("prep", ColumnTransformer([("num", tree_pipe, ais_cols)], remainder="drop")),
            ("clf", HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, l2_regularization=0.1, random_state=42)),
        ])),
    ]
    if audio_cols:
        experiments.extend([
            ("audio_extra_trees_balanced", audio_cols, Pipeline([
                ("prep", ColumnTransformer([("num", tree_pipe, audio_cols)], remainder="drop")),
                ("clf", ExtraTreesClassifier(
                    n_estimators=700,
                    min_samples_leaf=2,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1,
                )),
            ])),
            ("fused_extra_trees_balanced", ais_cols + audio_cols, Pipeline([
                ("prep", ColumnTransformer([("num", tree_pipe, ais_cols + audio_cols)], remainder="drop")),
                ("clf", ExtraTreesClassifier(
                    n_estimators=900,
                    min_samples_leaf=2,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1,
                )),
            ])),
            ("fused_random_forest_balanced", ais_cols + audio_cols, Pipeline([
                ("prep", ColumnTransformer([("num", tree_pipe, ais_cols + audio_cols)], remainder="drop")),
                ("clf", RandomForestClassifier(
                    n_estimators=700,
                    min_samples_leaf=3,
                    class_weight="balanced_subsample",
                    random_state=43,
                    n_jobs=-1,
                )),
            ])),
        ])

    fitted = {}
    for name, cols, model in experiments:
        x_train = train_df[cols] if cols else train_df[["sog"]]
        x_val = val_df[cols] if cols else val_df[["sog"]]
        result = evaluate(name, model, x_train, y_train, x_val, y_val)
        print(f"{name}: macro_f1={result['macro_f1']:.6f} accuracy={result['accuracy']:.6f}")
        results.append(result)
        fitted[name] = (cols, model)

    results_sorted = sorted(results, key=lambda r: r["macro_f1"], reverse=True)
    best_name = results_sorted[0]["model"]
    best_cols, best_model = fitted[best_name]

    # Refit the best configuration on train + labeled validation before predicting hidden test.
    trainval_df = pd.concat([train_df, val_df], ignore_index=True, sort=False)
    x_trainval = trainval_df[best_cols] if best_cols else trainval_df[["sog"]]
    y_trainval = trainval_df["ship_type"]
    x_test = test_df[best_cols] if best_cols else test_df[["sog"]]
    best_model.fit(x_trainval, y_trainval)
    test_pred = best_model.predict(x_test)

    submission = pd.read_csv(args.data_dir / "sample_submission_task1.csv")
    submission["predicted_class"] = test_pred
    submission.to_csv(args.out_dir / "submission_task1.csv", index=False)
    joblib.dump(best_model, args.out_dir / "best_task1_model.joblib")

    payload = {
        "classes": CLASSES,
        "best_model_by_val_macro_f1": best_name,
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "test_rows": int(len(test_df)),
        "ais_features": ais_cols,
        "audio_feature_count": int(len(audio_cols)),
        "results": results_sorted,
    }
    (args.out_dir / "task1_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    rows = []
    for result in results_sorted:
        rows.append({
            "model": result["model"],
            "macro_f1": result["macro_f1"],
            "accuracy": result["accuracy"],
            **{f"f1_{cls}": result["classification_report"][cls]["f1-score"] for cls in CLASSES},
        })
    pd.DataFrame(rows).to_csv(args.out_dir / "task1_scores.csv", index=False)


if __name__ == "__main__":
    main()
