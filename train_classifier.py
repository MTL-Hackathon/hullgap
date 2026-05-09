#!/usr/bin/env python3
"""
train_classifier.py

Trains a binary crystal stability classifier on the Materials Project dataset
produced by build_dataset.py. Compares Logistic Regression, Random Forest, and
Gradient Boosting via stratified 5-fold CV, retrains the winner on the full
training split, evaluates on a held-out test set, and saves the model.

Usage:
    python train_classifier.py
    python train_classifier.py --data data/results/training_data_balanced.csv
    python train_classifier.py --output-dir models
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ── constants ─────────────────────────────────────────────────────────────────

# Dropped: identifiers + energy_above_hull (direct data leakage — it defines
# is_stable, so any model using it would memorise the label not learn structure)
_DROP_COLS = {"material_id", "formula_pretty", "chemsys", "energy_above_hull", "is_stable"}

_CANDIDATES = {
    "LogisticRegression": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, C=0.1, random_state=42)),
    ]),
    "RandomForest": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=300, max_depth=6,
                                       min_samples_leaf=3, random_state=42)),
    ]),
    "GradientBoosting": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", GradientBoostingClassifier(n_estimators=200, max_depth=3,
                                           learning_rate=0.05, subsample=0.8,
                                           random_state=42)),
    ]),
}

_CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
_SCORING = ("accuracy", "f1", "roc_auc")


# ── data loading ──────────────────────────────────────────────────────────────

def load_data(path: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Load the balanced CSV and return (X, y, feature_names).

    Drops identifier and leakage columns; target is is_stable.
    """
    df = pd.read_csv(path)
    if "is_stable" not in df.columns:
        raise ValueError(f"'is_stable' column not found in {path}")

    feature_cols = [c for c in df.columns if c not in _DROP_COLS]
    X = df[feature_cols].values.astype(float)
    y = df["is_stable"].values.astype(int)

    print(f"Loaded {len(df)} rows from {path}")
    print(f"Features ({len(feature_cols)}): {feature_cols}")
    print(f"Class balance: stable={y.sum()}, unstable={len(y)-y.sum()}")
    return X, y, feature_cols


# ── model selection ───────────────────────────────────────────────────────────

def select_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> tuple[str, Pipeline, pd.DataFrame]:
    """
    Run stratified 5-fold CV on all candidate models.

    Returns (best_name, best_pipeline, cv_results_df) ranked by mean ROC-AUC.
    """
    print("\n5-fold stratified cross-validation")
    print("-" * 52)
    print(f"{'Model':<22} {'Acc':>7} {'F1':>7} {'AUC':>7}")
    print("-" * 52)

    rows = []
    for name, pipeline in _CANDIDATES.items():
        cv = cross_validate(pipeline, X_train, y_train, cv=_CV,
                            scoring=list(_SCORING), return_train_score=False)
        row = {
            "model": name,
            "acc_mean": cv["test_accuracy"].mean(),
            "acc_std": cv["test_accuracy"].std(),
            "f1_mean": cv["test_f1"].mean(),
            "f1_std": cv["test_f1"].std(),
            "auc_mean": cv["test_roc_auc"].mean(),
            "auc_std": cv["test_roc_auc"].std(),
        }
        rows.append(row)
        print(f"  {name:<20} {row['acc_mean']:.3f}   {row['f1_mean']:.3f}   {row['auc_mean']:.3f}")

    print("-" * 52)

    results = pd.DataFrame(rows).sort_values("auc_mean", ascending=False)
    best_name = results.iloc[0]["model"]
    print(f"\nBest model by AUC: {best_name}  (AUC {results.iloc[0]['auc_mean']:.3f} +/- {results.iloc[0]['auc_std']:.3f})")
    return best_name, _CANDIDATES[best_name], results


# ── evaluation ────────────────────────────────────────────────────────────────

def evaluate(
    pipeline: Pipeline,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: list[str],
) -> Pipeline:
    """
    Fit the pipeline on the training split and report held-out test metrics.

    Returns the fitted pipeline.
    """
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    print("\nHeld-out test set results  (20% of data)")
    print("-" * 52)
    print(classification_report(y_test, y_pred, target_names=["unstable", "stable"]))

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    auc = roc_auc_score(y_test, y_prob)
    print(f"ROC-AUC : {auc:.4f}")
    print(f"Confusion matrix:")
    print(f"  TN={tn}  FP={fp}")
    print(f"  FN={fn}  TP={tp}")

    _print_feature_importance(pipeline, feature_names)
    return pipeline


def _print_feature_importance(pipeline: Pipeline, feature_names: list[str]) -> None:
    clf = pipeline.named_steps["clf"]
    if hasattr(clf, "feature_importances_"):
        importances = clf.feature_importances_
        label = "Feature importances (Gini)"
    elif hasattr(clf, "coef_"):
        importances = np.abs(clf.coef_[0])
        label = "Feature importances (|coef|)"
    else:
        return

    order = np.argsort(importances)[::-1]
    print(f"\n{label}:")
    print("-" * 40)
    for i in order:
        bar = "#" * int(importances[i] * 40 / importances[order[0]])
        print(f"  {feature_names[i]:<30} {importances[i]:.4f}  {bar}")


# ── save ──────────────────────────────────────────────────────────────────────

def save_model(pipeline: Pipeline, out_dir: Path, name: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "stability_classifier.joblib"
    dump({"model": pipeline, "model_name": name}, model_path)
    print(f"\nModel saved -> {model_path.resolve()}")
    return model_path


# ── orchestration ─────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a crystal stability classifier on MP binary compound data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data", default="data/results/training_data_balanced.csv",
        help="Path to the balanced training CSV",
    )
    parser.add_argument(
        "--output-dir", default="models",
        help="Directory to save the trained model",
    )
    parser.add_argument(
        "--test-size", type=float, default=0.2,
        help="Fraction of data held out for final evaluation",
    )
    args = parser.parse_args()

    # 1 — load
    X, y, feature_names = load_data(args.data)

    # 2 — stratified train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=42,
    )
    print(f"\nTrain: {len(X_train)} samples  |  Test: {len(X_test)} samples")

    # 3 — cross-validate all candidates on training split only
    best_name, best_pipeline, cv_results = select_model(X_train, y_train)

    # 4 — fit winner on full train split, evaluate on held-out test
    fitted_pipeline = evaluate(
        best_pipeline, X_train, y_train, X_test, y_test, feature_names,
    )

    # 5 — save
    save_model(fitted_pipeline, Path(args.output_dir), best_name)


if __name__ == "__main__":
    main()
