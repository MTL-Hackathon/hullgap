#!/usr/bin/env python3
"""
evaluate_cobi.py

Evaluates the trained stability classifier on Co-Bi — the target system.

Steps:
  1. Reload the full MP dataset, strip out all Co-Bi rows, rebalance, retrain
     GradientBoosting so Co-Bi is never seen during training.
  2. Query Materials Project fresh for every Co-Bi entry.
  3. Apply identical feature engineering to the Co-Bi entries.
  4. Predict stability and compare against the true energy_above_hull labels.
  5. Report accuracy, ROC-AUC, and a confusion matrix.

Usage:
    python evaluate_cobi.py
    python evaluate_cobi.py --hull-threshold 0.05
"""

import argparse
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
load_dotenv()

_ORDERING_MAP = {"NM": 0, "FM": 1, "FiM": 2, "AFM": 3}
_ID_COLS      = {"material_id", "formula_pretty", "chemsys"}
_LEAK_COLS    = {"energy_above_hull", "is_stable"}
_DROP_COLS    = _ID_COLS | _LEAK_COLS

_SUMMARY_FIELDS = [
    "material_id", "formula_pretty", "chemsys", "symmetry",
    "nsites", "volume", "density", "formation_energy_per_atom",
    "energy_above_hull", "band_gap", "total_magnetization", "ordering",
]


# ── feature engineering (shared by train and test) ───────────────────────────

def engineer(df: pd.DataFrame, cs_columns: list[str] | None = None) -> pd.DataFrame:
    """
    Apply the same transformations used in build_dataset.py.

    If cs_columns is provided, one-hot columns are aligned to that list
    (missing columns filled with 0, extra columns dropped). Pass None for
    the training set to infer the columns from the data.
    """
    df = df.copy()

    df["volume_per_atom"] = df["volume"] / df["nsites"]
    df["composition_A_fraction"] = df.apply(_first_element_fraction, axis=1)
    df["band_gap"] = df["band_gap"].fillna(-1.0)
    df["total_magnetization"] = df["total_magnetization"].fillna(0.0)
    df["magnetic_ordering"] = (
        df["magnetic_ordering"]
        .map(lambda x: _ORDERING_MAP.get(str(x).strip(), -1) if pd.notna(x) else -1)
        .astype(int)
    )

    df["crystal_system"] = df["crystal_system"].fillna("Unknown")
    cs_dummies = pd.get_dummies(df["crystal_system"], prefix="cs").astype(int)
    df = pd.concat([df.drop(columns=["crystal_system"]), cs_dummies], axis=1)

    if cs_columns is not None:
        for col in cs_columns:
            if col not in df.columns:
                df[col] = 0
        extra = [c for c in df.columns if c.startswith("cs_") and c not in cs_columns]
        if extra:
            df = df.drop(columns=extra)

    return df


def _first_element_fraction(row: pd.Series) -> float:
    from pymatgen.core import Composition
    try:
        first_el = sorted(row["chemsys"].split("-"))[0]
        return float(Composition(row["formula_pretty"]).get_atomic_fraction(first_el))
    except Exception:
        return np.nan


# ── step 1: retrain without Co-Bi ────────────────────────────────────────────

def retrain_without_cobi(
    full_csv: str,
    hull_threshold: float,
) -> tuple[Pipeline, list[str], list[str]]:
    """
    Load the full dataset (already feature-engineered by build_dataset.py),
    drop Co-Bi rows, rebalance, retrain GradientBoosting.

    The full CSV already has cs_* one-hot columns and all derived features —
    no need to re-run engineer() on it.

    Returns (fitted_pipeline, feature_names, cs_column_names).
    """
    df = pd.read_csv(full_csv)

    # separate Co-Bi out completely
    mask_cobi = df["chemsys"] == "Bi-Co"
    n_cobi = mask_cobi.sum()
    df_train = df[~mask_cobi].copy()
    print(f"Full dataset : {len(df)} rows")
    print(f"Co-Bi removed: {n_cobi} rows")
    print(f"Training pool: {len(df_train)} rows")

    # drop rows missing critical fields and re-apply label at requested threshold
    df_train = df_train.dropna(subset=["formation_energy_per_atom", "energy_above_hull"])
    df_train = df_train[df_train["nsites"].notna() & (df_train["nsites"] > 0)]
    df_train["is_stable"] = (df_train["energy_above_hull"] <= hull_threshold).astype(int)

    # record which cs_* columns exist in the training data
    cs_columns = [c for c in df_train.columns if c.startswith("cs_")]

    # rebalance
    counts = df_train["is_stable"].value_counts()
    n_min = int(counts.min())
    df_balanced = pd.concat(
        [g.sample(n=n_min, random_state=42) for _, g in df_train.groupby("is_stable")],
        ignore_index=True,
    )
    print(f"After rebalance: {len(df_balanced)} rows "
          f"({n_min} stable / {n_min} unstable)")

    # build feature matrix — same _DROP_COLS as train_classifier.py
    feature_cols = [c for c in df_balanced.columns if c not in _DROP_COLS]
    X = df_balanced[feature_cols].values.astype(float)
    y = df_balanced["is_stable"].values.astype(int)

    # retrain
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", GradientBoostingClassifier(
            n_estimators=200, max_depth=3,
            learning_rate=0.05, subsample=0.8,
            random_state=42,
        )),
    ])
    pipeline.fit(X, y)
    print(f"Model retrained on {len(X)} samples, {len(feature_cols)} features.\n")

    return pipeline, feature_cols, cs_columns


# ── step 2: query Co-Bi from MP ───────────────────────────────────────────────

def query_cobi(api_key: str) -> pd.DataFrame:
    """Fetch all Co-Bi entries from Materials Project and return a raw DataFrame."""
    try:
        from mp_api.client import MPRester
    except ImportError:
        sys.exit("mp-api not installed — run: pip install mp-api")

    with MPRester(api_key) as mpr:
        docs = mpr.materials.summary.search(
            chemsys=["Co-Bi"],
            fields=_SUMMARY_FIELDS,
        )

    if not docs:
        sys.exit("No Co-Bi entries returned from Materials Project.")

    records = []
    for doc in docs:
        sym = doc.symmetry
        ordering = getattr(doc, "ordering", None)
        records.append({
            "material_id": str(doc.material_id),
            "formula_pretty": doc.formula_pretty,
            "chemsys": doc.chemsys,
            "crystal_system": (_enum_str(getattr(sym, "crystal_system", None)) if sym else None),
            "spacegroup_number": (getattr(sym, "number", None) if sym else None),
            "nsites": doc.nsites,
            "volume": doc.volume,
            "density": doc.density,
            "formation_energy_per_atom": doc.formation_energy_per_atom,
            "energy_above_hull": doc.energy_above_hull,
            "band_gap": doc.band_gap,
            "total_magnetization": doc.total_magnetization,
            "magnetic_ordering": _enum_str(ordering),
        })

    df = pd.DataFrame(records)
    print(f"Queried {len(df)} Co-Bi entries from Materials Project:")
    print(df[["material_id", "formula_pretty", "energy_above_hull"]].to_string(index=False))
    return df


def _enum_str(val) -> str | None:
    if val is None:
        return None
    return val.value if hasattr(val, "value") else str(val)


# ── step 3: predict and evaluate ─────────────────────────────────────────────

def predict_and_evaluate(
    pipeline: Pipeline,
    feature_cols: list[str],
    cs_columns: list[str],
    df_cobi: pd.DataFrame,
    hull_threshold: float,
) -> None:
    """Engineer Co-Bi features, predict, and report metrics."""
    df = df_cobi.dropna(subset=["formation_energy_per_atom", "energy_above_hull"]).copy()
    df["is_stable"] = (df["energy_above_hull"] <= hull_threshold).astype(int)

    df = engineer(df, cs_columns=cs_columns)

    # align to exact training feature columns (fill any gaps with 0)
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0
    X_test = df[feature_cols].values.astype(float)
    y_true = df["is_stable"].values.astype(int)

    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    # ── per-entry predictions ──────────────────────────────────────────────
    print("\nPer-entry predictions:")
    print("-" * 75)
    header = f"{'material_id':<14} {'formula':<10} {'e_above_hull':>13} {'true':>6} {'pred':>6} {'p(stable)':>10}"
    print(header)
    print("-" * 75)
    for i, row in df.reset_index(drop=True).iterrows():
        true_label = "stable" if y_true[i] == 1 else "unstable"
        pred_label = "stable" if y_pred[i] == 1 else "unstable"
        match = "" if y_true[i] == y_pred[i] else " <-- WRONG"
        print(f"  {row['material_id']:<12} {row['formula_pretty']:<10} "
              f"{row['energy_above_hull']:>12.4f}  {true_label:>8}  {pred_label:>8}"
              f"  {y_prob[i]:>8.3f}{match}")

    # ── aggregate metrics ──────────────────────────────────────────────────
    print("\n" + "=" * 40)
    print("EVALUATION METRICS (Co-Bi holdout)")
    print("=" * 40)

    acc = accuracy_score(y_true, y_pred)
    print(f"Accuracy : {acc:.4f}  ({int(acc * len(y_true))}/{len(y_true)} correct)")

    if len(np.unique(y_true)) > 1:
        auc = roc_auc_score(y_true, y_prob)
        print(f"ROC-AUC  : {auc:.4f}")
    else:
        print("ROC-AUC  : n/a (only one class present in Co-Bi test set)")

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    print(f"\nConfusion matrix (rows=true, cols=predicted):")
    print(f"                 pred:unstable  pred:stable")
    print(f"  true:unstable       {tn:>4}           {fp:>4}")
    print(f"  true:stable         {fn:>4}           {tp:>4}")

    print(f"\nNote: n={len(y_true)} - metrics are illustrative given the small Co-Bi holdout.")


# ── orchestration ─────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the stability classifier on held-out Co-Bi entries.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--full-data", default="data/results/training_data_full.csv",
        help="Full (unbalanced) dataset CSV produced by build_dataset.py",
    )
    parser.add_argument(
        "--hull-threshold", type=float, default=0.05,
        help="Energy-above-hull cutoff for is_stable=1 (eV/atom)",
    )
    args = parser.parse_args()

    api_key = os.environ.get("MP_API_KEY")
    if not api_key:
        sys.exit("MP_API_KEY environment variable is not set.")

    print("=" * 55)
    print("Step 1: Retrain GradientBoosting excluding Co-Bi")
    print("=" * 55)
    pipeline, feature_cols, cs_columns = retrain_without_cobi(
        args.full_data, args.hull_threshold,
    )

    print("=" * 55)
    print("Step 2: Query Co-Bi from Materials Project")
    print("=" * 55)
    df_cobi = query_cobi(api_key)

    print("\n" + "=" * 55)
    print("Step 3: Predict and evaluate")
    print("=" * 55)
    predict_and_evaluate(pipeline, feature_cols, cs_columns, df_cobi, args.hull_threshold)


if __name__ == "__main__":
    main()
