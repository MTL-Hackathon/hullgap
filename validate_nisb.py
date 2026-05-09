#!/usr/bin/env python3
"""
validate_nisb.py

Validates the trained GradientBoosting stability classifier against Ni-Sb
binary compounds from Materials Project — a chemical system never seen
during training.

Steps:
  1. Load the saved model from models/stability_classifier.joblib, or retrain
     from scratch on training_data_balanced.csv if the file is missing.
  2. Derive the exact feature columns from the training CSV.
  3. Confirm zero Ni-Sb overlap with the training set.
  4. Query Materials Project for all Ni-Sb binary entries.
  5. Apply identical preprocessing (feature engineering, encoding, imputation).
  6. Predict stability and produce a full validation report.
  7. Save per-entry results to validation_NiSb.csv.
  8. Flag entries where the model is confidently wrong (p > 0.75, wrong label).

Usage:
    python validate_nisb.py
    python validate_nisb.py --model models/stability_classifier.joblib
    python validate_nisb.py --hull-threshold 0.05 --output-dir .
"""

import argparse
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from joblib import dump, load as jload
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
load_dotenv()

_ORDERING_MAP   = {"NM": 0, "FM": 1, "FiM": 2, "AFM": 3}
_ID_COLS        = {"material_id", "formula_pretty", "chemsys"}
_LEAK_DROP_COLS = {"energy_above_hull", "is_stable"}
_ALL_DROP       = _ID_COLS | _LEAK_DROP_COLS

_SUMMARY_FIELDS = [
    "material_id", "formula_pretty", "chemsys", "symmetry",
    "nsites", "volume", "density", "formation_energy_per_atom",
    "energy_above_hull", "band_gap", "total_magnetization", "ordering",
]


# ── step 1: load model (or retrain fallback) ──────────────────────────────────

def load_or_retrain(
    model_path: str,
    data_path: str,
    hull_threshold: float,
) -> Pipeline:
    """
    Load the saved joblib pipeline. If the file is missing, retrain
    GradientBoosting from data_path with the original hyperparameters.
    """
    p = Path(model_path)
    if p.exists():
        artifact = jload(p)
        pipeline  = artifact["model"]
        name      = artifact.get("model_name", "unknown")
        print(f"Loaded model  : {p}  ({name})")
        return pipeline

    print(f"Model file not found at {p} -- retraining from {data_path}")
    df = pd.read_csv(data_path)
    df["is_stable"] = (df["energy_above_hull"] <= hull_threshold).astype(int)
    feat_cols = [c for c in df.columns if c not in _ALL_DROP]
    X = df[feat_cols].values.astype(float)
    y = df["is_stable"].values.astype(int)

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", GradientBoostingClassifier(
            n_estimators=200, max_depth=3,
            learning_rate=0.05, subsample=0.8,
            random_state=42,
        )),
    ])
    pipeline.fit(X, y)
    print(f"Retrained on {len(X)} samples, {len(feat_cols)} features.")
    return pipeline


# ── step 2 & 3: derive feature columns and verify no overlap ─────────────────

def get_training_features(data_path: str) -> tuple[list[str], list[str]]:
    """
    Return (feature_cols, cs_columns) derived from the training CSV.
    Also prints confirmation that Ni-Sb has zero rows in the training set.
    """
    df = pd.read_csv(data_path)

    # confirm Ni-Sb absence
    mask = df["chemsys"].str.contains("Ni", na=False) & df["chemsys"].str.contains("Sb", na=False)
    n_overlap = mask.sum()
    if n_overlap == 0:
        print("Overlap check : 0 Ni-Sb rows in training data -- clean holdout confirmed.")
    else:
        print(f"WARNING: {n_overlap} Ni-Sb rows found in training data!")
        print(df[mask][["material_id", "formula_pretty", "chemsys"]].to_string(index=False))

    feat_cols = [c for c in df.columns if c not in _ALL_DROP]
    cs_columns = [c for c in feat_cols if c.startswith("cs_")]
    print(f"Training features ({len(feat_cols)}): {feat_cols}")
    return feat_cols, cs_columns


# ── step 4: query Materials Project ──────────────────────────────────────────

def query_nisb(api_key: str) -> pd.DataFrame:
    """Fetch all Ni-Sb entries from Materials Project."""
    try:
        from mp_api.client import MPRester
    except ImportError:
        sys.exit("mp-api not installed -- run: pip install mp-api")

    with MPRester(api_key) as mpr:
        docs = mpr.materials.summary.search(
            chemsys=["Ni-Sb"],
            fields=_SUMMARY_FIELDS,
        )

    if not docs:
        sys.exit("No Ni-Sb entries returned from Materials Project.")

    records = []
    for doc in docs:
        sym = doc.symmetry
        ordering = getattr(doc, "ordering", None)
        records.append({
            "material_id":              str(doc.material_id),
            "formula_pretty":           doc.formula_pretty,
            "chemsys":                  doc.chemsys,
            "crystal_system":           _enum_str(getattr(sym, "crystal_system", None)) if sym else None,
            "spacegroup_number":        getattr(sym, "number", None) if sym else None,
            "nsites":                   doc.nsites,
            "volume":                   doc.volume,
            "density":                  doc.density,
            "formation_energy_per_atom": doc.formation_energy_per_atom,
            "energy_above_hull":        doc.energy_above_hull,
            "band_gap":                 doc.band_gap,
            "total_magnetization":      doc.total_magnetization,
            "magnetic_ordering":        _enum_str(ordering),
        })

    df = pd.DataFrame(records)
    print(f"\nQueried {len(df)} Ni-Sb entries from Materials Project.")
    return df


def _enum_str(val) -> str | None:
    if val is None:
        return None
    return val.value if hasattr(val, "value") else str(val)


# ── step 5: feature engineering ──────────────────────────────────────────────

def engineer_test(
    df: pd.DataFrame,
    feat_cols: list[str],
    cs_columns: list[str],
    hull_threshold: float,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Apply identical preprocessing to the raw MP query result.

    Returns (annotated_df, X, y_true).
    annotated_df retains material_id / formula / energy for the report.
    """
    df = df.copy()

    # drop rows missing critical fields
    n_before = len(df)
    df = df.dropna(subset=["formation_energy_per_atom", "energy_above_hull"])
    df = df[df["nsites"].notna() & (df["nsites"] > 0)]
    if len(df) < n_before:
        print(f"Dropped {n_before - len(df)} rows with null energies or zero nsites.")

    # stability label
    df["is_stable"] = (df["energy_above_hull"] <= hull_threshold).astype(int)

    # derived features
    df["volume_per_atom"]       = df["volume"] / df["nsites"]
    df["composition_A_fraction"] = df.apply(_first_element_fraction, axis=1)

    # fill missing
    df["band_gap"]           = df["band_gap"].fillna(-1.0)
    df["total_magnetization"] = df["total_magnetization"].fillna(0.0)

    # label-encode magnetic ordering
    df["magnetic_ordering"] = (
        df["magnetic_ordering"]
        .map(lambda x: _ORDERING_MAP.get(str(x).strip(), -1) if pd.notna(x) else -1)
        .astype(int)
    )

    # one-hot encode crystal_system, then align to training columns
    df["crystal_system"] = df["crystal_system"].fillna("Unknown")
    cs_dummies = pd.get_dummies(df["crystal_system"], prefix="cs").astype(int)
    df = pd.concat([df.drop(columns=["crystal_system"]), cs_dummies], axis=1)

    for col in cs_columns:       # add any training cs column absent in this batch
        if col not in df.columns:
            df[col] = 0
    extra_cs = [c for c in df.columns if c.startswith("cs_") and c not in cs_columns]
    if extra_cs:
        print(f"Note: dropping {len(extra_cs)} unseen crystal-system columns: {extra_cs}")
        df = df.drop(columns=extra_cs)

    # align to training feature order, filling any remaining gaps
    for col in feat_cols:
        if col not in df.columns:
            df[col] = 0
    X      = df[feat_cols].values.astype(float)
    y_true = df["is_stable"].values.astype(int)

    return df, X, y_true


def _first_element_fraction(row: pd.Series) -> float:
    from pymatgen.core import Composition
    try:
        first_el = sorted(row["chemsys"].split("-"))[0]
        return float(Composition(row["formula_pretty"]).get_atomic_fraction(first_el))
    except Exception:
        return np.nan


# ── step 6 & 7: report + save ─────────────────────────────────────────────────

def report_and_save(
    df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    output_path: Path,
    confidence_threshold: float = 0.75,
) -> None:
    """Print full validation report, save CSV, flag confident errors."""

    # ── per-entry table ───────────────────────────────────────────────────
    results = df[["material_id", "formula_pretty", "energy_above_hull"]].copy().reset_index(drop=True)
    results["true_label"]            = np.where(y_true == 1, "stable", "unstable")
    results["predicted_label"]       = np.where(y_pred == 1, "stable", "unstable")
    results["predicted_probability"] = np.round(y_prob, 4)
    results["correct"]               = y_true == y_pred

    print("\nPer-entry predictions:")
    print("-" * 90)
    header = (f"{'material_id':<14} {'formula':<10} {'e_hull':>10}  "
              f"{'true':>10}  {'pred':>10}  {'p(stable)':>10}  {'':>5}")
    print(header)
    print("-" * 90)
    for _, row in results.iterrows():
        flag = "" if row["correct"] else "<-- WRONG"
        print(
            f"  {row['material_id']:<12} {row['formula_pretty']:<10} "
            f"{row['energy_above_hull']:>10.4f}  "
            f"{row['true_label']:>10}  {row['predicted_label']:>10}  "
            f"{row['predicted_probability']:>10.3f}  {flag}"
        )

    # ── aggregate metrics ─────────────────────────────────────────────────
    n = len(y_true)
    n_stable   = int(y_true.sum())
    n_unstable = n - n_stable
    acc = accuracy_score(y_true, y_pred)

    print(f"\n{'=' * 50}")
    print("VALIDATION REPORT  --  Ni-Sb holdout")
    print(f"{'=' * 50}")
    print(f"Total Ni-Sb entries  : {n}")
    print(f"  stable   (is_stable=1): {n_stable}")
    print(f"  unstable (is_stable=0): {n_unstable}")
    print(f"\nAccuracy : {acc:.4f}  ({int(acc * n)}/{n} correct)")

    if len(np.unique(y_true)) > 1:
        auc = roc_auc_score(y_true, y_prob)
        print(f"ROC-AUC  : {auc:.4f}")
    else:
        print("ROC-AUC  : n/a (only one class present in Ni-Sb holdout)")

    print(f"\nClassification report:")
    print(classification_report(y_true, y_pred, target_names=["unstable", "stable"]))

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    print("Confusion matrix (rows=true, cols=predicted):")
    print(f"                 pred:unstable  pred:stable")
    print(f"  true:unstable       {tn:>5}          {fp:>5}")
    print(f"  true:stable         {fn:>5}          {tp:>5}")

    # ── confident errors ──────────────────────────────────────────────────
    wrong_mask = ~results["correct"]
    confident_wrong = results[
        wrong_mask & (results["predicted_probability"].apply(
            lambda p: p > confidence_threshold or (1 - p) > confidence_threshold
        ))
    ]

    print(f"\n{'=' * 50}")
    print(f"Confident errors  (p > {confidence_threshold} but wrong label)")
    print(f"{'=' * 50}")
    if confident_wrong.empty:
        print("  None -- no confidently wrong predictions.")
    else:
        print(f"  {len(confident_wrong)} entry/entries:\n")
        for _, row in confident_wrong.iterrows():
            p = row["predicted_probability"]
            confidence = p if row["predicted_label"] == "stable" else 1 - p
            print(
                f"  {row['material_id']}  {row['formula_pretty']:<10}  "
                f"e_hull={row['energy_above_hull']:.4f}  "
                f"true={row['true_label']}  pred={row['predicted_label']}  "
                f"confidence={confidence:.3f}"
            )

    # ── save CSV ──────────────────────────────────────────────────────────
    results.to_csv(output_path, index=False)
    print(f"\nPer-entry results saved -> {output_path.resolve()}")


# ── orchestration ─────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate the stability classifier on held-out Ni-Sb compounds.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model", default="models/stability_classifier.joblib",
        help="Path to saved joblib model (retrained from --data if missing)",
    )
    parser.add_argument(
        "--data", default="data/results/training_data_balanced.csv",
        help="Training CSV used to derive feature columns (and retrain if needed)",
    )
    parser.add_argument(
        "--hull-threshold", type=float, default=0.05,
        help="Energy-above-hull cutoff for is_stable=1 (eV/atom)",
    )
    parser.add_argument(
        "--output-dir", default=".",
        help="Directory to write validation_NiSb.csv",
    )
    parser.add_argument(
        "--confidence-threshold", type=float, default=0.75,
        help="Probability threshold for flagging confident errors",
    )
    args = parser.parse_args()

    api_key = os.environ.get("MP_API_KEY")
    if not api_key:
        sys.exit("MP_API_KEY environment variable is not set.")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 55)
    print("Step 1: Load model")
    print("=" * 55)
    pipeline = load_or_retrain(args.model, args.data, args.hull_threshold)

    print("\n" + "=" * 55)
    print("Step 2+3: Training features + overlap check")
    print("=" * 55)
    feat_cols, cs_columns = get_training_features(args.data)

    print("\n" + "=" * 55)
    print("Step 4: Query Ni-Sb from Materials Project")
    print("=" * 55)
    df_raw = query_nisb(api_key)

    print("\n" + "=" * 55)
    print("Step 5: Feature engineering")
    print("=" * 55)
    df_eng, X, y_true = engineer_test(
        df_raw, feat_cols, cs_columns, args.hull_threshold,
    )

    print("\n" + "=" * 55)
    print("Step 6+7+8: Predictions and report")
    print("=" * 55)
    y_pred = pipeline.predict(X)
    y_prob = pipeline.predict_proba(X)[:, 1]

    report_and_save(
        df_eng, y_true, y_pred, y_prob,
        output_path=out_dir / "validation_NiSb.csv",
        confidence_threshold=args.confidence_threshold,
    )


if __name__ == "__main__":
    main()
