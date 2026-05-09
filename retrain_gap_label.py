#!/usr/bin/env python3
"""
retrain_gap_label.py

Retrains the GradientBoosting stability classifier using a gap-label scheme
that discards entries in the ambiguous 0.035–0.075 eV/atom hull-distance zone,
then re-runs the Ni-Sb holdout validation and compares results to the
original hard-threshold run (validation_NiSb.csv).

Labeling scheme:
  clear stable   : energy_above_hull <  0.035  →  is_stable = 1
  ambiguous      : 0.035 <= energy_above_hull <= 0.075  →  DROPPED
  clear unstable : energy_above_hull >  0.075  →  is_stable = 0

Usage:
    python retrain_gap_label.py
    python retrain_gap_label.py --lower 0.035 --upper 0.075
    python retrain_gap_label.py --output-dir models
"""

import argparse
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from joblib import dump
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

_ORDERING_MAP = {"NM": 0, "FM": 1, "FiM": 2, "AFM": 3}
_ID_COLS      = {"material_id", "formula_pretty", "chemsys"}
_LEAK_COLS    = {"energy_above_hull", "is_stable"}
_ALL_DROP     = _ID_COLS | _LEAK_COLS

_SUMMARY_FIELDS = [
    "material_id", "formula_pretty", "chemsys", "symmetry",
    "nsites", "volume", "density", "formation_energy_per_atom",
    "energy_above_hull", "band_gap", "total_magnetization", "ordering",
]


# ── gap-label training ────────────────────────────────────────────────────────

def retrain_gap_model(
    full_csv: str,
    lower: float,
    upper: float,
) -> tuple[Pipeline, list[str], list[str], dict]:
    """
    Load full dataset, remove Ni-Sb, drop ambiguous zone, rebalance, retrain.

    Returns (pipeline, feature_cols, cs_columns, stats_dict).
    """
    df = pd.read_csv(full_csv)

    # remove Ni-Sb — must stay as a clean holdout
    nisb_mask = df["chemsys"].str.contains("Ni", na=False) & df["chemsys"].str.contains("Sb", na=False)
    df = df[~nisb_mask].copy()

    # drop ambiguous zone
    amb_mask  = (df["energy_above_hull"] >= lower) & (df["energy_above_hull"] <= upper)
    n_amb     = amb_mask.sum()
    df_clean  = df[~amb_mask].copy()

    # relabel with the gap-label boundary
    df_clean["is_stable"] = (df_clean["energy_above_hull"] < lower).astype(int)

    counts = df_clean["is_stable"].value_counts()
    n_stable   = int(counts.get(1, 0))
    n_unstable = int(counts.get(0, 0))

    print(f"Full training pool (Ni-Sb removed) : {len(df)}")
    print(f"  Ambiguous zone dropped ({lower}–{upper} eV/atom) : {n_amb}")
    print(f"  Clear stable   (< {lower})  : {n_stable}")
    print(f"  Clear unstable (> {upper}) : {n_unstable}")

    # rebalance by undersampling majority
    n_min = min(n_stable, n_unstable)
    df_bal = pd.concat(
        [g.sample(n=n_min, random_state=42) for _, g in df_clean.groupby("is_stable")],
        ignore_index=True,
    )
    print(f"  After rebalance : {len(df_bal)}  ({n_min} stable / {n_min} unstable)")

    cs_columns  = [c for c in df_bal.columns if c.startswith("cs_")]
    feature_cols = [c for c in df_bal.columns if c not in _ALL_DROP]
    X = df_bal[feature_cols].values.astype(float)
    y = df_bal["is_stable"].values.astype(int)

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", GradientBoostingClassifier(
            n_estimators=200, max_depth=3,
            learning_rate=0.05, subsample=0.8,
            random_state=42,
        )),
    ])
    pipeline.fit(X, y)
    print(f"  Retrained on {len(X)} samples, {len(feature_cols)} features.\n")

    stats = {
        "n_train": len(X),
        "n_stable_train": n_min,
        "n_unstable_train": n_min,
        "n_ambiguous_dropped": n_amb,
    }
    return pipeline, feature_cols, cs_columns, stats


# ── query Ni-Sb ───────────────────────────────────────────────────────────────

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
        sym      = doc.symmetry
        ordering = getattr(doc, "ordering", None)
        records.append({
            "material_id":               str(doc.material_id),
            "formula_pretty":            doc.formula_pretty,
            "chemsys":                   doc.chemsys,
            "crystal_system":            _enum_str(getattr(sym, "crystal_system", None)) if sym else None,
            "spacegroup_number":         getattr(sym, "number", None) if sym else None,
            "nsites":                    doc.nsites,
            "volume":                    doc.volume,
            "density":                   doc.density,
            "formation_energy_per_atom": doc.formation_energy_per_atom,
            "energy_above_hull":         doc.energy_above_hull,
            "band_gap":                  doc.band_gap,
            "total_magnetization":       doc.total_magnetization,
            "magnetic_ordering":         _enum_str(ordering),
        })

    return pd.DataFrame(records)


def _enum_str(val) -> str | None:
    if val is None:
        return None
    return val.value if hasattr(val, "value") else str(val)


# ── feature engineering for test data ────────────────────────────────────────

def engineer_test(
    df: pd.DataFrame,
    feat_cols: list[str],
    cs_columns: list[str],
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Apply identical feature engineering as training (no label logic here —
    caller handles gap-label filtering before calling this).
    """
    df = df.copy()
    df = df.dropna(subset=["formation_energy_per_atom", "energy_above_hull"])
    df = df[df["nsites"].notna() & (df["nsites"] > 0)]

    df["volume_per_atom"]        = df["volume"] / df["nsites"]
    df["composition_A_fraction"] = df.apply(_first_element_fraction, axis=1)
    df["band_gap"]               = df["band_gap"].fillna(-1.0)
    df["total_magnetization"]    = df["total_magnetization"].fillna(0.0)
    df["magnetic_ordering"] = (
        df["magnetic_ordering"]
        .map(lambda x: _ORDERING_MAP.get(str(x).strip(), -1) if pd.notna(x) else -1)
        .astype(int)
    )

    df["crystal_system"] = df["crystal_system"].fillna("Unknown")
    cs_dummies = pd.get_dummies(df["crystal_system"], prefix="cs").astype(int)
    df = pd.concat([df.drop(columns=["crystal_system"]), cs_dummies], axis=1)

    for col in cs_columns:
        if col not in df.columns:
            df[col] = 0
    extra = [c for c in df.columns if c.startswith("cs_") and c not in cs_columns]
    if extra:
        df = df.drop(columns=extra)
    for col in feat_cols:
        if col not in df.columns:
            df[col] = 0

    X = df[feat_cols].values.astype(float)
    return df, X


def _first_element_fraction(row: pd.Series) -> float:
    from pymatgen.core import Composition
    try:
        first_el = sorted(row["chemsys"].split("-"))[0]
        return float(Composition(row["formula_pretty"]).get_atomic_fraction(first_el))
    except Exception:
        return np.nan


# ── validate on Ni-Sb ────────────────────────────────────────────────────────

def validate_nisb(
    pipeline: Pipeline,
    feat_cols: list[str],
    cs_columns: list[str],
    df_raw: pd.DataFrame,
    lower: float,
    upper: float,
    confidence_thresh: float,
    output_path: Path,
) -> dict:
    """
    Apply gap-label filtering to Ni-Sb, engineer features, predict, report.

    Returns a metrics dict for comparison.
    """
    # split into kept / dropped
    amb_mask = (df_raw["energy_above_hull"] >= lower) & (df_raw["energy_above_hull"] <= upper)
    df_dropped = df_raw[amb_mask].copy()
    df_eval    = df_raw[~amb_mask].copy()

    print(f"Ni-Sb entries queried       : {len(df_raw)}")
    if not df_dropped.empty:
        print(f"  Dropped as ambiguous ({lower}–{upper})  : {len(df_dropped)}")
        for _, r in df_dropped.iterrows():
            print(f"    {r['material_id']}  {r['formula_pretty']:<10}  e_hull={r['energy_above_hull']:.4f}")
    print(f"  Kept for evaluation         : {len(df_eval)}")

    if df_eval.empty:
        sys.exit("No Ni-Sb entries remain after ambiguous-zone filtering.")

    # apply gap label to the kept entries
    df_eval["is_stable"] = (df_eval["energy_above_hull"] < lower).astype(int)

    df_eng, X = engineer_test(df_eval, feat_cols, cs_columns)
    y_true = df_eval["is_stable"].values.astype(int)

    y_pred = pipeline.predict(X)
    y_prob = pipeline.predict_proba(X)[:, 1]

    # per-entry table
    results = df_eval[["material_id", "formula_pretty", "energy_above_hull"]].copy().reset_index(drop=True)
    results["true_label"]            = np.where(y_true == 1, "stable", "unstable")
    results["predicted_label"]       = np.where(y_pred == 1, "stable", "unstable")
    results["predicted_probability"] = np.round(y_prob, 4)
    results["correct"]               = (y_true == y_pred)
    results["in_ambiguous_zone"]     = False

    # also record the dropped entries in the CSV (flagged separately)
    if not df_dropped.empty:
        dropped_rows = df_dropped[["material_id", "formula_pretty", "energy_above_hull"]].copy()
        dropped_rows["true_label"]            = "ambiguous"
        dropped_rows["predicted_label"]       = "dropped"
        dropped_rows["predicted_probability"] = np.nan
        dropped_rows["correct"]               = np.nan
        dropped_rows["in_ambiguous_zone"]     = True
        results = pd.concat([results, dropped_rows], ignore_index=True)

    results.to_csv(output_path, index=False)

    # metrics (on kept entries only)
    n          = len(y_true)
    n_stable   = int(y_true.sum())
    n_unstable = n - n_stable
    acc        = accuracy_score(y_true, y_pred)
    n_correct  = int(acc * n)

    print(f"\nPer-entry predictions (evaluated subset):")
    print("-" * 80)
    header = (f"  {'material_id':<13} {'formula':<10} {'e_hull':>10}  "
              f"{'true':>10}  {'pred':>10}  {'p(stable)':>10}")
    print(header)
    print("-" * 80)
    for _, row in results[~results["in_ambiguous_zone"]].iterrows():
        flag = "" if row["correct"] else "<-- WRONG"
        print(
            f"  {row['material_id']:<13} {row['formula_pretty']:<10} "
            f"{row['energy_above_hull']:>10.4f}  "
            f"{row['true_label']:>10}  {row['predicted_label']:>10}  "
            f"{row['predicted_probability']:>10.3f}  {flag}"
        )

    print(f"\n{'=' * 50}")
    print("VALIDATION METRICS  --  gap-label model")
    print(f"{'=' * 50}")
    print(f"Entries evaluated  : {n}  (of {len(df_raw)} queried; {len(df_dropped)} ambiguous dropped)")
    print(f"  stable   : {n_stable}")
    print(f"  unstable : {n_unstable}")
    print(f"\nAccuracy  : {acc:.4f}  ({n_correct}/{n})")

    if len(np.unique(y_true)) > 1:
        auc = roc_auc_score(y_true, y_prob)
        print(f"ROC-AUC   : {auc:.4f}")
    else:
        auc = float("nan")
        print("ROC-AUC   : n/a (only one class in evaluated subset)")

    print(f"\nClassification report:")
    present = sorted(np.unique(np.concatenate([y_true, y_pred])))
    label_names = {0: "unstable", 1: "stable"}
    print(classification_report(y_true, y_pred,
                                labels=present,
                                target_names=[label_names[l] for l in present],
                                zero_division=0))

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    print("Confusion matrix (rows=true, cols=predicted):")
    print(f"                 pred:unstable  pred:stable")
    print(f"  true:unstable       {tn:>5}          {fp:>5}")
    print(f"  true:stable         {fn:>5}          {tp:>5}")

    # confident errors
    kept = results[~results["in_ambiguous_zone"]].copy()
    wrong_mask = ~kept["correct"].astype(bool)
    confident_wrong = kept[
        wrong_mask & kept["predicted_probability"].apply(
            lambda p: p > confidence_thresh or (1 - p) > confidence_thresh
        )
    ]
    n_confident = len(confident_wrong)
    print(f"\nConfident errors (p > {confidence_thresh}, wrong label): {n_confident}")
    if not confident_wrong.empty:
        for _, r in confident_wrong.iterrows():
            p = r["predicted_probability"]
            conf = p if r["predicted_label"] == "stable" else 1 - p
            print(f"  {r['material_id']}  {r['formula_pretty']:<10}  "
                  f"e_hull={r['energy_above_hull']:.4f}  "
                  f"true={r['true_label']}  pred={r['predicted_label']}  conf={conf:.3f}")

    print(f"\nPer-entry results (including dropped) saved -> {output_path.resolve()}")

    return {
        "n_queried": len(df_raw), "n_evaluated": n, "n_dropped": len(df_dropped),
        "n_stable": n_stable, "n_unstable": n_unstable,
        "accuracy": acc, "auc": auc,
        "n_correct": n_correct, "n_errors": n - n_correct,
        "n_confident_errors": n_confident,
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
    }


# ── side-by-side comparison ───────────────────────────────────────────────────

def compare(prev_csv: str, new_metrics: dict, lower: float, upper: float) -> None:
    """Load previous validation_NiSb.csv and print a side-by-side comparison."""
    prev_df    = pd.read_csv(prev_csv)
    prev_true  = (prev_df["true_label"] == "stable").astype(int).values
    prev_pred  = (prev_df["predicted_label"] == "stable").astype(int).values
    prev_prob  = prev_df["predicted_probability"].values
    prev_acc   = accuracy_score(prev_true, prev_pred)
    prev_n     = len(prev_df)
    prev_auc   = roc_auc_score(prev_true, prev_prob) if len(np.unique(prev_true)) > 1 else float("nan")
    prev_errors         = int((prev_true != prev_pred).sum())
    prev_cm             = confusion_matrix(prev_true, prev_pred, labels=[0, 1])
    p_tn, p_fp, p_fn, p_tp = prev_cm.ravel()

    prev_conf_wrong = prev_df[
        (prev_df["true_label"] != prev_df["predicted_label"]) &
        prev_df["predicted_probability"].apply(
            lambda p: p > 0.75 or (1 - p) > 0.75
        )
    ]
    prev_confident = len(prev_conf_wrong)

    w = 26
    sep = "-" * 60
    print(f"\n{'=' * 60}")
    print("COMPARISON  --  original vs. gap-label model")
    print(f"{'=' * 60}")
    print(f"{'Metric':<{w}} {'Original (0.05 cutoff)':>16}  {'Gap-label (drop 0.035-0.075)':>20}")
    print(sep)
    print(f"{'Label scheme':<{w}} {'hard threshold':>16}  {'gap-label':>20}")
    print(f"{'Ni-Sb entries queried':<{w}} {prev_n:>16}  {new_metrics['n_queried']:>20}")
    print(f"{'  ambiguous (dropped)':<{w}} {'0':>16}  {new_metrics['n_dropped']:>20}")
    print(f"{'  evaluated':<{w}} {prev_n:>16}  {new_metrics['n_evaluated']:>20}")
    print(f"{'  stable':<{w}} {int(prev_true.sum()):>16}  {new_metrics['n_stable']:>20}")
    print(f"{'  unstable':<{w}} {int((1-prev_true).sum()):>16}  {new_metrics['n_unstable']:>20}")
    print(sep)
    print(f"{'Accuracy':<{w}} {prev_acc:>16.4f}  {new_metrics['accuracy']:>20.4f}")

    prev_auc_str = f"{prev_auc:.4f}" if not np.isnan(prev_auc) else "n/a"
    new_auc_str  = f"{new_metrics['auc']:.4f}" if not np.isnan(new_metrics['auc']) else "n/a (1 class)"
    print(f"{'ROC-AUC':<{w}} {prev_auc_str:>16}  {new_auc_str:>20}")
    print(sep)
    print(f"{'Confusion matrix':}")
    print(f"  {'TN':<{w-2}} {p_tn:>16}  {new_metrics['tn']:>20}")
    print(f"  {'FP':<{w-2}} {p_fp:>16}  {new_metrics['fp']:>20}")
    print(f"  {'FN':<{w-2}} {p_fn:>16}  {new_metrics['fn']:>20}")
    print(f"  {'TP':<{w-2}} {p_tp:>16}  {new_metrics['tp']:>20}")
    print(sep)
    print(f"{'Total errors':<{w}} {prev_errors:>16}  {new_metrics['n_errors']:>20}")
    print(f"{'Confident errors (>0.75)':<{w}} {prev_confident:>16}  {new_metrics['n_confident_errors']:>20}")
    print(f"{'=' * 60}")


# ── orchestration ─────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Retrain with gap-label scheme and compare Ni-Sb validation results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--full-data",  default="data/results/training_data_full.csv")
    parser.add_argument("--prev-validation", default="validation_NiSb.csv",
                        help="Per-entry CSV from the original validate_nisb.py run")
    parser.add_argument("--lower", type=float, default=0.035,
                        help="Lower bound of ambiguous zone (eV/atom, inclusive)")
    parser.add_argument("--upper", type=float, default=0.075,
                        help="Upper bound of ambiguous zone (eV/atom, inclusive)")
    parser.add_argument("--output-dir", default=".",
                        help="Directory to save new model and validation CSV")
    parser.add_argument("--confidence-threshold", type=float, default=0.75)
    args = parser.parse_args()

    api_key = os.environ.get("MP_API_KEY")
    if not api_key:
        sys.exit("MP_API_KEY environment variable is not set.")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 55)
    print("Step 1: Retrain with gap-label scheme")
    print("=" * 55)
    pipeline, feat_cols, cs_columns, train_stats = retrain_gap_model(
        args.full_data, args.lower, args.upper,
    )
    model_path = out_dir / "stability_classifier_gap_label.joblib"
    dump({"model": pipeline, "model_name": "GradientBoosting_gap_label",
          "lower": args.lower, "upper": args.upper}, model_path)
    print(f"Gap-label model saved -> {model_path.resolve()}")

    print("\n" + "=" * 55)
    print("Step 2: Query Ni-Sb from Materials Project")
    print("=" * 55)
    df_raw = query_nisb(api_key)

    print("\n" + "=" * 55)
    print("Step 3: Validate on Ni-Sb holdout")
    print("=" * 55)
    new_metrics = validate_nisb(
        pipeline, feat_cols, cs_columns, df_raw,
        lower=args.lower, upper=args.upper,
        confidence_thresh=args.confidence_threshold,
        output_path=out_dir / "validation_NiSb_gap_label.csv",
    )

    print("\n" + "=" * 55)
    print("Step 4: Compare to original run")
    print("=" * 55)
    compare(args.prev_validation, new_metrics, args.lower, args.upper)


if __name__ == "__main__":
    main()
