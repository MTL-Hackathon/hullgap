#!/usr/bin/env python3
"""
evaluate_fesb_mnbi.py

Queries Fe-Sb and Mn-Bi from Materials Project, combines them into a single
holdout set, and evaluates both classifiers:

  - Original model  : hard 0.05 eV/atom threshold
  - Gap-label model : ambiguous zone 0.035–0.075 eV/atom dropped in training

Reports for each model:
  * Total entries, class distribution
  * Accuracy, ROC-AUC, precision, recall, F1
  * Confusion matrix
  * Confident errors (p > 0.75, wrong label)

For the gap-label model additionally reports:
  * How many entries fall in the ambiguous zone
  * Accuracy excluding ambiguous entries
  * Accuracy including ambiguous entries (treating all ambiguous predictions as wrong)

NOTE: Mn-Bi (stored as Bi-Mn in MP) WAS included in the original training
query.  Mn-Bi rows are flagged throughout; Fe-Sb-only metrics are reported
separately as the truly clean holdout.

Usage:
    python evaluate_fesb_mnbi.py
"""

import argparse
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from joblib import load as jload
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

warnings.filterwarnings("ignore")
load_dotenv()

_ORDERING_MAP   = {"NM": 0, "FM": 1, "FiM": 2, "AFM": 3}
_ALL_DROP       = {"material_id", "formula_pretty", "chemsys",
                   "energy_above_hull", "is_stable"}
_SUMMARY_FIELDS = [
    "material_id", "formula_pretty", "chemsys", "symmetry",
    "nsites", "volume", "density", "formation_energy_per_atom",
    "energy_above_hull", "band_gap", "total_magnetization", "ordering",
]

ORIG_THRESHOLD = 0.05
GAP_LOWER      = 0.035
GAP_UPPER      = 0.075


# ── load models & feature metadata ───────────────────────────────────────────

def load_models(model_dir: str, train_csv: str) -> tuple:
    """
    Load both saved pipelines and derive training feature columns.
    Returns (orig_pipeline, gap_pipeline, feat_cols, cs_columns).
    """
    orig_art = jload(Path(model_dir) / "stability_classifier.joblib")
    gap_art  = jload(Path(model_dir) / "stability_classifier_gap_label.joblib")
    orig_pipe = orig_art["model"]
    gap_pipe  = gap_art["model"]
    print(f"Loaded: {orig_art['model_name']} (original)")
    print(f"Loaded: {gap_art['model_name']}  (lower={gap_art['lower']}, upper={gap_art['upper']})")

    df_train  = pd.read_csv(train_csv)
    feat_cols = [c for c in df_train.columns if c not in _ALL_DROP]
    cs_columns = [c for c in feat_cols if c.startswith("cs_")]
    print(f"Feature columns ({len(feat_cols)}): {feat_cols}\n")
    return orig_pipe, gap_pipe, feat_cols, cs_columns


# ── training overlap check ────────────────────────────────────────────────────

def check_overlap(train_csv: str, combined_df: pd.DataFrame) -> dict[str, int]:
    """
    Check which material_ids in combined_df also appear in the training CSV.
    Prints a clear warning for every system with overlap.
    Returns {chemsys: n_overlap}.
    """
    df_train = pd.read_csv(train_csv)
    train_ids = set(df_train["material_id"])

    overlaps = {}
    for sys in combined_df["chemsys"].unique():
        sys_df  = combined_df[combined_df["chemsys"] == sys]
        n_over  = sys_df["material_id"].isin(train_ids).sum()
        overlaps[sys] = int(n_over)
        if n_over > 0:
            ids = list(sys_df[sys_df["material_id"].isin(train_ids)]["material_id"])
            print(f"  [WARNING] {sys}: {n_over}/{len(sys_df)} entries were in training data!")
            print(f"            IDs: {ids}")
        else:
            print(f"  {sys}: 0/{len(sys_df)} overlap with training -- clean holdout.")
    return overlaps


# ── query MP ─────────────────────────────────────────────────────────────────

def query_systems(api_key: str, systems: list[str]) -> pd.DataFrame:
    """Query MP summary for the given chemical systems and return a raw DataFrame."""
    try:
        from mp_api.client import MPRester
    except ImportError:
        sys.exit("mp-api not installed -- run: pip install mp-api")

    records = []
    with MPRester(api_key) as mpr:
        for chemsys in systems:
            docs = mpr.materials.summary.search(chemsys=[chemsys], fields=_SUMMARY_FIELDS)
            print(f"  {chemsys}: {len(docs)} entries")
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


# ── feature engineering ───────────────────────────────────────────────────────

def engineer(df: pd.DataFrame, feat_cols: list[str], cs_columns: list[str]) -> np.ndarray:
    """Apply identical preprocessing to a raw MP DataFrame; return X matrix."""
    df = df.copy()
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

    return df[feat_cols].values.astype(float)


def _first_element_fraction(row: pd.Series) -> float:
    from pymatgen.core import Composition
    try:
        first_el = sorted(row["chemsys"].split("-"))[0]
        return float(Composition(row["formula_pretty"]).get_atomic_fraction(first_el))
    except Exception:
        return np.nan


# ── evaluation helpers ────────────────────────────────────────────────────────

def _print_metrics_block(
    label: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    df_meta: pd.DataFrame,
    confidence_thresh: float,
    note: str = "",
) -> dict:
    """Print full metrics block; return dict of scalar metrics."""
    n          = len(y_true)
    n_stable   = int(y_true.sum())
    n_unstable = n - n_stable

    print(f"\n  n={n}  (stable={n_stable}, unstable={n_unstable}){note}")

    if n == 0:
        print("  [no entries to evaluate]")
        return {}

    acc = accuracy_score(y_true, y_pred)
    print(f"  Accuracy  : {acc:.4f}  ({int(acc * n)}/{n})")

    if len(np.unique(y_true)) > 1:
        auc = roc_auc_score(y_true, y_prob)
        print(f"  ROC-AUC   : {auc:.4f}")
    else:
        auc = float("nan")
        print("  ROC-AUC   : n/a (single class)")

    present    = sorted(np.unique(np.concatenate([y_true, y_pred])))
    lmap       = {0: "unstable", 1: "stable"}
    print(f"\n  Classification report:")
    report_str = classification_report(
        y_true, y_pred,
        labels=present,
        target_names=[lmap[l] for l in present],
        zero_division=0,
    )
    for line in report_str.splitlines():
        print(f"    {line}")

    cm          = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    print(f"  Confusion matrix (rows=true, cols=predicted):")
    print(f"                   pred:unstable  pred:stable")
    print(f"    true:unstable       {tn:>5}          {fp:>5}")
    print(f"    true:stable         {fn:>5}          {tp:>5}")

    # confident errors
    wrong  = y_true != y_pred
    conf_p = np.maximum(y_prob, 1 - y_prob)  # confidence = distance from 0.5
    conf_wrong = df_meta[(wrong) & (conf_p > confidence_thresh)].copy()
    conf_wrong["p_stable"] = y_prob[(wrong) & (conf_p > confidence_thresh)]
    print(f"\n  Confident errors (p > {confidence_thresh}, wrong label): {len(conf_wrong)}")
    if not conf_wrong.empty:
        for _, r in conf_wrong.iterrows():
            p     = r["p_stable"]
            pred  = "stable" if p > 0.5 else "unstable"
            true  = "stable" if r["is_stable"] == 1 else "unstable"
            conf  = p if pred == "stable" else 1 - p
            print(f"    {r['material_id']:<12} {r['formula_pretty']:<10}  "
                  f"e_hull={r['energy_above_hull']:.4f}  "
                  f"true={true}  pred={pred}  conf={conf:.3f}")

    return {"n": n, "n_stable": n_stable, "n_unstable": n_unstable,
            "accuracy": acc, "auc": auc, "tn": tn, "fp": fp, "fn": fn, "tp": tp,
            "n_confident_errors": len(conf_wrong)}


def _per_entry_table(
    df_meta: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> None:
    """Print a compact per-entry prediction table."""
    print(f"\n  {'material_id':<14} {'formula':<10} {'chemsys':<8} "
          f"{'e_hull':>10}  {'true':>10}  {'pred':>10}  {'p(stable)':>10}  {'':>5}")
    print(f"  {'-'*85}")
    for i, (_, row) in enumerate(df_meta.iterrows()):
        flag = "" if y_true[i] == y_pred[i] else "<-- WRONG"
        tl   = "stable" if y_true[i] == 1 else "unstable"
        pl   = "stable" if y_pred[i] == 1 else "unstable"
        print(f"  {row['material_id']:<14} {row['formula_pretty']:<10} {row['chemsys']:<8} "
              f"{row['energy_above_hull']:>10.4f}  {tl:>10}  {pl:>10}  {y_prob[i]:>10.3f}  {flag}")


# ── original model evaluation ─────────────────────────────────────────────────

def evaluate_original(
    pipeline,
    X: np.ndarray,
    df: pd.DataFrame,
    confidence_thresh: float,
    overlaps: dict,
) -> dict:
    """Evaluate the original 0.05-threshold model on the combined holdout."""
    df = df.copy().reset_index(drop=True)
    df["is_stable"] = (df["energy_above_hull"] <= ORIG_THRESHOLD).astype(int)
    y_true = df["is_stable"].values
    y_pred = pipeline.predict(X)
    y_prob = pipeline.predict_proba(X)[:, 1]

    print(f"\n{'='*62}")
    print(f"ORIGINAL MODEL  (threshold = {ORIG_THRESHOLD} eV/atom)")
    print(f"{'='*62}")

    _per_entry_table(df, y_true, y_pred, y_prob)

    print(f"\n-- Combined (Fe-Sb + Mn-Bi) --")
    metrics_all = _print_metrics_block(
        "combined", y_true, y_pred, y_prob, df, confidence_thresh,
        note="  [WARNING: Mn-Bi in training]" if any(v > 0 for v in overlaps.values()) else "",
    )

    # Fe-Sb only (clean holdout)
    fesb_mask = df["chemsys"] == "Fe-Sb"
    if fesb_mask.sum() > 0:
        print(f"\n-- Fe-Sb only (truly clean holdout) --")
        metrics_fesb = _print_metrics_block(
            "Fe-Sb", y_true[fesb_mask], y_pred[fesb_mask], y_prob[fesb_mask],
            df[fesb_mask].reset_index(drop=True), confidence_thresh,
        )
    else:
        metrics_fesb = {}

    return {"combined": metrics_all, "fesb_only": metrics_fesb}


# ── gap-label model evaluation ────────────────────────────────────────────────

def evaluate_gap_label(
    pipeline,
    X: np.ndarray,
    df: pd.DataFrame,
    confidence_thresh: float,
    overlaps: dict,
) -> dict:
    """
    Evaluate the gap-label model.

    Reports three sub-evaluations:
      1. Excluding ambiguous entries (model's intended operating regime)
      2. Including all entries, ambiguous zone predictions counted as wrong
      3. Fe-Sb only (clean holdout), same two sub-evaluations
    """
    df = df.copy().reset_index(drop=True)
    e  = df["energy_above_hull"].values

    in_amb   = (e >= GAP_LOWER) & (e <= GAP_UPPER)
    n_amb    = int(in_amb.sum())
    y_pred   = pipeline.predict(X)
    y_prob   = pipeline.predict_proba(X)[:, 1]

    # ground truth for "excluded" uses the gap boundary
    y_true_gap  = (e < GAP_LOWER).astype(int)
    # ground truth for "included / as-wrong" uses the standard 0.05 threshold
    y_true_orig = (e <= ORIG_THRESHOLD).astype(int)

    print(f"\n{'='*62}")
    print(f"GAP-LABEL MODEL  (ambiguous zone: {GAP_LOWER}–{GAP_UPPER} eV/atom)")
    print(f"{'='*62}")

    _per_entry_table(df, y_true_orig, y_pred, y_prob)

    print(f"\n  Ambiguous zone entries ({GAP_LOWER}–{GAP_UPPER} eV/atom): {n_amb}")
    if n_amb:
        amb_df = df[in_amb][["material_id", "formula_pretty", "chemsys", "energy_above_hull"]]
        for _, r in amb_df.iterrows():
            print(f"    {r['material_id']:<12} {r['formula_pretty']:<10} {r['chemsys']:<8}  "
                  f"e_hull={r['energy_above_hull']:.4f}")

    # 1. Excluding ambiguous
    keep = ~in_amb
    print(f"\n-- Combined: EXCLUDING ambiguous ({keep.sum()} entries) --")
    df["is_stable"] = y_true_gap
    m_excl = _print_metrics_block(
        "gap_excl", y_true_gap[keep], y_pred[keep], y_prob[keep],
        df[keep].reset_index(drop=True), confidence_thresh,
    )

    # 2. Including all, ambiguous treated as wrong
    n_total   = len(df)
    # correct = non-ambiguous AND prediction matches 0.05-threshold label
    correct_vec = (~in_amb) & (y_pred == y_true_orig)
    acc_incl    = float(correct_vec.mean())
    n_correct   = int(correct_vec.sum())
    print(f"\n-- Combined: INCLUDING ambiguous (treated as wrong) ({n_total} entries) --")
    print(f"\n  n={n_total}  (stable={int(y_true_orig.sum())}, unstable={int((1-y_true_orig).sum())})")
    print(f"  Ambiguous entries (auto-wrong): {n_amb}")
    print(f"  Accuracy (amb=wrong): {acc_incl:.4f}  ({n_correct}/{n_total})")
    # For non-ambiguous entries only, AUC uses the same y_true_orig
    if len(np.unique(y_true_orig)) > 1:
        auc_incl = roc_auc_score(y_true_orig, y_prob)
        print(f"  ROC-AUC (all entries, 0.05 labels): {auc_incl:.4f}")
    else:
        auc_incl = float("nan")
        print("  ROC-AUC: n/a (single class)")

    m_incl = {"n": n_total, "accuracy_amb_as_wrong": acc_incl,
              "n_ambiguous": n_amb, "auc_incl": auc_incl}

    # 3. Fe-Sb only
    fesb = df["chemsys"] == "Fe-Sb"
    if fesb.sum():
        fesb_amb  = in_amb[fesb]
        fesb_keep = fesb & (~in_amb)
        print(f"\n-- Fe-Sb only (clean holdout): EXCLUDING ambiguous ({fesb_keep.sum()} entries) --")
        m_fesb_excl = _print_metrics_block(
            "gap_fesb_excl",
            y_true_gap[fesb_keep], y_pred[fesb_keep], y_prob[fesb_keep],
            df[fesb_keep].reset_index(drop=True), confidence_thresh,
        ) if fesb_keep.sum() > 0 else {}

        print(f"\n-- Fe-Sb only: INCLUDING ambiguous (treated as wrong) ({fesb.sum()} entries) --")
        correct_fesb = (~fesb_amb) & (y_pred[fesb] == y_true_orig[fesb])
        acc_fesb     = float(correct_fesb.mean())
        print(f"\n  n={fesb.sum()}  ambiguous={fesb_amb.sum()}")
        print(f"  Accuracy (amb=wrong): {acc_fesb:.4f}  ({int(correct_fesb.sum())}/{fesb.sum()})")
    else:
        m_fesb_excl = {}

    return {"excl": m_excl, "incl": m_incl}


# ── comparison table ──────────────────────────────────────────────────────────

def comparison_table(orig_metrics: dict, gap_metrics: dict) -> None:
    w   = 36
    sep = "-" * 70
    print(f"\n{'='*70}")
    print("SIDE-BY-SIDE COMPARISON  --  combined Fe-Sb + Mn-Bi holdout")
    print(f"{'='*70}")
    print(f"{'Metric':<{w}} {'Original (0.05)':>15}  {'Gap excl.':>10}  {'Gap incl.(amb=wrong)':>20}")
    print(sep)

    o   = orig_metrics["combined"]
    ge  = gap_metrics["excl"]
    gi  = gap_metrics["incl"]

    def _fmt(v, fmt=".4f"):
        return "n/a" if (isinstance(v, float) and np.isnan(v)) else format(v, fmt)

    print(f"{'Entries evaluated':<{w}} {o['n']:>15}  {ge.get('n', 0):>10}  {gi['n']:>20}")
    print(f"{'  of which ambiguous':<{w}} {'0':>15}  {gi['n_ambiguous']:>10}  {gi['n_ambiguous']:>20}")
    print(f"{'  stable':<{w}} {o['n_stable']:>15}  {ge.get('n_stable',0):>10}  {'(0.05 label)':>20}")
    print(f"{'  unstable':<{w}} {o['n_unstable']:>15}  {ge.get('n_unstable',0):>10}  {'':>20}")
    print(sep)
    print(f"{'Accuracy':<{w}} {_fmt(o['accuracy']):>15}  {_fmt(ge.get('accuracy', float('nan'))):>10}  {_fmt(gi['accuracy_amb_as_wrong']):>20}")
    print(f"{'ROC-AUC':<{w}} {_fmt(o['auc']):>15}  {_fmt(ge.get('auc', float('nan'))):>10}  {_fmt(gi['auc_incl']):>20}")
    print(sep)
    print(f"{'Confusion matrix':}")
    print(f"  {'TN':<{w-2}} {o['tn']:>15}  {ge.get('tn',0):>10}  {'':>20}")
    print(f"  {'FP':<{w-2}} {o['fp']:>15}  {ge.get('fp',0):>10}  {'':>20}")
    print(f"  {'FN':<{w-2}} {o['fn']:>15}  {ge.get('fn',0):>10}  {'':>20}")
    print(f"  {'TP':<{w-2}} {o['tp']:>15}  {ge.get('tp',0):>10}  {'':>20}")
    print(sep)
    print(f"{'Confident errors (>0.75)':<{w}} {o['n_confident_errors']:>15}  {ge.get('n_confident_errors',0):>10}  {'':>20}")
    print(f"{'='*70}")

    # caveat about training contamination
    print("\nCAVEAT: Mn-Bi (= Bi-Mn in MP) appeared in training data.")
    print("  For a truly clean holdout, refer to the Fe-Sb-only metrics above.")


# ── orchestration ─────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Evaluate both classifiers on Fe-Sb + Mn-Bi combined holdout.",
    )
    parser.add_argument("--model-dir",   default="models")
    parser.add_argument("--train-data",  default="data/results/training_data_balanced.csv")
    parser.add_argument("--full-data",   default="data/results/training_data_full.csv")
    parser.add_argument("--output-dir",  default=".")
    parser.add_argument("--confidence-threshold", type=float, default=0.75)
    args = parser.parse_args()

    api_key = os.environ.get("MP_API_KEY")
    if not api_key:
        sys.exit("MP_API_KEY is not set.")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1 — load models
    print("=" * 62)
    print("Step 1: Load models")
    print("=" * 62)
    orig_pipe, gap_pipe, feat_cols, cs_columns = load_models(args.model_dir, args.train_data)

    # 2 — query MP
    print("=" * 62)
    print("Step 2: Query Fe-Sb and Mn-Bi from Materials Project")
    print("=" * 62)
    df_raw = query_systems(api_key, ["Fe-Sb", "Mn-Bi"])
    print(f"Total queried: {len(df_raw)} entries\n")

    # 3 — overlap check
    print("=" * 62)
    print("Step 3: Training overlap check")
    print("=" * 62)
    overlaps = check_overlap(args.full_data, df_raw)

    # 4 — clean and engineer
    print("\n" + "=" * 62)
    print("Step 4: Feature engineering")
    print("=" * 62)
    df = df_raw.dropna(subset=["formation_energy_per_atom", "energy_above_hull"]).copy()
    df = df[df["nsites"].notna() & (df["nsites"] > 0)].reset_index(drop=True)
    if len(df) < len(df_raw):
        print(f"  Dropped {len(df_raw) - len(df)} rows with null energies.")
    X = engineer(df, feat_cols, cs_columns)
    print(f"  Feature matrix: {X.shape[0]} x {X.shape[1]}")

    # 5 — evaluate both models
    orig_metrics = evaluate_original(orig_pipe, X, df, args.confidence_threshold, overlaps)
    gap_metrics  = evaluate_gap_label(gap_pipe,  X, df, args.confidence_threshold, overlaps)

    # 6 — side-by-side comparison
    comparison_table(orig_metrics, gap_metrics)

    # 7 — save combined results CSV
    df["y_true_orig"]  = (df["energy_above_hull"] <= ORIG_THRESHOLD).astype(int)
    df["y_true_gap"]   = (df["energy_above_hull"] < GAP_LOWER).astype(int)
    df["in_amb_zone"]  = ((df["energy_above_hull"] >= GAP_LOWER) &
                          (df["energy_above_hull"] <= GAP_UPPER)).astype(int)
    df["pred_orig"]    = orig_pipe.predict(X)
    df["prob_orig"]    = orig_pipe.predict_proba(X)[:, 1].round(4)
    df["pred_gap"]     = gap_pipe.predict(X)
    df["prob_gap"]     = gap_pipe.predict_proba(X)[:, 1].round(4)
    df["in_training"]  = df["material_id"].apply(
        lambda mid: int(mid in set(pd.read_csv(args.full_data)["material_id"]))
    )

    out_path = out_dir / "validation_FeSb_MnBi.csv"
    save_cols = [
        "material_id", "formula_pretty", "chemsys", "energy_above_hull",
        "y_true_orig", "y_true_gap", "in_amb_zone", "in_training",
        "pred_orig", "prob_orig", "pred_gap", "prob_gap",
    ]
    df[save_cols].to_csv(out_path, index=False)
    print(f"\nFull results saved -> {out_path.resolve()}")


if __name__ == "__main__":
    main()
