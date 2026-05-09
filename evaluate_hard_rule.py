#!/usr/bin/env python3
"""
evaluate_hard_rule.py

Applies a post-prediction hard rule to the original GradientBoosting model:

    if formation_energy_per_atom > 0.0 eV/atom  →  override prediction to unstable

Re-evaluates on the Fe-Sb holdout (the clean holdout from evaluate_fesb_mnbi.py)
and reports entry-level changes plus before/after metric comparison.

Effective probability for overridden entries is set to 0.0 (maximum confidence
unstable), so ROC-AUC remains well-defined.
"""

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

HULL_THRESHOLD = 0.05
ORDERING_MAP   = {"NM": 0, "FM": 1, "FiM": 2, "AFM": 3}
CS_COLS        = ["cs_Cubic","cs_Hexagonal","cs_Monoclinic","cs_Orthorhombic",
                  "cs_Tetragonal","cs_Triclinic","cs_Trigonal"]
SUMMARY_FIELDS = [
    "material_id","formula_pretty","chemsys","symmetry","nsites","volume",
    "density","formation_energy_per_atom","energy_above_hull","band_gap",
    "total_magnetization","ordering",
]


# ── query ─────────────────────────────────────────────────────────────────────

def query_fesb(api_key: str) -> pd.DataFrame:
    from mp_api.client import MPRester
    with MPRester(api_key) as mpr:
        docs = mpr.materials.summary.search(chemsys=["Fe-Sb"], fields=SUMMARY_FIELDS)
    records = []
    for doc in docs:
        sym = doc.symmetry
        def es(v): return v.value if hasattr(v, "value") else str(v) if v else None
        records.append({
            "material_id":               str(doc.material_id),
            "formula_pretty":            doc.formula_pretty,
            "chemsys":                   doc.chemsys,
            "crystal_system":            es(getattr(sym, "crystal_system", None)) if sym else None,
            "spacegroup_number":         getattr(sym, "number", None) if sym else None,
            "nsites":                    doc.nsites,
            "volume":                    doc.volume,
            "density":                   doc.density,
            "formation_energy_per_atom": doc.formation_energy_per_atom,
            "energy_above_hull":         doc.energy_above_hull,
            "band_gap":                  doc.band_gap if doc.band_gap is not None else -1.0,
            "total_magnetization":       doc.total_magnetization if doc.total_magnetization is not None else 0.0,
            "magnetic_ordering":         es(getattr(doc, "ordering", None)),
        })
    return pd.DataFrame(records).sort_values("energy_above_hull").reset_index(drop=True)


# ── feature engineering ───────────────────────────────────────────────────────

def engineer(df: pd.DataFrame, feat_cols: list[str]) -> np.ndarray:
    from pymatgen.core import Composition
    df = df.copy()
    df["volume_per_atom"] = df["volume"] / df["nsites"]
    def frac(row):
        comp = Composition(row["formula_pretty"])
        first = sorted(comp.elements, key=lambda e: e.symbol)[0]
        return float(comp.get_atomic_fraction(first))
    df["composition_A_fraction"] = df.apply(frac, axis=1)
    df["band_gap"]            = df["band_gap"].fillna(-1.0)
    df["total_magnetization"] = df["total_magnetization"].fillna(0.0)
    df["magnetic_ordering"] = (
        df["magnetic_ordering"]
        .map(lambda x: ORDERING_MAP.get(str(x).strip(), -1) if pd.notna(x) else -1)
        .astype(int)
    )
    df["crystal_system"] = df["crystal_system"].fillna("Unknown")
    cs_dummies = pd.get_dummies(df["crystal_system"], prefix="cs").astype(int)
    df = pd.concat([df.drop(columns=["crystal_system"]), cs_dummies], axis=1)
    for col in CS_COLS:
        if col not in df.columns:
            df[col] = 0
    for col in feat_cols:
        if col not in df.columns:
            df[col] = 0
    return df[feat_cols].values.astype(float)


# ── metrics block ─────────────────────────────────────────────────────────────

def print_metrics(tag: str, y_true, y_pred, y_prob, df_meta):
    n           = len(y_true)
    acc         = accuracy_score(y_true, y_pred)
    auc         = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float("nan")
    cm          = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    print(f"  Accuracy : {acc:.4f}  ({int(acc*n)}/{n} correct)")
    auc_s = f"{auc:.4f}" if not np.isnan(auc) else "n/a"
    print(f"  ROC-AUC  : {auc_s}")

    present = sorted(np.unique(np.concatenate([y_true, y_pred])))
    lmap    = {0: "unstable", 1: "stable"}
    report  = classification_report(y_true, y_pred, labels=present,
                                    target_names=[lmap[l] for l in present],
                                    zero_division=0)
    print(f"\n  Classification report:")
    for line in report.splitlines():
        print(f"    {line}")

    print(f"  Confusion matrix (rows=true, cols=predicted):")
    print(f"                   pred:unstable  pred:stable")
    print(f"    true:unstable       {tn:>5}          {fp:>5}")
    print(f"    true:stable         {fn:>5}          {tp:>5}")

    wrong   = y_true != y_pred
    conf_p  = np.maximum(y_prob, 1 - y_prob)
    ce_mask = wrong & (conf_p > 0.75)
    ce_df   = df_meta[ce_mask].copy()
    ce_prob = y_prob[ce_mask]
    print(f"\n  Confident errors (p > 0.75, wrong label): {ce_mask.sum()}")
    for i, (_, row) in enumerate(ce_df.iterrows()):
        p    = ce_prob[i]
        pred = "stable" if p > 0.5 else "unstable"
        true = "stable" if row["is_stable"] == 1 else "unstable"
        conf = p if pred == "stable" else 1 - p
        print(f"    {row['material_id']:<12} {row['formula_pretty']:<10} "
              f"e_hull={row['energy_above_hull']:.4f}  fe={row['formation_energy_per_atom']:+.4f}  "
              f"true={true}  pred={pred}  conf={conf:.3f}")

    return {"accuracy": acc, "auc": auc, "tn": tn, "fp": fp, "fn": fn, "tp": tp,
            "n_confident_errors": int(ce_mask.sum())}


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    api_key = os.environ.get("MP_API_KEY")
    if not api_key:
        sys.exit("MP_API_KEY not set.")

    # load model + training feature columns
    art       = jload("models/stability_classifier.joblib")
    pipeline  = art["model"]
    df_train  = pd.read_csv("data/results/training_data_balanced.csv")
    drop      = {"material_id","formula_pretty","chemsys","energy_above_hull","is_stable"}
    feat_cols = [c for c in df_train.columns if c not in drop]

    # query Fe-Sb
    print("Querying Fe-Sb from Materials Project...")
    df = query_fesb(api_key)
    print(f"Retrieved {len(df)} entries.\n")

    df["is_stable"] = (df["energy_above_hull"] <= HULL_THRESHOLD).astype(int)
    X      = engineer(df, feat_cols)
    y_true = df["is_stable"].values

    # --- base model predictions ---
    y_pred_base = pipeline.predict(X)
    y_prob_base = pipeline.predict_proba(X)[:, 1]

    # --- hard rule ---
    fe_vals   = df["formation_energy_per_atom"].values
    rule_mask = fe_vals > 0.0          # entries where rule fires
    y_pred_rule = y_pred_base.copy()
    y_prob_rule = y_prob_base.copy()
    y_pred_rule[rule_mask] = 0         # override to unstable
    y_prob_rule[rule_mask] = 0.0       # override prob to 0 (max confidence unstable)

    # --- per-entry table ---
    print("Per-entry predictions (Fe-Sb holdout)")
    print("=" * 100)
    header = (f"  {'material_id':<12} {'formula':<10} {'fe/atom':>9} {'e_hull':>8}"
              f"  {'true':>10}  {'base pred':>10} {'p':>6}"
              f"  {'rule pred':>10} {'p':>6}  {'change'}")
    print(header)
    print("-" * 100)

    flipped, already_correct, newly_wrong = [], [], []
    for i, row in df.iterrows():
        true_l  = "stable"   if y_true[i]       == 1 else "unstable"
        base_l  = "stable"   if y_pred_base[i]  == 1 else "unstable"
        rule_l  = "stable"   if y_pred_rule[i]  == 1 else "unstable"
        fired   = bool(rule_mask[i])
        changed = y_pred_base[i] != y_pred_rule[i]

        if changed:
            if y_pred_rule[i] == y_true[i]:
                change_tag = "FIXED"
                flipped.append(row["material_id"])
            else:
                change_tag = "BROKE"
                newly_wrong.append(row["material_id"])
        elif fired:
            change_tag = "rule (no change)"
            already_correct.append(row["material_id"])
        else:
            change_tag = ""

        base_wrong = " *" if y_pred_base[i] != y_true[i] else "  "
        rule_wrong = " *" if y_pred_rule[i] != y_true[i] else "  "
        print(f"  {row['material_id']:<12} {row['formula_pretty']:<10}"
              f" {row['formation_energy_per_atom']:>+9.4f} {row['energy_above_hull']:>8.4f}"
              f"  {true_l:>10}  {base_l:>10}{base_wrong}{y_prob_base[i]:>6.3f}"
              f"  {rule_l:>10}{rule_wrong}{y_prob_rule[i]:>6.3f}  {change_tag}")

    print("-" * 100)
    print("  * = wrong prediction\n")

    # --- rule summary ---
    print(f"Hard rule fired on {rule_mask.sum()} entries (formation_energy_per_atom > 0):")
    for i in np.where(rule_mask)[0]:
        r = df.iloc[i]
        print(f"  {r['material_id']}  {r['formula_pretty']:<10}  "
              f"fe={r['formation_energy_per_atom']:+.4f}  "
              f"pred: {'stable' if y_pred_base[i]==1 else 'unstable'} "
              f"-> unstable  "
              f"({'FIXED' if y_pred_rule[i]==y_true[i] and y_pred_base[i]!=y_true[i] else 'already correct' if y_pred_base[i]==y_true[i] else 'broke'})")

    # --- before/after metrics ---
    print(f"\n{'='*55}")
    print("BEFORE  (original model, no hard rule)")
    print(f"{'='*55}")
    m_before = print_metrics("before", y_true, y_pred_base, y_prob_base, df)

    print(f"\n{'='*55}")
    print("AFTER   (original model + hard rule fe > 0 => unstable)")
    print(f"{'='*55}")
    m_after = print_metrics("after", y_true, y_pred_rule, y_prob_rule, df)

    # --- comparison ---
    w = 28
    print(f"\n{'='*55}")
    print("DELTA")
    print(f"{'='*55}")
    print(f"  {'Metric':<{w}} {'Before':>8}  {'After':>8}  {'Change':>8}")
    print(f"  {'-'*52}")

    def fmt(v): return "n/a" if isinstance(v, float) and np.isnan(v) else f"{v:.4f}"
    def delta(a, b):
        if isinstance(a, float) and np.isnan(a): return ""
        if isinstance(b, float) and np.isnan(b): return ""
        d = b - a
        return f"{d:+.4f}"

    for key, label in [("accuracy","Accuracy"), ("auc","ROC-AUC"),
                        ("tn","TN"), ("fp","FP"), ("fn","FN"), ("tp","TP"),
                        ("n_confident_errors","Confident errors")]:
        a, b = m_before[key], m_after[key]
        print(f"  {label:<{w}} {fmt(a):>8}  {fmt(b):>8}  {delta(a, b):>8}")

    print(f"\n  Predictions changed by hard rule: {int((y_pred_base != y_pred_rule).sum())}")
    if flipped:
        print(f"    Fixed (wrong -> correct) : {flipped}")
    if already_correct:
        print(f"    No change (already right): {already_correct}")
    if newly_wrong:
        print(f"    Broke (correct -> wrong) : {newly_wrong}")


if __name__ == "__main__":
    main()
