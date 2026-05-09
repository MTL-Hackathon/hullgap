#!/usr/bin/env python3
"""
retrain_endothermic.py

Adds a binary feature:
    is_endothermic = 1 if formation_energy_per_atom > 0.0 else 0

Retrains GradientBoosting with the 18-feature set, evaluates on the Fe-Sb
holdout, and compares accuracy and confident errors against the original
17-feature model.
"""

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

HULL_THRESHOLD = 0.05
ORDERING_MAP   = {"NM": 0, "FM": 1, "FiM": 2, "AFM": 3}
CS_COLS        = ["cs_Cubic","cs_Hexagonal","cs_Monoclinic","cs_Orthorhombic",
                  "cs_Tetragonal","cs_Triclinic","cs_Trigonal"]
SUMMARY_FIELDS = [
    "material_id","formula_pretty","chemsys","symmetry","nsites","volume",
    "density","formation_energy_per_atom","energy_above_hull","band_gap",
    "total_magnetization","ordering",
]
DROP_COLS = {"material_id","formula_pretty","chemsys","energy_above_hull","is_stable"}


# ── step 1: retrain with is_endothermic ───────────────────────────────────────

def retrain(train_csv: str) -> tuple[Pipeline, list[str]]:
    df = pd.read_csv(train_csv)

    # inject new feature
    df["is_endothermic"] = (df["formation_energy_per_atom"] > 0.0).astype(int)

    feat_cols = [c for c in df.columns if c not in DROP_COLS]
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
    print(f"New feature set: {feat_cols}\n")
    return pipeline, feat_cols


# ── step 2: query Fe-Sb ───────────────────────────────────────────────────────

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


# ── step 3: feature engineering ───────────────────────────────────────────────

def engineer(df: pd.DataFrame, feat_cols: list[str]) -> np.ndarray:
    from pymatgen.core import Composition
    df = df.copy()
    df["volume_per_atom"]        = df["volume"] / df["nsites"]
    df["is_endothermic"]         = (df["formation_energy_per_atom"] > 0.0).astype(int)
    def frac(row):
        comp  = Composition(row["formula_pretty"])
        first = sorted(comp.elements, key=lambda e: e.symbol)[0]
        return float(comp.get_atomic_fraction(first))
    df["composition_A_fraction"] = df.apply(frac, axis=1)
    df["band_gap"]               = df["band_gap"].fillna(-1.0)
    df["total_magnetization"]    = df["total_magnetization"].fillna(0.0)
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


# ── step 4: evaluate one model and return metrics ────────────────────────────

def evaluate(
    label: str,
    pipeline: Pipeline,
    feat_cols: list[str],
    df: pd.DataFrame,
    X: np.ndarray,
    y_true: np.ndarray,
) -> dict:
    y_pred = pipeline.predict(X)
    y_prob = pipeline.predict_proba(X)[:, 1]

    n   = len(y_true)
    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float("nan")
    cm  = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    print(f"\n{'='*55}")
    print(f"{label}")
    print(f"{'='*55}")
    print(f"  Features ({len(feat_cols)}): {feat_cols}")
    print(f"\n  Accuracy : {acc:.4f}  ({int(acc*n)}/{n})")
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

    # confident errors
    wrong   = y_true != y_pred
    conf_p  = np.maximum(y_prob, 1 - y_prob)
    ce_mask = wrong & (conf_p > 0.75)
    print(f"\n  Confident errors (p > 0.75, wrong label): {ce_mask.sum()}")
    ce_rows = []
    for i in np.where(ce_mask)[0]:
        r    = df.iloc[i]
        p    = y_prob[i]
        pred = "stable" if p > 0.5 else "unstable"
        true = "stable" if y_true[i] == 1 else "unstable"
        conf = p if pred == "stable" else 1 - p
        ce_rows.append({"mid": r["material_id"], "formula": r["formula_pretty"],
                        "fe": r["formation_energy_per_atom"], "e_hull": r["energy_above_hull"],
                        "true": true, "pred": pred, "conf": conf, "prob": p})
        print(f"    {r['material_id']:<12} {r['formula_pretty']:<10}"
              f"  fe={r['formation_energy_per_atom']:+.4f}  e_hull={r['energy_above_hull']:.4f}"
              f"  true={true}  pred={pred}  conf={conf:.3f}")

    return {"accuracy": acc, "auc": auc, "tn": tn, "fp": fp, "fn": fn, "tp": tp,
            "n_errors": int(wrong.sum()), "n_confident_errors": int(ce_mask.sum()),
            "y_pred": y_pred, "y_prob": y_prob, "conf_errors": ce_rows}


# ── step 5: per-entry side-by-side + feature importance ──────────────────────

def per_entry_comparison(df, y_true, m_orig, m_new):
    print(f"\n{'='*90}")
    print("Per-entry comparison  (Fe-Sb holdout)")
    print(f"{'='*90}")
    print(f"  {'material_id':<12} {'formula':<10} {'fe/atom':>9} {'e_hull':>8}"
          f"  {'true':>10}  {'orig':>10} {'p':>6}  {'new':>10} {'p':>6}  change")
    print(f"  {'-'*88}")
    for i, row in df.iterrows():
        tl   = "stable"   if y_true[i]            == 1 else "unstable"
        ol   = "stable"   if m_orig["y_pred"][i]  == 1 else "unstable"
        nl   = "stable"   if m_new["y_pred"][i]   == 1 else "unstable"
        ow   = " *" if m_orig["y_pred"][i] != y_true[i] else "  "
        nw   = " *" if m_new["y_pred"][i]  != y_true[i] else "  "
        changed = m_orig["y_pred"][i] != m_new["y_pred"][i]
        if changed:
            tag = "FIXED" if m_new["y_pred"][i] == y_true[i] else "BROKE"
        else:
            tag = ""
        print(f"  {row['material_id']:<12} {row['formula_pretty']:<10}"
              f" {row['formation_energy_per_atom']:>+9.4f} {row['energy_above_hull']:>8.4f}"
              f"  {tl:>10}  {ol:>10}{ow}{m_orig['y_prob'][i]:>6.3f}"
              f"  {nl:>10}{nw}{m_new['y_prob'][i]:>6.3f}  {tag}")
    print(f"  {'-'*88}")
    print("  * = wrong prediction")


def feature_importances(pipeline: Pipeline, feat_cols: list[str]):
    clf  = pipeline.named_steps["clf"]
    imp  = clf.feature_importances_
    rows = sorted(zip(feat_cols, imp), key=lambda x: x[1], reverse=True)
    print(f"\n  Feature importances (ranked):")
    print(f"  {'Feature':<32} {'Importance':>11}  Bar")
    print(f"  {'-'*56}")
    for feat, val in rows:
        bar = "#" * int(val / imp.max() * 30)
        print(f"  {feat:<32} {val:>11.4f}  {bar}")

    # focused group comparison
    endoth_imp = imp[feat_cols.index("is_endothermic")]
    fe_imp     = imp[feat_cols.index("formation_energy_per_atom")]
    cs_total   = sum(imp[feat_cols.index(c)] for c in CS_COLS if c in feat_cols)
    print(f"\n  is_endothermic            : {endoth_imp:.4f}  ({endoth_imp*100:.1f}%)")
    print(f"  formation_energy_per_atom : {fe_imp:.4f}  ({fe_imp*100:.1f}%)")
    print(f"  all cs_* combined         : {cs_total:.4f}  ({cs_total*100:.1f}%)")
    print(f"  is_endothermic / formation_energy ratio: {endoth_imp/fe_imp:.3f}")


def comparison_table(m_orig: dict, m_new: dict):
    w   = 28
    sep = "-" * 52
    print(f"\n{'='*52}")
    print("COMPARISON SUMMARY")
    print(f"{'='*52}")
    print(f"  {'Metric':<{w}} {'Original':>10}  {'+ is_endothermic':>14}")
    print(sep)

    def fmt(v):
        return "n/a" if isinstance(v, float) and np.isnan(v) else f"{v:.4f}"
    def dlt(a, b):
        if any(isinstance(x, float) and np.isnan(x) for x in [a, b]): return ""
        return f"{b-a:+.4f}"

    for key, label in [("accuracy","Accuracy"), ("auc","ROC-AUC"),
                        ("n_errors","Total errors"), ("n_confident_errors","Confident errors"),
                        ("tn","TN"), ("fp","FP"), ("fn","FN"), ("tp","TP")]:
        a, b = m_orig[key], m_new[key]
        print(f"  {label:<{w}} {fmt(a):>10}  {fmt(b):>14}  {dlt(a,b)}")
    print(f"{'='*52}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    api_key = os.environ.get("MP_API_KEY")
    if not api_key:
        sys.exit("MP_API_KEY not set.")

    # 1 — retrain
    print("=" * 55)
    print("Step 1: Retrain with is_endothermic feature")
    print("=" * 55)
    new_pipeline, new_feat_cols = retrain("data/results/training_data_balanced.csv")
    dump({"model": new_pipeline, "model_name": "GradientBoosting_endothermic"},
         "models/stability_classifier_endothermic.joblib")
    print("Saved -> models/stability_classifier_endothermic.joblib")

    # 2 — query Fe-Sb
    print("\n" + "=" * 55)
    print("Step 2: Query Fe-Sb holdout")
    print("=" * 55)
    df = query_fesb(api_key)
    df["is_stable"] = (df["energy_above_hull"] <= HULL_THRESHOLD).astype(int)
    y_true = df["is_stable"].values
    print(f"Retrieved {len(df)} entries  "
          f"(stable={y_true.sum()}, unstable={len(y_true)-y_true.sum()})")

    # 3 — load original model + derive its feature cols
    orig_art   = jload("models/stability_classifier.joblib")
    orig_pipe  = orig_art["model"]
    df_train   = pd.read_csv("data/results/training_data_balanced.csv")
    orig_feats = [c for c in df_train.columns if c not in DROP_COLS]

    # 4 — engineer for both feature sets
    X_orig = engineer(df, orig_feats)     # 17 features (no is_endothermic)
    X_new  = engineer(df, new_feat_cols)  # 18 features (with is_endothermic)

    # 5 — evaluate both
    m_orig = evaluate("ORIGINAL model (17 features, no is_endothermic)",
                      orig_pipe, orig_feats, df, X_orig, y_true)
    m_new  = evaluate("RETRAINED model (18 features, with is_endothermic)",
                      new_pipeline, new_feat_cols, df, X_new, y_true)

    # 6 — per-entry comparison
    per_entry_comparison(df, y_true, m_orig, m_new)

    # 7 — feature importances of new model
    print(f"\n{'='*55}")
    print("Feature importances — retrained model")
    print(f"{'='*55}")
    feature_importances(new_pipeline, new_feat_cols)

    # 8 — summary table
    comparison_table(m_orig, m_new)


if __name__ == "__main__":
    main()
