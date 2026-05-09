#!/usr/bin/env python3
"""
validate_final_cosn_nipb.py

Final validation of the 18-feature GradientBoosting model (is_endothermic)
against Co-Sn and Ni-Pb binary compounds from Materials Project.

WARNING: Co-Sn appears in training_data_balanced.csv (8 entries).
         Results for Co-Sn are marked CONTAMINATED.
         Ni-Pb is a clean holdout (0 overlap).

Reports:
  - Contamination check with explicit listing
  - Per-entry table for each system
  - Per-system and combined metrics (accuracy, ROC-AUC, confusion matrix, classification report)
  - Confident errors (p > 0.75, wrong label)
  - is_endothermic override cases
  - Comparison table vs prior rounds
  - Saves validation_CoSn_NiPb.csv
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
DROP_COLS = {"material_id","formula_pretty","chemsys","energy_above_hull","is_stable"}

# Prior results for comparison table
PRIOR_ROUNDS = [
    ("CV (248 samples)",            None,   0.870),
    ("Ni-Sb original (n=8)",        0.714,  0.917),
    ("Fe-Sb retrained (n=7)",       0.857,  0.917),
]


# ── contamination check ───────────────────────────────────────────────────────

def check_contamination(df_train: pd.DataFrame, systems: list[str]) -> dict[str, list[str]]:
    contamination = {}
    for sys in systems:
        parts = set(sys.split("-"))
        mask = df_train["chemsys"].apply(
            lambda c: set(str(c).split("-")) == parts if pd.notna(c) else False
        )
        hits = df_train[mask]["material_id"].tolist()
        contamination[sys] = hits
    return contamination


# ── query ─────────────────────────────────────────────────────────────────────

def query_system(api_key: str, chemsys: str) -> pd.DataFrame:
    from mp_api.client import MPRester
    with MPRester(api_key) as mpr:
        docs = mpr.materials.summary.search(chemsys=[chemsys], fields=SUMMARY_FIELDS)
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
    df["volume_per_atom"]    = df["volume"] / df["nsites"]
    df["is_endothermic"]     = (df["formation_energy_per_atom"] > 0.0).astype(int)
    def frac(row):
        try:
            comp  = Composition(row["formula_pretty"])
            first = sorted(comp.elements, key=lambda e: e.symbol)[0]
            return float(comp.get_atomic_fraction(first))
        except Exception:
            return np.nan
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


# ── per-entry table ───────────────────────────────────────────────────────────

def print_entry_table(df: pd.DataFrame, y_true, y_pred, y_prob,
                      contaminated_ids: set = None):
    print(f"  {'material_id':<12} {'formula':<10} {'fe/atom':>9} {'e_hull':>8}"
          f"  {'true':>10}  {'pred':>10} {'p':>6}  note")
    print(f"  {'-'*85}")
    for i, row in df.iterrows():
        tl   = "stable"   if y_true[i]   == 1 else "unstable"
        pl   = "stable"   if y_pred[i]   == 1 else "unstable"
        note = []
        if y_pred[i] != y_true[i]:
            note.append("WRONG")
        if row.get("is_endothermic", 0) == 1:
            note.append("endoth")
        if contaminated_ids and row["material_id"] in contaminated_ids:
            note.append("IN-TRAIN")
        print(f"  {row['material_id']:<12} {row['formula_pretty']:<10}"
              f" {row['formation_energy_per_atom']:>+9.4f} {row['energy_above_hull']:>8.4f}"
              f"  {tl:>10}  {pl:>10} {y_prob[i]:>6.3f}  {', '.join(note)}")
    print(f"  {'-'*85}")


# ── metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(y_true, y_pred, y_prob) -> dict:
    n   = len(y_true)
    acc = accuracy_score(y_true, y_pred)
    if len(np.unique(y_true)) > 1:
        auc = roc_auc_score(y_true, y_prob)
    else:
        auc = float("nan")
    cm          = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    wrong       = y_true != y_pred
    conf_p      = np.maximum(y_prob, 1 - y_prob)
    ce_mask     = wrong & (conf_p > 0.75)
    return {
        "n": n, "accuracy": acc, "auc": auc,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "n_errors": int(wrong.sum()), "n_confident_errors": int(ce_mask.sum()),
        "ce_mask": ce_mask,
    }


def print_metrics_block(tag: str, m: dict, y_true, y_pred, df=None, ce_mask=None):
    n   = m["n"]
    acc = m["accuracy"]
    auc = m["auc"]
    auc_s = f"{auc:.4f}" if not np.isnan(auc) else "n/a"
    print(f"  Accuracy : {acc:.4f}  ({int(acc*n)}/{n} correct)")
    print(f"  ROC-AUC  : {auc_s}")
    print(f"  TP={m['tp']}  FP={m['fp']}  TN={m['tn']}  FN={m['fn']}")

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
    print(f"    true:unstable       {m['tn']:>5}          {m['fp']:>5}")
    print(f"    true:stable         {m['fn']:>5}          {m['tp']:>5}")

    if ce_mask is not None and df is not None:
        y_prob = None  # must pass separately; handled by caller
    print(f"\n  Confident errors (p > 0.75, wrong): {m['n_confident_errors']}")


def print_confident_errors(df, y_true, y_pred, y_prob, ce_mask):
    if ce_mask.sum() == 0:
        print("    None.")
        return
    for i in np.where(ce_mask)[0]:
        r    = df.iloc[i]
        p    = y_prob[i]
        pred = "stable"   if p > 0.5 else "unstable"
        true = "stable"   if y_true[i] == 1 else "unstable"
        conf = p if pred == "stable" else 1 - p
        print(f"    {r['material_id']:<12} {r['formula_pretty']:<10}"
              f"  fe={r['formation_energy_per_atom']:+.4f}  e_hull={r['energy_above_hull']:.4f}"
              f"  true={true}  pred={pred}  conf={conf:.3f}")


# ── is_endothermic cases ──────────────────────────────────────────────────────

def report_endothermic_cases(df, y_true, y_pred, y_prob):
    mask = df["is_endothermic"].values == 1
    if mask.sum() == 0:
        print("  No is_endothermic=1 entries in this system.")
        return
    print(f"  {'material_id':<12} {'formula':<10} {'fe/atom':>9} {'e_hull':>8}"
          f"  {'true':>10}  {'pred':>10} {'p':>6}")
    print(f"  {'-'*78}")
    for i in np.where(mask)[0]:
        r  = df.iloc[i]
        tl = "stable"   if y_true[i]  == 1 else "unstable"
        pl = "stable"   if y_pred[i]  == 1 else "unstable"
        wr = " <WRONG" if y_pred[i] != y_true[i] else ""
        print(f"  {r['material_id']:<12} {r['formula_pretty']:<10}"
              f" {r['formation_energy_per_atom']:>+9.4f} {r['energy_above_hull']:>8.4f}"
              f"  {tl:>10}  {pl:>10} {y_prob[i]:>6.3f}{wr}")


# ── comparison table ──────────────────────────────────────────────────────────

def print_comparison_table(results: list[tuple]):
    """results: list of (label, acc, auc, n, contaminated_flag)"""
    w = 36
    print(f"\n{'='*70}")
    print("COMPARISON ACROSS ALL VALIDATION ROUNDS")
    print(f"{'='*70}")
    print(f"  {'Round':<{w}} {'n':>5}  {'Accuracy':>10}  {'ROC-AUC':>10}  {'Note'}")
    print(f"  {'-'*68}")
    for label, n, acc, auc, note in results:
        acc_s = f"{acc:.4f}" if acc is not None else "  n/a  "
        auc_s = f"{auc:.4f}" if auc is not None and not (isinstance(auc, float) and np.isnan(auc)) else "  n/a  "
        n_s   = str(n) if n is not None else "  n/a"
        print(f"  {label:<{w}} {n_s:>5}  {acc_s:>10}  {auc_s:>10}  {note}")
    print(f"{'='*70}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    api_key = os.environ.get("MP_API_KEY")
    if not api_key:
        sys.exit("MP_API_KEY not set.")

    # ── load model ─────────────────────────────────────────────────────────
    print("=" * 60)
    print("Step 1: Load 18-feature endothermic model")
    print("=" * 60)
    artifact  = jload("models/stability_classifier_endothermic.joblib")
    pipeline  = artifact["model"]
    model_name = artifact.get("model_name", "unknown")
    print(f"Loaded: {model_name}")

    df_train   = pd.read_csv("data/results/training_data_balanced.csv")
    feat_cols  = [c for c in df_train.columns if c not in DROP_COLS]
    # is_endothermic is NOT in the CSV but was added during retrain —
    # confirm it is expected by checking the model's input size
    n_model_features = pipeline.named_steps["clf"].n_features_in_
    if "is_endothermic" not in feat_cols:
        feat_cols.append("is_endothermic")
    print(f"Feature set ({len(feat_cols)}): {feat_cols}")
    print(f"Model expects {n_model_features} features  -- match: {len(feat_cols)==n_model_features}")

    # ── contamination check ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Step 2: Contamination check")
    print("=" * 60)
    contamination = check_contamination(df_train, ["Co-Sn", "Ni-Pb"])

    contaminated_systems = set()
    contaminated_ids     = set()
    for sys, ids in contamination.items():
        if ids:
            contaminated_systems.add(sys)
            contaminated_ids.update(ids)
            print(f"  *** CONTAMINATION DETECTED: {sys} has {len(ids)} entries in training data ***")
            mask = df_train["material_id"].isin(ids)
            sub  = df_train[mask][["material_id","formula_pretty","energy_above_hull"]].copy()
            for _, row in sub.iterrows():
                print(f"    {row['material_id']:<12} {row['formula_pretty']:<12}"
                      f"  e_hull={row['energy_above_hull']:.6f}")
        else:
            print(f"  {sys}: 0 overlap -- clean holdout confirmed.")

    if contaminated_systems:
        print(f"\n  PROCEEDING with CONTAMINATION WARNING for: {sorted(contaminated_systems)}")
        print(f"  Co-Sn results will be marked CONTAMINATED in all tables.")
        print(f"  Ni-Pb is the only valid clean holdout.\n")

    # ── query ──────────────────────────────────────────────────────────────
    print("=" * 60)
    print("Step 3: Query Materials Project")
    print("=" * 60)
    print("Querying Co-Sn...")
    df_cosn = query_system(api_key, "Co-Sn")
    print(f"  Retrieved {len(df_cosn)} Co-Sn entries.")

    print("Querying Ni-Pb...")
    df_nipb = query_system(api_key, "Ni-Pb")
    print(f"  Retrieved {len(df_nipb)} Ni-Pb entries.")

    # ── label ──────────────────────────────────────────────────────────────
    df_cosn["is_stable"] = (df_cosn["energy_above_hull"] <= HULL_THRESHOLD).astype(int)
    df_nipb["is_stable"] = (df_nipb["energy_above_hull"] <= HULL_THRESHOLD).astype(int)

    y_true_cosn = df_cosn["is_stable"].values
    y_true_nipb = df_nipb["is_stable"].values

    print(f"\n  Co-Sn: stable={y_true_cosn.sum()}  unstable={len(y_true_cosn)-y_true_cosn.sum()}"
          f"  [CONTAMINATED]")
    print(f"  Ni-Pb: stable={y_true_nipb.sum()}  unstable={len(y_true_nipb)-y_true_nipb.sum()}"
          f"  [clean]")

    # ── feature engineering ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Step 4: Feature engineering")
    print("=" * 60)
    X_cosn = engineer(df_cosn, feat_cols)
    X_nipb = engineer(df_nipb, feat_cols)

    # inject is_endothermic into df for reporting
    df_cosn["is_endothermic"] = (df_cosn["formation_energy_per_atom"] > 0.0).astype(int)
    df_nipb["is_endothermic"] = (df_nipb["formation_energy_per_atom"] > 0.0).astype(int)

    print(f"  Co-Sn feature matrix: {X_cosn.shape}")
    print(f"  Ni-Pb feature matrix: {X_nipb.shape}")

    # ── predict ────────────────────────────────────────────────────────────
    y_pred_cosn = pipeline.predict(X_cosn)
    y_prob_cosn = pipeline.predict_proba(X_cosn)[:, 1]
    y_pred_nipb = pipeline.predict(X_nipb)
    y_prob_nipb = pipeline.predict_proba(X_nipb)[:, 1]

    m_cosn = compute_metrics(y_true_cosn, y_pred_cosn, y_prob_cosn)
    m_nipb = compute_metrics(y_true_nipb, y_pred_nipb, y_prob_nipb)

    # combined
    y_true_all = np.concatenate([y_true_cosn, y_true_nipb])
    y_pred_all = np.concatenate([y_pred_cosn, y_pred_nipb])
    y_prob_all = np.concatenate([y_prob_cosn, y_prob_nipb])
    m_all      = compute_metrics(y_true_all, y_pred_all, y_prob_all)

    # ── per-entry tables ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Co-Sn per-entry  [*** CONTAMINATED -- entries seen during training ***]")
    print("=" * 60)
    print_entry_table(df_cosn.reset_index(drop=True), y_true_cosn, y_pred_cosn, y_prob_cosn,
                      contaminated_ids=contaminated_ids)

    print("\n" + "=" * 60)
    print("Ni-Pb per-entry  [clean holdout]")
    print("=" * 60)
    print_entry_table(df_nipb.reset_index(drop=True), y_true_nipb, y_pred_nipb, y_prob_nipb)

    # ── per-system metrics ─────────────────────────────────────────────────
    for tag, df_s, y_t, y_p, y_pr, m, flag in [
        ("Co-Sn [CONTAMINATED]", df_cosn.reset_index(drop=True),
         y_true_cosn, y_pred_cosn, y_prob_cosn, m_cosn, "CONTAMINATED"),
        ("Ni-Pb [clean holdout]", df_nipb.reset_index(drop=True),
         y_true_nipb, y_pred_nipb, y_prob_nipb, m_nipb, "CLEAN"),
        ("Combined Co-Sn + Ni-Pb", None,
         y_true_all, y_pred_all, y_prob_all, m_all, "MIXED"),
    ]:
        print(f"\n{'='*60}")
        print(f"Metrics: {tag}")
        print(f"{'='*60}")
        print_metrics_block(tag, m, y_t, y_p)
        print(f"\n  Confident errors:")
        ce_df = df_s if df_s is not None else pd.concat(
            [df_cosn.reset_index(drop=True), df_nipb.reset_index(drop=True)], ignore_index=True
        )
        print_confident_errors(ce_df, y_t, y_p, y_pr, m["ce_mask"])

    # ── is_endothermic cases ───────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("is_endothermic=1 cases (formation_energy_per_atom > 0)")
    print(f"{'='*60}")

    print("\n  Co-Sn [CONTAMINATED]:")
    report_endothermic_cases(df_cosn.reset_index(drop=True),
                             y_true_cosn, y_pred_cosn, y_prob_cosn)

    print("\n  Ni-Pb [clean]:")
    report_endothermic_cases(df_nipb.reset_index(drop=True),
                             y_true_nipb, y_pred_nipb, y_prob_nipb)

    # ── save CSV ───────────────────────────────────────────────────────────
    out_path = Path("data/results/validation_CoSn_NiPb.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def make_result_df(df_s, y_t, y_p, y_pr, system, contaminated):
        r = df_s[["material_id","formula_pretty","chemsys",
                   "formation_energy_per_atom","energy_above_hull",
                   "is_endothermic"]].copy().reset_index(drop=True)
        r["true_label"]            = np.where(y_t == 1, "stable", "unstable")
        r["predicted_label"]       = np.where(y_p == 1, "stable", "unstable")
        r["predicted_probability"] = np.round(y_pr, 4)
        r["correct"]               = y_t == y_p
        r["contaminated"]          = contaminated
        conf_p = np.maximum(y_pr, 1 - y_pr)
        r["confident_error"] = (y_t != y_p) & (conf_p > 0.75)
        return r

    df_out = pd.concat([
        make_result_df(df_cosn.reset_index(drop=True),
                       y_true_cosn, y_pred_cosn, y_prob_cosn, "Co-Sn", True),
        make_result_df(df_nipb.reset_index(drop=True),
                       y_true_nipb, y_pred_nipb, y_prob_nipb, "Ni-Pb", False),
    ], ignore_index=True)

    df_out.to_csv(out_path, index=False)
    print(f"\nSaved -> {out_path.resolve()}")

    # ── comparison table ───────────────────────────────────────────────────
    acc_cosn = m_cosn["accuracy"]
    auc_cosn = m_cosn["auc"]
    acc_nipb = m_nipb["accuracy"]
    auc_nipb = m_nipb["auc"]
    acc_all  = m_all["accuracy"]
    auc_all  = m_all["auc"]

    def _safe_auc(v):
        return None if (isinstance(v, float) and np.isnan(v)) else v

    comparison_rows = [
        ("CV -- GBC 5-fold (n=248)",          248,    None,     0.870,   "training"),
        ("Ni-Sb original (n=8)",               8,     0.714,    0.917,   "clean holdout"),
        ("Fe-Sb retrained endothermic (n=7)",  7,     0.857,    0.917,   "clean holdout"),
        (f"Co-Sn endothermic (n={len(df_cosn)})",
          len(df_cosn), acc_cosn, _safe_auc(auc_cosn), "CONTAMINATED"),
        (f"Ni-Pb endothermic (n={len(df_nipb)})",
          len(df_nipb), acc_nipb, _safe_auc(auc_nipb), "clean holdout"),
        (f"Combined Co-Sn+Ni-Pb (n={len(df_cosn)+len(df_nipb)})",
          len(df_cosn)+len(df_nipb), acc_all, _safe_auc(auc_all), "mixed"),
    ]

    print_comparison_table(comparison_rows)

    # ── final note ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("SUMMARY NOTE")
    print(f"{'='*60}")
    print(f"  Co-Sn results are NOT valid holdout evidence.")
    print(f"  {len(contaminated_ids)} of {len(df_cosn)} Co-Sn entries were seen during training.")
    print(f"  Ni-Pb (n={len(df_nipb)}) is the only uncontaminated final test system.")
    nipb_stable = int(y_true_nipb.sum())
    nipb_unstable = len(y_true_nipb) - nipb_stable
    print(f"  Ni-Pb class balance: {nipb_stable} stable / {nipb_unstable} unstable")
    if not np.isnan(auc_nipb):
        print(f"  Ni-Pb accuracy={acc_nipb:.4f}  AUC={auc_nipb:.4f}")
    else:
        print(f"  Ni-Pb accuracy={acc_nipb:.4f}  AUC=n/a (single class)")


if __name__ == "__main__":
    main()
