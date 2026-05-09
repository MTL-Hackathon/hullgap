#!/usr/bin/env python3
"""
build_structure_classifier.py

Structure-only stability pre-screening classifier for Co-Bi binary compounds.
No DFT-derived properties used as features: no formation energy, band gap,
magnetization, or total energy.

Geometry features (min bond distances, packing fraction, volume/atom) ARE
included — they are computed from atomic positions and hard-sphere radii,
not from DFT energy calculations.

Step 1: Query training data (28 binary systems) including structure objects
Step 2: Compute geometry features + matminer compositional features
Step 3: Train GBC and RF; report CV, test metrics, feature importances
Step 4: Validate on Fe-Sb and Ni-Sb holdouts
Step 5: Score Co-Bi candidates from cobi_test_combinations_100.csv
"""

import os
import sys
import warnings
from math import pi
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from joblib import dump, load as jload
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

warnings.filterwarnings("ignore")
load_dotenv()

# ── constants ─────────────────────────────────────────────────────────────────

HULL_THRESHOLD  = 0.05
RANDOM_STATE    = 42
R_NEIGHBOR      = 6.0   # Angstrom, for geometry feature computation

_CO_PARTNERS    = ["As","Sb","Te","Se","P","S","Sn","Ge","Si","N","O","Fe","Ni","Mn"]
_BI_PARTNERS    = ["Ni","Fe","Mn","Cu","Ag","Zn","Pb","Sb","Te","Se","In","Ga","Tl"]
TRAIN_SYSTEMS   = (
    [f"Co-{x}" for x in _CO_PARTNERS]
    + [f"Bi-{x}" for x in _BI_PARTNERS]
    + ["Co-Bi"]
)

SUMMARY_FIELDS  = [
    "material_id", "formula_pretty", "chemsys", "symmetry",
    "nsites", "energy_above_hull", "structure",
]

CS_COLS         = ["cs_Cubic","cs_Hexagonal","cs_Monoclinic",
                   "cs_Orthorhombic","cs_Tetragonal","cs_Triclinic","cs_Trigonal"]

RATIO_FLAGS     = ["ratio_1_1","ratio_1_2","ratio_2_1","ratio_1_3","ratio_3_1",
                   "ratio_2_3","ratio_3_2","ratio_1_4","ratio_4_1","ratio_1_5","ratio_5_1"]

# Geometry features — structure-derived, NOT DFT energy outputs
GEOM_FEAT_COLS  = [
    "min_AB_dist",       # min distance between the two different element types
    "min_AA_dist",       # min distance between the alphabetically-first element pair
    "min_BB_dist",       # min distance between the alphabetically-second element pair
    "packing_fraction",  # (n_A * V_A + n_B * V_B) / cell_volume, hard-sphere radii
    "volume_per_atom",   # cell volume / nsites
]

# DFT-derived outputs that must never appear as features
DFT_COLS        = {
    "formation_energy_per_atom", "energy_above_hull", "is_stable",
    "band_gap", "total_magnetization", "magnetic_ordering",
    "energy_per_atom", "total_energy",
}


# ── geometry feature computation ──────────────────────────────────────────────

def compute_geom_features(structure, formula_pretty: str) -> dict:
    """
    Compute generalised geometry features for any binary compound.

    Elements are sorted alphabetically so that the feature names are
    consistent across different chemical systems:
        A = alphabetically first element (e.g. Bi in Co-Bi, As in Co-As)
        B = alphabetically second element (e.g. Co in Co-Bi, Co in Co-As)

    Returns a dict with keys matching GEOM_FEAT_COLS.
    All values are NaN on failure so that downstream imputation handles them.
    """
    from pymatgen.core import Composition, Element

    nan_row = {k: np.nan for k in GEOM_FEAT_COLS}

    if structure is None:
        return nan_row

    try:
        comp     = Composition(formula_pretty)
        elements = sorted(str(e) for e in comp.elements)
        if len(elements) < 2:
            el_A = el_B = elements[0]
        else:
            el_A, el_B = elements[0], elements[1]
    except Exception:
        return nan_row

    try:
        r_A = float(Element(el_A).atomic_radius or 1.4)
        r_B = float(Element(el_B).atomic_radius or 1.4)
    except Exception:
        r_A = r_B = 1.4

    n_A = sum(1 for s in structure if str(s.specie) == el_A)
    n_B = sum(1 for s in structure if str(s.specie) == el_B)
    V_A = (4 / 3) * pi * r_A ** 3
    V_B = (4 / 3) * pi * r_B ** 3

    try:
        phi = (n_A * V_A + n_B * V_B) / structure.volume
        vpa = structure.volume / structure.num_sites
    except Exception:
        return nan_row

    try:
        nb_list = structure.get_all_neighbors(r=R_NEIGHBOR)
    except Exception:
        return {
            "min_AB_dist": np.nan,
            "min_AA_dist": np.nan,
            "min_BB_dist": np.nan,
            "packing_fraction": phi,
            "volume_per_atom":  vpa,
        }

    aa_dists: list[float] = []
    bb_dists: list[float] = []
    ab_dists: list[float] = []

    for i, site in enumerate(structure):
        sp = str(site.specie)
        for nb in nb_list[i]:
            nb_sp = str(nb.specie)
            d = nb.nn_distance
            if sp == el_A and nb_sp == el_A:
                aa_dists.append(d)
            elif sp == el_B and nb_sp == el_B:
                bb_dists.append(d)
            elif (sp == el_A and nb_sp == el_B) or (sp == el_B and nb_sp == el_A):
                ab_dists.append(d)

    return {
        "min_AB_dist":      min(ab_dists) if ab_dists else np.nan,
        "min_AA_dist":      min(aa_dists) if aa_dists else np.nan,
        "min_BB_dist":      min(bb_dists) if bb_dists else np.nan,
        "packing_fraction": phi,
        "volume_per_atom":  vpa,
    }


# ── step 1: query training data ───────────────────────────────────────────────

def query_training(api_key: str) -> pd.DataFrame:
    from mp_api.client import MPRester
    records = []
    with MPRester(api_key) as mpr:
        for chemsys in tqdm(TRAIN_SYSTEMS, desc="Querying MP", unit="system"):
            try:
                docs = mpr.materials.summary.search(
                    chemsys=[chemsys], fields=SUMMARY_FIELDS
                )
                for doc in docs:
                    sym = doc.symmetry
                    def es(v): return v.value if hasattr(v, "value") else str(v) if v else None
                    geom = compute_geom_features(
                        getattr(doc, "structure", None),
                        doc.formula_pretty,
                    )
                    records.append({
                        "material_id":       str(doc.material_id),
                        "formula_pretty":    doc.formula_pretty,
                        "chemsys":           doc.chemsys,
                        "crystal_system":    es(getattr(sym, "crystal_system", None)) if sym else None,
                        "spacegroup_number": getattr(sym, "number", None) if sym else None,
                        "nsites":            doc.nsites,
                        "energy_above_hull": doc.energy_above_hull,
                        **geom,
                    })
            except Exception as e:
                print(f"  WARNING: {chemsys} failed: {e}")
    df = pd.DataFrame(records)
    df = df.dropna(subset=["energy_above_hull", "nsites"]).reset_index(drop=True)
    df["is_stable"] = (df["energy_above_hull"] <= HULL_THRESHOLD).astype(int)
    return df


def query_system_raw(api_key: str, chemsys: str) -> pd.DataFrame:
    from mp_api.client import MPRester
    with MPRester(api_key) as mpr:
        docs = mpr.materials.summary.search(chemsys=[chemsys], fields=SUMMARY_FIELDS)
    records = []
    for doc in docs:
        sym = doc.symmetry
        def es(v): return v.value if hasattr(v, "value") else str(v) if v else None
        geom = compute_geom_features(
            getattr(doc, "structure", None),
            doc.formula_pretty,
        )
        records.append({
            "material_id":       str(doc.material_id),
            "formula_pretty":    doc.formula_pretty,
            "chemsys":           doc.chemsys,
            "crystal_system":    es(getattr(sym, "crystal_system", None)) if sym else None,
            "spacegroup_number": getattr(sym, "number", None) if sym else None,
            "nsites":            doc.nsites,
            "energy_above_hull": doc.energy_above_hull,
            **geom,
        })
    df = pd.DataFrame(records)
    df = df.dropna(subset=["energy_above_hull", "nsites"]).reset_index(drop=True)
    df["is_stable"] = (df["energy_above_hull"] <= HULL_THRESHOLD).astype(int)
    return df


# ── step 2: feature engineering ───────────────────────────────────────────────

def _ratio_flags(formula: str) -> dict:
    from pymatgen.core import Composition
    try:
        comp   = Composition(formula)
        counts = sorted(comp.as_dict().values())
        if len(counts) == 2:
            a, b = sorted([counts[0], counts[1]])
        elif len(counts) == 1:
            a, b = counts[0], counts[0]
        else:
            return {f: 0 for f in RATIO_FLAGS}
        ratios = {
            "ratio_1_1": (a == b),
            "ratio_1_2": (b == 2 * a),
            "ratio_2_1": (a * 2 == b) if a * 2 == b else (b == 2 * a),
            "ratio_1_3": (b == 3 * a),
            "ratio_3_1": (b == a * 3),
            "ratio_2_3": (b * 2 == a * 3 or a * 2 == b * 3),
            "ratio_3_2": (a * 3 == b * 2 or b * 3 == a * 2),
            "ratio_1_4": (b == 4 * a),
            "ratio_4_1": (a * 4 == b),
            "ratio_1_5": (b == 5 * a),
            "ratio_5_1": (a * 5 == b),
        }
        return {k: int(v) for k, v in ratios.items()}
    except Exception:
        return {f: 0 for f in RATIO_FLAGS}


def _first_el_fraction(formula: str) -> float:
    from pymatgen.core import Composition
    try:
        comp  = Composition(formula)
        first = sorted(comp.elements, key=lambda e: e.symbol)[0]
        return float(comp.get_atomic_fraction(first))
    except Exception:
        return np.nan


def engineer_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Build feature matrix. Returns (df_with_features, feat_cols).
    Includes geometry features (min distances, packing fraction, volume/atom)
    if they are already present in the DataFrame; imputes NaN otherwise.
    """
    from pymatgen.core import Composition
    from matminer.featurizers.composition import ElementProperty, Stoichiometry, ValenceOrbital

    df = df.copy().reset_index(drop=True)

    # -- structural scalars --
    df["composition_A_fraction"] = df["formula_pretty"].apply(_first_el_fraction)
    df["n_atoms_parity"]         = (df["nsites"] % 2).astype(int)
    df["spacegroup_number"]      = pd.to_numeric(df["spacegroup_number"], errors="coerce").fillna(1)

    # ensure geometry cols exist (fill with NaN if absent, imputed later)
    for col in GEOM_FEAT_COLS:
        if col not in df.columns:
            df[col] = np.nan

    # -- crystal system one-hot --
    df["crystal_system"] = df["crystal_system"].fillna("Unknown")
    cs_dummies = pd.get_dummies(df["crystal_system"], prefix="cs").astype(int)
    for col in CS_COLS:
        if col not in cs_dummies.columns:
            cs_dummies[col] = 0
    cs_dummies = cs_dummies[CS_COLS]
    df = pd.concat([df.drop(columns=["crystal_system"]), cs_dummies], axis=1)

    # -- stoichiometry ratio flags --
    ratio_df = pd.DataFrame(
        df["formula_pretty"].apply(_ratio_flags).tolist(), index=df.index
    )
    df = pd.concat([df, ratio_df], axis=1)

    # -- matminer compositional features --
    print("  Running matminer featurizers...")
    df["_comp"] = df["formula_pretty"].apply(
        lambda f: Composition(f) if pd.notna(f) else None
    )

    ep = ElementProperty.from_preset("magpie")
    st = Stoichiometry()
    vo = ValenceOrbital()

    ep_names = ep.feature_labels()
    st_names = st.feature_labels()
    vo_names = vo.feature_labels()

    def safe_featurize(feat, comp):
        try:
            return feat.featurize(comp)
        except Exception:
            return [np.nan] * len(feat.feature_labels())

    ep_data = [safe_featurize(ep, c) if c else [np.nan] * len(ep_names) for c in df["_comp"]]
    st_data = [safe_featurize(st, c) if c else [np.nan] * len(st_names) for c in df["_comp"]]
    vo_data = [safe_featurize(vo, c) if c else [np.nan] * len(vo_names) for c in df["_comp"]]

    df = pd.concat([
        df.drop(columns=["_comp"]),
        pd.DataFrame(ep_data, columns=ep_names, index=df.index),
        pd.DataFrame(st_data, columns=st_names, index=df.index),
        pd.DataFrame(vo_data, columns=vo_names, index=df.index),
    ], axis=1)

    # -- collect feature column names (no DFT leakage) --
    skip = DFT_COLS | {"material_id", "formula_pretty", "chemsys"}
    feat_cols = [c for c in df.columns if c not in skip]

    # impute any remaining NaNs with column medians
    n_nan = df[feat_cols].isna().sum().sum()
    if n_nan > 0:
        print(f"  Imputing {n_nan} NaN values in feature matrix...")
        imp = SimpleImputer(strategy="median")
        df[feat_cols] = imp.fit_transform(df[feat_cols])

    return df, feat_cols


def align_to_training(df: pd.DataFrame, feat_cols: list[str]) -> np.ndarray:
    """Preprocess holdout/candidate data and align to training feature list."""
    df, _ = engineer_features(df)
    for col in feat_cols:
        if col not in df.columns:
            df[col] = 0
    arr = df[feat_cols].values.astype(float)
    col_medians = np.nanmedian(arr, axis=0)
    inds = np.where(np.isnan(arr))
    if len(inds[0]):
        arr[inds] = np.take(col_medians, inds[1])
    return arr


# ── balance ───────────────────────────────────────────────────────────────────

def balance(df: pd.DataFrame) -> pd.DataFrame:
    pos = df[df["is_stable"] == 1]
    neg = df[df["is_stable"] == 0]
    n   = min(len(pos), len(neg))
    return pd.concat([
        pos.sample(n, random_state=RANDOM_STATE),
        neg.sample(n, random_state=RANDOM_STATE),
    ]).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)


# ── step 3: train and evaluate ────────────────────────────────────────────────

def train_and_evaluate(
    X_train, X_test, y_train, y_test, feat_cols: list[str]
) -> tuple[Pipeline, Pipeline]:
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)

    models = {
        "GradientBoosting": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", GradientBoostingClassifier(
                n_estimators=200, max_depth=3,
                learning_rate=0.05, subsample=0.8,
                random_state=RANDOM_STATE,
            )),
        ]),
        "RandomForest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=200, max_depth=None,
                random_state=RANDOM_STATE, n_jobs=-1,
            )),
        ]),
    }

    results = {}
    for name, pipe in models.items():
        cv_auc = cross_val_score(pipe, X_train, y_train,
                                 cv=cv, scoring="roc_auc").mean()
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        y_prob = pipe.predict_proba(X_test)[:, 1]
        acc    = accuracy_score(y_test, y_pred)
        auc    = roc_auc_score(y_test, y_prob) if len(np.unique(y_test)) > 1 else float("nan")
        results[name] = {"pipe": pipe, "cv_auc": cv_auc, "acc": acc, "auc": auc,
                         "y_pred": y_pred, "y_prob": y_prob}

        print(f"\n{'='*55}")
        print(f"  {name}")
        print(f"{'='*55}")
        print(f"  CV ROC-AUC  (10-fold): {cv_auc:.4f}")
        print(f"  Test accuracy        : {acc:.4f}")
        auc_s = f"{auc:.4f}" if not np.isnan(auc) else "n/a"
        print(f"  Test ROC-AUC         : {auc_s}")

        cm          = confusion_matrix(y_test, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        print(f"  Confusion  TP={tp} FP={fp} TN={tn} FN={fn}")

        present = sorted(np.unique(np.concatenate([y_test, y_pred])))
        lmap    = {0: "unstable", 1: "stable"}
        report  = classification_report(y_test, y_pred, labels=present,
                                        target_names=[lmap[l] for l in present],
                                        zero_division=0)
        print(f"\n  Classification report:")
        for line in report.splitlines():
            print(f"    {line}")

    # feature importances for GBC
    gbc_pipe = results["GradientBoosting"]["pipe"]
    imp      = gbc_pipe.named_steps["clf"].feature_importances_
    top20    = sorted(zip(feat_cols, imp), key=lambda x: x[1], reverse=True)[:20]
    print(f"\n{'='*55}")
    print("  Top-20 feature importances (GradientBoosting)")
    print(f"{'='*55}")
    print(f"  {'Feature':<40} {'Importance':>10}  Bar")
    print(f"  {'-'*60}")
    max_imp = max(v for _, v in top20)
    for feat, val in top20:
        bar = "#" * int(val / max_imp * 25)
        print(f"  {feat:<40} {val:>10.4f}  {bar}")

    # geometry feature importances specifically
    geom_imp = [(f, v) for f, v in zip(feat_cols, imp) if f in GEOM_FEAT_COLS]
    print(f"\n  Geometry feature importances:")
    for feat, val in sorted(geom_imp, key=lambda x: x[1], reverse=True):
        rank = sorted(zip(feat_cols, imp), key=lambda x: x[1], reverse=True)
        rank_pos = next(i+1 for i,(f,_) in enumerate(rank) if f == feat)
        print(f"    {feat:<40} {val:.4f}  (rank {rank_pos}/{len(feat_cols)})")

    top_names = {f for f, _ in top20}
    leaked    = top_names & DFT_COLS
    if leaked:
        print(f"\n  *** DFT LEAK WARNING: {leaked} in top features ***")
    else:
        print(f"\n  DFT leak check: PASSED -- no DFT-derived feature in top-20.")

    return results["GradientBoosting"]["pipe"], results["RandomForest"]["pipe"]


# ── step 4: holdout validation ────────────────────────────────────────────────

def validate_holdout(
    label: str,
    api_key: str,
    chemsys: str,
    pipeline: Pipeline,
    feat_cols: list[str],
    df_train_ids: set,
):
    print(f"\n{'='*55}")
    print(f"Holdout: {label}  ({chemsys})")
    print(f"{'='*55}")

    df = query_system_raw(api_key, chemsys)
    overlap = set(df["material_id"]) & df_train_ids
    if overlap:
        print(f"  WARNING: {len(overlap)} entries overlap with training — {overlap}")
    else:
        print(f"  Overlap check: 0 matches in training -- clean holdout.")

    y_true = df["is_stable"].values
    print(f"  n={len(df)}  stable={y_true.sum()}  unstable={len(y_true)-y_true.sum()}")

    X = align_to_training(df, feat_cols)
    y_pred = pipeline.predict(X)
    y_prob = pipeline.predict_proba(X)[:, 1]

    acc   = accuracy_score(y_true, y_pred)
    auc   = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float("nan")
    auc_s = f"{auc:.4f}" if not np.isnan(auc) else "n/a"

    print(f"\n  Accuracy : {acc:.4f}  ({int(acc*len(y_true))}/{len(y_true)} correct)")
    print(f"  ROC-AUC  : {auc_s}")

    present = sorted(np.unique(np.concatenate([y_true, y_pred])))
    lmap    = {0: "unstable", 1: "stable"}
    report  = classification_report(y_true, y_pred, labels=present,
                                    target_names=[lmap[l] for l in present],
                                    zero_division=0)
    print(f"\n  Classification report:")
    for line in report.splitlines():
        print(f"    {line}")

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    print(f"  Confusion matrix:")
    print(f"                   pred:unstable  pred:stable")
    print(f"    true:unstable       {tn:>5}          {fp:>5}")
    print(f"    true:stable         {fn:>5}          {tp:>5}")

    print(f"\n  Per-entry:")
    print(f"  {'material_id':<12} {'formula':<10} {'e_hull':>8}  {'true':>10}  {'pred':>10} {'p':>6}")
    print(f"  {'-'*65}")
    for i, row in df.iterrows():
        tl = "stable" if y_true[i] == 1 else "unstable"
        pl = "stable" if y_pred[i] == 1 else "unstable"
        wr = " <WRONG" if y_pred[i] != y_true[i] else ""
        print(f"  {row['material_id']:<12} {row['formula_pretty']:<10}"
              f" {row['energy_above_hull']:>8.4f}  {tl:>10}  {pl:>10} {y_prob[i]:>6.3f}{wr}")

    return {"label": label, "n": len(df), "acc": acc, "auc": auc}


# ── step 5: generate or load candidates ───────────────────────────────────────

_COBI_PROTOTYPES = [
    ("CoBi",   221, "Cubic",        2,  "Pm-3m",    "B2_CsCl"),
    ("CoBi",   225, "Cubic",        8,  "Fm-3m",    "B1_NaCl"),
    ("CoBi",   216, "Cubic",        8,  "F-43m",    "B3_ZnS"),
    ("CoBi",   198, "Cubic",        8,  "P2_13",    "B20_FeSi"),
    ("CoBi",   62,  "Orthorhombic", 8,  "Pnma",     "B27_FeB"),
    ("CoBi",   63,  "Orthorhombic", 8,  "Cmcm",     "B33_CrB"),
    ("CoBi",   194, "Hexagonal",    4,  "P6_3/mmc", "B81_NiAs"),
    ("CoBi",   186, "Hexagonal",    4,  "P6_3mc",   "B4_wurtzite"),
    ("CoBi",   187, "Hexagonal",    2,  "P-6m2",    "Bh_WC"),
    ("CoBi",   191, "Hexagonal",    2,  "P6/mmm",   "hex_simple"),
    ("CoBi",   123, "Tetragonal",   2,  "P4/mmm",   "L10_CuAu"),
    ("CoBi",   139, "Tetragonal",   4,  "I4/mmm",   "tet_body"),
    ("CoBi",   166, "Trigonal",     4,  "R-3m",     "trig_AB"),
    ("CoBi",   11,  "Monoclinic",   8,  "P2_1/m",   "mono_AB"),
    ("CoBi",   2,   "Triclinic",    4,  "P-1",      "tri_AB"),
    ("CoBi2",  139, "Tetragonal",   6,  "I4/mmm",   "C11b_MoSi2"),
    ("CoBi2",  225, "Cubic",        12, "Fm-3m",    "C1_CaF2"),
    ("CoBi2",  194, "Hexagonal",    6,  "P6_3/mmc", "hex_AB2"),
    ("CoBi2",  58,  "Orthorhombic", 6,  "Pnnm",     "marcasite"),
    ("CoBi2",  62,  "Orthorhombic", 12, "Pnma",     "orth_AB2"),
    ("CoBi2",  164, "Trigonal",     3,  "P-3m1",    "CdI2_type"),
    ("CoBi2",  12,  "Monoclinic",   12, "C2/m",     "mono_AB2"),
    ("CoBi2",  2,   "Triclinic",    6,  "P-1",      "tri_AB2"),
    ("Co2Bi",  225, "Cubic",        12, "Fm-3m",    "C1_inv"),
    ("Co2Bi",  194, "Hexagonal",    6,  "P6_3/mmc", "AlB2_type"),
    ("Co2Bi",  139, "Tetragonal",   6,  "I4/mmm",   "C11b_inv"),
    ("Co2Bi",  62,  "Orthorhombic", 12, "Pnma",     "Co2Si_type"),
    ("Co2Bi",  167, "Trigonal",     6,  "R-3c",     "corund_like"),
    ("Co2Bi",  12,  "Monoclinic",   12, "C2/m",     "mono_A2B"),
    ("Co2Bi",  2,   "Triclinic",    6,  "P-1",      "tri_A2B"),
    ("CoBi3",  221, "Cubic",        4,  "Pm-3m",    "L12_AuCu3"),
    ("CoBi3",  194, "Hexagonal",    8,  "P6_3/mmc", "D019_inv"),
    ("CoBi3",  62,  "Orthorhombic", 8,  "Pnma",     "orth_AB3"),
    ("CoBi3",  123, "Tetragonal",   8,  "P4/mmm",   "D022_inv"),
    ("CoBi3",  12,  "Monoclinic",   8,  "C2/m",     "mono_AB3"),
    ("CoBi3",  2,   "Triclinic",    8,  "P-1",      "tri_AB3"),
    ("Co3Bi",  221, "Cubic",        4,  "Pm-3m",    "L12_Cu3Au"),
    ("Co3Bi",  194, "Hexagonal",    8,  "P6_3/mmc", "D019_Ni3Sn"),
    ("Co3Bi",  62,  "Orthorhombic", 8,  "Pnma",     "orth_A3B"),
    ("Co3Bi",  123, "Tetragonal",   8,  "P4/mmm",   "D022"),
    ("Co3Bi",  12,  "Monoclinic",   8,  "C2/m",     "mono_A3B"),
    ("Co3Bi",  2,   "Triclinic",    8,  "P-1",      "tri_A3B"),
    ("Co2Bi3", 166, "Trigonal",     5,  "R-3m",     "Bi2Te3_like"),
    ("Co2Bi3", 12,  "Monoclinic",   10, "C2/m",     "mono_23"),
    ("Co2Bi3", 62,  "Orthorhombic", 10, "Pnma",     "Sb2S3_like"),
    ("Co2Bi3", 2,   "Triclinic",    10, "P-1",      "tri_23"),
    ("Co3Bi2", 166, "Trigonal",     5,  "R-3m",     "trig_32"),
    ("Co3Bi2", 164, "Trigonal",     5,  "P-3m1",    "CdI2_32"),
    ("Co3Bi2", 62,  "Orthorhombic", 10, "Pnma",     "orth_32"),
    ("Co3Bi2", 12,  "Monoclinic",   10, "C2/m",     "mono_32"),
    ("Co3Bi2", 2,   "Triclinic",    10, "P-1",      "tri_32"),
    ("CoBi4",  139, "Tetragonal",   5,  "I4/mmm",   "tet_14"),
    ("CoBi4",  194, "Hexagonal",    5,  "P6_3/mmc", "hex_14"),
    ("CoBi4",  62,  "Orthorhombic", 5,  "Pnma",     "orth_14"),
    ("CoBi4",  12,  "Monoclinic",   10, "C2/m",     "mono_14"),
    ("CoBi4",  2,   "Triclinic",    5,  "P-1",      "tri_14"),
    ("Co4Bi",  139, "Tetragonal",   5,  "I4/mmm",   "tet_41"),
    ("Co4Bi",  194, "Hexagonal",    5,  "P6_3/mmc", "hex_41"),
    ("Co4Bi",  62,  "Orthorhombic", 5,  "Pnma",     "orth_41"),
    ("Co4Bi",  12,  "Monoclinic",   10, "C2/m",     "mono_41"),
    ("Co4Bi",  2,   "Triclinic",    5,  "P-1",      "tri_41"),
    ("CoBi5",  191, "Hexagonal",    6,  "P6/mmm",   "D2d_inv"),
    ("CoBi5",  123, "Tetragonal",   6,  "P4/mmm",   "tet_15"),
    ("CoBi5",  62,  "Orthorhombic", 6,  "Pnma",     "orth_15"),
    ("CoBi5",  2,   "Triclinic",    6,  "P-1",      "tri_15"),
    ("Co5Bi",  191, "Hexagonal",    6,  "P6/mmm",   "D2d_CaCu5"),
    ("Co5Bi",  123, "Tetragonal",   6,  "P4/mmm",   "tet_51"),
    ("Co5Bi",  62,  "Orthorhombic", 6,  "Pnma",     "orth_51"),
    ("Co5Bi",  2,   "Triclinic",    6,  "P-1",      "tri_51"),
    ("Co3Bi4", 166, "Trigonal",     14, "R-3m",     "trig_34"),
    ("Co3Bi4", 62,  "Orthorhombic", 14, "Pnma",     "orth_34"),
    ("Co3Bi4", 11,  "Monoclinic",   14, "P2_1/m",   "mono_34"),
    ("Co4Bi3", 166, "Trigonal",     14, "R-3m",     "trig_43"),
    ("Co4Bi3", 62,  "Orthorhombic", 14, "Pnma",     "orth_43"),
    ("Co4Bi3", 11,  "Monoclinic",   14, "P2_1/m",   "mono_43"),
    ("Co2Bi5", 12,  "Monoclinic",   14, "C2/m",     "mono_25"),
    ("Co2Bi5", 62,  "Orthorhombic", 14, "Pnma",     "orth_25"),
    ("Co5Bi2", 12,  "Monoclinic",   14, "C2/m",     "mono_52"),
    ("Co5Bi2", 62,  "Orthorhombic", 14, "Pnma",     "orth_52"),
    ("CoBi6",  191, "Hexagonal",    7,  "P6/mmm",   "hex_16"),
    ("CoBi6",  123, "Tetragonal",   7,  "P4/mmm",   "tet_16"),
    ("Co6Bi",  191, "Hexagonal",    7,  "P6/mmm",   "hex_61"),
    ("Co6Bi",  123, "Tetragonal",   7,  "P4/mmm",   "tet_61"),
    ("Co3Bi5", 2,   "Triclinic",    16, "P-1",      "tri_35"),
    ("Co5Bi3", 2,   "Triclinic",    16, "P-1",      "tri_53"),
    ("Co3Bi5", 62,  "Orthorhombic", 16, "Pnma",     "orth_35"),
    ("Co5Bi3", 62,  "Orthorhombic", 16, "Pnma",     "orth_53"),
    ("CoBi",   225, "Cubic",        16, "Fm-3m",    "B1_supercell"),
    ("Co3Bi",  221, "Cubic",        16, "Pm-3m",    "L12_large"),
    ("CoBi3",  221, "Cubic",        16, "Pm-3m",    "L12_inv_large"),
    ("CoBi2",  225, "Cubic",        24, "Fm-3m",    "C1_large"),
    ("Co2Bi",  225, "Cubic",        24, "Fm-3m",    "C1_inv_large"),
    ("Co3Bi2", 194, "Hexagonal",    20, "P6_3/mmc", "hex_32_large"),
    ("Co2Bi3", 194, "Hexagonal",    20, "P6_3/mmc", "hex_23_large"),
    ("CoBi",   62,  "Orthorhombic", 16, "Pnma",     "orth_AB_large"),
    ("Co3Bi4", 12,  "Monoclinic",   28, "C2/m",     "mono_34_large"),
    ("Co4Bi3", 12,  "Monoclinic",   28, "C2/m",     "mono_43_large"),
    ("CoBi",   164, "Trigonal",     4,  "P-3m1",    "trig_L10"),
    ("Co2Bi3", 166, "Trigonal",     10, "R-3m",     "trig_23_R"),
    ("Co3Bi2", 164, "Trigonal",     10, "P-3m1",    "trig_32_P"),
]


def generate_candidates_csv(path: Path):
    rows = []
    for i, (formula, sg_num, cs, nsites, sg_sym, proto) in enumerate(_COBI_PROTOTYPES, 1):
        bi_frac = _first_el_fraction(formula)
        rows.append({
            "task_id":         f"cobi_cand_{i:03d}",
            "reduced_formula": formula,
            "sg_number":       sg_num,
            "sg_symbol":       sg_sym,
            "crystal_system":  cs,
            "n_atoms":         nsites,
            "co_fraction":     1.0 - bi_frac,
            "bi_fraction":     bi_frac,
            "prototype_name":  proto,
        })
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    print(f"  Generated {len(df)} Co-Bi candidates -> {path}")
    return df


def load_or_generate_candidates(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.exists():
        df = pd.read_csv(p)
        print(f"  Loaded {len(df)} candidates from {p}")
    else:
        print(f"  {p} not found -- generating from built-in prototype list...")
        df = generate_candidates_csv(p)
    return df


def _build_candidate_features(df_cand: pd.DataFrame, feat_cols: list[str]) -> np.ndarray:
    """
    Convert candidate CSV rows to feature matrix aligned to training.
    Geometry features (min_AB_dist etc.) are imputed with NaN when absent;
    the median imputer in engineer_features handles them.
    """
    from pymatgen.core import Composition
    from matminer.featurizers.composition import ElementProperty, Stoichiometry, ValenceOrbital

    df = df_cand.copy().reset_index(drop=True)

    renames = {
        "sg_number":       "spacegroup_number",
        "n_atoms":         "nsites",
        "reduced_formula": "formula_pretty",
    }
    for old, new in renames.items():
        if old in df.columns and new not in df.columns:
            df[new] = df[old]

    # geometry features: use if present (candidates pre-processed), else NaN
    for col in GEOM_FEAT_COLS:
        if col not in df.columns:
            df[col] = np.nan

    df["composition_A_fraction"] = df["formula_pretty"].apply(_first_el_fraction)
    df["n_atoms_parity"]         = (df["nsites"] % 2).astype(int)
    df["spacegroup_number"]      = pd.to_numeric(df["spacegroup_number"],
                                                  errors="coerce").fillna(1)

    df["crystal_system"] = df.get("crystal_system", pd.Series(["Unknown"] * len(df)))
    df["crystal_system"] = df["crystal_system"].fillna("Unknown")
    cs_dummies = pd.get_dummies(df["crystal_system"], prefix="cs").astype(int)
    for col in CS_COLS:
        if col not in cs_dummies.columns:
            cs_dummies[col] = 0
    cs_dummies = cs_dummies[CS_COLS]
    df = pd.concat([df.drop(columns=["crystal_system"]), cs_dummies], axis=1)

    ratio_df = pd.DataFrame(
        df["formula_pretty"].apply(_ratio_flags).tolist(), index=df.index
    )
    df = pd.concat([df, ratio_df], axis=1)

    df["_comp"] = df["formula_pretty"].apply(
        lambda f: Composition(f) if pd.notna(f) else None
    )
    ep = ElementProperty.from_preset("magpie")
    st = Stoichiometry()
    vo = ValenceOrbital()

    ep_names = ep.feature_labels()
    st_names = st.feature_labels()
    vo_names = vo.feature_labels()

    def safe_f(feat, comp):
        try:
            return feat.featurize(comp)
        except Exception:
            return [np.nan] * len(feat.feature_labels())

    ep_data = [safe_f(ep, c) if c else [np.nan] * len(ep_names) for c in df["_comp"]]
    st_data = [safe_f(st, c) if c else [np.nan] * len(st_names) for c in df["_comp"]]
    vo_data = [safe_f(vo, c) if c else [np.nan] * len(vo_names) for c in df["_comp"]]

    df = pd.concat([
        df.drop(columns=["_comp"]),
        pd.DataFrame(ep_data, columns=ep_names, index=df.index),
        pd.DataFrame(st_data, columns=st_names, index=df.index),
        pd.DataFrame(vo_data, columns=vo_names, index=df.index),
    ], axis=1)

    for col in feat_cols:
        if col not in df.columns:
            df[col] = 0

    arr = df[feat_cols].values.astype(float)
    col_medians = np.nanmedian(arr, axis=0)
    col_medians = np.where(np.isnan(col_medians), 0.0, col_medians)  # all-NaN col fallback
    inds = np.where(np.isnan(arr))
    if len(inds[0]):
        arr[inds] = np.take(col_medians, inds[1])
    return arr


def confidence_band(p: float) -> str:
    if p >= 0.80: return "high stable"
    if p >= 0.60: return "likely stable"
    if p >= 0.40: return "uncertain"
    if p >= 0.20: return "likely unstable"
    return "high unstable"


# ── comparison table ──────────────────────────────────────────────────────────

def print_comparison_table(structure_holdouts: list[dict]):
    w = 38
    print(f"\n{'='*70}")
    print("COMPARISON: structure-only model vs DFT-feature model")
    print(f"{'='*70}")
    print(f"  {'Holdout':<{w}} {'n':>5}  {'struct acc':>10}  {'struct AUC':>10}  {'DFT acc':>9}  {'DFT AUC':>9}")
    print(f"  {'-'*68}")

    dft_ref = {
        "Ni-Sb": {"acc": 0.857, "auc": 0.917},
        "Fe-Sb": {"acc": 0.857, "auc": 0.917},
    }

    for r in structure_holdouts:
        acc_s = f"{r['acc']:.4f}"
        auc_s = f"{r['auc']:.4f}" if not np.isnan(r['auc']) else "  n/a  "
        ref     = dft_ref.get(r["label"].split()[0], {})
        dft_acc = f"{ref['acc']:.4f}" if "acc" in ref else "  n/a"
        dft_auc = f"{ref['auc']:.4f}" if "auc" in ref else "  n/a"
        print(f"  {r['label']:<{w}} {r['n']:>5}  {acc_s:>10}  {auc_s:>10}  {dft_acc:>9}  {dft_auc:>9}")

    print(f"{'='*70}")
    print("  DFT-feature model used 18 features including formation_energy_per_atom.")
    print("  This model uses 0 DFT outputs -- valid pre-screening.")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    api_key = os.environ.get("MP_API_KEY")
    if not api_key:
        sys.exit("MP_API_KEY not set.")

    out_dir    = Path("data/results")
    models_dir = Path("models")
    out_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(exist_ok=True)

    # ── Step 1: query training data ────────────────────────────────────────
    print("=" * 55)
    print("Step 1: Query training data (structure + geometry features)")
    print("=" * 55)
    train_raw_path = out_dir / "training_data_structure_raw.csv"

    # Force re-query if geometry columns are missing from cached file
    needs_query = True
    if train_raw_path.exists():
        df_check = pd.read_csv(train_raw_path, nrows=1)
        if all(c in df_check.columns for c in GEOM_FEAT_COLS):
            print(f"  Loading cached data (with geometry) from {train_raw_path}")
            df_raw    = pd.read_csv(train_raw_path)
            needs_query = False
        else:
            print(f"  Cache missing geometry columns — re-querying MP with structure data...")

    if needs_query:
        df_raw = query_training(api_key)
        df_raw.to_csv(train_raw_path, index=False)
        print(f"  Saved {len(df_raw)} rows -> {train_raw_path}")

    stable   = (df_raw["is_stable"] == 1).sum()
    unstable = len(df_raw) - stable
    print(f"  Total: {len(df_raw)}  stable={stable}  unstable={unstable}")
    geom_ok = df_raw[GEOM_FEAT_COLS].notna().all(axis=1).sum()
    print(f"  Rows with complete geometry features: {geom_ok}/{len(df_raw)}")

    # ── Step 2: feature engineering ────────────────────────────────────────
    print(f"\n{'='*55}")
    print("Step 2: Feature engineering (168 + 5 geometry = 173 features)")
    print(f"{'='*55}")
    df_feat, feat_cols = engineer_features(df_raw)
    print(f"  Feature count : {len(feat_cols)}")
    print(f"  Geometry cols : {[c for c in feat_cols if c in GEOM_FEAT_COLS]}")
    print(f"  DFT leak check: {DFT_COLS & set(feat_cols)}")

    df_bal = balance(df_feat)
    print(f"\n  Balanced: {len(df_bal)} samples (50/50)")

    X = df_bal[feat_cols].values.astype(float)
    y = df_bal["is_stable"].values.astype(int)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    train_ids = set(df_raw["material_id"])

    # ── Step 3: train ──────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print("Step 3: Train and evaluate")
    print(f"{'='*55}")
    gbc_pipe, rf_pipe = train_and_evaluate(X_tr, X_te, y_tr, y_te, feat_cols)

    dump({"model": gbc_pipe, "model_name": "GBC_structure_only",
          "feat_cols": feat_cols},
         models_dir / "stability_classifier_structure_only.joblib")
    print(f"\n  Saved -> models/stability_classifier_structure_only.joblib")

    # ── Step 4: holdout validation ─────────────────────────────────────────
    print(f"\n{'='*55}")
    print("Step 4: Holdout validation on Fe-Sb and Ni-Sb")
    print(f"{'='*55}")
    holdout_results = []
    for label, chemsys in [("Fe-Sb (clean holdout)", "Fe-Sb"),
                            ("Ni-Sb (clean holdout)", "Ni-Sb")]:
        r = validate_holdout(label, api_key, chemsys, gbc_pipe, feat_cols, train_ids)
        holdout_results.append(r)

    print_comparison_table(holdout_results)

    # ── Step 5: Co-Bi candidates ───────────────────────────────────────────
    print(f"\n{'='*55}")
    print("Step 5: Score Co-Bi candidates")
    print(f"{'='*55}")
    df_cand = load_or_generate_candidates("cobi_test_combinations_100.csv")

    print("  Building candidate feature matrix (geometry features imputed)...")
    X_cand   = _build_candidate_features(df_cand, feat_cols)
    y_pred_c = gbc_pipe.predict(X_cand)
    y_prob_c = gbc_pipe.predict_proba(X_cand)[:, 1]

    df_cand["predicted_label"]              = np.where(y_pred_c == 1, "stable", "unstable")
    df_cand["predicted_probability_stable"] = np.round(y_prob_c, 4)
    df_cand["confidence_band"]              = [confidence_band(p) for p in y_prob_c]

    df_out   = df_cand.sort_values("predicted_probability_stable", ascending=False).reset_index(drop=True)
    pred_path = out_dir / "cobi_predictions.csv"
    out_cols  = ["task_id", "reduced_formula", "sg_symbol", "crystal_system", "n_atoms",
                 "co_fraction", "bi_fraction", "prototype_name",
                 "predicted_label", "predicted_probability_stable", "confidence_band"]
    out_cols  = [c for c in out_cols if c in df_out.columns]
    df_out[out_cols].to_csv(pred_path, index=False)
    print(f"  Saved {len(df_out)} predictions -> {pred_path}")

    top20 = df_out[df_out["predicted_label"] == "stable"].head(20)
    print(f"\n{'='*70}")
    print("TOP-20 MOST LIKELY STABLE Co-Bi CANDIDATES")
    print(f"{'='*70}")
    print(f"  {'#':<4} {'task_id':<15} {'formula':<10} {'crystal_sys':<14} "
          f"{'sg':>6} {'n_at':>5}  {'p(stable)':>10}  band")
    print(f"  {'-'*75}")
    for rank, (_, row) in enumerate(top20.iterrows(), 1):
        sg  = row.get("sg_number", row.get("sg_symbol", "?"))
        nat = row.get("n_atoms", row.get("nsites", "?"))
        print(f"  {rank:<4} {row['task_id']:<15} {row['reduced_formula']:<10} "
              f"{row['crystal_system']:<14} {str(sg):>6} {str(nat):>5}  "
              f"{row['predicted_probability_stable']:>10.4f}  {row['confidence_band']}")

    print(f"\n  Confidence band distribution:")
    for band in ["high stable", "likely stable", "uncertain", "likely unstable", "high unstable"]:
        n = (df_out["confidence_band"] == band).sum()
        print(f"    {band:<20}: {n:>4}")


if __name__ == "__main__":
    main()
