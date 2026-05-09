#!/usr/bin/env python3
"""
screen_candidates.py

Reusable CLI wrapper around the trained structure-only stability classifier.
Accepts a metadata CSV + a directory of CIF files, scores every candidate,
and copies the shortlisted CIFs to the output directory for downstream MLIP.

Usage:
    python screen_candidates.py METADATA_CSV CIF_DIR [options]

Examples:
    python screen_candidates.py metadata.csv cifs/
    python screen_candidates.py metadata.csv cifs/ --top-n 100
    python screen_candidates.py metadata.csv cifs/ --skip-cif-copy --output-dir dry_run/
    python screen_candidates.py metadata.csv cifs/ --top-n 20 --max-per-composition 3
    python screen_candidates.py metadata.csv cifs/ --no-diversity --output-dir results/run1/
"""

import argparse
import json
import shutil
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import load as jload
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ── constants ──────────────────────────────────────────────────────────────────

REQUIRED_COLS = ["cif_filename", "formula", "space_group", "n_atoms", "n_Co", "n_Bi"]

# Metadata columns that pass through to outputs unchanged (not model features)
PASSTHROUGH_COLS = ["volume", "density", "fingerprint"]

# Geometry features — computed by preprocess_candidates.py, used as model inputs.
# Co-Bi specific names are mapped to generic names in derive_columns() so they
# align with how the model was trained (alphabetically sorted element pairs).
# A = alphabetically first element = Bi in Co-Bi
# B = alphabetically second element = Co in Co-Bi
GEOM_FEAT_COLS = [
    "min_AB_dist",       # min Bi-Co distance  (← min_CoBi in metadata)
    "min_AA_dist",       # min Bi-Bi distance  (← min_BiBi in metadata)
    "min_BB_dist",       # min Co-Co distance  (← min_CoCo in metadata)
    "packing_fraction",  # hard-sphere packing (← packing_fraction in metadata)
    "volume_per_atom",   # cell vol / n_atoms  (← volume_per_atom in metadata)
]

CS_COLS = [
    "cs_Cubic", "cs_Hexagonal", "cs_Monoclinic", "cs_Orthorhombic",
    "cs_Tetragonal", "cs_Triclinic", "cs_Trigonal",
]

RATIO_FLAGS = [
    "ratio_1_1", "ratio_1_2", "ratio_2_1", "ratio_1_3", "ratio_3_1",
    "ratio_2_3", "ratio_3_2", "ratio_1_4", "ratio_4_1", "ratio_1_5", "ratio_5_1",
]

BAND_ORDER = ["high stable", "likely stable", "uncertain",
              "likely unstable", "high unstable"]

BATCH_SIZE = 500  # rows per featurization batch for large files


# ── step 1: load model and feature list ───────────────────────────────────────

def load_model_and_features(model_path: str, features_path: str):
    """
    Load the trained sklearn Pipeline and the ordered feature column list.

    Returns (pipeline, feat_cols: list[str]).
    Raises FileNotFoundError with a clear message if either file is missing.
    """
    mp = Path(model_path)
    fp = Path(features_path)

    if not mp.exists():
        raise FileNotFoundError(
            f"Model not found: {mp}\n"
            f"Run save_model.py to export the trained model first."
        )
    if not fp.exists():
        raise FileNotFoundError(
            f"Feature column file not found: {fp}\n"
            f"Run save_model.py to export the feature list first."
        )

    pipeline  = jload(mp)
    with open(fp) as f:
        feat_cols = json.load(f)

    return pipeline, feat_cols


# ── step 1b: derive columns from new metadata format ─────────────────────────

def derive_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute derived columns from the new metadata CSV format so that the
    rest of the pipeline sees the same internal column names as before.

    Adds:
        task_id          -- cif_filename without .cif extension
        reduced_formula  -- pymatgen reduced formula from full formula
        crystal_system   -- capitalized crystal system from space_group number
        co_fraction      -- n_Co / n_atoms
        sg_number        -- alias for space_group (downstream diversity code)
        formula_pretty   -- alias for reduced_formula (featurizer input)
        nsites           -- alias for n_atoms (featurizer input)
        spacegroup_number-- alias for space_group (featurizer input)
    """
    from pymatgen.core import Composition
    from pymatgen.symmetry.groups import SpaceGroup

    df = df.copy()

    df["task_id"] = df["cif_filename"].str.replace(r"\.cif$", "", regex=True)

    def _reduced(formula):
        try:
            return Composition(formula).reduced_formula
        except Exception:
            return formula

    def _crystal_system(sg_num):
        try:
            return SpaceGroup.from_int_number(int(sg_num)).crystal_system.capitalize()
        except Exception:
            return "Unknown"

    df["reduced_formula"] = df["formula"].apply(_reduced)
    df["crystal_system"]  = df["space_group"].apply(_crystal_system)
    df["co_fraction"]     = df["n_Co"] / df["n_atoms"]

    # internal aliases expected by featurize() and diversified_top_n()
    df["sg_number"]         = df["space_group"]
    df["formula_pretty"]    = df["reduced_formula"]
    df["nsites"]            = df["n_atoms"]
    df["spacegroup_number"] = df["space_group"]

    # Map Co-Bi specific distance column names to the generic names used during
    # training (alphabetically sorted: A=Bi, B=Co for the Co-Bi system).
    # If any source column is missing the generic column is set to NaN and
    # imputed with the training median inside featurize().
    col_map = {
        "min_CoBi":          "min_AB_dist",
        "min_BiBi":          "min_AA_dist",
        "min_CoCo":          "min_BB_dist",
        # packing_fraction and volume_per_atom keep the same name
    }
    for src, dst in col_map.items():
        if src in df.columns:
            df[dst] = pd.to_numeric(df[src], errors="coerce")
        elif dst not in df.columns:
            df[dst] = np.nan

    for col in ["packing_fraction", "volume_per_atom"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = np.nan

    return df


# ── step 1c: validate CIF files ───────────────────────────────────────────────

def validate_cif_files(df: pd.DataFrame, cif_dir: Path,
                       output_dir: Path) -> tuple[pd.DataFrame, list[str]]:
    """
    Check that every CIF referenced in df exists on disk.

    Adds a boolean `cif_present` column to df.
    Writes missing filenames to {output_dir}/missing_cifs.txt.

    Returns (df_with_flag, missing_filenames).
    """
    present = []
    missing = []
    for fname in df["cif_filename"]:
        if (cif_dir / fname).exists():
            present.append(True)
        else:
            present.append(False)
            missing.append(fname)

    df = df.copy()
    df["cif_present"] = present

    if missing:
        txt_path = output_dir / "missing_cifs.txt"
        txt_path.write_text("\n".join(missing), encoding="utf-8")
        print(f"  WARNING: {len(missing)} CIF files not found -> {txt_path}")

    return df, missing


# ── step 7b: copy shortlisted CIFs ───────────────────────────────────────────

def copy_shortlist_cifs(shortlist: pd.DataFrame,
                        cif_dir: Path, output_dir: Path) -> int:
    """
    Copy each shortlisted candidate's CIF file to {output_dir}/shortlist_cifs/.
    Skips rows where cif_present is False.

    Returns the number of files successfully copied.
    """
    dest_dir = output_dir / "shortlist_cifs"
    dest_dir.mkdir(exist_ok=True)

    copied = 0
    for _, row in shortlist.iterrows():
        if not row.get("cif_present", True):
            continue
        src = cif_dir / row["cif_filename"]
        shutil.copy2(src, dest_dir / row["cif_filename"])
        copied += 1

    return copied


# ── step 2: validate input ────────────────────────────────────────────────────

def validate_input(df: pd.DataFrame) -> None:
    """
    Check that all required columns are present.
    Raises SystemExit with a clear error listing missing columns.
    """
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        sys.exit(
            f"ERROR: Input CSV is missing required columns: {missing}\n"
            f"Required: {REQUIRED_COLS}\n"
            f"Found:    {df.columns.tolist()}"
        )
    if len(df) == 0:
        sys.exit("ERROR: Input CSV is empty.")


# ── step 3: featurize ─────────────────────────────────────────────────────────

def _composition_A_fraction(formula: str) -> float:
    """Fraction of the alphabetically first element — matches training convention."""
    from pymatgen.core import Composition
    try:
        comp  = Composition(formula)
        first = sorted(comp.elements, key=lambda e: e.symbol)[0]
        return float(comp.get_atomic_fraction(first))
    except Exception:
        return np.nan


def _ratio_flags(formula: str) -> dict:
    """Boolean stoichiometry-ratio prototype flags."""
    from pymatgen.core import Composition
    try:
        comp   = Composition(formula)
        counts = sorted(comp.as_dict().values())
        a, b   = (sorted([counts[0], counts[1]]) if len(counts) >= 2
                  else (counts[0], counts[0]))
        return {
            "ratio_1_1": int(a == b),
            "ratio_1_2": int(b == 2 * a),
            "ratio_2_1": int(b == 2 * a),
            "ratio_1_3": int(b == 3 * a),
            "ratio_3_1": int(b == 3 * a),
            "ratio_2_3": int(b * 2 == a * 3 or a * 2 == b * 3),
            "ratio_3_2": int(a * 3 == b * 2 or b * 3 == a * 2),
            "ratio_1_4": int(b == 4 * a),
            "ratio_4_1": int(b == 4 * a),
            "ratio_1_5": int(b == 5 * a),
            "ratio_5_1": int(b == 5 * a),
        }
    except Exception:
        return {f: 0 for f in RATIO_FLAGS}


def _featurize_one_composition(formula: str, ep, st, vo) -> dict | None:
    """
    Compute all matminer compositional features for a single formula string.
    Returns a flat dict of {feature_name: value}, or None on failure.
    """
    from pymatgen.core import Composition
    try:
        comp = Composition(formula)
        row  = {}
        row.update(zip(ep.feature_labels(), ep.featurize(comp)))
        row.update(zip(st.feature_labels(), st.featurize(comp)))
        row.update(zip(vo.feature_labels(), vo.featurize(comp)))
        row.update(_ratio_flags(formula))
        row["composition_A_fraction"] = _composition_A_fraction(formula)
        return row
    except Exception:
        return None


def featurize(df: pd.DataFrame, feat_cols: list[str],
              error_log: list) -> np.ndarray:
    """
    Build the feature matrix from a dataframe of candidates.

    Featurization is composition-cached: each unique formula is featurized
    once regardless of how many rows share it. Rows that fail featurization
    are recorded in `error_log` (list of dicts, mutated in place) and receive
    all-zero compositional features so the pipeline can still score them.

    Progress is reported via tqdm at the unique-composition level.

    Returns X of shape (n_rows, len(feat_cols)).
    """
    from matminer.featurizers.composition import (
        ElementProperty, Stoichiometry, ValenceOrbital,
    )

    ep = ElementProperty.from_preset("magpie")
    st = Stoichiometry()
    vo = ValenceOrbital()

    n_comp_features = len(ep.feature_labels()) + len(st.feature_labels()) + \
                      len(vo.feature_labels()) + len(RATIO_FLAGS) + 1  # +A_frac

    # -- cache compositional features per unique formula --
    unique_formulas = df["formula_pretty"].dropna().unique()
    comp_cache: dict[str, dict] = {}

    for formula in tqdm(unique_formulas, desc="Featurizing compositions", unit="formula"):
        result = _featurize_one_composition(formula, ep, st, vo)
        if result is None:
            error_log.append({"formula": formula, "error": "featurization failed"})
            comp_cache[formula] = {}
        else:
            comp_cache[formula] = result

    # -- structural features (per row, cheap) --
    df = df.copy().reset_index(drop=True)
    df["spacegroup_number"] = pd.to_numeric(df["spacegroup_number"],
                                             errors="coerce").fillna(1)
    df["n_atoms_parity"]    = (df["nsites"] % 2).astype(int)

    # one-hot crystal system
    df["crystal_system"] = df["crystal_system"].fillna("Unknown")
    cs_dummies = pd.get_dummies(df["crystal_system"], prefix="cs").astype(int)
    for col in CS_COLS:
        if col not in cs_dummies.columns:
            cs_dummies[col] = 0
    cs_dummies = cs_dummies[CS_COLS]
    df = pd.concat([df.drop(columns=["crystal_system"]), cs_dummies], axis=1)

    # -- merge compositional cache into dataframe --
    comp_df = pd.DataFrame(
        [comp_cache.get(f, {}) for f in df["formula_pretty"]],
        index=df.index,
    )
    df = pd.concat([df, comp_df], axis=1)

    # -- ensure geometry features are present (NaN if not computed) --
    for col in GEOM_FEAT_COLS:
        if col not in df.columns:
            df[col] = np.nan

    # -- align to training feature order --
    for col in feat_cols:
        if col not in df.columns:
            df[col] = 0

    arr = df[feat_cols].values.astype(float)

    # impute NaN with column medians
    col_medians = np.nanmedian(arr, axis=0)
    nan_r, nan_c = np.where(np.isnan(arr))
    if len(nan_r):
        arr[nan_r, nan_c] = col_medians[nan_c]

    return arr


# ── step 4: predict ───────────────────────────────────────────────────────────

def predict(pipeline, X: np.ndarray) -> np.ndarray:
    """
    Run inference and return p(stable) for each row.
    Processes in batches to avoid memory issues on very large inputs.
    """
    n      = len(X)
    probs  = np.empty(n, dtype=float)
    n_batches = max(1, n // BATCH_SIZE)

    for start in tqdm(range(0, n, BATCH_SIZE), desc="Scoring", unit="batch",
                      total=n_batches + (1 if n % BATCH_SIZE else 0)):
        end             = min(start + BATCH_SIZE, n)
        probs[start:end] = pipeline.predict_proba(X[start:end])[:, 1]

    return probs


# ── step 5: confidence bands ──────────────────────────────────────────────────

def assign_confidence_bands(probs: np.ndarray) -> list[str]:
    """Map probability scores to labelled confidence bands."""
    def _band(p: float) -> str:
        if p >= 0.80: return "high stable"
        if p >= 0.60: return "likely stable"
        if p >= 0.40: return "uncertain"
        if p >= 0.20: return "likely unstable"
        return "high unstable"

    return [_band(p) for p in probs]


# ── step 6: diversified top-N selection ──────────────────────────────────────

def diversified_top_n(df: pd.DataFrame, top_n: int,
                      max_per_comp: int, max_per_sg: int) -> pd.DataFrame:
    """
    Greedy diversity-aware selection.

    Iterates candidates sorted by p_stable descending and accepts each one
    only if neither its reduced_formula count nor its sg_number count has
    reached its cap. Stops when the shortlist reaches top_n or all candidates
    are exhausted.

    Returns the shortlist as a DataFrame with a rank_shortlist column.
    """
    counts_comp: dict[str, int] = {}
    counts_sg:   dict[str, int] = {}
    accepted: list[int] = []

    for idx, row in df.sort_values("p_stable", ascending=False).iterrows():
        comp = str(row["reduced_formula"])
        sg   = str(row["sg_number"])
        if (counts_comp.get(comp, 0) < max_per_comp and
                counts_sg.get(sg, 0) < max_per_sg):
            accepted.append(idx)
            counts_comp[comp] = counts_comp.get(comp, 0) + 1
            counts_sg[sg]     = counts_sg.get(sg, 0) + 1
        if len(accepted) >= top_n:
            break

    shortlist = df.loc[accepted].copy().reset_index(drop=True)
    shortlist["rank_shortlist"] = shortlist.index + 1
    return shortlist


# ── step 7: summary ───────────────────────────────────────────────────────────

def print_and_save_summary(df_full: pd.DataFrame, shortlist: pd.DataFrame,
                           output_dir: Path, args,
                           n_cifs_copied: int = 0,
                           n_missing_cifs: int = 0) -> None:
    """
    Print a structured summary to stdout and save it to summary.txt.
    Includes CIF copy stats in Section F.
    """
    lines: list[str] = []

    def emit(*parts, **kwargs):
        text = " ".join(str(p) for p in parts)
        print(text, **kwargs)
        lines.append(text)

    top_n = args.top_n
    sep   = lambda n=70: emit("-" * n)
    hdr   = lambda t, n=70: (emit("=" * n), emit(t), emit("=" * n))

    hdr("SCREENING SUMMARY")
    emit(f"  Input metadata   : {args.metadata_csv}")
    emit(f"  CIF directory    : {args.cif_dir}")
    emit(f"  Model            : {args.model}")
    emit(f"  Total candidates : {len(df_full)}")
    emit(f"  Shortlist size   : {len(shortlist)} "
         f"({'no-diversity, raw top-N' if args.no_diversity else f'diversified top-{top_n}'})")

    # -- Band distribution --
    emit()
    hdr("SECTION A -- Confidence band distribution")
    emit(f"  {'Band':<20}  {'Count':>6}  {'Pct':>7}  Bar")
    sep()
    for band in BAND_ORDER:
        n   = (df_full["confidence_band"] == band).sum()
        pct = n / len(df_full) * 100
        bar = "#" * int(pct / 2)
        emit(f"  {band:<20}  {n:>6}  {pct:>6.1f}%  {bar}")
    sep()
    emit(f"  {'TOTAL':<20}  {len(df_full):>6}  100.0%")

    # -- Top 10 shortlist preview --
    emit()
    hdr("SECTION B -- Top 10 shortlist preview")
    col_hdr = (f"  {'SL#':<4} {'rank':>5} {'task_id':<18} {'formula':<12} "
               f"{'crystal_sys':<14} {'sg':>6} {'n_at':>5}  {'p(stable)':>10}  band")
    emit(col_hdr)
    sep(90)
    for _, row in shortlist.head(10).iterrows():
        emit(f"  {row['rank_shortlist']:<4} {row['rank_overall']:>5} "
             f"{str(row.get('task_id','')):<18} {str(row['reduced_formula']):<12} "
             f"{str(row.get('crystal_system','')):<14} "
             f"{str(row.get('sg_number',''))[:6]:>6} "
             f"{int(row['nsites']):>5}  {row['p_stable']:>10.4f}  "
             f"{row['confidence_band']}")
    sep(90)

    # -- Composition distribution in shortlist --
    emit()
    hdr("SECTION C -- Composition distribution in shortlist")
    comp_grp = (shortlist.groupby("reduced_formula")["p_stable"]
                .agg(count="count", mean_p="mean")
                .sort_values("count", ascending=False))
    emit(f"  {'formula':<14}  {'count':>6}  {'mean p(stable)':>14}")
    sep(42)
    for formula, row in comp_grp.iterrows():
        bar = "#" * int(row["mean_p"] * 15)
        emit(f"  {formula:<14}  {row['count']:>6}  {row['mean_p']:>14.4f}  {bar}")

    # -- Crystal system distribution in shortlist --
    emit()
    hdr("SECTION D -- Crystal system distribution in shortlist")
    cs_grp = (shortlist.groupby("crystal_system")["p_stable"]
              .agg(count="count", mean_p="mean")
              .sort_values("count", ascending=False))
    emit(f"  {'crystal_system':<16}  {'count':>6}  {'mean p(stable)':>14}")
    sep(44)
    for cs, row in cs_grp.iterrows():
        bar = "#" * int(row["mean_p"] * 15)
        emit(f"  {cs:<16}  {row['count']:>6}  {row['mean_p']:>14.4f}  {bar}")

    # -- Diversity stats --
    if not args.no_diversity:
        emit()
        hdr("SECTION E -- Diversity stats")
        naive_top = df_full.sort_values("p_stable", ascending=False).head(top_n)
        naive_sg_counts  = naive_top["sg_number"].value_counts()
        naive_comp_counts = naive_top["reduced_formula"].value_counts()
        top_sg_naive   = naive_sg_counts.index[0]
        top_sg_n_naive = naive_sg_counts.iloc[0]
        top_comp_naive   = naive_comp_counts.index[0]
        top_comp_n_naive = naive_comp_counts.iloc[0]

        div_sg_counts  = shortlist["sg_number"].value_counts()
        div_comp_counts = shortlist["reduced_formula"].value_counts()

        emit(f"  Without diversity constraints (naive top-{top_n}):")
        emit(f"    Most common sg:      sg {top_sg_naive} -- {top_sg_n_naive}/{top_n} slots")
        emit(f"    Most common formula: {top_comp_naive} -- {top_comp_n_naive}/{top_n} slots")
        emit()
        emit(f"  With diversity (max {args.max_per_composition}/formula, "
             f"{args.max_per_spacegroup}/sg):")
        emit(f"    Unique formulas in shortlist: {shortlist['reduced_formula'].nunique()}")
        emit(f"    Unique sg in shortlist:       {shortlist['sg_number'].nunique()}")
        emit(f"    Shortlist size achieved:      {len(shortlist)}/{top_n}")

    # -- CIF copy stats --
    emit()
    hdr("SECTION F -- CIF file status")
    n_present = int(df_full.get("cif_present", pd.Series([True]*len(df_full))).sum())
    emit(f"  CIF files present  : {n_present}/{len(df_full)}")
    emit(f"  CIF files missing  : {n_missing_cifs}")
    if args.skip_cif_copy:
        emit(f"  CIF copy           : skipped (--skip-cif-copy)")
    else:
        cif_out = Path(args.output_dir) / "shortlist_cifs"
        emit(f"  CIFs copied        : {n_cifs_copied} -> {cif_out}/")

    (output_dir / "summary.txt").write_text("\n".join(lines), encoding="utf-8")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Screen Co-Bi candidates with the structure-only stability classifier.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("metadata_csv",
                        help="Path to metadata CSV with CIF filenames and structural info.")
    parser.add_argument("cif_dir",
                        help="Directory containing the CIF files referenced in metadata_csv.")
    parser.add_argument("--model",
                        default="models/structure_only_classifier.joblib",
                        help="Path to trained sklearn Pipeline (.joblib).")
    parser.add_argument("--feature-columns",
                        default="models/feature_columns.json",
                        dest="feature_columns",
                        help="Path to JSON list of training feature column names.")
    parser.add_argument("--output-dir",
                        default="screening_results",
                        dest="output_dir",
                        help="Directory for output files.")
    parser.add_argument("--top-n",
                        type=int, default=50, dest="top_n",
                        help="Size of final diversified shortlist.")
    parser.add_argument("--max-per-composition",
                        type=int, default=5, dest="max_per_composition",
                        help="Max candidates per reduced_formula in shortlist.")
    parser.add_argument("--max-per-spacegroup",
                        type=int, default=8, dest="max_per_spacegroup",
                        help="Max candidates per sg_number in shortlist.")
    parser.add_argument("--no-diversity",
                        action="store_true", dest="no_diversity",
                        help="Skip diversity constraints; output naive top-N.")
    parser.add_argument("--skip-cif-copy",
                        action="store_true", dest="skip_cif_copy",
                        help="Skip copying CIF files to output (useful for dry runs).")
    args = parser.parse_args()

    t0 = time.perf_counter()

    cif_dir    = Path(args.cif_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not cif_dir.is_dir():
        sys.exit(f"ERROR: CIF directory not found: {cif_dir}")

    # ── 1. load model ──────────────────────────────────────────────────────
    print(f"Loading model from {args.model}...")
    pipeline, feat_cols = load_model_and_features(args.model, args.feature_columns)
    print(f"  Model loaded  |  {len(feat_cols)} training features")

    # ── 2. load and validate metadata ─────────────────────────────────────
    print(f"\nLoading metadata: {args.metadata_csv}")
    df = pd.read_csv(args.metadata_csv)
    print(f"  {len(df)} rows, columns: {df.columns.tolist()}")
    validate_input(df)

    # ── 3. derive columns from metadata format ─────────────────────────────
    print("\nDeriving reduced_formula, crystal_system, co_fraction...")
    df = derive_columns(df)
    print(f"  Unique reduced formulas : {df['reduced_formula'].nunique()}")
    print(f"  Crystal systems found   : {sorted(df['crystal_system'].unique())}")

    # ── 4. validate CIF files ──────────────────────────────────────────────
    print(f"\nChecking CIF files in {cif_dir}...")
    df, missing_cifs = validate_cif_files(df, cif_dir, output_dir)
    n_present = int(df["cif_present"].sum())
    print(f"  Present: {n_present}/{len(df)}  |  Missing: {len(missing_cifs)}")

    # ── 5. featurize ───────────────────────────────────────────────────────
    print(f"\nFeaturizing {len(df)} candidates "
          f"({df['formula_pretty'].nunique()} unique compositions)...")
    error_log: list[dict] = []
    X = featurize(df, feat_cols, error_log)
    print(f"  Feature matrix: {X.shape}")

    if error_log:
        err_path = output_dir / "featurization_errors.csv"
        pd.DataFrame(error_log).to_csv(err_path, index=False)
        print(f"  WARNING: {len(error_log)} compositions failed featurization "
              f"-> {err_path}")

    # ── 6. predict ─────────────────────────────────────────────────────────
    print(f"\nScoring {len(X)} candidates...")
    probs = predict(pipeline, X)

    # ── 7. assemble results ────────────────────────────────────────────────
    df["p_stable"]        = np.round(probs, 4)
    df["predicted_label"] = np.where(probs >= 0.5, "stable", "unstable")
    df["confidence_band"] = assign_confidence_bands(probs)
    df_sorted = df.sort_values("p_stable", ascending=False).reset_index(drop=True)
    df_sorted["rank_overall"] = df_sorted.index + 1

    # ── 8. diversity selection (CIF-present candidates only) ───────────────
    df_eligible = df_sorted[df_sorted["cif_present"]].copy()

    if args.no_diversity:
        shortlist = df_eligible.head(args.top_n).copy()
        shortlist["rank_shortlist"] = range(1, len(shortlist) + 1)
        print(f"\nNo-diversity mode: taking raw top-{args.top_n} "
              f"(from {len(df_eligible)} CIF-present candidates).")
    else:
        print(f"\nApplying diversity constraints "
              f"(max {args.max_per_composition}/formula, "
              f"{args.max_per_spacegroup}/sg) "
              f"over {len(df_eligible)} CIF-present candidates...")
        shortlist = diversified_top_n(
            df_eligible, args.top_n,
            args.max_per_composition, args.max_per_spacegroup,
        )
        print(f"  Shortlist: {len(shortlist)} candidates "
              f"({shortlist['reduced_formula'].nunique()} formulas, "
              f"{shortlist['spacegroup_number'].nunique()} space groups)")

    # ── 9. copy CIFs ───────────────────────────────────────────────────────
    n_cifs_copied = 0
    if args.skip_cif_copy:
        print("\nSkipping CIF copy (--skip-cif-copy).")
    else:
        print(f"\nCopying {len(shortlist)} CIF files to "
              f"{output_dir / 'shortlist_cifs'}...")
        n_cifs_copied = copy_shortlist_cifs(shortlist, cif_dir, output_dir)
        print(f"  Copied: {n_cifs_copied}")

    # ── 10. save CSVs ──────────────────────────────────────────────────────
    df_sorted.to_csv(output_dir / "predictions_full.csv", index=False)
    print(f"\nSaved predictions_full.csv ({len(df_sorted)} rows) "
          f"-> {output_dir / 'predictions_full.csv'}")

    sl_path = output_dir / f"shortlist_top_{len(shortlist)}.csv"
    shortlist.to_csv(sl_path, index=False)
    print(f"Saved shortlist ({len(shortlist)} rows) -> {sl_path}")

    # ── 11. summary ────────────────────────────────────────────────────────
    print()
    print_and_save_summary(df_sorted, shortlist, output_dir, args,
                           n_cifs_copied=n_cifs_copied,
                           n_missing_cifs=len(missing_cifs))
    print(f"\nSaved summary -> {output_dir / 'summary.txt'}")

    # ── timing ─────────────────────────────────────────────────────────────
    elapsed = time.perf_counter() - t0
    rps     = len(df) / elapsed
    print(f"\nTotal processed  : {len(df)}")
    print(f"Shortlist size   : {len(shortlist)}")
    print(f"CIFs copied      : {n_cifs_copied}")
    print(f"Missing CIFs     : {len(missing_cifs)}")
    print(f"Elapsed          : {elapsed:.1f}s  ({rps:.0f} rows/sec)")


if __name__ == "__main__":
    main()
