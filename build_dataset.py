#!/usr/bin/env python3
"""
build_dataset.py

Queries the Materials Project API for binary compounds in Co-X and Bi-X
chemical systems, engineers features, labels thermodynamic stability,
balances classes by undersampling, and exports two CSV training datasets.

Usage:
    python build_dataset.py
    python build_dataset.py --output-dir data/ml --hull-threshold 0.1
"""

import argparse
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

warnings.filterwarnings("ignore")

load_dotenv()

# ── chemical systems ──────────────────────────────────────────────────────────

_CO_PARTNERS = ["As", "Sb", "Te", "Se", "P", "S", "Sn", "Ge", "Si", "N", "O", "Fe", "Ni", "Mn"]
_BI_PARTNERS = ["Ni", "Fe", "Mn", "Cu", "Ag", "Zn", "Pb", "Sb", "Te", "Se", "In", "Ga", "Tl"]

ALL_SYSTEMS: list[str] = (
    [f"Co-{x}" for x in _CO_PARTNERS]
    + [f"Bi-{x}" for x in _BI_PARTNERS]
    + ["Co-Bi"]
)

# ── constants ─────────────────────────────────────────────────────────────────

# Magnetic ordering label encoding; anything not listed maps to -1 (Unknown)
_ORDERING_MAP = {"NM": 0, "FM": 1, "FiM": 2, "AFM": 3}

_SUMMARY_FIELDS = [
    "material_id",
    "formula_pretty",
    "chemsys",
    "symmetry",
    "nsites",
    "volume",
    "density",
    "formation_energy_per_atom",
    "energy_above_hull",
    "band_gap",
    "total_magnetization",
    "ordering",
]


# ── data access ───────────────────────────────────────────────────────────────

def query_materials(api_key: str, chemsys_list: list[str]) -> pd.DataFrame:
    """
    Query the MP summary endpoint for all materials in chemsys_list.

    Returns a raw DataFrame with one row per material.
    Logs a warning and continues if any individual system query fails.
    """
    try:
        from mp_api.client import MPRester
    except ImportError:
        sys.exit("mp-api not installed — run: pip install mp-api")

    records: list[dict] = []

    with MPRester(api_key) as mpr:
        for chemsys in tqdm(chemsys_list, desc="Querying MP", unit="system"):
            try:
                docs = mpr.materials.summary.search(
                    chemsys=[chemsys],
                    fields=_SUMMARY_FIELDS,
                )
                for doc in docs:
                    records.append(_flatten_doc(doc))
            except Exception as exc:
                print(f"  [warn] {chemsys}: {exc}")

    if not records:
        sys.exit("No records returned. Verify MP_API_KEY and network access.")

    df = pd.DataFrame(records)
    print(f"\nFetched {len(df):,} entries across {df['chemsys'].nunique()} chemical systems.")
    return df


def _flatten_doc(doc) -> dict:
    """Flatten one SummaryDoc into a plain dict, resolving nested objects."""
    sym = doc.symmetry
    ordering = getattr(doc, "ordering", None)

    return {
        "material_id": str(doc.material_id),
        "formula_pretty": doc.formula_pretty,
        "chemsys": doc.chemsys,
        "spacegroup_number": (getattr(sym, "number", None) if sym else None),
        "crystal_system": (_enum_str(getattr(sym, "crystal_system", None)) if sym else None),
        "nsites": doc.nsites,
        "volume": doc.volume,
        "density": doc.density,
        "formation_energy_per_atom": doc.formation_energy_per_atom,
        "energy_above_hull": doc.energy_above_hull,
        "band_gap": doc.band_gap,
        "total_magnetization": doc.total_magnetization,
        "magnetic_ordering": _enum_str(ordering),
    }


def _enum_str(val) -> str | None:
    """Convert an enum (or anything with .value) to its string value."""
    if val is None:
        return None
    return val.value if hasattr(val, "value") else str(val)


# ── feature engineering ───────────────────────────────────────────────────────

def clean_and_engineer(df: pd.DataFrame, hull_threshold: float) -> pd.DataFrame:
    """
    Drop invalid rows, derive new features, encode categoricals, label stability.

    Steps:
      1. Drop rows missing formation_energy_per_atom, energy_above_hull, or nsites
      2. Add is_stable label (1 if energy_above_hull <= hull_threshold)
      3. Compute volume_per_atom and composition_A_fraction
      4. Fill missing band_gap (-1) and total_magnetization (0)
      5. Label-encode magnetic_ordering: NM=0, FM=1, FiM=2, AFM=3, Unknown=-1
      6. One-hot encode crystal_system into cs_* boolean columns
    """
    df = df.copy()

    n_before = len(df)
    df = df.dropna(subset=["formation_energy_per_atom", "energy_above_hull"])
    df = df[df["nsites"].notna() & (df["nsites"] > 0)]
    dropped = n_before - len(df)
    if dropped:
        print(f"  Dropped {dropped} rows (null energies or invalid nsites).")

    df["is_stable"] = (df["energy_above_hull"] <= hull_threshold).astype(int)

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

    return df


def _first_element_fraction(row: pd.Series) -> float:
    """Return the atomic fraction of the alphabetically first element in the chemsys."""
    from pymatgen.core import Composition  # cached by Python's import system
    try:
        first_el = sorted(row["chemsys"].split("-"))[0]
        return float(Composition(row["formula_pretty"]).get_atomic_fraction(first_el))
    except Exception:
        return np.nan


# ── class balancing ───────────────────────────────────────────────────────────

def balance_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Undersample the majority class to match the minority class count.

    Uses random_state=42 for reproducibility.
    Returns the original DataFrame unchanged if either class is empty.
    """
    counts = df["is_stable"].value_counts()

    print("\nClass distribution before balancing:")
    for label, count in counts.sort_index().items():
        tag = "stable" if label == 1 else "unstable"
        print(f"  {tag:>8} ({label}): {count:,}")

    minority_n = int(counts.min())
    if minority_n == 0:
        print("  [warn] One class has zero entries — skipping balancing.")
        return df

    balanced = pd.concat(
        [grp.sample(n=minority_n, random_state=42) for _, grp in df.groupby("is_stable")],
        ignore_index=True,
    )

    print("\nClass distribution after balancing:")
    for label, count in balanced["is_stable"].value_counts().sort_index().items():
        tag = "stable" if label == 1 else "unstable"
        print(f"  {tag:>8} ({label}): {count:,}")

    return balanced


# ── orchestration ─────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a binary crystal stability dataset from Materials Project.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output-dir", default=".", metavar="DIR",
        help="Directory to write training_data_full.csv and training_data_balanced.csv",
    )
    parser.add_argument(
        "--hull-threshold", type=float, default=0.05, metavar="EV",
        help="Energy-above-hull cutoff (eV/atom) for is_stable=1",
    )
    args = parser.parse_args()

    api_key = os.environ.get("MP_API_KEY")
    if not api_key:
        sys.exit(
            "MP_API_KEY is not set.\n"
            "Either export it in your shell or add it to a .env file:\n"
            "  MP_API_KEY=your_key_here"
        )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Hull threshold  : {args.hull_threshold} eV/atom")
    print(f"Output directory: {out_dir.resolve()}")
    print(f"Systems to query: {len(ALL_SYSTEMS)}\n")

    # 1 — fetch from MP
    raw_df = query_materials(api_key, ALL_SYSTEMS)

    # 2 — clean and engineer features
    print("\nCleaning and engineering features...")
    full_df = clean_and_engineer(raw_df, args.hull_threshold)

    # 3 — save full dataset
    full_path = out_dir / "training_data_full.csv"
    full_df.to_csv(full_path, index=False)
    print(f"\nSaved: {full_path}  ({len(full_df):,} rows)")

    # 4 — balance and save
    print("\nBalancing dataset...")
    balanced_df = balance_dataset(full_df)

    bal_path = out_dir / "training_data_balanced.csv"
    balanced_df.to_csv(bal_path, index=False)
    print(f"Saved: {bal_path}  ({len(balanced_df):,} rows)")

    # 5 — summary report
    _print_summary(full_df, balanced_df, full_path, bal_path)


def _print_summary(
    full_df: pd.DataFrame,
    balanced_df: pd.DataFrame,
    full_path: Path,
    bal_path: Path,
) -> None:
    sep = "-" * 60
    print(f"\n{sep}\nSUMMARY\n{sep}")

    print("\nEntries per chemical system (full dataset):")
    for chemsys, n in full_df["chemsys"].value_counts().sort_index().items():
        print(f"  {chemsys:<12} {n:>5}")

    feature_cols = [
        c for c in full_df.columns
        if c not in {"material_id", "formula_pretty", "chemsys", "is_stable"}
    ]
    print(f"\nFeature columns ({len(feature_cols)}):")
    for col in feature_cols:
        print(f"  {col}")

    print(f"\ntraining_data_full.csv     : {len(full_df):,} rows  ->  {full_path.resolve()}")
    print(f"training_data_balanced.csv : {len(balanced_df):,} rows  ->  {bal_path.resolve()}")
    print(sep)


if __name__ == "__main__":
    main()
