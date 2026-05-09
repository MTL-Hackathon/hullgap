#!/usr/bin/env python3
"""Geometry pre-processor: scale and filter raw candidate CIFs before screening.

Slots between candidate generation (pyxtal/random) and screen_candidates.py.

Pipeline:
    pyxtal cifs  ->  preprocess_candidates.py  ->  screen_candidates.py  ->  MLIP

For each CIF:
  1. Load structure
  2. Scale lattice via Vegard's law so nearest-neighbour distance matches
     the expected metallic bond length for Co-Bi
  3. Apply hard distance and packing-fraction filter
  4. Save scaled CIF to output directory
  5. Compute distance columns for the screener metadata CSV

Usage:
    python scripts/preprocess_candidates.py METADATA_CSV CIF_DIR [options]

Examples:
    python scripts/preprocess_candidates.py \\
        data/results/candidate_metadata_screener.csv \\
        data/candidates/Co-Bi/ \\
        --output-dir data/candidates/Co-Bi-scaled/

    # Then pass straight to screener:
    python screen_candidates.py \\
        data/candidates/Co-Bi-scaled/metadata_scaled.csv \\
        data/candidates/Co-Bi-scaled/cifs/
"""

from __future__ import annotations

import argparse
import logging
import sys
from math import pi
from pathlib import Path

import numpy as np
import pandas as pd
from pymatgen.core import Element, Structure
from pymatgen.io.cif import CifWriter
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
R_CO = 1.25   # metallic radius, Angstrom
R_BI = 1.54

V_CO = (4 / 3) * pi * R_CO ** 3   # 8.181 A^3
V_BI = (4 / 3) * pi * R_BI ** 3   # 15.312 A^3

D_MIN_CoCo = 2.00
D_MIN_BiBi = 2.50
D_MIN_CoBi = 2.10
PHI_MIN    = 0.20
PHI_MAX    = 0.95
R_NEIGHBOR = 6.0


# ---------------------------------------------------------------------------
# Step 1 — Vegard's law lattice scaling
# ---------------------------------------------------------------------------

def _target_bond(structure: Structure) -> float:
    """Expected nearest-neighbour distance for this Co-Bi structure."""
    has_co = any(str(s.specie) == "Co" for s in structure)
    has_bi = any(str(s.specie) == "Bi" for s in structure)
    if has_co and has_bi:
        return R_CO + R_BI        # 2.79 A  Co-Bi bond
    if has_bi:
        return 2 * R_BI           # 3.08 A  Bi-Bi
    return 2 * R_CO               # 2.50 A  Co-Co


def scale_to_physical(structure: Structure) -> Structure:
    """
    Rescale the unit cell so the current nearest-neighbour distance matches
    the expected metallic bond length.  Uses Structure.scale_lattice().
    """
    nb = structure.get_all_neighbors(r=4.0)
    all_dists = [n.nn_distance for site_nb in nb for n in site_nb if n.nn_distance > 0.1]
    if not all_dists:
        return structure           # can't scale — return as-is

    d_now    = min(all_dists)
    d_target = _target_bond(structure)
    if d_now <= 0:
        return structure

    scale = d_target / d_now
    new_vol = structure.volume * (scale ** 3)
    scaled  = structure.copy()
    scaled.scale_lattice(new_vol)
    return scaled


# ---------------------------------------------------------------------------
# Step 2 — Hard distance + packing filter
# ---------------------------------------------------------------------------

def _min_pair_dist(structure: Structure, sp_a: str, sp_b: str,
                   nb_list: list) -> float | None:
    """
    Minimum distance between species A and B from a precomputed neighbour list.
    Returns None if no such pair exists within R_NEIGHBOR.
    """
    dists: list[float] = []
    for i, site in enumerate(structure):
        if str(site.specie) != sp_a:
            continue
        for nb in nb_list[i]:
            if str(nb.specie) == sp_b:
                dists.append(nb.nn_distance)
    return min(dists) if dists else None


def filter_structure(structure: Structure) -> tuple[bool, str]:
    """
    Return (passes, reason).  reason is '' when passes is True.
    Rejects structures with atomic clashes or unphysical packing.
    """
    n_co = sum(1 for s in structure if str(s.specie) == "Co")
    n_bi = sum(1 for s in structure if str(s.specie) == "Bi")

    # packing fraction
    phi = (n_co * V_CO + n_bi * V_BI) / structure.volume
    if phi < PHI_MIN:
        return False, f"phi={phi:.3f} < {PHI_MIN} (too sparse)"
    if phi > PHI_MAX:
        return False, f"phi={phi:.3f} > {PHI_MAX} (too dense)"

    nb_list = structure.get_all_neighbors(r=R_NEIGHBOR)

    d_coco = _min_pair_dist(structure, "Co", "Co", nb_list)
    if d_coco is not None and d_coco < D_MIN_CoCo:
        return False, f"Co-Co={d_coco:.3f} < {D_MIN_CoCo} A"

    d_bibi = _min_pair_dist(structure, "Bi", "Bi", nb_list)
    if d_bibi is not None and d_bibi < D_MIN_BiBi:
        return False, f"Bi-Bi={d_bibi:.3f} < {D_MIN_BiBi} A"

    d_cobi = _min_pair_dist(structure, "Co", "Bi", nb_list)
    if d_cobi is not None and d_cobi < D_MIN_CoBi:
        return False, f"Co-Bi={d_cobi:.3f} < {D_MIN_CoBi} A"

    return True, ""


# ---------------------------------------------------------------------------
# Step 3 — Distance columns for screener metadata
# ---------------------------------------------------------------------------

def compute_distance_cols(structure: Structure) -> dict:
    """Return min_CoCo, min_BiBi, min_CoBi as strings ('' if no such pair)."""
    nb_list = structure.get_all_neighbors(r=R_NEIGHBOR)
    d_coco  = _min_pair_dist(structure, "Co", "Co", nb_list)
    d_bibi  = _min_pair_dist(structure, "Bi", "Bi", nb_list)
    d_cobi  = _min_pair_dist(structure, "Co", "Bi", nb_list)
    return {
        "min_CoCo": f"{d_coco:.6f}" if d_coco is not None else "",
        "min_BiBi": f"{d_bibi:.6f}" if d_bibi is not None else "",
        "min_CoBi": f"{d_cobi:.6f}" if d_cobi is not None else "",
        "volume":            round(structure.volume, 6),
        "volume_per_atom":   round(structure.volume / structure.num_sites, 6),
        "packing_fraction":  round(
            (sum(1 for s in structure if str(s.specie) == "Co") * V_CO +
             sum(1 for s in structure if str(s.specie) == "Bi") * V_BI)
            / structure.volume, 6
        ),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Scale + filter raw candidate CIFs before screener.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("metadata_csv",
                        help="Input metadata CSV (cif_filename, formula, space_group, "
                             "n_atoms, n_Co, n_Bi required).")
    parser.add_argument("cif_dir",
                        help="Directory containing the input CIF files.")
    parser.add_argument("--output-dir", default="data/candidates/Co-Bi-scaled",
                        dest="output_dir",
                        help="Output directory for scaled CIFs and metadata.")
    parser.add_argument("--no-scale", action="store_true", dest="no_scale",
                        help="Skip Vegard scaling; only apply distance filter.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)

    cif_dir    = Path(args.cif_dir)
    output_dir = Path(args.output_dir)
    cifs_out   = output_dir / "cifs"
    cifs_out.mkdir(parents=True, exist_ok=True)
    errors_log = output_dir / "preprocess_errors.log"

    if not cif_dir.is_dir():
        print(f"ERROR: CIF directory not found: {cif_dir}")
        return 1

    df_in = pd.read_csv(args.metadata_csv)
    required = ["cif_filename", "formula", "space_group", "n_atoms", "n_Co", "n_Bi"]
    missing_cols = [c for c in required if c not in df_in.columns]
    if missing_cols:
        print(f"ERROR: missing columns in input CSV: {missing_cols}")
        return 1

    n_total       = len(df_in)
    n_load_error  = 0
    n_reject_phi  = 0
    n_reject_dist = 0
    n_saved       = 0
    error_lines: list[str] = []

    out_rows: list[dict] = []

    for _, row in tqdm(df_in.iterrows(), total=n_total,
                       desc="Preprocessing", unit="struct"):
        fname    = row["cif_filename"]
        cif_path = cif_dir / fname

        if not cif_path.exists():
            error_lines.append(f"MISSING: {fname}")
            n_load_error += 1
            continue

        try:
            struct = Structure.from_file(str(cif_path))
        except Exception as exc:
            error_lines.append(f"LOAD_ERROR: {fname} — {exc}")
            n_load_error += 1
            continue

        # --- Vegard's law scaling ---
        if not args.no_scale:
            try:
                struct = scale_to_physical(struct)
            except Exception as exc:
                error_lines.append(f"SCALE_ERROR: {fname} — {exc}")
                # continue without scaling rather than skipping entirely
                pass

        # --- hard filter ---
        passes, reason = filter_structure(struct)
        if not passes:
            if "phi" in reason:
                n_reject_phi += 1
            else:
                n_reject_dist += 1
            error_lines.append(f"REJECTED: {fname} — {reason}")
            continue

        # --- save scaled CIF ---
        try:
            CifWriter(struct).write_file(str(cifs_out / fname))
        except Exception as exc:
            error_lines.append(f"WRITE_ERROR: {fname} — {exc}")
            n_load_error += 1
            continue

        # --- build output metadata row ---
        dist_cols = compute_distance_cols(struct)
        out_row   = row.to_dict()
        out_row.update(dist_cols)
        # update n_atoms in case primitive cell changed
        out_row["n_atoms"] = struct.num_sites
        out_rows.append(out_row)
        n_saved += 1

    # write errors log
    if error_lines:
        errors_log.write_text("\n".join(error_lines), encoding="utf-8")

    # write metadata CSV
    df_out = pd.DataFrame(out_rows)
    meta_out = output_dir / "metadata_scaled.csv"
    df_out.to_csv(meta_out, index=False)

    # --- summary ---
    n_rejected = n_reject_phi + n_reject_dist
    print(f"\n{'='*60}")
    print(f"  Preprocessing complete")
    print(f"{'='*60}")
    print(f"  Input structures      : {n_total}")
    print(f"  Load / write errors   : {n_load_error}")
    print(f"  Rejected (packing)    : {n_reject_phi}")
    print(f"  Rejected (distances)  : {n_reject_dist}")
    print(f"  Passed + saved        : {n_saved}  ({100*n_saved/n_total:.1f}%)")
    print(f"  Scaled CIFs           : {cifs_out}")
    print(f"  Metadata CSV          : {meta_out}")
    if error_lines:
        print(f"  Error log             : {errors_log}")
    print(f"\nNext step:")
    print(f"  python screen_candidates.py {meta_out} {cifs_out}/")

    return 0


if __name__ == "__main__":
    sys.exit(main())
