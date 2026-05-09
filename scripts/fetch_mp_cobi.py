#!/usr/bin/env python3
"""Fetch all Co-Bi binary structures from Materials Project and save as CIF + CSV.

Usage:
    python scripts/fetch_mp_cobi.py

Outputs:
    co_bi_mp_structures/cifs/   one CIF per structure
    co_bi_mp_structures/metadata.csv
"""

from __future__ import annotations

import hashlib
import os
import sys
import warnings
from math import gcd, pi
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
OUTPUT_DIR = Path("./co_bi_mp_structures")
CIFS_DIR = OUTPUT_DIR / "cifs"
METADATA_CSV = OUTPUT_DIR / "metadata.csv"

V_CO = (4 / 3) * pi * (1.25 ** 3)   # 8.181 A^3
V_BI = (4 / 3) * pi * (1.54 ** 3)   # 15.312 A^3
R_NEIGHBOR = 8.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def stability_label(e_above_hull: float | None) -> str:
    if e_above_hull is None:
        return "unknown"
    if e_above_hull <= 0.0:
        return "stable"
    if e_above_hull <= 0.025:
        return "metastable"
    if e_above_hull <= 0.100:
        return "unstable_near"
    return "unstable"


def primitive_ratio(n_co: int, n_bi: int) -> tuple[int, int]:
    g = gcd(n_co, n_bi)
    return n_co // g, n_bi // g


def make_filename(n_co: int, n_bi: int, m: int, n: int, z: int,
                  sg_number: int, mp_id: str, formula: str) -> str:
    mp_num = int(mp_id.replace("mp-", "").replace("mvc-", ""))
    raw = formula + str(sg_number) + mp_id
    h = hashlib.sha1(raw.encode()).hexdigest()[:10]
    return f"Co{n_co}_Bi{n_bi}_m{m}_n{n}_z{z}_sg{sg_number:03d}_mp{mp_num:06d}_{h}.cif"


def compute_fingerprint(structure) -> str:
    neighbors = structure.get_all_neighbors(r=4.0)
    if not neighbors or not neighbors[0]:
        first_shell: list[float] = []
    else:
        first_shell = sorted([round(nb.nn_distance, 2) for nb in neighbors[0]][:6])
    raw = str((
        round(structure.volume / structure.num_sites, 2),
        structure.get_space_group_info()[1],
        tuple(first_shell),
    ))
    return hashlib.sha1(raw.encode()).hexdigest()


def compute_min_distances(structure) -> tuple[str, str, str]:
    """Return (min_CoCo, min_BiBi, min_CoBi) as strings; '' if no such pair."""
    co_indices = [i for i, s in enumerate(structure) if str(s.specie) == "Co"]
    bi_indices = [i for i, s in enumerate(structure) if str(s.specie) == "Bi"]

    all_neighbors = structure.get_all_neighbors(r=R_NEIGHBOR)

    coco_dists: list[float] = []
    bibi_dists: list[float] = []
    cobi_dists: list[float] = []

    for i in co_indices:
        for nb in all_neighbors[i]:
            sp = str(nb.specie)
            if sp == "Co":
                coco_dists.append(nb.nn_distance)
            elif sp == "Bi":
                cobi_dists.append(nb.nn_distance)

    for i in bi_indices:
        for nb in all_neighbors[i]:
            sp = str(nb.specie)
            if sp == "Bi":
                bibi_dists.append(nb.nn_distance)
            elif sp == "Co" and i < nb.index:
                pass  # already captured above

    # Co-Bi from Bi side to catch any missed (symmetric)
    for i in bi_indices:
        for nb in all_neighbors[i]:
            if str(nb.specie) == "Co":
                cobi_dists.append(nb.nn_distance)

    min_coco = f"{min(coco_dists):.6f}" if coco_dists else ""
    min_bibi = f"{min(bibi_dists):.6f}" if bibi_dists else ""
    min_cobi = f"{min(cobi_dists):.6f}" if cobi_dists else ""
    return min_coco, min_bibi, min_cobi


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    # Check API key
    api_key = os.environ.get("MP_API_KEY", "")
    if not api_key:
        print("ERROR: MP_API_KEY environment variable is not set.")
        print("Set your MP API key:  export MP_API_KEY='your_key'")
        print("Get a free key at:    https://materialsproject.org/api")
        return 1

    CIFS_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 1 — fetch from MP
    # ------------------------------------------------------------------
    print("Querying Materials Project for Co-Bi binary structures...")
    try:
        from mp_api.client import MPRester
        with MPRester(api_key) as mpr:
            results = mpr.materials.summary.search(
                chemsys="Co-Bi",
                fields=[
                    "material_id",
                    "formula_pretty",
                    "structure",
                    "symmetry",
                    "energy_above_hull",
                    "formation_energy_per_atom",
                    "is_stable",
                    "nsites",
                    "volume",
                    "density",
                ],
            )
    except Exception as exc:
        print(f"ERROR: MP query failed:\n{exc}")
        return 1

    print(f"Found {len(results)} entries. Processing...\n")

    # ------------------------------------------------------------------
    # Step 2 — process each entry
    # ------------------------------------------------------------------
    metadata_rows: list[dict] = []
    skipped_disordered = 0
    skipped_neighbor = 0
    saved = 0

    summary_rows: list[dict] = []

    for entry in tqdm(results, desc="Structures", unit="struct"):
        mp_id: str = entry.material_id
        formula: str = entry.formula_pretty

        structure = entry.structure
        if structure is None:
            print(f"  WARNING: {mp_id} has no structure, skipping.")
            continue

        # Disordered check
        if not structure.is_ordered:
            print(f"  WARNING: {mp_id} ({formula}) is disordered — skipping.")
            skipped_disordered += 1
            continue

        print(f"  Processing {mp_id}: {formula} ...", end=" ", flush=True)

        # Convert to primitive P1
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            struct_p1 = structure.get_primitive_structure()

        # Composition
        comp = struct_p1.composition
        n_co = int(round(comp.get("Co", 0)))
        n_bi = int(round(comp.get("Bi", 0)))

        if n_co == 0 and n_bi == 0:
            print("no Co/Bi atoms — skip.")
            continue

        m, n = primitive_ratio(max(n_co, 1), max(n_bi, 1))
        if n_co == 0:
            m = 0
        if n_bi == 0:
            n = 0
        z = (n_co // m) if m > 0 else (n_bi // n if n > 0 else 1)

        # Space group
        try:
            sg_symbol, sg_number = struct_p1.get_space_group_info()
        except Exception:
            sg_symbol, sg_number = "P1", 1

        # Neighbors — check they exist
        try:
            all_nb = struct_p1.get_all_neighbors(r=R_NEIGHBOR)
            if any(len(nb) == 0 for nb in all_nb):
                print("empty neighbor shell — skip.")
                skipped_neighbor += 1
                continue
        except Exception as exc:
            print(f"neighbor error ({exc}) — skip.")
            skipped_neighbor += 1
            continue

        # Min distances
        min_coco, min_bibi, min_cobi = compute_min_distances(struct_p1)

        # Packing fraction
        volume = struct_p1.volume
        packing = (n_co * V_CO + n_bi * V_BI) / volume

        # Density — try entry attr first
        try:
            density = float(entry.density)
        except (AttributeError, TypeError):
            density = float(struct_p1.density)

        # Fingerprint
        try:
            fp = compute_fingerprint(struct_p1)
        except Exception:
            fp = ""

        # Filename
        fname = make_filename(n_co, n_bi, m, n, z, sg_number, mp_id, formula)
        cif_path = CIFS_DIR / fname

        # Write CIF
        try:
            cif_text = struct_p1.to(fmt="cif")
            cif_path.write_text(cif_text, encoding="utf-8")
        except Exception as exc:
            print(f"CIF write failed ({exc}) — skip.")
            continue

        # Energy fields
        e_hull = entry.energy_above_hull
        e_form = entry.formation_energy_per_atom
        stab = stability_label(e_hull)
        mp_num = int(mp_id.replace("mp-", "").replace("mvc-", ""))

        # Wyckoff sites string
        try:
            from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
            sga = SpacegroupAnalyzer(struct_p1)
            sym_ds = sga.get_symmetry_dataset()
            wyckoff_str = " ".join(sym_ds["wyckoffs"]) if sym_ds else ""
        except Exception:
            wyckoff_str = ""

        metadata_rows.append({
            "cif_filename":               fname,
            "formula":                    formula,
            "m":                          m,
            "n":                          n,
            "z":                          z,
            "space_group":                sg_number,
            "trial":                      0,
            "seed":                       mp_num,
            "n_Co":                       n_co,
            "n_Bi":                       n_bi,
            "n_atoms":                    struct_p1.num_sites,
            "volume":                     round(volume, 6),
            "density":                    round(density, 6),
            "packing_fraction":           round(packing, 6),
            "min_CoCo":                   min_coco,
            "min_BiBi":                   min_bibi,
            "min_CoBi":                   min_cobi,
            "fingerprint":                fp,
            # extra test labels
            "mp_id":                      mp_id,
            "formation_energy_per_atom":  e_form,
            "energy_above_hull":          e_hull,
            "stability_label":            stab,
        })

        summary_rows.append({
            "mp_id":              mp_id,
            "formula":            formula,
            "space_group":        sg_number,
            "sg_symbol":          sg_symbol,
            "energy_above_hull":  e_hull,
            "stability_label":    stab,
        })

        saved += 1
        print(f"SG {sg_number} ({sg_symbol}) — saved.")

    # ------------------------------------------------------------------
    # Step 3 — write CSV
    # ------------------------------------------------------------------
    df = pd.DataFrame(metadata_rows)
    df.to_csv(METADATA_CSV, index=False)
    print(f"\nMetadata written to {METADATA_CSV}")

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    print("\n" + "=" * 90)
    print(f"  {'mp_id':<14} {'formula':<14} {'SG':>4}  {'sg_symbol':<12} {'e_hull':>10}  stability")
    print("-" * 90)
    summary_rows.sort(key=lambda r: (r["energy_above_hull"] or 999))
    for r in summary_rows:
        e = f"{r['energy_above_hull']:.4f}" if r["energy_above_hull"] is not None else "N/A"
        print(f"  {r['mp_id']:<14} {r['formula']:<14} {r['space_group']:>4}  "
              f"{r['sg_symbol']:<12} {e:>10}  {r['stability_label']}")
    print("=" * 90)

    print(f"\nTotal found     : {len(results)}")
    print(f"Skipped (disordered) : {skipped_disordered}")
    print(f"Skipped (neighbors)  : {skipped_neighbor}")
    print(f"Saved           : {saved}")
    print(f"Output          : {OUTPUT_DIR.resolve()}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
