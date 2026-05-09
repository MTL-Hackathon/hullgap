#!/usr/bin/env python3
"""Build AFLOW prototype structures for Co-Bi candidate generation.

AFLOW's prototype encyclopedia is used as the reference, but CIF files are
constructed directly with pymatgen from literature Wyckoff positions rather
than downloaded — the AFLOW HTTP endpoint is intermittently unavailable.
When AFLOW comes back online the download path (Step 1) will be tried first
and the embedded fallback used only on failure.

Prototype reference: https://aflowlib.org/prototype-encyclopedia/

Usage:
    python scripts/aflow_prototypes.py

Outputs:
    data/prototypes/aflow/<label>.cif              source prototype CIFs
    data/candidates/Co-Bi/aflow_<label>_CoBi.cif  substituted candidates
    data/results/candidate_metadata.csv            rows appended
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
from pymatgen.core import Element, Lattice, Structure
from pymatgen.io.cif import CifWriter
from pymatgen.symmetry.groups import SpaceGroup

ROOT = Path(__file__).parent.parent
PROTOTYPES_DIR = ROOT / "data" / "prototypes" / "aflow"
CANDIDATES_DIR = ROOT / "data" / "candidates" / "Co-Bi"
RESULTS_DIR = ROOT / "data" / "results"
METADATA_CSV = RESULTS_DIR / "candidate_metadata.csv"

AFLOW_BASE = "https://aflowlib.org/prototype-encyclopedia/"

R_CO = 1.26    # Angstrom, used in Vegard's law
R_BI = 1.60

MIN_COBI_DIST = 2.5
MAX_COBI_DIST = 4.5
MIN_VOL_PER_ATOM = 20.0
MAX_VOL_PER_ATOM = 120.0
MAX_ATOMS = 40

# ---------------------------------------------------------------------------
# Prototype definitions (Wyckoff coordinates from literature)
# Each entry: label, AFLOW name, Freedman target, space group number,
# lattice factory, and asymmetric unit [(element, [x,y,z]), ...]
# ---------------------------------------------------------------------------
PROTOTYPES: list[dict[str, Any]] = [
    {
        "label": "AB_oC8_63_c_a",
        "name": "TlI",
        "freedman": "beta-CoBi",
        "sg": 63,
        "lattice": Lattice.orthorhombic(4.578, 12.461, 5.248),
        # Tl on 4c (0, y, 1/4), I on 4a (0, 0, 0)
        "asym": [("Tl", [0.0, 0.360, 0.25]),
                 ("I",  [0.0, 0.000, 0.00])],
    },
    {
        "label": "AB2_tI12_139_a_h",
        "name": "CuAl2",
        "freedman": "beta-CoBi2",
        "sg": 140,
        "lattice": Lattice.tetragonal(6.063, 4.874),
        # Cu on 4a (0, 0, 1/4), Al on 8h (x, x+0.5, 0)
        "asym": [("Cu", [0.000, 0.000, 0.250]),
                 ("Al", [0.158, 0.658, 0.000])],
    },
    {
        "label": "AB_hP4_194_c_a",
        "name": "NiAs",
        "freedman": "alpha-CoBi",
        "sg": 194,
        "lattice": Lattice.hexagonal(3.619, 5.010),
        # Ni on 2a (0,0,0), As on 2c (1/3, 2/3, 1/4)
        "asym": [("Ni", [0.000,    0.000,    0.000]),
                 ("As", [1/3,      2/3,      0.250])],
    },
    {
        "label": "AB2_hP3_191_a_d",
        "name": "AlB2",
        "freedman": "",
        "sg": 191,
        "lattice": Lattice.hexagonal(3.009, 3.262),
        # Al on 1a (0,0,0), B on 2d (1/3, 2/3, 1/2)
        "asym": [("Al", [0.000, 0.000, 0.000]),
                 ("B",  [1/3,   2/3,   0.500])],
    },
    {
        "label": "AB3_oC16_63_c_cf",
        "name": "PuBr3",
        "freedman": "",
        "sg": 63,
        "lattice": Lattice.orthorhombic(7.964, 11.584, 4.058),
        # Pu on 4c (y=0.25), Br on 4c (y=0.564) and 8f (x=0.33)
        "asym": [("Pu", [0.000, 0.250, 0.250]),
                 ("Br", [0.000, 0.564, 0.250]),
                 ("Br", [0.330, 0.000, 0.500])],
    },
    {
        "label": "AB_hP2_187_d_a",
        "name": "WC",
        "freedman": "",
        "sg": 187,
        "lattice": Lattice.hexagonal(2.906, 2.837),
        # W on 1d (1/3, 2/3, 1/2), C on 1a (0,0,0)
        "asym": [("W", [1/3,  2/3,  0.500]),
                 ("C", [0.0,  0.0,  0.000])],
    },
    {
        "label": "AB_oP8_62_c_c",
        "name": "FeB",
        "freedman": "",
        "sg": 62,
        "lattice": Lattice.orthorhombic(4.053, 5.495, 2.952),
        # Fe on 4c (x=0.180, z=0.125), B on 4c (x=0.036, z=0.610)
        "asym": [("Fe", [0.180, 0.250, 0.125]),
                 ("B",  [0.036, 0.250, 0.610])],
    },
    {
        "label": "AB_cP2_221_b_a",
        "name": "CsCl",
        "freedman": "",
        "sg": 221,
        "lattice": Lattice.cubic(4.123),
        # Cs on 1b (1/2,1/2,1/2), Cl on 1a (0,0,0)
        "asym": [("Cs", [0.5, 0.5, 0.5]),
                 ("Cl", [0.0, 0.0, 0.0])],
    },
]


# ---------------------------------------------------------------------------
# Step 1: Attempt AFLOW download (opportunistic, falls back to embedded)
# ---------------------------------------------------------------------------

def try_download_cif(label: str) -> str | None:
    """Try to fetch the prototype CIF from AFLOW.  Returns text or None."""
    log = logging.getLogger(__name__)
    for suffix in (".CIF", "-001.CIF"):
        url = f"{AFLOW_BASE}{label}{suffix}"
        try:
            resp = requests.get(url, timeout=10, verify=False)
            if resp.status_code == 200 and len(resp.text) > 100:
                log.info("  Downloaded from AFLOW: %s", url)
                return resp.text
        except requests.RequestException:
            pass
    return None


# ---------------------------------------------------------------------------
# Step 2: Build structure from embedded Wyckoff definition
# ---------------------------------------------------------------------------

def build_from_wyckoff(proto: dict[str, Any]) -> Structure:
    """Expand Wyckoff asymmetric unit to full unit cell via space group orbits."""
    sg = SpaceGroup.from_int_number(proto["sg"])
    species: list[str] = []
    coords: list[list[float]] = []
    tol = 1e-3

    for element, coord in proto["asym"]:
        orbit = sg.get_orbit(np.array(coord, dtype=float), tol=tol)
        for pos in orbit:
            wrapped = np.mod(pos, 1.0)
            if all(not np.allclose(wrapped, c, atol=tol) for c in coords):
                species.append(element)
                coords.append(wrapped.tolist())

    return Structure(proto["lattice"], species, coords)


def load_prototype(proto: dict[str, Any]) -> tuple[Structure, str]:
    """Return (structure, source) — source is 'AFLOW' or 'embedded'."""
    log = logging.getLogger(__name__)
    PROTOTYPES_DIR.mkdir(parents=True, exist_ok=True)
    cif_path = PROTOTYPES_DIR / f"{proto['label']}.cif"

    # Try live download first
    cif_text = try_download_cif(proto["label"])
    if cif_text:
        cif_path.write_text(cif_text, encoding="utf-8")
        try:
            return Structure.from_file(str(cif_path)), "AFLOW"
        except Exception as exc:
            log.warning("  Downloaded CIF failed to parse (%s); using embedded", exc)

    # Fallback: build from embedded Wyckoff data
    structure = build_from_wyckoff(proto)
    CifWriter(structure).write_file(str(cif_path))
    log.info("  Built from embedded Wyckoff data -> %s atoms", len(structure))
    return structure, "embedded"


# ---------------------------------------------------------------------------
# Step 3: Identify sites, substitute Co/Bi, Vegard's law scaling
# ---------------------------------------------------------------------------

def identify_sites(structure: Structure) -> tuple[str, str]:
    """Return (metal, anion) by Pauling electronegativity (metal = lower EN)."""
    elements = sorted(
        {str(el) for el in structure.composition.elements},
        key=lambda sym: Element(sym).X,
    )
    return elements[0], elements[-1]


def substitute_and_scale(structure: Structure, metal: str, anion: str) -> Structure:
    """Replace metal->Co and anion->Bi; rescale lattice via Vegard's law."""
    log = logging.getLogger(__name__)
    new = structure.copy()
    new.replace_species({metal: "Co", anion: "Bi"})

    r_metal = Element(metal).atomic_radius
    r_anion = Element(anion).atomic_radius
    if r_metal is None or r_anion is None:
        log.warning("  Missing atomic radius for %s or %s — no Vegard scaling", metal, anion)
        return new

    scale = (R_CO + R_BI) / (float(r_metal) + float(r_anion))
    lat = new.lattice
    new_lat = Lattice.from_parameters(
        lat.a * scale, lat.b * scale, lat.c * scale,
        lat.alpha, lat.beta, lat.gamma,
    )
    return Structure(new_lat, new.species, new.frac_coords)


# ---------------------------------------------------------------------------
# Step 4: Sanity check
# ---------------------------------------------------------------------------

def min_cobi_distance(structure: Structure) -> float:
    """Minimum Co-Bi distance accounting for periodic boundary conditions."""
    min_d = float("inf")
    for i, site in enumerate(structure):
        if str(site.specie) != "Co":
            continue
        for nn in structure.get_neighbors(site, r=MAX_COBI_DIST + 1.0):
            if str(nn.specie) == "Bi":
                min_d = min(min_d, nn.nn_distance)
    return min_d


def sanity_check(structure: Structure) -> tuple[bool, str]:
    """Return (passed, reason).  reason is '' when passed."""
    n = len(structure)
    if n > MAX_ATOMS:
        return False, f"too many atoms ({n})"

    vpa = structure.volume / n
    if vpa < MIN_VOL_PER_ATOM:
        return False, f"vol/atom too small ({vpa:.1f} A3)"
    if vpa > MAX_VOL_PER_ATOM:
        return False, f"vol/atom too large ({vpa:.1f} A3)"

    d = min_cobi_distance(structure)
    if d < MIN_COBI_DIST:
        return False, f"Co-Bi too close ({d:.2f} A)"
    if d > MAX_COBI_DIST:
        return False, f"Co-Bi too far ({d:.2f} A)"

    return True, ""


# ---------------------------------------------------------------------------
# Step 5: Save
# ---------------------------------------------------------------------------

def save_candidate(structure: Structure, label: str) -> Path:
    CANDIDATES_DIR.mkdir(parents=True, exist_ok=True)
    path = CANDIDATES_DIR / f"aflow_{label}_CoBi.cif"
    CifWriter(structure).write_file(str(path))
    return path


def append_metadata(rows: list[dict[str, Any]]) -> None:
    new_df = pd.DataFrame(rows)
    if METADATA_CSV.exists():
        combined = pd.concat([pd.read_csv(METADATA_CSV), new_df], ignore_index=True)
    else:
        combined = new_df
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    combined.to_csv(METADATA_CSV, index=False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    log = logging.getLogger(__name__)

    import warnings
    warnings.filterwarnings("ignore", message=".*InsecureRequestWarning.*")

    summary: list[dict[str, Any]] = []
    metadata_rows: list[dict[str, Any]] = []

    for proto in PROTOTYPES:
        label = proto["label"]
        log.info("--- %s (%s) ---", label, proto["name"])

        # Load / build prototype
        try:
            source_struct, src = load_prototype(proto)
        except Exception as exc:
            log.error("  Failed to load %s: %s", label, exc)
            summary.append({**proto, "status": f"load error: {exc}", "file": ""})
            continue

        sg_orig = source_struct.get_space_group_info()[0]
        metal, anion = identify_sites(source_struct)
        log.info(
            "  Source: %s  SG=%s  n=%d  metal=%s  anion=%s  src=%s",
            source_struct.composition.reduced_formula, sg_orig,
            len(source_struct), metal, anion, src,
        )

        # Substitute + scale
        candidate = substitute_and_scale(source_struct, metal, anion)
        formula = candidate.composition.reduced_formula

        # Sanity check
        passed, reason = sanity_check(candidate)
        if not passed:
            log.warning("  REJECTED: %s", reason)
            summary.append({**proto, "status": f"rejected: {reason}", "file": ""})
            continue

        sg_out = candidate.get_space_group_info()[0]
        vpa = candidate.volume / len(candidate)
        d_min = min_cobi_distance(candidate)
        log.info(
            "  OK -> %s  SG=%s  n=%d  vol/atom=%.1f A3  min Co-Bi=%.2f A",
            formula, sg_out, len(candidate), vpa, d_min,
        )

        cif_out = save_candidate(candidate, label)

        summary.append({
            **proto,
            "status": "OK",
            "sg_out": sg_out,
            "formula": formula,
            "n_atoms": len(candidate),
            "file": str(cif_out.relative_to(ROOT)),
        })
        metadata_rows.append({
            "candidate_id": f"aflow_{label}",
            "source": "AFLOW",
            "source_label": f"{label} ({proto['name']})",
            "target_formula": formula,
            "space_group": sg_out,
            "n_atoms": len(candidate),
            "stoichiometry": formula,
            "file_path": str(cif_out.relative_to(ROOT)),
        })

    # Update metadata CSV
    if metadata_rows:
        append_metadata(metadata_rows)
        log.info("Appended %d rows to %s", len(metadata_rows), METADATA_CSV)

    # Step 6: Summary table
    print("\n" + "=" * 82)
    print(f"  {'prototype':<26} {'sg_out':<14} {'stoich':<10} {'n':>3}  status")
    print("-" * 82)
    for r in summary:
        sg = r.get("sg_out", "-")
        formula = r.get("formula", "-")
        n = r.get("n_atoms", "-")
        print(f"  {r['label']:<26} {sg:<14} {formula:<10} {str(n):>3}  {r['status']}")

    n_ok = sum(1 for r in summary if r["status"] == "OK")
    print("=" * 82)
    print(f"\nTotal AFLOW prototypes processed : {len(PROTOTYPES)}")
    print(f"Valid Co-Bi candidates generated : {n_ok}")

    freedman = [
        ("TlI  / Cmcm",    "beta-CoBi",   "beta-CoBi"),
        ("CuAl2 / I4/mcm", "beta-CoBi2",  "beta-CoBi2"),
        ("NiAs / P63mmc",  "alpha-CoBi",  "alpha-CoBi"),
    ]
    print("\nFreedman structure coverage:")
    for label_str, target, key in freedman:
        covered = any(r.get("freedman") == key and r["status"] == "OK" for r in summary)
        mark = "OK" if covered else "MISS"
        print(f"  {label_str:<22} -> {target:<14} [{mark}]")

    return 0


if __name__ == "__main__":
    sys.exit(main())
