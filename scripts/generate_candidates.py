#!/usr/bin/env python3
"""Generate Co-Bi candidate crystal structures by prototype substitution.

Queries Materials Project for Ni-Bi, Fe-Bi, Mn-Bi, Cu-Bi structures,
replaces the transition metal with Co (preserving cell geometry and atomic
positions), applies stoichiometry / size / distance filters, deduplicates
with StructureMatcher, and writes CIF files plus a metadata CSV.

Validation checks which of the 4 Freedman (JACS 2025) high-pressure Co-Bi
prototypes were recovered.

Usage (defaults cover the full task):
    python scripts/generate_candidates.py

    python scripts/generate_candidates.py \\
        --systems Ni-Bi Fe-Bi Mn-Bi Cu-Bi \\
        --out-dir data/candidates/Co-Bi \\
        --max-atoms 40 \\
        --min-dist 2.0
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from mp_api.client import MPRester
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Composition, Structure
from pymatgen.io.cif import CifWriter

load_dotenv()

ROOT = Path(__file__).parent.parent
RESULTS_DIR = ROOT / "data" / "results"

# Transition metals in the source systems → all map to Co
REPLACE_ELEMENTS: dict[str, str] = {"Ni": "Co", "Fe": "Co", "Mn": "Co", "Cu": "Co"}

# Freedman (JACS 2025) confirmed high-pressure Co-Bi structures to validate against
# Space group numbers used for matching (robust across symbol notation variants).
FREEDMAN_TARGETS: list[dict[str, Any]] = [
    {
        "name": "α-CoBi₂",
        "source_system": "Ni-Bi",
        "source_formula": "NiBi2",
        "target_formula": "CoBi2",
        "sg_number": 12,
        "sg_symbol": "C2/m",
        "note": "",
    },
    {
        "name": "β-CoBi",
        "source_system": "Ni-Bi",
        "source_formula": "NiBi",
        "target_formula": "CoBi",
        "sg_number": 63,
        "sg_symbol": "Cmcm",
        "note": "",
    },
    {
        "name": "β-CoBi₂",
        "source_system": "Fe-Bi",
        "source_formula": "FeBi2",
        "target_formula": "CoBi2",
        "sg_number": 140,
        "sg_symbol": "I4/mcm",
        "note": "",
    },
    {
        "name": "α-CoBi",
        "source_system": "Cu-Bi",
        "source_formula": "Cu11Bi7",
        "target_formula": "Co11Bi7",
        "sg_number": 194,
        "sg_symbol": "P6_3/mmc",
        "note": "11:7 ratio → filtered by stoichiometry constraint (not 1:1/1:2/1:3)",
    },
]


# ---------------------------------------------------------------------------
# Fetching
# ---------------------------------------------------------------------------

def fetch_source_structures(
    api_key: str,
    systems: list[str],
) -> list[dict[str, Any]]:
    """Return all MP structures for the given chemical systems."""
    log = logging.getLogger(__name__)
    docs: list[dict[str, Any]] = []
    with MPRester(api_key) as mpr:
        for system in systems:
            log.info("Querying %s ...", system)
            results = mpr.materials.summary.search(
                chemsys=[system],
                fields=["material_id", "formula_pretty", "symmetry", "structure"],
            )
            n = 0
            for r in results:
                structure: Structure | None = r.structure
                if structure is None:
                    try:
                        structure = mpr.get_structure_by_material_id(r.material_id)
                    except Exception as exc:
                        log.warning("Cannot fetch structure %s: %s", r.material_id, exc)
                        continue
                docs.append(
                    {
                        "mp_id": r.material_id,
                        "formula": r.formula_pretty,
                        "system": system,
                        "sg_symbol": r.symmetry.symbol if r.symmetry else "?",
                        "sg_number": r.symmetry.number if r.symmetry else 0,
                        "structure": structure,
                    }
                )
                n += 1
            log.info("  %d structures fetched", n)
    log.info("Total source structures: %d", len(docs))
    return docs


# ---------------------------------------------------------------------------
# Prototype substitution
# ---------------------------------------------------------------------------

def substitute_co(source: Structure, source_element: str) -> Structure:
    """Replace source_element with Co; preserve cell and all atomic positions."""
    new = source.copy()
    new.replace_species({source_element: "Co"})
    return new


def find_source_element(structure: Structure) -> str | None:
    """Return the transition metal element to replace, or None if not found."""
    present = {str(el) for el in structure.composition.elements}
    return next((el for el in REPLACE_ELEMENTS if el in present), None)


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------

def has_allowed_stoichiometry(structure: Structure) -> bool:
    """Accept only Co:Bi = 1:1, 1:2, 1:3 in the reduced formula."""
    comp = structure.composition.reduced_composition
    if {str(el) for el in comp.elements} != {"Co", "Bi"}:
        return False
    co = round(comp["Co"])
    bi = round(comp["Bi"])
    return co == 1 and bi in {1, 2, 3}


def has_distance_clash(structure: Structure, min_dist: float) -> bool:
    """Return True if any interatomic distance is below min_dist (Å)."""
    dist = structure.distance_matrix.copy()
    np.fill_diagonal(dist, np.inf)
    return float(dist.min()) < min_dist


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def deduplicate(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove structurally equivalent candidates using StructureMatcher."""
    log = logging.getLogger(__name__)
    matcher = StructureMatcher(
        ltol=0.2, stol=0.3, angle_tol=5, primitive_cell=True, scale=True
    )
    unique: list[dict[str, Any]] = []
    for cand in candidates:
        if not any(matcher.fit(cand["structure"], u["structure"]) for u in unique):
            unique.append(cand)
    log.info("Deduplication: %d → %d unique", len(candidates), len(unique))
    return unique


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_candidates(
    candidates: list[dict[str, Any]],
    out_dir: Path,
) -> list[dict[str, Any]]:
    """Write one CIF per candidate and return metadata rows."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for cand in candidates:
        struct = cand["structure"]
        formula_red = struct.composition.reduced_formula
        candidate_id = f"{cand['source_mp_id']}_{formula_red}"
        cif_path = out_dir / f"{candidate_id}.cif"
        CifWriter(struct).write_file(str(cif_path))
        rows.append(
            {
                "candidate_id": candidate_id,
                "source_system": cand["source_system"],
                "source_formula": cand["source_formula"],
                "source_mp_id": cand["source_mp_id"],
                "target_formula": formula_red,
                "n_atoms": len(struct),
                "space_group": cand["sg_symbol"],
                "file_path": str(cif_path.relative_to(ROOT)),
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_freedman(
    candidates: list[dict[str, Any]],
    source_docs: list[dict[str, Any]],
) -> None:
    """Report recovery of the 4 Freedman (JACS 2025) Co-Bi prototypes."""
    print("\n" + "=" * 70)
    print("Freedman prototype validation (MIT JACS 2025)")
    print("=" * 70)
    for proto in FREEDMAN_TARGETS:
        proto_comp = Composition(proto["source_formula"]).reduced_composition
        source_hits = [
            d for d in source_docs
            if Composition(d["formula"]).reduced_composition == proto_comp
            and d["system"] == proto["source_system"]
            and d["sg_number"] == proto["sg_number"]
        ]
        in_candidates = any(
            c["source_mp_id"] == h["mp_id"]
            for h in source_hits
            for c in candidates
        )

        if not source_hits:
            status = "MISS  — not found in Materials Project"
        elif in_candidates:
            mp_ids = ", ".join(h["mp_id"] for h in source_hits)
            status = f"OK    — captured  ({mp_ids})"
        else:
            status = "MISS  — found in MP but filtered out"
            if proto["note"]:
                status += f"  [{proto['note']}]"

        print(
            f"  {proto['name']:<12}  "
            f"[{proto['source_formula']:<10} {proto['sg_symbol']:<12}]  "
            f"{status}"
        )
    print("=" * 70)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(candidates: list[dict[str, Any]]) -> None:
    print(f"\nTotal candidates: {len(candidates)}")

    stoich = Counter(
        c["structure"].composition.reduced_formula for c in candidates
    )
    print("\nBy stoichiometry:")
    for formula, count in sorted(stoich.items()):
        print(f"  {formula:<12}  {count}")

    by_system = Counter(c["source_system"] for c in candidates)
    print("\nBy source system:")
    for system, count in sorted(by_system.items()):
        print(f"  {system:<10}  {count}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate Co-Bi candidates by prototype substitution from MP."
    )
    p.add_argument(
        "--systems",
        nargs="+",
        default=["Ni-Bi", "Fe-Bi", "Mn-Bi", "Cu-Bi"],
        help="Source chemical systems to query (default: Ni-Bi Fe-Bi Mn-Bi Cu-Bi).",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "data" / "candidates" / "Co-Bi",
        help="Directory for output CIF files.",
    )
    p.add_argument(
        "--max-atoms",
        type=int,
        default=40,
        help="Reject structures with more than this many atoms (default: 40).",
    )
    p.add_argument(
        "--min-dist",
        type=float,
        default=2.0,
        help="Reject structures with any interatomic distance below this (Å, default: 2.0).",
    )
    return p.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    log = logging.getLogger(__name__)
    args = _parse_args()

    api_key = os.environ.get("MP_API_KEY", "")
    if not api_key:
        log.error("MP_API_KEY not set — copy .env.example to .env and add your key.")
        return 1

    # --- Fetch ---
    log.info("=== Step 1: Fetching source structures ===")
    source_docs = fetch_source_structures(api_key, args.systems)

    # --- Substitute + filter ---
    log.info("\n=== Step 2: Prototype substitution and filtering ===")
    candidates: list[dict[str, Any]] = []
    rejected: Counter = Counter()

    for doc in source_docs:
        source_el = find_source_element(doc["structure"])
        if source_el is None:
            continue

        struct = substitute_co(doc["structure"], source_el)

        if not has_allowed_stoichiometry(struct):
            rejected["stoichiometry"] += 1
            log.debug("Rejected (stoichiometry) %s %s", doc["mp_id"], doc["formula"])
            continue
        if len(struct) > args.max_atoms:
            rejected["size"] += 1
            log.debug("Rejected (size %d) %s %s", len(struct), doc["mp_id"], doc["formula"])
            continue
        if has_distance_clash(struct, args.min_dist):
            rejected["distance"] += 1
            log.debug("Rejected (distance clash) %s %s", doc["mp_id"], doc["formula"])
            continue

        candidates.append(
            {
                "structure": struct,
                "source_mp_id": doc["mp_id"],
                "source_formula": doc["formula"],
                "source_system": doc["system"],
                "sg_symbol": doc["sg_symbol"],
                "sg_number": doc["sg_number"],
            }
        )

    log.info(
        "Passed: %d  |  Rejected — stoich: %d  size: %d  distance: %d",
        len(candidates),
        rejected["stoichiometry"],
        rejected["size"],
        rejected["distance"],
    )

    # --- Deduplicate ---
    log.info("\n=== Step 3: Deduplication ===")
    candidates = deduplicate(candidates)

    # --- Save ---
    log.info("\n=== Step 4: Saving to %s ===", args.out_dir)
    metadata_rows = save_candidates(candidates, args.out_dir)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = RESULTS_DIR / "candidate_metadata.csv"
    pd.DataFrame(metadata_rows).to_csv(out_csv, index=False)
    log.info("Metadata saved to %s", out_csv)

    # --- Validate ---
    validate_freedman(candidates, source_docs)

    # --- Summary ---
    print_summary(candidates)

    return 0


if __name__ == "__main__":
    sys.exit(main())
