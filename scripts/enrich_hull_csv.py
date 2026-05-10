"""
Add crystal_system column to mattersim hull CSVs using the relaxed CIF files.

Reads each relaxed CIF, runs pymatgen SpacegroupAnalyzer with a loose
tolerance, and writes the crystal system back into the CSV.

Usage:
    python scripts/enrich_hull_csv.py
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "data" / "results"
MATTERGEN_DIR = PROJECT_ROOT / "data" / "mattergen"


def crystal_system_from_cif(cif_path: Path, symprec: float = 0.1) -> str:
    try:
        struct = Structure.from_file(str(cif_path))
        sga = SpacegroupAnalyzer(struct, symprec=symprec)
        return sga.get_crystal_system().capitalize()
    except Exception as exc:
        log.warning("  Could not analyze %s: %s", cif_path.name, exc)
        return "Unknown"


def enrich_csv(csv_path: Path) -> None:
    system = csv_path.stem.replace("_mattersim_hull", "")
    relaxed_dir = MATTERGEN_DIR / system / "relaxed"

    if not relaxed_dir.exists():
        log.warning("No relaxed dir for %s, skipping", system)
        return

    df = pd.read_csv(csv_path)
    if "crystal_system" in df.columns:
        log.info("%s already has crystal_system, skipping", csv_path.name)
        return

    cif_map: dict[int, Path] = {}
    for cif in relaxed_dir.glob("*.cif"):
        parts = cif.stem.split("_")
        try:
            idx = int(parts[1])
            cif_map[idx] = cif
        except (IndexError, ValueError):
            continue

    crystal_systems: list[str] = []
    for _, row in df.iterrows():
        idx = int(row["idx"])
        cif_path = cif_map.get(idx)
        if cif_path:
            cs = crystal_system_from_cif(cif_path)
            log.info("  [%d] %s -> %s", idx, row["formula"], cs)
        else:
            cs = "Unknown"
            log.warning("  [%d] %s -> no CIF found", idx, row["formula"])
        crystal_systems.append(cs)

    df["crystal_system"] = crystal_systems
    df.to_csv(csv_path, index=False)
    log.info("Updated %s (%d rows)", csv_path.name, len(df))


def main() -> None:
    csvs = sorted(RESULTS_DIR.glob("*_mattersim_hull.csv"))
    log.info("Found %d mattersim hull CSVs", len(csvs))
    for csv_path in csvs:
        log.info("Processing %s ...", csv_path.name)
        enrich_csv(csv_path)
    log.info("Done")


if __name__ == "__main__":
    main()
