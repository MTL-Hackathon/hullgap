#!/usr/bin/env python3
"""
CLI: formation energies and convex-hull distance for validated DFT totals.

Uses elemental reference energies on the same VASP energy scale; suitable only
for relative ranking within this hackathon workflow.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from hullgap.dft.dft_hull import load_elemental_references, score_dft_candidates, write_hull_scores


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Score DFT energies vs convex hull using reference elemental energies.")
    p.add_argument("--system", type=str, required=True, help="Chemical system, e.g. Co-Bi.")
    p.add_argument("--dft-energies", type=Path, required=True, help="Parsed DFT results CSV.")
    p.add_argument("--out", type=Path, required=True, help="Output hull scores CSV.")
    p.add_argument(
        "--elemental-refs",
        type=Path,
        default=Path("dft/reference_energies.yaml"),
        help="YAML mapping element -> energy_per_atom_eV.",
    )
    return p.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _parse_args()
    df = pd.read_csv(args.dft_energies)
    df.columns = [c.strip() for c in df.columns]
    refs = load_elemental_references(args.elemental_refs)
    scored = score_dft_candidates(df, args.system, refs)
    write_hull_scores(scored, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
