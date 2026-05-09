#!/usr/bin/env python3
"""
CLI: write VASP relaxation folders for shortlisted candidates.

PBE + spin-polarized settings suit magnetic Co-containing binaries for a first
relaxation pass. POTCAR files are not generated.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from hullgap.dft.make_vasp_inputs import generate_inputs_from_candidate_list


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate VASP input sets from a DFT candidate list.")
    p.add_argument("--candidate-list", type=Path, required=True)
    p.add_argument("--outdir", type=Path, required=True)
    p.add_argument("--preset", type=str, default="coarse_relax")
    p.add_argument("--kppa", type=int, default=1000, help="K-point density (pymatgen automatic_density).")
    return p.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _parse_args()
    df = pd.read_csv(args.candidate_list)
    df.columns = [c.strip() for c in df.columns]
    logging.info("Loaded %d candidate rows from %s", len(df), args.candidate_list)
    # tqdm wrapper: re-implement loop inside library or tqdm on rows - library uses internal loop.
    # We pass tqdm by monkey-patching is heavy; keep simple log in library.
    generate_inputs_from_candidate_list(df, args.outdir, preset=args.preset, kppa=args.kppa)
    logging.info("Done writing inputs under %s", args.outdir.resolve())
    return 0


if __name__ == "__main__":
    sys.exit(main())
