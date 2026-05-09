#!/usr/bin/env python3
"""
CLI: harvest energies and structures from finished QE pw.x run directories.

Intended after calculations complete; does not require the QE binary.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from hullgap.dft.parse_qe_outputs import parse_run_tree, write_energy_table


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Parse QE pw.x outputs under a run tree into a CSV.")
    p.add_argument("--run-dir", type=Path, required=True, help="Root directory containing candidate run folders.")
    p.add_argument("--out", type=Path, required=True, help="Output CSV path.")
    return p.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _parse_args()
    df = parse_run_tree(args.run_dir)
    write_energy_table(df, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
