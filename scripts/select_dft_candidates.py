#!/usr/bin/env python3
"""
CLI: pick top MLIP-ranked structures for targeted DFT validation.

DFT validates only a short list after the MLIP pipeline; it does not replace
high-throughput screening.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from hullgap.dft.select_candidates import load_hull_scores_csv, select_top_candidates, write_candidate_list


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Select top candidates for DFT from MLIP hull scores.")
    p.add_argument("--hull-scores", type=Path, required=True, help="Path to hull_scores CSV.")
    p.add_argument("--relaxed-dir", type=Path, required=True, help="Directory with relaxed CIFs.")
    p.add_argument("--top-n", type=int, default=10, help="Maximum number of candidates.")
    p.add_argument("--max-atoms", type=int, default=40, help="Skip structures larger than this.")
    p.add_argument("--out", type=Path, required=True, help="Output CSV path.")
    return p.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _parse_args()
    df = load_hull_scores_csv(args.hull_scores)
    selected = select_top_candidates(df, args.relaxed_dir, top_n=args.top_n, max_atoms=args.max_atoms)
    write_candidate_list(selected, args.out)
    logging.info("Selected %d candidates for DFT.", len(selected))
    return 0


if __name__ == "__main__":
    sys.exit(main())
