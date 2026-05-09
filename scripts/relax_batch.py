#!/usr/bin/env python3
"""
CLI: batch-relax candidate crystal structures with an MLIP.

Usage
-----
python scripts/relax_batch.py \
    --input  data/candidates/Co-Bi \
    --output data/relaxed/Co-Bi \
    --model  chgnet \
    --fmax   0.05 \
    --max-steps 300

With trajectory + step-log:

python scripts/relax_batch.py \
    --input  data/candidates/Co-Bi \
    --output data/relaxed/Co-Bi \
    --model  chgnet \
    --fmax   0.05 \
    --max-steps 300 \
    --save-trajectory \
    --save-step-log

MLIP energies are screening predictions — candidates should be validated with
DFT before any claims of thermodynamic stability.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from hullgap.calculators import AVAILABLE_MODELS, get_calculator
from hullgap.relax import relax_structure

logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Batch-relax candidate structures with a universal MLIP."
    )
    p.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Directory containing candidate CIF files.",
    )
    p.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Directory to write relaxed CIF files.",
    )
    p.add_argument(
        "--model",
        type=str,
        default="chgnet",
        choices=list(AVAILABLE_MODELS),
        help="MLIP model to use (default: chgnet).",
    )
    p.add_argument(
        "--fmax",
        type=float,
        default=0.05,
        help="Force convergence threshold in eV/Å (default: 0.05).",
    )
    p.add_argument(
        "--max-steps",
        type=int,
        default=300,
        help="Maximum optimiser steps per structure (default: 300).",
    )
    p.add_argument(
        "--relax-cell",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Relax the unit cell (default: yes).",
    )
    p.add_argument(
        "--results-csv",
        type=Path,
        default=None,
        help="Custom path for results CSV. Default: data/results/relaxation_results_<SYSTEM>_<MODEL>.csv",
    )

    # Trajectory / step-log options
    p.add_argument(
        "--save-trajectory",
        action="store_true",
        default=False,
        help="Save .traj and .extxyz trajectory files for each structure.",
    )
    p.add_argument(
        "--trajectory-dir",
        type=Path,
        default=None,
        help="Directory for trajectory files. Default: data/trajectories/<SYSTEM>",
    )
    p.add_argument(
        "--save-step-log",
        action="store_true",
        default=False,
        help="Save a per-step CSV with energy, forces, volume, and cell parameters.",
    )
    return p.parse_args()


def _infer_system_name(input_dir: Path) -> str:
    """Best-effort system name from the input directory (e.g. 'Co-Bi')."""
    return input_dir.resolve().name


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    args = _parse_args()
    input_dir: Path = args.input.resolve()
    output_dir: Path = args.output.resolve()
    model: str = args.model
    system = _infer_system_name(input_dir)

    if not input_dir.is_dir():
        logger.error("Input directory does not exist: %s", input_dir)
        return 1

    cif_files = sorted(input_dir.glob("*.cif"))
    if not cif_files:
        logger.error("No .cif files found in %s", input_dir)
        return 1

    logger.info(
        "Found %d candidate CIF(s) in %s — model=%s  fmax=%.3f  max_steps=%d",
        len(cif_files),
        input_dir,
        model,
        args.fmax,
        args.max_steps,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve trajectory directory
    traj_dir: str | None = None
    if args.save_trajectory:
        td = args.trajectory_dir or Path("data/trajectories") / system
        td.mkdir(parents=True, exist_ok=True)
        traj_dir = str(td)
        logger.info("Trajectories will be saved to %s", traj_dir)

    try:
        calc = get_calculator(model)
    except ImportError as exc:
        logger.error("%s", exc)
        return 1

    results: list[dict] = []
    all_step_rows: list[dict] = []

    for cif_path in tqdm(cif_files, desc=f"Relaxing ({model})", unit="struct"):
        out_cif = output_dir / cif_path.name
        rec = relax_structure(
            input_file=str(cif_path),
            output_file=str(out_cif),
            model=model,
            fmax=args.fmax,
            max_steps=args.max_steps,
            relax_cell=args.relax_cell,
            save_trajectory=args.save_trajectory,
            trajectory_dir=traj_dir,
            save_step_log=args.save_step_log,
            _calculator=calc,
        )

        step_log = rec.pop("_step_log", None)
        if step_log:
            all_step_rows.extend(step_log)

        results.append(rec)

    # --- Write main results CSV ---
    df = pd.DataFrame(results)

    if args.results_csv is not None:
        csv_path = args.results_csv
    else:
        csv_path = Path("data/results") / f"relaxation_results_{system}_{model}.csv"

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    logger.info("Wrote relaxation results to %s", csv_path)

    # --- Write per-step log CSV ---
    if all_step_rows:
        step_csv = Path("data/results") / f"relaxation_steps_{system}_{model}.csv"
        step_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(all_step_rows).to_csv(step_csv, index=False)
        logger.info("Wrote per-step log (%d rows) to %s", len(all_step_rows), step_csv)

    n_ok = (df["status"] != "failed_relaxation").sum()
    n_fail = (df["status"] == "failed_relaxation").sum()
    logger.info("Done: %d succeeded, %d failed out of %d total.", n_ok, n_fail, len(df))

    return 0


if __name__ == "__main__":
    sys.exit(main())
