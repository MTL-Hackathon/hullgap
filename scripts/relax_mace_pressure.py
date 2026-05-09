#!/usr/bin/env python3
"""
Relax structures under applied pressure using MACE-MP-0 with multiprocessing.

Reads extxyz files (e.g. from generate_random_structures.py), relaxes each
structure with an applied hydrostatic pressure via FIRE, and writes results
to CSV.

Usage:
    python scripts/relax_mace_pressure.py --structures data/candidates/Co-Bi --pressure 50
    python scripts/relax_mace_pressure.py --structures data/candidates/Co-Bi --pressure 0 --out data/results
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import signal
import sys
import time
import warnings
from math import gcd
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD.*")

# ---- globals set once per worker by _init_worker ----
_calc = None
_pressure_ev_a3 = None
_fmax = None
_max_steps = None


def _init_worker(model: str, pressure_gpa: float, fmax: float, max_steps: int):
    """Each worker loads its own MACE calculator to avoid pickling issues."""
    global _calc, _pressure_ev_a3, _fmax, _max_steps

    import torch
    torch.set_num_threads(1)

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    warnings.filterwarnings("ignore")

    from ase import units
    from mace.calculators import mace_mp

    _calc = mace_mp(model=model, device="cpu", default_dtype="float32")
    _pressure_ev_a3 = pressure_gpa * units.GPa
    _fmax = fmax
    _max_steps = max_steps


def reduced_formula(n_co: int, n_bi: int) -> str:
    if n_co == 0 and n_bi == 0:
        return "?"
    g = gcd(n_co, n_bi) if n_co and n_bi else max(n_co, n_bi)
    co = n_co // g if n_co else 0
    bi = n_bi // g if n_bi else 0

    def p(el: str, n: int) -> str:
        return "" if n == 0 else el if n == 1 else f"{el}{n}"

    return p("Co", co) + p("Bi", bi)


def _relax_one(task: dict) -> dict | None:
    """Worker function: relax one structure. Receives serialised Atoms data."""
    from ase import Atoms, units
    from ase.filters import FrechetCellFilter
    from ase.optimize import FIRE

    atoms = Atoms(
        symbols=task["symbols"],
        positions=np.array(task["positions"]),
        cell=np.array(task["cell"]),
        pbc=True,
    )

    try:
        atoms.calc = _calc
        ecf = FrechetCellFilter(atoms, scalar_pressure=_pressure_ev_a3)
        opt = FIRE(ecf, logfile=None)
        opt.run(fmax=_fmax, steps=_max_steps)

        energy_ev = float(atoms.get_potential_energy())
        volume_a3 = float(atoms.get_volume())
        enthalpy_ev = energy_ev + _pressure_ev_a3 * volume_a3

        symbols = atoms.get_chemical_symbols()
        n_co = symbols.count("Co")
        n_bi = symbols.count("Bi")
        n_atoms = len(atoms)

        return {
            "n_Co": n_co,
            "n_Bi": n_bi,
            "n_atoms": n_atoms,
            "formula": reduced_formula(n_co, n_bi),
            "x_Co": n_co / n_atoms if n_atoms > 0 else 0.0,
            "energy_eV": energy_ev,
            "volume_A3": volume_a3,
            "enthalpy_eV": enthalpy_ev,
            "enthalpy_eV_atom": enthalpy_ev / n_atoms,
            "pressure_GPa": task["pressure_gpa"],
            "converged": True,
            "n_steps": opt.nsteps,
            "source_file": task["source_file"],
            "source_index": task["source_index"],
        }
    except Exception:
        return None


def main() -> None:
    ap = argparse.ArgumentParser(description="Relax Co-Bi structures with MACE-MP-0")
    ap.add_argument("--structures", default="data/candidates/Co-Bi",
                    help="directory of extxyz files (default: data/candidates/Co-Bi)")
    ap.add_argument("--pressure", type=float, default=50.0,
                    help="external pressure in GPa (default: 50)")
    ap.add_argument("--model", default="small",
                    help="MACE-MP-0 model size: small/medium/large (default: small)")
    ap.add_argument("--fmax", type=float, default=0.05,
                    help="force convergence criterion eV/A (default: 0.05)")
    ap.add_argument("--max-steps", type=int, default=100,
                    help="max optimizer steps per structure (default: 100)")
    ap.add_argument("--workers", type=int, default=0,
                    help="number of worker processes (default: cpu_count - 2)")
    ap.add_argument("--out", default="data/results",
                    help="output directory (default: data/results/)")
    args = ap.parse_args()

    struct_dir = Path(args.structures)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_workers = args.workers or max(1, os.cpu_count() - 2)

    # ---- load all structures and serialise for pickling ----
    from ase.io import read as ase_read

    extxyz_files = sorted(struct_dir.glob("*.extxyz"))
    if not extxyz_files:
        raise SystemExit(f"No .extxyz files found in {struct_dir}")

    tasks: list[dict] = []
    for fpath in extxyz_files:
        structures = ase_read(str(fpath), index=":", format="extxyz")
        for i, atoms in enumerate(structures):
            tasks.append({
                "symbols": atoms.get_chemical_symbols(),
                "positions": atoms.get_positions().tolist(),
                "cell": atoms.get_cell().tolist(),
                "pressure_gpa": args.pressure,
                "source_file": fpath.name,
                "source_index": i,
            })

    print(f"{len(tasks)} structures loaded from {len(extxyz_files)} files.")
    print(f"Launching {n_workers} workers with MACE-MP-0 ({args.model}) at {args.pressure} GPa ...")

    t0 = time.time()

    ctx = mp.get_context("spawn")
    with ctx.Pool(
        processes=n_workers,
        initializer=_init_worker,
        initargs=(args.model, args.pressure, args.fmax, args.max_steps),
    ) as pool:
        all_results: list[dict] = []
        n_fail = 0

        for i, result in enumerate(pool.imap_unordered(_relax_one, tasks), 1):
            if result is not None:
                all_results.append(result)
            else:
                n_fail += 1

            if i % 20 == 0 or i == len(tasks):
                elapsed = time.time() - t0
                speed = i / elapsed
                eta = (len(tasks) - i) / speed if speed > 0 else 0
                print(f"  [{i}/{len(tasks)}]  ok={len(all_results)}  "
                      f"fail={n_fail}  {speed:.1f} struct/s  "
                      f"ETA {eta:.0f}s")

    if not all_results:
        raise SystemExit("All relaxations failed")

    df = pd.DataFrame(all_results)
    csv_path = out_dir / f"mace_relaxed_{int(args.pressure)}GPa.csv"
    df.to_csv(csv_path, index=False)

    elapsed = time.time() - t0
    print(f"\nDone: {len(all_results)} ok, {n_fail} failed in {elapsed:.1f}s "
          f"({len(all_results)/elapsed:.1f} struct/s)")
    print(f"Results: {csv_path}")


if __name__ == "__main__":
    main()
