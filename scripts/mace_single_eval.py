#!/usr/bin/env python3
"""
Relax one CIF from reports/cobi_top20_shortlist with MACE-MP-0
using multiprocessing, then compute formation energy relative to
elemental references (Co and Bi).

Usage (from project root, with .venv activated):
    python scripts/mace_single_eval.py
"""

from __future__ import annotations

import multiprocessing as mp
import os
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD.*")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SHORTLIST_DIR = PROJECT_ROOT / "reports" / "cobi_top20_shortlist"

# Standard elemental reference structures:
#   Co  — HCP (space group 194, P6_3/mmc), 2 atoms/cell
#   Bi  — rhombohedral A7 (space group 166, R-3m), 2 atoms/cell
ELEMENTAL_STRUCTURES = {
    "Co": {
        "symbols": ["Co", "Co"],
        "cell": [
            [2.507, 0.0, 0.0],
            [-1.2535, 2.1710, 0.0],
            [0.0, 0.0, 4.069],
        ],
        "scaled_positions": [
            [1 / 3, 2 / 3, 1 / 4],
            [2 / 3, 1 / 3, 3 / 4],
        ],
    },
    "Bi": {
        "symbols": ["Bi", "Bi"],
        "cell": [
            [4.546, 0.0, 0.0],
            [-2.273, 3.936, 0.0],
            [0.0, 0.0, 11.862],
        ],
        "scaled_positions": [
            [0.0, 0.0, 0.2339],
            [0.0, 0.0, 0.7661],
        ],
    },
}


# ---- worker functions (each process loads its own calculator) ----

_calc = None


def _init_worker(model: str):
    global _calc
    import torch

    torch.set_num_threads(1)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    warnings.filterwarnings("ignore")

    from mace.calculators import mace_mp

    _calc = mace_mp(model=model, device="cpu", default_dtype="float64")


def _relax_task(task: dict) -> dict:
    """Relax a single structure and return energy info."""
    from ase import Atoms
    from ase.filters import FrechetCellFilter
    from ase.optimize import FIRE

    if "scaled_positions" in task:
        atoms = Atoms(
            symbols=task["symbols"],
            scaled_positions=task["scaled_positions"],
            cell=task["cell"],
            pbc=True,
        )
    else:
        atoms = Atoms(
            symbols=task["symbols"],
            positions=task["positions"],
            cell=task["cell"],
            pbc=True,
        )

    atoms.calc = _calc
    ecf = FrechetCellFilter(atoms)
    opt = FIRE(ecf, logfile=None)
    opt.run(fmax=0.01, steps=500)

    energy = float(atoms.get_potential_energy())
    n_atoms = len(atoms)
    return {
        "label": task["label"],
        "energy_eV": energy,
        "n_atoms": n_atoms,
        "energy_eV_per_atom": energy / n_atoms,
        "converged": opt.converged(),
        "n_steps": opt.nsteps,
        "volume_A3": float(atoms.get_volume()),
    }


def main() -> None:
    from ase.io import read as ase_read

    cif_files = sorted(SHORTLIST_DIR.glob("*.cif"))
    if not cif_files:
        raise SystemExit(f"No CIF files found in {SHORTLIST_DIR}")

    print(f"Found {len(cif_files)} CIF files in {SHORTLIST_DIR.name}/\n")

    tasks: list[dict] = []

    # Elemental references first
    for elem, info in ELEMENTAL_STRUCTURES.items():
        tasks.append(
            {
                "label": f"ref_{elem}",
                "symbols": info["symbols"],
                "cell": info["cell"],
                "scaled_positions": info["scaled_positions"],
            }
        )

    # All 20 candidates
    cif_meta: list[dict] = []
    for cif_path in cif_files:
        atoms = ase_read(str(cif_path))
        symbols = atoms.get_chemical_symbols()
        n_Co = symbols.count("Co")
        n_Bi = symbols.count("Bi")
        short_name = cif_path.stem[:40]
        tasks.append(
            {
                "label": short_name,
                "symbols": symbols,
                "positions": atoms.get_positions().tolist(),
                "cell": atoms.get_cell().tolist(),
            }
        )
        cif_meta.append({"name": short_name, "n_Co": n_Co, "n_Bi": n_Bi, "n_atoms": len(atoms)})
        print(f"  {short_name}  Co{n_Co}Bi{n_Bi} ({len(atoms)} atoms)")

    n_workers = min(len(tasks), max(1, os.cpu_count() - 1))
    print(f"\nRelaxing {len(tasks)} structures ({len(cif_files)} candidates + 2 refs) "
          f"with MACE-MP-0 using {n_workers} workers …\n")

    t0 = time.time()
    ctx = mp.get_context("spawn")
    with ctx.Pool(
        processes=n_workers,
        initializer=_init_worker,
        initargs=("medium",),
    ) as pool:
        completed = 0
        results_map: dict[str, dict] = {}
        for result in pool.imap_unordered(_relax_task, tasks):
            completed += 1
            results_map[result["label"]] = result
            elapsed = time.time() - t0
            speed = completed / elapsed if elapsed > 0 else 0
            eta = (len(tasks) - completed) / speed if speed > 0 else 0
            print(f"  [{completed}/{len(tasks)}] {result['label']:<40}  "
                  f"E={result['energy_eV_per_atom']:.4f} eV/at  "
                  f"({speed:.1f} struct/s, ETA {eta:.0f}s)")

    elapsed = time.time() - t0
    print(f"\nAll done in {elapsed:.1f}s\n")

    # ---- Extract reference energies ----
    mu_Co = results_map["ref_Co"]["energy_eV_per_atom"]
    mu_Bi = results_map["ref_Bi"]["energy_eV_per_atom"]

    print(f"  Reference energies:")
    print(f"    μ(Co) = {mu_Co:.4f} eV/atom   (HCP)")
    print(f"    μ(Bi) = {mu_Bi:.4f} eV/atom   (A7)")

    # ---- Formation energy for every candidate ----
    rows = []
    for meta in cif_meta:
        r = results_map[meta["name"]]
        n_Co = meta["n_Co"]
        n_Bi = meta["n_Bi"]
        n_atoms = meta["n_atoms"]
        E_total = r["energy_eV"]
        E_ref = n_Co * mu_Co + n_Bi * mu_Bi
        E_form = (E_total - E_ref) / n_atoms
        rows.append({
            "name": meta["name"],
            "n_Co": n_Co,
            "n_Bi": n_Bi,
            "n_atoms": n_atoms,
            "E_eV_per_atom": r["energy_eV_per_atom"],
            "E_form_eV_per_atom": E_form,
            "converged": r["converged"],
            "steps": r["n_steps"],
        })

    import pandas as pd

    df = pd.DataFrame(rows).sort_values("E_form_eV_per_atom")

    print(f"\n{'='*90}")
    print(f" {'#':<3} {'Name':<42} {'Comp':<10} {'E/at':>8} {'E_form':>8} {'Conv':>5} {'Steps':>5}")
    print(f"{'-'*90}")
    for i, (_, row) in enumerate(df.iterrows(), 1):
        tag = "✓" if row["converged"] else "✗"
        comp = f"Co{row['n_Co']}Bi{row['n_Bi']}"
        print(f" {i:<3} {row['name']:<42} {comp:<10} "
              f"{row['E_eV_per_atom']:>8.4f} {row['E_form_eV_per_atom']:>8.4f} {tag:>5} {row['steps']:>5}")
    print(f"{'='*90}")

    n_stable = (df["E_form_eV_per_atom"] < 0).sum()
    n_marginal = ((df["E_form_eV_per_atom"] >= 0) & (df["E_form_eV_per_atom"] < 0.05)).sum()
    print(f"\n  {n_stable} predicted stable (E_form < 0)")
    print(f"  {n_marginal} marginal (0 < E_form < 0.05 eV/atom)")
    print(f"  {len(df) - n_stable - n_marginal} clearly unstable (E_form >= 0.05 eV/atom)")

    csv_path = PROJECT_ROOT / "reports" / "mace_shortlist_formation_energies.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n  Results saved to {csv_path}")


if __name__ == "__main__":
    main()
