"""Model-agnostic MLIP relaxation of crystal structures.

This module provides the main ``relax_structure`` function used by the batch
relaxation CLI.  It is intentionally thin: it reads a structure file, attaches
an ASE calculator (via :mod:`hullgap.calculators`), runs a geometry
optimisation, and returns a result dict suitable for CSV serialisation.

MLIP energies are *screening* energies — they prioritise candidates for DFT
validation but do **not** prove thermodynamic stability on their own.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
from ase.filters import FrechetCellFilter
from ase.io import read as ase_read
from ase.optimize import LBFGS
from pymatgen.io.ase import AseAtomsAdaptor

from hullgap.calculators import get_calculator

logger = logging.getLogger(__name__)


def _candidate_id_from_path(path: Path) -> str:
    return path.stem


def relax_structure(
    input_file: str,
    output_file: str,
    model: str = "chgnet",
    fmax: float = 0.05,
    max_steps: int = 300,
    relax_cell: bool = True,
    _calculator=None,
) -> dict:
    """Relax a single crystal structure with an MLIP and write the result.

    Parameters
    ----------
    input_file
        Path to the input CIF (or any ASE-readable structure file).
    output_file
        Destination path for the relaxed CIF.
    model
        MLIP backend name (``"chgnet"`` or ``"mace"``).
    fmax
        Force convergence threshold in eV/Angstrom.
    max_steps
        Maximum optimiser steps.
    relax_cell
        Whether to relax the unit-cell (volume + shape) alongside positions.
    _calculator
        Pre-loaded ASE calculator.  When *None* (the default) a fresh
        calculator is created via :func:`get_calculator`.  Passing a
        calculator avoids reloading the model for every structure in a
        batch run.

    Returns
    -------
    dict
        Relaxation result record (see column list in the README).
    """
    input_path = Path(input_file)
    output_path = Path(output_file)
    candidate_id = _candidate_id_from_path(input_path)

    result: dict = {
        "candidate_id": candidate_id,
        "formula": "",
        "status": "failed_relaxation",
        "initial_file": str(input_path),
        "relaxed_file": "",
        "energy_total_eV": np.nan,
        "energy_per_atom_eV": np.nan,
        "max_force_eV_A": np.nan,
        "volume_per_atom": np.nan,
        "n_steps": 0,
        "model_name": model,
        "error_message": "",
    }

    try:
        atoms = ase_read(str(input_path))
        result["formula"] = atoms.get_chemical_formula(mode="metal")

        calc = _calculator if _calculator is not None else get_calculator(model)
        atoms.calc = calc

        if relax_cell:
            opt_target = FrechetCellFilter(atoms)
        else:
            opt_target = atoms

        optimiser = LBFGS(opt_target, logfile=None)

        t0 = time.perf_counter()
        converged = optimiser.run(fmax=fmax, steps=max_steps)
        elapsed = time.perf_counter() - t0

        n_steps = optimiser.nsteps
        energy = atoms.get_potential_energy()
        forces = atoms.get_forces()
        max_force = float(np.max(np.linalg.norm(forces, axis=1)))
        n_atoms = len(atoms)
        volume = atoms.get_volume()

        pmg_struct = AseAtomsAdaptor.get_structure(atoms)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pmg_struct.to(filename=str(output_path))

        result.update(
            {
                "formula": pmg_struct.composition.reduced_formula,
                "status": "converged" if converged else "max_steps_reached",
                "relaxed_file": str(output_path),
                "energy_total_eV": float(energy),
                "energy_per_atom_eV": float(energy) / n_atoms,
                "max_force_eV_A": max_force,
                "volume_per_atom": volume / n_atoms,
                "n_steps": n_steps,
            }
        )
        logger.info(
            "%s  %s  E/atom=%.4f eV  fmax=%.4f  steps=%d  (%.1fs)",
            candidate_id,
            result["formula"],
            result["energy_per_atom_eV"],
            max_force,
            n_steps,
            elapsed,
        )
    except Exception as exc:  # noqa: BLE001
        result["error_message"] = str(exc)
        logger.error("Failed to relax %s: %s", candidate_id, exc)

    return result
