"""Model-agnostic MLIP relaxation of crystal structures.

This module provides the main ``relax_structure`` function used by the batch
relaxation CLI.  It is intentionally thin: it reads a structure file, attaches
an ASE calculator (via :mod:`hullgap.calculators`), runs a geometry
optimisation, and returns a result dict suitable for CSV serialisation.

Optionally saves a full ASE trajectory (``.traj`` / ``.extxyz``) and a
per-step log of energy, forces, volume, and cell parameters.

MLIP energies are *screening* energies — they prioritise candidates for DFT
validation but do **not** prove thermodynamic stability on their own.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.filters import FrechetCellFilter
from ase.io import read as ase_read
from ase.io import write as ase_write
from ase.io.trajectory import Trajectory
from ase.optimize import LBFGS
from pymatgen.io.ase import AseAtomsAdaptor

from hullgap.calculators import get_calculator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-step recorder
# ---------------------------------------------------------------------------

@dataclass
class _StepRecord:
    """One snapshot captured during relaxation."""
    step: int
    energy_eV: float
    energy_per_atom_eV: float
    fmax_eV_A: float
    volume_A3: float
    volume_per_atom_A3: float
    a_A: float
    b_A: float
    c_A: float
    alpha_deg: float
    beta_deg: float
    gamma_deg: float


@dataclass
class StepLogger:
    """Callback that records per-step relaxation data from an ASE Atoms object."""

    atoms: Atoms
    records: list[_StepRecord] = field(default_factory=list)
    _step: int = field(default=0, init=False)

    def __call__(self) -> None:
        atoms = self.atoms
        energy = float(atoms.get_potential_energy())
        forces = atoms.get_forces()
        fmax = float(np.max(np.linalg.norm(forces, axis=1)))
        n = len(atoms)
        vol = atoms.get_volume()
        cell = atoms.cell.cellpar()  # [a, b, c, alpha, beta, gamma]

        self.records.append(_StepRecord(
            step=self._step,
            energy_eV=energy,
            energy_per_atom_eV=energy / n,
            fmax_eV_A=fmax,
            volume_A3=vol,
            volume_per_atom_A3=vol / n,
            a_A=float(cell[0]),
            b_A=float(cell[1]),
            c_A=float(cell[2]),
            alpha_deg=float(cell[3]),
            beta_deg=float(cell[4]),
            gamma_deg=float(cell[5]),
        ))
        self._step += 1

    def to_dicts(self, candidate_id: str, model: str, traj_file: str = "") -> list[dict]:
        """Return records as a list of flat dicts for CSV export."""
        return [
            {
                "candidate_id": candidate_id,
                "model_name": model,
                "step": r.step,
                "energy_eV": r.energy_eV,
                "energy_per_atom_eV": r.energy_per_atom_eV,
                "fmax_eV_A": r.fmax_eV_A,
                "volume_A3": r.volume_A3,
                "volume_per_atom_A3": r.volume_per_atom_A3,
                "a_A": r.a_A,
                "b_A": r.b_A,
                "c_A": r.c_A,
                "alpha_deg": r.alpha_deg,
                "beta_deg": r.beta_deg,
                "gamma_deg": r.gamma_deg,
                "trajectory_file": traj_file,
            }
            for r in self.records
        ]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _candidate_id_from_path(path: Path) -> str:
    return path.stem


def _save_extxyz(traj_path: Path, extxyz_path: Path) -> None:
    """Convert an ASE .traj to a portable .extxyz file."""
    traj = Trajectory(str(traj_path), mode="r")
    ase_write(str(extxyz_path), list(traj), format="extxyz")
    traj.close()


# ---------------------------------------------------------------------------
# Main relaxation entry point
# ---------------------------------------------------------------------------

def relax_structure(
    input_file: str,
    output_file: str,
    model: str = "chgnet",
    fmax: float = 0.05,
    max_steps: int = 300,
    relax_cell: bool = True,
    save_trajectory: bool = False,
    trajectory_dir: str | None = None,
    save_step_log: bool = False,
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
    save_trajectory
        If *True*, write ``.traj`` and ``.extxyz`` trajectory files.
    trajectory_dir
        Directory for trajectory files.  Required when *save_trajectory*
        is *True*.  Files are written as
        ``<trajectory_dir>/<candidate_id>_<model>.traj`` (and ``.extxyz``).
    save_step_log
        If *True*, record per-step energy / force / cell data and include
        it in the returned dict under the key ``"_step_log"``.
    _calculator
        Pre-loaded ASE calculator.  When *None* (the default) a fresh
        calculator is created via :func:`get_calculator`.  Passing a
        calculator avoids reloading the model for every structure in a
        batch run.

    Returns
    -------
    dict
        Relaxation result record (see column list in the README).
        When *save_step_log* is *True*, includes an extra key
        ``"_step_log"`` with a list of per-step dicts.
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
        "trajectory_file": "",
    }

    traj_writer: Trajectory | None = None
    traj_path: Path | None = None
    extxyz_path: Path | None = None

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

        # --- trajectory writer ---
        if save_trajectory and trajectory_dir is not None:
            traj_dir = Path(trajectory_dir)
            traj_dir.mkdir(parents=True, exist_ok=True)
            traj_path = traj_dir / f"{candidate_id}_{model}.traj"
            extxyz_path = traj_dir / f"{candidate_id}_{model}.extxyz"
            traj_writer = Trajectory(str(traj_path), mode="w", atoms=atoms)
            optimiser.attach(traj_writer.write)

        # --- step logger ---
        step_logger: StepLogger | None = None
        if save_step_log or save_trajectory:
            step_logger = StepLogger(atoms=atoms)
            optimiser.attach(step_logger)

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

        # --- finalise trajectory ---
        if traj_writer is not None:
            traj_writer.close()
            traj_writer = None
            result["trajectory_file"] = str(traj_path)
            if extxyz_path is not None:
                _save_extxyz(traj_path, extxyz_path)
                logger.info("Trajectory: %s  (+.extxyz)", traj_path)

        # --- attach step log to result ---
        if step_logger is not None:
            traj_str = str(traj_path) if traj_path else ""
            result["_step_log"] = step_logger.to_dicts(
                candidate_id, model, traj_str
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
    finally:
        if traj_writer is not None:
            traj_writer.close()

    return result
