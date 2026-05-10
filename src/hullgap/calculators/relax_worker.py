"""Multiprocessing-safe relaxation worker for MLIP calculators.

This module exists as a standalone file (not inline in a notebook) so that
``multiprocessing`` with the ``spawn`` start method can pickle and import it.
"""

from __future__ import annotations

import time
import traceback

import numpy as np
from ase.filters import FrechetCellFilter
from ase.optimize import LBFGS
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor


def relax_single(args: tuple) -> dict:
    """Relax one structure with the given MLIP model.

    Parameters
    ----------
    args
        Tuple of (proto_name, struct_dict, model_name, fmax, max_steps).
    """
    proto_name, struct_dict, model_name, fmax, max_steps = args

    from hullgap.calculators import get_calculator

    struct = Structure.from_dict(struct_dict)
    atoms = AseAtomsAdaptor.get_atoms(struct)

    try:
        calc = get_calculator(model_name)
        atoms.calc = calc

        opt_target = FrechetCellFilter(atoms)
        opt = LBFGS(opt_target, logfile=None)

        t0 = time.perf_counter()
        converged = opt.run(fmax=fmax, steps=max_steps)
        wall_time = time.perf_counter() - t0

        energy = float(atoms.get_potential_energy())
        forces = atoms.get_forces()
        max_force = float(np.max(np.linalg.norm(forces, axis=1)))
        n_atoms = len(atoms)
        volume = atoms.get_volume()

        return {
            "prototype": proto_name,
            "model": model_name,
            "formula": atoms.get_chemical_formula(mode="metal"),
            "n_atoms": n_atoms,
            "energy_total_eV": energy,
            "energy_per_atom_eV": energy / n_atoms,
            "max_force_eV_A": max_force,
            "volume_per_atom_A3": volume / n_atoms,
            "n_steps": opt.nsteps,
            "converged": bool(converged),
            "wall_time_s": wall_time,
            "status": "converged" if converged else "max_steps_reached",
            "error": "",
        }
    except Exception as exc:
        return {
            "prototype": proto_name,
            "model": model_name,
            "formula": struct.composition.reduced_formula,
            "n_atoms": len(struct),
            "energy_total_eV": np.nan,
            "energy_per_atom_eV": np.nan,
            "max_force_eV_A": np.nan,
            "volume_per_atom_A3": np.nan,
            "n_steps": 0,
            "converged": False,
            "wall_time_s": 0.0,
            "status": "failed",
            "error": traceback.format_exc(),
        }
