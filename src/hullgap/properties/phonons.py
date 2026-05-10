"""Phonon stability and thermal property calculations via MatterSim + phonopy."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ase import Atoms
    from ase.calculators.calculator import Calculator

logger = logging.getLogger(__name__)


def compute_phonons(
    atoms: Atoms,
    calc: Calculator | None = None,
    supercell_matrix: tuple[int, int, int] = (2, 2, 2),
    temperature: float = 300.0,
    work_dir: str | Path | None = None,
    amplitude: float = 0.01,
) -> dict:
    """Compute phonon properties for a relaxed structure.

    Uses MatterSim's PhononWorkflow (frozen-phonon method with phonopy).
    The calculator attached to *atoms* is used if *calc* is not provided.

    Parameters
    ----------
    atoms
        Relaxed ASE Atoms object. Must have a calculator attached or *calc*
        must be provided.
    calc
        ASE calculator to use for force evaluations. If None, uses atoms.calc.
    supercell_matrix
        Diagonal supercell expansion, e.g. (2,2,2) gives a 2x2x2 supercell.
    temperature
        Temperature (K) at which to evaluate thermal properties.
    work_dir
        Directory for intermediate phonopy files. Uses a temp dir if None.
    amplitude
        Finite displacement amplitude in Angstrom.

    Returns
    -------
    dict with keys:
        has_imaginary : bool
        min_frequency_THz : float (negative means imaginary)
        free_energy_kJ_mol : float
        entropy_J_K_mol : float
        heat_capacity_J_K_mol : float
    """
    try:
        from mattersim.applications.phonon import PhononWorkflow
    except ImportError as exc:
        raise ImportError(
            "MatterSim phonon support requires mattersim. "
            "Install with: pip install mattersim"
        ) from exc

    if calc is not None:
        atoms.calc = calc

    if atoms.calc is None:
        raise ValueError("atoms must have a calculator attached or calc must be provided")

    if work_dir is None:
        work_dir = tempfile.mkdtemp(prefix="hullgap_phonon_")
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    sc_matrix = np.diag(supercell_matrix)

    logger.info(
        "Running phonon calculation: supercell=%s, amplitude=%.3f A",
        supercell_matrix,
        amplitude,
    )

    ph = PhononWorkflow(
        atoms=atoms,
        find_prim=False,
        work_dir=str(work_dir),
        amplitude=amplitude,
        supercell_matrix=sc_matrix,
    )

    has_imaginary, frequencies = ph.run()

    all_freqs = np.array(frequencies).flatten()
    min_freq = float(np.min(all_freqs)) if len(all_freqs) > 0 else 0.0

    thermal = _get_thermal_properties(ph, temperature)

    result = {
        "has_imaginary": bool(has_imaginary),
        "min_frequency_THz": min_freq,
        "free_energy_kJ_mol": thermal.get("free_energy_kJ_mol"),
        "entropy_J_K_mol": thermal.get("entropy_J_K_mol"),
        "heat_capacity_J_K_mol": thermal.get("heat_capacity_J_K_mol"),
    }

    logger.info(
        "Phonon result: imaginary=%s, min_freq=%.3f THz, Cv=%.2f J/(K·mol)",
        result["has_imaginary"],
        result["min_frequency_THz"],
        result["heat_capacity_J_K_mol"] or 0.0,
    )

    return result


def _get_thermal_properties(ph: object, temperature: float) -> dict:
    """Extract thermal properties from the PhononWorkflow's phonopy object."""
    try:
        phonopy_obj = ph.phonon
        phonopy_obj.run_mesh([20, 20, 20])
        phonopy_obj.run_thermal_properties(
            t_min=temperature, t_max=temperature, t_step=1
        )
        tp = phonopy_obj.get_thermal_properties_dict()

        idx = 0
        return {
            "free_energy_kJ_mol": float(tp["free_energy"][idx]),
            "entropy_J_K_mol": float(tp["entropy"][idx]),
            "heat_capacity_J_K_mol": float(tp["heat_capacity"][idx]),
        }
    except Exception as exc:
        logger.warning("Could not extract thermal properties: %s", exc)
        return {
            "free_energy_kJ_mol": None,
            "entropy_J_K_mol": None,
            "heat_capacity_J_K_mol": None,
        }
