"""Elastic property calculations from MLIP stress tensors.

Computes the full 6x6 elastic tensor via the stress-strain method,
then derives bulk modulus, shear modulus, Young's modulus, and Poisson's
ratio using Voigt-Reuss-Hill averaging.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ase import Atoms
    from ase.calculators.calculator import Calculator

logger = logging.getLogger(__name__)

# Voigt notation: xx=0, yy=1, zz=2, yz=3, xz=4, xy=5
_STRAIN_MODES = [
    np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=float),  # e_xx
    np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=float),  # e_yy
    np.array([[0, 0, 0], [0, 0, 0], [0, 0, 1]], dtype=float),  # e_zz
    np.array([[0, 0, 0], [0, 0, 0.5], [0, 0.5, 0]], dtype=float),  # e_yz
    np.array([[0, 0, 0.5], [0, 0, 0], [0.5, 0, 0]], dtype=float),  # e_xz
    np.array([[0, 0.5, 0], [0.5, 0, 0], [0, 0, 0]], dtype=float),  # e_xy
]


def compute_elastic_properties(
    atoms: Atoms,
    calc: Calculator | None = None,
    max_strain: float = 0.01,
    n_steps: int = 5,
) -> dict:
    """Compute elastic constants from stress-strain finite differences.

    Applies small strains along 6 independent modes, evaluates the stress
    tensor at each deformation, and fits the elastic tensor Cij.

    Parameters
    ----------
    atoms
        Relaxed ASE Atoms object (equilibrium geometry).
    calc
        ASE calculator for stress evaluation. Uses atoms.calc if None.
    max_strain
        Maximum strain magnitude (symmetric: -max_strain to +max_strain).
    n_steps
        Number of strain steps (odd recommended for symmetric sampling).

    Returns
    -------
    dict with keys:
        elastic_tensor_GPa : 6x6 numpy array
        bulk_modulus_GPa : float (VRH average)
        shear_modulus_GPa : float (VRH average)
        youngs_modulus_GPa : float
        poisson_ratio : float
    """
    if calc is not None:
        atoms = atoms.copy()
        atoms.calc = calc
    elif atoms.calc is None:
        raise ValueError("atoms must have a calculator attached or calc must be provided")

    logger.info("Computing elastic tensor: max_strain=%.4f, n_steps=%d", max_strain, n_steps)

    cell0 = atoms.get_cell().array.copy()
    strains = np.linspace(-max_strain, max_strain, n_steps)

    # Cij[i, j] = dσ_i / dε_j (Voigt notation)
    Cij = np.zeros((6, 6))

    for j, mode in enumerate(_STRAIN_MODES):
        stresses_voigt = []
        for eps in strains:
            deformed = _apply_strain(atoms, cell0, eps * mode)
            stress = deformed.get_stress(voigt=True)  # eV/A^3, Voigt order
            stresses_voigt.append(stress)

        stresses_voigt = np.array(stresses_voigt)  # (n_steps, 6)

        for i in range(6):
            # Linear fit: σ_i = Cij[i,j] * ε + offset
            # ASE stress convention is negative of Cauchy stress
            coeffs = np.polyfit(strains, -stresses_voigt[:, i], 1)
            Cij[i, j] = coeffs[0]

    # Convert from eV/A^3 to GPa
    eV_per_A3_to_GPa = 160.21766208
    Cij_GPa = Cij * eV_per_A3_to_GPa

    # Symmetrize
    Cij_GPa = 0.5 * (Cij_GPa + Cij_GPa.T)

    bulk_v, shear_v = _voigt_average(Cij_GPa)
    bulk_r, shear_r = _reuss_average(Cij_GPa)

    bulk_vrh = 0.5 * (bulk_v + bulk_r)
    shear_vrh = 0.5 * (shear_v + shear_r)

    if shear_vrh > 0 and (3 * bulk_vrh + shear_vrh) > 0:
        youngs = 9 * bulk_vrh * shear_vrh / (3 * bulk_vrh + shear_vrh)
        poisson = (3 * bulk_vrh - 2 * shear_vrh) / (6 * bulk_vrh + 2 * shear_vrh)
    else:
        youngs = 0.0
        poisson = 0.0

    result = {
        "elastic_tensor_GPa": Cij_GPa,
        "bulk_modulus_GPa": float(bulk_vrh),
        "shear_modulus_GPa": float(shear_vrh),
        "youngs_modulus_GPa": float(youngs),
        "poisson_ratio": float(poisson),
    }

    logger.info(
        "Elastic properties: K=%.1f GPa, G=%.1f GPa, E=%.1f GPa, ν=%.3f",
        bulk_vrh, shear_vrh, youngs, poisson,
    )

    return result


def _apply_strain(atoms: Atoms, cell0: np.ndarray, strain_matrix: np.ndarray) -> Atoms:
    """Apply a strain tensor to atoms and return the deformed copy."""
    deformed = atoms.copy()
    deformed.calc = atoms.calc

    deformation = np.eye(3) + strain_matrix
    new_cell = cell0 @ deformation.T
    deformed.set_cell(new_cell, scale_atoms=True)
    return deformed


def _voigt_average(C: np.ndarray) -> tuple[float, float]:
    """Voigt (upper bound) averages for bulk and shear modulus."""
    K_v = (
        (C[0, 0] + C[1, 1] + C[2, 2])
        + 2 * (C[0, 1] + C[0, 2] + C[1, 2])
    ) / 9.0

    G_v = (
        (C[0, 0] + C[1, 1] + C[2, 2])
        - (C[0, 1] + C[0, 2] + C[1, 2])
        + 3 * (C[3, 3] + C[4, 4] + C[5, 5])
    ) / 15.0

    return float(K_v), float(G_v)


def _reuss_average(C: np.ndarray) -> tuple[float, float]:
    """Reuss (lower bound) averages for bulk and shear modulus."""
    try:
        S = np.linalg.inv(C)
    except np.linalg.LinAlgError:
        logger.warning("Elastic tensor is singular; Reuss average unavailable")
        return _voigt_average(C)

    K_r = 1.0 / (
        (S[0, 0] + S[1, 1] + S[2, 2]) + 2 * (S[0, 1] + S[0, 2] + S[1, 2])
    )

    G_r = 15.0 / (
        4 * (S[0, 0] + S[1, 1] + S[2, 2])
        - 4 * (S[0, 1] + S[0, 2] + S[1, 2])
        + 3 * (S[3, 3] + S[4, 4] + S[5, 5])
    )

    return float(K_r), float(G_r)
