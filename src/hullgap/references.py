"""Elemental reference energies via MLIP relaxation.

Relaxes standard ground-state bulk structures for each element using a
given ASE calculator and caches the per-atom energies to a YAML file so
they only need to be computed once.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from ase import Atoms
from ase.build import bulk
from ase.filters import FrechetCellFilter
from ase.optimize import LBFGS

logger = logging.getLogger(__name__)

GROUND_STATE_STRUCTURES: dict[str, dict[str, Any]] = {
    "Li": {"name": "Li", "crystalstructure": "bcc", "a": 3.49},
    "Na": {"name": "Na", "crystalstructure": "bcc", "a": 4.23},
    "K":  {"name": "K",  "crystalstructure": "bcc", "a": 5.23},
    "Be": {"name": "Be", "crystalstructure": "hcp", "a": 2.29, "c": 3.58},
    "Mg": {"name": "Mg", "crystalstructure": "hcp", "a": 3.21, "c": 5.21},
    "Ca": {"name": "Ca", "crystalstructure": "fcc", "a": 5.58},
    "Sr": {"name": "Sr", "crystalstructure": "fcc", "a": 6.08},
    "Ba": {"name": "Ba", "crystalstructure": "bcc", "a": 5.02},
    "Sc": {"name": "Sc", "crystalstructure": "hcp", "a": 3.31, "c": 5.27},
    "Ti": {"name": "Ti", "crystalstructure": "hcp", "a": 2.95, "c": 4.68},
    "V":  {"name": "V",  "crystalstructure": "bcc", "a": 3.02},
    "Cr": {"name": "Cr", "crystalstructure": "bcc", "a": 2.88},
    "Mn": {"name": "Mn", "crystalstructure": "bcc", "a": 8.91},
    "Fe": {"name": "Fe", "crystalstructure": "bcc", "a": 2.87},
    "Co": {"name": "Co", "crystalstructure": "hcp", "a": 2.51, "c": 4.07},
    "Ni": {"name": "Ni", "crystalstructure": "fcc", "a": 3.52},
    "Cu": {"name": "Cu", "crystalstructure": "fcc", "a": 3.61},
    "Zn": {"name": "Zn", "crystalstructure": "hcp", "a": 2.66, "c": 4.95},
    "Ga": {"name": "Ga", "crystalstructure": "orthorhombic", "a": 4.52},
    "Al": {"name": "Al", "crystalstructure": "fcc", "a": 4.05},
    "Si": {"name": "Si", "crystalstructure": "diamond", "a": 5.43},
    "Ge": {"name": "Ge", "crystalstructure": "diamond", "a": 5.66},
    "Sn": {"name": "Sn", "crystalstructure": "diamond", "a": 6.49},
    "Pb": {"name": "Pb", "crystalstructure": "fcc", "a": 4.95},
    "Bi": {"name": "Bi", "crystalstructure": "rhombohedral"},
    "Sb": {"name": "Sb", "crystalstructure": "rhombohedral"},
    "As": {"name": "As", "crystalstructure": "rhombohedral"},
    "Se": {"name": "Se", "crystalstructure": "hexagonal"},
    "Te": {"name": "Te", "crystalstructure": "hexagonal"},
    "B":  {"name": "B",  "crystalstructure": "rhombohedral"},
    "C":  {"name": "C",  "crystalstructure": "diamond", "a": 3.57},
    "N":  {"name": "N",  "crystalstructure": "diamond", "a": 5.65},
    "O":  {"name": "O",  "crystalstructure": "fcc", "a": 6.83},
    "F":  {"name": "F",  "crystalstructure": "fcc", "a": 7.28},
    "S":  {"name": "S",  "crystalstructure": "fcc", "a": 10.47},
    "P":  {"name": "P",  "crystalstructure": "diamond", "a": 7.17},
    "Y":  {"name": "Y",  "crystalstructure": "hcp", "a": 3.65, "c": 5.73},
    "Zr": {"name": "Zr", "crystalstructure": "hcp", "a": 3.23, "c": 5.15},
    "Nb": {"name": "Nb", "crystalstructure": "bcc", "a": 3.30},
    "Mo": {"name": "Mo", "crystalstructure": "bcc", "a": 3.15},
    "Ru": {"name": "Ru", "crystalstructure": "hcp", "a": 2.71, "c": 4.28},
    "Rh": {"name": "Rh", "crystalstructure": "fcc", "a": 3.80},
    "Pd": {"name": "Pd", "crystalstructure": "fcc", "a": 3.89},
    "Ag": {"name": "Ag", "crystalstructure": "fcc", "a": 4.09},
    "Cd": {"name": "Cd", "crystalstructure": "hcp", "a": 2.98, "c": 5.62},
    "In": {"name": "In", "crystalstructure": "tetragonal", "a": 3.25, "c": 4.95},
    "Hf": {"name": "Hf", "crystalstructure": "hcp", "a": 3.19, "c": 5.05},
    "Ta": {"name": "Ta", "crystalstructure": "bcc", "a": 3.30},
    "W":  {"name": "W",  "crystalstructure": "bcc", "a": 3.16},
    "Re": {"name": "Re", "crystalstructure": "hcp", "a": 2.76, "c": 4.46},
    "Os": {"name": "Os", "crystalstructure": "hcp", "a": 2.73, "c": 4.32},
    "Ir": {"name": "Ir", "crystalstructure": "fcc", "a": 3.84},
    "Pt": {"name": "Pt", "crystalstructure": "fcc", "a": 3.92},
    "Au": {"name": "Au", "crystalstructure": "fcc", "a": 4.08},
    "La": {"name": "La", "crystalstructure": "hcp", "a": 3.77, "c": 12.14},
    "Ce": {"name": "Ce", "crystalstructure": "fcc", "a": 5.16},
}


def _build_element_atoms(symbol: str) -> Atoms:
    """Build an ASE Atoms object for a pure element's ground state."""
    if symbol in GROUND_STATE_STRUCTURES:
        params = GROUND_STATE_STRUCTURES[symbol].copy()
        name = params.pop("name")
        try:
            return bulk(name, **params)
        except Exception:
            pass

    try:
        return bulk(symbol)
    except Exception as exc:
        raise ValueError(
            f"Cannot build bulk structure for {symbol}. "
            "Add it to GROUND_STATE_STRUCTURES in references.py."
        ) from exc


def _relax_and_get_energy(
    atoms: Atoms,
    calculator: Any,
    fmax: float = 0.01,
    max_steps: int = 500,
) -> float:
    """Relax *atoms* with *calculator* and return energy per atom."""
    atoms = atoms.copy()
    atoms.calc = calculator
    opt_target = FrechetCellFilter(atoms)
    opt = LBFGS(opt_target, logfile=None)
    opt.run(fmax=fmax, steps=max_steps)
    return float(atoms.get_potential_energy()) / len(atoms)


def get_elemental_references(
    elements: list[str],
    calculator: Any,
    cache_path: Path | None = None,
) -> dict[str, float]:
    """Return ``{element: energy_per_atom}`` for each element.

    If *cache_path* points to an existing YAML file, cached values are
    loaded first; only missing elements are relaxed and appended.
    """
    cached: dict[str, float] = {}
    if cache_path is not None and cache_path.exists():
        with cache_path.open(encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        cached = {str(k): float(v) for k, v in raw.items() if isinstance(v, (int, float))}

    refs: dict[str, float] = {}
    updated = False
    for el in elements:
        if el in cached:
            refs[el] = cached[el]
            logger.info("Reference %s: %.6f eV/atom (cached)", el, refs[el])
            continue
        logger.info("Computing reference energy for %s …", el)
        atoms = _build_element_atoms(el)
        e_per_atom = _relax_and_get_energy(atoms, calculator)
        refs[el] = e_per_atom
        cached[el] = e_per_atom
        updated = True
        logger.info("Reference %s: %.6f eV/atom", el, e_per_atom)

    if updated and cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(cached, f, default_flow_style=False)
        logger.info("Wrote reference cache to %s", cache_path)

    return refs
