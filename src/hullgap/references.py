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

_GPa_TO_EV_A3 = 1.0 / 160.21766208

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


def _relax_and_get_enthalpy_per_atom(
    atoms: Atoms,
    calculator: Any,
    fmax: float = 0.01,
    max_steps: int = 500,
    pressure_GPa: float = 0.0,
) -> float:
    """Relax *atoms* at *pressure_GPa* and return enthalpy (E+PV) per atom."""
    atoms = atoms.copy()
    atoms.calc = calculator
    scalar_pressure = pressure_GPa * _GPa_TO_EV_A3
    opt_target = FrechetCellFilter(atoms, scalar_pressure=scalar_pressure)
    opt = LBFGS(opt_target, logfile=None)
    opt.run(fmax=fmax, steps=max_steps)
    energy = float(atoms.get_potential_energy())
    volume = float(atoms.get_volume())
    return (energy + scalar_pressure * volume) / len(atoms)


def get_elemental_references(
    elements: list[str],
    calculator: Any,
    cache_path: Path | None = None,
    pressure_GPa: float = 0.0,
) -> dict[str, float]:
    """Return ``{element: enthalpy_per_atom}`` for each element at *pressure_GPa*.

    Values are enthalpies H = (E + PV) / N so they can be used directly as
    chemical potentials when building the convex hull at finite pressure.

    Cache keys are ``"{el}@{P:.2f}GPa"`` for P > 0 and ``"{el}"`` for 0 GPa
    (backward-compatible with existing caches).
    """
    cached: dict[str, float] = {}
    if cache_path is not None and cache_path.exists():
        with cache_path.open(encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        cached = {str(k): float(v) for k, v in raw.items() if isinstance(v, (int, float))}

    def _cache_key(el: str) -> str:
        return el if pressure_GPa == 0.0 else f"{el}@{pressure_GPa:.2f}GPa"

    refs: dict[str, float] = {}
    updated = False
    for el in elements:
        key = _cache_key(el)
        if key in cached:
            refs[el] = cached[key]
            logger.info("Reference %s @ %.2f GPa: %.6f eV/atom (cached)",
                        el, pressure_GPa, refs[el])
            continue
        logger.info("Computing reference enthalpy for %s @ %.2f GPa …",
                    el, pressure_GPa)
        atoms = _build_element_atoms(el)
        h_per_atom = _relax_and_get_enthalpy_per_atom(
            atoms, calculator, pressure_GPa=pressure_GPa
        )
        refs[el] = h_per_atom
        cached[key] = h_per_atom
        updated = True
        logger.info("Reference %s @ %.2f GPa: %.6f eV/atom", el, pressure_GPa, h_per_atom)

    if updated and cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(cached, f, default_flow_style=False)
        logger.info("Wrote reference cache to %s", cache_path)

    return refs
