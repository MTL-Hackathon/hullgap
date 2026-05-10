"""Generalized binary convex-hull utilities.

Works for any A-B binary system (not hardcoded to Co-Bi).  Provides the
same core functions as ``hullgap.dft.dft_hull`` but parameterized on an
arbitrary element_B.
"""

from __future__ import annotations

import numpy as np
from pymatgen.core import Composition, Element


def parse_system_string(system: str) -> list[str]:
    """Parse ``'Li-O'`` or ``'Li\\u2013O'`` into element symbols."""
    s = system.replace("\u2013", "-").replace("\u2014", "-").strip()
    return [p.strip() for p in s.split("-") if p.strip()]


def element_fraction(composition: Composition, element: str) -> float:
    """Mole fraction of *element* in *composition*; 0 if absent."""
    try:
        return float(composition.get_atomic_fraction(Element(element)))
    except Exception:  # noqa: BLE001
        return 0.0


def formation_energy_per_atom(
    E_total: float,
    composition: Composition,
    refs: dict[str, float],
) -> float:
    """Formation energy per atom: ``(E_total - sum n_i * mu_i) / N``."""
    ref_sum = 0.0
    for el, amt in composition.get_el_amt_dict().items():
        sym = str(el)
        if sym not in refs:
            raise KeyError(f"Missing reference energy for element {sym}.")
        ref_sum += refs[sym] * float(amt)
    return (E_total - ref_sum) / composition.num_atoms


def _cross_2d(o: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    return float(
        (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    )


def lower_convex_hull_2d(points: np.ndarray) -> np.ndarray:
    """Lower convex hull of 2-D ``(x, energy)`` points.

    Uses a monotone-chain algorithm, keeping only downward-facing turns.
    """
    if len(points) == 0:
        return points
    uniq = sorted({(float(p[0]), float(p[1])) for p in points})
    pts = np.array(uniq, dtype=float)
    pts = pts[np.argsort(pts[:, 0])]
    lower: list[list[float]] = []
    for p in pts:
        p_list = [float(p[0]), float(p[1])]
        while (
            len(lower) >= 2
            and _cross_2d(
                np.array(lower[-2]), np.array(lower[-1]), np.array(p_list)
            )
            <= 1e-12
        ):
            lower.pop()
        lower.append(p_list)
    return np.array(lower, dtype=float)


def hull_energy_at_x(hull: np.ndarray, xq: float) -> float:
    """Piecewise-linear interpolation of the lower hull at *xq*."""
    if hull is None or len(hull) == 0:
        return float("nan")
    hull = hull[np.argsort(hull[:, 0])]
    xs, ys = hull[:, 0], hull[:, 1]
    if xq <= xs[0]:
        return float(ys[0])
    if xq >= xs[-1]:
        return float(ys[-1])
    i = int(np.searchsorted(xs, xq, side="right") - 1)
    i = max(0, min(i, len(xs) - 2))
    x0, x1 = float(xs[i]), float(xs[i + 1])
    y0, y1 = float(ys[i]), float(ys[i + 1])
    if abs(x1 - x0) < 1e-14:
        return min(y0, y1)
    return y0 + (y1 - y0) * (xq - x0) / (x1 - x0)
