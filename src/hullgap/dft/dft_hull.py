"""
Formation energies and convex-hull distance for a few DFT totals after MLIP.

Elemental reference energies (per atom) define the formation-energy zero at
pure Co and pure Bi corners. A 2D lower convex hull in (x_Bi, E_form/atom)
is computed with numpy. DFT energies come from Quantum ESPRESSO pw.x (PBE/PAW);
reference energies must use the same pseudopotential and cutoff setup.

This remains a hackathon prioritization workflow, not a claim of absolute
thermodynamic stability.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from pymatgen.core import Composition, Element

logger = logging.getLogger(__name__)


def parse_system_string(system: str) -> list[str]:
    """Parse 'Co-Bi' or 'Co–Bi' into element symbols."""
    s = system.replace("–", "-").replace("—", "-").strip()
    return [p.strip() for p in s.split("-") if p.strip()]


def load_elemental_references(path: Path) -> dict[str, float]:
    """Load YAML mapping element symbol -> reference energy per atom (eV)."""
    with path.open(encoding="utf-8") as handle:
        data: dict[str, Any] = yaml.safe_load(handle) or {}
    out: dict[str, float] = {}
    for key, val in data.items():
        if isinstance(key, str) and (key.startswith("#") or key.startswith("_")):
            continue
        if isinstance(val, (int, float)):
            out[str(key)] = float(val)
    return out


def formation_energy_per_atom(E_total: float, composition: Composition, refs: dict[str, float]) -> float:
    """
    Formation energy per atom:
    (E_total - sum_i n_i * mu_i) / N_atoms
    """
    ref_sum = 0.0
    for el, amt in composition.get_el_amt_dict().items():
        sym = str(el)
        if sym not in refs:
            raise KeyError(f"Missing reference energy for element {sym} in YAML.")
        ref_sum += float(refs[sym]) * float(amt)
    n = composition.num_atoms
    return (E_total - ref_sum) / n


def x_bi_fraction(composition: Composition) -> float:
    """Mole fraction of Bi atoms (Co–Bi MVP); 0 if Bi absent."""
    try:
        return float(composition.get_atomic_fraction(Element("Bi")))
    except Exception:  # noqa: BLE001
        return 0.0


def _cross_2d(o: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    return float((a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]))


def lower_convex_hull_2d(points: np.ndarray) -> np.ndarray:
    """
    Lower convex hull of 2D points (x, energy), x sorted ascending.

    Monotone chain; removes upper-facing turns.
    """
    if len(points) == 0:
        return points
    uniq = sorted({(float(p[0]), float(p[1])) for p in points})
    pts = np.array(uniq, dtype=float)
    pts = pts[np.argsort(pts[:, 0])]
    lower: list[list[float]] = []
    for p in pts:
        p_list = [float(p[0]), float(p[1])]
        while len(lower) >= 2 and _cross_2d(np.array(lower[-2]), np.array(lower[-1]), np.array(p_list)) <= 1e-12:
            lower.pop()
        lower.append(p_list)
    return np.array(lower, dtype=float)


def hull_energy_at_x(hull: np.ndarray, xq: float) -> float:
    """Interpolate the lower hull energy at composition xq (piecewise linear)."""
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


def label_from_e_above(e_above: float | None, converged: bool, parse_ok: bool) -> str:
    if not parse_ok:
        return "failed_dft"
    if not converged or e_above is None or e_above != e_above:
        return "dft_unstable"
    if e_above <= 0.05:
        return "dft_validated_near_hull"
    if e_above <= 0.10:
        return "dft_validated_metastable"
    return "dft_unstable"


def score_dft_candidates(
    dft_energies: pd.DataFrame,
    system: str,
    elemental_refs: dict[str, float],
) -> pd.DataFrame:
    """
    Join DFT totals with reference energies; compute formation E and e_above_hull.

    Binary MVP: hull uses (0, 0) and (1, 0) formation-energy anchors at pure
    elements plus all converged compounds.
    """
    elements = parse_system_string(system)
    for el in elements:
        if el not in elemental_refs:
            raise KeyError(f"Element {el} from system {system!r} missing in elemental reference YAML.")
    df = dft_energies.copy()

    hull_points: list[tuple[float, float]] = [(0.0, 0.0), (1.0, 0.0)]

    for _, row in df.iterrows():
        status = str(row.get("status", "")).lower()
        converged = bool(row.get("converged", False))
        if status == "failed" or not converged:
            continue
        formula = str(row.get("formula", "")).strip()
        if not formula:
            continue
        try:
            E_tot = float(row["dft_energy_total_eV"])
        except (TypeError, ValueError, KeyError):
            continue
        if E_tot != E_tot:
            continue
        try:
            comp = Composition(formula)
        except Exception:  # noqa: BLE001
            continue
        try:
            e_form = formation_energy_per_atom(E_tot, comp, elemental_refs)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Skip hull point %s: %s", formula, exc)
            continue
        xb = x_bi_fraction(comp)
        hull_points.append((xb, e_form))

    hull_arr = lower_convex_hull_2d(np.array(hull_points, dtype=float))

    rows_out: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        cid = row.get("candidate_id", "")
        formula = str(row.get("formula", "")).strip()
        status = str(row.get("status", "")).lower()
        converged = bool(row.get("converged", False))

        parse_ok = status != "failed" and bool(formula)
        try:
            E_tot = float(row.get("dft_energy_total_eV", float("nan")))
        except (TypeError, ValueError):
            E_tot = float("nan")
        if E_tot != E_tot:
            parse_ok = False

        if not parse_ok or not formula:
            rows_out.append(
                {
                    "candidate_id": cid,
                    "formula": formula,
                    "x_Bi": "",
                    "dft_energy_per_atom_eV": row.get("dft_energy_per_atom_eV", ""),
                    "formation_energy_eV_atom": "",
                    "dft_e_above_hull_eV_atom": "",
                    "validation_status": "failed_dft",
                }
            )
            continue

        try:
            comp = Composition(formula)
        except Exception:  # noqa: BLE001
            rows_out.append(
                {
                    "candidate_id": cid,
                    "formula": formula,
                    "x_Bi": "",
                    "dft_energy_per_atom_eV": row.get("dft_energy_per_atom_eV", ""),
                    "formation_energy_eV_atom": "",
                    "dft_e_above_hull_eV_atom": "",
                    "validation_status": "failed_dft",
                }
            )
            continue

        try:
            e_form = formation_energy_per_atom(E_tot, comp, elemental_refs)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Formation energy failed for %s: %s", cid, exc)
            rows_out.append(
                {
                    "candidate_id": cid,
                    "formula": formula,
                    "x_Bi": x_bi_fraction(comp),
                    "dft_energy_per_atom_eV": row.get("dft_energy_per_atom_eV", ""),
                    "formation_energy_eV_atom": "",
                    "dft_e_above_hull_eV_atom": "",
                    "validation_status": "failed_dft",
                }
            )
            continue

        e_per_atom = E_tot / comp.num_atoms
        xb = x_bi_fraction(comp)
        e_hull = hull_energy_at_x(hull_arr, xb)
        e_above: float | None = None
        if e_hull == e_hull:
            e_above = float(e_form - e_hull)

        val = label_from_e_above(e_above, converged, parse_ok=True)

        rows_out.append(
            {
                "candidate_id": cid,
                "formula": formula,
                "x_Bi": xb,
                "dft_energy_per_atom_eV": e_per_atom,
                "formation_energy_eV_atom": e_form,
                "dft_e_above_hull_eV_atom": e_above if e_above is not None else "",
                "validation_status": val,
            }
        )

    return pd.DataFrame(rows_out)


def write_hull_scores(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    logger.info("Wrote DFT hull scores to %s", out_path)
