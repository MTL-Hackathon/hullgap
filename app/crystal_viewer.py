"""Interactive crystal structure viewer and screening dashboard.

Run with:
    streamlit run app/crystal_viewer.py
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from pymatgen.core import Composition, Element, Lattice, Structure
from pymatgen.io.ase import AseAtomsAdaptor

REPO = Path(__file__).resolve().parent.parent
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

RESULTS_DIR = REPO / "data" / "results"
RELAXED_DIR = REPO / "data" / "relaxed"
DFT_DIR = REPO / "dft"

RADII = {"Co": 1.25, "Bi": 1.55}

ELEMENT_COLORS = {
    "Co": "#3366cc", "Bi": "#cc6633", "Ni": "#228b22", "Sb": "#8b008b",
    "Fe": "#b22222", "Mn": "#ff8c00", "Cr": "#4682b4", "Ti": "#708090",
}

ELEMENT_SIZES = {
    "Co": 8, "Bi": 12, "Ni": 8, "Sb": 11, "Fe": 8, "Mn": 8, "Cr": 8, "Ti": 9,
}

# ---------------------------------------------------------------------------
# Structure generation
# ---------------------------------------------------------------------------

def avg_radius(species_list: list[str]) -> float:
    return np.mean([RADII.get(s, 1.4) for s in species_list])


def make_candidates(el_a: str, el_b: str) -> list[tuple[str, Structure]]:
    """Generate candidate structures across the full composition range."""
    r_avg = avg_radius([el_a, el_b])
    candidates = []

    a = 2 * r_avg / np.sqrt(3) * 2
    candidates.append(("CsCl_B2", Structure(
        Lattice.cubic(a), [el_a, el_b],
        [[0, 0, 0], [0.5, 0.5, 0.5]],
    )))

    a = 2 * r_avg * np.sqrt(2)
    candidates.append(("NaCl_B1", Structure(
        Lattice.cubic(a), [el_a] * 4 + [el_b] * 4,
        [[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5],
         [0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.5], [0.5, 0.5, 0.5]],
    )))

    a = 4 * r_avg / np.sqrt(3)
    candidates.append(("ZnS_B3", Structure(
        Lattice.cubic(a), [el_a] * 4 + [el_b] * 4,
        [[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5],
         [0.25, 0.25, 0.25], [0.75, 0.75, 0.25],
         [0.75, 0.25, 0.75], [0.25, 0.75, 0.75]],
    )))

    a_hex = 2 * r_avg
    c_hex = a_hex * 1.63
    candidates.append(("NiAs_B81", Structure(
        Lattice.hexagonal(a_hex, c_hex), [el_a, el_a, el_b, el_b],
        [[0, 0, 0], [0, 0, 0.5],
         [1 / 3, 2 / 3, 0.25], [2 / 3, 1 / 3, 0.75]],
    )))

    a = 2 * r_avg * 2.1
    candidates.append(("FeSi_B20", Structure(
        Lattice.cubic(a), [el_a] * 4 + [el_b] * 4,
        [[0.137, 0.137, 0.137], [0.637, 0.363, 0.863],
         [0.363, 0.863, 0.637], [0.863, 0.637, 0.363],
         [0.845, 0.845, 0.845], [0.345, 0.655, 0.155],
         [0.655, 0.155, 0.345], [0.155, 0.345, 0.655]],
    )))

    a = 2 * r_avg * np.sqrt(2)
    candidates.append(("Cu3Au_L12", Structure(
        Lattice.cubic(a), [el_a] * 3 + [el_b],
        [[0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5], [0, 0, 0]],
    )))

    a_hex = 2 * r_avg * np.sqrt(2)
    c_hex = a_hex * 0.816
    candidates.append(("Ni3Sn_D019", Structure(
        Lattice.hexagonal(a_hex, c_hex),
        [el_a] * 6 + [el_b] * 2,
        [[5 / 6, 2 / 3, 1 / 4], [1 / 6, 1 / 3, 3 / 4],
         [1 / 3, 1 / 6, 1 / 4], [2 / 3, 5 / 6, 3 / 4],
         [1 / 2, 1 / 2, 1 / 4], [1 / 2, 1 / 2, 3 / 4],
         [0, 0, 1 / 4], [0, 0, 3 / 4]],
    )))

    a = 2 * r_avg * np.sqrt(2)
    candidates.append(("Au3Cu_L12", Structure(
        Lattice.cubic(a), [el_a] + [el_b] * 3,
        [[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]],
    )))

    a_hex = 2 * r_avg * np.sqrt(2)
    c_hex = a_hex * 0.816
    candidates.append(("Sn3Ni_D019", Structure(
        Lattice.hexagonal(a_hex, c_hex),
        [el_a] * 2 + [el_b] * 6,
        [[0, 0, 1 / 4], [0, 0, 3 / 4],
         [5 / 6, 2 / 3, 1 / 4], [1 / 6, 1 / 3, 3 / 4],
         [1 / 3, 1 / 6, 1 / 4], [2 / 3, 5 / 6, 3 / 4],
         [1 / 2, 1 / 2, 1 / 4], [1 / 2, 1 / 2, 3 / 4]],
    )))

    a = 2 * r_avg * np.sqrt(2)
    c = a * 2.45
    candidates.append(("MoSi2_C11b", Structure(
        Lattice.tetragonal(a, c), [el_a] * 2 + [el_b] * 4,
        [[0, 0, 0], [0.5, 0.5, 0.5],
         [0, 0, 1 / 3], [0, 0, 2 / 3],
         [0.5, 0.5, 1 / 3 + 0.5], [0.5, 0.5, 2 / 3 - 0.5]],
    )))

    a = 2 * r_avg * 2
    candidates.append(("CaF2_C1", Structure(
        Lattice.cubic(a), [el_a] * 4 + [el_b] * 8,
        [[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5],
         [0.25, 0.25, 0.25], [0.75, 0.75, 0.25],
         [0.75, 0.25, 0.75], [0.25, 0.75, 0.75],
         [0.25, 0.25, 0.75], [0.75, 0.75, 0.75],
         [0.75, 0.25, 0.25], [0.25, 0.75, 0.25]],
    )))

    candidates.append(("CaF2_C1_inv", Structure(
        Lattice.cubic(a), [el_a] * 8 + [el_b] * 4,
        [[0.25, 0.25, 0.25], [0.75, 0.75, 0.25],
         [0.75, 0.25, 0.75], [0.25, 0.75, 0.75],
         [0.25, 0.25, 0.75], [0.75, 0.75, 0.75],
         [0.75, 0.25, 0.25], [0.25, 0.75, 0.25],
         [0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]],
    )))

    a_hex = 2 * r_avg
    c_hex = a_hex * 0.81
    candidates.append(("CaCu5_D2d", Structure(
        Lattice.hexagonal(a_hex, c_hex),
        [el_a] * 5 + [el_b],
        [[1 / 3, 2 / 3, 0], [2 / 3, 1 / 3, 0],
         [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0.5],
         [0, 0, 0]],
    )))

    candidates.append(("CaCu5_D2d_inv", Structure(
        Lattice.hexagonal(a_hex, c_hex),
        [el_a] + [el_b] * 5,
        [[0, 0, 0],
         [1 / 3, 2 / 3, 0], [2 / 3, 1 / 3, 0],
         [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0.5]],
    )))

    return candidates


# ---------------------------------------------------------------------------
# 3D visualisation helpers
# ---------------------------------------------------------------------------

def _parallelepiped_edges(lattice_matrix: np.ndarray) -> list[np.ndarray]:
    """Return the 12 edge segments of a parallelepiped defined by a 3x3 matrix."""
    corners_frac = np.array([
        [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
        [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1],
    ], dtype=float)
    corners = corners_frac @ lattice_matrix
    edges = []
    for i, j in combinations(range(8), 2):
        diff = corners_frac[i] - corners_frac[j]
        if np.count_nonzero(diff) == 1:
            edges.append(np.array([corners[i], corners[j]]))
    return edges


def build_structure_figure(
    struct: Structure,
    name: str,
    supercell: tuple[int, int, int] = (2, 2, 2),
) -> go.Figure:
    """Build an interactive 3D plotly figure for a crystal structure."""
    unit_lattice = struct.lattice.matrix.copy()
    unit_params = struct.lattice.parameters

    if any(s > 1 for s in supercell):
        struct = struct.copy()
        struct.make_supercell(supercell)

    super_lattice = struct.lattice.matrix
    cart_coords = struct.cart_coords
    species = [str(s.specie) for s in struct]

    fig = go.Figure()

    for sym in sorted(set(species)):
        mask = np.array([s == sym for s in species])
        pts = cart_coords[mask]
        color = ELEMENT_COLORS.get(sym, "#888888")
        size = ELEMENT_SIZES.get(sym, 8)
        fig.add_trace(go.Scatter3d(
            x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
            mode="markers",
            marker=dict(size=size, color=color, opacity=0.9,
                        line=dict(width=0.5, color="black")),
            name=sym,
            hovertemplate=(
                f"<b>{sym}</b><br>"
                "x: %{x:.3f} Å<br>y: %{y:.3f} Å<br>z: %{z:.3f} Å"
                "<extra></extra>"
            ),
        ))

    for edge in _parallelepiped_edges(super_lattice):
        fig.add_trace(go.Scatter3d(
            x=edge[:, 0], y=edge[:, 1], z=edge[:, 2],
            mode="lines", line=dict(color="rgba(160,160,160,0.4)", width=2),
            showlegend=False, hoverinfo="skip",
        ))

    uc_legend_shown = False
    for edge in _parallelepiped_edges(unit_lattice):
        fig.add_trace(go.Scatter3d(
            x=edge[:, 0], y=edge[:, 1], z=edge[:, 2],
            mode="lines", line=dict(color="#e03030", width=5),
            name="unit cell" if not uc_legend_shown else "",
            showlegend=not uc_legend_shown, legendgroup="unit_cell",
            hoverinfo="skip",
        ))
        uc_legend_shown = True

    uc_corners_frac = np.array([
        [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
        [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1],
    ], dtype=float)
    uc_corners = uc_corners_frac @ unit_lattice
    fig.add_trace(go.Scatter3d(
        x=uc_corners[:, 0], y=uc_corners[:, 1], z=uc_corners[:, 2],
        mode="markers", marker=dict(size=3, color="#e03030", symbol="diamond"),
        showlegend=False, legendgroup="unit_cell",
        hovertemplate="unit cell corner<extra></extra>",
    ))

    all_pts = np.vstack([cart_coords, *_parallelepiped_edges(super_lattice)])
    pad = 0.5
    ranges = {
        ax: [all_pts[:, i].min() - pad, all_pts[:, i].max() + pad]
        for i, ax in enumerate(["x", "y", "z"])
    }

    formula = struct.composition.reduced_formula
    title = (
        f"{name} — {formula}  |  unit cell: "
        f"({unit_params[0]:.2f}, {unit_params[1]:.2f}, {unit_params[2]:.2f} Å) "
        f"({unit_params[3]:.1f}°, {unit_params[4]:.1f}°, {unit_params[5]:.1f}°)"
        f"  |  supercell: {supercell[0]}×{supercell[1]}×{supercell[2]}"
    )

    fig.update_layout(
        title=dict(text=title, font=dict(size=13)),
        scene=dict(
            xaxis=dict(title="x (Å)", range=ranges["x"]),
            yaxis=dict(title="y (Å)", range=ranges["y"]),
            zaxis=dict(title="z (Å)", range=ranges["z"]),
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        height=600,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01,
                    font=dict(size=12)),
    )
    return fig


# ---------------------------------------------------------------------------
# MLIP relaxation helpers
# ---------------------------------------------------------------------------

def _get_element_b(system: str) -> str:
    return system.split("-")[1]


def _x_b_fraction(comp: Composition, el_b: str) -> float:
    """Mole fraction of element B in a binary composition."""
    try:
        return float(comp.get_atomic_fraction(Element(el_b)))
    except Exception:
        return 0.0


def _relax_single(
    struct: Structure, calc, fmax: float = 0.05, max_steps: int = 300,
) -> tuple[Structure, float, float, int, str]:
    """Relax one structure and return (relaxed_struct, E_total, fmax, steps, status)."""
    from ase.filters import FrechetCellFilter
    from ase.optimize import LBFGS

    atoms = AseAtomsAdaptor.get_atoms(struct)
    atoms.calc = calc
    opt = LBFGS(FrechetCellFilter(atoms), logfile=None)
    converged = opt.run(fmax=fmax, steps=max_steps)
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    fmax_val = float(np.max(np.linalg.norm(forces, axis=1)))
    relaxed = AseAtomsAdaptor.get_structure(atoms)
    status = "converged" if converged else "max_steps_reached"
    return relaxed, float(energy), fmax_val, opt.nsteps, status


@st.cache_resource(show_spinner="Loading MLIP model...")
def _load_calculator(model: str = "chgnet"):
    from hullgap.calculators import get_calculator
    return get_calculator(model)


def run_mlip_screening(
    candidates: list[tuple[str, Structure]],
    el_a: str,
    el_b: str,
    progress_bar,
    model: str = "chgnet",
) -> pd.DataFrame:
    """Relax all candidates with an MLIP and compute formation energies."""
    calc = _load_calculator(model)

    # Elemental references
    elem_structs = {
        "Co": Structure(Lattice.hexagonal(2.507, 4.069), ["Co", "Co"],
                        [[1/3, 2/3, 1/4], [2/3, 1/3, 3/4]]),
        "Bi": Structure(Lattice.rhombohedral(4.746, 57.23), ["Bi", "Bi"],
                        [[0.234, 0.234, 0.234], [0.766, 0.766, 0.766]]),
    }

    refs: dict[str, float] = {}
    for el in [el_a, el_b]:
        if el in elem_structs:
            _, e_tot, _, _, _ = _relax_single(elem_structs[el], calc, fmax=0.02, max_steps=500)
            natoms = len(elem_structs[el])
            refs[el] = e_tot / natoms
        else:
            st.warning(f"No elemental reference for {el}; formation energies will be absent.")

    rows = []
    n = len(candidates)
    for i, (name, struct) in enumerate(candidates):
        progress_bar.progress((i + 1) / n, text=f"Relaxing {name} ({i+1}/{n})")
        try:
            relaxed, e_tot, fmax_val, steps, status = _relax_single(struct, calc)
            n_atoms = len(relaxed)
            comp = relaxed.composition
            formula = comp.reduced_formula
            x_b = _x_b_fraction(comp, el_b)

            fe = np.nan
            if len(refs) == 2:
                ref_sum = sum(refs[str(el)] * amt for el, amt in comp.get_el_amt_dict().items())
                fe = (e_tot - ref_sum) / n_atoms

            rows.append({
                "model": model,
                "prototype": name,
                "formula": formula,
                "n_atoms": n_atoms,
                f"x_{el_b}": x_b,
                "energy_total_eV": e_tot,
                "energy_per_atom_eV": e_tot / n_atoms,
                "fmax_eV_A": fmax_val,
                "formation_energy_eV_atom": fe,
                "status": status,
                "n_steps": steps,
            })
        except Exception as exc:
            rows.append({
                "model": model,
                "prototype": name,
                "formula": struct.composition.reduced_formula,
                "n_atoms": len(struct),
                f"x_{el_b}": _x_b_fraction(struct.composition, el_b),
                "energy_total_eV": np.nan,
                "energy_per_atom_eV": np.nan,
                "fmax_eV_A": np.nan,
                "formation_energy_eV_atom": np.nan,
                "status": "failed",
                "n_steps": 0,
                "error": str(exc),
            })

    df = pd.DataFrame(rows)

    if "formation_energy_eV_atom" in df.columns:
        valid = df["formation_energy_eV_atom"].dropna()
        if len(valid) >= 2:
            from hullgap.dft.dft_hull import lower_convex_hull_2d, hull_energy_at_x
            x_col = f"x_{el_b}"
            pts = np.column_stack([df.loc[valid.index, x_col].values, valid.values])
            hull_pts = np.array([[0.0, 0.0], [1.0, 0.0]] + pts.tolist())
            hull = lower_convex_hull_2d(hull_pts)
            e_above = []
            for _, row in df.iterrows():
                fe_val = row["formation_energy_eV_atom"]
                if pd.isna(fe_val):
                    e_above.append(np.nan)
                else:
                    e_hull = hull_energy_at_x(hull, row[x_col])
                    e_above.append(fe_val - e_hull)
            df["e_above_hull_eV_atom"] = e_above
            df["stability_label"] = df["e_above_hull_eV_atom"].apply(
                lambda v: "on_hull" if not pd.isna(v) and abs(v) < 1e-6 else "unstable"
            )

    df = df.sort_values("formation_energy_eV_atom", ascending=True, na_position="last")
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Hull plot
# ---------------------------------------------------------------------------

MODEL_STYLES = {
    "chgnet": {"color": "#3366cc", "symbol": "circle"},
    "mace":   {"color": "#22aa44", "symbol": "diamond"},
}


def build_hull_figure(
    mlip_df: pd.DataFrame,
    el_b: str,
    dft_df: pd.DataFrame | None = None,
) -> go.Figure:
    """2D convex hull plot: x_B vs formation energy per atom."""
    from hullgap.dft.dft_hull import lower_convex_hull_2d

    fig = go.Figure()
    x_col = f"x_{el_b}"

    has_model_col = "model" in mlip_df.columns
    models = mlip_df["model"].unique().tolist() if has_model_col else ["chgnet"]

    for model_name in models:
        style = MODEL_STYLES.get(model_name, {"color": "#888888", "symbol": "circle"})
        subset = mlip_df[mlip_df["model"] == model_name] if has_model_col else mlip_df
        valid = subset.dropna(subset=["formation_energy_eV_atom"])
        if valid.empty:
            continue

        fig.add_trace(go.Scatter(
            x=valid[x_col], y=valid["formation_energy_eV_atom"],
            mode="markers+text",
            text=valid["prototype"],
            textposition="top center",
            textfont=dict(size=9),
            marker=dict(size=10, color=style["color"], symbol=style["symbol"],
                        line=dict(width=1, color="black")),
            name=f"{model_name} candidates",
            legendgroup=model_name,
            hovertemplate=(
                f"<b>%{{text}}</b> ({model_name})<br>"
                f"x_{el_b}: %{{x:.3f}}<br>"
                "ΔE: %{y:.4f} eV/atom<extra></extra>"
            ),
        ))

        hull_pts = np.array([[0.0, 0.0], [1.0, 0.0]]
                            + list(zip(valid[x_col], valid["formation_energy_eV_atom"])))
        hull = lower_convex_hull_2d(hull_pts)
        fig.add_trace(go.Scatter(
            x=hull[:, 0], y=hull[:, 1],
            mode="lines",
            line=dict(color=style["color"], width=2, dash="dash"),
            name=f"{model_name} hull",
            legendgroup=model_name,
            hoverinfo="skip",
        ))

    if dft_df is not None and not dft_df.empty:
        dft_valid = dft_df.dropna(subset=["formation_energy_eV_atom"])
        if not dft_valid.empty:
            fig.add_trace(go.Scatter(
                x=dft_valid["x_Bi"], y=dft_valid["formation_energy_eV_atom"],
                mode="markers+text",
                text=dft_valid["candidate_id"],
                textposition="bottom center",
                textfont=dict(size=9, color="#cc3333"),
                marker=dict(size=14, color="#cc3333", symbol="star",
                            line=dict(width=1.5, color="black")),
                name="DFT validated",
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    "x_Bi: %{x:.3f}<br>"
                    "ΔE (DFT): %{y:.4f} eV/atom<extra></extra>"
                ),
            ))

    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 0],
        mode="lines",
        line=dict(color="gray", width=1, dash="dot"),
        showlegend=False, hoverinfo="skip",
    ))

    fig.update_layout(
        title="Formation energy convex hull",
        xaxis_title=f"x_{el_b} (mole fraction)",
        yaxis_title="Formation energy (eV/atom)",
        height=500,
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
        hovermode="closest",
    )
    return fig


# ---------------------------------------------------------------------------
# DFT helpers
# ---------------------------------------------------------------------------

def _run_qe_for_candidate(
    prototype: str,
    struct: Structure,
    system: str,
    pseudo_dir: Path,
) -> dict:
    """Generate QE inputs, run pw.x, parse output, and return result dict."""
    from hullgap.dft.make_qe_inputs import write_run_folder
    from hullgap.dft.parse_qe_outputs import parse_single_run_dir

    run_base = DFT_DIR / "runs" / system
    run_base.mkdir(parents=True, exist_ok=True)
    run_dir = run_base / prototype

    cif_path = run_dir / "input.cif"
    run_dir.mkdir(parents=True, exist_ok=True)
    struct.to(filename=str(cif_path))

    write_run_folder(
        structure=struct,
        run_dir=run_dir,
        candidate_id=prototype,
        formula=struct.composition.reduced_formula,
        source_file=cif_path,
        preset="coarse_relax",
        pseudo_dir=pseudo_dir,
    )

    pw_x = "pw.x"
    has_mpirun = subprocess.run(
        ["which", "mpirun"], capture_output=True
    ).returncode == 0
    has_pw = subprocess.run(
        ["which", "pw.x"], capture_output=True
    ).returncode == 0

    if not has_pw:
        return {
            "candidate_id": prototype,
            "formula": struct.composition.reduced_formula,
            "status": "failed",
            "converged": False,
            "error_message": "pw.x not found on PATH. Install Quantum ESPRESSO.",
        }

    if has_mpirun:
        cmd = ["mpirun", "-np", "4", pw_x, "-in", "pw.in"]
    else:
        cmd = [pw_x, "-in", "pw.in"]

    proc = subprocess.run(cmd, cwd=str(run_dir), capture_output=True, text=True,
                          timeout=3600)

    pw_out = run_dir / "pw.out"
    pw_out.write_text(proc.stdout + proc.stderr, encoding="utf-8")

    return parse_single_run_dir(run_dir)


def score_dft_results(
    dft_results: list[dict],
    system: str,
    el_b: str,
) -> pd.DataFrame:
    """Compute formation energies for DFT results using reference energies."""
    from hullgap.dft.dft_hull import (
        formation_energy_per_atom, load_elemental_references, x_bi_fraction,
    )

    ref_path = DFT_DIR / "reference_energies.yaml"
    if not ref_path.exists():
        st.warning("No DFT reference energies found at dft/reference_energies.yaml")
        return pd.DataFrame(dft_results)

    refs = load_elemental_references(ref_path)

    rows = []
    for r in dft_results:
        row = dict(r)
        formula = str(r.get("formula", ""))
        e_tot = r.get("dft_energy_total_eV")

        if formula and e_tot is not None and r.get("converged", False):
            try:
                comp = Composition(formula)
                fe = formation_energy_per_atom(e_tot, comp, refs)
                row["formation_energy_eV_atom"] = fe
                row["x_Bi"] = x_bi_fraction(comp)
            except Exception:
                row["formation_energy_eV_atom"] = np.nan
                row["x_Bi"] = np.nan
        else:
            row["formation_energy_eV_atom"] = np.nan
            row["x_Bi"] = np.nan
        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_sweep_data(system: str) -> pd.DataFrame | None:
    csv_path = RESULTS_DIR / f"stoichiometry_sweep_{system}_chgnet.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return None


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(
        page_title="HullGap Crystal Viewer",
        page_icon="🔬",
        layout="wide",
    )
    st.title("HullGap — Crystal Structure Dashboard")

    with st.sidebar:
        st.header("System")
        el_a = st.text_input("Element A", value="Co")
        el_b = st.text_input("Element B", value="Bi")
        system = f"{el_a}-{el_b}"

        st.divider()
        candidates = make_candidates(el_a, el_b)
        candidate_names = [name for name, _ in candidates]
        candidate_map = dict(candidates)

        selected = st.selectbox("Select prototype", candidate_names, index=0)

        st.divider()
        sc_a = st.slider("Supercell a", 1, 4, 2)
        sc_b = st.slider("Supercell b", 1, 4, 2)
        sc_c = st.slider("Supercell c", 1, 4, 2)

    tab_viz, tab_screen, tab_dft = st.tabs([
        "Structure Viewer",
        "MLIP Screening",
        "DFT Validation",
    ])

    # ---- Tab 1: Structure Viewer ----
    with tab_viz:
        struct = candidate_map[selected]
        col_viz, col_info = st.columns([3, 1])

        with col_viz:
            fig = build_structure_figure(struct, selected, supercell=(sc_a, sc_b, sc_c))
            st.plotly_chart(fig, use_container_width=True)

        with col_info:
            st.subheader("Structure info")
            cp = struct.lattice.parameters
            st.markdown(f"**Formula:** {struct.composition.reduced_formula}")
            st.markdown(f"**Atoms:** {len(struct)}")
            st.markdown(f"**Lattice:**")
            st.markdown(f"  a = {cp[0]:.3f} Å, b = {cp[1]:.3f} Å, c = {cp[2]:.3f} Å")
            st.markdown(f"  α = {cp[3]:.1f}°, β = {cp[4]:.1f}°, γ = {cp[5]:.1f}°")
            st.markdown(f"**Volume:** {struct.lattice.volume:.2f} ų")

            mlip_df = st.session_state.get("mlip_results")
            if mlip_df is not None and selected in mlip_df["prototype"].values:
                row = mlip_df[mlip_df["prototype"] == selected].iloc[0]
                st.divider()
                st.subheader("MLIP results")
                st.markdown(f"**E/atom:** {row['energy_per_atom_eV']:.4f} eV")
                fe = row.get("formation_energy_eV_atom")
                if not pd.isna(fe):
                    st.markdown(f"**Formation E:** {fe:.4f} eV/atom")
                hull_dist = row.get("e_above_hull_eV_atom")
                label = row.get("stability_label", "")
                if hull_dist is not None and not pd.isna(hull_dist):
                    color = "green" if label == "on_hull" else "red"
                    st.markdown(
                        f"**E above hull:** "
                        f":{color}[{hull_dist:.4f} eV/atom ({label})]"
                    )

    # ---- Tab 2: MLIP Screening ----
    with tab_screen:
        st.subheader("MLIP screening — relax & rank all candidates")

        col_models, col_btn = st.columns([2, 1])
        with col_models:
            selected_models = st.multiselect(
                "MLIP models",
                ["chgnet", "mace"],
                default=["chgnet", "mace"],
                key="mlip_models",
            )
        with col_btn:
            st.markdown("")
            run_mlip = st.button(
                f"Run {' + '.join(selected_models) if selected_models else 'MLIP'} on all candidates",
                type="primary",
                use_container_width=True,
                disabled=len(selected_models) == 0,
            )

        if run_mlip:
            all_dfs = []
            for model_name in selected_models:
                progress = st.progress(0, text=f"Loading {model_name}...")
                result_df = run_mlip_screening(
                    candidates, el_a, el_b, progress, model=model_name,
                )
                all_dfs.append(result_df)
                csv_path = RESULTS_DIR / f"stoichiometry_sweep_{system}_{model_name}.csv"
                RESULTS_DIR.mkdir(parents=True, exist_ok=True)
                result_df.to_csv(csv_path, index=False)
                progress.empty()

            combined = pd.concat(all_dfs, ignore_index=True)
            combined = combined.sort_values(
                "formation_energy_eV_atom", ascending=True, na_position="last",
            ).reset_index(drop=True)
            st.session_state["mlip_results"] = combined
            st.success(
                f"Relaxed {len(candidates)} candidates × "
                f"{len(selected_models)} models. Results saved."
            )

        mlip_df = st.session_state.get("mlip_results")
        if mlip_df is not None:
            st.markdown("#### Candidates ranked by formation energy")

            display_cols = [
                "model", "prototype", "formula", "n_atoms", f"x_{el_b}",
                "energy_per_atom_eV", "formation_energy_eV_atom",
                "e_above_hull_eV_atom", "stability_label", "status", "n_steps",
            ]
            available = [c for c in display_cols if c in mlip_df.columns]

            col_table, col_struct = st.columns([1, 1])

            with col_table:
                st.dataframe(mlip_df[available], use_container_width=True, hide_index=True)

            with col_struct:
                st.markdown("#### Structure preview")
                screen_proto = st.selectbox(
                    "Select candidate to visualize",
                    mlip_df["prototype"].tolist(),
                    index=0,
                    key="screen_proto_select",
                )
                if screen_proto in candidate_map:
                    struct_fig = build_structure_figure(
                        candidate_map[screen_proto], screen_proto,
                        supercell=(sc_a, sc_b, sc_c),
                    )
                    st.plotly_chart(struct_fig, use_container_width=True)
                else:
                    st.warning(f"Prototype {screen_proto} not found in generated candidates.")

            st.divider()
            st.markdown("#### Convex hull")
            dft_df = st.session_state.get("dft_results")
            fig = build_hull_figure(mlip_df, el_b, dft_df=dft_df)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info(
                "No MLIP results yet. Click **Run MLIP on all candidates** to "
                "relax structures with CHGNet and rank by formation energy."
            )

    # ---- Tab 3: DFT Validation ----
    with tab_dft:
        st.subheader("DFT validation — Quantum ESPRESSO")

        mlip_df = st.session_state.get("mlip_results")
        if mlip_df is None:
            st.info("Run MLIP screening first to select candidates for DFT.")
        else:
            st.markdown(
                "Select candidates from the MLIP-ranked list below, "
                "then click **Run DFT** to validate with Quantum ESPRESSO."
            )

            dft_candidates = mlip_df[["prototype", "formula", "formation_energy_eV_atom",
                                       "e_above_hull_eV_atom", "stability_label"]].copy()
            dft_candidates = dft_candidates.dropna(subset=["formation_energy_eV_atom"])
            dft_candidates.insert(0, "select", False)

            edited = st.data_editor(
                dft_candidates,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "select": st.column_config.CheckboxColumn("Select", default=False),
                },
                disabled=["prototype", "formula", "formation_energy_eV_atom",
                           "e_above_hull_eV_atom", "stability_label"],
            )

            selected_rows = edited[edited["select"] == True]  # noqa: E712

            pseudo_dir_str = st.text_input(
                "Pseudopotential directory",
                value=str(Path.home() / "software" / "pseudopotentials" / "pbe"),
            )
            pseudo_dir = Path(pseudo_dir_str)

            col_dft_btn, col_dft_info = st.columns([1, 3])
            with col_dft_btn:
                run_dft = st.button(
                    f"Run DFT on {len(selected_rows)} selected",
                    type="primary",
                    disabled=len(selected_rows) == 0,
                    use_container_width=True,
                )

            if run_dft and len(selected_rows) > 0:
                dft_results = []
                progress = st.progress(0, text="Starting DFT calculations...")
                n_sel = len(selected_rows)

                for i, (_, row) in enumerate(selected_rows.iterrows()):
                    proto = row["prototype"]
                    progress.progress(
                        (i + 1) / n_sel,
                        text=f"Running QE for {proto} ({i+1}/{n_sel})",
                    )
                    struct = candidate_map[proto]
                    result = _run_qe_for_candidate(proto, struct, system, pseudo_dir)
                    dft_results.append(result)

                progress.empty()

                dft_scored = score_dft_results(dft_results, system, el_b)
                st.session_state["dft_results"] = dft_scored

                n_ok = dft_scored["converged"].sum() if "converged" in dft_scored.columns else 0
                st.success(f"DFT complete: {n_ok}/{n_sel} converged.")

            dft_df = st.session_state.get("dft_results")
            if dft_df is not None and not dft_df.empty:
                st.divider()
                st.markdown("#### DFT results")

                dft_show = [
                    "candidate_id", "formula", "status", "converged",
                    "dft_energy_per_atom_eV", "formation_energy_eV_atom",
                    "x_Bi",
                ]
                avail = [c for c in dft_show if c in dft_df.columns]
                st.dataframe(dft_df[avail], use_container_width=True, hide_index=True)

                st.divider()
                st.markdown("#### Hull plot (MLIP + DFT)")
                fig = build_hull_figure(mlip_df, el_b, dft_df=dft_df)
                st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
