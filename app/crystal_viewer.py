"""Interactive crystal structure viewer for HullGap candidates.

Run with:
    streamlit run app/crystal_viewer.py
"""

from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from pymatgen.core import Composition, Lattice, Structure

REPO = Path(__file__).resolve().parent.parent
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

RESULTS_DIR = REPO / "data" / "results"
RELAXED_DIR = REPO / "data" / "relaxed"

RADII = {"Co": 1.25, "Bi": 1.55}

ELEMENT_COLORS = {
    "Co": "#3366cc",
    "Bi": "#cc6633",
    "Ni": "#228b22",
    "Sb": "#8b008b",
    "Fe": "#b22222",
    "Mn": "#ff8c00",
    "Cr": "#4682b4",
    "Ti": "#708090",
}

ELEMENT_SIZES = {
    "Co": 8, "Bi": 12, "Ni": 8, "Sb": 11, "Fe": 8, "Mn": 8, "Cr": 8, "Ti": 9,
}


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
    """Build an interactive 3D plotly figure for a crystal structure.

    Draws the full supercell with a light outer box and a bold colored
    unit-cell box anchored at the origin corner so both are clearly
    distinguishable.
    """
    unit_lattice = struct.lattice.matrix.copy()
    unit_params = struct.lattice.parameters

    if any(s > 1 for s in supercell):
        struct = struct.copy()
        struct.make_supercell(supercell)

    super_lattice = struct.lattice.matrix
    cart_coords = struct.cart_coords
    species = [str(s.specie) for s in struct]

    fig = go.Figure()

    # --- Atoms ---
    for sym in sorted(set(species)):
        mask = np.array([s == sym for s in species])
        pts = cart_coords[mask]
        color = ELEMENT_COLORS.get(sym, "#888888")
        size = ELEMENT_SIZES.get(sym, 8)
        fig.add_trace(go.Scatter3d(
            x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
            mode="markers",
            marker=dict(
                size=size,
                color=color,
                opacity=0.9,
                line=dict(width=0.5, color="black"),
            ),
            name=sym,
            hovertemplate=(
                f"<b>{sym}</b><br>"
                "x: %{x:.3f} Å<br>y: %{y:.3f} Å<br>z: %{z:.3f} Å"
                "<extra></extra>"
            ),
        ))

    # --- Supercell outer box (light grey, thin) ---
    super_edges = _parallelepiped_edges(super_lattice)
    for edge in super_edges:
        fig.add_trace(go.Scatter3d(
            x=edge[:, 0], y=edge[:, 1], z=edge[:, 2],
            mode="lines",
            line=dict(color="rgba(160,160,160,0.4)", width=2),
            showlegend=False,
            hoverinfo="skip",
        ))

    # --- Unit cell box at origin (bold, colored) ---
    uc_edges = _parallelepiped_edges(unit_lattice)
    uc_legend_shown = False
    for edge in uc_edges:
        fig.add_trace(go.Scatter3d(
            x=edge[:, 0], y=edge[:, 1], z=edge[:, 2],
            mode="lines",
            line=dict(color="#e03030", width=5),
            name="unit cell" if not uc_legend_shown else "",
            showlegend=not uc_legend_shown,
            legendgroup="unit_cell",
            hoverinfo="skip",
        ))
        uc_legend_shown = True

    # --- Unit cell corner spheres for extra visibility ---
    uc_corners_frac = np.array([
        [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
        [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1],
    ], dtype=float)
    uc_corners = uc_corners_frac @ unit_lattice
    fig.add_trace(go.Scatter3d(
        x=uc_corners[:, 0], y=uc_corners[:, 1], z=uc_corners[:, 2],
        mode="markers",
        marker=dict(size=3, color="#e03030", symbol="diamond"),
        showlegend=False,
        legendgroup="unit_cell",
        hovertemplate="unit cell corner<extra></extra>",
    ))

    all_pts = np.vstack([cart_coords, *super_edges])
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
        legend=dict(
            yanchor="top", y=0.99, xanchor="left", x=0.01,
            font=dict(size=12),
        ),
    )
    return fig


def load_sweep_data(system: str) -> pd.DataFrame | None:
    """Load stoichiometry sweep CSV if it exists."""
    csv_path = RESULTS_DIR / f"stoichiometry_sweep_{system}_chgnet.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return None


def main() -> None:
    st.set_page_config(
        page_title="HullGap Crystal Viewer",
        page_icon="🔬",
        layout="wide",
    )
    st.title("Crystal Structure Viewer")
    st.caption("Interactive 3D visualization of candidate crystal structures")

    with st.sidebar:
        st.header("Configuration")
        el_a = st.text_input("Element A", value="Co")
        el_b = st.text_input("Element B", value="Bi")
        system = f"{el_a}-{el_b}"

        st.divider()

        candidates = make_candidates(el_a, el_b)
        candidate_names = [name for name, _ in candidates]

        sweep_df = load_sweep_data(system)

        selected = st.selectbox(
            "Select prototype",
            candidate_names,
            index=0,
        )

        st.divider()
        sc_a = st.slider("Supercell a", 1, 4, 2)
        sc_b = st.slider("Supercell b", 1, 4, 2)
        sc_c = st.slider("Supercell c", 1, 4, 2)

    struct = dict(candidates)[selected]

    col_viz, col_info = st.columns([3, 1])

    with col_viz:
        fig = build_structure_figure(struct, selected, supercell=(sc_a, sc_b, sc_c))
        st.plotly_chart(fig, use_container_width=True)

    with col_info:
        st.subheader("Structure info")
        cp = struct.lattice.parameters
        st.markdown(f"**Formula:** {struct.composition.reduced_formula}")
        st.markdown(f"**Atoms:** {len(struct)}")
        st.markdown(f"**Space group:** P 1 (prototype)")
        st.markdown(f"**Lattice:**")
        st.markdown(f"  a = {cp[0]:.3f} Å")
        st.markdown(f"  b = {cp[1]:.3f} Å")
        st.markdown(f"  c = {cp[2]:.3f} Å")
        st.markdown(f"  α = {cp[3]:.1f}°, β = {cp[4]:.1f}°, γ = {cp[5]:.1f}°")
        st.markdown(f"**Volume:** {struct.lattice.volume:.2f} ų")

        if sweep_df is not None and selected in sweep_df["prototype"].values:
            row = sweep_df[sweep_df["prototype"] == selected].iloc[0]
            st.divider()
            st.subheader("MLIP results")
            st.markdown(f"**E/atom:** {row['energy_per_atom_eV']:.4f} eV")
            st.markdown(f"**Formation E:** {row['formation_energy_eV_atom']:.4f} eV/atom")
            hull_dist = row.get("e_above_hull_eV_atom", None)
            label = row.get("stability_label", "")
            if hull_dist is not None:
                color = "green" if label == "on_hull" else "red"
                st.markdown(
                    f"**E above hull:** "
                    f":{color}[{hull_dist:.4f} eV/atom ({label})]"
                )
            st.markdown(f"**Status:** {row['status']}")
            st.markdown(f"**Relax steps:** {int(row['n_steps'])}")

    if sweep_df is not None:
        st.divider()
        st.subheader("All candidates — MLIP screening summary")
        display_cols = [
            "prototype", "formula", "n_atoms", "x_Bi",
            "energy_per_atom_eV", "formation_energy_eV_atom",
            "e_above_hull_eV_atom", "stability_label", "status",
        ]
        available_cols = [c for c in display_cols if c in sweep_df.columns]
        styled = sweep_df[available_cols].style.apply(
            lambda row: [
                "background-color: #d4edda" if row.get("stability_label") == "on_hull"
                else "" for _ in row
            ],
            axis=1,
        )
        st.dataframe(styled, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
