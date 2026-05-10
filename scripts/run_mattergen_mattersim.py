"""
MatterGen + MatterSim convex hull pipeline.

Generates candidate crystal structures for a binary chemical system,
relaxes them with MatterSim, computes formation energies, builds the
convex hull, plots it, and saves results (CSV + CIF + PNG).

Usage:
    python scripts/run_mattergen_mattersim.py --element-a Li --element-b O
    python scripts/run_mattergen_mattersim.py --element-a Li --element-b O --n-candidates 4

Requires patched @torch.jit.script decorators in torch_sparse and
torch_runstats (see notebook 08 header for details).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from ase.filters import FrechetCellFilter
from ase.optimize import LBFGS
from pymatgen.core import Composition
from pymatgen.io.ase import AseAtomsAdaptor

from hullgap.calculators import get_calculator
from hullgap.hull import (
    element_fraction,
    formation_energy_per_atom,
    hull_energy_at_x,
    lower_convex_hull_2d,
)
from hullgap.properties.elastic import compute_elastic_properties
from hullgap.properties.phonons import compute_phonons
from hullgap.references import get_elemental_references

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

_GPa_TO_EV_A3 = 1.0 / 160.21766208


# ── CLI ──────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MatterGen + MatterSim hull pipeline")
    p.add_argument("--element-a", default="Li")
    p.add_argument("--element-b", default="O")
    p.add_argument("--n-candidates", type=int, default=32)
    p.add_argument("--guidance", type=float, default=2.0)
    p.add_argument("--e-above-hull-target", type=float, default=0.0)
    p.add_argument("--pressure-gpa", type=float, default=0.0,
                   help="External hydrostatic pressure in GPa (default: 0). "
                        "Hull is built on formation enthalpy H=E+PV at this pressure.")
    p.add_argument("--skip-properties", action="store_true",
                   help="Skip phonon and elastic property calculations")
    p.add_argument("--n-property-candidates", type=int, default=5,
                   help="Number of top candidates to characterize (phonons + elastic)")
    return p.parse_args()


# ── Step 1: MatterGen generation ─────────────────────────────────────


def step1_generate(system: str, n: int, guidance: float,
                   e_above_hull: float, output_dir: Path):
    """Generate candidate structures with MatterGen.

    MatterGen is a diffusion model that denoises over 1000 steps (fixed by
    the D3PM discrete schedule baked into the checkpoint).  With classifier-
    free guidance it runs ~2 forward passes per step through GemNetT on GPU.
    Expect ~9 min for a batch of 32 structures.
    """
    log.info("=== Step 1: MatterGen generation ===")
    log.info("System: %s  |  n_candidates: %d  |  guidance: %.1f",
             system, n, guidance)

    from mattergen.common.utils.data_classes import MatterGenCheckpointInfo
    from mattergen.generator import CrystalGenerator

    checkpoint_info = MatterGenCheckpointInfo.from_hf_hub(
        "chemical_system_energy_above_hull",
    )
    log.info("Checkpoint loaded from HuggingFace")

    generator = CrystalGenerator(
        checkpoint_info=checkpoint_info,
        properties_to_condition_on={
            "chemical_system": system,
            "energy_above_hull": e_above_hull,
        },
        batch_size=n,
        num_batches=1,
        diffusion_guidance_factor=guidance,
    )

    log.info("Starting generation (1000 denoising steps on GPU) ...")
    structures = generator.generate(output_dir=output_dir)
    log.info("Generated %d structures", len(structures))

    for i, s in enumerate(structures[:8]):
        log.info("  [%d] %-10s  %d atoms  vol=%.1f A³",
                 i, s.composition.reduced_formula, s.num_sites, s.volume)
    if len(structures) > 8:
        log.info("  ... (%d more)", len(structures) - 8)
    return structures


# ── Step 2: Elemental reference energies ─────────────────────────────


def step2_references(elements: list[str], calc,
                     cache_path: Path,
                     pressure_GPa: float = 0.0) -> dict[str, float]:
    """Compute or load cached per-atom enthalpies for pure elements."""
    log.info("=== Step 2: Elemental reference enthalpies @ %.2f GPa ===", pressure_GPa)
    refs = get_elemental_references(
        elements=elements,
        calculator=calc,
        cache_path=cache_path,
        pressure_GPa=pressure_GPa,
    )
    for el, h in refs.items():
        log.info("  %s: %.6f eV/atom", el, h)
    return refs


# ── Step 3: Relax candidates with MatterSim ──────────────────────────


def step3_relax(structures, calc,
                pressure_GPa: float = 0.0,
                target_elements: set[str] | None = None) -> tuple[pd.DataFrame, list]:
    """Relax all candidates and return (results DataFrame, relaxed_structures).

    The returned ``relaxed_structures`` list contains post-relaxation pymatgen
    Structures; failed entries fall back to the original geometry so downstream
    indexing stays stable.  Structures with elements outside *target_elements*
    are skipped entirely (no reference energy needed for the foreign element).
    """
    log.info("=== Step 3: MatterSim relaxation @ %.2f GPa ===", pressure_GPa)
    scalar_pressure = pressure_GPa * _GPa_TO_EV_A3
    results = []
    relaxed_structures: list = []
    for i, struct in enumerate(structures):
        if target_elements is not None:
            present = {str(el) for el in struct.composition.elements}
            foreign = present - target_elements
            if foreign:
                log.warning("  [%d] %-10s  skipped — foreign elements: %s",
                            i, struct.composition.reduced_formula, foreign)
                continue

        atoms = AseAtomsAdaptor.get_atoms(struct)
        atoms.calc = calc

        try:
            opt_target = FrechetCellFilter(atoms, scalar_pressure=scalar_pressure)
            opt = LBFGS(opt_target, logfile=None)
            opt.run(fmax=0.02, steps=500)

            e_total = float(atoms.get_potential_energy())
            volume = float(atoms.get_volume())
            n_atoms = len(atoms)
            fmax = float(np.max(np.linalg.norm(atoms.get_forces(), axis=1)))
            enthalpy = e_total + scalar_pressure * volume

            status = "converged" if fmax < 0.02 else "max_steps"
            log.info("  [%d] %-10s  H=%.4f eV/atom  fmax=%.4f  %s",
                     i, struct.composition.reduced_formula,
                     enthalpy / n_atoms, fmax, status)
            results.append({
                "idx": i,
                "formula": struct.composition.reduced_formula,
                "n_atoms": n_atoms,
                "e_total_eV": e_total,
                "e_per_atom_eV": e_total / n_atoms,
                "enthalpy_eV": enthalpy,
                "enthalpy_per_atom_eV": enthalpy / n_atoms,
                "fmax_eV_A": fmax,
                "volume_A3": volume,
                "status": status,
            })
            # Capture the post-relaxation geometry as a pymatgen Structure so
            # the CIF written in step6_save matches the CSV stats.
            relaxed_structures.append(AseAtomsAdaptor.get_structure(atoms))
        except Exception as exc:
            log.error("  [%d] FAILED: %s", i, exc)
            results.append({
                "idx": i,
                "formula": struct.composition.reduced_formula,
                "n_atoms": struct.num_sites,
                "e_total_eV": np.nan,
                "e_per_atom_eV": np.nan,
                "enthalpy_eV": np.nan,
                "enthalpy_per_atom_eV": np.nan,
                "fmax_eV_A": np.nan,
                "volume_A3": np.nan,
                "status": f"failed: {exc}",
            })
            # Keep the index aligned with `structures`; the original geometry
            # is the best fallback even though no relaxation completed.
            relaxed_structures.append(struct)

    df = pd.DataFrame(results)
    n_conv = (df.status == "converged").sum()
    n_max = (df.status == "max_steps").sum()
    n_fail = df.status.str.startswith("failed").sum()
    log.info("Relaxed %d structures  (converged: %d, max_steps: %d, failed: %d)",
             len(df), n_conv, n_max, n_fail)
    return df, relaxed_structures


# ── Step 4: Formation energy + convex hull ───────────────────────────


def step4_hull(df: pd.DataFrame, refs: dict[str, float],
               element_b: str,
               pressure_GPa: float = 0.0) -> tuple[pd.DataFrame, np.ndarray]:
    """Score candidates and return (scored_df, hull_vertices).

    At pressure_GPa > 0 the hull is built on formation enthalpy ΔH/atom
    (using the ``enthalpy_eV`` column); at 0 GPa it falls back to total energy.
    """
    log.info("=== Step 4: Formation enthalpy + convex hull @ %.2f GPa ===", pressure_GPa)

    energy_col = "enthalpy_eV" if pressure_GPa > 0.0 else "e_total_eV"
    df_ok = df[~df[energy_col].isna()].copy()
    if df_ok.empty:
        log.error("No successfully relaxed structures!")
        return df_ok, np.empty((0, 2))

    df_ok["x_B"] = df_ok["formula"].apply(
        lambda f: element_fraction(Composition(f), element_b)
    )

    def _calc_eform(row):
        comp = Composition(row["formula"]) * (
            row["n_atoms"] / Composition(row["formula"]).num_atoms
        )
        return formation_energy_per_atom(row[energy_col], comp, refs)

    df_ok["e_form_eV_atom"] = df_ok.apply(_calc_eform, axis=1)

    anchors = np.array([[0.0, 0.0], [1.0, 0.0]])
    candidate_pts = np.column_stack([df_ok.x_B.values, df_ok.e_form_eV_atom.values])
    hull = lower_convex_hull_2d(np.vstack([anchors, candidate_pts]))

    df_ok["e_above_hull_eV_atom"] = df_ok.apply(
        lambda r: r["e_form_eV_atom"] - hull_energy_at_x(hull, r["x_B"]),
        axis=1,
    )
    df_ok["on_hull"] = df_ok["e_above_hull_eV_atom"] <= 0.025

    n_stable = df_ok.on_hull.sum()
    log.info("Scored %d candidates: %d on/near hull, %d above",
             len(df_ok), n_stable, len(df_ok) - n_stable)
    log.info("\n%s", df_ok.sort_values("e_above_hull_eV_atom").to_string(index=False))
    return df_ok, hull


# ── Step 5: Plot ─────────────────────────────────────────────────────


def step5_plot(df_ok: pd.DataFrame, hull: np.ndarray,
               element_a: str, element_b: str, save_path: Path,
               pressure_GPa: float = 0.0) -> None:
    """Draw the convex hull diagram and save as PNG."""
    log.info("=== Step 5: Plot convex hull ===")

    fig, ax = plt.subplots(figsize=(10, 6))

    above = df_ok[~df_ok.on_hull]
    on = df_ok[df_ok.on_hull]

    hull_sorted = hull[np.argsort(hull[:, 0])]

    # Filled stability region
    fill_x = np.concatenate([[hull_sorted[0, 0]], hull_sorted[:, 0], [hull_sorted[-1, 0]]])
    fill_y = np.concatenate([[0.0], hull_sorted[:, 1], [0.0]])
    ax.fill(fill_x, fill_y, color="#289ff0", alpha=0.08, zorder=1)

    # Hull boundary
    ax.plot(hull_sorted[:, 0], hull_sorted[:, 1], "-", color="#289ff0",
            lw=2.5, zorder=5, label="Convex hull", solid_capstyle="round")

    # Above-hull candidates
    if len(above) > 0:
        ax.scatter(above.x_B, above.e_form_eV_atom, c="#94a3b8", s=55,
                   alpha=0.6, edgecolors="white", linewidths=0.5,
                   label=f"Above hull ({len(above)})", zorder=6)

    # On-hull candidates
    if len(on) > 0:
        ax.scatter(on.x_B, on.e_form_eV_atom, c="#289ff0", s=100,
                   edgecolors="white", linewidths=1.0,
                   label=f"On / near hull ({len(on)})", zorder=7)

    # Elemental endpoints
    ax.scatter([0, 1], [0, 0], marker="D", c="#289ff0", s=80,
               edgecolors="white", linewidths=1.0, zorder=8)
    ax.annotate(element_a, (0, 0), textcoords="offset points",
                xytext=(-14, -16), fontsize=10, fontweight="bold", color="#1e3a5f")
    ax.annotate(element_b, (1, 0), textcoords="offset points",
                xytext=(6, -16), fontsize=10, fontweight="bold", color="#1e3a5f")

    # Label on-hull formulas
    for j, (_, row) in enumerate(on.iterrows()):
        y_off = 10 if j % 2 == 0 else -16
        ax.annotate(
            row["formula"],
            (row["x_B"], row["e_form_eV_atom"]),
            textcoords="offset points", xytext=(6, y_off),
            fontsize=8.5, color="#1e3a5f", fontweight="semibold",
            path_effects=[pe.withStroke(linewidth=2.5, foreground="white")],
        )

    # Drop lines from above-hull points to hull
    for _, row in above.iterrows():
        e_hull = hull_energy_at_x(hull_sorted, row["x_B"])
        ax.plot([row["x_B"], row["x_B"]], [row["e_form_eV_atom"], e_hull],
                ls=":", lw=0.7, color="#94a3b8", alpha=0.5, zorder=2)

    ax.axhline(0, color="#cbd5e1", lw=1.0, ls="--", zorder=3)
    pressure_label = f"  |  P = {pressure_GPa:.1f} GPa" if pressure_GPa > 0 else ""
    y_label = "Formation enthalpy (eV/atom)" if pressure_GPa > 0 else "Formation energy (eV/atom)"
    ax.set_xlabel(f"x({element_b})  \u2014  mole fraction of {element_b}", fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.set_title(
        f"{element_a}\u2013{element_b}  convex hull   (MatterGen + MatterSim){pressure_label}",
        fontsize=14, fontweight="bold",
    )
    ax.set_xlim(-0.05, 1.05)
    y_min = min(df_ok.e_form_eV_atom.min(), hull_sorted[:, 1].min())
    ax.set_ylim(y_min * 1.25, max(0.15, df_ok.e_form_eV_atom.max() * 1.1 + 0.05))
    ax.legend(fontsize=9.5, loc="upper right", framealpha=0.9)
    ax.grid(True, alpha=0.15)

    fig.tight_layout()
    plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Plot saved to %s", save_path)


# ── Step 6: Save results ────────────────────────────────────────────


def step6_save(df_ok: pd.DataFrame, structures, results_csv: Path,
               output_dir: Path, system: str) -> None:
    """Write CSV + relaxed CIF files."""
    log.info("=== Step 6: Save results ===")

    df_ok.to_csv(results_csv, index=False)
    log.info("Results CSV saved to %s", results_csv)

    relaxed_dir = output_dir / "relaxed"
    relaxed_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    for i, struct in enumerate(structures):
        if i in df_ok.idx.values:
            cif_path = (relaxed_dir
                        / f"{system}_{i:03d}_{struct.composition.reduced_formula}.cif")
            struct.to(filename=str(cif_path))
            saved += 1
    log.info("Saved %d relaxed CIF files to %s", saved, relaxed_dir)


# ── Step 7: Phonon stability ──────────────────────────────────────────


def step7_phonons(df_ok: pd.DataFrame, structures, calc,
                  n_top: int = 5) -> pd.DataFrame:
    """Compute phonon properties for the top n_top candidates."""
    log.info("=== Step 7: Phonon stability (top %d candidates) ===", n_top)

    top_idx = df_ok.nsmallest(n_top, "e_above_hull_eV_atom").index

    df_ok["dynamically_stable"] = pd.NA
    df_ok["min_freq_THz"] = np.nan
    df_ok["Cv_300K_J_K_mol"] = np.nan
    df_ok["free_energy_300K_kJ_mol"] = np.nan

    for idx in top_idx:
        row = df_ok.loc[idx]
        struct_idx = int(row["idx"])
        formula = row["formula"]

        log.info("  Phonons for [%d] %s ...", struct_idx, formula)
        atoms = AseAtomsAdaptor.get_atoms(structures[struct_idx])

        try:
            result = compute_phonons(atoms, calc=calc)
            df_ok.at[idx, "dynamically_stable"] = not result["has_imaginary"]
            df_ok.at[idx, "min_freq_THz"] = result["min_frequency_THz"]
            df_ok.at[idx, "Cv_300K_J_K_mol"] = result["heat_capacity_J_K_mol"]
            df_ok.at[idx, "free_energy_300K_kJ_mol"] = result["free_energy_kJ_mol"]

            stability = "STABLE" if not result["has_imaginary"] else "UNSTABLE"
            log.info("    %s  min_freq=%.3f THz  Cv=%.2f J/(K·mol)",
                     stability, result["min_frequency_THz"],
                     result["heat_capacity_J_K_mol"] or 0.0)
        except Exception as exc:
            log.error("    Phonon calculation failed: %s", exc)

    n_stable = df_ok["dynamically_stable"].sum()
    n_computed = df_ok["dynamically_stable"].notna().sum()
    log.info("Phonon results: %d/%d dynamically stable", n_stable, n_computed)
    return df_ok


# ── Step 8: Elastic properties ───────────────────────────────────────


def step8_elastic(df_ok: pd.DataFrame, structures, calc,
                  n_top: int = 5) -> pd.DataFrame:
    """Compute elastic properties for the top n_top candidates."""
    log.info("=== Step 8: Elastic properties (top %d candidates) ===", n_top)

    top_idx = df_ok.nsmallest(n_top, "e_above_hull_eV_atom").index

    df_ok["bulk_modulus_GPa"] = np.nan
    df_ok["shear_modulus_GPa"] = np.nan
    df_ok["youngs_modulus_GPa"] = np.nan
    df_ok["poisson_ratio"] = np.nan

    for idx in top_idx:
        row = df_ok.loc[idx]
        struct_idx = int(row["idx"])
        formula = row["formula"]

        log.info("  Elastic for [%d] %s ...", struct_idx, formula)
        atoms = AseAtomsAdaptor.get_atoms(structures[struct_idx])

        try:
            result = compute_elastic_properties(atoms, calc=calc)
            df_ok.at[idx, "bulk_modulus_GPa"] = result["bulk_modulus_GPa"]
            df_ok.at[idx, "shear_modulus_GPa"] = result["shear_modulus_GPa"]
            df_ok.at[idx, "youngs_modulus_GPa"] = result["youngs_modulus_GPa"]
            df_ok.at[idx, "poisson_ratio"] = result["poisson_ratio"]

            log.info("    K=%.1f GPa  G=%.1f GPa  E=%.1f GPa  ν=%.3f",
                     result["bulk_modulus_GPa"], result["shear_modulus_GPa"],
                     result["youngs_modulus_GPa"], result["poisson_ratio"])
        except Exception as exc:
            log.error("    Elastic calculation failed: %s", exc)

    n_computed = df_ok["bulk_modulus_GPa"].notna().sum()
    log.info("Elastic results computed for %d candidates", n_computed)
    return df_ok


# ── Step 9: Property summary plot ────────────────────────────────────


def step9_property_plot(df_ok: pd.DataFrame, element_a: str,
                        element_b: str, save_path: Path) -> None:
    """Plot property summary for characterized candidates."""
    log.info("=== Step 9: Property summary plot ===")

    has_props = df_ok[df_ok["bulk_modulus_GPa"].notna()].copy()
    if has_props.empty:
        log.warning("No property data to plot")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Bulk modulus vs energy above hull
    ax = axes[0]
    colors = has_props["dynamically_stable"].map(
        {True: "#22c55e", False: "#ef4444"}
    ).fillna("#94a3b8")

    ax.scatter(
        has_props["e_above_hull_eV_atom"] * 1000,
        has_props["bulk_modulus_GPa"],
        c=colors, s=80, edgecolors="white", linewidths=0.8, zorder=5,
    )
    for _, row in has_props.iterrows():
        ax.annotate(
            row["formula"],
            (row["e_above_hull_eV_atom"] * 1000, row["bulk_modulus_GPa"]),
            textcoords="offset points", xytext=(6, 4), fontsize=8,
        )
    ax.set_xlabel("Energy above hull (meV/atom)")
    ax.set_ylabel("Bulk modulus (GPa)")
    ax.set_title("Mechanical stiffness vs stability")
    ax.axvline(25, ls="--", color="#cbd5e1", lw=1)
    ax.grid(True, alpha=0.15)

    # Right: bar chart of moduli for top candidates
    ax = axes[1]
    formulas = has_props["formula"].values
    x = np.arange(len(formulas))
    width = 0.35

    ax.bar(x - width / 2, has_props["bulk_modulus_GPa"], width,
           label="Bulk (K)", color="#3b82f6", alpha=0.8)
    ax.bar(x + width / 2, has_props["shear_modulus_GPa"], width,
           label="Shear (G)", color="#f59e0b", alpha=0.8)

    # Phonon stability markers
    for i, (_, row) in enumerate(has_props.iterrows()):
        if row.get("dynamically_stable") is True:
            ax.text(i, -8, "✓", ha="center", fontsize=12, color="#22c55e",
                    fontweight="bold")
        elif row.get("dynamically_stable") is False:
            ax.text(i, -8, "✗", ha="center", fontsize=12, color="#ef4444",
                    fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(formulas, fontsize=9, rotation=30, ha="right")
    ax.set_ylabel("Modulus (GPa)")
    ax.set_title(f"{element_a}–{element_b} top candidates")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.15, axis="y")

    fig.tight_layout()
    plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Property plot saved to %s", save_path)


# ── main ─────────────────────────────────────────────────────────────


def main():
    args = parse_args()
    system = f"{args.element_a}-{args.element_b}"
    pressure_GPa: float = args.pressure_gpa
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Device: %s  |  torch: %s", device, torch.__version__)
    if device == "cuda":
        log.info("GPU:    %s", torch.cuda.get_device_name(0))
    if pressure_GPa > 0:
        log.info("Pressure: %.2f GPa  (hull built on formation enthalpy H=E+PV)", pressure_GPa)

    # Pressure-aware output names: e.g. "Li-O_10GPa_mattersim_hull.csv"
    p_suffix = f"_{pressure_GPa:.0f}GPa" if pressure_GPa > 0 else ""
    output_dir = PROJECT_ROOT / "data" / "mattergen" / system
    ref_cache = PROJECT_ROOT / "data" / "mattersim_references.yaml"
    results_csv = (PROJECT_ROOT / "data" / "results"
                   / f"{system}{p_suffix}_mattersim_hull.csv")
    plot_path = results_csv.with_suffix(".png")
    output_dir.mkdir(parents=True, exist_ok=True)
    results_csv.parent.mkdir(parents=True, exist_ok=True)

    structures = step1_generate(
        system=system,
        n=args.n_candidates,
        guidance=args.guidance,
        e_above_hull=args.e_above_hull_target,
        output_dir=output_dir,
    )

    calc = get_calculator("mattersim", device=device)

    refs = step2_references(
        elements=[args.element_a, args.element_b],
        calc=calc,
        cache_path=ref_cache,
        pressure_GPa=pressure_GPa,
    )

    df, relaxed_structures = step3_relax(structures, calc,
                                          pressure_GPa=pressure_GPa,
                                          target_elements={args.element_a, args.element_b})

    df_ok, hull = step4_hull(df, refs, args.element_b, pressure_GPa=pressure_GPa)

    if not df_ok.empty:
        step5_plot(df_ok, hull, args.element_a, args.element_b, plot_path,
                   pressure_GPa=pressure_GPa)
        # Use relaxed structures so CIFs on disk match the CSV stats.
        step6_save(df_ok, relaxed_structures, results_csv, output_dir, system)

        if not args.skip_properties:
            n_top = args.n_property_candidates
            df_ok = step7_phonons(df_ok, relaxed_structures, calc, n_top=n_top)
            df_ok = step8_elastic(df_ok, relaxed_structures, calc, n_top=n_top)

            prop_plot_path = results_csv.with_name(
                f"{system}{p_suffix}_properties.png"
            )
            step9_property_plot(df_ok, args.element_a, args.element_b,
                               prop_plot_path)

            # Re-save CSV with property columns
            df_ok.to_csv(results_csv, index=False)
            log.info("Updated results CSV with property data: %s", results_csv)

    log.info("=== Done ===")


if __name__ == "__main__":
    main()
