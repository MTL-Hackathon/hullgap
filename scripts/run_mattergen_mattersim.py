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
from hullgap.references import get_elemental_references

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ── CLI ──────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MatterGen + MatterSim hull pipeline")
    p.add_argument("--element-a", default="Li")
    p.add_argument("--element-b", default="O")
    p.add_argument("--n-candidates", type=int, default=32)
    p.add_argument("--guidance", type=float, default=2.0)
    p.add_argument("--e-above-hull-target", type=float, default=0.0)
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
                     cache_path: Path) -> dict[str, float]:
    """Compute or load cached per-atom energies for pure elements."""
    log.info("=== Step 2: Elemental reference energies ===")
    refs = get_elemental_references(
        elements=elements,
        calculator=calc,
        cache_path=cache_path,
    )
    for el, e in refs.items():
        log.info("  %s: %.6f eV/atom", el, e)
    return refs


# ── Step 3: Relax candidates with MatterSim ──────────────────────────


def step3_relax(structures, calc) -> pd.DataFrame:
    """Relax all candidates and return a DataFrame of results."""
    log.info("=== Step 3: MatterSim relaxation ===")
    results = []
    for i, struct in enumerate(structures):
        atoms = AseAtomsAdaptor.get_atoms(struct)
        atoms.calc = calc

        try:
            opt_target = FrechetCellFilter(atoms)
            opt = LBFGS(opt_target, logfile=None)
            opt.run(fmax=0.02, steps=500)

            e_total = float(atoms.get_potential_energy())
            n_atoms = len(atoms)
            fmax = float(np.max(np.linalg.norm(atoms.get_forces(), axis=1)))

            status = "converged" if fmax < 0.02 else "max_steps"
            log.info("  [%d] %-10s  E=%.4f eV/atom  fmax=%.4f  %s",
                     i, struct.composition.reduced_formula,
                     e_total / n_atoms, fmax, status)
            results.append({
                "idx": i,
                "formula": struct.composition.reduced_formula,
                "n_atoms": n_atoms,
                "e_total_eV": e_total,
                "e_per_atom_eV": e_total / n_atoms,
                "fmax_eV_A": fmax,
                "volume_A3": atoms.get_volume(),
                "status": status,
            })
        except Exception as exc:
            log.error("  [%d] FAILED: %s", i, exc)
            results.append({
                "idx": i,
                "formula": struct.composition.reduced_formula,
                "n_atoms": struct.num_sites,
                "e_total_eV": np.nan,
                "e_per_atom_eV": np.nan,
                "fmax_eV_A": np.nan,
                "volume_A3": np.nan,
                "status": f"failed: {exc}",
            })

    df = pd.DataFrame(results)
    n_conv = (df.status == "converged").sum()
    n_max = (df.status == "max_steps").sum()
    n_fail = df.status.str.startswith("failed").sum()
    log.info("Relaxed %d structures  (converged: %d, max_steps: %d, failed: %d)",
             len(df), n_conv, n_max, n_fail)
    return df


# ── Step 4: Formation energy + convex hull ───────────────────────────


def step4_hull(df: pd.DataFrame, refs: dict[str, float],
               element_b: str) -> tuple[pd.DataFrame, np.ndarray]:
    """Score candidates and return (scored_df, hull_vertices)."""
    log.info("=== Step 4: Formation energy + convex hull ===")

    df_ok = df[~df.e_total_eV.isna()].copy()
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
        return formation_energy_per_atom(row["e_total_eV"], comp, refs)

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
               element_a: str, element_b: str, save_path: Path) -> None:
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
    ax.set_xlabel(f"x({element_b})  \u2014  mole fraction of {element_b}", fontsize=12)
    ax.set_ylabel("Formation energy (eV/atom)", fontsize=12)
    ax.set_title(f"{element_a}\u2013{element_b}  convex hull   (MatterGen + MatterSim)",
                 fontsize=14, fontweight="bold")
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


# ── main ─────────────────────────────────────────────────────────────


def main():
    args = parse_args()
    system = f"{args.element_a}-{args.element_b}"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Device: %s  |  torch: %s", device, torch.__version__)
    if device == "cuda":
        log.info("GPU:    %s", torch.cuda.get_device_name(0))

    output_dir = PROJECT_ROOT / "data" / "mattergen" / system
    ref_cache = PROJECT_ROOT / "data" / "mattersim_references.yaml"
    results_csv = PROJECT_ROOT / "data" / "results" / f"{system}_mattersim_hull.csv"
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
    )

    df = step3_relax(structures, calc)

    df_ok, hull = step4_hull(df, refs, args.element_b)

    if not df_ok.empty:
        step5_plot(df_ok, hull, args.element_a, args.element_b, plot_path)
        step6_save(df_ok, structures, results_csv, output_dir, system)

    log.info("=== Done ===")


if __name__ == "__main__":
    main()
