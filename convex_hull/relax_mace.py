#!/usr/bin/env python3
"""
Generate, relax, and score binary crystal structures using MACE-MP-0.

Three-phase pipeline:
  1. Generate random structures across the A-B composition space.
  2. Relax each structure with MACE-MP-0 (multiprocessing).
  3. Build a convex hull and identify stable compositions.

Usage:
    python relax_mace.py --elements Co Bi
    python relax_mace.py --elements Fe Bi --n-per-comp 80 --pressure 0 --workers 6
    python relax_mace.py --elements Co Bi --skip-generate --structures my_structs/
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import time
import warnings
from math import gcd
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD.*")

# ---------------------------------------------------------------------------
# Composition grid -- element-agnostic ratios (n_A, n_B) per formula unit
# ---------------------------------------------------------------------------
COMPOSITIONS: list[tuple[int, int]] = [
    (1, 0),  # pure A
    (0, 1),  # pure B
    (1, 8), (1, 6), (1, 5), (1, 4), (1, 3), (1, 2), (2, 3),
    (1, 1),
    (3, 2), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1), (8, 1),
]

NFORM_RANGE = {"pure": (1, 8), "binary": (1, 4)}
MAX_ATOMS = 24
ANGLE_RANGE = (50.0, 130.0)
MAX_PLACEMENT_ATTEMPTS = 300

# ---------------------------------------------------------------------------
# Phase 1 -- Structure generation
# ---------------------------------------------------------------------------

def _element_volume(symbol: str) -> float:
    """Estimate ambient volume-per-atom (A^3) from atomic radius."""
    try:
        from pymatgen.core import Element
        el = Element(symbol)
        r = el.atomic_radius
        if r is None:
            r = 1.5
        return (4.0 / 3.0) * np.pi * float(r) ** 3
    except Exception:
        return 15.0


def estimate_vol_range(el_a: str, el_b: str) -> tuple[float, float]:
    """Volume-per-atom range (A^3) for random lattice generation."""
    va, vb = _element_volume(el_a), _element_volume(el_b)
    v_mid = (va + vb) / 2.0
    return (max(4.0, v_mid * 0.5), v_mid * 2.0)


def estimate_min_sep(el_a: str, el_b: str) -> float:
    """Minimum interatomic distance (A) from covalent radii."""
    try:
        from pymatgen.core import Element
        ra = Element(el_a).atomic_radius or 1.3
        rb = Element(el_b).atomic_radius or 1.3
        return float(min(ra, rb)) * 0.85 * 2.0
    except Exception:
        return 1.8


def random_lattice(
    rng: np.random.Generator, n_atoms: int, vol_range: tuple[float, float],
) -> np.ndarray:
    """Return a random 3x3 cell matrix for *n_atoms* atoms."""
    vol = rng.uniform(*vol_range) * n_atoms
    raw = rng.uniform(0.7, 1.4, size=3)
    raw *= (vol / np.prod(raw)) ** (1.0 / 3.0)
    a, b, c = raw

    alpha = np.radians(rng.uniform(*ANGLE_RANGE))
    beta = np.radians(rng.uniform(*ANGLE_RANGE))
    gamma = np.radians(rng.uniform(*ANGLE_RANGE))

    v_a = np.array([a, 0.0, 0.0])
    v_b = np.array([b * np.cos(gamma), b * np.sin(gamma), 0.0])
    cx = c * np.cos(beta)
    cy = c * (np.cos(alpha) - np.cos(beta) * np.cos(gamma)) / np.sin(gamma)
    cz_sq = c * c - cx * cx - cy * cy
    if cz_sq < 0.01:
        return random_lattice(rng, n_atoms, vol_range)
    v_c = np.array([cx, cy, np.sqrt(cz_sq)])

    return np.array([v_a, v_b, v_c])


def place_atoms(
    rng: np.random.Generator, cell: np.ndarray, n_atoms: int, min_sep: float,
) -> np.ndarray | None:
    """Place *n_atoms* in *cell* with minimum separation via rejection."""
    positions: list[np.ndarray] = []
    inv_cell = np.linalg.inv(cell)

    for _ in range(n_atoms):
        for _ in range(MAX_PLACEMENT_ATTEMPTS):
            frac = rng.uniform(0.0, 1.0, size=3)
            cart = frac @ cell
            ok = True
            for prev in positions:
                diff = cart - prev
                frac_diff = diff @ inv_cell
                frac_diff -= np.round(frac_diff)
                mic = frac_diff @ cell
                if np.linalg.norm(mic) < min_sep:
                    ok = False
                    break
            if ok:
                positions.append(cart)
                break
        else:
            return None
    return np.array(positions)


def make_atoms(
    rng: np.random.Generator,
    el_a: str, el_b: str,
    n_a: int, n_b: int,
    vol_range: tuple[float, float],
    min_sep: float,
):
    """Build one random Atoms object for a given composition."""
    from ase import Atoms

    n_atoms = n_a + n_b
    cell = random_lattice(rng, n_atoms, vol_range)
    pos = place_atoms(rng, cell, n_atoms, min_sep)
    if pos is None:
        return None
    symbols = [el_a] * n_a + [el_b] * n_b
    rng.shuffle(symbols)
    return Atoms(symbols=symbols, positions=pos, cell=cell, pbc=True)


def generate_structures(
    el_a: str, el_b: str, n_per_comp: int, out_dir: Path, seed: int = 42,
) -> list[dict]:
    """Generate random structures and return serialised task dicts."""
    from ase.io import write as ase_write

    rng = np.random.default_rng(seed)
    vol_range = estimate_vol_range(el_a, el_b)
    min_sep = estimate_min_sep(el_a, el_b)

    print(f"Generating structures for {el_a}-{el_b}  "
          f"(vol/atom {vol_range[0]:.1f}–{vol_range[1]:.1f} A^3, "
          f"min_sep {min_sep:.2f} A)")

    struct_dir = out_dir / "structures"
    struct_dir.mkdir(parents=True, exist_ok=True)
    tasks: list[dict] = []

    for n_a_fu, n_b_fu in COMPOSITIONS:
        is_pure = n_a_fu == 0 or n_b_fu == 0
        nf_lo, nf_hi = NFORM_RANGE["pure" if is_pure else "binary"]

        batch = []
        while len(batch) < n_per_comp:
            nform = int(rng.integers(nf_lo, nf_hi + 1))
            n_a = n_a_fu * nform
            n_b = n_b_fu * nform
            if n_a + n_b > MAX_ATOMS:
                continue
            atoms = make_atoms(rng, el_a, el_b, n_a, n_b, vol_range, min_sep)
            if atoms is None:
                continue
            batch.append(atoms)

        if is_pure:
            tag = f"{el_a}{n_a_fu}" if n_a_fu else f"{el_b}{n_b_fu}"
        else:
            tag = f"{el_a}{n_a_fu}{el_b}{n_b_fu}"

        fname = struct_dir / f"{tag}.extxyz"
        ase_write(str(fname), batch, format="extxyz")
        print(f"  {tag}: {len(batch)} structures -> {fname.name}")

        for i, atoms in enumerate(batch):
            tasks.append({
                "symbols": atoms.get_chemical_symbols(),
                "positions": atoms.get_positions().tolist(),
                "cell": atoms.get_cell().tolist(),
                "source_file": fname.name,
                "source_index": i,
            })

    print(f"Total: {len(tasks)} structures generated")
    return tasks


# ---------------------------------------------------------------------------
# Phase 2 -- MACE relaxation
# ---------------------------------------------------------------------------

_calc = None
_pressure_ev_a3 = None
_fmax = None
_max_steps = None


def _init_worker(model: str, pressure_gpa: float, fmax: float, max_steps: int):
    """Each worker loads its own MACE calculator to avoid pickling issues."""
    global _calc, _pressure_ev_a3, _fmax, _max_steps

    import torch
    torch.set_num_threads(1)

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    warnings.filterwarnings("ignore")

    from ase import units
    from mace.calculators import mace_mp

    _calc = mace_mp(model=model, device="cpu", default_dtype="float32")
    _pressure_ev_a3 = pressure_gpa * units.GPa
    _fmax = fmax
    _max_steps = max_steps


def reduced_formula(el_a: str, el_b: str, n_a: int, n_b: int) -> str:
    if n_a == 0 and n_b == 0:
        return "?"
    g = gcd(n_a, n_b) if n_a and n_b else max(n_a, n_b)
    ra, rb = n_a // g, n_b // g

    def p(el: str, n: int) -> str:
        return "" if n == 0 else el if n == 1 else f"{el}{n}"

    return p(el_a, ra) + p(el_b, rb)


def _relax_one(task: dict) -> dict | None:
    """Worker function: relax one structure."""
    from ase import Atoms
    from ase.filters import FrechetCellFilter
    from ase.optimize import FIRE

    atoms = Atoms(
        symbols=task["symbols"],
        positions=np.array(task["positions"]),
        cell=np.array(task["cell"]),
        pbc=True,
    )
    el_a, el_b = task["element_a"], task["element_b"]

    try:
        atoms.calc = _calc
        ecf = FrechetCellFilter(atoms, scalar_pressure=_pressure_ev_a3)
        opt = FIRE(ecf, logfile=None)
        opt.run(fmax=_fmax, steps=_max_steps)

        energy_ev = float(atoms.get_potential_energy())
        volume_a3 = float(atoms.get_volume())
        enthalpy_ev = energy_ev + _pressure_ev_a3 * volume_a3

        symbols = atoms.get_chemical_symbols()
        n_a = symbols.count(el_a)
        n_b = symbols.count(el_b)
        n_atoms = len(atoms)

        return {
            "element_A": el_a,
            "element_B": el_b,
            "n_A": n_a,
            "n_B": n_b,
            "n_atoms": n_atoms,
            "formula": reduced_formula(el_a, el_b, n_a, n_b),
            "x_A": n_a / n_atoms if n_atoms > 0 else 0.0,
            "energy_eV": energy_ev,
            "volume_A3": volume_a3,
            "enthalpy_eV": enthalpy_ev,
            "enthalpy_eV_atom": enthalpy_ev / n_atoms,
            "pressure_GPa": task["pressure_gpa"],
            "converged": True,
            "n_steps": opt.nsteps,
            "source_file": task["source_file"],
            "source_index": task["source_index"],
        }
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Phase 3 -- Convex hull
# ---------------------------------------------------------------------------

def _cross_2d(o: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    return float((a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]))


def lower_convex_hull_2d(points: np.ndarray) -> np.ndarray:
    """Lower convex hull of 2D points (x, energy) via monotone chain."""
    if len(points) == 0:
        return points
    uniq = sorted({(float(p[0]), float(p[1])) for p in points})
    pts = np.array(uniq, dtype=float)
    pts = pts[np.argsort(pts[:, 0])]
    lower: list[list[float]] = []
    for p in pts:
        pl = [float(p[0]), float(p[1])]
        while len(lower) >= 2 and _cross_2d(
            np.array(lower[-2]), np.array(lower[-1]), np.array(pl)
        ) <= 1e-12:
            lower.pop()
        lower.append(pl)
    return np.array(lower, dtype=float)


def hull_energy_at_x(hull: np.ndarray, xq: float) -> float:
    """Interpolate the lower hull energy at composition xq."""
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


def compute_hull(
    df: pd.DataFrame, el_a: str, el_b: str, tol: float = 0.025,
) -> pd.DataFrame:
    """Compute formation enthalpies, build hull, label stable phases.

    Parameters
    ----------
    df : DataFrame with columns n_A, n_B, n_atoms, enthalpy_eV, enthalpy_eV_atom
    el_a, el_b : element symbols
    tol : energy above hull tolerance (eV/atom) for "near-hull" label

    Returns
    -------
    DataFrame with added columns:
        ref_H_A_eV_atom, ref_H_B_eV_atom,
        formation_enthalpy_eV_atom, e_above_hull_eV_atom, on_hull
    """
    # Filter out unphysical MACE energies (collapsed/exploded structures)
    median_h = df["enthalpy_eV_atom"].median()
    energy_bound = max(abs(median_h) * 5.0, 50.0)
    physical = df["enthalpy_eV_atom"].between(median_h - energy_bound, median_h + energy_bound)
    n_dropped = (~physical).sum()
    if n_dropped > 0:
        print(f"  Dropped {n_dropped} structures with unphysical energies")
    df = df[physical].copy()

    pure_a = df[df["n_B"] == 0]
    pure_b = df[df["n_A"] == 0]

    if pure_a.empty or pure_b.empty:
        raise ValueError(
            f"Need at least one pure {el_a} and one pure {el_b} structure "
            "to compute formation enthalpies."
        )

    mu_a = pure_a["enthalpy_eV_atom"].min()
    mu_b = pure_b["enthalpy_eV_atom"].min()

    df = df.copy()
    df["ref_H_A_eV_atom"] = mu_a
    df["ref_H_B_eV_atom"] = mu_b

    df["formation_enthalpy_eV_atom"] = (
        df["enthalpy_eV"] - df["n_A"] * mu_a - df["n_B"] * mu_b
    ) / df["n_atoms"]

    # Anchor hull at pure-element endpoints (formation enthalpy = 0 by definition)
    hull_pts = [(0.0, 0.0), (1.0, 0.0)]
    binary = df[(df["n_A"] > 0) & (df["n_B"] > 0)]
    for _, row in binary.iterrows():
        hull_pts.append((1.0 - row["x_A"], row["formation_enthalpy_eV_atom"]))
    hull = lower_convex_hull_2d(np.array(hull_pts, dtype=float))

    df["e_above_hull_eV_atom"] = df.apply(
        lambda row: float(
            row["formation_enthalpy_eV_atom"]
            - hull_energy_at_x(hull, 1.0 - row["x_A"])
        ),
        axis=1,
    )
    df["on_hull"] = df["e_above_hull_eV_atom"] <= tol

    return df


def print_hull_summary(df: pd.DataFrame, el_a: str, el_b: str) -> None:
    """Print a table of stable compositions."""
    stable = df[df["on_hull"]].copy()
    if stable.empty:
        print("\nNo stable compositions found within tolerance.")
        return

    best = (
        stable.sort_values("e_above_hull_eV_atom")
        .groupby("formula", sort=False)
        .first()
        .reset_index()
        .sort_values("x_A")
    )

    print(f"\n{'='*60}")
    print(f" Stable / near-hull compositions for {el_a}-{el_b}")
    print(f"{'='*60}")
    print(f" {'Formula':<14} {'x_'+el_a:<8} {'H_form (eV/at)':<16} {'E_above_hull':<14} {'N_atoms'}")
    print(f" {'-'*14} {'-'*8} {'-'*16} {'-'*14} {'-'*7}")

    for _, row in best.iterrows():
        print(f" {row['formula']:<14} {row['x_A']:<8.3f} "
              f"{row['formation_enthalpy_eV_atom']:<16.4f} "
              f"{row['e_above_hull_eV_atom']:<14.4f} {int(row['n_atoms'])}")

    print(f"{'='*60}")
    print(f" {len(best)} unique stable compositions from {len(stable)} structures")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate, relax, and score binary structures with MACE-MP-0",
    )
    ap.add_argument("--elements", nargs=2, required=True, metavar=("A", "B"),
                    help="two element symbols, e.g. Co Bi")
    ap.add_argument("--n-per-comp", type=int, default=80,
                    help="structures per composition (default: 80)")
    ap.add_argument("--skip-generate", action="store_true",
                    help="skip generation; read pre-existing extxyz from --structures")
    ap.add_argument("--structures", default=None,
                    help="directory of extxyz files (used with --skip-generate)")
    ap.add_argument("--pressure", type=float, default=0.0,
                    help="external pressure in GPa (default: 0)")
    ap.add_argument("--model", default="small",
                    help="MACE-MP-0 model size: small/medium/large (default: small)")
    ap.add_argument("--fmax", type=float, default=0.05,
                    help="force convergence criterion eV/A (default: 0.05)")
    ap.add_argument("--max-steps", type=int, default=100,
                    help="max optimizer steps per structure (default: 100)")
    ap.add_argument("--workers", type=int, default=0,
                    help="number of worker processes (default: cpu_count - 2)")
    ap.add_argument("--hull-tol", type=float, default=0.025,
                    help="energy above hull tolerance in eV/atom (default: 0.025)")
    ap.add_argument("--out", default="results",
                    help="output directory (default: results/)")
    ap.add_argument("--seed", type=int, default=42,
                    help="random seed for structure generation (default: 42)")
    args = ap.parse_args()

    el_a, el_b = args.elements
    system = f"{el_a}-{el_b}"
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    n_workers = args.workers or max(1, os.cpu_count() - 2)

    # ---- Phase 1: Generate or load structures ----
    if args.skip_generate:
        from ase.io import read as ase_read

        struct_dir = Path(args.structures or "structures")
        extxyz_files = sorted(struct_dir.glob("*.extxyz"))
        if not extxyz_files:
            raise SystemExit(f"No .extxyz files found in {struct_dir}")

        tasks: list[dict] = []
        for fpath in extxyz_files:
            structures = ase_read(str(fpath), index=":", format="extxyz")
            for i, atoms in enumerate(structures):
                tasks.append({
                    "symbols": atoms.get_chemical_symbols(),
                    "positions": atoms.get_positions().tolist(),
                    "cell": atoms.get_cell().tolist(),
                    "source_file": fpath.name,
                    "source_index": i,
                })
        print(f"{len(tasks)} structures loaded from {len(extxyz_files)} files.")
    else:
        tasks = generate_structures(el_a, el_b, args.n_per_comp, out_dir, args.seed)

    for t in tasks:
        t["pressure_gpa"] = args.pressure
        t["element_a"] = el_a
        t["element_b"] = el_b

    # ---- Phase 2: MACE relaxation ----
    print(f"\nLaunching {n_workers} workers with MACE-MP-0 ({args.model}) "
          f"at {args.pressure} GPa ...")

    t0 = time.time()
    ctx = mp.get_context("spawn")

    with ctx.Pool(
        processes=n_workers,
        initializer=_init_worker,
        initargs=(args.model, args.pressure, args.fmax, args.max_steps),
    ) as pool:
        all_results: list[dict] = []
        n_fail = 0

        for i, result in enumerate(pool.imap_unordered(_relax_one, tasks), 1):
            if result is not None:
                all_results.append(result)
            else:
                n_fail += 1

            if i % 20 == 0 or i == len(tasks):
                elapsed = time.time() - t0
                speed = i / elapsed
                eta = (len(tasks) - i) / speed if speed > 0 else 0
                print(f"  [{i}/{len(tasks)}]  ok={len(all_results)}  "
                      f"fail={n_fail}  {speed:.1f} struct/s  "
                      f"ETA {eta:.0f}s")

    if not all_results:
        raise SystemExit("All relaxations failed")

    elapsed = time.time() - t0
    print(f"\nRelaxation done: {len(all_results)} ok, {n_fail} failed "
          f"in {elapsed:.1f}s ({len(all_results)/elapsed:.1f} struct/s)")

    # ---- Phase 3: Convex hull ----
    df = pd.DataFrame(all_results)

    try:
        df = compute_hull(df, el_a, el_b, tol=args.hull_tol)
    except ValueError as exc:
        print(f"\nWarning: could not compute hull: {exc}")
        csv_all = out_dir / f"mace_relaxed_{system}_{int(args.pressure)}GPa.csv"
        df.to_csv(csv_all, index=False)
        print(f"Results (no hull): {csv_all}")
        return

    csv_all = out_dir / f"mace_relaxed_{system}_{int(args.pressure)}GPa.csv"
    df.to_csv(csv_all, index=False)
    print(f"\nFull results: {csv_all}")

    stable = df[df["on_hull"]].sort_values("x_A")
    csv_stable = out_dir / f"mace_stable_{system}_{int(args.pressure)}GPa.csv"
    stable.to_csv(csv_stable, index=False)
    print(f"Stable phases: {csv_stable}")

    print_hull_summary(df, el_a, el_b)


if __name__ == "__main__":
    main()
