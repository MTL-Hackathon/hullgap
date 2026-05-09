#!/usr/bin/env python3
"""
Generate random crystal structures for a binary convex hull search.

Mimics AIRSS: random lattices, random fractional coordinates, minimum
separation constraints.  Writes structures as extxyz files.

Usage:
    python scripts/generate_random_structures.py --system Co-Bi --n-per-comp 80
    python scripts/generate_random_structures.py --system Co-Bi --n-per-comp 20 --out data/candidates/Co-Bi
"""

from __future__ import annotations

import argparse
import itertools
import os
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import write

RNG = np.random.default_rng(42)

# Volume-per-atom ranges (A^3) at ~50 GPa -- compressed relative to ambient.
VOL_PER_ATOM_RANGE = (6.0, 16.0)
MIN_SEP = 1.8  # minimum interatomic distance in Angstrom
MAX_PLACEMENT_ATTEMPTS = 300
ANGLE_RANGE = (50.0, 130.0)


COMPOSITIONS: list[tuple[int, int]] = [
    # (n_Co, n_Bi) per formula unit
    (1, 0),  # pure Co
    (0, 1),  # pure Bi
    (1, 8), (1, 6), (1, 5), (1, 4), (1, 3), (1, 2), (2, 3),
    (1, 1),
    (3, 2), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1), (8, 1),
]

NFORM_RANGE = {
    "pure": (1, 8),
    "binary": (1, 4),
}


def random_lattice(n_atoms: int) -> np.ndarray:
    """Return a 3x3 cell matrix for *n_atoms* atoms with random shape."""
    vol = RNG.uniform(*VOL_PER_ATOM_RANGE) * n_atoms
    # random lengths with aspect-ratio constraint
    raw = RNG.uniform(0.7, 1.4, size=3)
    raw *= (vol / np.prod(raw)) ** (1.0 / 3.0)
    a, b, c = raw

    alpha = np.radians(RNG.uniform(*ANGLE_RANGE))
    beta = np.radians(RNG.uniform(*ANGLE_RANGE))
    gamma = np.radians(RNG.uniform(*ANGLE_RANGE))

    # Triclinic cell vectors
    v_a = np.array([a, 0.0, 0.0])
    v_b = np.array([b * np.cos(gamma), b * np.sin(gamma), 0.0])
    cx = c * np.cos(beta)
    cy = c * (np.cos(alpha) - np.cos(beta) * np.cos(gamma)) / np.sin(gamma)
    cz_sq = c * c - cx * cx - cy * cy
    if cz_sq < 0.01:
        return random_lattice(n_atoms)
    cz = np.sqrt(cz_sq)
    v_c = np.array([cx, cy, cz])

    return np.array([v_a, v_b, v_c])


def place_atoms(cell: np.ndarray, n_atoms: int) -> np.ndarray | None:
    """Place *n_atoms* in *cell* with minimum separation via rejection."""
    positions: list[np.ndarray] = []
    inv_cell = np.linalg.inv(cell)

    for _ in range(n_atoms):
        for _ in range(MAX_PLACEMENT_ATTEMPTS):
            frac = RNG.uniform(0.0, 1.0, size=3)
            cart = frac @ cell
            ok = True
            for prev in positions:
                diff = cart - prev
                # minimum-image in fractional coords
                frac_diff = diff @ inv_cell
                frac_diff -= np.round(frac_diff)
                mic = frac_diff @ cell
                if np.linalg.norm(mic) < MIN_SEP:
                    ok = False
                    break
            if ok:
                positions.append(cart)
                break
        else:
            return None
    return np.array(positions)


def make_atoms_from_symbols(symbols: list[str]) -> Atoms | None:
    """Build one random Atoms object for the given symbol list."""
    n_atoms = len(symbols)
    cell = random_lattice(n_atoms)
    pos = place_atoms(cell, n_atoms)
    if pos is None:
        return None
    atoms = Atoms(symbols=symbols, positions=pos, cell=cell, pbc=True)
    return atoms


def generate_all(
    el_a: str, el_b: str, compositions: list[tuple[int, int]],
    n_per_comp: int, out_dir: Path,
) -> int:
    """Generate random structures for every composition and write extxyz."""
    out_dir.mkdir(parents=True, exist_ok=True)
    total = 0

    for n_a_fu, n_b_fu in compositions:
        is_pure = n_a_fu == 0 or n_b_fu == 0
        nf_lo, nf_hi = NFORM_RANGE["pure" if is_pure else "binary"]

        generated = 0
        batch: list[Atoms] = []

        while generated < n_per_comp:
            nform = int(RNG.integers(nf_lo, nf_hi + 1))
            n_a = n_a_fu * nform
            n_b = n_b_fu * nform
            n_atoms = n_a + n_b
            if n_atoms > 24:
                continue

            symbols = [el_a] * n_a + [el_b] * n_b
            RNG.shuffle(symbols)
            atoms = make_atoms_from_symbols(symbols)
            if atoms is None:
                continue

            atoms.info[f"n_{el_a}"] = n_a
            atoms.info[f"n_{el_b}"] = n_b
            atoms.info["composition"] = Atoms(symbols).get_chemical_formula(mode="metal")
            batch.append(atoms)
            generated += 1

        def _tag(el, n):
            return "" if n == 0 else el if n == 1 else f"{el}{n}"
        tag = _tag(el_a, n_a_fu) + _tag(el_b, n_b_fu)

        fname = out_dir / f"{tag}.extxyz"
        write(str(fname), batch, format="extxyz")
        total += len(batch)
        print(f"  {tag}: {len(batch)} structures -> {fname.name}")

    return total


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate random binary crystal structures")
    ap.add_argument("--system", type=str, default="Co-Bi",
                    help="binary system, e.g. Co-Bi (default: Co-Bi)")
    ap.add_argument("--n-per-comp", type=int, default=80,
                    help="structures per composition (default 80)")
    ap.add_argument("--out", type=str, default=None,
                    help="output directory (default: data/candidates/<SYSTEM>)")
    args = ap.parse_args()

    elements = [e.strip() for e in args.system.replace("–", "-").split("-")]
    if len(elements) != 2:
        raise SystemExit(f"Expected binary system like Co-Bi, got {args.system!r}")
    el_a, el_b = elements

    out = Path(args.out) if args.out else Path(f"data/candidates/{args.system}")
    print(f"Generating {args.n_per_comp} random structures per composition for {el_a}-{el_b} ...")
    n = generate_all(el_a, el_b, COMPOSITIONS, args.n_per_comp, out)
    print(f"Total: {n} structures written to {out}/")


if __name__ == "__main__":
    main()
