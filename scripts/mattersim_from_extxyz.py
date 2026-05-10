"""
MatterSim hull pipeline — post-MatterGen entry point.

Picks up after MatterGen has already run by reading an existing
generated_crystals.extxyz file and running steps 2-9:

  references → relax → hull → plot → save CIFs → phonons → elastic

Usage:
    python scripts/mattersim_from_extxyz.py --element-a Bi --element-b Co
    python scripts/mattersim_from_extxyz.py --element-a Bi --element-b Co \\
        --extxyz data/mattergen/Bi-Co/generated_crystals.extxyz \\
        --pressure-gpa 10 --skip-properties

Structures containing elements outside the target system are silently
skipped — no reference energy is needed for the foreign element.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch
from ase.io import read as ase_read
from pymatgen.io.ase import AseAtomsAdaptor

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).parent))

from run_mattergen_mattersim import (  # noqa: E402
    step2_references,
    step3_relax,
    step4_hull,
    step5_plot,
    step6_save,
    step7_phonons,
    step8_elastic,
    step9_property_plot,
)

from hullgap.calculators import get_calculator  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="MatterSim hull pipeline starting from an existing extxyz"
    )
    p.add_argument("--element-a", required=True)
    p.add_argument("--element-b", required=True)
    p.add_argument(
        "--extxyz",
        default=None,
        help="Path to generated_crystals.extxyz. "
             "Defaults to data/mattergen/{A}-{B}/generated_crystals.extxyz",
    )
    p.add_argument("--pressure-gpa", type=float, default=0.0,
                   help="External hydrostatic pressure in GPa (default: 0)")
    p.add_argument("--skip-properties", action="store_true",
                   help="Skip phonon and elastic property calculations")
    p.add_argument("--n-property-candidates", type=int, default=5)
    return p.parse_args()


def load_structures(extxyz_path: Path, target_elements: set[str]) -> list:
    """Read extxyz, convert to pymatgen, drop off-spec structures."""
    log.info("Loading structures from %s", extxyz_path)
    atoms_list = ase_read(str(extxyz_path), index=":")
    log.info("Read %d structures", len(atoms_list))

    structures = []
    skipped = 0
    for i, atoms in enumerate(atoms_list):
        present = set(atoms.get_chemical_symbols())
        foreign = present - target_elements
        if foreign:
            log.warning("  [%d] %-12s  skipped — foreign elements: %s",
                        i, atoms.get_chemical_formula(), foreign)
            skipped += 1
            continue
        structures.append((i, AseAtomsAdaptor.get_structure(atoms)))

    log.info("Keeping %d structures (%d skipped)", len(structures), skipped)

    # Re-index so downstream steps get a plain list with original indices
    # preserved in a wrapper that step3_relax can enumerate.
    # We return a list of pymatgen Structure objects; the original index is
    # lost but step3 assigns its own sequential idx.
    return [s for _, s in structures]


def main() -> None:
    args = parse_args()
    system = f"{args.element_a}-{args.element_b}"
    pressure_GPa: float = args.pressure_gpa
    target_elements = {args.element_a, args.element_b}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("System: %s  |  device: %s  |  torch: %s",
             system, device, torch.__version__)
    if device == "cuda":
        log.info("GPU: %s", torch.cuda.get_device_name(0))
    if pressure_GPa > 0:
        log.info("Pressure: %.2f GPa  (hull on formation enthalpy H=E+PV)", pressure_GPa)

    extxyz_path = Path(args.extxyz) if args.extxyz else (
        PROJECT_ROOT / "data" / "mattergen" / system / "generated_crystals.extxyz"
    )
    if not extxyz_path.exists():
        log.error("extxyz not found: %s", extxyz_path)
        sys.exit(1)

    p_suffix = f"_{pressure_GPa:.0f}GPa" if pressure_GPa > 0 else ""
    output_dir  = PROJECT_ROOT / "data" / "mattergen" / system
    ref_cache   = PROJECT_ROOT / "data" / "mattersim_references.yaml"
    results_csv = (PROJECT_ROOT / "data" / "results"
                   / f"{system}{p_suffix}_mattersim_hull.csv")
    plot_path   = results_csv.with_suffix(".png")
    results_csv.parent.mkdir(parents=True, exist_ok=True)

    structures = load_structures(extxyz_path, target_elements)
    if not structures:
        log.error("No valid structures after filtering — aborting.")
        sys.exit(1)

    calc = get_calculator("mattersim", device=device)

    refs = step2_references(
        elements=[args.element_a, args.element_b],
        calc=calc,
        cache_path=ref_cache,
        pressure_GPa=pressure_GPa,
    )

    df, relaxed_structures = step3_relax(structures, calc,
                                          pressure_GPa=pressure_GPa,
                                          target_elements=target_elements)

    df_ok, hull = step4_hull(df, refs, args.element_b,
                             pressure_GPa=pressure_GPa)

    if df_ok.empty:
        log.error("No candidates survived hull scoring — aborting.")
        sys.exit(1)

    step5_plot(df_ok, hull, args.element_a, args.element_b, plot_path,
               pressure_GPa=pressure_GPa)
    step6_save(df_ok, relaxed_structures, results_csv, output_dir, system)

    if not args.skip_properties:
        n_top = args.n_property_candidates
        df_ok = step7_phonons(df_ok, relaxed_structures, calc, n_top=n_top)
        df_ok = step8_elastic(df_ok, relaxed_structures, calc, n_top=n_top)

        prop_plot_path = results_csv.with_name(f"{system}{p_suffix}_properties.png")
        step9_property_plot(df_ok, args.element_a, args.element_b, prop_plot_path)

        df_ok.to_csv(results_csv, index=False)
        log.info("Updated results CSV with property data: %s", results_csv)

    log.info("=== Done ===")


if __name__ == "__main__":
    main()
