"""
Generate Quantum ESPRESSO pw.x input files for MLIP-shortlisted structures.

Co-rich binaries can be magnetic; nspin=2 and starting_magnetization are
included for physically reasonable first relaxations. Pseudopotentials
(UPF files) must be available at the path given by pseudo_dir.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from pymatgen.core import Structure
from pymatgen.io.pwscf import PWInput
from tqdm import tqdm

logger = logging.getLogger(__name__)

PSEUDO_DIR_DEFAULT = Path("/home/hhoechter/software/pseudopotentials/pbe")

_DEFAULT_MAG = 0.1
_MAGMOM_BY_ELEMENT: dict[str, float] = {
    "Co": 0.5,
    "Fe": 0.7,
    "Ni": 0.3,
    "Mn": 0.8,
}

_PSEUDO_BY_ELEMENT: dict[str, str] = {
    "Co": "Co.pbe-spn-kjpaw_psl.0.3.1.UPF",
    "Bi": "Bi.pbe-dn-kjpaw_psl.1.0.0.UPF",
}

RY_TO_EV = 13.605693009


def _kpoints_grid(structure: Structure, kppa: int = 300) -> tuple[int, int, int]:
    """Compute k-mesh dimensions from k-points-per-atom density."""
    lengths = structure.lattice.abc
    n_atoms = len(structure)
    target = max(1, kppa / n_atoms)
    vol_recip = structure.lattice.reciprocal_lattice.volume
    k_per_dir = (target * vol_recip) ** (1.0 / 3.0)
    grid = []
    recip_abc = structure.lattice.reciprocal_lattice.abc
    for b_len in recip_abc:
        n = max(1, int(np.round(k_per_dir * b_len / max(recip_abc))))
        grid.append(n)
    return (grid[0], grid[1], grid[2])


def _pseudo_dict(structure: Structure) -> dict[str, str]:
    """Map each element symbol to its UPF filename."""
    out: dict[str, str] = {}
    for el in structure.composition.elements:
        sym = str(el.symbol)
        if sym in _PSEUDO_BY_ELEMENT:
            out[sym] = _PSEUDO_BY_ELEMENT[sym]
        else:
            out[sym] = f"{sym}.pbe-n-kjpaw_psl.1.0.0.UPF"
            logger.warning(
                "No default pseudopotential for %s; guessing %s. "
                "You may need to download it.",
                sym,
                out[sym],
            )
    return out


def coarse_relax_pw_input(
    structure: Structure,
    pseudo_dir: Path,
    ecutwfc: float = 40.0,
    ecutrho: float = 320.0,
    kppa: int = 300,
) -> PWInput:
    """
    PBE + spin-polarized coarse relaxation for hackathon screening.

    ecutwfc ~ 40 Ry (≈ 544 eV kinetic cutoff); ecutrho ~ 320 Ry for PAW.
    """
    pseudo = _pseudo_dict(structure)
    kgrid = _kpoints_grid(structure, kppa=kppa)

    has_magnetic = any(str(s.specie.symbol) in _MAGMOM_BY_ELEMENT for s in structure)

    control = {
        "calculation": "vc-relax",
        "restart_mode": "from_scratch",
        "pseudo_dir": str(pseudo_dir.resolve()),
        "outdir": "./tmp",
        "tprnfor": True,
        "tstress": True,
        "forc_conv_thr": 1.0e-3,
        "etot_conv_thr": 1.0e-5,
    }

    system: dict[str, object] = {
        "ecutwfc": ecutwfc,
        "ecutrho": ecutrho,
        "occupations": "smearing",
        "smearing": "mp",
        "degauss": 0.02,
    }

    if has_magnetic:
        system["nspin"] = 2
        for el in structure.composition.elements:
            sym = str(el.symbol)
            mag = _MAGMOM_BY_ELEMENT.get(sym, _DEFAULT_MAG)
            system[f"starting_magnetization({_species_index(structure, sym)})"] = mag

    electrons = {
        "conv_thr": 1.0e-6,
        "mixing_beta": 0.3,
    }

    ions = {
        "ion_dynamics": "bfgs",
    }

    cell = {
        "cell_dynamics": "bfgs",
        "press": 0.0,
    }

    return PWInput(
        structure,
        pseudo=pseudo,
        control=control,
        system=system,
        electrons=electrons,
        ions=ions,
        cell=cell,
        kpoints_mode="automatic",
        kpoints_grid=kgrid,
        kpoints_shift=(0, 0, 0),
    )


def _species_index(structure: Structure, symbol: str) -> int:
    """1-based species index matching QE ntyp ordering."""
    species_ordered = list(dict.fromkeys(str(s.specie.symbol) for s in structure))
    return species_ordered.index(symbol) + 1


def write_run_folder(
    structure: Structure,
    run_dir: Path,
    candidate_id: str,
    formula: str,
    source_file: Path,
    preset: str,
    pseudo_dir: Path,
    kppa: int = 300,
) -> None:
    """Write pw.x input file and metadata.yaml under run_dir."""
    run_dir.mkdir(parents=True, exist_ok=True)

    preset_fn = {"coarse_relax": coarse_relax_pw_input}.get(preset)
    if preset_fn is None:
        raise ValueError(f"Unknown preset: {preset}")

    pw_input = preset_fn(structure, pseudo_dir=pseudo_dir, kppa=kppa)
    pw_input.write_file(str(run_dir / "pw.in"))

    meta = {
        "candidate_id": candidate_id,
        "formula": formula,
        "source_file": str(source_file),
        "preset": preset,
        "pseudo_dir": str(pseudo_dir.resolve()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    with (run_dir / "metadata.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(meta, handle, sort_keys=False)


def generate_inputs_from_candidate_list(
    candidate_list: pd.DataFrame,
    outdir: Path,
    preset: str = "coarse_relax",
    pseudo_dir: Path | None = None,
    kppa: int = 300,
) -> None:
    """
    For each row, read relaxed CIF and emit a QE pw.x input folder.

    Parameters
    ----------
    candidate_list
        Must include candidate_id and relaxed_file columns (aliases accepted).
    outdir
        Base directory, e.g. dft/inputs/Co-Bi; each candidate becomes a subfolder.
    preset
        Name of QE recipe (only coarse_relax implemented).
    pseudo_dir
        Path to directory containing UPF pseudopotential files.
    kppa
        K-point density (per atom).
    """
    if pseudo_dir is None:
        pseudo_dir = PSEUDO_DIR_DEFAULT

    df = candidate_list.copy()
    col_id = "candidate_id" if "candidate_id" in df.columns else None
    col_relaxed = "relaxed_file" if "relaxed_file" in df.columns else None
    if col_id is None:
        for alt in ("id", "candidate"):
            if alt in df.columns:
                col_id = alt
                break
    if col_relaxed is None:
        for alt in ("structure_path", "initial_file_path"):
            if alt in df.columns:
                col_relaxed = alt
                break
    if col_id is None or col_relaxed is None:
        raise ValueError("candidate_list must contain candidate_id and relaxed_file (or aliases).")

    col_formula = "formula" if "formula" in df.columns else None

    outdir = outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    for _, row in tqdm(df.iterrows(), total=len(df), desc="QE inputs"):
        cid = str(row[col_id]).strip()
        path = Path(str(row[col_relaxed]).strip()).expanduser()
        if not path.is_file():
            logger.error("Missing structure file for %s: %s", cid, path)
            continue
        try:
            struct = Structure.from_file(path)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to read structure for %s: %s", cid, exc)
            continue

        formula = str(row[col_formula]) if col_formula and pd.notna(row[col_formula]) else struct.composition.reduced_formula
        run_dir = outdir / cid
        write_run_folder(struct, run_dir, cid, formula, path.resolve(), preset=preset, pseudo_dir=pseudo_dir, kppa=kppa)
        logger.info("Wrote QE inputs to %s", run_dir)
