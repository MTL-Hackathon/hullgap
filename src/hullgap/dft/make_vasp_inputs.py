"""
Generate PBE, spin-polarized VASP relaxation inputs for MLIP-shortlisted structures.

Co-rich binaries can be magnetic; ISPIN=2 and MAGMOM initialization are included
for physically reasonable first relaxations. POTCAR files are intentionally not
written so pseudopotential choice stays with the user / cluster policy.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from tqdm import tqdm
from pymatgen.core import Structure
from pymatgen.io.vasp.inputs import Incar, Kpoints, Poscar

logger = logging.getLogger(__name__)

# Element-specific initial magnetic moments (Bohr magnetons) for ISPIN=2 relaxations.
_DEFAULT_MAG = 0.6
_MAGMOM_BY_ELEMENT: dict[str, float] = {
    "Co": 3.0,
    "Fe": 4.0,
    "Ni": 2.0,
    "Mn": 5.0,
}


def magmom_list_for_structure(structure: Structure) -> list[float]:
    """Return one MAGMOM per site in structure site order."""
    return [_MAGMOM_BY_ELEMENT.get(str(site.specie.symbol), _DEFAULT_MAG) for site in structure]


def coarse_relax_incar(structure: Structure) -> Incar:
    """
    PBE + spin-polarized coarse relaxation settings for hackathon screening.

    This is an MVP preset for relative ranking after MLIP, not production-grade
    thermodynamic convergence.
    """
    incar_dict: dict[str, Any] = {
        "SYSTEM": "HullGap candidate",
        "ENCUT": 520,
        "EDIFF": 1e-5,
        "EDIFFG": -0.03,
        "ISMEAR": 1,
        "SIGMA": 0.2,
        "IBRION": 2,
        "NSW": 100,
        "ISIF": 3,
        "ISPIN": 2,
        "LREAL": "Auto",
        "LASPH": True,
        "LWAVE": False,
        "LCHARG": False,
        "MAGMOM": magmom_list_for_structure(structure),
        "PREC": "Accurate",
    }
    return Incar(incar_dict)


def kpoints_for_structure(structure: Structure, kppa: int = 1000) -> Kpoints:
    """Automatic k-mesh from pymatgen (k-points per reciprocal atom)."""
    return Kpoints.automatic_density(structure, int(kppa), force_gamma=False)


def write_run_folder(
    structure: Structure,
    run_dir: Path,
    candidate_id: str,
    formula: str,
    source_file: Path,
    preset: str,
    kppa: int = 1000,
) -> None:
    """Write POSCAR, INCAR, KPOINTS, and metadata.yaml under run_dir."""
    run_dir.mkdir(parents=True, exist_ok=True)

    preset_incar = {"coarse_relax": coarse_relax_incar}.get(preset)
    if preset_incar is None:
        raise ValueError(f"Unknown preset: {preset}")
    incar = preset_incar(structure)

    Poscar(structure).write_file(str(run_dir / "POSCAR"))
    incar.write_file(str(run_dir / "INCAR"))
    kpoints_for_structure(structure, kppa=kppa).write_file(str(run_dir / "KPOINTS"))

    meta = {
        "candidate_id": candidate_id,
        "formula": formula,
        "source_file": str(source_file),
        "preset": preset,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    with (run_dir / "metadata.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(meta, handle, sort_keys=False)


def generate_inputs_from_candidate_list(
    candidate_list: pd.DataFrame,
    outdir: Path,
    preset: str = "coarse_relax",
    kppa: int = 1000,
) -> None:
    """
    For each row, read relaxed CIF/POSCAR-like path and emit a VASP folder.

    Parameters
    ----------
    candidate_list
        Must include candidate_id and relaxed_file columns (aliases accepted).
    outdir
        Base directory, e.g. dft/inputs/Co-Bi; each candidate becomes a subfolder.
    preset
        Name of INCAR/KPOINTS recipe (only coarse_relax implemented).
    kppa
        K-point density passed to pymatgen automatic_density.
    """
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

    for _, row in tqdm(df.iterrows(), total=len(df), desc="VASP inputs"):
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
        write_run_folder(struct, run_dir, cid, formula, path.resolve(), preset=preset, kppa=kppa)
        logger.info("Wrote VASP inputs to %s", run_dir)
