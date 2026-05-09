"""
Parse finished (or crashed) VASP runs into a tabular summary for HullGap.

Designed to run without the VASP binary: only vasprun.xml / OUTCAR (and
optionally POSCAR) are read.
"""

from __future__ import annotations

import logging
import re
import traceback
from pathlib import Path

import pandas as pd
from pymatgen.core import Structure
from pymatgen.io.vasp.outputs import Outcar, Vasprun

logger = logging.getLogger(__name__)

_TOTEN_RE = re.compile(r"free\s+energy\s+TOTEN\s*=\s*([-+0-9.eE]+)")


def _last_toten_ev(outcar_path: Path) -> float | None:
    """Best-effort parse of the last TOTEN line from OUTCAR."""
    try:
        text = outcar_path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None
    last: float | None = None
    for match in _TOTEN_RE.finditer(text):
        try:
            last = float(match.group(1))
        except ValueError:
            continue
    return last


def _try_total_magnetization(vasprun: Vasprun | None, outcar_path: Path | None) -> float | None:
    if outcar_path and outcar_path.is_file():
        try:
            oc = Outcar(str(outcar_path))
            if oc.total_mag is not None:
                return float(oc.total_mag)
        except Exception:  # noqa: BLE001
            pass
    if vasprun is not None:
        # Some pymatgen versions expose magnetization on Vasprun; keep optional.
        mag = getattr(vasprun, "total_mag", None)
        if mag is not None:
            return float(mag)
    return None


def parse_single_run_dir(run_dir: Path) -> dict[str, object]:
    """
    Parse one candidate folder (may contain vasprun.xml and/or OUTCAR).

    candidate_id is inferred from the directory name (leaf run folder).
    """
    candidate_id = run_dir.name
    formula = ""
    status = "failed"
    converged = False
    dft_energy_total_eV: float | None = None
    dft_energy_per_atom_eV: float | None = None
    n_atoms: int | None = None
    volume_per_atom: float | None = None
    total_magnetization: float | None = None
    final_structure_path = ""
    error_message = ""

    vasprun_path = run_dir / "vasprun.xml"
    outcar_path = run_dir / "OUTCAR"
    poscar_path = run_dir / "POSCAR"

    vasprun: Vasprun | None = None

    try:
        if vasprun_path.is_file():
            vasprun = Vasprun(str(vasprun_path), parse_dos=False, parse_eigen=False)
            final_struct = vasprun.final_structure
            formula = final_struct.composition.reduced_formula
            n_atoms = len(final_struct)
            dft_energy_total_eV = float(vasprun.final_energy)
            dft_energy_per_atom_eV = dft_energy_total_eV / n_atoms if n_atoms else None
            volume_per_atom = float(final_struct.volume / n_atoms) if n_atoms else None
            converged = bool(vasprun.converged)
            total_magnetization = _try_total_magnetization(vasprun, outcar_path if outcar_path.is_file() else None)

            cif_path = run_dir / "final_structure.cif"
            final_struct.to(filename=str(cif_path))
            final_structure_path = str(cif_path.resolve())
            status = "ok" if converged else "not_converged"
        elif outcar_path.is_file() and poscar_path.is_file():
            struct = Structure.from_file(str(poscar_path))
            formula = struct.composition.reduced_formula
            n_atoms = len(struct)
            dft_energy_total_eV = _last_toten_ev(outcar_path)
            if dft_energy_total_eV is None:
                raise ValueError("Could not parse energy from OUTCAR (no TOTEN lines).")
            dft_energy_per_atom_eV = dft_energy_total_eV / n_atoms if n_atoms else None
            volume_per_atom = float(struct.volume / n_atoms) if n_atoms else None
            total_magnetization = _try_total_magnetization(None, outcar_path)
            status = "partial_outcar"
            converged = False
            error_message = "vasprun.xml missing; energies parsed from OUTCAR only."
        else:
            error_message = "No vasprun.xml found; OUTCAR/POSCAR pair also missing."
    except Exception as exc:  # noqa: BLE001
        status = "failed"
        error_message = f"{exc}\n{traceback.format_exc()}"

    return {
        "candidate_id": candidate_id,
        "formula": formula,
        "status": status,
        "converged": converged,
        "dft_energy_total_eV": dft_energy_total_eV,
        "dft_energy_per_atom_eV": dft_energy_per_atom_eV,
        "n_atoms": n_atoms if n_atoms is not None else "",
        "volume_per_atom": volume_per_atom if volume_per_atom is not None else "",
        "total_magnetization": total_magnetization if total_magnetization is not None else "",
        "final_structure_path": final_structure_path,
        "error_message": error_message,
    }


def discover_run_roots(root: Path) -> list[Path]:
    """
    Return directories that look like leaf VASP runs (contain vasprun.xml or OUTCAR).

    If root itself is a run, return [root]. Else collect directories containing
    vasprun.xml, de-duplicated.
    """
    root = root.resolve()
    if (root / "vasprun.xml").is_file() or (root / "OUTCAR").is_file():
        return [root]

    runs: list[Path] = []
    for path in sorted(root.rglob("vasprun.xml")):
        runs.append(path.parent)
    if not runs:
        for path in sorted(root.rglob("OUTCAR")):
            runs.append(path.parent)
    seen: set[Path] = set()
    ordered: list[Path] = []
    for p in runs:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            ordered.append(rp)
    return ordered


def parse_run_tree(run_root: Path) -> pd.DataFrame:
    """Parse all discovered runs under run_root into a DataFrame."""
    dirs = discover_run_roots(run_root)
    if not dirs:
        logger.warning("No VASP outputs found under %s", run_root)
        return pd.DataFrame()

    rows = [parse_single_run_dir(d) for d in dirs]
    return pd.DataFrame(rows)


def write_energy_table(df: pd.DataFrame, out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    logger.info("Wrote %d parsed runs to %s", len(df), out_csv)
