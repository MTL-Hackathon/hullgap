"""
Parse finished (or crashed) Quantum ESPRESSO pw.x runs into a tabular summary.

Reads pw.x stdout output files. Does not require the QE binary to be installed
for parsing.
"""

from __future__ import annotations

import logging
import re
import traceback
from pathlib import Path

import pandas as pd
from pymatgen.core import Structure
from pymatgen.io.pwscf import PWOutput

logger = logging.getLogger(__name__)

RY_TO_EV = 13.605693009

_TOTAL_ENERGY_RE = re.compile(r"!\s+total energy\s+=\s+([-+0-9.eEdD]+)\s+Ry")
_TOTAL_MAG_RE = re.compile(r"total magnetization\s+=\s+([-+0-9.eEdD]+)\s+Bohr")
_CONVERGENCE_RE = re.compile(r"convergence has been achieved")
_CELL_RE = re.compile(r"Begin final coordinates")
_NATOMS_RE = re.compile(r"number of atoms/cell\s+=\s+(\d+)")
_VOLUME_RE = re.compile(r"unit-cell volume\s+=\s+([-+0-9.eEdD]+)")
_ATOMIC_POS_BLOCK_RE = re.compile(
    r"ATOMIC_POSITIONS\s+\S+\s*\n((?:\s*[A-Z][a-z]?\s+[-+0-9.]+\s+[-+0-9.]+\s+[-+0-9.]+.*\n)+)",
    re.MULTILINE,
)
_ATOM_LINE_RE = re.compile(r"^\s*([A-Z][a-z]?)\s+[-+0-9.]+\s+[-+0-9.]+\s+[-+0-9.]+", re.MULTILINE)


def _parse_final_energy_ry(text: str) -> float | None:
    """Extract last '! total energy' in Ry."""
    last: float | None = None
    for m in _TOTAL_ENERGY_RE.finditer(text):
        try:
            last = float(m.group(1).replace("d", "e").replace("D", "E"))
        except ValueError:
            continue
    return last


def _parse_total_magnetization(text: str) -> float | None:
    last: float | None = None
    for m in _TOTAL_MAG_RE.finditer(text):
        try:
            last = float(m.group(1).replace("d", "e").replace("D", "E"))
        except ValueError:
            continue
    return last


def _parse_n_atoms(text: str) -> int | None:
    m = _NATOMS_RE.search(text)
    if m:
        return int(m.group(1))
    return None


def _parse_volume_bohr3(text: str) -> float | None:
    last: float | None = None
    for m in _VOLUME_RE.finditer(text):
        try:
            last = float(m.group(1))
        except ValueError:
            continue
    return last


def _find_pw_output(run_dir: Path) -> Path | None:
    """Find the main pw.x stdout file (pw.out, *.out, or *.log)."""
    for name in ["pw.out", "pwscf.out"]:
        p = run_dir / name
        if p.is_file():
            return p
    for ext in ["*.out", "*.log"]:
        found = sorted(run_dir.glob(ext))
        if found:
            return found[0]
    return None


def _try_read_final_structure(run_dir: Path) -> Structure | None:
    """Try to read final structure from QE data-file-schema.xml or the input."""
    schema = run_dir / "tmp" / "pwscf.save" / "data-file-schema.xml"
    if schema.is_file():
        try:
            from pymatgen.io.pwscf import PWInput as _

            # pymatgen doesn't parse data-file-schema.xml directly;
            # fall back to the input structure for now.
        except Exception:  # noqa: BLE001
            pass
    pw_in = run_dir / "pw.in"
    if pw_in.is_file():
        try:
            return PWInput.from_file(str(pw_in)).structure
        except Exception:  # noqa: BLE001
            pass
    return None


def _extract_species_from_text(text: str) -> str:
    """Extract reduced formula from ATOMIC_POSITIONS block."""
    from collections import Counter
    from pymatgen.core import Composition

    block = _ATOMIC_POS_BLOCK_RE.search(text)
    if block:
        atoms = _ATOM_LINE_RE.findall(block.group(1))
        if atoms:
            try:
                return Composition(Counter(atoms)).reduced_formula
            except Exception:  # noqa: BLE001
                pass
    return ""


def _formula_from_pw_output_or_input(text: str, run_dir: Path) -> str:
    """Best-effort formula from pw.out ATOMIC_POSITIONS or pw.in."""
    formula = _extract_species_from_text(text)
    if formula:
        return formula

    pw_in = run_dir / "pw.in"
    if pw_in.is_file():
        try:
            in_text = pw_in.read_text(encoding="utf-8", errors="ignore")
            formula = _extract_species_from_text(in_text)
        except Exception:  # noqa: BLE001
            pass
    return formula


def parse_single_run_dir(run_dir: Path) -> dict[str, object]:
    """
    Parse one candidate folder containing QE pw.x output.

    candidate_id is inferred from the directory name.
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

    out_file = _find_pw_output(run_dir)

    try:
        if out_file is not None and out_file.is_file():
            text = out_file.read_text(encoding="utf-8", errors="ignore")

            e_ry = _parse_final_energy_ry(text)
            if e_ry is not None:
                dft_energy_total_eV = e_ry * RY_TO_EV

            n_atoms = _parse_n_atoms(text)
            if dft_energy_total_eV is not None and n_atoms:
                dft_energy_per_atom_eV = dft_energy_total_eV / n_atoms

            vol_bohr3 = _parse_volume_bohr3(text)
            bohr_to_ang = 0.529177249
            if vol_bohr3 is not None and n_atoms:
                volume_per_atom = (vol_bohr3 * bohr_to_ang**3) / n_atoms

            converged = bool(_CONVERGENCE_RE.search(text))
            total_magnetization = _parse_total_magnetization(text)

            struct = _try_read_final_structure(run_dir)
            if struct is not None:
                formula = struct.composition.reduced_formula
                if n_atoms is None:
                    n_atoms = len(struct)
                cif_path = run_dir / "final_structure.cif"
                struct.to(filename=str(cif_path))
                final_structure_path = str(cif_path.resolve())

            if not formula:
                formula = _formula_from_pw_output_or_input(text, run_dir)

            status = "ok" if converged else "not_converged"
        else:
            error_message = "No pw.x output file found (pw.out, *.out, *.log)."
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
    Return directories that look like QE runs (contain pw.out, *.out, or pw.in).

    If root itself is a run, return [root].
    """
    root = root.resolve()
    if _find_pw_output(root) is not None:
        return [root]

    runs: list[Path] = []
    for path in sorted(root.rglob("pw.out")):
        runs.append(path.parent)
    if not runs:
        for path in sorted(root.rglob("*.out")):
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
        logger.warning("No QE outputs found under %s", run_root)
        return pd.DataFrame()

    rows = [parse_single_run_dir(d) for d in dirs]
    return pd.DataFrame(rows)


def write_energy_table(df: pd.DataFrame, out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    logger.info("Wrote %d parsed runs to %s", len(df), out_csv)
