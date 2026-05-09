"""
Select a small subset of MLIP-scored candidates for expensive DFT validation.

Reads hull-score tables produced upstream (MLIP relaxation + hull analysis),
applies sanity filters, and ranks by energy above hull for hand-off to DFT.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from pymatgen.core import Structure

logger = logging.getLogger(__name__)

# Column aliases for robustness across pipeline CSV variants
COL_CANDIDATE_ID = ("candidate_id", "id", "candidate")
COL_FORMULA = ("formula", "pretty_formula", "full_formula")
COL_SYSTEM = ("system", "target_system", "chemical_system")
COL_RELAXED = ("relaxed_file", "initial_file_path", "relaxed_path", "structure_path")
COL_E_AH = (
    "e_above_hull_eV_atom",
    "delta_to_existing_hull_eV_atom",
    "mlip_e_above_hull_eV_atom",
)
COL_STATUS = ("predicted_status", "status", "relax_status")
COL_N_ATOMS = ("n_atoms", "natoms", "num_atoms")
COL_PROTOTYPE = ("source_prototype", "prototype_label", "prototype")


def _first_present(df: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    for name in candidates:
        if name in df.columns:
            return name
    return None


def _normalize_failed_mask(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip().str.lower()
    bad = {"failed_relaxation", "failed", "error", "nan", "none"}
    return s.isin(bad)


def _resolve_relaxed_path(relaxed_dir: Path, raw: str) -> Path | None:
    """Resolve a path from CSV against cwd and relaxed_dir."""
    if not raw or not str(raw).strip():
        return None
    p = Path(raw.strip())
    if p.is_file():
        return p.resolve()
    trial = Path.cwd() / p
    if trial.is_file():
        return trial.resolve()
    name = p.name
    trial = relaxed_dir / name
    if trial.is_file():
        return trial.resolve()
    trial = relaxed_dir / p
    if trial.is_file():
        return trial.resolve()
    return None


def _n_atoms_from_row(row: pd.Series, relaxed_path: Path | None) -> int | None:
    for key in COL_N_ATOMS:
        if key in row.index and pd.notna(row[key]):
            try:
                return int(row[key])
            except (TypeError, ValueError):
                continue
    if relaxed_path and relaxed_path.is_file():
        try:
            struct = Structure.from_file(relaxed_path)
            return len(struct)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not read structure for n_atoms from %s: %s", relaxed_path, exc)
    return None


def select_top_candidates(
    hull_scores: pd.DataFrame,
    relaxed_dir: Path,
    top_n: int,
    max_atoms: int | None = None,
) -> pd.DataFrame:
    """
    Filter and rank candidates for downstream DFT.

    Parameters
    ----------
    hull_scores
        DataFrame from hull_scores CSV.
    relaxed_dir
        Directory containing relaxed CIFs (used to verify paths).
    top_n
        Maximum number of rows to return after sorting.
    max_atoms
        If set, drop rows with n_atoms greater than this threshold.
    """
    df = hull_scores.copy()
    relaxed_dir = relaxed_dir.resolve()

    col_id = _first_present(df, COL_CANDIDATE_ID)
    col_formula = _first_present(df, COL_FORMULA)
    col_system = _first_present(df, COL_SYSTEM)
    col_relaxed = _first_present(df, COL_RELAXED)
    col_e_ah = _first_present(df, COL_E_AH)
    col_status = _first_present(df, COL_STATUS)
    col_proto = _first_present(df, COL_PROTOTYPE)

    missing = [n for n, c in [("candidate_id", col_id), ("relaxed_file", col_relaxed)] if c is None]
    if missing:
        raise ValueError(f"hull_scores CSV missing required columns (need one of each group): {missing}")

    if col_e_ah is None:
        logger.warning("No energy-above-hull column found; sorting will be arbitrary.")
        df["_sort_eah"] = 0.0
    else:
        df["_sort_eah"] = pd.to_numeric(df[col_e_ah], errors="coerce")

    if col_status is not None:
        mask_ok = ~_normalize_failed_mask(df[col_status].fillna(""))
        df = df.loc[mask_ok]
        logger.info("After removing failed-status rows: %d candidates", len(df))

    rows_out: list[dict[str, object]] = []
    for _, row in df.iterrows():
        raw_path = str(row[col_relaxed]) if pd.notna(row[col_relaxed]) else ""
        resolved = _resolve_relaxed_path(relaxed_dir, raw_path)
        if resolved is None:
            logger.debug("Skipping missing relaxed file for %s: %s", row.get(col_id), raw_path)
            continue

        n_atoms = _n_atoms_from_row(row, resolved)
        if max_atoms is not None and n_atoms is not None and n_atoms > max_atoms:
            continue

        e_ah = float(row["_sort_eah"]) if pd.notna(row["_sort_eah"]) else float("nan")

        rows_out.append(
            {
                "candidate_id": row[col_id],
                "formula": row[col_formula] if col_formula else "",
                "system": row[col_system] if col_system and col_system in row else "",
                "relaxed_file": str(resolved),
                "mlip_e_above_hull_eV_atom": e_ah,
                "n_atoms": n_atoms if n_atoms is not None else "",
                "source_prototype": row[col_proto] if col_proto and col_proto in row else "",
            }
        )

    out = pd.DataFrame(rows_out)
    if out.empty:
        return out

    out = out.sort_values(by="mlip_e_above_hull_eV_atom", ascending=True, na_position="last")
    return out.head(top_n).reset_index(drop=True)


def load_hull_scores_csv(path: Path) -> pd.DataFrame:
    """Load hull scores with utf-8 and strip column names."""
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return df


def write_candidate_list(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    logger.info("Wrote %d rows to %s", len(df), out_path)
