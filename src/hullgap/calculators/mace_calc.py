"""MACE-MP calculator wrapper for ASE relaxations (optional backup model)."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ase.calculators.calculator import Calculator

logger = logging.getLogger(__name__)


def get_mace_calculator(model_path: str | None = None) -> Calculator:
    """Load MACE-MP and return an ASE calculator.

    Parameters
    ----------
    model_path
        Path to a local MACE model file.  If *None*, uses the default
        MACE-MP-0 medium foundation model downloaded on first use.

    Raises
    ------
    ImportError
        If ``mace-torch`` is not installed.
    """
    try:
        from mace.calculators import mace_mp
    except ImportError as exc:
        raise ImportError(
            "MACE-MP is not installed. Install it with:\n"
            "  pip install mace-torch\n"
            "MACE is optional; CHGNet is the default model."
        ) from exc

    logger.info("Loading MACE-MP foundation model …")
    calc = mace_mp(model=model_path, default_dtype="float64")
    logger.info("MACE-MP calculator ready.")
    return calc
