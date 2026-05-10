"""MACE-MP calculator wrapper for ASE relaxations."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ase.calculators.calculator import Calculator

logger = logging.getLogger(__name__)


def get_mace_calculator(
    model_path: str | None = None,
    device: str = "cpu",
) -> Calculator:
    """Load MACE-MP and return an ASE calculator.

    Parameters
    ----------
    model_path
        Path to a local MACE model file.  If *None*, uses the default
        MACE-MP-0 medium foundation model downloaded on first use.
    device
        Torch device string. Defaults to ``"cpu"`` for safe multiprocessing.
    """
    from mace.calculators import mace_mp

    logger.info("Loading MACE-MP foundation model (device=%s) …", device)
    calc = mace_mp(model=model_path, default_dtype="float64", device=device)
    logger.info("MACE-MP calculator ready.")
    return calc
