"""MatterSim calculator wrapper for ASE relaxations."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ase.calculators.calculator import Calculator

logger = logging.getLogger(__name__)


def get_mattersim_calculator(device: str | None = None) -> Calculator:
    """Load the pretrained MatterSim model and return an ASE calculator.

    Parameters
    ----------
    device
        Torch device string (``"cuda"``, ``"cpu"``).  When *None* the
        calculator auto-detects CUDA availability.
    """
    try:
        import torch
        from mattersim.forcefield.potential import MatterSimCalculator
    except ImportError as exc:
        raise ImportError(
            "MatterSim is not installed. Install it with:\n"
            "  pip install mattersim"
        ) from exc

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info("Loading MatterSim calculator (device=%s) …", device)
    calc = MatterSimCalculator(device=device)
    logger.info("MatterSim calculator ready.")
    return calc
