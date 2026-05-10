"""CHGNet calculator wrapper for ASE relaxations."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ase.calculators.calculator import Calculator

logger = logging.getLogger(__name__)


def get_chgnet_calculator(use_device: str = "cpu") -> Calculator:
    """Load the pretrained CHGNet model and return an ASE calculator.

    Parameters
    ----------
    use_device
        Torch device string. Defaults to ``"cpu"`` for safe multiprocessing.
    """
    try:
        from chgnet.model.dynamics import CHGNetCalculator
    except ImportError as exc:
        raise ImportError(
            "CHGNet is not installed. Install it with:\n"
            "  pip install chgnet"
        ) from exc

    logger.info("Loading CHGNet pretrained model (device=%s) …", use_device)
    calc = CHGNetCalculator(use_device=use_device)
    logger.info("CHGNet calculator ready.")
    return calc
