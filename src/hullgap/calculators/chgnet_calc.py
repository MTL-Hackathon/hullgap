"""CHGNet calculator wrapper for ASE relaxations."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ase.calculators.calculator import Calculator

logger = logging.getLogger(__name__)


def get_chgnet_calculator() -> Calculator:
    """Load the pretrained CHGNet model and return an ASE calculator.

    Raises
    ------
    ImportError
        If ``chgnet`` is not installed.
    """
    try:
        from chgnet.model.dynamics import CHGNetCalculator
    except ImportError as exc:
        raise ImportError(
            "CHGNet is not installed. Install it with:\n"
            "  pip install chgnet\n"
            "See INSTALL.md for details."
        ) from exc

    logger.info("Loading CHGNet pretrained model …")
    calc = CHGNetCalculator()
    logger.info("CHGNet calculator ready.")
    return calc
