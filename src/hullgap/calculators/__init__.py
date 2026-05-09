"""Calculator backends for MLIP-based structure relaxation."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ase.calculators.calculator import Calculator

AVAILABLE_MODELS = ("chgnet", "mace")


def get_calculator(model: str = "chgnet") -> Calculator:
    """Return an ASE-compatible calculator for the requested MLIP backend.

    Parameters
    ----------
    model
        One of ``"chgnet"`` or ``"mace"``.

    Raises
    ------
    ValueError
        Unknown model name.
    ImportError
        Backend package is not installed.
    """
    model = model.lower().strip()
    if model == "chgnet":
        from hullgap.calculators.chgnet_calc import get_chgnet_calculator

        return get_chgnet_calculator()
    if model == "mace":
        from hullgap.calculators.mace_calc import get_mace_calculator

        return get_mace_calculator()
    raise ValueError(
        f"Unknown model {model!r}. Available: {', '.join(AVAILABLE_MODELS)}"
    )
