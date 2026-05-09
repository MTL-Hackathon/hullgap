"""
FastAPI server for Project BROT frontend.

Wraps the dummy backend logic from backend.py as REST endpoints.
Run with:  uvicorn ui.server:app --port 8000 --reload
"""

from __future__ import annotations

import time
from math import gcd
from typing import List, Optional

import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Project BROT API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

RNG = np.random.default_rng()

COMPOSITIONS: list[tuple[int, int]] = [
    (1, 0),
    (0, 1),
    (1, 8), (1, 6), (1, 5), (1, 4), (1, 3), (1, 2), (2, 3),
    (1, 1),
    (3, 2), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1), (8, 1),
]

NFORM_RANGE = {"pure": (1, 8), "binary": (1, 4)}
MAX_ATOMS = 24


def _reduced_formula(n_a: int, n_b: int, el_a: str, el_b: str) -> str:
    if n_a == 0:
        return el_b
    if n_b == 0:
        return el_a
    g = gcd(n_a, n_b)
    ra, rb = n_a // g, n_b // g
    part_a = el_a if ra == 1 else f"{el_a}{ra}"
    part_b = el_b if rb == 1 else f"{el_b}{rb}"
    return part_a + part_b


class GenerateRequest(BaseModel):
    element_a: str
    element_b: str
    n_candidates: int = 50


class CandidateItem(BaseModel):
    composition: str
    formula: str
    n_atoms: int
    x_B: float
    formation_energy_eV_atom: float
    e_above_hull_eV_atom: float
    predicted_stable: bool


class ValidateRequest(BaseModel):
    candidates: List[CandidateItem]


class MaceResultItem(CandidateItem):
    mace_energy_eV_atom: float
    mace_e_above_hull_eV_atom: float
    mace_stable: bool


@app.post("/generate", response_model=List[CandidateItem])
def generate(req: GenerateRequest):
    el_a, el_b = req.element_a, req.element_b
    target = max(req.n_candidates, 1)
    per_comp = max(1, target // len(COMPOSITIONS))
    rows: list[dict] = []

    for n_a_fu, n_b_fu in COMPOSITIONS:
        is_pure = n_a_fu == 0 or n_b_fu == 0
        nf_lo, nf_hi = NFORM_RANGE["pure" if is_pure else "binary"]

        for _ in range(per_comp):
            nform = int(RNG.integers(nf_lo, nf_hi + 1))
            n_a = n_a_fu * nform
            n_b = n_b_fu * nform
            n_atoms = n_a + n_b
            if n_atoms > MAX_ATOMS or n_atoms == 0:
                continue

            formula = _reduced_formula(n_a, n_b, el_a, el_b)
            x_b = n_b / n_atoms

            fe = float(RNG.normal(-0.15, 0.12))
            if is_pure:
                fe = 0.0
            e_hull = max(0.0, fe + float(RNG.uniform(0.0, 0.10)))

            rows.append({
                "composition": f"{el_a}{n_a}{el_b}{n_b}" if n_b else f"{el_a}{n_a}",
                "formula": formula,
                "n_atoms": n_atoms,
                "x_B": round(x_b, 4),
                "formation_energy_eV_atom": round(fe, 4),
                "e_above_hull_eV_atom": round(e_hull, 4),
                "predicted_stable": e_hull < 0.025,
            })

            if len(rows) >= target:
                break
        if len(rows) >= target:
            break

    while len(rows) < target:
        n_a_fu, n_b_fu = COMPOSITIONS[int(RNG.integers(2, len(COMPOSITIONS)))]
        nform = int(RNG.integers(1, 4))
        n_a, n_b = n_a_fu * nform, n_b_fu * nform
        n_atoms = n_a + n_b
        if n_atoms > MAX_ATOMS or n_atoms == 0:
            continue
        formula = _reduced_formula(n_a, n_b, el_a, el_b)
        x_b = n_b / n_atoms
        fe = float(RNG.normal(-0.15, 0.12))
        e_hull = max(0.0, fe + float(RNG.uniform(0.0, 0.10)))
        rows.append({
            "composition": f"{el_a}{n_a}{el_b}{n_b}",
            "formula": formula,
            "n_atoms": n_atoms,
            "x_B": round(x_b, 4),
            "formation_energy_eV_atom": round(fe, 4),
            "e_above_hull_eV_atom": round(e_hull, 4),
            "predicted_stable": e_hull < 0.025,
        })

    result = sorted(rows[:target], key=lambda r: r["e_above_hull_eV_atom"])
    return result


@app.post("/validate", response_model=List[MaceResultItem])
def validate(req: ValidateRequest):
    results: list[dict] = []

    for c in req.candidates:
        time.sleep(float(RNG.uniform(0.02, 0.08)))

        mace_fe = c.formation_energy_eV_atom + float(RNG.normal(0.0, 0.03))
        mace_eh = max(0.0, mace_fe + float(RNG.uniform(0.0, 0.06)))

        results.append({
            **c.model_dump(),
            "mace_energy_eV_atom": round(mace_fe, 4),
            "mace_e_above_hull_eV_atom": round(mace_eh, 4),
            "mace_stable": mace_eh < 0.020,
        })

    results.sort(key=lambda r: r["mace_e_above_hull_eV_atom"])
    return results
