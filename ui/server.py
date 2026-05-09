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

CRYSTAL_SYSTEMS = [
    "Triclinic", "Monoclinic", "Orthorhombic",
    "Tetragonal", "Trigonal", "Hexagonal", "Cubic",
]
CRYSTAL_SYSTEM_WEIGHTS = [0.25, 0.20, 0.15, 0.12, 0.08, 0.10, 0.10]


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
    crystal_system: str


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
                "crystal_system": RNG.choice(CRYSTAL_SYSTEMS, p=CRYSTAL_SYSTEM_WEIGHTS),
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
            "crystal_system": RNG.choice(CRYSTAL_SYSTEMS, p=CRYSTAL_SYSTEM_WEIGHTS),
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


# ---------------------------------------------------------------------------
# Crystal structure endpoint
# ---------------------------------------------------------------------------

ELEMENT_RADII = {"Co": 1.25, "Bi": 1.55}


def _avg_radius(elements: list[str]) -> float:
    return float(np.mean([ELEMENT_RADII.get(e, 1.4) for e in elements]))


def _cubic_lattice(a: float) -> list[list[float]]:
    return [[a, 0, 0], [0, a, 0], [0, 0, a]]


def _hexagonal_lattice(a: float, c: float) -> list[list[float]]:
    return [[a, 0, 0], [-a / 2, a * np.sqrt(3) / 2, 0], [0, 0, c]]


def _tetragonal_lattice(a: float, c: float) -> list[list[float]]:
    return [[a, 0, 0], [0, a, 0], [0, 0, c]]


def _make_prototypes(el_a: str, el_b: str) -> dict[str, dict]:
    """Generate all prototype structures for a binary system (pure numpy)."""
    r = _avg_radius([el_a, el_b])
    protos: dict[str, dict] = {}

    # CsCl (B2)
    a = 2 * r / np.sqrt(3) * 2
    protos["CsCl_B2"] = {
        "lattice": _cubic_lattice(a),
        "species": [el_a, el_b],
        "frac_coords": [[0, 0, 0], [0.5, 0.5, 0.5]],
    }

    # NaCl (B1)
    a = 2 * r * np.sqrt(2)
    protos["NaCl_B1"] = {
        "lattice": _cubic_lattice(a),
        "species": [el_a] * 4 + [el_b] * 4,
        "frac_coords": [
            [0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5],
            [0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.5], [0.5, 0.5, 0.5],
        ],
    }

    # ZnS (B3)
    a = 4 * r / np.sqrt(3)
    protos["ZnS_B3"] = {
        "lattice": _cubic_lattice(a),
        "species": [el_a] * 4 + [el_b] * 4,
        "frac_coords": [
            [0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5],
            [0.25, 0.25, 0.25], [0.75, 0.75, 0.25],
            [0.75, 0.25, 0.75], [0.25, 0.75, 0.75],
        ],
    }

    # NiAs (B81)
    a_hex = 2 * r
    c_hex = a_hex * 1.63
    protos["NiAs_B81"] = {
        "lattice": _hexagonal_lattice(a_hex, c_hex),
        "species": [el_a, el_a, el_b, el_b],
        "frac_coords": [
            [0, 0, 0], [0, 0, 0.5],
            [1 / 3, 2 / 3, 0.25], [2 / 3, 1 / 3, 0.75],
        ],
    }

    # FeSi (B20)
    a = 2 * r * 2.1
    protos["FeSi_B20"] = {
        "lattice": _cubic_lattice(a),
        "species": [el_a] * 4 + [el_b] * 4,
        "frac_coords": [
            [0.137, 0.137, 0.137], [0.637, 0.363, 0.863],
            [0.363, 0.863, 0.637], [0.863, 0.637, 0.363],
            [0.845, 0.845, 0.845], [0.345, 0.655, 0.155],
            [0.655, 0.155, 0.345], [0.155, 0.345, 0.655],
        ],
    }

    # Cu3Au (L12)
    a = 2 * r * np.sqrt(2)
    protos["Cu3Au_L12"] = {
        "lattice": _cubic_lattice(a),
        "species": [el_a] * 3 + [el_b],
        "frac_coords": [
            [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5], [0, 0, 0],
        ],
    }

    # Ni3Sn (D019)
    a_hex = 2 * r * np.sqrt(2)
    c_hex = a_hex * 0.816
    protos["Ni3Sn_D019"] = {
        "lattice": _hexagonal_lattice(a_hex, c_hex),
        "species": [el_a] * 6 + [el_b] * 2,
        "frac_coords": [
            [5 / 6, 2 / 3, 1 / 4], [1 / 6, 1 / 3, 3 / 4],
            [1 / 3, 1 / 6, 1 / 4], [2 / 3, 5 / 6, 3 / 4],
            [1 / 2, 1 / 2, 1 / 4], [1 / 2, 1 / 2, 3 / 4],
            [0, 0, 1 / 4], [0, 0, 3 / 4],
        ],
    }

    # Au3Cu (L12 inverse)
    a = 2 * r * np.sqrt(2)
    protos["Au3Cu_L12"] = {
        "lattice": _cubic_lattice(a),
        "species": [el_a] + [el_b] * 3,
        "frac_coords": [
            [0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5],
        ],
    }

    # Sn3Ni (D019 inverse)
    a_hex = 2 * r * np.sqrt(2)
    c_hex = a_hex * 0.816
    protos["Sn3Ni_D019"] = {
        "lattice": _hexagonal_lattice(a_hex, c_hex),
        "species": [el_a] * 2 + [el_b] * 6,
        "frac_coords": [
            [0, 0, 1 / 4], [0, 0, 3 / 4],
            [5 / 6, 2 / 3, 1 / 4], [1 / 6, 1 / 3, 3 / 4],
            [1 / 3, 1 / 6, 1 / 4], [2 / 3, 5 / 6, 3 / 4],
            [1 / 2, 1 / 2, 1 / 4], [1 / 2, 1 / 2, 3 / 4],
        ],
    }

    # MoSi2 (C11b)
    a = 2 * r * np.sqrt(2)
    c = a * 2.45
    protos["MoSi2_C11b"] = {
        "lattice": _tetragonal_lattice(a, c),
        "species": [el_a] * 2 + [el_b] * 4,
        "frac_coords": [
            [0, 0, 0], [0.5, 0.5, 0.5],
            [0, 0, 1 / 3], [0, 0, 2 / 3],
            [0.5, 0.5, 1 / 3 + 0.5], [0.5, 0.5, 2 / 3 - 0.5],
        ],
    }

    # CaF2 (C1)
    a = 2 * r * 2
    protos["CaF2_C1"] = {
        "lattice": _cubic_lattice(a),
        "species": [el_a] * 4 + [el_b] * 8,
        "frac_coords": [
            [0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5],
            [0.25, 0.25, 0.25], [0.75, 0.75, 0.25],
            [0.75, 0.25, 0.75], [0.25, 0.75, 0.75],
            [0.25, 0.25, 0.75], [0.75, 0.75, 0.75],
            [0.75, 0.25, 0.25], [0.25, 0.75, 0.25],
        ],
    }

    # CaF2 inverse
    protos["CaF2_C1_inv"] = {
        "lattice": _cubic_lattice(a),
        "species": [el_a] * 8 + [el_b] * 4,
        "frac_coords": [
            [0.25, 0.25, 0.25], [0.75, 0.75, 0.25],
            [0.75, 0.25, 0.75], [0.25, 0.75, 0.75],
            [0.25, 0.25, 0.75], [0.75, 0.75, 0.75],
            [0.75, 0.25, 0.25], [0.25, 0.75, 0.25],
            [0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5],
        ],
    }

    # CaCu5 (D2d)
    a_hex = 2 * r
    c_hex = a_hex * 0.81
    protos["CaCu5_D2d"] = {
        "lattice": _hexagonal_lattice(a_hex, c_hex),
        "species": [el_a] * 5 + [el_b],
        "frac_coords": [
            [1 / 3, 2 / 3, 0], [2 / 3, 1 / 3, 0],
            [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0.5],
            [0, 0, 0],
        ],
    }

    # CaCu5 inverse
    protos["CaCu5_D2d_inv"] = {
        "lattice": _hexagonal_lattice(a_hex, c_hex),
        "species": [el_a] + [el_b] * 5,
        "frac_coords": [
            [0, 0, 0],
            [1 / 3, 2 / 3, 0], [2 / 3, 1 / 3, 0],
            [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0.5],
        ],
    }

    return protos


class StructureRequest(BaseModel):
    element_a: str
    element_b: str
    prototype: str


class StructureResponse(BaseModel):
    prototype: str
    formula: str
    n_atoms: int
    lattice_matrix: list[list[float]]
    lattice_params: dict
    volume: float
    species: list[str]
    frac_coords: list[list[float]]
    cart_coords: list[list[float]]


class PrototypeListResponse(BaseModel):
    prototypes: list[str]


@app.post("/structure", response_model=StructureResponse)
def get_structure(req: StructureRequest):
    protos = _make_prototypes(req.element_a, req.element_b)
    if req.prototype not in protos:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail=f"Unknown prototype: {req.prototype}")

    proto = protos[req.prototype]
    lat = np.array(proto["lattice"])
    frac = np.array(proto["frac_coords"])
    cart = (frac @ lat).tolist()

    # Compute lattice parameters
    a_vec, b_vec, c_vec = lat[0], lat[1], lat[2]
    a_len = float(np.linalg.norm(a_vec))
    b_len = float(np.linalg.norm(b_vec))
    c_len = float(np.linalg.norm(c_vec))
    alpha = float(np.degrees(np.arccos(
        np.clip(np.dot(b_vec, c_vec) / (b_len * c_len), -1, 1)
    )))
    beta = float(np.degrees(np.arccos(
        np.clip(np.dot(a_vec, c_vec) / (a_len * c_len), -1, 1)
    )))
    gamma = float(np.degrees(np.arccos(
        np.clip(np.dot(a_vec, b_vec) / (a_len * b_len), -1, 1)
    )))
    volume = float(abs(np.dot(a_vec, np.cross(b_vec, c_vec))))

    species = proto["species"]
    counts: dict[str, int] = {}
    for s in species:
        counts[s] = counts.get(s, 0) + 1
    g = gcd(*counts.values()) if len(counts) > 1 else list(counts.values())[0]
    formula_parts = []
    for el, cnt in counts.items():
        n = cnt // g
        formula_parts.append(el if n == 1 else f"{el}{n}")
    formula = "".join(formula_parts)

    return {
        "prototype": req.prototype,
        "formula": formula,
        "n_atoms": len(species),
        "lattice_matrix": lat.tolist(),
        "lattice_params": {
            "a": round(a_len, 4),
            "b": round(b_len, 4),
            "c": round(c_len, 4),
            "alpha": round(alpha, 2),
            "beta": round(beta, 2),
            "gamma": round(gamma, 2),
        },
        "volume": round(volume, 4),
        "species": species,
        "frac_coords": proto["frac_coords"],
        "cart_coords": cart,
    }


@app.get("/prototypes", response_model=PrototypeListResponse)
def list_prototypes():
    protos = _make_prototypes("A", "B")
    return {"prototypes": list(protos.keys())}
