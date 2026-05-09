"""Query Materials Project for binary systems and compute void scores.

Output: data/results/void_scores.csv
"""

import os
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from mp_api.client import MPRester

load_dotenv()

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data" / "results"
DATA_DIR.mkdir(parents=True, exist_ok=True)

SYSTEMS = [
    "Co-Bi", "Fe-Bi", "Ni-Bi", "Mn-Bi", "Co-Sb", "Co-Te", "Co-Se",
    "Ru-Bi", "Mo-Bi", "W-Bi", "Ti-Si", "Co-Si", "Ni-Si",
    "Hf-N", "Zr-N", "Ta-N", "Ti-N",
]

# void_score = 1 / (stable_phases + 1)
# Score is 1.0 when no stable phases exist; decreases as more stable phases are found.


def query_system(mpr: MPRester, system: str) -> dict[str, Any]:
    results = mpr.materials.summary.search(
        chemsys=[system],
        fields=["material_id", "energy_above_hull"],
    )
    total = len(results)
    stable = sum(
        1 for r in results
        if r.energy_above_hull is not None and r.energy_above_hull == 0.0
    )
    return {
        "system": system,
        "total_entries": total,
        "stable_phases": stable,
        "zero_stable_phases": stable == 0,
        "void_score": round(1.0 / (stable + 1), 6),
    }


def main() -> None:
    api_key = os.environ.get("MP_API_KEY", "")
    if not api_key:
        raise ValueError("MP_API_KEY not set — copy .env.example to .env and fill it in.")

    rows = []
    with MPRester(api_key) as mpr:
        for system in SYSTEMS:
            print(f"Querying {system} ...", end=" ", flush=True)
            row = query_system(mpr, system)
            rows.append(row)
            print(f"total={row['total_entries']}  stable={row['stable_phases']}  void_score={row['void_score']}")

    out = DATA_DIR / "void_scores.csv"
    df = pd.DataFrame(rows).sort_values("void_score", ascending=False).reset_index(drop=True)
    df.to_csv(out, index=False)
    print(f"\nSaved {out}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
