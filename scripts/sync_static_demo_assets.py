#!/usr/bin/env python3
"""Populate ui/frontend/public/demo with hull CSVs and relaxed CIFs for GitHub Pages static export."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS_SRC = ROOT / "data" / "results"
MATTERGEN_SRC = ROOT / "data" / "mattergen"
DEMO_DST = ROOT / "ui" / "frontend" / "public" / "demo"


def _parse_idx_from_cif_stem(stem: str) -> int | None:
    parts = stem.split("_")
    if len(parts) < 2:
        return None
    try:
        return int(parts[1], 10)
    except ValueError:
        return None


def main() -> None:
    if DEMO_DST.exists():
        shutil.rmtree(DEMO_DST)
    (DEMO_DST / "results").mkdir(parents=True)
    (DEMO_DST / "mattergen").mkdir(parents=True)

    cif_index: dict[str, dict[str, str]] = {}

    for csv_path in sorted(RESULTS_SRC.glob("*_mattersim_hull.csv")):
        shutil.copy2(csv_path, DEMO_DST / "results" / csv_path.name)
        system = csv_path.name.replace("_mattersim_hull.csv", "")
        relaxed = MATTERGEN_SRC / system / "relaxed"
        if not relaxed.is_dir():
            continue
        dest_rel = DEMO_DST / "mattergen" / system / "relaxed"
        dest_rel.mkdir(parents=True, exist_ok=True)
        idx_map: dict[str, str] = {}
        for cif in sorted(relaxed.glob("*.cif")):
            shutil.copy2(cif, dest_rel / cif.name)
            idx = _parse_idx_from_cif_stem(cif.stem)
            if idx is None:
                continue
            key = str(idx)
            if key not in idx_map:
                idx_map[key] = cif.name
        if idx_map:
            cif_index[system] = idx_map

    extra = RESULTS_SRC / "cobi_predictions.csv"
    if extra.is_file():
        shutil.copy2(extra, DEMO_DST / "results" / extra.name)

    (DEMO_DST / "cif-index.json").write_text(
        json.dumps(cif_index, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    systems = sorted({p.stem.replace("_mattersim_hull", "") for p in (DEMO_DST / "results").glob("*_mattersim_hull.csv")})
    (DEMO_DST / "systems.json").write_text(
        json.dumps({"systems": systems}, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"sync_static_demo_assets: {len(systems)} hull CSVs, {len(cif_index)} systems with CIFs -> {DEMO_DST}")


if __name__ == "__main__":
    main()
