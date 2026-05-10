#!/usr/bin/env python3
"""Build data/demo/candidates_curated.csv: up to MAX_ROWS hull rows per system that has MatterGen + MatterSim relaxed data."""
from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data" / "results"
MATTERGEN = ROOT / "data" / "mattergen"
OUT = ROOT / "data" / "demo" / "candidates_curated.csv"
MAX_ROWS = 20


def _candidate_paths() -> list[Path]:
    paths: list[Path] = []
    for csv_path in sorted(RESULTS.glob("*_mattersim_hull.csv")):
        system = csv_path.name.replace("_mattersim_hull.csv", "")
        relaxed = MATTERGEN / system / "relaxed"
        if relaxed.is_dir():
            paths.append(csv_path)
    return paths


def _union_fieldnames(paths: list[Path]) -> list[str]:
    """Stable union: columns appear in first-seen order across sorted files."""
    seen: set[str] = set()
    union: list[str] = []
    for csv_path in paths:
        with csv_path.open(newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            hdr = next(reader, None)
        if not hdr:
            continue
        for col in hdr:
            c = col.strip()
            if c and c not in seen:
                seen.add(c)
                union.append(c)
    return union


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    paths = _candidate_paths()
    if not paths:
        OUT.write_text("", encoding="utf-8")
        print(f"No systems matched -> empty {OUT}")
        return

    data_cols = _union_fieldnames(paths)
    fieldnames = ["system", *data_cols]

    n_written = 0
    n_systems = 0
    with OUT.open("w", newline="", encoding="utf-8") as out_f:
        writer = csv.DictWriter(
            out_f,
            fieldnames=fieldnames,
            extrasaction="ignore",
            restval="",
        )
        writer.writeheader()

        for csv_path in paths:
            system = csv_path.name.replace("_mattersim_hull.csv", "")
            n_systems += 1
            with csv_path.open(newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for i, row in enumerate(reader):
                    if i >= MAX_ROWS:
                        break
                    if not any((v or "").strip() for v in row.values()):
                        continue
                    out_row = {"system": system}
                    for k in data_cols:
                        out_row[k] = row.get(k, "") or ""
                    writer.writerow(out_row)
                    n_written += 1

    print(f"Wrote {OUT} with {n_written} rows across {n_systems} systems (max {MAX_ROWS} each).")


if __name__ == "__main__":
    main()
