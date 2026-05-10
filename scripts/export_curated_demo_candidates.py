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


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out_rows: list[list[str]] = []
    hull_header: list[str] | None = None

    for csv_path in sorted(RESULTS.glob("*_mattersim_hull.csv")):
        system = csv_path.name.replace("_mattersim_hull.csv", "")
        relaxed = MATTERGEN / system / "relaxed"
        if not relaxed.is_dir():
            continue

        with csv_path.open(newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            rows = list(reader)
        if len(rows) < 2:
            continue
        hdr = rows[0]
        if hull_header is None:
            hull_header = hdr
            out_rows.append(["system", *hdr])
        elif hdr != hull_header:
            raise SystemExit(
                f"Header mismatch in {csv_path.name}: expected {hull_header!r}, got {hdr!r}"
            )

        for parts in rows[1 : 1 + MAX_ROWS]:
            if not parts or all(not c.strip() for c in parts):
                continue
            out_rows.append([system, *parts])

    with OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerows(out_rows)

    n_data = max(0, len(out_rows) - 1)
    n_systems = len({r[0] for r in out_rows[1:]}) if n_data else 0
    print(f"Wrote {OUT} with {n_data} rows across {n_systems} systems (max {MAX_ROWS} each).")
    if n_data == 0:
        print("No systems matched (need data/results/*_mattersim_hull.csv + data/mattergen/<SYS>/relaxed/).")


if __name__ == "__main__":
    main()
