#!/usr/bin/env python3
"""
Generate a conservative ordered Co-Bi crystal candidate dataset.

The generator intentionally avoids decoration/enumeration workflows and asks
pyxtal directly for ordered Co-Bi structures over bounded stoichiometry, Z,
space-group, and trial grids. Candidates are loosely screened for distances and
packing fraction, deduplicated by a fast structural fingerprint, and streamed to
CIF plus metadata for later MLIP filtering.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import math
import multiprocessing as mp
import os
import pickle
import random
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
from pymatgen.core import Structure
from pyxtal import pyxtal
from tqdm import tqdm


MAX_REDUCED_FORMULA_ATOMS = 8
MAX_Z = 4
MAX_ATOMS_PER_CELL = 32
N_TRIALS_PER_TASK = 1
N_WORKERS = max(1, (os.cpu_count() or 1) - 1)
QUICK_TEST_TIMEOUT_SECONDS = 5
FULL_RUN_TIMEOUT_SECONDS = 10

D_MIN_COCO = 1.85
D_MIN_BIBI = 2.30
D_MIN_COBI = 1.95
PHI_MIN = 0.12
PHI_MAX = 1.10
R_NEIGHBOR = 6.0
OUTPUT_DIR = "./co_bi_dataset"

CO_RADIUS = 1.25
BI_RADIUS = 1.54
TARGET_COBI_DISTANCE = CO_RADIUS + BI_RADIUS

METADATA_COLUMNS = [
    "cif_filename",
    "formula",
    "m",
    "n",
    "z",
    "space_group",
    "trial",
    "seed",
    "n_Co",
    "n_Bi",
    "n_atoms",
    "volume",
    "density",
    "packing_fraction",
    "min_CoCo",
    "min_BiBi",
    "min_CoBi",
    "fingerprint",
]


@dataclass(frozen=True)
class Task:
    m: int
    n: int
    z: int
    sg: int
    trial: int

    @property
    def n_co(self) -> int:
        return self.m * self.z

    @property
    def n_bi(self) -> int:
        return self.n * self.z

    @property
    def n_atoms(self) -> int:
        return self.n_co + self.n_bi


def deterministic_seed(task: Task) -> int:
    text = f"Co-Bi|m={task.m}|n={task.n}|z={task.z}|sg={task.sg}|trial={task.trial}"
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") % (2**32 - 1)


def reduced_stoichiometries(max_reduced_formula_atoms: int) -> list[tuple[int, int]]:
    stoichs: list[tuple[int, int]] = []
    for m in range(1, max_reduced_formula_atoms):
        for n in range(1, max_reduced_formula_atoms):
            if m + n <= max_reduced_formula_atoms and math.gcd(m, n) == 1:
                stoichs.append((m, n))
    return stoichs


def generate_tasks(
    max_reduced_formula_atoms: int,
    max_z: int,
    max_atoms: int,
    n_trials: int,
    quick_test: bool,
) -> list[Task]:
    if quick_test:
        stoichs = [(1, 1), (1, 2), (1, 3)]
        z_values = [1, 2]
        space_groups = [1, 2, 12, 62, 63, 140, 194, 225]
        n_trials = 1
    else:
        stoichs = reduced_stoichiometries(max_reduced_formula_atoms)
        z_values = list(range(1, max_z + 1))
        space_groups = list(range(1, 231))

    tasks: list[Task] = []
    for m, n in stoichs:
        for z in z_values:
            n_atoms = (m + n) * z
            if n_atoms > max_atoms:
                continue
            for sg in space_groups:
                for trial in range(n_trials):
                    tasks.append(Task(m=m, n=n, z=z, sg=sg, trial=trial))
    return tasks


def pyxtal_from_random(task: Task, seed: int):
    random.seed(seed)
    np.random.seed(seed)

    xtal = pyxtal()
    kwargs = {
        "dim": 3,
        "group": task.sg,
        "species": ["Co", "Bi"],
        "numIons": [task.n_co, task.n_bi],
    }
    signature = inspect.signature(xtal.from_random)
    for seed_kw in ("random_state", "seed"):
        if seed_kw in signature.parameters:
            kwargs[seed_kw] = seed
            break
    xtal.from_random(**kwargs)
    return xtal


def to_pymatgen_structure(xtal) -> Structure:
    if hasattr(xtal, "to_pymatgen"):
        structure = xtal.to_pymatgen()
    elif hasattr(xtal, "to_pymatgen_structure"):
        structure = xtal.to_pymatgen_structure()
    else:
        raise RuntimeError("pyxtal object does not expose a pymatgen conversion method")
    if not isinstance(structure, Structure):
        structure = Structure.from_sites(structure.sites)
    return structure


def scale_to_target_cobi_distance(structure: Structure) -> Structure:
    min_cobi = minimum_pair_distance(structure, "Co", "Bi")
    if not math.isfinite(min_cobi) or min_cobi <= 0:
        return structure.copy()
    scale = TARGET_COBI_DISTANCE / min_cobi
    new_volume = structure.volume * scale**3
    scaled = structure.copy()
    scaled.scale_lattice(new_volume)
    return scaled


def minimum_pair_distance(structure: Structure, species_a: str, species_b: str) -> float:
    species = [site.specie.symbol for site in structure]
    distances = structure.distance_matrix
    best = math.inf
    for i, sp_i in enumerate(species):
        if sp_i != species_a:
            continue
        start = i + 1 if species_a == species_b else 0
        for j in range(start, len(species)):
            if i == j or species[j] != species_b:
                continue
            d = float(distances[i, j])
            if d > 1e-8 and d < best:
                best = d
    return best


def packing_fraction(structure: Structure) -> float:
    sphere_volume = 0.0
    for site in structure:
        radius = CO_RADIUS if site.specie.symbol == "Co" else BI_RADIUS
        sphere_volume += (4.0 / 3.0) * math.pi * radius**3
    return sphere_volume / structure.volume


def prefilter_metrics(structure: Structure) -> tuple[bool, dict[str, float]]:
    min_coco = minimum_pair_distance(structure, "Co", "Co")
    min_bibi = minimum_pair_distance(structure, "Bi", "Bi")
    min_cobi = minimum_pair_distance(structure, "Co", "Bi")
    phi = packing_fraction(structure)

    ok = (
        (not math.isfinite(min_coco) or min_coco >= D_MIN_COCO)
        and (not math.isfinite(min_bibi) or min_bibi >= D_MIN_BIBI)
        and math.isfinite(min_cobi)
        and min_cobi >= D_MIN_COBI
        and PHI_MIN <= phi <= PHI_MAX
    )
    return ok, {
        "min_CoCo": min_coco,
        "min_BiBi": min_bibi,
        "min_CoBi": min_cobi,
        "packing_fraction": phi,
    }


def fmt_float(value: float) -> str:
    if not math.isfinite(value):
        return ""
    return f"{value:.6f}"


def fast_fingerprint(structure: Structure, task: Task) -> str:
    species = [site.specie.symbol for site in structure]
    distances = structure.distance_matrix
    pair_bins: list[str] = []
    for i in range(len(structure)):
        for j in range(i + 1, len(structure)):
            d = float(distances[i, j])
            if d <= R_NEIGHBOR:
                pair = "".join(sorted((species[i], species[j])))
                pair_bins.append(f"{pair}:{d:.2f}")
    pair_bins.sort()

    lattice = structure.lattice
    lattice_part = (
        f"a={lattice.a:.2f}|b={lattice.b:.2f}|c={lattice.c:.2f}|"
        f"al={lattice.alpha:.1f}|be={lattice.beta:.1f}|ga={lattice.gamma:.1f}"
    )
    payload = "|".join(
        [
            f"Co{task.n_co}Bi{task.n_bi}",
            f"sg{task.sg}",
            lattice_part,
            ";".join(pair_bins),
        ]
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def generate_candidate_no_timeout(task: Task) -> dict:
    seed = deterministic_seed(task)
    try:
        xtal = pyxtal_from_random(task, seed)
        structure = to_pymatgen_structure(xtal)
        structure = scale_to_target_cobi_distance(structure)
        ok, metrics = prefilter_metrics(structure)
        if not ok:
            return {"status": "rejected", "task": task, "seed": seed, "metrics": metrics}

        fingerprint = fast_fingerprint(structure, task)
        formula = f"Co{task.n_co}Bi{task.n_bi}"
        cif_filename = (
            f"Co{task.n_co}_Bi{task.n_bi}_m{task.m}_n{task.n}_"
            f"z{task.z}_sg{task.sg:03d}_trial{task.trial:03d}_{fingerprint[:10]}.cif"
        )
        metadata = {
            "cif_filename": cif_filename,
            "formula": formula,
            "m": task.m,
            "n": task.n,
            "z": task.z,
            "space_group": task.sg,
            "trial": task.trial,
            "seed": seed,
            "n_Co": task.n_co,
            "n_Bi": task.n_bi,
            "n_atoms": task.n_atoms,
            "volume": fmt_float(structure.volume),
            "density": fmt_float(structure.density),
            "packing_fraction": fmt_float(metrics["packing_fraction"]),
            "min_CoCo": fmt_float(metrics["min_CoCo"]),
            "min_BiBi": fmt_float(metrics["min_BiBi"]),
            "min_CoBi": fmt_float(metrics["min_CoBi"]),
            "fingerprint": fingerprint,
        }
        return {
            "status": "accepted",
            "task": task,
            "seed": seed,
            "fingerprint": fingerprint,
            "cif_filename": cif_filename,
            "cif": structure.to(fmt="cif"),
            "metadata": metadata,
        }
    except Exception as exc:
        return {
            "status": "error",
            "task": task,
            "seed": seed,
            "error": repr(exc),
            "traceback": traceback.format_exc(),
        }


def worker_loop(worker_id: int, task_queue, result_queue) -> None:
    while True:
        item = task_queue.get()
        if item is None:
            return

        task_id, task = item
        try:
            result = generate_candidate_no_timeout(task)
        except BaseException as exc:
            result = {
                "status": "error",
                "task": task,
                "seed": deterministic_seed(task),
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            }
        result["task_id"] = task_id
        result["worker_id"] = worker_id
        result_queue.put(result)


def load_seen_fingerprints(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open("rb") as handle:
        loaded = pickle.load(handle)
    return set(loaded)


def save_seen_fingerprints(path: Path, seen: set[str]) -> None:
    tmp_path = path.with_suffix(".tmp")
    with tmp_path.open("wb") as handle:
        pickle.dump(seen, handle, protocol=pickle.HIGHEST_PROTOCOL)
    tmp_path.replace(path)


def existing_metadata_fingerprints(metadata_path: Path) -> set[str]:
    if not metadata_path.exists():
        return set()
    with metadata_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return {row["fingerprint"] for row in reader if row.get("fingerprint")}


def append_metadata_row(metadata_path: Path, row: dict) -> None:
    write_header = not metadata_path.exists() or metadata_path.stat().st_size == 0
    with metadata_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=METADATA_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def append_error(error_path: Path, result: dict) -> None:
    task: Task = result["task"]
    with error_path.open("a", encoding="utf-8") as handle:
        handle.write(
            f"[{datetime.now(timezone.utc).isoformat()}] "
            f"m={task.m} n={task.n} z={task.z} sg={task.sg} trial={task.trial} "
            f"seed={result.get('seed')} error={result.get('error')}\n"
        )
        if result.get("traceback"):
            handle.write(result["traceback"])
            handle.write("\n")


def start_worker(ctx, worker_id: int, result_queue):
    task_queue = ctx.Queue()
    process = ctx.Process(target=worker_loop, args=(worker_id, task_queue, result_queue))
    process.start()
    return {"process": process, "queue": task_queue}


def stop_worker(worker: dict) -> None:
    process = worker["process"]
    if process.is_alive():
        try:
            worker["queue"].put(None)
        except Exception:
            pass
        process.join(1)
    if process.is_alive():
        process.terminate()
        process.join(2)
    if process.is_alive():
        process.kill()
        process.join(2)


def iter_results(tasks: list[Task], workers: int, timeout_seconds: int) -> Iterable[dict]:
    if not tasks:
        return

    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    worker_count = min(max(1, workers), len(tasks))
    worker_state = {
        worker_id: start_worker(ctx, worker_id, result_queue) for worker_id in range(worker_count)
    }
    running: dict[int, dict] = {}
    next_task_id = 0
    completed = 0

    def assign(worker_id: int) -> None:
        nonlocal next_task_id
        if next_task_id >= len(tasks):
            return
        task = tasks[next_task_id]
        worker_state[worker_id]["queue"].put((next_task_id, task))
        running[worker_id] = {
            "task_id": next_task_id,
            "task": task,
            "started": time.monotonic(),
        }
        next_task_id += 1

    try:
        for worker_id in range(worker_count):
            assign(worker_id)

        while completed < len(tasks):
            try:
                result = result_queue.get(timeout=0.1)
            except Exception:
                result = None

            if result is not None:
                worker_id = result.get("worker_id")
                current = running.get(worker_id)
                if current and result.get("task_id") == current["task_id"]:
                    running.pop(worker_id, None)
                    completed += 1
                    yield result
                    assign(worker_id)

            now = time.monotonic()
            for worker_id, current in list(running.items()):
                process = worker_state[worker_id]["process"]
                if not process.is_alive():
                    running.pop(worker_id, None)
                    stop_worker(worker_state[worker_id])
                    worker_state[worker_id] = start_worker(ctx, worker_id, result_queue)
                    completed += 1
                    task = current["task"]
                    yield {
                        "status": "error",
                        "task": task,
                        "seed": deterministic_seed(task),
                        "error": f"generation worker exited with code {process.exitcode}",
                        "traceback": "",
                    }
                    assign(worker_id)
                    continue

                if now - current["started"] > timeout_seconds:
                    task = current["task"]
                    running.pop(worker_id, None)
                    stop_worker(worker_state[worker_id])
                    worker_state[worker_id] = start_worker(ctx, worker_id, result_queue)
                    completed += 1
                    yield {
                        "status": "error",
                        "task": task,
                        "seed": deterministic_seed(task),
                        "error": f"generation timeout after {timeout_seconds}s",
                        "traceback": "",
                    }
                    assign(worker_id)
    finally:
        for worker in worker_state.values():
            stop_worker(worker)


def write_summary(summary_path: Path, args: argparse.Namespace, counts: dict, elapsed_s: float) -> None:
    lines = [
        "Co-Bi dataset generation summary",
        f"finished_utc: {datetime.now(timezone.utc).isoformat()}",
        f"output_dir: {Path(args.output_dir).resolve()}",
        f"quick_test: {args.quick_test}",
        f"max_reduced_formula_atoms: {args.max_reduced_formula_atoms}",
        f"max_z: {args.max_z}",
        f"max_atoms: {args.max_atoms}",
        f"trials: {args.trials}",
        f"workers: {args.workers}",
        f"tasks_total: {counts['tasks_total']}",
        f"accepted_written: {counts['accepted_written']}",
        f"duplicates_skipped: {counts['duplicates_skipped']}",
        f"prefilter_rejected: {counts['prefilter_rejected']}",
        f"errors: {counts['errors']}",
        f"elapsed_seconds: {elapsed_s:.2f}",
        "",
        "Scientific bounds",
        "system: Co-Bi only",
        "ordered_structures_only: true",
        "partial_occupancy: false",
        "space_groups: 1-230",
        "MAX_Z: 4",
        "MAX_ATOMS_PER_CELL: 32",
        "EnumerateStructureTransformation: not used",
    ]
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate ordered Co-Bi crystal candidates with pyxtal."
    )
    parser.add_argument("--quick-test", action="store_true", help="Run the bounded smoke-test grid.")
    parser.add_argument("--trials", type=int, default=N_TRIALS_PER_TASK, help="Trials per valid task.")
    parser.add_argument("--workers", type=int, default=N_WORKERS, help="Multiprocessing workers.")
    parser.add_argument("--output-dir", default=OUTPUT_DIR, help="Dataset output directory.")
    parser.add_argument("--max-z", type=int, default=MAX_Z, help="Maximum Z multiplier.")
    parser.add_argument(
        "--max-reduced-formula-atoms",
        type=int,
        default=MAX_REDUCED_FORMULA_ATOMS,
        help="Maximum m+n for reduced Co_m Bi_n stoichiometries.",
    )
    parser.add_argument("--max-atoms", type=int, default=MAX_ATOMS_PER_CELL, help="Maximum atoms per cell.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.quick_test:
        args.trials = 1
        args.output_dir = "co_bi_dataset_quicktest"

    if args.max_z > MAX_Z:
        raise ValueError(f"--max-z may not exceed the scientific bound MAX_Z={MAX_Z}")
    if args.max_atoms > MAX_ATOMS_PER_CELL:
        raise ValueError(
            f"--max-atoms may not exceed the scientific bound MAX_ATOMS_PER_CELL={MAX_ATOMS_PER_CELL}"
        )
    if args.max_reduced_formula_atoms > MAX_REDUCED_FORMULA_ATOMS:
        raise ValueError(
            "--max-reduced-formula-atoms may not exceed the scientific bound "
            f"{MAX_REDUCED_FORMULA_ATOMS}"
        )
    if args.trials < 1:
        raise ValueError("--trials must be at least 1")
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")

    start = time.time()
    output_dir = Path(args.output_dir)
    cifs_dir = output_dir / "cifs"
    metadata_path = output_dir / "metadata.csv"
    error_path = output_dir / "errors.log"
    checkpoint_path = output_dir / "checkpoint_seen_fingerprints.pkl"
    summary_path = output_dir / "run_summary.txt"
    cifs_dir.mkdir(parents=True, exist_ok=True)
    error_path.touch(exist_ok=True)

    tasks = generate_tasks(
        max_reduced_formula_atoms=args.max_reduced_formula_atoms,
        max_z=args.max_z,
        max_atoms=args.max_atoms,
        n_trials=args.trials,
        quick_test=args.quick_test,
    )

    seen = load_seen_fingerprints(checkpoint_path)
    seen.update(existing_metadata_fingerprints(metadata_path))

    counts = {
        "tasks_total": len(tasks),
        "accepted_written": 0,
        "duplicates_skipped": 0,
        "prefilter_rejected": 0,
        "errors": 0,
    }

    print(f"Generated {len(tasks)} tasks; writing to {output_dir}", flush=True)
    print(f"Loaded {len(seen)} seen fingerprints for resume", flush=True)
    timeout_seconds = QUICK_TEST_TIMEOUT_SECONDS if args.quick_test else FULL_RUN_TIMEOUT_SECONDS
    print(f"Per-generation timeout: {timeout_seconds}s", flush=True)

    progress = tqdm(total=len(tasks), unit="task", dynamic_ncols=True)
    for idx, result in enumerate(iter_results(tasks, args.workers, timeout_seconds), start=1):
        progress.update(1)
        status = result["status"]
        if status == "accepted":
            fingerprint = result["fingerprint"]
            if fingerprint in seen:
                counts["duplicates_skipped"] += 1
            else:
                seen.add(fingerprint)
                cif_path = cifs_dir / result["cif_filename"]
                cif_path.write_text(result["cif"], encoding="utf-8")
                append_metadata_row(metadata_path, result["metadata"])
                counts["accepted_written"] += 1
                if counts["accepted_written"] % 25 == 0:
                    save_seen_fingerprints(checkpoint_path, seen)
        elif status == "rejected":
            counts["prefilter_rejected"] += 1
        elif status == "error":
            counts["errors"] += 1
            append_error(error_path, result)
        else:
            counts["errors"] += 1
            append_error(error_path, {**result, "error": f"unknown status {status!r}"})

        if idx % 100 == 0 or idx == len(tasks):
            print(
                f"{idx}/{len(tasks)} done | accepted={counts['accepted_written']} "
                f"dupes={counts['duplicates_skipped']} rejected={counts['prefilter_rejected']} "
                f"errors={counts['errors']}",
                flush=True,
            )
    progress.close()

    save_seen_fingerprints(checkpoint_path, seen)
    write_summary(summary_path, args, counts, time.time() - start)
    print(f"Finished. CIFs: {counts['accepted_written']} | metadata: {metadata_path}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        raise SystemExit(130)
