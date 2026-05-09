#!/usr/bin/env bash
# Run QE pw.x in a single candidate directory (pw.in required).
#
# Typical workflow:
#   mkdir -p dft/runs/Co-Bi
#   cp -r dft/inputs/Co-Bi/candidate_demo_001 dft/runs/Co-Bi/
#   bash scripts/run_qe_candidate.sh dft/runs/Co-Bi/candidate_demo_001 4
#
# Requires pw.x on PATH (Quantum ESPRESSO).

set -euo pipefail

RUNDIR="${1:?Usage: $0 <run_directory> [mpi_ranks]}"
NP="${2:-4}"

if [[ ! -d "$RUNDIR" ]]; then
  echo "error: not a directory: $RUNDIR" >&2
  exit 1
fi

if [[ ! -f "$RUNDIR/pw.in" ]]; then
  echo "error: missing $RUNDIR/pw.in" >&2
  exit 1
fi

if ! command -v pw.x &>/dev/null; then
  echo "error: pw.x not found on PATH." >&2
  echo "  Install Quantum ESPRESSO and add its bin/ to PATH." >&2
  echo "  See .cursor/rules/qe-setup.mdc for instructions." >&2
  exit 1
fi

echo "Running pw.x in $RUNDIR with $NP MPI ranks..."
if command -v mpirun &>/dev/null; then
  (cd "$RUNDIR" && mpirun -np "$NP" pw.x -in pw.in > pw.out 2>&1)
else
  echo "warning: mpirun not found, running pw.x serially." >&2
  (cd "$RUNDIR" && pw.x -in pw.in > pw.out 2>&1)
fi

echo "Done. Output written to $RUNDIR/pw.out"
echo "Parse results from repo root:"
echo "  python scripts/parse_dft_results.py --run-dir dft/runs/Co-Bi --out dft/results/dft_energies_Co-Bi.csv"
