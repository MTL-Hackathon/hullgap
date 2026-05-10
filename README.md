# HullGap

**Demo:** https://mtl-hackathon.github.io/hullgap/ — static UI over precomputed hull CSVs + CIFs (no live MP/MACE on Pages).

Hackathon MVP: generate binary crystal candidates (MatterGen or PyXtal), relax with MLIPs, score vs a convex hull, optional QE validation, Next.js viewer.

**Contributors:** Henri Höchter, Benedikt Lezius, Gregor Vollherbst, j.koch01.

## Install

Python **3.10–3.11**. Core vs extras in `pyproject.toml`.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .                 # CHGNet, pymatgen, ASE, PyXtal, mp-api, …
pip install -e ".[mace]"         # + MACE-MP-0
pip install -e ".[matter]"       # + MatterGen + MatterSim (GPU recommended)
echo "MP_API_KEY=…" > .env       # Materials Project API
```

## What to do

**Web app (full):** `uvicorn ui.server:app --port 8000` from repo root; `cd ui/frontend && npm i && npm run dev` → http://localhost:3000.

**Pipeline (example):** `python scripts/run_mattergen_mattersim.py --element-a Co --element-b Bi --n-candidates 100` → `data/mattergen/Co-Bi/`, `data/results/Co-Bi_mattersim_hull.csv`.

**Pages:** Settings → Pages → deploy from branch **`gh-pages`** / `/`. Workflow [.github/workflows/deploy-github-pages.yml](.github/workflows/deploy-github-pages.yml) builds and pushes that branch.

## Functionality

- Candidate **generation** (MatterGen diffusion, PyXtal prototypes).
- **Relaxation** and hull scoring with MLIPs (**CHGNet**, **MatterSim**, **MACE-MP-0**); cross-check / second opinion with MACE where configured.
- **Convex hull** (formation energy, energy above hull); UI tables, 3D viewer, hull chart.
- **Materials Project** phases in the UI (needs `MP_API_KEY`); survey / void work in `outputs/` also references **AFLOW**, **COD**, **JARVIS**, **NOMAD**.
- Optional **Quantum ESPRESSO** (`pw.x`) validation — see `scripts/` under `dft/` and `.cursor/rules/qe-setup.mdc`.

MLIPs approximate DFT; mixing MLIP energies with database DFT on one hull is screening-only, not a stability proof.

Repository: https://github.com/MTL-Hackathon/hullgap
