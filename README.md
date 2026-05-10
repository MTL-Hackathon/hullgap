---
title: HullGap
emoji: 🔬
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: 1.39.0
app_file: app/crystal_viewer.py
pinned: false
python_version: "3.11"
---

# HullGap

**MLIP-guided discovery of missing stable crystal structures in underexplored chemical systems.**

HullGap is an end-to-end materials-discovery pipeline that:

1. **Generates** candidate crystal structures for a target binary system (MatterGen diffusion or PyXtal prototype sweep).
2. **Relaxes** them with a universal machine-learning interatomic potential (MatterSim, CHGNet, or MACE-MP-0).
3. **Scores** candidates against the existing convex hull and flags candidates that sit on or below it.
4. **Validates** top picks with a second MLIP and, optionally, spin-polarized PBE in Quantum ESPRESSO.
5. **Visualizes** the results in a Next.js web app with an interactive 3D crystal viewer and convex-hull plot.

The original motivating case is **Co–Bi**, a system flagged in the literature as having missing stable phases. The repo currently ships pre-computed results for **18 binary systems** under `data/mattergen/`.

> ⚠️ **Scientific caveat.** MLIP energies are screening predictions, not stability proofs. Mixing MLIP and DFT-database energies on one hull is convenient but not internally consistent. Treat ranked candidates as priorities for DFT or experimental follow-up, not as discoveries.

---

## Quick start

### Install

```bash
mamba create -n hullgap python=3.11 -y
mamba activate hullgap
pip install -e .                      # core: CHGNet, pymatgen, ASE, PyXtal, mp-api
pip install -e ".[mace]"              # + MACE-MP-0
pip install -e ".[matter]"            # + MatterGen + MatterSim (GPU recommended)
pip install -e ".[properties]"        # + phonopy (elastic / phonon stability)
pip install -e ".[notebook]"          # + JupyterLab
pip install -e ".[all]"               # everything
```

Python is pinned to **3.10–3.11** (`mattergen` / `pyxtal` / `torch 2.2.x` compatibility).

Set up Materials Project access:

```bash
echo "MP_API_KEY=your_key_here" > .env
```

### Run the web UI

Two processes — backend + frontend.

```bash
# Backend (FastAPI on :8000)
uvicorn ui.server:app --port 8000 --reload

# Frontend (Next.js on :3000)
cd ui/frontend
npm install
npm run dev
```

Open http://localhost:3000.

### GitHub Pages (static demo)

The workflow [.github/workflows/deploy-github-pages.yml](.github/workflows/deploy-github-pages.yml) exports the Next.js UI as static HTML, bundles hull CSVs and relaxed CIFs from `data/results` and `data/mattergen`, and deploys to GitHub Pages on pushes to `main`.

**Enable Pages (required once, or deploy returns HTTP 404):**

1. Repo **Settings → Pages** (or org policy must allow Pages for this repo).
2. Under **Build and deployment**, set **Source** to **GitHub Actions** (not “Deploy from a branch”). Save if prompted.
3. Re-run the workflow (**Actions → Deploy GitHub Pages → Run workflow**) or push to `main` again.

If the **deploy** job still fails with `HttpError: Not Found` / “Ensure GitHub Pages has been enabled”, an org owner may need to allow GitHub Pages for the organization, or (for forks) Pages may be disabled—use the upstream repo or a non-fork copy. Private repos need a plan that includes GitHub Pages.

Project sites use base path `/<repository-name>` automatically. For a repository named `username.github.io`, the workflow sets an empty base path (site served at the domain root).

Local static build:

```bash
python scripts/sync_static_demo_assets.py
cd ui/frontend
NEXT_PUBLIC_STATIC_EXPORT=true NEXT_PUBLIC_BASE_PATH=/your-repo-name npm run build
# output: ui/frontend/out/
```

The static demo does not include Materials Project overlays or MACE validation (those need server-side routes or local API keys).

### Run the pipeline from the CLI

End-to-end MatterGen → MatterSim → hull for one binary:

```bash
python scripts/run_mattergen_mattersim.py \
    --element-a Co \
    --element-b Bi \
    --n-candidates 100
```

Outputs land in `data/mattergen/Co-Bi/` (CIFs + extxyz + trajectories + relaxed structures) and `data/results/Co-Bi_mattersim_hull.csv`.

---

## Architecture

```
                ┌─────────────────────────────┐
                │  Target system (e.g. Co–Bi) │
                └──────────────┬──────────────┘
                               │
           ┌───────────────────┴────────────────────┐
           │                                        │
   ┌───────▼────────┐                     ┌─────────▼──────────┐
   │   MatterGen    │                     │     PyXtal         │
   │  (diffusion)   │                     │ (prototype sweep)  │
   └───────┬────────┘                     └─────────┬──────────┘
           │                                        │
           └──────────────┬─────────────────────────┘
                          │  candidate CIFs
                  ┌───────▼────────┐
                  │  MLIP relax    │   MatterSim │ CHGNet │ MACE-MP-0
                  └───────┬────────┘
                          │  relaxed CIFs + energies
                  ┌───────▼────────┐
                  │  Hull scoring  │   formation E, e_above_hull
                  │  (hull.py)     │
                  └───────┬────────┘
                          │
        ┌─────────────────┼──────────────────┐
        │                 │                  │
 ┌──────▼──────┐  ┌───────▼───────┐  ┌───────▼────────┐
 │  Next.js UI │  │  Cross-MLIP   │  │  QE pw.x DFT   │
 │  + viewer   │  │  validation   │  │  (validation)  │
 └─────────────┘  └───────────────┘  └────────────────┘
```

---

## Repository layout

```
hullgap/
├── pyproject.toml
├── src/hullgap/
│   ├── relax.py                    # relax_structure() — model-agnostic LBFGS driver
│   ├── hull.py                     # 2D convex hull, formation energy, hull distance
│   ├── references.py               # elemental ground-state energies (cached YAML)
│   ├── calculators/
│   │   ├── __init__.py             # get_calculator(model="chgnet"|"mace"|"mattersim")
│   │   ├── chgnet_calc.py
│   │   ├── mace_calc.py
│   │   ├── mattersim_calc.py
│   │   └── relax_worker.py         # spawn-safe worker for parallel relaxations
│   ├── dft/
│   │   ├── dft_hull.py             # build hull from QE pw.x energies
│   │   ├── make_qe_inputs.py       # spin-polarized pw.x inputs (nspin=2 for Co-rich)
│   │   ├── parse_qe_outputs.py     # parse total E, magnetization, convergence
│   │   └── select_candidates.py    # rank + shortlist for DFT
│   └── properties/
│       ├── elastic.py              # stress–strain elastic tensor (Voigt-Reuss-Hill)
│       └── phonons.py              # frozen-phonon dynamical stability via phonopy
│
├── scripts/
│   ├── run_mattergen_mattersim.py  # full pipeline: generate → relax → hull → CSV/PNG
│   ├── mace_single_eval.py         # MACE-MP-0 cross-validation on Co-Bi top-20
│   ├── enrich_hull_csv.py          # add crystal_system column via SpacegroupAnalyzer
│   └── backfill_relaxed_cifs.py    # re-relax legacy MatterGen CIFs with MatterSim
│
├── notebooks/
│   ├── 02_stoichiometry_sweep.ipynb        # PyXtal prototype sweep + CHGNet hull
│   ├── 03_mlip_relaxation.ipynb            # batch relax CIFs (CHGNet or MACE)
│   ├── 04_relaxation_visualization.ipynb   # convergence diagnostics
│   ├── 05_dft_validation.ipynb             # QE workflow walkthrough
│   ├── 06_chgnet_vs_mace.ipynb             # cross-model agreement
│   ├── 06_high_pressure_verification.ipynb
│   ├── 07_mp_api_binary_survey.ipynb       # MP coverage / void scoring
│   └── 08_mattergen_mattersim_hull.ipynb   # end-to-end pipeline notebook
│
├── ui/
│   ├── server.py                   # FastAPI backend (see endpoints below)
│   └── frontend/                   # Next.js 16 + React 19 + Three.js + recharts
│
├── app/
│   └── crystal_viewer.py           # Legacy Streamlit dashboard (still works)
│
├── models/
│   ├── stability_classifier_structure_only.joblib
│   ├── structure_only_classifier.joblib
│   └── feature_columns.json
│
├── data/
│   ├── mattergen/<SYSTEM>/         # generated_crystals.extxyz, *.zip, relaxed/*.cif
│   ├── results/<SYSTEM>_mattersim_hull.csv
│   └── mattersim_references.yaml
│
├── outputs/                        # binary-survey CSVs, void-score plots, MACE shortlist
└── reports/                        # figures, tables, slides_assets
```

---

## Backends and models

| Model        | Role                                  | Extra        | Notes                              |
|--------------|---------------------------------------|--------------|------------------------------------|
| **CHGNet**   | Default screening MLIP                | core         | CPU-friendly                        |
| **MACE-MP-0**| Cross-validation MLIP                 | `[mace]`     | Slower, second opinion on top-N     |
| **MatterSim**| Primary relaxer in current pipeline   | `[matter]`   | GPU recommended                    |
| **MatterGen**| Generative diffusion model            | `[matter]`   | GPU strongly recommended            |
| **QE pw.x**  | DFT validation (PBE, spin-polarized)  | external     | See `src/hullgap/dft/`              |

`get_calculator(model="...")` returns an ASE-compatible calculator and is the single entry point used by `relax.py` and the batch scripts.

---

## Web UI

**Backend** — `ui/server.py` (FastAPI, CORS allowed for `http://localhost:3000`):

| Method | Path                  | Purpose                                                       |
|--------|-----------------------|---------------------------------------------------------------|
| POST   | `/generate`           | Generate candidates for `{element_a, element_b, n_candidates}` |
| POST   | `/validate`           | MACE-MP cross-validation of a candidate list                   |
| POST   | `/structure`          | Build a structure from a prototype + composition               |
| POST   | `/structure_by_idx`   | Load a relaxed CIF by `{system, idx}`                          |
| GET    | `/prototypes`         | List available prototype names                                 |

**Frontend** — `ui/frontend/` (Next.js App Router):

- 3D crystal viewer (`crystal-viewer.tsx`) using `@react-three/fiber`
- Convex-hull plot (`hull-chart.tsx`) using `recharts`
- Element selector / periodic-table map (`element-map.tsx`)
- Candidate / results tables, workspace shell, animated landing
- Proxy API routes under `app/api/*` forward to the FastAPI backend

The legacy Streamlit dashboard at `app/crystal_viewer.py` is still functional (`streamlit run app/crystal_viewer.py`) but the Next.js app is the primary UI.

---

## DFT validation

For high-confidence shortlisting, top MLIP candidates can be re-relaxed in **Quantum ESPRESSO** with PBE, `nspin=2`, and per-element `starting_magnetization` (Co-containing binaries can be magnetic).

```bash
python scripts/select_dft_candidates.py \
    --hull-scores data/results/Co-Bi_mattersim_hull.csv \
    --top-n 10 --max-atoms 40 \
    --out dft/results/dft_candidate_list.csv

python scripts/make_qe_inputs.py \
    --candidate-list dft/results/dft_candidate_list.csv \
    --outdir dft/inputs/Co-Bi \
    --preset coarse_relax \
    --pseudo-dir /path/to/pbe_pseudos

bash scripts/run_qe_candidate.sh dft/runs/Co-Bi/<candidate_id> 4

python scripts/parse_dft_results.py --run-dir dft/runs/Co-Bi --out dft/results/dft_energies_Co-Bi.csv
python scripts/score_dft_hull.py --system Co-Bi \
    --dft-energies dft/results/dft_energies_Co-Bi.csv \
    --elemental-refs dft/reference_energies.yaml \
    --out dft/results/dft_hull_scores_Co-Bi.csv
```

`dft/reference_energies.yaml` must use the same pseudopotentials and `ecutwfc` as the candidate runs. See `.cursor/rules/qe-setup.mdc` for QE installation notes. The walkthrough notebook is `notebooks/05_dft_validation.ipynb`.

---

## Pre-computed results

`data/mattergen/` contains MatterGen + MatterSim outputs for the following 18 binary systems (each with `generated_crystals.extxyz`, the generated CIF zip, the diffusion-trajectory zip, and a `relaxed/` directory):

```
Bi-Fe   Bi-Mn   Bi-Ru   Co-Bi   Co-Sb   Co-Si   Co-Te   Li-O    Mo-Bi
N-Hf    N-Ta    N-Zr    Ni-Bi   Ni-Si   Se-Co   Ti-N    Ti-Si   W-Bi
```

Hull CSVs for each system live alongside in `data/results/<SYSTEM>_mattersim_hull.csv` with formation energy, hull distance, and Spglib-derived crystal system. Elemental reference energies are cached in `data/mattersim_references.yaml`.

`outputs/` holds the broader binary-database survey (Materials Project, AFLOW, COD, JARVIS, NOMAD) used to score system-level voids, plus `cobi_top20_shortlist.zip` and `mace_shortlist_formation_energies.csv` from the cross-MLIP study.

---

## Limitations

- **Energy scales mix.** Default hulls compare MLIP candidate energies to MP DFT energies. Use Mode B (re-evaluate MP entries with the same MLIP) for an internally consistent hull.
- **MLIPs miss magnetism nuance.** Co-rich phases need spin-polarized DFT for credible totals.
- **Generated structures are not synthesized structures.** Stable on-paper ≠ accessible in the lab. Synthesis kinetics are out of scope.
- **18 systems are pre-computed**, but the pipeline is general — pass any `--element-a` / `--element-b` to `run_mattergen_mattersim.py`.
