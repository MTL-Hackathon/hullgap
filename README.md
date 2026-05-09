# HullGap Task Breakdown

## Project goal

Build a fast, credible MVP for an automated materials-discovery workflow that identifies underexplored microelectronics-relevant chemical systems, generates candidate crystal structures, relaxes them using machine-learning interatomic potentials (MLIPs), and ranks candidates by predicted thermodynamic stability relative to the current convex hull.

Core demo statement:

> Given a target chemical system, HullGap finds database voids, generates plausible candidate crystal structures by prototype transfer, relaxes them with a universal MLIP, and flags candidates predicted to be stable or near-stable relative to the existing database hull.

---

## MVP strategy

Use a **database-void-guided prototype-transfer pipeline**.

Avoid starting with a fully generative crystal model or full active-learning DFT loop. Those are scientifically attractive but too slow for a hackathon MVP.

Recommended first pipeline:

```text
Select microelectronics-relevant chemical systems
        ↓
Query Materials Project and score database sparsity
        ↓
Generate candidate structures by prototype substitution
        ↓
Deduplicate and sanity-check structures
        ↓
Relax candidates with CHGNet or another universal MLIP
        ↓
Compute formation energies and energy above hull
        ↓
Rank candidates and visualize hull updates
        ↓
Validate on Co–Bi and at least one known stable benchmark system
```

---

## Team structure: five parallel workstreams

## Person 1 — Database void finder and target selection

### Goal

Identify chemical systems where existing databases appear incomplete but where chemistry and microelectronics relevance make discovery plausible.

### Main responsibilities

* Build a script or notebook to query Materials Project for selected binary systems.
* Count database coverage per system.
* Score systems by novelty, sparsity, and relevance.
* Produce a ranked target list for candidate generation.

### Starting systems

```text
Co-Bi, Fe-Bi, Ni-Bi, Mn-Bi
Co-Sb, Co-Te, Co-Se
Ru-Bi, Mo-Bi, W-Bi
Ti-Si, Co-Si, Ni-Si
Hf-N, Zr-N, Ta-N, Ti-N
```

Co–Bi should be treated as the first benchmark because the task description explicitly motivates the problem using Co–Bi database incompleteness.

### Metrics to compute

For each chemical system:

* number of Materials Project entries,
* number of stable Materials Project phases,
* number of unique stoichiometries,
* minimum energy above hull,
* whether the system has no stable database compounds,
* whether chemically related systems contain stable compounds,
* qualitative microelectronics tag.

### Output file

```text
data/results/void_scores.csv
```

Suggested columns:

```text
system,
n_mp_entries,
n_stable_mp,
n_unique_stoich,
min_e_above_hull,
has_zero_stable_phases,
related_systems_with_stable_phases,
microelectronics_tag,
void_score
```

### Early milestone

Within the first few hours, produce a ranked table of at least 20 systems with void scores.

---

## Person 2 — Prototype library and candidate generator

### Goal

Generate plausible crystal structures for target systems as quickly as possible.

### Status: Co-Bi ordered pyxtal dataset generated

An ordered Co-Bi candidate set has been generated with `scripts/generate_cobi_dataset.py`
using pyxtal across reduced stoichiometries with `gcd(m,n)=1`, `m+n<=8`, `Z<=4`,
all space groups 1-230, and a maximum of 32 atoms per cell. To keep the repository
usable on GitHub, the 3,952 generated CIFs are stored in one zip archive:

```text
data/candidates/Co-Bi_pyxtal_dataset.zip
```

The archive contains:

```text
Co-Bi/*.cif
candidate_metadata.csv
```

The candidate metadata contract is also available as a standalone CSV:

```text
data/results/candidate_metadata.csv
```

### Main responsibilities

* Build a prototype-transfer pipeline.
* Pull structures from related chemical systems.
* Substitute elements into target systems.
* Generate candidate CIF or POSCAR files.
* Deduplicate candidates.
* Remove obviously invalid structures.

### Candidate-generation approach

Use chemically related systems as prototype sources. Examples:

```text
NiBi2 prototype → CoBi2
FeSb2 prototype → CoBi2
CoSb3 prototype → CoBi3
MnBi prototype → CoBi
RuSb2 prototype → RuBi2
```

### Rules for first pass

* Focus on binary systems first.
* Generate stoichiometries from 1:4 to 4:1.
* Keep cells small, preferably ≤40 atoms.
* Reject structures with unrealistically short interatomic distances.
* Deduplicate using structure matching.
* Save all candidate structures with clear metadata.

### Output files

Candidate structures:

```text
data/candidates/<SYSTEM>/*.cif
```

Metadata table:

```text
data/results/candidate_metadata.csv
```

Suggested columns:

```text
candidate_id,
target_system,
formula,
source_system,
source_formula,
source_material_id,
prototype_label,
n_atoms,
initial_file_path
```

### Early milestone

Generate at least 50 Co–Bi candidates and 50 candidates across two additional systems.

---

## Person 3 — MLIP relaxation pipeline

### Goal

Relax generated candidate structures and compute approximate screening energies using a universal MLIP. MLIP energies prioritise candidates for DFT validation — they do **not** prove thermodynamic stability on their own.

### Status: implemented

The relaxation pipeline is built around a model-agnostic interface that currently supports two backends:

| Model | Role | Package | Install |
|-------|------|---------|---------|
| **CHGNet** | default / primary | `chgnet` | `pip install -e .` (core dep) |
| **MACE-MP** | optional backup / cross-validation | `mace-torch` | `pip install -e ".[mace]"` |

### Code layout

```text
src/hullgap/
  relax.py                       # relax_structure() — model-agnostic entry point
  calculators/
    __init__.py                  # get_calculator(model) factory
    chgnet_calc.py               # CHGNet backend
    mace_calc.py                 # MACE-MP backend

scripts/
  relax_batch.py                 # CLI: batch-relax a directory of CIFs
```

### Quick start

Default screening run (CHGNet):

```bash
python scripts/relax_batch.py \
    --input  data/candidates/Co-Bi \
    --output data/relaxed/Co-Bi \
    --model  chgnet \
    --fmax   0.05 \
    --max-steps 300
```

Top-candidate rerun with tighter settings:

```bash
python scripts/relax_batch.py \
    --input  data/candidates/Co-Bi \
    --output data/relaxed/Co-Bi \
    --model  chgnet \
    --fmax   0.02 \
    --max-steps 1000
```

Cross-validation with MACE-MP (requires `pip install -e ".[mace]"`):

```bash
python scripts/relax_batch.py \
    --input  data/candidates/Co-Bi \
    --output data/relaxed/Co-Bi \
    --model  mace \
    --fmax   0.05 \
    --max-steps 300
```

### Relaxation settings

```text
Default screening:     fmax = 0.05 eV/Å   max_steps = 300   relax_cell = yes
Top-candidate rerun:   fmax = 0.02 eV/Å   max_steps = 1000  relax_cell = yes
```

### Output files

Relaxed structures:

```text
data/relaxed/<SYSTEM>/*.cif
```

Relaxation results CSV (one per system × model):

```text
data/results/relaxation_results_<SYSTEM>_<MODEL>.csv
```

Columns:

```text
candidate_id, formula, status, initial_file, relaxed_file,
energy_total_eV, energy_per_atom_eV, max_force_eV_A,
volume_per_atom, n_steps, model_name, error_message
```

Status values: `converged`, `max_steps_reached`, `failed_relaxation`.

### Fault tolerance

The batch script never crashes the whole run because one structure fails. Failed structures are logged with `status=failed_relaxation` and the exception message saved in `error_message`.

### Early milestone

Relax at least one known structure and one generated Co–Bi candidate end-to-end.

---

## Person 4 — Convex hull scoring and visualization

### Goal

Convert relaxed MLIP energies into stability rankings and visualizations.

### Main responsibilities

* Build the current Materials Project convex hull for each system.
* Compute formation energies for candidates.
* Compute energy above hull or distance below the existing hull.
* Insert candidates into the hull and determine whether they become new hull vertices.
* Produce hull plots and ranked candidate tables.

### Important scientific caveat

Raw MLIP energies and DFT database energies are not perfectly consistent. Therefore, report results carefully:

> Candidates are MLIP-screened predictions. They should be prioritized for DFT validation before being claimed as truly stable.

If time allows, compute two hull modes:

### Mode A — MP-reference hull

Compare candidate MLIP energies against the current Materials Project hull.

Pros:

* fast,
* intuitive,
* directly tied to the task wording.

Cons:

* mixes MLIP and DFT energy scales.

### Mode B — MLIP-recomputed hull

Evaluate MLIP energies for both existing MP structures and generated candidates, then build a hull entirely in the MLIP energy system.

Pros:

* internally consistent,
* better scientific framing.

Cons:

* requires evaluating more structures.

### Output files

Hull scores:

```text
data/results/hull_scores.csv
```

Suggested columns:

```text
candidate_id,
system,
formula,
composition_fraction,
formation_energy_eV_atom,
e_above_hull_eV_atom,
delta_to_existing_hull_eV_atom,
predicted_status,
source_prototype,
relaxed_file
```

Status labels:

```text
new_hull_vertex
near_hull
metastable
unstable
failed_relaxation
```

Figures:

```text
reports/figures/<SYSTEM>_hull.png
reports/figures/<SYSTEM>_candidate_ranking.png
```

### Early milestone

Produce one Co–Bi convex hull plot with candidate points overlaid.

---

## Person 5 — Validation, demo, and presentation

### Goal

Make the project coherent, credible, and judge-ready.

### Main responsibilities

* Own the README, project description, and presentation narrative.
* Define validation benchmarks.
* Build a simple Streamlit demo if time permits.
* Track assumptions, limitations, and scientific claims.
* Convert outputs from the other workstreams into final figures and slides.

### Validation levels

### Level 1 — Pipeline sanity benchmark

Test whether the pipeline can recover known stable structures in a system where stable phases are already known.

Example procedure:

```text
Take known stable phase from a benchmark system
pretend it is missing
generate it through prototype transfer
relax with MLIP
check whether it appears near or on the hull
```

### Level 2 — Co–Bi benchmark

Use Co–Bi as the main motivating case because it is named in the task. Show whether the workflow generates plausible low-energy Co–Bi candidates.

### Level 3 — Cross-model agreement

If time permits, run top candidates through a second MLIP and compare ranking stability.

### Demo app pages

If building a Streamlit app, keep it simple:

```text
Page 1: Target system void scores
Page 2: Candidate structures generated
Page 3: Relaxation results
Page 4: Convex hull plot
Page 5: Top candidate ranking and validation flags
```

### Output files

```text
README.md
reports/project_description.md
reports/validation_plan.md
reports/tables/top_candidates.csv
reports/figures/pipeline_diagram.png
app/streamlit_app.py
```

### Early milestone

Have a project title, one-paragraph description, and pipeline diagram ready before the technical pipeline is fully done.

---

# Shared software stack

## Core Python stack

```text
Python 3.11
pymatgen
mp-api
ASE
CHGNet
torch
numpy
scipy
pandas
matplotlib
plotly
streamlit
tqdm
pyyaml
joblib
python-dotenv
```

## Optional packages

```text
mace-torch
matgl
matbench-discovery
spglib
seekpath
phonopy
jupyterlab
ipywidgets
```

## Recommended installation

```bash
mamba create -n hullgap python=3.11 -y
mamba activate hullgap
pip install pymatgen mp-api ase chgnet pandas numpy scipy matplotlib plotly streamlit tqdm pyyaml joblib python-dotenv
```

Optional:

```bash
pip install mace-torch matgl matbench-discovery
```

---

# Repository structure

```text
hullgap/
  README.md
  environment.yml
  .env.example

  data/
    raw/
      mp_entries/
      prototypes/
    candidates/
    relaxed/
    results/

  notebooks/
    01_void_finder.ipynb
    02_candidate_generation.ipynb
    03_relaxation_debug.ipynb
    04_hull_analysis.ipynb

  src/
    hullgap/
      __init__.py
      relax.py                         # model-agnostic relaxation interface
      calculators/
        __init__.py                    # get_calculator() factory
        chgnet_calc.py                 # CHGNet backend
        mace_calc.py                   # MACE-MP backend (optional)
      dft/
        __init__.py
        dft_hull.py
        make_qe_inputs.py
        parse_qe_outputs.py
        select_candidates.py

  scripts/
    relax_batch.py                     # batch MLIP relaxation CLI
    select_dft_candidates.py
    make_qe_inputs.py
    parse_dft_results.py
    score_dft_hull.py
    run_qe_candidate.sh

  app/
    streamlit_app.py

  reports/
    figures/
    tables/
    slides_assets/
```

---

# File contracts between team members

Use CSV and CIF files as simple contracts so everyone can work in parallel.

```text
Person 1 outputs:
  data/results/void_scores.csv

Person 2 consumes:
  data/results/void_scores.csv

Person 2 outputs:
  data/results/candidate_metadata.csv
  data/candidates/*/*.cif

Person 3 consumes:
  data/results/candidate_metadata.csv
  data/candidates/*/*.cif

Person 3 outputs:
  data/results/relaxation_results.csv
  data/relaxed/*/*.cif

Person 4 consumes:
  data/results/relaxation_results.csv
  Materials Project entries

Person 4 outputs:
  data/results/hull_scores.csv
  reports/figures/*.png

Person 5 consumes:
  all result CSVs and figures
```

---

# Suggested command-line workflow

```bash
python scripts/query_systems.py \
  --systems Co-Bi Fe-Bi Ni-Bi Co-Sb Ru-Bi W-Bi Hf-N Ta-N \
  --out data/results/void_scores.csv

python scripts/generate_candidates.py \
  --target Co-Bi \
  --prototype-systems Ni-Bi Fe-Bi Mn-Bi Co-Sb Co-As \
  --outdir data/candidates/Co-Bi

python scripts/relax_batch.py \
  --input data/candidates/Co-Bi \
  --output data/relaxed/Co-Bi \
  --model chgnet \
  --fmax 0.05 \
  --max-steps 300

python scripts/score_hull.py \
  --system Co-Bi \
  --relaxed data/relaxed/Co-Bi \
  --out data/results/hull_scores_Co-Bi.csv

streamlit run app/streamlit_app.py
```

## Targeted DFT validation (after MLIP)

The MLIP pipeline ranks many candidates; **DFT is a small validation layer**, not a replacement for MLIP throughput. Only a handful of top structures receive PBE, spin-polarized **Quantum ESPRESSO** (pw.x) relaxations. Co-containing binaries can be magnetic, so `nspin=2` and `starting_magnetization` settings matter for a credible first pass. Outputs support **prioritization** for later higher-accuracy study; they are **not** final claims of thermodynamic stability.

```bash
python scripts/select_dft_candidates.py \
  --hull-scores data/results/hull_scores_Co-Bi.csv \
  --relaxed-dir data/relaxed/Co-Bi \
  --top-n 10 \
  --max-atoms 40 \
  --out dft/results/dft_candidate_list.csv

python scripts/make_qe_inputs.py \
  --candidate-list dft/results/dft_candidate_list.csv \
  --outdir dft/inputs/Co-Bi \
  --preset coarse_relax \
  --pseudo-dir /path/to/pseudopotentials/pbe

# Run QE pw.x (copy inputs to dft/runs/Co-Bi/<candidate_id>/ first):
bash scripts/run_qe_candidate.sh dft/runs/Co-Bi/candidate_demo_001 4

# After QE runs complete:
python scripts/parse_dft_results.py \
  --run-dir dft/runs/Co-Bi \
  --out dft/results/dft_energies_Co-Bi.csv

python scripts/score_dft_hull.py \
  --system Co-Bi \
  --dft-energies dft/results/dft_energies_Co-Bi.csv \
  --elemental-refs dft/reference_energies.yaml \
  --out dft/results/dft_hull_scores_Co-Bi.csv
```

Calibrate `dft/reference_energies.yaml` to the same PBE pseudopotential / ecutwfc convention as your elemental reference runs. See [`.cursor/rules/qe-setup.mdc`](.cursor/rules/qe-setup.mdc) for Quantum ESPRESSO installation instructions.

For an interactive walkthrough with plots, install `pip install -e ".[notebook]"` and open [`notebooks/05_dft_validation.ipynb`](notebooks/05_dft_validation.ipynb) (run Jupyter from the repo root).

---

# Time plan

## First 2 hours

### Person 1

* Set up Materials Project API.
* Query first 10–20 systems.
* Produce first void-score table.

### Person 2

* Pull related structures for Co–Bi prototype transfer.
* Generate first Co–Bi candidate CIFs.

### Person 3

* Install CHGNet.
* Relax one known structure and one generated candidate.

### Person 4

* Build first Materials Project hull plot for Co–Bi or another binary system.

### Person 5

* Draft project title, one-paragraph description, and pipeline diagram.

### Milestone

```text
We can query MP, generate one candidate, relax one structure, and plot one hull.
```

---

## Hours 2–6

### Person 1

* Finalize top target systems.
* Hand top 3–5 systems to candidate-generation workstream.

### Person 2

* Generate 50–300 Co–Bi candidates.
* Start candidates for 1–2 additional systems.

### Person 3

* Batch-relax Co–Bi candidates.
* Save clean relaxation result table.

### Person 4

* Compute hull scores for Co–Bi candidates.
* Create first hull plot with candidate points.

### Person 5

* Assemble first demo narrative.
* Start README and slide skeleton.

### Milestone

```text
End-to-end Co–Bi workflow:
candidate generation → MLIP relaxation → hull ranking → plot.
```

---

## Hours 6–12

### Person 1

* Expand void scoring to 20–50 systems.

### Person 2

* Generate candidates for top 3–5 systems.

### Person 3

* Continue batch relaxations.
* Rerun top candidates with tighter settings.

### Person 4

* Build summary table across systems.
* Create top-candidate plots.

### Person 5

* Refine claims, limitations, and validation story.
* Prepare demo and presentation assets.

### Milestone

```text
Ranked top-candidate table across several microelectronics-relevant systems.
```

---

## Final stretch

Priority order:

1. Make Co–Bi end-to-end result clean.
2. Add one benchmark where a known stable phase is recovered.
3. Add 2–4 additional target systems.
4. Run a second MLIP on only the top candidates if feasible.
5. Polish figures and demo.
6. Prepare final 5–10 minute presentation.

---

# Final deliverables

## Required by event

* Project title.
* One-paragraph description.
* 5–10 minute presentation with Q&A.

## Strong MVP deliverables

* Automated pipeline script or notebook.
* Database void ranking table.
* Candidate structure-generation method.
* MLIP relaxation results.
* Convex-hull comparison plot.
* Ranked top-candidate table.
* Validation benchmark.
* Clear microelectronics relevance statement.

---

# Suggested project title

**HullGap: MLIP-Guided Discovery of Missing Stable Crystal Structures for Microelectronics**

---

# Suggested one-paragraph description

**HullGap** is an automated materials-discovery workflow for identifying missing stable or near-stable crystal structures in underexplored microelectronics-relevant chemical systems. Starting from materials databases, the pipeline scores chemical systems for convex-hull voids, generates candidate structures by chemically informed prototype substitution, relaxes them with universal machine-learning interatomic potentials, and ranks candidates by predicted formation energy and energy above hull. We validate the approach on Co–Bi, a system highlighted by recent experimental evidence as missing stable phases in existing database hulls, and extend the workflow to device-relevant intermetallics, pnictides, nitrides, and contact/barrier material systems. The output is a ranked set of candidate structures that can be prioritized for DFT validation or experimental synthesis.

---

# Critical path

The critical path is:

```text
candidate structures → MLIP relaxation → hull scoring → ranked table and hull plot
```

If the team gets stuck, focus all effort on:

1. Co–Bi candidates,
2. CHGNet relaxation,
3. convex hull plot,
4. ranked table,
5. clean validation story.
