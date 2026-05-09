# Co-Bi Stability Pre-Screening Classifier

**Project:** hullgap  
**Model type:** Structure-only Gradient Boosting Classifier  
**Status:** Research prototype — validated on two held-out chemical systems

---

## 1. Purpose

Generating Co-Bi candidate crystal structures is cheap; running MLIP relaxations or DFT on all of them is not. This classifier acts as a fast first filter: given a metadata CSV and a directory of CIF files, it scores each candidate by its predicted thermodynamic stability probability and outputs a diversified ranked shortlist, along with copies of the corresponding CIFs, ready to pass directly into an MLIP relaxation workflow. The classifier uses only information that is knowable from a hypothetical crystal structure — composition, space group, and site count — so it can be applied to candidate structures before any energy calculation has been performed.

---

## 2. Pipeline Position

```
Candidate              Structure-only         MLIP               DFT
generator       -->    screen_candidates  -->  relaxation   -->   validation
(e.g. USPEX,           (this classifier)       (CHGNet,
 RandSpg, manual)                              MACE, ...)
```

The classifier sits between generation and MLIP. Its job is to shrink a pool of hundreds or thousands of candidates down to a tractable shortlist of ~50–100 before any expensive calculation runs. It does not replace MLIP or DFT — it decides which candidates are worth the compute.

---

## 3. What the Classifier Does

1. Reads a metadata CSV with one row per candidate structure
2. Validates that all referenced CIF files exist on disk
3. Derives composition and symmetry information from the metadata
4. Computes 168 structure-only features per candidate (matminer + symmetry)
5. Scores each candidate with a trained Gradient Boosting model
6. Applies diversity-aware selection to produce a shortlist that covers multiple compositions and space groups
7. Copies the shortlisted CIF files into a single output folder for the next pipeline stage

---

## 4. How to Use It

### Installation

The classifier requires Python 3.10–3.11. Install the project dependencies plus the ML stack:

```bash
pip install -e ".[notebook]"            # project deps from pyproject.toml
pip install scikit-learn matminer       # ML stack (not in pyproject.toml yet)
```

Set your Materials Project API key in a `.env` file or environment:

```bash
echo "MP_API_KEY=your_key_here" >> .env
```

### Basic command

```bash
python screen_candidates.py metadata.csv cifs/
```

### Common options

| Option | Default | Description |
|---|---|---|
| `--top-n N` | 50 | Size of final shortlist |
| `--output-dir DIR` | `screening_results/` | Where outputs are written |
| `--max-per-composition N` | 5 | Cap on how many of the same formula enter the shortlist |
| `--max-per-spacegroup N` | 8 | Cap on how many of the same space group enter the shortlist |
| `--no-diversity` | off | Skip diversity caps; take raw top-N by probability |
| `--skip-cif-copy` | off | Score only; don't copy CIFs (useful for dry runs) |
| `--model PATH` | `models/structure_only_classifier.joblib` | Trained model |
| `--feature-columns PATH` | `models/feature_columns.json` | Feature column ordering |

### Examples

```bash
# Screen with top-100 shortlist, tight diversity
python screen_candidates.py metadata.csv cifs/ --top-n 100 --max-per-composition 3

# Dry run: score only, no file copying
python screen_candidates.py metadata.csv cifs/ --skip-cif-copy --output-dir dry_run/

# No diversity constraints
python screen_candidates.py metadata.csv cifs/ --no-diversity --top-n 20
```

### Required input CSV columns

| Column | Description |
|---|---|
| `cif_filename` | Filename (with `.cif`) inside the CIF directory |
| `formula` | Full unit-cell formula (e.g. `Co2 Bi4`) |
| `space_group` | International space group number (integer) |
| `n_atoms` | Total atoms in unit cell |
| `n_Co` | Number of Co atoms |
| `n_Bi` | Number of Bi atoms |

All other columns (`volume`, `density`, `packing_fraction`, `min_CoCo`, `min_BiBi`, `min_CoBi`, `fingerprint`, etc.) are passed through to outputs unchanged.

### Output structure

```
screening_results/
├── predictions_full.csv      # All candidates, scored and ranked
├── shortlist_top_50.csv      # Diversified shortlist with rank_shortlist column
├── shortlist_cifs/           # CIF files for the shortlisted candidates
│   ├── candidate_001.cif
│   ├── candidate_002.cif
│   └── ...
├── summary.txt               # Human-readable report (same as terminal output)
├── missing_cifs.txt          # Filenames not found on disk (if any)
└── featurization_errors.csv  # Compositions that failed featurization (if any)
```

**Added columns in output CSVs:**

| Column | Description |
|---|---|
| `reduced_formula` | Pymatgen reduced formula (derived) |
| `crystal_system` | Crystal system derived from space group (derived) |
| `co_fraction` | n_Co / n_atoms (derived) |
| `p_stable` | Predicted probability of stability (model output) |
| `predicted_label` | `stable` or `unstable` at threshold p = 0.5 |
| `confidence_band` | Five-tier label (see below) |
| `rank_overall` | Rank among all candidates by p_stable |
| `rank_shortlist` | Rank within the shortlist (shortlist CSV only) |
| `cif_present` | Whether the CIF file was found on disk |

**Confidence bands:**

| Band | p(stable) range |
|---|---|
| high stable | > 0.80 |
| likely stable | 0.60 – 0.80 |
| uncertain | 0.40 – 0.60 |
| likely unstable | 0.20 – 0.40 |
| high unstable | < 0.20 |

---

## 5. Model Description

| Property | Value |
|---|---|
| Algorithm | Gradient Boosting Classifier (sklearn) |
| Estimators | 200 |
| Max depth | 3 |
| Learning rate | 0.05 |
| Subsample | 0.8 |
| Random state | 42 |
| Input features | 168 |
| Training samples | 202 (balanced from 248 queried) |
| Stability label | `energy_above_hull` ≤ 0.05 eV/atom |

### Training data

248 binary intermetallic compounds from the Materials Project across 28 chemical systems: Co paired with As, Sb, Te, Se, P, S, Sn, Ge, Si, N, O, Fe, Ni, Mn; and Bi paired with Ni, Fe, Mn, Cu, Ag, Zn, Pb, Sb, Te, Se, In, Ga, Tl; plus Co-Bi itself. The dataset was class-balanced by undersampling the majority class (unstable) to 101 samples per class.

### Feature set (168 total, zero DFT outputs)

| Feature group | Count | Source |
|---|---|---|
| ElementProperty (MAGPIE preset) | 132 | matminer — mean, range, std of elemental properties (electronegativity, atomic radius, valence electrons, Mendeleev number, melting point, etc.) |
| Stoichiometry descriptors | 6 | matminer — L2 norm, max, min of stoichiometric vector |
| Valence orbital fractions | 8 | matminer — fraction of s, p, d, f valence electrons |
| Space group number | 1 | Integer, 1–230 |
| Crystal system (one-hot) | 7 | Cubic, Hexagonal, Monoclinic, Orthorhombic, Tetragonal, Triclinic, Trigonal |
| nsites | 1 | Atoms per unit cell |
| Composition A fraction | 1 | Atomic fraction of alphabetically first element |
| n_atoms parity | 1 | 1 if odd, 0 if even |
| Stoichiometry ratio flags | 11 | Boolean flags for 1:1, 1:2, 2:1, 1:3, 3:1, 2:3, 3:2, 1:4, 4:1, 1:5, 5:1 |

The top-5 features by Gini importance are: `spacegroup_number`, `nsites`, `MagpieData mean MeltingT`, `MagpieData avg_dev GSvolume_pa`, `MagpieData mean SpaceGroupNumber`. No DFT-derived property appears in the top 20.

---

## 6. Validated Performance

| Validation set | n | Accuracy | ROC-AUC |
|---|---|---|---|
| 10-fold CV (training distribution) | 248 | — | **0.683** |
| Fe-Sb holdout (unseen chemistry) | 7 | 0.714 | **0.750** |
| Ni-Sb holdout (unseen chemistry) | 8 | 0.375 | **0.929** |

**For reference — earlier DFT-feature model** (18 features including `formation_energy_per_atom`, *cannot be used for pre-screening*):

| Validation set | n | Accuracy | ROC-AUC |
|---|---|---|---|
| 10-fold CV | 248 | — | 0.870 |
| Fe-Sb holdout | 7 | 0.857 | 0.917 |
| Ni-Sb holdout | 8 | 0.857 | 0.917 |

**Interpretation:** The Ni-Sb result is the most informative. Accuracy (0.375) looks bad but is misleading: the Ni-Sb holdout is heavily class-imbalanced (7 stable, 1 unstable), and the model's threshold-0.5 decision boundary is mis-calibrated on this out-of-distribution chemistry. The ROC-AUC of 0.929 means the model's *ranking* of the 8 compounds by stability probability is nearly perfect — it correctly places the one unstable compound near the bottom of the ranked list. For a pre-screening tool used by ranking candidates and picking the top-N, AUC is the operationally relevant metric. The accuracy gap vs. the DFT-feature model (~0.48 AUC points on CV) is the honest cost of removing formation energy as a feature.

---

## 7. Known Limitations

- **Small training set.** 248 compounds spanning 28 binary systems is modest. Prototypes common in Co-Bi but rare in Co-X/Bi-X training systems will be underrepresented.

- **Probabilities are not calibrated.** `p_stable` should be treated as a ranking score, not a literal probability. A candidate with p = 0.85 is not 85% likely to be stable — it just ranks above one with p = 0.70. Do not apply a probability threshold to gate candidates; use rank or confidence band instead.

- **Strong Pnma bias.** Without diversity caps, the naive top-50 is dominated by Pnma (sg=62) orthorhombic structures. The `--max-per-spacegroup` cap mitigates this, but Pnma candidates genuinely score highest and should be prioritised in the top tier.

- **No positional information.** Two CIF files with identical formula, space group, and nsites get identical predictions regardless of their actual atomic coordinates. The model cannot distinguish between a well-optimised geometry and a random one with the same metadata.

- **Composition bias.** The model systematically favours Bi-rich compounds (mean p(stable) ≈ 0.85 for CoBi₂–CoBi₃) and disfavours Co-rich compounds (mean p(stable) ≈ 0.11 for Co₃Bi–Co₅Bi). This is consistent with known Co-Bi phase diagram behaviour, but may suppress legitimate Co-rich candidates if the training chemistry is not fully representative.

- **Out-of-distribution chemistry.** The model was trained on Co-X and Bi-X binaries; it has never seen a Co-Bi compound during training. Predictions on Co-Bi are genuine extrapolation. The Ni-Sb and Fe-Sb holdouts suggest this generalises reasonably well in terms of ranking, but it has not been validated directly on Co-Bi compounds with known DFT energies.

---

## 8. Design Decisions

**Why structure-only features?**  
An earlier model used 18 features including `formation_energy_per_atom`, `band_gap`, and `total_magnetization` and achieved CV AUC 0.870 — substantially better. Those features are, however, DFT outputs. A hypothetical candidate structure has no formation energy until it has been calculated, which is precisely what we are trying to decide whether to do. Structure-only features are weaker but they are the only features that exist at the point of pre-screening. The performance gap (~0.19 AUC points) is the irreducible cost of operating before DFT.

**Why diversity-aware selection?**  
In test runs on 100 Co-Bi candidates, naïve top-N selection placed 8 of 20 slots in Pnma and 7 in CoBi stoichiometry — a shortlist dominated by one prototype. Diversity caps (`--max-per-composition`, `--max-per-spacegroup`) spread selections across 9 formulas and 11 space groups for the same shortlist size, giving downstream MLIP runs broader coverage of chemical space. The model's confidence scores still determine priority within each cap.

**Why 0.05 eV/atom as the hull threshold?**  
This is the standard in the materials ML literature and matches the Materials Project's own stability flag. An earlier experiment with a "gap label" scheme — dropping all compounds with `energy_above_hull` between 0.035 and 0.075 eV/atom as ambiguous and relabelling with tighter bounds — reduced training set size (202 → 182 samples) and produced worse or equivalent performance on the Fe-Sb holdout while making the model's predictions harder to interpret. The 0.05 threshold was retained.

---

## 9. File Inventory

| File | Purpose |
|---|---|
| `screen_candidates.py` | **Main CLI tool.** Takes metadata CSV + CIF directory, runs scoring, outputs shortlist + copied CIFs. |
| `save_model.py` | One-time export: reads `models/stability_classifier_structure_only.joblib` and writes the two files expected by `screen_candidates.py`. Run once after training. |
| `models/structure_only_classifier.joblib` | Trained sklearn Pipeline (StandardScaler + GradientBoostingClassifier). |
| `models/feature_columns.json` | Ordered list of 168 feature names; controls column alignment at inference time. |
| `build_structure_classifier.py` | Full training script: queries Materials Project, engineers features, trains GBC + RF, validates on Fe-Sb and Ni-Sb holdouts, scores Co-Bi candidates. Reproduces the model from scratch. |
| `build_dataset.py` | Original dataset builder used for the DFT-feature model (queries MP, engineers DFT features, balances classes). Not used by the structure-only pipeline but retained as the canonical data source. |
| `train_classifier.py` | Original classifier training for the DFT-feature model (LR/RF/GBC comparison, 80/20 split, saves best model). |
| `score_cobi_candidates.py` | Standalone scoring script for the 100-candidate Co-Bi test set with 8-section report. Predecessor to `screen_candidates.py`; retained for reference. |
| `evaluate_fesb_mnbi.py` | Holdout evaluation on Fe-Sb and Mn-Bi (Mn-Bi was found to be contaminated). |
| `validate_nisb.py` | Holdout validation on Ni-Sb with both original and gap-label models. |
| `validate_final_cosn_nipb.py` | Final holdout validation on Co-Sn (contaminated) and Ni-Pb (clean) using the endothermic model. |
| `retrain_endothermic.py` | Adds `is_endothermic` binary feature to DFT-feature model and re-evaluates on Fe-Sb. |
| `retrain_gap_label.py` | Gap-label experiment: drops ambiguous hull entries and retrains. |
| `evaluate_hard_rule.py` | Post-prediction hard rule experiment: override prediction to unstable when formation energy > 0. |
| `pyproject.toml` | Project metadata and dependencies. |
| `data/results/training_data_balanced.csv` | Balanced training set (202 rows, DFT features). |
| `data/results/training_data_structure_raw.csv` | Raw queried data for structure-only training (248 rows, no DFT features). |
| `cobi_test_combinations_100.csv` | 100 Co-Bi prototype candidates generated for initial scoring tests. |
| `data/results/cobi_predictions.csv` / `cobi_predictions_full.csv` | Scoring results for the 100-candidate test set. |

---

## 10. Reproducing the Training

```bash
# 1. Set credentials
echo "MP_API_KEY=your_key_here" >> .env

# 2. Install dependencies
pip install -e "."
pip install scikit-learn matminer

# 3. Run the full structure-only training pipeline
python build_structure_classifier.py
```

`build_structure_classifier.py` handles all five steps in sequence:
1. Queries Materials Project for 28 binary systems (~30 seconds, cached to `data/results/training_data_structure_raw.csv` on first run)
2. Engineers 168 structure-only features via matminer (~1 minute)
3. Trains GBC and RF with 10-fold CV; prints metrics and feature importances
4. Validates on Fe-Sb and Ni-Sb holdouts (two additional MP queries)
5. Scores `cobi_test_combinations_100.csv` and saves predictions

After training, export the model for `screen_candidates.py`:

```bash
python save_model.py
```

**Approximate runtime:** 3–5 minutes on a modern laptop with a fast network connection. The MP queries dominate the first run; subsequent runs use the cached CSV.

---

## 11. Next Steps for the Pipeline

- **Validate the shortlist with MLIP.** Run CHGNet or MACE relaxation on the shortlisted CIFs and compute formation energy / energy above hull. This is the ground-truth test of whether the classifier's ranking is useful.

- **Measure ranking quality against MLIP energies.** Once MLIP energies are available for a sample of candidates, compute Spearman rank correlation between `p_stable` and MLIP stability. This gives an operational measure of classifier usefulness independent of the binary accuracy metric.

- **Expand training data if shortlist quality is poor.** The most impactful improvement would be adding more training systems similar to Co-Bi (transition metal–heavy metal binaries). A training set of 1,000–2,000 compounds would substantially narrow the AUC gap with the DFT-feature model.

- **Add lattice-parameter features for a second-stage filter.** After MLIP relaxation, the relaxed volume, c/a ratio, and interatomic distances are available. A second lightweight classifier trained on these features could further refine the shortlist before DFT.

- **Calibrate probabilities.** Apply Platt scaling or isotonic regression so that `p_stable` can be interpreted as a literal probability. This requires a held-out calibration set with known labels — the MLIP energies from the step above would serve this purpose.
