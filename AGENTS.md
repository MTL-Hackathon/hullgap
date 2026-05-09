# Agents

## Cursor Cloud specific instructions

### Environment

- Python **3.11** is required (`pyproject.toml` specifies `>=3.10,<3.12`). The VM's default system Python is 3.12, which is outside the supported range. Python 3.11 is installed from the deadsnakes PPA; the virtualenv at `.venv` uses it.
- Activate the virtualenv before running any command: `source /workspace/.venv/bin/activate`
- The package is installed in editable mode (`pip install -e .`), so changes under `src/hullgap/` are picked up immediately.

### Running the project

- Setup and installation steps are documented in `INSTALL.md`.
- The project is a pure-Python CLI pipeline with no web server, database, or Docker dependency.
- Four CLI scripts exist under `scripts/` (DFT validation layer). Run any of them with `--help` for usage.
- The full MLIP pipeline scripts referenced in `README.md` (e.g. `query_systems.py`, `generate_candidates.py`, `relax_batch.py`, `score_hull.py`) are not yet implemented.

### Testing

- No automated test suite exists yet. Validate changes by running the CLI scripts with `--help` and the smoke test: `python -c "import hullgap; print('OK')"`.
- For a full integration check, create synthetic data and exercise the DFT pipeline modules (`hullgap.dft.select_candidates`, `hullgap.dft.make_vasp_inputs`, `hullgap.dft.dft_hull`, `hullgap.dft.parse_vasp_outputs`).

### Lint / type-check

- No linter or formatter is configured in the repo yet. Standard `ruff` or `flake8` can be run manually against `src/` and `scripts/`.

### External services

- The Materials Project API (`mp-api`) requires an `MP_API_KEY` environment variable for live queries. The API key is not needed for the existing DFT-validation scripts, which operate on local CSV/CIF files.
