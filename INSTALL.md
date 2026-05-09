# Installation

## Requirements

- Python 3.10 or 3.11
- Git
- Optional: CUDA-capable GPU for faster MLIP inference

## Dependencies

Core packages (installed automatically with `pip install -e .`):

- `pymatgen>=2024.1` — crystal structure manipulation and analysis
- `pandas>=2.0` — tabular data handling
- `numpy>=1.24` — numerical arrays
- `pyyaml>=6.0` — YAML config parsing
- `tqdm>=4.65` — progress bars
- `mp-api>=0.39` — Materials Project REST client
- `matplotlib>=3.8` — plotting (convex hull diagrams, etc.)
- `python-dotenv>=1.0` — load `.env` files

Optional MLIP packages (install with `pip install -e ".[mlip]"`):

- `mace-torch>=0.3` — MACE machine-learned interatomic potential
- `ase>=3.22` — Atomic Simulation Environment (structure I/O, dynamics)
- `torch>=1.12` — PyTorch backend for MACE

## Setup

```bash
git clone https://github.com/MTL-Hackathon/hullgap.git
cd hullgap

python -m venv .venv
source .venv/bin/activate

pip install -U pip setuptools wheel
pip install -e .

# If you need MLIP relaxation:
pip install -e ".[mlip]"
```

On Windows, activate the virtual environment with `.venv\Scripts\activate` instead of `source .venv/bin/activate`.

## Environment variables

Copy `.env.example` to `.env` and fill in your Materials Project API key:

```bash
cp .env.example .env
# then edit .env and set MP_API_KEY=<your key>
```

Get your key at https://materialsproject.org/api.

## Smoke test

After installation, confirm Python and the editable package:

```bash
python --version
python -c "import hullgap; print('hullgap import OK')"
```

You should see Python 3.10.x or 3.11.x and `hullgap import OK`.
