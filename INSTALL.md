# Installation

## Requirements

- Python 3.10 or 3.11
- Git
- Optional: CUDA-capable GPU for faster MLIP inference
- Optional: Quantum ESPRESSO 7.5 for DFT validation (see below)

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
- `ase>=3.22` — Atomic Simulation Environment (structure I/O, dynamics)
- `chgnet>=0.3` — CHGNet universal MLIP (default relaxation model)
- `scipy>=1.10` — scientific computing (convex hulls, optimization)

Optional packages:

- `mace-torch>=0.3` — MACE-MP interatomic potential (install with `pip install -e ".[mace]"`)
- `jupyterlab`, `ipykernel` — notebook support (install with `pip install -e ".[notebook]"`)
- `streamlit>=1.30` — BROT web UI (install with `pip install -e ".[ui]"`)

To install everything at once: `pip install -e ".[all]"`

## Setup

```bash
git clone https://github.com/MTL-Hackathon/hullgap.git
cd hullgap

python -m venv .venv
source .venv/bin/activate

pip install -U pip setuptools wheel
pip install -e .

# Optional: Jupyter for notebooks
# pip install -e ".[notebook]"
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

After installation, confirm Python, the editable package, and CHGNet:

```bash
python --version
python -c "import hullgap; print('hullgap import OK')"
python -c "from chgnet.model.dynamics import CHGNetCalculator; print('CHGNet import OK')"
```

You should see Python 3.10.x or 3.11.x, `hullgap import OK`, and `CHGNet import OK`.

## MLIP models

### CHGNet (default, installed with `pip install -e .`)

CHGNet is a universal machine-learned interatomic potential that ships with a
pretrained model.  It is installed automatically as a core dependency.

Quick relaxation test:

```bash
python scripts/relax_batch.py \
    --input data/relaxed/Co-Bi \
    --output /tmp/hullgap_test_relax \
    --model chgnet \
    --fmax 0.05 \
    --max-steps 10
```

### MACE-MP (optional backup model)

MACE-MP is a second universal MLIP that can be used for cross-model validation.
It is **not** required for the MVP.

```bash
pip install -e ".[mace]"
python -c "from mace.calculators import mace_mp; print('MACE import OK')"
```

After installation, pass `--model mace` to `scripts/relax_batch.py`.  If
`mace-torch` is not installed, the script will print a clear error message and
exit without crashing.

## Streamlit UI — Project BROT (optional)

The BROT (**B**eyond-DFT **R**apid **O**ptimization **T**oolkit) web UI provides
an interactive interface for candidate generation and MACE validation.

```bash
pip install -e ".[ui]"
streamlit run ui/app.py
```

The app opens at `http://localhost:8501`. Select two elements, set a candidate
count, and step through the generation → selection → MACE validation workflow.

## Quantum ESPRESSO (optional, for DFT validation)

QE is only needed if you want to run actual DFT calculations after MLIP screening.

### Build dependencies (Ubuntu/Debian)

```bash
sudo apt-get install -y gfortran libopenmpi-dev libblas-dev liblapack-dev libfftw3-dev
```

### Download and build

```bash
mkdir -p ~/software && cd ~/software
curl -L -o qe-7.5-ReleasePack.tar.gz \
  "https://www.quantum-espresso.org/rdm-download/8/v7-5/c660239520162325dd0c670ba0a4b65c/qe-7.5-ReleasePack.tar.gz"
tar xzf qe-7.5-ReleasePack.tar.gz
cd qe-7.5
./configure --prefix=$HOME/software/qe-7.5-install
make -j$(nproc) pw
```

### Add to PATH

```bash
echo 'export PATH="$HOME/software/qe-7.5/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
pw.x --version   # should print "Program PWSCF v.7.5"
```

### Pseudopotentials

Download PBE PAW pseudopotentials for your target elements:

```bash
mkdir -p ~/software/pseudopotentials/pbe && cd ~/software/pseudopotentials/pbe
curl -L -O "https://pseudopotentials.quantum-espresso.org/upf_files/Co.pbe-spn-kjpaw_psl.0.3.1.UPF"
curl -L -O "https://pseudopotentials.quantum-espresso.org/upf_files/Bi.pbe-dn-kjpaw_psl.1.0.0.UPF"
```

Pass `--pseudo-dir ~/software/pseudopotentials/pbe` to `scripts/make_qe_inputs.py`.
