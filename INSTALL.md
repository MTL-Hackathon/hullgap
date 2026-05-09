# Installation

## Requirements

- Python 3.10 or 3.11
- Git
- Optional: CUDA-capable GPU for faster MLIP inference

## Setup

```bash
git clone https://github.com/MTL-Hackathon/hullgap.git
cd hullgap

python -m venv .venv
source .venv/bin/activate

pip install -U pip setuptools wheel
pip install -e .
```

On Windows, activate the virtual environment with `.venv\Scripts\activate` instead of `source .venv/bin/activate`.

## Smoke test

After installation, confirm Python and the editable package:

```bash
python --version
python -c "import hullgap; print('hullgap import OK')"
```

You should see Python 3.10.x or 3.11.x and `hullgap import OK`.
