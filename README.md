# Tau Supersymmetry

Search for supersymmetric particles in tau lepton final states using machine learning. Built on ATLAS Run 2/Run 3 data with XGBoost and PyTorch classifiers.

## Setup

```bash
# Install dependencies
make setup

# Pull processed data (requires DVC remote)
make dvc-pull
```

Requires Python 3.13+ and [uv](https://docs.astral.sh/uv/).

## Usage

### Preprocessing

Reads ROOT ntuples, applies selection cuts, computes event weights, merges samples, and saves to parquet:

```bash
make preprocess
```

Override config via Hydra CLI:

```bash
uv run python preprocess.py analysis.channel=2 regions@analysis=sr_ch2_compressed
```

### Training

```bash
make train
```

### Full pipeline

```bash
make pipeline        # preprocess + train
make repro           # DVC-tracked reproducible run
```

### Experiment tracking

```bash
make ui              # MLflow UI at http://localhost:5000
```

## Project structure

```
.
├── configs/              # Hydra configs (features, models, regions, samples)
├── src/
│   ├── processing/       # Preprocessing pipeline (cuts, merging, I/O)
│   ├── models/           # Model definitions
│   └── visualization/    # Plotting utilities
├── tests/                # pytest test suite
├── notebooks/            # Exploratory analysis
├── data/                 # Raw and processed data (DVC-tracked)
├── preprocess.py         # Preprocessing entry point
├── train.py              # Training entry point
├── dvc.yaml              # DVC pipeline definition
└── Makefile              # Developer workflow
```

## Development

```bash
make test            # Run tests
make lint            # Run ruff
make format          # Run pre-commit hooks
make clean           # Remove caches
```
