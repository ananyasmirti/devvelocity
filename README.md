# DevVelocity

A data science and machine learning project for developer productivity modeling.

## Project Structure

- `notebooks/` - Jupyter notebooks for exploration and analysis
- `data/` - Datasets (CSV files should be added manually and are ignored by git)
- `models/` - Saved trained models
- `rbac_middleware.py` - RBAC logic for the MCP server
- `telemetry_pipeline.py` - Live telemetry ingestion and feature computation
- `mcp_server.py` - MCP server exposing telemetry and inference tools
- `train_model.py` - Train the LightGBM efficiency model

## Setup

### Requirements
- Python 3.11
- `venv` for isolation
- Homebrew installed on macOS for dependencies like `libomp`

### Installation

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### macOS extra dependency

If LightGBM fails to load with `libomp` errors, install OpenMP support:

```bash
brew install libomp
export LDFLAGS="-L/opt/homebrew/opt/libomp/lib"
export CPPFLAGS="-I/opt/homebrew/opt/libomp/include"
```

## Data

The repo ignores large CSV data files, so do not commit them.
Place the survey dataset at:

```bash
data/2024 - survey_results_public.csv
```

Then train the model.

## Training the model

```bash
source venv/bin/activate
python train_model.py
```

This will:
- load the Stack Overflow survey CSV
- engineer SPACE framework features
- train a LightGBM classifier
- save the model bundle to `models/devvelocity_model.pkl`

## Running the MCP server

```bash
source venv/bin/activate
python mcp_server.py
```

The MCP server exposes these tools:

- `get_my_metrics` — own telemetry summary
- `get_my_efficiency_score` — own efficiency score
- `get_team_aggregate` — anonymized team aggregates
- `get_developer_score` — individual score for a specified developer
- `get_raw_telemetry` — raw telemetry events for a developer
- `trigger_model_retrain` — enqueue model retraining

## Telemetry database

The active telemetry database is:

```bash
telemetry.db
```

`demo_telemetry.db` is only used for the demo helper in `telemetry_pipeline.py`.

## Notes

- `data/*.csv` is ignored by git to avoid large file pushes.
- If you need to remove a large file from history, use `git filter-branch` or `git filter-repo`.
