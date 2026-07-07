# Random Occlusion (RO) — PadChest-GR Scripts

## Quick Reference

### Submit all 9 models at once

```bash
./scripts/ro/padchest-gr/submit_all.sh --experiment ro_p100 --seed 3
```

| Argument       | Default   | Description                          |
|----------------|-----------|--------------------------------------|
| `--experiment` | `ro_p100` | Experiment name (see table below)    |
| `--seed`       | `3`       | Random seed for bbox selection       |
| `--dry-run`    |           | Print what would run without submitting |

### Submit a single model

```bash
sbatch scripts/ro/padchest-gr/run_medgemma.sh <experiment> <seed>
```

Both positional args are optional (defaults: `ro_p100`, `3`):

```bash
sbatch scripts/ro/padchest-gr/run_medgemma.sh ro_p50 7
```

### Run locally (no SLURM)

```bash
bash scripts/ro/padchest-gr/run_medgemma.sh ro_p25 3
```

## Experiment Values

| Experiment | Noise strength |
|------------|----------------|
| `ro_p25`   | 25%            |
| `ro_p50`   | 50%            |
| `ro_p75`   | 75%            |
| `ro_p100`  | 100%           |

## Models & Scripts

| Script             | Model key          | Venv              | GPU batch (`--num-images`) |
|--------------------|--------------------|-------------------|----------------------------|
| `run_medgemma.sh`  | `medgemma`         | `.venv_RRG`       | 100                        |
| `run_maira2.sh`    | `maira-2`          | `.venv_RRG`       | 24                         |
| `run_cxrmateed.sh` | `cxrmateed`        | `.venv_RRG`       | 128                        |
| `run_nv_reason_cxr.sh` | `nv-reason-cxr-3b` | `.venv_nv`    | 128                        |
| `run_chexone.sh`   | `chexone`          | `.venv_nv`        | 128                        |
| `run_libra.sh`     | `libra`            | `.SC_Libra_venv`  | 24                         |
| `run_llavarad.sh`  | `llavarad`         | `.SC_Libra_venv`  | 24                         |
| `run_chexagent.sh` | `chexagent`        | `.venv_chexagent` | 24                         |
| `run_radialog.sh`  | `radialog`         | `.radialog_venv`  | 32                         |

## Monitor & Cancel

```bash
# Check experiment status (SLURM + output files)
./scripts/ro/status.sh

# Check running jobs
squeue -u $USER

# Follow logs
tail -f logs/ro/*.out

# Cancel all your jobs
scancel -u $USER
```

## Output

Results go to `outputs/<experiment>/<model_id>::seed=<seed>.json`.
