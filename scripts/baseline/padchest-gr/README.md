# Baseline — PadChest-GR Scripts

## Quick Reference

### Submit all 9 models at once

```bash
./scripts/baseline/padchest-gr/submit_all.sh
```

| Argument          | Description                             |
|-------------------|-----------------------------------------|
| `--dry-run`       | Print what would run without submitting |
| `--skip-existing` | Skip models whose output already exists |

### Submit a single model

```bash
sbatch scripts/baseline/padchest-gr/run_medgemma.sh
```

### Run locally (no SLURM)

```bash
bash scripts/baseline/padchest-gr/run_medgemma.sh
```

### Check experiment status

```bash
./scripts/baseline/status.sh
```

## Models & Scripts

| Script             | Model key          | Venv              | GPU batch (`--num-images`) |
|--------------------|--------------------|-------------------|----------------------------|
| `run_medgemma.sh`  | `medgemma`         | `.venv_RRG`       | 100                        |
| `run_maira2.sh`    | `maira-2`          | `.venv_RRG`       | 6                          |
| `run_cxrmateed.sh` | `cxrmateed`        | `.venv_RRG`       | 128                        |
| `run_nv_reason_cxr.sh` | `nv-reason-cxr-3b` | `.venv_nv`    | 128                        |
| `run_chexone.sh`   | `chexone`          | `.venv_nv`        | 8                          |
| `run_libra.sh`     | `libra`            | `.SC_Libra_venv`  | 24                         |
| `run_llavarad.sh`  | `llavarad`         | `.SC_Libra_venv`  | 24                         |
| `run_chexagent.sh` | `chexagent`        | `.venv_chexagent` | 40                         |
| `run_radialog.sh`  | `radialog`         | `.radialog_venv`  | 32                         |

## Monitor & Cancel

```bash
# Check running jobs
squeue -u $USER

# Check experiment status (SLURM + output files)
./scripts/baseline/status.sh

# Follow logs
tail -f logs/baseline/*.out

# Cancel all your jobs
scancel -u $USER
```

## Output

Results go to `outputs/baseline/padchest-gr/<model_id>.json`.
