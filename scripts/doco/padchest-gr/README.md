# Drop Object Class Occlusion (DOCO) — PadChest-GR Scripts

DOCO is identical to OCO (Object Class Occlusion) except that **samples without any valid disease bounding boxes are dropped** from the dataset before inference. This ensures every evaluated sample actually has its disease region occluded, removing the confound of unmodified images diluting the results.

## Quick Reference

### Submit all 9 models at once

```bash
./scripts/doco/padchest-gr/submit_all.sh --experiment doco_p100 --seed 3
```

| Argument       | Default     | Description                          |
|----------------|-------------|--------------------------------------|
| `--experiment` | `doco_p100` | Experiment name (see table below)    |
| `--seed`       | `3`         | Random seed for bbox selection       |
| `--dry-run`    |             | Print what would run without submitting |

### Submit a single model

```bash
sbatch scripts/doco/padchest-gr/run_medgemma.sh <experiment> <seed>
```

Both positional args are optional (defaults: `doco_p100`, `3`):

```bash
sbatch scripts/doco/padchest-gr/run_medgemma.sh doco_p50 7
```

### Run locally (no SLURM)

```bash
bash scripts/doco/padchest-gr/run_medgemma.sh doco_p25 3
```

## Difference from OCO

| Aspect | OCO | DOCO |
|--------|-----|------|
| Occlusion | All annotated bboxes | All annotated bboxes |
| Samples without bboxes | **Kept** (image unmodified) | **Dropped** |
| Dataset size | Full category dataset | Subset with valid bboxes only |

At runtime you will see a log line like:
```
[DOCO] Dropped 42/350 samples without bboxes (308 remaining)
```

## Experiment Values

| Experiment  | Occlusion strength |
|-------------|--------------------|
| `doco_p25`  | 25%                |
| `doco_p50`  | 50%                |
| `doco_p75`  | 75%                |
| `doco_p100` | 100%               |

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
# Check experiment status (SLURM + output files)
./scripts/doco/status.sh

# Check running jobs
squeue -u $USER

# Follow logs
tail -f logs/doco/*.out

# Cancel all your jobs
scancel -u $USER
```

## Output

Results go to `outputs/doco/<pXX>/padchest-gr/<model_id>::seed=<seed>.json`.
