# Evaluation Multi-Runner

Batch-evaluate all model outputs under `outputs/`, grouped by experiment and percentage level (pXX).

## Quick start

```bash
# Preview what will run
bash scripts/run_all_evals.sh --dry-run

# Submit to SLURM (all 65 files)
sbatch scripts/run_all_evals.sh

# Only a specific experiment
sbatch scripts/run_all_evals.sh --experiment ro

# With bootstrap CI and CheXbert breakdowns
sbatch scripts/run_all_evals.sh --bootstrap-ci --save-breakdowns

# Per-experiment aggregated CSV (one row per model)
sbatch scripts/run_all_evals.sh --output-mode per-experiment --parallel 8

# Run with 8 parallel workers
sbatch scripts/run_all_evals.sh --parallel 8
# Combine flags
sbatch scripts/run_all_evals.sh --experiment baseline --save-breakdowns --report-chexbert-f1
```

## Flags

### Runner-specific

| Flag | Description |
|------|-------------|
| `--dry-run` | Preview commands without running |
| `--experiment <name>` | Filter by experiment (`baseline`, `ro`, `all_noise`) |
| `--parallel <N>` | Max parallel evaluations (default: `4`) |

### Forwarded to `run_eval.py`

| Flag | Type | Default |
|------|------|---------|
| `--output-mode` | value | `per-file` |
| `--scorers` | value | `CheXbert,F1-RadGraph,BLEU-1,BLEU-4,ROUGE-L` |
| `--run-name` | value | `mimic_cxr_eval` |
| `--green-model-name` | value | `StanfordAIMI/GREEN-radllama2-7b` |
| `--bootstrap-ci` | boolean | off |
| `--save-breakdowns` | boolean | off |
| `--report-chexbert-f1` | boolean | off |
| `--compute-green` | boolean | off |

## Output structure

Results mirror the `outputs/` layout under `results/`:

```
results/
  baseline/padchest-gr/<model>.csv
  all_noise/padchest-gr/<model>.csv
  ro/p00/padchest-gr/<model>.csv
  ro/p20/padchest-gr/<model>.csv
  ...
  ro/p100/padchest-gr/<model>.csv
```

Logs go to `logs/eval/`.
