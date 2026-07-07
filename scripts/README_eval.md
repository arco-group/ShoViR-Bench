# Evaluation Multi-Runner

Batch-evaluate all model outputs under `outputs/`, grouped by experiment and percentage level (pXX).

Two runner scripts are available:

| Script | Evaluator | Default experiments |
|--------|-----------|---------------------|
| `run_all_evals.sh` | `run_eval.py` (BLEU, ROUGE, RadGraph-F1, CheXbert) | all except `ro`, `doco`, `oco` |
| `run_all_evals_chexbert_class.sh` | `run_eval_chexbert_class.py` (CheXbert F1 per target class) | `ro`, `doco`, `oco` only |

---

## `run_all_evals.sh`

### Quick start

```bash
# Preview what will run (excludes ro/doco/oco by default)
bash scripts/run_all_evals.sh --dry-run

# Submit to SLURM
sbatch scripts/run_all_evals.sh

# Also include ro, doco, oco occlusion experiments
sbatch scripts/run_all_evals.sh --include-occlusion

# Only a specific experiment
sbatch scripts/run_all_evals.sh --experiment baseline
sbatch scripts/run_all_evals.sh --experiment ro --include-occlusion

# With bootstrap CI and CheXbert breakdowns
sbatch scripts/run_all_evals.sh --bootstrap-ci --save-breakdowns

# Per-experiment aggregated CSV (one row per model)
sbatch scripts/run_all_evals.sh --output-mode per-experiment --parallel 8

# Combine flags
sbatch scripts/run_all_evals.sh --experiment baseline --save-breakdowns --report-chexbert-f1
```

### Flags

#### Runner-specific

| Flag | Description |
|------|-------------|
| `--dry-run` | Preview commands without running |
| `--experiment <name>` | Filter by experiment (`baseline`, `ro`, `all_noise`, …) |
| `--parallel <N>` | Max parallel evaluations (default: `4`) |
| `--include-occlusion` | Also include `ro`, `doco`, `oco` experiments (excluded by default) |

#### Forwarded to `run_eval.py`

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
| `--skip-existing` | boolean | off |

---

## `run_all_evals_chexbert_class.sh`

Runs `run_eval_chexbert_class.py`, which computes **CheXbert F1 per target category** — each sample contributes only to its own target class (e.g. Edema, Atelectasis). Restricted to the occlusion experiments (`ro`, `doco`, `oco`) by default.

### Quick start

```bash
# Preview
bash scripts/run_all_evals_chexbert_class.sh --dry-run

# Submit to SLURM (all of ro + doco + oco)
sbatch scripts/run_all_evals_chexbert_class.sh

# Only one occlusion experiment
sbatch scripts/run_all_evals_chexbert_class.sh --experiment ro

# With bootstrap CI
sbatch scripts/run_all_evals_chexbert_class.sh --bootstrap-ci

# Per-experiment aggregated CSV
sbatch scripts/run_all_evals_chexbert_class.sh --output-mode per-experiment
```

### Flags

#### Runner-specific

| Flag | Description |
|------|-------------|
| `--dry-run` | Preview commands without running |
| `--experiment <name>` | Filter to one experiment (`ro`, `doco`, or `oco`) |
| `--parallel <N>` | Max parallel evaluations (default: `4`) |
| `--skip-existing` | Skip if the result CSV already exists |

#### Forwarded to `run_eval_chexbert_class.py`

| Flag | Type | Default |
|------|------|---------|
| `--output-mode` | value | `per-file` |
| `--uncertain-mode` | value | `rrg-` |
| `--n-resamples` | value | `500` |
| `--seed` | value | `3` |
| `--bootstrap-ci` | boolean | off |

---

## Output structure

Results mirror the `outputs/` layout under `results/`:

```
results/
  baseline/padchest-gr/<model>.csv
  all_noise/padchest-gr/<model>.csv
  ro/p25/padchest-gr/<model>.csv
  ro/p50/padchest-gr/<model>.csv
  ...
  doco/p25/padchest-gr/<model>.csv
  oco/p25/padchest-gr/<model>.csv
```

Logs go to `logs/eval/`.
- `run_all_evals.sh` logs: `logs/eval/eval_<N>.log`
- `run_all_evals_chexbert_class.sh` logs: `logs/eval/eval_chexbert_class_<N>.log`
