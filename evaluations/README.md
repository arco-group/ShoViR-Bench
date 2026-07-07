# Setup

> **Deprecated.** Metrics computation has moved to [radscore](https://github.com/fruffini/radscore), a standalone package. See the [Evaluation Metrics](../README.md#evaluation-metrics) section of the top-level README for install and usage instructions. The pipeline below is kept only for reference during the migration.

This project is tested with **Python 3.10.4**. 

## 1) Create and activate a virtual environment (Python 3.10.4)

module load Python/3.10.4-GCCcore-11.3.0


```bash
module purge
module load Python/3.10.4-GCCcore-11.3.0
source .venv_eval/bin/activate

python evaluations/run_eval.py --filepath outputs/baseline/padchest-gr/google__medgemma-1.5-4b-it_medgemma_default.json --output-mode per-file
python evaluations/run_eval.py --filepath outputs/baseline/padchest-gr/aehrc__cxrmate-rrg24_cxrmateed_default.json --output-mode per-file

python evaluations/run_eval.py --filepath outputs/baseline/padchest-gr/Chantal__RaDialog-interactive-radiology-report-generation_radialog_default.json --output-mode per-file

python evaluations/run_eval.py --filepath outputs/baseline/padchest-gr/microsoft__maira-2_maira2_default.json --output-mode per-file

python evaluations/run_eval.py --filepath outputs/baseline/padchest-gr/nvidia__NV-Reason-CXR-3B_nv_reason_default.json --output-mode per-file

python evaluations/run_eval.py --filepath outputs/baseline/padchest-gr/StanfordAIMI__CheXagent-2-3b-srrg-findings_chexagent_default.json --output-mode per-file


python evaluations/run_eval.py --filepath outputs/baseline/padchest-gr/google__medgemma-1.5-4b-it_medgemma_default.json



```

> On HPC systems, you may need to load the correct Python module first (e.g., `module load Python/3.10.4`).

---

## 2) Add GREEN as a git submodule

We use Stanford AIMI's **GREEN** as a submodule:

```bash
git submodule add https://github.com/Stanford-AIMI/GREEN.git third_party/GREEN
git submodule update --init --recursive
```

If you cloned this repository without submodules, run:

```bash
git submodule update --init --recursive
```

---

## 3) Lighten `GREEN/setup.py` (recommended for easier installs)

In some environments, `GREEN` may declare overly strict constraints or heavy dependencies that are not needed for our usage.

Recommended approach:
- Open `third_party/GREEN/setup.py`
- **Remove overly strict `Requires-Python` pins** (e.g., `==3.12.1`) and/or unnecessary heavyweight dependencies
- Keep only what is required for the scoring functionality used in this project

> We intentionally keep GREEN as a submodule so we can vendor minimal changes while still tracking upstream updates.

---

## 4) Install requirements

Install this project requirements:

```bash
python -m pip install -r requirements.txt
```

If you also want to install GREEN in editable mode (so imports work without touching `PYTHONPATH`):

```bash
python -m pip install -e third_party/GREEN
```

---

## Notes / Troubleshooting

- Prefer `python -m pip ...` over `pip ...` to avoid mixing interpreters.
- If you see shared library errors like `libpythonX.Y.so not found`, ensure you are using the same Python module/version you used to create the virtual environment.
- If you update the GREEN submodule to a new commit, remember to commit the updated submodule pointer:

  ```bash
  cd third_party/GREEN
  git fetch
  git checkout <commit-or-tag>
  cd ../..
  git add third_party/GREEN
  git commit -m "Bump GREEN submodule"
  ```