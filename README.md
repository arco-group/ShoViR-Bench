# Radiology Benchmark

Minimal benchmark scaffold to run Hugging Face vision-language models on radiology images.

## Layout

- `src/benchmark/models/`: model registry and per-model packages (start with MedGemma)
- `src/benchmark/prompts.py`: prompts keyed by model
- `src/benchmark/cli.py`: CLI runner
- `src/benchmark/hf_runner.py`: HF pipeline runner

## Environment

This project targets Python 3.12.

```bash
python3.12 -m venv .venv_rrg
source .venv_rrg/bin/activate
pip install -r requirements.txt
```

If you need the NV/CUDA-specific pins, install from `requirements_nv.txt` instead.

## Usage

```bash
python src/main.py \
  --model medgemma \
  --data /path/to/images \
  --experiment baseline \
  --cache-dir ./model_caching
```

## Notes

- Images are loaded as RGB and sent to the HF `image-to-text` pipeline.
- Output is JSONL with model info, prompt, and generated text per image.
- Output path defaults to `outputs/<experiment>/<model>.jsonl` unless `--output` is set.
- Optional flags: `--device cuda:0` or `--dtype float16` for GPU runs.
- For CUDA-enabled torch, install the matching wheel from `https://pytorch.org/get-started/locally/`.
