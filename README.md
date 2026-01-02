# Radiology Benchmark

Minimal benchmark scaffold to run Hugging Face vision-language models on radiology images.

## Layout

- `src/benchmark/models.py`: model registry (start with MedGemma)
- `src/benchmark/prompts.py`: prompts keyed by model
- `src/benchmark/cli.py`: CLI runner
- `src/benchmark/hf_runner.py`: HF pipeline runner

## Usage

```bash
pip install transformers pillow torch

python src/main.py \
  --model medgemma \
  --data /path/to/images \
  --output outputs/medgemma.jsonl \
  --cache-dir ./model_caching
```

## Notes

- Images are loaded as RGB and sent to the HF `image-to-text` pipeline.
- Output is JSONL with model info, prompt, and generated text per image.
- Optional flags: `--device cuda:0` or `--dtype float16` for GPU runs.
