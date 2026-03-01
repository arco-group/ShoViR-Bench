"""Compute efficiency statistics for a single model.

Run this script inside the model's own virtual environment to measure:
  - Total and trainable parameter count
  - Mean inference time per image (wall-clock, CUDA-synced)
  - Throughput (images / second)
  - GFLOPs and GMACs for a single prefill forward pass

FLOPs/MACs are computed using the following priority:
  1. torch.utils.flop_counter.FlopCounterMode  (PyTorch ≥ 2.0, no extra deps)
  2. calflops                                   (if installed: pip install calflops)
  3. 2 × N_params × seq_len approximation       (always works, less accurate)

Results are saved to results/compute_stats/<model_key>.json.

Usage (test with 5 synthetic images, no dataset required):
    python -m src.postprocessing.compute_model_stats \\
        --model medgemma --n-images 5 --device cuda:0 --dtype bfloat16

    # Parameter + FLOPs count only, no timing (CPU is fine):
    python -m src.postprocessing.compute_model_stats \\
        --model cxrmateed --n-images 0 --device cpu
"""
from __future__ import annotations

import argparse
import inspect
import json
import os
import time
from pathlib import Path

import numpy as np
from PIL import Image

# ── Paths ───────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "results" / "compute_stats"

# ── Model name mapping (key → display name) ─────────────────────────────────
NAME_MAP: dict[str, str] = {
    "medgemma":         "MedGemma",
    "maira-2":          "MAIRA-2",
    "cxrmateed":        "CXRMate",
    "chexagent":        "CheXagent-2",
    "chexone":          "CheXOne",
    "nv-reason-cxr-3b": "NV-Reason-CXR",
    "libra":            "LIBRA-v1",
    "llavarad":         "LIBRA-LLaVA",
    "radialog":         "RaDialog",
}


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_synthetic_images(n: int, size: tuple[int, int] = (512, 512)) -> list[Image.Image]:
    """Generate n random grayscale images (uint8, simulating CXR)."""
    rng = np.random.default_rng(42)
    images = []
    for _ in range(n):
        arr = rng.integers(50, 220, size=size, dtype=np.uint8)
        images.append(Image.fromarray(arr, mode="L").convert("RGB"))
    return images


def _count_params(model) -> tuple[int, int]:
    """Return (total_params, trainable_params)."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def _filter_forward_kwargs(raw_model, inputs: dict) -> dict:
    """Keep only kwargs accepted by raw_model.forward() to avoid TypeError."""
    try:
        sig = inspect.signature(raw_model.forward)
        valid = set(sig.parameters)
        if "kwargs" in valid or "**kwargs" in str(sig):
            return inputs  # forward accepts **kwargs → pass everything
        return {k: v for k, v in inputs.items() if k in valid}
    except (ValueError, TypeError):
        return inputs


def _compute_flops_macs(model_instance, image: Image.Image, prompt: str) -> dict:
    """
    Compute GFLOPs and GMACs for one prefill forward pass.

    Tries three methods in order:
      1. FlopCounterMode  (torch.utils.flop_counter, PyTorch ≥ 2.0)
      2. calflops         (pip install calflops)
      3. 2·N·L estimate   (Chinchilla-style, always available)

    FLOPs here count multiply-accumulate as 2 operations (standard convention).
    MACs = FLOPs / 2.
    """
    import torch

    # Prepare inputs for a single image
    _, _, inputs, _ = model_instance.prepare_inputs(image, prompt)
    raw_model = model_instance._model
    raw_model.eval()
    fwd_inputs = _filter_forward_kwargs(raw_model, dict(inputs))

    # ── Method 1: FlopCounterMode (built-in, no extra deps) ──────────────
    try:
        from torch.utils.flop_counter import FlopCounterMode  # type: ignore

        with torch.no_grad():
            with FlopCounterMode(display=False) as counter:
                raw_model(**fwd_inputs)

        total_flops = counter.get_total_flops()
        return {
            "gflops":       round(total_flops / 1e9, 3),
            "gmacs":        round(total_flops / 2e9, 3),
            "flops_method": "FlopCounterMode",
        }
    except Exception as exc:
        print(f"  [FlopCounterMode] failed: {exc}")

    # ── Method 2: calflops (optional install) ─────────────────────────────
    try:
        from calflops import calculate_flops  # type: ignore

        flops, macs, _ = calculate_flops(
            model=raw_model,
            kwargs=fwd_inputs,
            output_as_string=False,
            print_results=False,
        )
        return {
            "gflops":       round(flops / 1e9, 3),
            "gmacs":        round(macs / 1e9, 3),
            "flops_method": "calflops",
        }
    except ImportError:
        pass
    except Exception as exc:
        print(f"  [calflops] failed: {exc}")

    # ── Method 3: 2·N·L approximation ────────────────────────────────────
    # For a transformer: FLOPs ≈ 2 × N_params × seq_len
    # (Hoffmann et al., "Training Compute-Optimal Large Language Models", 2022)
    try:
        n_params = sum(p.numel() for p in raw_model.parameters())
        seq_len: int | None = None
        for key in ("input_ids", "attention_mask"):
            t = fwd_inputs.get(key)
            if t is not None and hasattr(t, "shape"):
                seq_len = int(t.shape[-1])
                break
        if seq_len is None:
            raise ValueError("Cannot determine sequence length from inputs.")

        flops = 2 * n_params * seq_len
        return {
            "gflops":       round(flops / 1e9, 3),
            "gmacs":        round(flops / 2e9, 3),
            "flops_method": f"2NL_approx(L={seq_len})",
        }
    except Exception as exc:
        print(f"  [2NL approx] failed: {exc}")

    return {"gflops": None, "gmacs": None, "flops_method": None}


# ── Main stat-collection routine ─────────────────────────────────────────────

def run_stats(
    model_key: str,
    n_images: int,
    device: str,
    dtype: str | None,
    cache_dir: str | None,
) -> dict:
    """Load the model, measure parameters / FLOPs / timing; return stats dict."""
    import torch

    from src.benchmark.models import MODEL_CLASSES, MODEL_SPECS
    from src.benchmark.prompts import PROMPTS

    spec = MODEL_SPECS[model_key]
    cls  = MODEL_CLASSES[model_key]
    prompt = PROMPTS.get(spec.prompt_key, "Describe the findings.")

    print(f"Loading model: {model_key} ({spec.model_id}) ...")
    model_instance = cls(
        name=spec.key,
        device=device,
        dtype=dtype,
        model_id=spec.model_id,
        task=spec.task,
        caching=spec.caching,
        trust_remote_code=True,
        generation_max_tokens=spec.generation_max_tokens,
        cache_dir=cache_dir or str(ROOT / ".models_cache"),
    )

    raw_model, _ = model_instance._ensure_loaded()
    n_total, n_trainable = _count_params(raw_model)
    print(f"  Parameters: {n_total / 1e9:.3f} B total  |  {n_trainable / 1e9:.3f} B trainable")

    result: dict = {
        "model_key":          model_key,
        "model_id":           spec.model_id,
        "display_name":       NAME_MAP.get(model_key, model_key),
        "n_params_total":     n_total,
        "n_params_trainable": n_trainable,
        "n_params_total_B":   round(n_total / 1e9, 4),
        "device":             device,
        "dtype":              dtype,
    }

    if n_images > 0:
        images = _make_synthetic_images(n_images)
        prompt_text = prompt

        # ── FLOPs / MACs (single forward pass on image 0) ────────────────
        print("  Computing FLOPs / MACs (single prefill forward pass) ...")
        flop_stats = _compute_flops_macs(model_instance, images[0], prompt_text)
        result.update(flop_stats)
        if flop_stats["gflops"] is not None:
            print(
                f"  GFLOPs: {flop_stats['gflops']:.1f}  |  "
                f"GMACs: {flop_stats['gmacs']:.1f}  "
                f"[method: {flop_stats['flops_method']}]"
            )
        else:
            print("  GFLOPs / GMACs: unavailable")

        # ── Inference timing (generate, N images) ────────────────────────
        print(f"  Timing {n_images} synthetic generate() calls ...")

        # 1 warm-up pass (not timed)
        try:
            model_instance([images[0]], prompt_text)
        except Exception:
            model_instance(images[0], prompt_text)

        if torch.cuda.is_available() and "cuda" in device:
            torch.cuda.synchronize()

        times: list[float] = []
        for img in images:
            t0 = time.perf_counter()
            try:
                model_instance([img], prompt_text)
            except Exception:
                model_instance(img, prompt_text)
            if torch.cuda.is_available() and "cuda" in device:
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)

        mean_t = float(np.mean(times))
        std_t  = float(np.std(times))
        result.update({
            "n_images_tested":       n_images,
            "mean_inference_time_s": round(mean_t, 4),
            "std_inference_time_s":  round(std_t, 4),
            "throughput_img_per_s":  round(1.0 / mean_t, 4),
        })
        print(f"  Throughput: {mean_t:.2f} s/image  ({1/mean_t:.2f} img/s)")

    else:
        # Parameters + FLOPs only (no generate() timing)
        images = _make_synthetic_images(1)
        print("  Computing FLOPs / MACs (single prefill forward pass) ...")
        flop_stats = _compute_flops_macs(model_instance, images[0], prompt)
        result.update(flop_stats)
        if flop_stats["gflops"] is not None:
            print(
                f"  GFLOPs: {flop_stats['gflops']:.1f}  |  "
                f"GMACs: {flop_stats['gmacs']:.1f}  "
                f"[method: {flop_stats['flops_method']}]"
            )
        result.update({
            "n_images_tested":       0,
            "mean_inference_time_s": None,
            "std_inference_time_s":  None,
            "throughput_img_per_s":  None,
        })

    return result


# ── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute efficiency stats (params, FLOPs, MACs, throughput) for one model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", required=True,
                        help="Model key (e.g. medgemma, maira-2, cxrmateed).")
    parser.add_argument("--n-images", type=int, default=5,
                        help="Synthetic images for generate() timing. "
                             "Use 0 to skip timing (FLOPs + params only).")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype",  default="bfloat16",
                        help="Torch dtype (float16, bfloat16, float32).")
    parser.add_argument("--cache-dir", default=None,
                        help="HuggingFace model cache dir.")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory for JSON. Default: results/compute_stats/")
    args = parser.parse_args()

    out_dir = Path(args.output_dir) if args.output_dir else OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("PYTHONPATH", str(ROOT))

    stats = run_stats(
        model_key=args.model,
        n_images=args.n_images,
        device=args.device,
        dtype=args.dtype,
        cache_dir=args.cache_dir,
    )

    out_path = out_dir / f"{args.model}.json"
    with out_path.open("w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nSaved: {out_path}")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
