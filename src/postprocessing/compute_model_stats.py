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

def _math_sdp_context():
    """Return a context manager that forces PyTorch to use the math (non-Flash)
    SDPA kernel so that FlopCounterMode can instrument attention operations.

    Flash / mem-efficient attention uses custom CUDA kernels that FlopCounterMode
    cannot count; switching to the math backend makes every attention call a
    plain sequence of matmuls/softmaxes that the counter handles correctly.

    Tries the PyTorch ≥ 2.3 API first, then the 2.0–2.2 deprecated API,
    then falls back to a no-op context so the rest of the code is unaffected.
    """
    import contextlib
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel  # type: ignore
        return sdpa_kernel([SDPBackend.MATH])
    except Exception:
        pass
    try:
        import torch
        return torch.backends.cuda.sdp_kernel(  # type: ignore[attr-defined]
            enable_flash=False, enable_math=True, enable_mem_efficient=False
        )
    except Exception:
        return contextlib.nullcontext()

class _ProfilerFlopCounter:
    """Drop-in for FlopCounterMode when torch.utils.flop_counter is unavailable
    (PyTorch < 2.1).  Uses torch.profiler with with_flops=True instead."""

    def __init__(self):
        self._total_flops: int = 0
        self._prof = None

    def __enter__(self):
        import torch
        from torch.profiler import profile, ProfilerActivity  # type: ignore
        self._prof = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            with_flops=True,
        )
        self._prof.__enter__()
        return self

    def __exit__(self, *args):
        self._prof.__exit__(*args)
        self._total_flops = sum(
            int(evt.flops) for evt in self._prof.key_averages() if evt.flops
        )

    def get_total_flops(self) -> int:
        return self._total_flops


def _flop_counter():
    """Return FlopCounterMode (PyTorch ≥ 2.1) or _ProfilerFlopCounter (< 2.1).

    Both expose the same interface: use as a context manager, then call
    .get_total_flops() after the block.
    """
    try:
        from torch.utils.flop_counter import FlopCounterMode  # type: ignore
        return FlopCounterMode(display=False)
    except (ImportError, ModuleNotFoundError):
        return _ProfilerFlopCounter()


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

    raw_model = model_instance._model
    raw_model.eval()

    # Prepare inputs for a single image.
    # Index positionally: base returns (model, processor, inputs, input_len),
    # Libra returns (model, tokenizer, inputs, input_len, stop_str) — 5 values.
    # Non-standard models (e.g. CXRMate) may fail prepare_inputs entirely;
    # in that case skip directly to the 2·N·L approximation.
    try:
        prepared = model_instance.prepare_inputs(image, prompt)
        inputs = prepared[2]
        fwd_inputs = _filter_forward_kwargs(raw_model, dict(inputs))
    except Exception as exc:
        print(f"  [prepare_inputs] failed ({exc}); trying processor-based fallback.")
        fwd_inputs = None

    # Fallback for models whose processor is a torchvision Transform (e.g. CXRMate):
    # apply it directly to get pixel_values, then build a minimal forward dict.
    if fwd_inputs is None:
        try:
            import torch
            proc = model_instance._processor
            if proc is not None and callable(proc):
                pixel_values = proc(image)                      # (C, H, W)
                # CXRMate forward expects (B=1, S=1, C, H, W)
                if pixel_values.dim() == 3:
                    pixel_values = pixel_values.unsqueeze(0).unsqueeze(0)
                device = next(raw_model.parameters()).device
                raw_fwd = {"pixel_values": pixel_values.to(device)}

                # Encoder-decoder models (e.g. CXRMate) also need decoder_input_ids
                # and decoder_attention_mask for a single forward step; without the
                # mask, some models produce a None attention bias that causes
                # FlopCounterMode to crash on '.dtype'.
                cfg = getattr(raw_model, "config", None)
                if cfg is not None and getattr(cfg, "is_encoder_decoder", False):
                    start_id = (
                        getattr(cfg, "decoder_start_token_id", None)
                        or getattr(cfg, "bos_token_id", None)
                        or 0
                    )
                    raw_fwd["decoder_input_ids"] = torch.tensor([[start_id]], device=device)
                    raw_fwd["decoder_attention_mask"] = torch.ones(
                        (1, 1), dtype=torch.long, device=device
                    )

                fwd_inputs = _filter_forward_kwargs(raw_model, raw_fwd)
                print("  [processor fallback] built pixel_values input.")
        except Exception as exc2:
            print(f"  [processor fallback] failed ({exc2}); will use 2·N·L approximation.")

    # ── Method 1: FlopCounterMode (built-in, no extra deps) ──────────────
    if fwd_inputs is not None:
        try:
            # Force math SDPA so FlopCounterMode can instrument attention.
            # Flash / mem-efficient kernels are opaque to the counter and would
            # cause it to report 0 or raise an error for those ops.
            with torch.no_grad(), _math_sdp_context():
                with _flop_counter() as counter:
                    raw_model(**fwd_inputs)

            total_flops = counter.get_total_flops()
            return {
                "gflops":       round(total_flops / 1e9, 3),
                "gmacs":        round(total_flops / 2e9, 3),
                "flops_method": "FlopCounterMode",
            }
        except Exception as exc:
            print(f"  [FlopCounterMode] failed: {exc}")

    # ── Method 1b: FlopCounterMode — encoder + decoder separately ────────
    # Encoder-decoder models (e.g. CXRMate) can fail with the combined forward
    # because intermediate tensors produced by the encoder are None-initialised
    # in the combined path and trip FlopCounterMode's dtype hooks.  Running the
    # sub-modules in sequence avoids that: the encoder outputs real tensors that
    # the decoder then consumes.
    if (
        fwd_inputs is not None
        and "pixel_values" in fwd_inputs
        and hasattr(raw_model, "encoder")
    ):
        try:
            total_flops = 0

            # ── Encoder pass ──────────────────────────────────────────────
            enc_fwd = _filter_forward_kwargs(
                raw_model.encoder,
                {"pixel_values": fwd_inputs["pixel_values"]},
            )
            with torch.no_grad(), _math_sdp_context():
                with _flop_counter() as counter:
                    enc_out = raw_model.encoder(**enc_fwd)
            total_flops += counter.get_total_flops()

            # ── Decoder pass ──────────────────────────────────────────────
            if hasattr(raw_model, "decoder") and "decoder_input_ids" in fwd_inputs:
                enc_hidden = getattr(enc_out, "last_hidden_state", None)
                if enc_hidden is not None:
                    dec_fwd = _filter_forward_kwargs(
                        raw_model.decoder,
                        {
                            "input_ids": fwd_inputs["decoder_input_ids"],
                            "encoder_hidden_states": enc_hidden,
                        },
                    )
                    with torch.no_grad(), _math_sdp_context():
                        with _flop_counter() as counter:
                            raw_model.decoder(**dec_fwd)
                    total_flops += counter.get_total_flops()

            return {
                "gflops":       round(total_flops / 1e9, 3),
                "gmacs":        round(total_flops / 2e9, 3),
                "flops_method": "FlopCounterMode(enc+dec)",
            }
        except Exception as exc:
            print(f"  [FlopCounterMode enc+dec] failed: {exc}")

    # ── Method 1c: FlopCounterMode — LLaVA-style (encode_images + LLM) ──
    # RaDialog uses BioViL-T (ResNet50) as vision encoder. Its forward has a
    # non-standard signature and returns a dataclass with .patch_embeddings
    # (not .last_hidden_state), so manually splitting vision_tower + projector
    # is fragile.  encode_images() already handles the full pipeline:
    #   vision_tower(img) → .patch_embeddings → flatten → mm_projector
    # We measure it in one call, then measure the LLM text-prefill separately.
    if (
        fwd_inputs is not None
        and hasattr(model_instance, "_vis_transforms")
        and hasattr(raw_model, "encode_images")
        and "input_ids" in fwd_inputs
    ):
        try:
            total_flops = 0
            device = next(raw_model.parameters()).device
            dtype  = next(raw_model.parameters()).dtype

            # Mirror the exact preprocessing used in inference:
            # preprocess_image (remap_to_uint8 + grayscale) then vis_transforms.
            img = (
                model_instance.preprocess_image(image)
                if hasattr(model_instance, "preprocess_image")
                else image
            )
            image_tensor = (
                model_instance._vis_transforms(img)
                .unsqueeze(0)
                .to(device=device, dtype=dtype)
            )

            # ── Vision encoder + projector ─────────────────────────────────
            with torch.no_grad(), _math_sdp_context():
                with _flop_counter() as counter:
                    raw_model.encode_images(image_tensor)
            total_flops += counter.get_total_flops()

            # ── LLM text-prefill pass ──────────────────────────────────────
            # When images=None, LLaVA's forward skips the vision path and
            # runs as a pure language model on the text input_ids.
            llm_fwd = _filter_forward_kwargs(
                raw_model,
                {k: v for k, v in fwd_inputs.items()
                 if k in ("input_ids", "attention_mask", "position_ids")},
            )
            if llm_fwd.get("input_ids") is not None:
                with torch.no_grad(), _math_sdp_context():
                    with _flop_counter() as counter:
                        raw_model(**llm_fwd)
                total_flops += counter.get_total_flops()

            return {
                "gflops":       round(total_flops / 1e9, 3),
                "gmacs":        round(total_flops / 2e9, 3),
                "flops_method": "FlopCounterMode(vision+llm)",
            }
        except Exception as exc:
            print(f"  [FlopCounterMode vision+llm] failed: {exc}")

    # ── Method 1d: FlopCounterMode — LLaVA `images` param (e.g. RaDialog) ──
    # LlavaLlamaForCausalLM.forward() accepts an `images` tensor directly and
    # runs the full multimodal forward (vision_tower → mm_projector → LLM) in
    # one call.  Build inputs from _vis_transforms + _tokenizer.
    if (
        hasattr(model_instance, "_vis_transforms")
        and hasattr(model_instance, "_tokenizer")
        and "images" in inspect.signature(raw_model.forward).parameters
    ):
        try:
            device = next(raw_model.parameters()).device
            dtype  = next(raw_model.parameters()).dtype

            img = (
                model_instance.preprocess_image(image)
                if hasattr(model_instance, "preprocess_image")
                else image
            )
            image_tensor = (
                model_instance._vis_transforms(img)
                .unsqueeze(0)
                .to(device=device, dtype=dtype)
            )

            tokenizer = model_instance._tokenizer
            text_enc  = tokenizer(prompt, return_tensors="pt")
            input_ids = text_enc["input_ids"].to(device)

            llava_fwd = _filter_forward_kwargs(
                raw_model,
                {"input_ids": input_ids, "images": image_tensor},
            )

            with torch.no_grad(), _math_sdp_context():
                with _flop_counter() as counter:
                    raw_model(**llava_fwd)

            total_flops = counter.get_total_flops()
            return {
                "gflops":       round(total_flops / 1e9, 3),
                "gmacs":        round(total_flops / 2e9, 3),
                "flops_method": "FlopCounterMode(llava-images)",
            }
        except Exception as exc:
            print(f"  [FlopCounterMode llava-images] failed: {exc}")

    # ── Method 2: calflops (optional install) ─────────────────────────────
    if fwd_inputs is not None:
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
        if fwd_inputs is not None:
            # For vision encoder-decoders (e.g. CXRMate), prefer pixel_values
            # spatial extent over decoder_input_ids (which is just seq_len=1 for the
            # BOS start token and would give a wildly underestimated FLOPs figure).
            pv = fwd_inputs.get("pixel_values")
            if pv is not None and hasattr(pv, "shape") and pv.ndim >= 4:
                # shape is (B, [S,] C, H, W); H*W approximates the patch-token count
                seq_len = int(pv.shape[-1] * pv.shape[-2])
            else:
                for key in ("input_ids", "attention_mask", "decoder_input_ids"):
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

    # _ensure_loaded() returns (model, processor) for most models, but some
    # override it with extra return values (e.g. cxrmateed returns 3).
    raw_model = model_instance._ensure_loaded()[0]
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
