import argparse
import json

from PIL import Image
from transformers import pipeline

from ...prompts import PROMPTS
from ..base import _resolve_dtype
from . import MODEL_SPEC


def main() -> None:
    parser = argparse.ArgumentParser(description="Debug CXRMateED locally.")
    parser.add_argument("image", help="Path to an image to run.")
    parser.add_argument("--prompt", help="Override the default prompt.")
    parser.add_argument("--device", help="Device string (e.g., cuda:0).")
    parser.add_argument("--dtype", help="Torch dtype (e.g., float16).")
    parser.add_argument("--cache-dir", help="Hugging Face cache directory.")
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()

    prompt = args.prompt or PROMPTS.get(MODEL_SPEC.prompt_key, "")
    pipe_kwargs: dict[str, object] = {
        "task": MODEL_SPEC.task,
        "model": MODEL_SPEC.model_id,
        "trust_remote_code": args.trust_remote_code,
    }
    if args.cache_dir:
        pipe_kwargs["cache_dir"] = args.cache_dir
    if args.device:
        pipe_kwargs["device"] = args.device
    if args.dtype:
        pipe_kwargs["torch_dtype"] = _resolve_dtype(args.dtype)

    image = Image.open(args.image).convert("RGB")
    pipe = pipeline(**pipe_kwargs)
    output = pipe(image, prompt=prompt) if prompt else pipe(image)
    print(json.dumps(output, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
