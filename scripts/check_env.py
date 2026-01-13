import sys


def main() -> int:
    try:
        import torch
        import transformers
        from PIL import Image
    except ImportError as exc:
        print(f"Import failed: {exc}")
        return 1

    print("Python:", sys.version.split()[0])
    print("Torch:", torch.__version__)
    print("Transformers:", transformers.__version__)
    print("Pillow:", Image.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA device:", torch.cuda.get_device_name(0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
