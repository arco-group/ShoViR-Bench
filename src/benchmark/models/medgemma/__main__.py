from . import MODEL_SPEC, MedGemma


def main() -> None:
    MedGemma.run_debug_main(MODEL_SPEC, "Debug MedGemma locally.")


if __name__ == "__main__":
    main()

