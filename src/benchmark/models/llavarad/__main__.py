from . import MODEL_SPEC, LLaVARad


def main() -> None:
    LLaVARad.run_debug_main(MODEL_SPEC, "Debug LLaVARad locally.")


if __name__ == "__main__":
    main()