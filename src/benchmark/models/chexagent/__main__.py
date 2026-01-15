from . import MODEL_SPEC, CheXagent


def main() -> None:
    CheXagent.run_debug_main(MODEL_SPEC, "Debug CheXagent locally.")


if __name__ == "__main__":
    main()
