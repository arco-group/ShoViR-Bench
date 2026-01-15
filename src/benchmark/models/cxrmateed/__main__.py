from . import CXRMateED, MODEL_SPEC


def main() -> None:
    CXRMateED.run_debug_main(MODEL_SPEC, "Debug CXRMateED locally.")


if __name__ == "__main__":
    main()
