from . import MODEL_SPEC, MambaXray


def main() -> None:
    MambaXray.run_debug_main(MODEL_SPEC, "Debug MambaXray locally.")


if __name__ == "__main__":
    main()
