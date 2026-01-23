from . import MODEL_SPEC, NVReasonCXR


def main() -> None:
    NVReasonCXR.run_debug_main(MODEL_SPEC, "Debug NV-Reason CXR locally.")


if __name__ == "__main__":
    main()
