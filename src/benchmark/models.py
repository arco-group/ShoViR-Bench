from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    key: str
    model_id: str
    prompt_key: str
    task: str
    supports_images: bool = True


MODEL_SPECS = {
    "medgemma": ModelSpec(
        key="medgemma",
        model_id="google/medgemma-7b-it",
        prompt_key="medgemma_default",
        task="image-to-text",
        supports_images=True,
    )
}
