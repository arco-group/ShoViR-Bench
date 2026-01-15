from ..base import BaseHFLM
from ..spec import ModelSpec


class MedGemma(BaseHFLM):
    pass


MODEL_SPEC = ModelSpec(
    key="medgemma",
    model_id="google/medgemma-1.5-4b-it",
    prompt_key="medgemma_default",
    task="image-to-text",
    supports_images=True,
)

MODEL_CLASS = MedGemma