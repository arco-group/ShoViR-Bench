from ..base import BaseHFLM
from ..spec import ModelSpec


class MambaXray(BaseHFLM):
    pass


MODEL_SPEC = ModelSpec(
    key="mambaxray",
    model_id="TODO",
    prompt_key="medgemma_default",
    task="image-to-text",
    supports_images=True,
)

MODEL_CLASS = MambaXray
