from ..base import BaseHFLM
from ..spec import ModelSpec


class Libra(BaseHFLM):
    pass


MODEL_SPEC = ModelSpec(
    key="libra",
    model_id="TODO",
    prompt_key="medgemma_default",
    task="image-to-text",
    supports_images=True,
)
