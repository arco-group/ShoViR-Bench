from ..base import BaseHFLM
from ..spec import ModelSpec


class Maira2(BaseHFLM):
    pass


MODEL_SPEC = ModelSpec(
    key="maira-2",
    model_id="TODO",
    prompt_key="medgemma_default",
    task="image-to-text",
    supports_images=True,
)

MODEL_CLASS = Maira2
