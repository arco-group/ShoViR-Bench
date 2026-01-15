from ..base import BaseHFLM
from ..spec import ModelSpec


class CheXagent(BaseHFLM):
    pass


MODEL_SPEC = ModelSpec(
    key="chexagent",
    model_id="TODO",
    prompt_key="medgemma_default",
    task="image-to-text",
    supports_images=True,
)

MODEL_CLASS = CheXagent
