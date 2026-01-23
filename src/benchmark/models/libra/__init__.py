from ..base import BaseHFLM
from ..spec import ModelSpec


class Libra(BaseHFLM):
    pass


MODEL_SPEC = ModelSpec(
    key="libra",
    model_id="X-iZhang/libra-v1.0-3b",
    prompt_key="libra_default",
    task="image-to-text",
    supports_images=True,
)

MODEL_CLASS = Libra
