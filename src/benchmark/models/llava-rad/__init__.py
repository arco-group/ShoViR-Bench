from ..base import BaseHFLM
from ..spec import ModelSpec


class LLaVARad(BaseHFLM):
    pass


MODEL_SPEC = ModelSpec(
    key="llava-rad",
    model_id="microsoft/llava-rad",
    prompt_key="llava-rad_default",
    task="image-to-text",
    supports_images=True,
)

MODEL_CLASS = LLaVARad