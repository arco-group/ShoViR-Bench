from ..base import BaseHFLM
from ..spec import ModelSpec


class MedGemma(BaseHFLM):
    def build_chat_inputs(
        self,
        processor,
        image,
        prompt_text: str,
        *,
        user_text: str = "Analyze this image.",
        device: str | None = None,
        torch_dtype=None,
    ):
        return super().build_chat_inputs(
            processor,
            image,
            prompt_text,
            user_text=user_text,
            device=device,
            torch_dtype=torch_dtype,
        )

MODEL_SPEC = ModelSpec(
    key="medgemma",
    model_id="google/medgemma-1.5-4b-it",
    prompt_key="medgemma_default",
    task="image-to-text",
    supports_images=True,
    generation_max_tokens = 450,
    caching = True,
)

MODEL_CLASS = MedGemma
