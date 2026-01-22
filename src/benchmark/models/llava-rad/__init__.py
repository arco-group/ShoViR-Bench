from ..base import BaseHFLM
from ..spec import ModelSpec


class LLaVARad(BaseHFLM):
    def _ensure_loaded(self):
        # Load processor
        if self._processor is None:

            self._processor = load_processor_llava(
                self.model_id,
                cache_dir=self.cache_dir,
                torch_dtype=self._torch_dtype,
            )
            
        # Load model
        if self._model is None:
            
            self._model = load_model_llava(
                self.model_id,
                cache_dir=self.cache_dir,
                torch_dtype=self._torch_dtype,
            )


        if self.device is not None:
            self._model.to(self.device)

        return self._model, self._processor
    pass


MODEL_SPEC = ModelSpec(
    key="llava-rad",
    model_id="microsoft/llava-rad",
    prompt_key="llava-rad_default",
    task="image-to-text",
    supports_images=True,
)

MODEL_CLASS = LLaVARad