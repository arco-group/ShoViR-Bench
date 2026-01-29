import torch

from ..base import BaseHFLM
from ..chat_template import _prepare_inputs
from ..spec import ModelSpec


class Maira2(BaseHFLM):
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
        if device is None:
            device = self.device
        if torch_dtype is None:
            torch_dtype = self._torch_dtype

        indication = prompt_text
        if user_text and user_text != "Analyze this image.":
            indication = f"{prompt_text}\n{user_text}"

        inputs = processor.format_and_preprocess_reporting_input(
            current_frontal=image,
            current_lateral=None,
            prior_frontal=None,
            indication=indication,
            technique="",
            comparison="",
            prior_report=None,
            return_tensors="pt",
            get_grounding=False,
        )

        input_lens = None
        if "attention_mask" in inputs and inputs["attention_mask"] is not None:
            input_lens = inputs["attention_mask"].sum(dim=-1).to(torch.long)
        elif "input_ids" in inputs and inputs["input_ids"] is not None:
            input_lens = torch.full(
                (inputs["input_ids"].shape[0],),
                inputs["input_ids"].shape[-1],
                dtype=torch.long,
            )

        inputs = _prepare_inputs(inputs, device=device, torch_dtype=torch_dtype)

        if input_lens is not None and input_lens.numel() == 1:
            input_lens = int(input_lens.item())
        return inputs, input_lens


MODEL_SPEC = ModelSpec(
    key="maira-2",
    model_id="microsoft/maira-2",
    prompt_key="maira2_default",
    task="image-to-text",
    generation_max_tokens = 450,
    caching = True,
    supports_images=True,
)

MODEL_CLASS = Maira2
