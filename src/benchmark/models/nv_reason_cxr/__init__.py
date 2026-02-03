from __future__ import annotations

import torch

from ..base import BaseHFLM
from ..chat_template import _prepare_inputs
from ..spec import ModelSpec


class NVReasonCXR(BaseHFLM):
    def build_chat_inputs(
        self,
        processor,
        image,
        prompt_text: str,
        *,
        user_text: str = "Provide a comprehensive image analysis, and list all abnormalities.",
        device: str | None = None,
        torch_dtype=None,
    ):
        if device is None:
            device = self.device
        if torch_dtype is None:
            torch_dtype = self._torch_dtype

        message_text = prompt_text
        if user_text and user_text != "Provide a comprehensive image analysis, and list all abnormalities.":
            message_text = user_text

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": message_text},
                ],
            }
        ]
        text = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(text=text, images=[image], return_tensors="pt")

        input_lens = None
        if "attention_mask" in inputs and inputs["attention_mask"] is not None:
            input_lens = inputs["attention_mask"].sum(dim=-1).to(torch.long)
        elif "input_ids" in inputs and inputs["input_ids"] is not None:
            input_lens = torch.full(
                (inputs["input_ids"].shape[0],),
                inputs["input_ids"].shape[-1],
                dtype=torch.long,
            )

        inputs = _prepare_inputs(inputs, device=device, dtype=torch_dtype)
        inputs.pop("num_crops", None)

        if input_lens is not None and input_lens.numel() == 1:
            input_lens = int(input_lens.item())
        return inputs, input_lens


MODEL_SPEC = ModelSpec(
    key="nv-reason-cxr-3b",
    model_id="nvidia/NV-Reason-CXR-3B",
    prompt_key="nv_reason_default",
    task="image-to-text",
    supports_images=True,
    generation_max_tokens=2048,
    caching = True,
)

MODEL_CLASS = NVReasonCXR
