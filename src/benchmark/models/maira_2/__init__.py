import torch

from ..base import BaseHFLM
from ..chat_template import _prepare_inputs
from ..spec import ModelSpec


class Maira2(BaseHFLM):
    def _batch_call(
        self,
        images: list,
        prompt_text: str,
        *,
        user_text: str = "Analyze this image.",
        drop_config: dict[str, object] | None = None,
    ) -> list[dict[str, str]]:
        """Batch N images into one generate() using MAIRA's custom processor."""
        model, processor = self._ensure_loaded()

        batch_input_ids = []
        batch_attention = []
        batch_pixel = []
        input_lens = []

        for img in images:
            inputs, input_len = self.build_chat_inputs(
                processor, img, prompt_text, user_text=user_text,
            )
            if drop_config:
                inputs = self._apply_drop_config(inputs, processor, model, drop_config)

            batch_input_ids.append(inputs["input_ids"].squeeze(0))
            if "attention_mask" in inputs:
                batch_attention.append(inputs["attention_mask"].squeeze(0))
            if "pixel_values" in inputs:
                batch_pixel.append(inputs["pixel_values"].squeeze(0))
            input_lens.append(input_len if input_len is not None else inputs["input_ids"].shape[-1])

        # Pad input_ids and attention_mask to the same length
        tok = getattr(processor, "tokenizer", processor)
        pad_id = getattr(tok, "pad_token_id", 0) or 0

        max_len = max(ids.shape[-1] for ids in batch_input_ids)
        padded_ids = []
        padded_attn = []
        for i, ids in enumerate(batch_input_ids):
            pad_len = max_len - ids.shape[-1]
            padded_ids.append(
                torch.nn.functional.pad(ids, (pad_len, 0), value=pad_id)
            )
            if batch_attention:
                padded_attn.append(
                    torch.nn.functional.pad(batch_attention[i], (pad_len, 0), value=0)
                )
            # Adjust input_len for the padding added on the left
            input_lens[i] = input_lens[i] + pad_len

        batched = {"input_ids": torch.stack(padded_ids)}
        if padded_attn:
            batched["attention_mask"] = torch.stack(padded_attn)
        if batch_pixel:
            batched["pixel_values"] = torch.stack(batch_pixel)

        batched = _prepare_inputs(batched, device=self.device, torch_dtype=self._torch_dtype)

        with torch.no_grad():
            output_ids = model.generate(
                **batched,
                max_new_tokens=self.generation_max_tokens,
                do_sample=False,
                use_cache=self.caching,
            )

        results = []
        for i in range(output_ids.shape[0]):
            generated = output_ids[i, input_lens[i]:]
            text = processor.decode(generated, skip_special_tokens=True)
            results.append({"generated_text": text})
        return results

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
