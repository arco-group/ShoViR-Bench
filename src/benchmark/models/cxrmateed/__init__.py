from ..base import BaseHFLM
from ..spec import ModelSpec
import torch

class CXRMateED(BaseHFLM):
    def __call__(
        self,
        image,
        prompt_text: str,
        *,
        user_text: str = "Analyze this image.",
        drop_config: dict[str, object] | None = None,
    ) -> list[dict[str, str]]:
        model, processor = self._ensure_loaded()
        tokenizer = getattr(processor, "tokenizer", processor)

        if isinstance(image, dict):
            example = dict(image)
        else:
            example = {
                "images": self.preprocess_image(image),
                "prompt_text": prompt_text,
            }
            if user_text and user_text != "Analyze this image.":
                example["user_text"] = user_text

        from torchvision import transforms
        to_tensor = transforms.ToTensor()
        if "images" in example and not torch.is_tensor(example["images"]):
            example["images"] = to_tensor(example["images"])
        device = self.device
        example = {
            key: value.to(device).unsqueeze(0) if torch.is_tensor(value) else value
            for key, value in example.items()
        }

        (
            inputs_embeds,
            attention_mask,
            token_type_ids,
            position_ids,
            bos_token_ids,
        ) = model.prepare_inputs(tokenizer=tokenizer, **example)

        special_token_ids = []
        sep_token_id = getattr(tokenizer, "sep_token_id", None)
        if sep_token_id is not None:
            special_token_ids.append(sep_token_id)

        with torch.no_grad():
            output = model.generate(
                input_ids=bos_token_ids,
                decoder_inputs_embeds=inputs_embeds,
                decoder_token_type_ids=token_type_ids,
                prompt_attention_mask=attention_mask,
                prompt_position_ids=position_ids,
                special_token_ids=special_token_ids,
                max_length=256,
                num_beams=4,
                return_dict_in_generate=True,
            )
        output_ids = output["sequences"]

        split_token_ids = []
        for token_id in (sep_token_id, getattr(tokenizer, "eos_token_id", None)):
            if token_id is not None:
                split_token_ids.append(token_id)
        findings, impression = model.split_and_decode_sections(output_ids, split_token_ids, tokenizer)
        sections: list[str] = []
        for finding, impression_item in zip(findings, impression):
            sections.append(f"Findings:\t{finding}\nImpression:\t{impression_item}")
        return [{"generated_text": "\n\n".join(sections)}]
    
MODEL_SPEC = ModelSpec(
    key="cxrmateed",
    model_id="aehrc/cxrmate-ed",
    prompt_key="medgemma_default",
    task="image-to-text",
    supports_images=True,

)

MODEL_CLASS = CXRMateED
