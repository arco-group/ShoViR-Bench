import torch
from ..base import BaseHFLM
from ..chat_template import _prepare_inputs
from ..spec import ModelSpec

class CheXagent(BaseHFLM):
    pass



class CheXagent(BaseHFLM):
    pass
    def __call__(
        self,
        image,
        prompt_text: str,
        *,
        user_text: str = "Analyze this image.",
        drop_config: dict[str, object] | None = None,
    ) -> list[dict[str, str]]:
        image = self.preprocess_image(image)
        model, processor = self._ensure_loaded()

        anatomies = [
            "Airway",
            "Breathing",
            "Cardiac",
            "Diaphragm",
            (
                "Everything else (e.g., mediastinal contours, bones, "
                "soft tissues, tubes, valves, and pacemakers)"
            ),
        ]
        prompts = [
            f'Please provide a detailed description of "{anatomy}" in the chest X-ray'
            for anatomy in anatomies
        ]
        anatomies = ["View"] + anatomies
        prompts = ["Determine the view of this CXR"] + prompts

        findings_chunks: list[str] = []
        for anatomy_idx, prompt in enumerate(prompts):
            inputs, _ = self._build_user_only_inputs(
                processor,
                image,
                prompt,
            )
            inputs = self._apply_drop_config(inputs, processor, model, drop_config)
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=self.generation_max_tokens,
                    do_sample=False,
                    num_beams=1,
                    temperature=1,
                    top_p=1.,
                    use_cache=self.caching,
                )
            decoded = processor.batch_decode(output_ids, skip_special_tokens=True)
            text = decoded[0] if decoded else ""
            if anatomy_idx != 0:
                findings_chunks.append(text.strip())

        findings = " ".join(findings_chunks).strip().replace("</s>", "").replace(' .', ' ')[0:]
        return [{"generated_text": findings}]

    def _build_user_only_inputs(
        self,
        processor,
        image,
        prompt_text: str,
        *,
        device: str | None = None,
        torch_dtype=None,
    ):
        if device is None:
            device = self.device
        if torch_dtype is None:
            torch_dtype = self._torch_dtype

        if hasattr(processor, "apply_chat_template"):
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text},
                        {"type": "image", "image": image},
                    ],
                }
            ]
            try:
                inputs = processor.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                )
            except Exception:
                inputs = processor(images=image, text=prompt_text, return_tensors="pt")
        else:
            inputs = processor(images=image, text=prompt_text, return_tensors="pt")

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
        inputs.pop("num_crops", None)

        if input_lens is not None and input_lens.numel() == 1:
            input_lens = int(input_lens.item())
        return inputs, input_lens


MODEL_SPEC = ModelSpec(
    key="chexagent",
    model_id="StanfordAIMI/CheXagent-8b",
    prompt_key="chexagent_default",
    task="image-to-text",
    supports_images=True,
    generation_max_tokens = 512,
    caching = True ,
)

MODEL_CLASS = CheXagent