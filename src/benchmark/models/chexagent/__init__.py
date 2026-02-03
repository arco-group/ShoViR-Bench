import torch
from ..base import BaseHFLM
from ..spec import ModelSpec


class CheXagent(BaseHFLM):
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

    def _single_image_call(
        self,
        image,
        prompt_text: str,
        *,
        user_text: str = "Analyze this image.",
        drop_config: dict[str, object] | None = None,
    ) -> dict[str, str]:
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
            inputs, _ = self.build_chat_inputs(
                processor,
                image,
                prompt_text,
                user_text=prompt,
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
        return {"generated_text": findings}

    def __call__(
        self,
        image,
        prompt_text: str,
        *,
        user_text: str = "Analyze this image.",
        drop_config: dict[str, object] | None = None,
    ) -> list[dict[str, str]]:
        if isinstance(image, list):
            return [
                self._single_image_call(
                    img, prompt_text, user_text=user_text, drop_config=drop_config,
                )
                for img in image
            ]
        return [self._single_image_call(
            image, prompt_text, user_text=user_text, drop_config=drop_config,
        )]


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
