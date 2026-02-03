import transformers
from ..base import BaseHFLM
from ..spec import ModelSpec
import torch


class CXRMateED(BaseHFLM):

    def _ensure_loaded(self):
        if self._model is not None:
            return self._model, self._processor, self._tokenizer

        tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_id, cache_dir=self.cache_dir)
        model = transformers.AutoModel.from_pretrained(self.model_id, trust_remote_code=self.trust_remote_code, cache_dir=self.cache_dir).to(device=self.device)
        from torchvision.transforms import v2
        test_transforms = v2.Compose(
        [
            v2.PILToTensor(),
            v2.Grayscale(num_output_channels=3),
            v2.Resize(size=model.config.encoder.image_size, antialias=True),
            v2.CenterCrop(size=[model.config.encoder.image_size]*2),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=model.config.encoder.image_mean, std=model.config.encoder.image_std),
        ]
        )

        self._model = model
        self._processor = test_transforms
        self._tokenizer = tokenizer
        return model, test_transforms, tokenizer


    def __call__(
        self,
        image,
        prompt_text: str,
        *,
        user_text: str = "Analyze this image.",
        drop_config: dict[str, object] | None = None,
    ) -> list[dict[str, str]]:
        model, test_transforms, tokenizer = self._ensure_loaded()
        self.preprocess_image = test_transforms
        # Shape: (C, H, W) -> (B, S, C, H, W) where B=1 (batch), S=1 (single image per study)
        images = self.preprocess_image(image).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            
            output_ids = model.generate(
                pixel_values=images.to(device=self.device),
                max_length=512,
                num_beams=1,
                bad_words_ids=[[tokenizer.convert_tokens_to_ids('[NF]')], [tokenizer.convert_tokens_to_ids('[NI]')]],
            )   
        findings, impression = model.split_and_decode_sections(output_ids, tokenizer)

        for i, j in zip(findings, impression):
            print(f'Findings: {i}\nImpression: {j}\n')
        return [{"findings": findings[0], "impression": impression[0]}]
    
MODEL_SPEC = ModelSpec(
    key="cxrmateed",
    model_id="aehrc/cxrmate-rrg24",
    prompt_key="cxrmateed_default",
    task="image-to-text",
    supports_images=True,
    caching = True,

)

MODEL_CLASS = CXRMateED
