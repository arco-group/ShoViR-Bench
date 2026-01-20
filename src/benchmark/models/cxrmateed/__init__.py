import transformers
from ..base import BaseHFLM
from ..spec import ModelSpec
import torch
from torchvision import transforms

class CXRMateED(BaseHFLM):

    def _ensure_loaded(self):
        encoder_decoder = transformers.AutoModel.from_pretrained(self.model_id, trust_remote_code=True).to(self.device)
        encoder_decoder.eval()
        tokenizer = transformers.PreTrainedTokenizerFast.from_pretrained(self.model_id )
        image_processor = transformers.AutoFeatureExtractor.from_pretrained(self.model_id)

        test_transforms = transforms.Compose(
            [
                transforms.Resize(size=image_processor.size['shortest_edge']),
                transforms.CenterCrop(size=[
                    image_processor.size['shortest_edge'],
                    image_processor.size['shortest_edge'],
                ]
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=image_processor.image_mean,
                    std=image_processor.image_std,
                ),
            ]
        )
        return encoder_decoder, test_transforms, tokenizer


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
        images = self.preprocess_image(image.convert('RGB')).unsqueeze(0)
        with torch.no_grad():
            outputs = model.generate(
                
                pixel_values=images.to(self.device),
                special_token_ids=[tokenizer.sep_token_id],
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                return_dict_in_generate=True,
                use_cache=False,
                max_length=256,
                num_beams=4,
            )

        # Findings and impression sections (exclude previous impression section):
        findings, impression = model.split_and_decode_sections(
            outputs.sequences,
            [tokenizer.sep_token_id, tokenizer.eos_token_id],
            tokenizer,
        )
        for i, j in zip(findings, impression):
            print(f'Findings: {i}\nImpression: {j}\n')
        return [{"findings": findings[0], "impression": impression[0]}]
    
MODEL_SPEC = ModelSpec(
    key="cxrmateed",
    model_id="aehrc/cxrmate-single-tf",
    prompt_key="medgemma_default",
    task="image-to-text",
    supports_images=True,

)

MODEL_CLASS = CXRMateED
