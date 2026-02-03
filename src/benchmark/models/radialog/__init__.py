from pathlib import Path

import numpy as np
import torch
from PIL import Image
from huggingface_hub import snapshot_download

from ..base import BaseHFLM
from ..spec import ModelSpec
import sys
sys.path.insert(0, "RaDialog-interactive-radiology-report-generation")
from torchvision.transforms import Compose, Resize, ToTensor, CenterCrop, transforms

class RaDialog(BaseHFLM):

    @staticmethod
    def init_chexpert_predictor():
        from chexpert_train import LitIGClassifier, ExpandChannels
        ckpt_path = f"RaDialog-interactive-radiology-report-generation/findings_classifier/checkpoints/chexpert_train/ChexpertClassifier.ckpt"
        chexpert_cols = ["No Finding", "Enlarged Cardiomediastinum",
                        "Cardiomegaly", "Lung Opacity",
                        "Lung Lesion", "Edema",
                        "Consolidation", "Pneumonia",
                        "Atelectasis", "Pneumothorax",
                        "Pleural Effusion", "Pleural Other",
                        "Fracture", "Support Devices"]
        model = LitIGClassifier.load_from_checkpoint(ckpt_path, num_classes=14, class_names=chexpert_cols, strict=False)
        model.eval()
        model.cuda()
        model.half()
        cp_transforms = Compose([Resize(512), CenterCrop(488), ToTensor(), ExpandChannels()])

        return model, np.asarray(model.class_names), cp_transforms
    def _ensure_loaded(self):
        if self._model is not None:
            return self._model, self._tokenizer

        # Import RaDialog dependencies
        from LLAVA_Biovil.llava.model.builder import load_pretrained_model
        from LLAVA_Biovil.llava.mm_utils import remap_to_uint8
        from utils import create_chest_xray_transform_for_inference

        # Download model files
        model_path = snapshot_download(
            repo_id=self.model_id,
            revision="main",
            cache_dir=self.cache_dir,
        )
        model_path = Path(model_path)

        self._tokenizer, self._model, image_processor, context_len = load_pretrained_model(
            model_path,
            model_base='liuhaotian/llava-v1.5-7b',
            model_name="llava-v1.5-7b-task-lora_radialog_instruct_llava_biovil_unfrozen_2e-5_5epochs_v5_checkpoint-21000",
            load_8bit=False,
            load_4bit=False,
        )
        
        if self.device is not None:
            model = self._model.to(self.device)

        model.config.tokenizer_padding_side = "left"

        # Initialize CheXpert predictor for findings
        cp_model, cp_class_names, cp_transforms = self.init_chexpert_predictor()

        # Store components
        self._cp_model = cp_model
        self._cp_class_names = cp_class_names
        self._cp_transforms = cp_transforms
        self._vis_transforms = create_chest_xray_transform_for_inference(512, center_crop_size=448)
        self._remap_to_uint8 = remap_to_uint8

        # Return model and tokenizer (processor equivalent)
        return self._model, self._tokenizer

    def _single_image_call(
        self,
        image: Image.Image,
        prompt_text: str,
        *,
        user_text: str = "Analyze this image.",
        drop_config: dict[str, object] | None = None,
    ) -> dict[str, str]:
        from LLAVA_Biovil.llava.mm_utils import tokenizer_image_token, KeywordsStoppingCriteria
        from LLAVA_Biovil.llava.conversation import SeparatorStyle, conv_vicuna_v1
        from LLAVA_Biovil.llava.constants import IMAGE_TOKEN_INDEX

        model, tokenizer = self._ensure_loaded()

        # Preprocess image
        image = self.preprocess_image(image)

        # Get CheXpert predictions for findings
        cp_image = self._cp_transforms(image)
        with torch.no_grad():
            logits = self._cp_model(cp_image[None].half().to(self.device))
        preds_probs = torch.sigmoid(logits)
        preds = preds_probs > 0.5
        pred = preds[0].cpu().numpy()
        findings = self._cp_class_names[pred].tolist()
        findings = ', '.join(findings).lower().strip()

        # Build conversation
        conv = conv_vicuna_v1.copy()
        report_gen_prompt = (
            f"<image>. Predicted Findings: {findings}. "
            "You are to act as a radiologist and write the finding section of a chest x-ray "
            "radiology report for this X-ray image and the given predicted findings. "
            "Write in the style of a radiologist, write one fluent text without enumeration, "
            "be concise and don't provide explanations or reasons."
        )

        conv.append_message("USER", report_gen_prompt)
        conv.append_message("ASSISTANT", None)
        text_input = conv.get_prompt()

        # Prepare image tensor
        image_tensor = self._vis_transforms(image).unsqueeze(0)
        image_tensor = image_tensor.to(model.device, dtype=torch.bfloat16)

        # Prepare input ids
        input_ids = tokenizer_image_token(
            text_input, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt'
        ).unsqueeze(0).to(model.device)

        # Set up stopping criteria
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        stopping_criteria = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)

        # Generate report
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                do_sample=False,
                use_cache=True,
                max_new_tokens=self.generation_max_tokens,
                stopping_criteria=[stopping_criteria],
                pad_token_id=tokenizer.pad_token_id,
            )

        generated_text = tokenizer.decode(
            output_ids[0, input_ids.shape[1]:]
        ).strip().replace("</s>", "")

        return {"generated_text": generated_text}

    def _batch_call(
        self,
        images: list[Image.Image],
        prompt_text: str,
        *,
        user_text: str = "Analyze this image.",
        drop_config: dict[str, object] | None = None,
    ) -> list[dict[str, str]]:
        from LLAVA_Biovil.llava.mm_utils import tokenizer_image_token, KeywordsStoppingCriteria
        from LLAVA_Biovil.llava.conversation import SeparatorStyle, conv_vicuna_v1
        from LLAVA_Biovil.llava.constants import IMAGE_TOKEN_INDEX

        model, tokenizer = self._ensure_loaded()

        # Preprocess all images
        preprocessed_images = [self.preprocess_image(img) for img in images]

        # Batch CheXpert predictions
        cp_images = torch.stack([self._cp_transforms(img) for img in preprocessed_images])
        with torch.no_grad():
            logits = self._cp_model(cp_images.half().to(self.device))
        preds_probs = torch.sigmoid(logits)
        preds = (preds_probs > 0.5).cpu().numpy()

        # Build conversations for each image (each has different findings)
        all_input_ids = []
        all_image_tensors = []
        for i, (img, pred) in enumerate(zip(preprocessed_images, preds)):
            findings = self._cp_class_names[pred].tolist()
            findings_str = ', '.join(findings).lower().strip()

            conv = conv_vicuna_v1.copy()
            report_gen_prompt = (
                f"<image>. Predicted Findings: {findings_str}. "
                "You are to act as a radiologist and write the finding section of a chest x-ray "
                "radiology report for this X-ray image and the given predicted findings. "
                "Write in the style of a radiologist, write one fluent text without enumeration, "
                "be concise and don't provide explanations or reasons."
            )
            conv.append_message("USER", report_gen_prompt)
            conv.append_message("ASSISTANT", None)
            text_input = conv.get_prompt()

            input_ids = tokenizer_image_token(
                text_input, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt'
            )
            all_input_ids.append(input_ids)

            image_tensor = self._vis_transforms(img)
            all_image_tensors.append(image_tensor)

        # Left-pad input_ids to same length
        max_len = max(ids.shape[0] for ids in all_input_ids)
        padded_input_ids = []
        for ids in all_input_ids:
            pad_len = max_len - ids.shape[0]
            if pad_len > 0:
                padding = torch.full((pad_len,), tokenizer.pad_token_id, dtype=ids.dtype)
                ids = torch.cat([padding, ids])
            padded_input_ids.append(ids)

        input_ids_batch = torch.stack(padded_input_ids).to(model.device)
        attention_mask = (input_ids_batch != tokenizer.pad_token_id).long()

        # Stack image tensors
        image_tensors_batch = torch.stack(all_image_tensors).to(model.device, dtype=torch.bfloat16)

        # Set up stopping criteria
        stop_str = conv_vicuna_v1.sep if conv_vicuna_v1.sep_style != SeparatorStyle.TWO else conv_vicuna_v1.sep2
        stopping_criteria = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids_batch)

        # Generate reports in batch
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids_batch,
                images=image_tensors_batch,
                attention_mask=attention_mask,
                do_sample=False,
                use_cache=True,
                max_new_tokens=self.generation_max_tokens,
                stopping_criteria=[stopping_criteria],
                pad_token_id=tokenizer.pad_token_id,
            )

        # Decode each output
        results = []
        for i in range(len(images)):
            generated_text = tokenizer.decode(
                output_ids[i, input_ids_batch.shape[1]:]
            ).strip().replace("</s>", "")
            results.append({"generated_text": generated_text})

        return results

    def __call__(
        self,
        image,
        prompt_text: str,
        *,
        user_text: str = "Analyze this image.",
        drop_config: dict[str, object] | None = None,
    ) -> list[dict[str, str]]:
        if isinstance(image, list):
            return self._batch_call(
                image, prompt_text, user_text=user_text, drop_config=drop_config,
            )
        return self._batch_call(
            [image], prompt_text, user_text=user_text, drop_config=drop_config,
        )

    def preprocess_image(self, image: Image.Image) -> Image.Image:
        """Preprocess image: remap to uint8 and convert to grayscale."""
        # Default preprocessing for RaDialog
        arr = self._remap_to_uint8(np.array(image))
        image = Image.fromarray(arr).convert("L")
        return image


MODEL_SPEC = ModelSpec(
    key="radialog",
    model_id="Chantal/RaDialog-interactive-radiology-report-generation",
    prompt_key="radialog_default",
    task="image-to-text",
    generation_max_tokens=300,
    caching=True,
    supports_images=True,
)

MODEL_CLASS = RaDialog
