import abc
import random
import re


class BaseHFLM(abc.ABC):
    def __init__(
        self,
        name: str,
        device: str | None,
        dtype: str | None,
        model_id: str,
        task: str,
        trust_remote_code: bool = False,
        cache_dir: str | None = None,
    ) -> None:
        self.device = device
        self.dtype = dtype
        self.name = name
        self.model_id = model_id
        self.task = task
        self.trust_remote_code = trust_remote_code
        self.cache_dir = cache_dir

    def build_pipeline(self):
        from transformers import pipeline

        pipeline_kwargs: dict[str, object] = {
            "task": self.task,
            "model": self.model_id,
            "trust_remote_code": self.trust_remote_code,
        }
        if self.cache_dir is not None:
            pipeline_kwargs["cache_dir"] = self.cache_dir
        if self.device is not None:
            pipeline_kwargs["device"] = self.device
        if self.dtype is not None:
            pipeline_kwargs["torch_dtype"] = _resolve_dtype(self.dtype)
        return pipeline(**pipeline_kwargs)

    @abc.abstractmethod
    def process_img(self, paths):
        raise NotImplementedError()

    @abc.abstractmethod
    def get_likelihood_prompt(self, question, options):
        raise NotImplementedError()

    @abc.abstractmethod
    def get_logits(self, pixel_values, prompt_ids, ans_ids):
        raise NotImplementedError()

    @abc.abstractmethod
    def compute_scores(self, likelihood, ans_indices, length_norm):
        raise NotImplementedError()

    @abc.abstractmethod
    def get_prompt(self, question, options):
        raise NotImplementedError()

    def parse_response(self, response, target, options):
        print(f"Response: {response}; Target: {options[target]}")
        choice_style = [chr(i) for i in range(ord("A"), ord("Z") + 1)]
        prediction = re.findall(r"\(([A-Z])\)", response)
        if len(prediction) == 0:
            prediction = choice_style[random.choice(list(range(len(options))))]
        else:
            prediction = prediction[0]
        target = choice_style[target]
        return prediction.lower() == target.lower()


def _resolve_dtype(value: str):
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("torch is required when setting --dtype") from exc

    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if value in mapping:
        return mapping[value]
    if value.startswith("torch."):
        attr = value.split(".", 1)[1]
        return getattr(torch, attr)
    return value
