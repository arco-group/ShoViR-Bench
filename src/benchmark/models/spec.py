from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    key: str
    model_id: str
    prompt_key: str
    task: str
    supports_images: bool = True
    generation_max_tokens: int = 128
    caching: bool = False

