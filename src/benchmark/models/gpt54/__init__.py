import base64
import io
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from PIL import Image

from ..base import BaseHFLM
from ..spec import ModelSpec


class GPT54(BaseHFLM):

    def _ensure_loaded(self):
        return None, None

    def _encode_image(self, image: Image.Image) -> str:
        image = image.resize((224, 224), Image.LANCZOS)
        buf = io.BytesIO()
        image.save(buf, format="JPEG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def _call_one(self, client, img: Image.Image, prompt_text: str) -> str:
        b64 = self._encode_image(img)
        response = client.chat.completions.create(
            model=self.model_id,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                        },
                    ],
                }
            ],
            max_completion_tokens=self.generation_max_tokens,
        )
        return response.choices[0].message.content or ""

    def __call__(
        self,
        image,
        prompt_text: str,
        *,
        user_text: str = "",
        drop_config: dict | None = None,
    ) -> list[dict[str, str]]:
        from openai import OpenAI

        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

        if not isinstance(image, list):
            image = [image]

        # Submit all requests concurrently; preserve original order via index
        futures = {}
        with ThreadPoolExecutor(max_workers=len(image)) as pool:
            for i, img in enumerate(image):
                futures[pool.submit(self._call_one, client, img, prompt_text)] = i

        results = [None] * len(image)
        for future, i in futures.items():
            results[i] = {"generated_text": future.result()}

        return results


MODEL_SPEC = ModelSpec(
    key="gpt54",
    model_id="gpt-5.4",
    prompt_key="gpt54_default",
    task="image-to-text",
    supports_images=True,
    generation_max_tokens=512,
    caching=False,
)

MODEL_CLASS = GPT54
