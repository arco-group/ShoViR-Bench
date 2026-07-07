import io
import os
from concurrent.futures import ThreadPoolExecutor

from PIL import Image

from ..base import BaseHFLM
from ..spec import ModelSpec


class Gemini(BaseHFLM):

    def _ensure_loaded(self):
        return None, None

    def _call_one(self, client, img: Image.Image, prompt_text: str) -> str:
        from google.genai import types

        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        image_bytes = buf.getvalue()

        response = client.models.generate_content(
            model=self.model_id,
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
                prompt_text,
            ],
            config=types.GenerateContentConfig(
                max_output_tokens=self.generation_max_tokens,
            ),
        )
        return response.text or ""

    def __call__(
        self,
        image,
        prompt_text: str,
        *,
        user_text: str = "",
        drop_config: dict | None = None,
    ) -> list[dict[str, str]]:
        from google import genai

        client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

        if not isinstance(image, list):
            image = [image]

        futures = {}
        with ThreadPoolExecutor(max_workers=len(image)) as pool:
            for i, img in enumerate(image):
                futures[pool.submit(self._call_one, client, img, prompt_text)] = i

        results = [None] * len(image)
        for future, i in futures.items():
            results[i] = {"generated_text": future.result()}

        return results


MODEL_SPEC = ModelSpec(
    key="gemini",
    model_id="gemini-2.0-flash",
    prompt_key="gemini_default",
    task="image-to-text",
    supports_images=True,
    generation_max_tokens=512,
    caching=False,
)

MODEL_CLASS = Gemini
