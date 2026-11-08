import os
from pathlib import Path

from google import genai
from google.genai import types

_client = None


def _get():
    global _client
    if _client is None:
        _client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
    return _client


def chat_text(model, prompt, temperature, max_tokens):
    cfg = types.GenerateContentConfig(temperature=temperature, max_output_tokens=max_tokens)
    r = _get().models.generate_content(model=model, contents=prompt, config=cfg)
    return r.text or ""


def chat_image(model, prompt, image_path, temperature, max_tokens):
    img = types.Part.from_bytes(data=Path(image_path).read_bytes(), mime_type="image/png")
    cfg = types.GenerateContentConfig(temperature=temperature, max_output_tokens=max_tokens)
    r = _get().models.generate_content(model=model, contents=[prompt, img], config=cfg)
    return r.text or ""
