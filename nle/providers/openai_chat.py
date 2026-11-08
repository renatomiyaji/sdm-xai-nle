import base64
import os
from pathlib import Path

from openai import OpenAI

_client = None


def client():
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    return _client


def chat_text(model, prompt, temperature, max_tokens):
    r = client().chat.completions.create(
        model=model, messages=[{"role": "user", "content": prompt}],
        temperature=temperature, max_completion_tokens=max_tokens)
    return r.choices[0].message.content or ""


def chat_image(model, prompt, image_path, temperature, max_tokens):
    data = base64.standard_b64encode(Path(image_path).read_bytes()).decode()
    r = client().chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{data}"}},
        ]}],
        temperature=temperature,
        max_completion_tokens=max_tokens,
    )
    return r.choices[0].message.content or ""
