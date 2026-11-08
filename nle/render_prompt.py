from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "prompts"


def load(name):
    return (ROOT / name).read_text(encoding="utf-8")


def render(template_name, **fields):
    text = load(template_name)
    for key, value in fields.items():
        text = text.replace("{" + key + "}", value)
    return text
