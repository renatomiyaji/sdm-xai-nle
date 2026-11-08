import shutil
import subprocess
import tempfile
from pathlib import Path


def unstructured_png(path):
    try:
        from unstructured.partition.image import partition_image
    except ImportError as e:
        raise RuntimeError("pip install -r requirements-nle.txt (unstructured[pdf,image])") from e
    elements = partition_image(filename=str(path))
    return "\n".join(el.text for el in elements if getattr(el, "text", None))


def docling_png(path):
    exe = shutil.which("docling")
    if not exe:
        raise RuntimeError("Docling CLI not on PATH; install docling and ensure docling convert is available")
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "out.md"
        subprocess.run([exe, "convert", str(path), "--to", "md", "--output", str(out.parent)], check=True)
        md = list(Path(tmp).glob("*.md"))
        if not md:
            raise RuntimeError("Docling produced no markdown output")
        return md[0].read_text(encoding="utf-8")


def extract(engine, image_path):
    if engine == "unstructured":
        return unstructured_png(image_path)
    if engine == "docling":
        return docling_png(image_path)
    raise ValueError(engine)
