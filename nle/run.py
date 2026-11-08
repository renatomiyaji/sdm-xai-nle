"""Phase B: generate NLEs from configs/experiments.yaml and output/figures/shap/*.png

  cd article-code
  python -m nle.run --dry-run
  python -m nle.run --experiment baseline_gpt41
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nle.providers import gemini as g_provider
from nle.providers import ocr_text, openai_chat
from nle.render_prompt import render
from nle.shap_metadata import structured_metadata, visual_reference

FIG = ROOT / "output" / "figures" / "shap"
OUT = ROOT / "output" / "nle"
CFG = ROOT / "configs"


def load_yaml(name):
    return yaml.safe_load((CFG / name).read_text(encoding="utf-8"))


def model_cfg(models, key):
    return models["models"][key]


def resolve_model(models, key):
    m = model_cfg(models, key)
    if m["provider"] == "openai":
        return "openai", m["model"], m["temperature"], m["max_tokens"]
    if m["provider"] == "gemini":
        return "gemini", m["model"], m["temperature"], m["max_tokens"]
    raise ValueError(m)


def generate(provider, model, prompt, image, temp, tokens):
    if provider == "openai":
        if image:
            return openai_chat.chat_image(model, prompt, image, temp, tokens)
        return openai_chat.chat_text(model, prompt, temp, tokens)
    if image:
        return g_provider.chat_image(model, prompt, image, temp, tokens)
    return g_provider.chat_text(model, prompt, temp, tokens)


def plot_path(pattern, family, plot_key):
    return FIG / pattern.format(family=family)


def run_one(exp, family, plot_key, models, dry_run):
    conf = load_yaml("experiments.yaml")
    plot_label = conf["plot_labels"][plot_key]
    png = plot_path(conf["plot_files"][plot_key], family, plot_key)
    run_id = f"{exp['id']}__{family}__{plot_key}"
    record = {"run_id": run_id, "experiment": exp, "family": family, "plot": plot_key,
              "plot_label": plot_label, "image": str(png), "timestamp": datetime.now(timezone.utc).isoformat()}

    if dry_run:
        record["status"] = "dry_run"
        print(run_id, "->", png)
        return record

    if not png.is_file():
        record["status"] = "missing_image"
        return record

    pipeline = exp["pipeline"]
    if pipeline == "baseline":
        prov, model, temp, tokens = resolve_model(models, exp["generator"])
        prompt = render("baseline_llm_synthesis.txt", PLOT_TYPE=plot_label,
                        STRUCTURED_METADATA=structured_metadata(family, plot_key))
        record["nle"] = generate(prov, model, prompt, None, temp, tokens)
    elif pipeline == "ocr":
        text = ocr_text.extract(exp["ocr"], png)
        prov, model, temp, tokens = resolve_model(models, exp["generator"])
        prompt = render("ocr_llm_synthesis.txt", PLOT_TYPE=plot_label, EXTRACTED_TEXT=text)
        record["ocr_text"] = text
        record["nle"] = generate(prov, model, prompt, None, temp, tokens)
    elif pipeline == "direct":
        prov, model, temp, tokens = resolve_model(models, exp["generator"])
        prompt = render("vlm_direct.txt", PLOT_TYPE=plot_label)
        record["nle"] = generate(prov, model, prompt, png, temp, tokens)
    elif pipeline == "hybrid":
        e_prov, e_model, e_temp, e_tokens = resolve_model(models, exp["extractor"])
        g_prov, g_model, g_temp, g_tokens = resolve_model(models, exp["generator"])
        p1 = render("hybrid_vlm_extract.txt", PLOT_TYPE=plot_label)
        summary = generate(e_prov, e_model, p1, png, e_temp, e_tokens)
        p2 = render("hybrid_llm_synthesis.txt", PLOT_TYPE=plot_label, VISUAL_SUMMARY=summary)
        record["visual_summary"] = summary
        record["nle"] = generate(g_prov, g_model, p2, None, g_temp, g_tokens)
    else:
        raise ValueError(pipeline)

    record["status"] = "ok"
    record["visual_reference"] = visual_reference(family, plot_label, record.get("ocr_text", ""))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{run_id}.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--experiment", help="single experiment id from experiments.yaml")
    args = p.parse_args()
    models = load_yaml("model_params.yaml")
    conf = load_yaml("experiments.yaml")
    exps = [e for e in conf["experiments"] if not args.experiment or e["id"] == args.experiment]
    for exp in exps:
        for family in conf["families"]:
            for plot_key in exp["plots"]:
                run_one(exp, family, plot_key, models, args.dry_run)


if __name__ == "__main__":
    main()
