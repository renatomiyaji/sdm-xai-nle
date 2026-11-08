"""Phase C: score NLE JSON files with the G-Eval rubric.

  cd article-code
  python -m evaluation.run_judge
  python -m evaluation.run_judge --cross-judge
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nle.providers import gemini as g_provider
from nle.providers import openai_chat
NLE = ROOT / "output" / "nle"
SCORES = ROOT / "output" / "scores"
RUBRIC = Path(__file__).resolve().parent / "judge_rubric.txt"


def render_rubric(plot_label, nle, reference):
    text = RUBRIC.read_text(encoding="utf-8")
    for key, val in [("PLOT_TYPE", plot_label), ("NLE", nle), ("VISUAL_REFERENCE", reference)]:
        text = text.replace("{" + key + "}", val)
    return text


def parse_scores(raw):
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
    return json.loads(raw)


def judge(provider, model, prompt, temperature, max_tokens):
    if provider == "openai":
        return openai_chat.chat_text(model, prompt, temperature, max_tokens)
    return g_provider.chat_text(model, prompt, temperature, max_tokens)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cross-judge", action="store_true")
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()
    models = yaml.safe_load((ROOT / "configs" / "model_params.yaml").read_text(encoding="utf-8"))
    primary = models["models"]["gpt-5.1-judge"]
    primary_model = os.getenv("JUDGE_MODEL", primary["model"])
    paths = sorted(NLE.glob("*.json"))
    if args.limit:
        paths = paths[: args.limit]
    SCORES.mkdir(parents=True, exist_ok=True)
    for path in paths:
        rec = json.loads(path.read_text(encoding="utf-8"))
        if rec.get("status") != "ok" or "nle" not in rec:
            continue
        prompt = render_rubric(rec["plot_label"], rec["nle"], rec.get("visual_reference", ""))
        primary_raw = judge(primary["provider"], primary_model, prompt, primary["temperature"], primary["max_tokens"])
        out = {"run_id": rec["run_id"], "scores_primary": parse_scores(primary_raw), "raw_primary": primary_raw}
        if args.cross_judge:
            alt = models["models"]["gemini-2.5-pro"]
            alt_raw = judge(alt["provider"], alt["model"], prompt, 1.0, alt["max_tokens"])
            out["scores_cross"] = parse_scores(alt_raw)
            out["raw_cross"] = alt_raw
        (SCORES / f"{rec['run_id']}.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(rec["run_id"], out["scores_primary"])


if __name__ == "__main__":
    main()
