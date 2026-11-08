"""Aggregate judge scores (Table tab:full_expanded_results) and optional Pearson cross-judge."""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCORES = ROOT / "output" / "scores"
NLE = ROOT / "output" / "nle"
DIMS = ("soundness", "fluency", "completeness", "context_awareness")


def load_meta(run_id):
    p = NLE / f"{run_id}.json"
    if not p.is_file():
        return {}
    return json.loads(p.read_text(encoding="utf-8")).get("experiment", {})


def main():
    by_key = defaultdict(list)
    primary_totals, cross_totals = [], []
    for path in sorted(SCORES.glob("*.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        meta = load_meta(row["run_id"])
        key = (meta.get("pipeline"), meta.get("approach"), meta.get("technique"))
        s = row["scores_primary"]
        by_key[key].append(s)
        primary_totals.append(sum(s[d] for d in DIMS) / 4)
        if "scores_cross" in row:
            c = row["scores_cross"]
            cross_totals.append(sum(c[d] for d in DIMS) / 4)
    if len(primary_totals) > 1 and len(primary_totals) == len(cross_totals):
        r = float(np.corrcoef(primary_totals, cross_totals)[0, 1])
        print(f"Pearson Total primary vs cross-judge: r = {r:.3f} ({len(primary_totals)} pairs)")
    print("\n| Pipeline | Approach | Technique | Total (mean ± SD) |")
    print("|---|---|---|---|")
    for key, rows in sorted(by_key.items()):
        totals = [sum(r[d] for d in DIMS) / 4 for r in rows]
        print(f"| {key[0]} | {key[1]} | {key[2]} | {np.mean(totals):.3f} ± {np.std(totals):.3f} |")


if __name__ == "__main__":
    main()
