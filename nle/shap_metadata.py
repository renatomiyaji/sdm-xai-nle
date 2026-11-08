import json
from pathlib import Path

CACHE = Path(__file__).resolve().parents[1] / "output" / "shap_cache"


def load_family(family):
    path = CACHE / f"{family}.json"
    if not path.is_file():
        raise FileNotFoundError(f"Run phase A first: missing {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def structured_metadata(family, plot_key):
    meta = load_family(family)
    lines = [
        f"Plot type: {plot_key}",
        f"Family: {family}",
        f"E[f(X)] (expected value class 1): {meta['expected_value']:.4f}",
    ]
    if plot_key == "waterfall":
        lines.append(f"Instance index: {meta['instance_index']}, f(x) = {meta['instance_proba']:.4f}")
        lines.append("Per-feature SHAP at instance:")
        for row in meta["waterfall"]:
            lines.append(f"  - {row['feature']}: value={row['value']:.4f}, SHAP={row['shap']:+.4f}")
    lines.append("Global mean |SHAP| ranking:")
    for feat in meta["feature_ranking"]:
        lines.append(f"  - {feat}: {meta['mean_abs_shap'][feat]:.4f}")
    return "\n".join(lines)


def visual_reference(family, plot_label, extra=""):
    meta = load_family(family)
    base = structured_metadata(family, plot_label)
    if extra:
        base += "\n\nOCR / auxiliary text:\n" + extra
    return base
