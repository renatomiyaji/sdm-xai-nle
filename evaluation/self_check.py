"""Parser check for judge JSON output format (no API)."""
import json

from evaluation.run_judge import parse_scores

sample = '{"soundness": 0.85, "fluency": 0.90, "completeness": 0.88, "context_awareness": 0.87}'
parsed = parse_scores(sample)
assert set(parsed) == {"soundness", "fluency", "completeness", "context_awareness"}
print("judge JSON parser ok", parsed)
