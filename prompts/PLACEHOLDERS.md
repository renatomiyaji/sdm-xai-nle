# Placeholders

Replace each placeholder by plain string substitution before sending the prompt.

| Placeholder | Used in | Value |
|---|---|---|
| `{PLOT_TYPE}` | all prompts and the judge rubric | SHAP visualization type, e.g. `beeswarm summary plot`, `waterfall plot`, `partial dependence plot` |
| `{STRUCTURED_METADATA}` | `baseline_llm_synthesis.txt` | Text from `output/shap_cache/*.json` via `nle/shap_metadata.py` |
| `{EXTRACTED_TEXT}` | `ocr_llm_synthesis.txt` | Raw text extracted by the OCR engine from the plot image (axis labels, variable names, numbers) |
| `{VISUAL_SUMMARY}` | `hybrid_llm_synthesis.txt` | Full output of the Stage 1 VLM (`hybrid_vlm_extract.txt`) |
| `{NLE}` | `../evaluation/judge_rubric.txt` | Candidate explanation being scored |
| `{VISUAL_REFERENCE}` | `../evaluation/judge_rubric.txt` | Extracted plot metadata used as the reference |

`vlm_direct.txt` and `hybrid_vlm_extract.txt` are sent together with the SHAP plot image.
