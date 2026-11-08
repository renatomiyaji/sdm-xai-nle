# Code and materials — ECOCOM manuscript replication

Repository: [github.com/renatomiyaji/sdm-xai-nle](https://github.com/renatomiyaji/sdm-xai-nle).

Scripts, configs, prompts and evaluation rubric for *What is the model telling us? Transforming complex
visual outputs into clear ecological explanations using Large Language Models* (GoAmazon 2014/15, Amazonian
*Thraupidae* / *Tyrannidae*).

Run commands from this repository root. Outputs go to `output/` (gitignored).

## Data (not in the repository)

- Atmospheric grid: Miyaji et al. (2021), https://doi.org/10.3390/data6120126
- Occurrences: GBIF https://doi.org/10.15468/dl.ppwbzv and ICMBio https://www.gov.br/icmbio

Build a local CSV: `Latitude`, `Longitude`, nine predictors (same order as the study matrix), then columns
`Especie_<Genus epithet>` (0/1). Species per family are listed in `scripts/sdm_common.py`.

**Pseudo-absences:** the code uses `ClusterCentroids` on the training split (notebook `SDM_XAI.ipynb`). The
manuscript text describes K-Means centroids on background samples; metrics target Table `table:metrics`.

**Scaling:** MinMax is fit on the full grid before the 70/30 split (notebook behaviour), not train-only.

## Phase A — SDM, metrics and SHAP figures

```bash
pip install -r requirements.txt
python -m scripts.run_all /path/to/study_matrix.csv
```

Or separately:

```bash
python -m scripts.train_classifiers /path/to/study_matrix.csv   # → output/metrics.json
python -m scripts.reproduce_figures /path/to/study_matrix.csv   # → output/figures/, output/shap_cache/
```

- Classifiers: Logistic Regression, Random Forest, SVM (3-fold grid on train, metrics on 30% hold-out).
- Figures: habitat map (SVM, *Thraupidae*), occurrence map, four SHAP plot types × both families under
  `output/figures/shap/` and `output/figures/maps/`.
- Check: *Thraupidae* waterfall instance must satisfy E[f(X)] ≈ 0.504 and f(x) ≈ 0.534 (Fig. 7).

## Phase B — Natural language explanations

```bash
pip install -r requirements-nle.txt
cp .env.example .env   # OPENAI_API_KEY, GOOGLE_API_KEY; optional JUDGE_MODEL
python -m nle.run --dry-run
python -m nle.run --experiment baseline_gpt41
```

- Experiment matrix: `configs/experiments.yaml` (core rows from Table `tab:full_expanded_results`).
- Model settings: `configs/model_params.yaml`.
- Prompts: `prompts/` (including `baseline_llm_synthesis.txt` for text-only baseline).
- OCR backends implemented: **Unstructured**, **Docling** (CLI on PATH). Other techniques in the manuscript
  (Dots, Paddle, Granite, Qwen, DeepSeek, Nanonets, etc.) are **not** automated here — reproduce via the same
  prompts and your provider of choice.

## Phase C — LLM-as-judge

```bash
python -m evaluation.run_judge
python -m evaluation.run_judge --cross-judge
python -m evaluation.aggregate
```

Rubric: `evaluation/judge_rubric.txt`. Primary judge defaults to `JUDGE_MODEL` (manuscript: GPT-5.1, temp 1.0);
cross-judge uses Gemini 2.5 Pro. Scores land in `output/scores/`; `aggregate.py` prints group means and Pearson
$r$ when both judges ran.

## Layout

```
configs/          model_params.yaml, experiments.yaml
scripts/          sdm_common, sdm_models, train_classifiers, reproduce_figures, run_all
prompts/          generation templates
nle/              run.py, providers (OpenAI, Gemini, OCR)
evaluation/       judge_rubric.txt, run_judge.py, aggregate.py
output/           metrics, figures, shap_cache, nle, scores (not committed)
```