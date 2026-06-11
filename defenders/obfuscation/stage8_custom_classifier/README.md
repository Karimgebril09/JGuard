# Stage 8 Custom Classifier (BeaverTails Baseline)

This module trains and serves a TF-IDF + Logistic Regression baseline classifier for stage 8.

Primary objective: `safe` vs `unsafe`.

Secondary objective: when predicted `unsafe`, also emit a harmfulness category.

## Tasks

Primary output:

- `safe`
- `unsafe`

Secondary output (only for unsafe predictions):

- `data_exposure`
- `illegal_actions`
- `toxicity`
- `hallucination_facilitation`

## Train

Run from repository root:

```bash
python -m defenses.obfuscation.stage8_custom_classifier.training_baseline --split 30k_train
```

Example with optional downsampling caps for larger splits:

```bash
python -m defenses.obfuscation.stage8_custom_classifier.training_baseline \
  --split 330k_train \
  --max-unsafe-per-class 12000 \
  --safe-samples 5000 \
  --max-total-rows 50000
```

Artifacts are written to `defenses/obfuscation/stage8_custom_classifier/artifacts/`.

Generated artifacts:

- `tfidf_vectorizer.joblib`
- `binary_classifier.joblib`
- `binary_label_encoder.joblib`
- `unsafe_category_classifier.joblib`
- `unsafe_category_label_encoder.joblib`
- `report.txt`
- `report.json`
- `metadata.json`

## Fine-Tune DistilBERT

Run from repository root:

```bash
python -m defenses.obfuscation.stage8_custom_classifier.training_finetune_distilbert --split 30k_train --epochs 2
```

This trains two transformer classifiers:

- Primary: binary `safe` vs `unsafe`
- Auxiliary: unsafe category (`data_exposure`, `illegal_actions`, `toxicity`, `hallucination_facilitation`)

Fine-tuned artifacts are written to `defenses/obfuscation/stage8_custom_classifier/distilbert_artifacts/` (or your custom `--output-dir`), including `report.txt`, `report.json`, and `metadata.json`.

## Use In Pipeline

```python
from defenses.obfuscation.pipeline import run_obfuscation_pipeline
from defenses.obfuscation.stage8_custom_classifier import load_stage8_custom_classifier

classifier = load_stage8_custom_classifier(
    "defenses/obfuscation/stage8_custom_classifier/artifacts"
)

result = run_obfuscation_pipeline(
    "some prompt text",
    stage8_classifier=classifier,
)
```

The returned structure is compatible with `classify_stage8` expectations in stage 8.
