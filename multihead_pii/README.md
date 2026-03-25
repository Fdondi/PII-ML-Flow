# Multi-Head PII

## What it implements

- Shared encoder (`ModernBERT`) with three heads:
  - proposal head (BIO token tags)
  - type head (`NONE` + PII label taxonomy)
  - sensitivity head (`REDACT|KEEP`) with continuous `redact_probability`
- Deterministic decoder for overlap resolution and final redaction spans.
- Standalone train/infer/evaluate entrypoints.

## Files

| File | Responsibility |
|---|---|
| `config.py` | Config dataclass + JSON load/save |
| `labels.py` | Label vocabularies and mappings |
| `dataset.py` | JSONL dataset adapters; canonical home for `_extract_value`, `LABEL_ALIASES`, and `REGEX_TYPE_MAP` |
| `model.py` | Shared encoder + multi-head architecture; `load_checkpoint` and `build_model_from_checkpoint` helpers |
| `losses.py` | Multitask losses |
| `decoder.py` | Postprocessing and conflict handling; canonical home for regex patterns (`EMAIL_PATTERN`, `PHONE_PATTERN`, `IPV4_PATTERN`, `IBAN_PATTERN`, `CREDIT_CARD_PATTERN`) |
| `train.py` | Training CLI |
| `infer.py` | Inference CLI |
| `evaluate.py` | Evaluation CLI |
| `span_credit.py` | Overlap credit scoring |
| `type_comparison.py` | Value-key comparison for TP/FP/FN accounting |
| `ui.py` | Streamlit PDF redaction UI |

### Shared utilities — single source of truth

| Symbol | Defined in | Used by |
|---|---|---|
| `EMAIL_PATTERN`, `PHONE_PATTERN`, `IPV4_PATTERN`, `IBAN_PATTERN`, `CREDIT_CARD_PATTERN` | `decoder.py` | `decoder.py`, `dataset.py` |
| `_extract_value` | `dataset.py` | `infer.py`, `evaluate.py` |
| `LABEL_ALIASES` | `dataset.py` | `dataset.py`, `train_modernbert_span_classifier.py` |
| `load_checkpoint`, `build_model_from_checkpoint` | `model.py` | `infer.py`, `evaluate.py` |

## Data formats

### Main dataset JSONL

Each row can be either:

1) Span-only format:

```json
{"text":"...", "spans":[{"start":10,"end":20,"label":"EMAIL"}]}
```

2) Rich item format:

```json
{"text":"...", "items":[{"start":10,"end":20,"category":"REAL_PII","label":"EMAIL"}]}
```

Sensitivity targets from `items` are mapped as:

- `REAL_PII` -> `REDACT`
- `PII_LOOKALIKE` -> `KEEP`

### Optional sensitivity companion JSONL

Use this when main rows only have `spans` and you want explicit stage-C supervision.
Rows must align 1:1 by line number with the main dataset:

```json
{"sensitivity_spans":[{"start":10,"end":20,"sensitivity":"REDACT"}]}
```

Allowed sensitivity values: `REDACT`, `KEEP`.

## BIO proposal tags

`BIO` is the token-level format used by the proposal head to mark candidate boundaries.

- `B-ENTITY`: first token of an entity-like span
- `I-ENTITY`: continuation token in the same span
- `O`: outside any entity-like span

Example:

```text
Text:   Jane  Doe   called  support@example.com
Tags:   B     I     O       B
```

How this is used in the pipeline:

1. The proposal head predicts BIO tags for each token.
2. BIO spans are decoded into candidate spans (high recall).
3. Candidate spans are merged with regex candidates (patterns from `decoder.py`).
4. Type and sensitivity heads score each candidate.
5. Decoder outputs final redaction decisions and confidence values.

This means BIO is only for boundary proposal, not final type/sensitivity decisions.

## Train

```bash
python -m multihead_pii.train --train train.gpt-5-nano.jsonl --valid valid.gpt-5-nano.jsonl --config configs/multihead_v1.json --output outputs_multihead
```

Add companion labels if needed:

```bash
python -m multihead_pii.train  --train train.gpt-5-nano.jsonl  --valid valid.gpt-5-nano.jsonl --train-sensitivity train.sensitivity.jsonl --valid-sensitivity valid.sensitivity.jsonl
```

## Inference

```bash
python -m multihead_pii.infer --input valid.gpt-5-nano.jsonl --checkpoint outputs_multihead/multihead_model.pt --output outputs_multihead/predictions.jsonl
```

`redactions` contains spans chosen by decoder thresholding, each with:
- `start`, `end`, `value` — character-level span
- `label` — PII type
- `decision` — `REDACT`
- `redact_score` — `type_confidence × redact_probability`
- `redact_probability`, `type_confidence` — raw head scores

## PDF redaction UI

Run a small UI to upload a PDF and view extracted redactions:

```bash
python -m streamlit run multihead_pii/ui.py
```

The UI:
- lets you pick a checkpoint path
- extracts text page-by-page from the uploaded PDF
- runs inference and shows detected redacted components
- renders a redacted text preview for each page

## Evaluate

```bash
python -m multihead_pii.evaluate  --valid valid.gpt-5-nano.jsonl --checkpoint outputs_multihead/multihead_model.pt --output outputs_multihead/eval_report.json
```

Evaluation includes both discrete and continuous sensitivity metrics:

- `sensitivity_candidate_accuracy` (thresholded/argmax)
- `sensitivity_redact_probability_mae`
- `sensitivity_redact_probability_brier`

Overlap-aware span scoring (for non-exact matches): if predicted span length is `N`, gold span length is `M`, and token overlap is `K > 0`, the overlap score is `1 / 2^(M + N - K)`. Exact matches still score `1.0`.

## Notes

- Checkpoints and reports are isolated under `outputs_multihead/`.
- `train_modernbert_span_classifier.py` is a legacy standalone script; it imports `LABEL_ALIASES` from this package rather than defining its own copy.
