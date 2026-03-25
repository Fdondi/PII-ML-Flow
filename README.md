# How to run

* (Suggested): create virtual python environment.
* `pip install -r requirements.txt`
* add an `.env` file with `OPENAI_API_KEY` for generation (or otherwise set the environment variable)
* generate data:
  ```bash
  python generate_challenging_span_data.py --train-size 1500 --valid-size 300 --model gpt-5-nano --max_dollars 3
  ```
* (suggested) inspect data quality:
  ```bash
  python export_declared_spans.py   # writes human/LLM-readable .review.md files
  ```
* train, evaluate, and infer using the `multihead_pii` package (see [`multihead_pii/README.md`](multihead_pii/README.md)):
  ```bash
  python -m multihead_pii.train    --train train.jsonl --valid valid.jsonl --config configs/multihead_v1.json --output outputs_multihead
  python -m multihead_pii.evaluate --valid valid.jsonl --checkpoint outputs_multihead/multihead_model.pt
  python -m multihead_pii.infer    --input valid.jsonl --checkpoint outputs_multihead/multihead_model.pt --output outputs_multihead/predictions.jsonl
  ```

## MLflow tracking

MLflow experiment tracking is built into the training pipeline.
When `mlflow` is installed, every `train` run is logged automatically — no extra flags needed.

```bash
mlflow ui   # visit http://127.0.0.1:5000 to browse runs
```

See [`multihead_pii/README.md`](multihead_pii/README.md#mlflow-tracking) for the full list of logged parameters, metrics, and artifacts.

### Hyperparameter search with Optuna + MLflow

`hparam_search.py` runs Optuna TPE trials and logs each one as a separate MLflow run.
With MLflow 3.x the Optuna study is persisted via `MlflowStorage`, so searches can be paused and resumed.

```bash
python hparam_search.py \
  --train train.jsonl \
  --valid valid.jsonl \
  --n-trials 20 \
  --experiment pii-hparam-search \
  --output-root hparam_outputs
```

Key flags:

```
--n-trials INT       Number of Optuna trials (default: 10)
--experiment TEXT    MLflow experiment name shared by all trials (default: pii-hparam-search)
--base-config PATH   Base config JSON to override with sampled hyperparameters
--output-root PATH   Root directory; one subdirectory per trial (default: hparam_outputs)
--seed INT           TPE sampler seed (default: 0)
```

## Overlap-aware span scoring

Training and evaluation include partial credit for non-exact overlapping spans.
If predicted span length is `N`, gold span length is `M`, and token overlap is `K > 0`, the overlap score is:

`1 / 2^(M + N - K)`

Exact matches still score `1.0`, and overlap-based metrics are reported alongside exact-match metrics in evaluation output.

## Local generation with LM Studio

Use this when you want local generation instead of OpenAI-hosted models.

1. Start LM Studio.
2. Load a model in LM Studio.
3. Start the OpenAI-compatible local server in LM Studio (`http://127.0.0.1:1234/v1` by default).
4. Copy the loaded model id from LM Studio (this is the value for `--model` when using `--local-base-url`).

Run:

```bash
python generate_challenging_span_data.py \
  --local \
  --model "<lmstudio-model-id>" \
  --local-base-url "http://127.0.0.1:1234/v1" \
  --train-size 1500 \
  --valid-size 300
```

Direct in-process GGUF loading (without LM Studio API):

```bash
python generate_challenging_span_data.py \
  --local \
  --model "E:\path\to\your-model.gguf" \
  --train-size 1500 \
  --valid-size 300
```

Notes:

- `--model` is used for both hosted and local runs.
- `--local` forces local mode.
- `--local-base-url` also enables local mode automatically.
- With `--local-base-url`, `--model` must be the model id served by LM Studio (not a `.gguf` file path).
- `--local-api-key` is optional; default is `lm-studio`.
- If `--local` is set and `--local-base-url` is omitted, the script tries `http://127.0.0.1:1234/v1` automatically and uses it when `--model` is found in `/v1/models`.
- If direct in-process GGUF loading fails on your machine (for example with a native `llama-cpp` crash), use LM Studio API mode as above.
