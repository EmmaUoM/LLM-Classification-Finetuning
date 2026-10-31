# LLM Preference Classification with Fine-tuning and MLflow

Predict whether a human prefers response A, response B, or a tie for the same prompt.
This project is the modular Python version of the updated `finetuning-emma.ipynb`.
It includes real fine-tuning on the competition training set, data cleaning,
group-disjoint development validation, model bundles, and separate offline inference.

The default trains the custom DeBERTa branch. Gemma 2 and Llama 3 can also be
fine-tuned sequentially using 4-bit NF4 weights, LoRA adapters, and a trainable
classification head. General pretrained models are the default starting points.

## Files

Keep the Python modules together in the repository root.

| File | Responsibility |
| --- | --- |
| `main.py` | Command-line entry point, run directories, logs, and failure handling |
| `config.py` | Arguments, shared class order, configuration, and logger |
| `data.py` | CSV validation and cleaning, grouped splits, A/B-swapped inputs, real labels, paired semantic features, and optional FAISS features |
| `models.py` | Custom DeBERTa architecture, QLoRA factories, model bundles, and legacy checkpoint loading |
| `inference.py` | Gemma 2, Llama 3, and custom classifier inference with A/B restoration |
| `evaluation.py` | Probability checks, weighted ensemble, and evaluation metrics |
| `tracking.py` | MLflow parameters, metrics, artifacts, failure records, and runtime/GPU monitoring |
| `utils.py` | Seeds, hashes, source snapshots, environment snapshots, manifests, and JSON |
| `pipeline.py` | Execution of all pipeline stages |
| `training.py` | Actual gradient updates, validation, early stopping, best checkpoints, and parameter-update audits |
| `requirements-mlops.txt` | Additional MLflow dependencies |
| `requirements.txt` | Full default DeBERTa runtime, including MLOps dependencies |
| `requirements-qlora.txt` | Optional Gemma/Llama QLoRA runtime |
| `requirements-faiss.txt` | Optional FAISS retrieval-feature dependencies |
| `requirements-dev.txt` | Test dependencies |
| `configs/` | Training and inference JSON configuration examples |
| `tests/` | CPU behavior, checkpoint round-trip, CLI, and MLflow integration checks |

## Environment

Python 3.10 or newer is required. Keep the CUDA-enabled PyTorch build supplied
by Kaggle or your GPU environment, then install:

```bash
python -m pip install -r requirements.txt
```

For optional Gemma/Llama QLoRA:

```bash
python -m pip install -r requirements-qlora.txt
```

QLoRA requires an NVIDIA CUDA GPU and compatible bitsandbytes binaries. The
notebook's old Python-3.10-specific Triton/xformers wheels are no longer required.
One model is loaded on `cuda:0` at a time; the old two-GPU custom inference modules
are not used by the new training pipeline.

Hugging Face models may be downloaded during training, or provided as local
standard Hugging Face directories with `--local-files-only`. Gemma/Llama access
may require accepting the model license and authenticating through your existing
Hugging Face login or environment. Never put access tokens in committed files.

## Train

Attach the Kaggle competition data and enable GPU access. By default, all cleaned
rows assigned to the training partition are used for three epochs:

```bash
python main.py --mode train \
    --input-root /kaggle/input \
    --output-dir /kaggle/working
```

Or use the example configuration:

```bash
python main.py --config configs/train.json
```

Start with a smaller end-to-end run before a complete training job:

```bash
python main.py --mode train \
    --input-root /kaggle/input \
    --output-dir /kaggle/working \
    --max-train-rows 1000 --max-val-rows 300 --epochs 1
```

Use local pretrained models when Internet is unavailable:

```bash
python main.py --mode train \
    --input-root /kaggle/input --output-dir /kaggle/working \
    --model-source deberta=/kaggle/input/my-deberta-base \
    --sentence-model /kaggle/input/my-sentence-transformer \
    --local-files-only
```

Train all three branches after installing the optional QLoRA runtime:

```bash
python main.py --mode train --input-root /kaggle/input \
    --output-dir /kaggle/working --train-backends deberta gemma llama
```

Change individual model directories or pin Hub revisions with repeated
`--model-source BACKEND=VALUE` and `--model-revision BACKEND=COMMIT_HASH` arguments.
Models initialize from public DeBERTa-v3-base, Gemma-2-9b-it, and
Meta-Llama-3-8B-Instruct by default. Old custom inference checkpoints are not
assumed to be standard Hugging Face bases.

Use `python main.py --help` for CLI options. A JSON configuration can set any
`Config` field; explicit CLI values override it.

## Cleaning and validation

Input CSVs need unique test IDs and the columns `prompt`, `response_a`, and
`response_b`. The training CSV also needs the real one-hot target columns:

```text
winner_model_a, winner_model_b, winner_tie
```

Cleaning safely parses JSON/Python-list conversation strings without `eval`,
normalizes Unicode and control characters, and preserves case, punctuation,
line structure, and code indentation. Invalid training labels/rows are reported
and removed. Duplicate pairs include A/B-reversed pairs; conflicting pair labels
are removed together. Test rows and their ID order are preserved.

The development holdout is produced by `StratifiedGroupKFold` using the complete
normalized prompt as the group. The default is one of five folds, approximately
20% validation. The same prompt and duplicate response pair cannot cross the split.
Only the training partition receives random A/B augmentation; labels and features
are swapped consistently. Model-name and fold columns are not used as features.

Three text components receive separate token budgets, with unused space assigned
to longer components. This avoids truncating away the whole second answer just
because it appears at the end of a concatenation.

The holdout selects the best epoch and supports early stopping. It is a development
set, not an untouched final test. A competition-finetuned starting checkpoint may
already have seen holdout examples; audit its provenance before interpreting scores.
There is no implicit full-data retraining after model selection.

## Semantic features and FAISS

The custom classifier uses a DeBERTa text representation plus two cosine features:
`cos(prompt_i, response_a_i)` and `cos(prompt_i, response_b_i)`. The frozen sentence
encoder is saved with the custom model bundle. Direct MLP feature fusion replaces
the old single-key attention block.

Paired cosine is the default and does not require FAISS. It does not search other
prompts in the current test batch, so a row's features are independent of unrelated
test rows. `data.FaissPromptIndex` is an optional utility for retrieval experiments:
fit it on training-prompt embeddings only, then transform response embeddings
against that fixed reference index. Install `requirements-faiss.txt` to use it.
Its retrieval features must not be silently substituted for the paired features
expected by a trained model.

## Saved models and inference

Every execution gets an isolated directory under `<output-dir>/runs/`:

```text
runs/<run_id>/
  config.json
  pipeline.log
  source/ and source_manifest.json
  data_quality_report.json
  train_split.csv and validation_split.csv
  models/<backend>/
    bundle_config.json and manifest.json
    training_history.csv
    training_update_audit.json
    validation_metrics.json and confusion_matrix.csv
    MODEL_CARD.md
  ensemble_validation.json
  result.json and run_summary.json
```

The custom bundle contains backbone configuration, fine-tuned safetensors,
tokenizer, and its sentence encoder when enabled. Gemma/Llama bundles contain
LoRA adapters and the trained classifier head; attach the matching base weights
separately for offline inference. Bundles produced by the updated notebook use
the same architecture version and can be loaded here.

Training mode does not export a competition submission. In a separate inference
run, supply a saved bundle and the current test data:

```bash
python main.py --mode submit \
    --input-root /kaggle/input --output-dir /kaggle/working \
    --bundle deberta=/kaggle/input/my-trained-models/runs/TRAIN_RUN_ID/models/deberta
```

Or edit the placeholder path in `configs/submit.example.json` and pass `--config`.
For Gemma/Llama, also repeat `--base-model BACKEND=LOCAL_BASE_DIRECTORY` as needed.
Submission mode forces offline model loading and never reads `train.csv`.

Both A/B orders are evaluated by default, restored to the original class order,
and averaged. Use `--no-tta` to disable this. Available models are combined using
fixed weights Gemma=0.50, Llama=0.25, DeBERTa=0.25, normalized over the explicitly
provided models. Override with `--ensemble-weights GEMMA LLAMA DEBERTA`.

The final file is `<output-dir>/submission.csv`, with columns:

```text
id,winner_model_a,winner_model_b,winner_tie
```

For Kaggle, use `/kaggle/working` as the output directory. Save and run the inference
notebook with Internet disabled, then submit the successful version's output.
A plain Python project can be attached as an Input or copied into the notebook's
working directory before invoking `python main.py ...`.

## Evaluation and MLflow

Training records actual train loss, validation log loss, accuracy, macro-F1,
confusion matrices, and trainable-parameter counts. Cross-entropy, backward,
AdamW, gradient accumulation/clipping, mixed precision, and scheduling perform
real updates. `training_update_audit.json` compares selected parameter tensors
before and after training. Best epoch selection uses validation log loss.

Inference without labels records confidence, entropy, predicted class fractions,
and descriptive input length statistics. It does not fabricate accuracy or loss.
To evaluate on separately labeled held-out rows, provide `--labels-csv PATH`;
its IDs must match the inference CSV exactly and its three winner columns must
contain valid one-hot labels. Rows are joined by ID, not assumed to share order.

MLflow is enabled by default and stores metadata in `<output-dir>/mlflow.db`,
with artifacts in `<output-dir>/mlartifacts/`. It tracks configuration, source/data
fingerprints, package versions, stage times, peak GPU allocation, model artifacts,
and run status. Source snapshots include all maintained modules, tests, configuration
examples, and dependency files. Failures retain local diagnostics and are marked
failed when a tracking run has started.

```bash
mlflow ui --backend-store-uri sqlite:////absolute/path/to/output/mlflow.db --port 5000
```

Open `http://127.0.0.1:5000` locally. If copying output to a different machine,
copy the model/artifact folders as well; stored artifact URIs point to their original
absolute locations and need relocation for browsing them in a moved MLflow UI.
The portable model bundles can be loaded directly without the tracking database.
Use `--disable-mlflow` to keep local reports only, or `--no-log-models` to avoid
copying model weights into MLflow's artifact store.

## Legacy checkpoint compatibility

`models.LegacyCustomDebertaModel` retains the old architecture.
`models.load_custom_checkpoint(path, trusted=True)` resolves old notebook/module
class locations when loading a trusted full-object checkpoint. New training and
inference use safetensors/PEFT bundles and do not invoke that pickle loader.
Legacy models expect their original feature definition and are not interchangeable
with the new paired-feature classifier.

## Verification

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

Tests exercise safe parsing, label QA, duplicate/conflict handling, grouped splits,
A/B feature/label restoration, probability checks, real CPU gradient updates using
tiny DeBERTa/Llama models, model save/reload equivalence, offline submission without
training data, parameter audits, source hashing, and real MLflow SQLite success/failure
logging. Tiny random models are test fixtures, not claimed predictive results.
Full-size GPU training, 4-bit CUDA execution, and Kaggle leaderboard scoring require
the actual environment and data; no performance scores are prefilled here.
