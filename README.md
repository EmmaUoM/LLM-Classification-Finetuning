# LLM Preference Classification with Fine-tuning and MLflow

Predict whether a human prefers response A, response B, or a tie for the same prompt.
This project is the modular Python version of the updated three-model Kaggle notebook.
It includes real fine-tuning on the competition training set, data cleaning,
complete grouped five-fold development cross-validation, model bundles, and separate offline inference.

The default trains **DeBERTa, Gemma 2, and Llama 3 sequentially in every fold**.
Gemma/Llama initialize from the same local Kaggle checkpoints used by the reference
notebook, using 4-bit NF4 weights, LoRA adapters, and a trainable classification head.
A complete training manifest connects all folds to separate offline submission.
These Gemma/Llama checkpoints have already been fine-tuned on the competition task;
local development scores may therefore contain overlap with their original training data.

## Files

Keep the Python modules together in the repository root.

| File | Responsibility |
| --- | --- |
| `main.py` | Command-line entry point, run directories, logs, and failure handling |
| `config.py` | Arguments, shared class order, configuration, and logger |
| `data.py` | CSV validation and cleaning, grouped splits, A/B-swapped inputs, real labels, and paired semantic features |
| `models.py` | Custom DeBERTa, Kaggle fused-checkpoint compatibility, QLoRA factories, shared 4-bit bases, model bundles, and legacy loading |
| `inference.py` | Gemma 2, Llama 3, and custom classifier inference with A/B restoration |
| `evaluation.py` | Probability checks, weighted ensemble, and evaluation metrics |
| `tracking.py` | MLflow parameters, metrics, artifacts, failure records, and runtime/GPU monitoring |
| `utils.py` | Seeds, hashes, source snapshots, environment snapshots, manifests, and JSON |
| `pipeline.py` | Preflight checks, execution, complete-fold manifests, OOF statistics, and multi-fold submission |
| `submission_support.py` | Portable training manifests, Kaggle Input discovery, offline base resolution, and dependency wheels |
| `training.py` | Actual gradient updates, validation, early stopping, best checkpoints, and parameter-update audits |
| `requirements-mlops.txt` | Additional MLflow dependencies |
| `requirements.txt` | DeBERTa and tracking runtime |
| `requirements-qlora.txt` | Default three-model runtime, including Gemma/Llama QLoRA |
| `requirements-dev.txt` | Test dependencies |
| `configs/` | Training and inference JSON configuration examples |
| `tests/` | CPU behavior, checkpoint round-trip, CLI, and MLflow integration checks |

## Environment

Python 3.10 or newer is required. Keep the CUDA-enabled PyTorch build supplied
by Kaggle or your GPU environment, then install the three-model runtime:

```bash
python -m pip install -r requirements-qlora.txt
```

A DeBERTa-only run needs `requirements.txt` and `--train-backends deberta`.
The dependency overlay matches the notebook: Transformers 4.46.3, safetensors 0.8.0,
MLflow skinny 2.22.2, and PyArrow 23.0.1. Keep Kaggle's existing CUDA/PyTorch build.

QLoRA requires an NVIDIA CUDA GPU and compatible bitsandbytes binaries. The
notebook's old Python-3.10-specific Triton/xformers wheels are no longer required.
One model is loaded on `cuda:0` at a time. The loader preserves the original Kaggle
checkpoints' fused `gate_up_proj` and multi-layer `score` head using standard
Transformers attention. Missing or unexpected weights in this layout are errors,
so a path change cannot silently leave part of the model randomly initialized.

The DeBERTa backbone may be downloaded during training or supplied as a local
standard Hugging Face directory. Gemma/Llama training requires complete local
checkpoints; missing Inputs and Hub IDs fail before any tokenizer or model download.

## Kaggle checkpoint Inputs

Add the following Inputs in the Kaggle notebook before running this project:

| Model or asset | Source / legacy mounted directory |
| --- | --- |
| Gemma 2 | [tascj0/lmsys-checkpoints-0-0805](https://www.kaggle.com/datasets/tascj0/lmsys-checkpoints-0-0805), `/kaggle/input/lmsys-checkpoints-0-0805` |
| Llama 3 | [tascj0/lmsys-checkpoints-3-0805](https://www.kaggle.com/datasets/tascj0/lmsys-checkpoints-3-0805), `/kaggle/input/lmsys-checkpoints-3-0805` |
| Sentence Transformer | Reference notebook's `some-pack`, containing `sentence-transformer-model/modules.json` |
| Competition CSVs | LLM Classification Finetuning competition Input |

Both legacy mounts and `/kaggle/input/datasets/<owner>/<slug>` are recognized.
Discovery also finds a unique model directory nested under a mounted dataset.
It checks `config.json` for model checkpoints and `modules.json` for Sentence
Transformers. Missing or ambiguous Inputs fail with instructions to attach the
dataset or select an exact path; no guessed path or Hub fallback is returned.
Gemma/Llama read these complete local checkpoints without a Hugging Face login.
DeBERTa retains the current paired-feature architecture and defaults to
`microsoft/deberta-v3-base`; provide a standard local backbone with
`--model-source deberta=...` to avoid downloading it. The reference notebook's old
`custom_model_complete.pth` has a different architecture and is not used by this
training pipeline.

The reference source is [blue0924/finetuning-test2](https://www.kaggle.com/code/blue0924/finetuning-test2).
Local model loading does not install Python packages. Initial dependency installation
and the default export of offline wheels still require Internet access.

## Train

Attach the Kaggle competition data and enable GPU access. By default, training
visits all five grouped folds. Each fold initializes fresh models, trains on its
other four partitions for up to three epochs, and saves its own best checkpoints:

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
    --single-fold --max-train-rows 1000 --max-val-rows 300 --epochs 1 \
    --no-export-offline-dependencies
```

For an intentionally DeBERTa-only run with locally supplied models:

```bash
python main.py --mode train \
    --input-root /kaggle/input --output-dir /kaggle/working \
    --train-backends deberta --no-export-offline-dependencies \
    --model-source deberta=/kaggle/input/my-deberta-base \
    --sentence-model /kaggle/input/my-sentence-transformer \
    --local-files-only
```

All three branches are enabled by default; this is the equivalent explicit command:

```bash
python main.py --mode train --input-root /kaggle/input \
    --output-dir /kaggle/working --train-backends deberta gemma llama
```

Change individual model directories or pin the DeBERTa Hub revision with repeated
`--model-source BACKEND=VALUE` and `--model-revision BACKEND=COMMIT_HASH` arguments.
DeBERTa initializes from its standard pretrained backbone, while Gemma/Llama use
the local Kaggle sources above. Gemma/Llama accept local standard Hugging Face
directories as well as the Kaggle fused layout. `--base-model BACKEND=DIRECTORY`
and JSON `base_model_overrides` also override training sources; they take precedence
over `model_sources` and are reapplied during validation, preflight, and model loading.
Changing an override invalidates the previous source check, including in fold copies.

Startup logs include `Pipeline version: kaggle-local-sources-v2` and the resolved
training source for each enabled model. Default Kaggle Inputs are resolved before
importing the ML stack. Submission only uses saved bundles and bases; it does not require the
original checkpoint or Sentence Transformer Inputs. If using this project from an
interactive notebook after replacing source files, restart the session and rerun
configuration before training. Configuration objects from an older module are rejected.

Use `python main.py --help` for CLI options. A JSON configuration can set any
`Config` field; explicit CLI values override it.

Full cross-validation is the default (`cross_validation: true`, `n_splits: 5`).
For one selected fold, use `--single-fold --holdout-fold 2`, or set
`cross_validation: false` in JSON. `--cross-validation` overrides a single-fold
JSON configuration. Keep `max_train_rows` and `max_val_rows` unset for a full-data
run. With three backends enabled, five folds train 15 models sequentially; expect
roughly five times the work of the old single-fold run.

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

Development folds are produced by `StratifiedGroupKFold` using the complete
normalized prompt as the group. All five folds are visited by default, with
approximately 20% validation in each fold. The same prompt and duplicate response
pair cannot cross a fold's training/validation split. The seed and split definition
stay fixed across folds, so every cleaned row is validated exactly once in a
full-data run. The pipeline rejects duplicate or missing out-of-fold predictions.
Only the training partition receives random A/B augmentation; labels and features
are swapped consistently. Model-name and fold columns are not used as features.

Three text components receive separate token budgets, with unused space assigned
to longer components. This avoids truncating away the whole second answer just
because it appears at the end of a concatenation.

Each fold's validation set selects its best epoch and supports early stopping.
Fold and out-of-fold scores are development estimates, not an untouched final test.
A competition-finetuned starting checkpoint may already have seen validation
examples; audit its provenance before interpreting scores.
There is no implicit full-data retraining after model selection.

## Semantic features

The custom classifier uses a DeBERTa text representation plus two cosine features:
`cos(prompt_i, response_a_i)` and `cos(prompt_i, response_b_i)`. The frozen sentence
encoder is saved with the custom model bundle. Direct MLP feature fusion replaces
the old single-key attention block.

Cosine features compare each response with its own prompt. They do not search other
prompts in the current test batch, so a row's features are independent of unrelated
test rows. Training and inference use the same paired-feature definition.

## Saved models and inference

Every execution gets an isolated directory under `<output-dir>/runs/`. A default
cross-validation run contains:

```text
runs/<run_id>/
  config.json and pipeline.log
  cv_progress.json
  fold_metrics.csv
  oof_predictions.csv
  cv_summary.json
  result.json and run_summary.json
  folds/fold_0/ ... folds/fold_4/
```

`fold_metrics.csv` contains each fold's ensemble validation metrics.
`cv_summary.json` includes their unweighted means and sample standard deviations,
plus metrics computed from pooled out-of-fold (OOF) probabilities. These are
separate summaries; pooled metrics need not equal the fold means. OOF rows retain
the cleaned source order, with one prediction per validated ID and a `fold` column.
The summary records validation coverage and whether row limits were enabled;
sampled debugging runs are not presented as full-data validation.
`cv_progress.json` retains completed fold IDs if training fails partway through.
`training_preflight.json` records source accessibility before the first model trains.
Each fold's `backend_status.json` records each requested model's progress.
`<output-dir>/training_manifest.json` is marked `complete` only when every requested
model and fold has finished. A failed or interrupted group is not auto-submitted.

Each fold directory (or a `--single-fold` run directory) contains:

```text
runs/<run_id>/
  config.json
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
LoRA adapters and the trained classifier head. Training also exports one shared
4-bit base per Gemma/Llama model under `submission_assets/base_models/<group>/`,
without merging any fold's LoRA. Adapters refer to these bases using relative paths.
The exported base retains the layout marker needed to reload its fused MLP/head.
Dependency wheels are saved under `submission_assets/wheels/` for the same Python
major/minor version. `--no-export-quantized-bases` and
`--no-export-offline-dependencies` opt out of these exports.
Bundles and complete manifests produced by the updated notebook are supported.

Training mode does not export a competition submission. After training finishes,
change only the mode while keeping the same configuration:

```bash
python main.py --config configs/train.json --mode submit
```

Auto-discovery reads the complete manifest under the output directory or attached
Kaggle Inputs. For a new Kaggle session, save the training version's output, attach
it as Input with its `training_manifest.json`, `runs/`, and `submission_assets/`
directories intact, and run:

```bash
python main.py --config configs/submit.example.json
```

Each model's saved folds are predicted separately, averaged with equal fold weights,
then combined using the configured model weights. A three-model training group is
not automatically reduced to DeBERTa if another model or fold is missing.
The entry point installs saved dependency wheels offline before importing the ML
stack; use `--no-install-offline-dependencies` when dependencies are already managed.
The wheelhouse records the required Python major/minor version.

Explicitly select saved folds when needed. Repeat `--bundle` for multiple folds of
the same backend; a single explicit model remains supported:

```bash
python main.py --mode submit --output-dir /kaggle/working \
    --bundle deberta=/kaggle/input/my-trained-models/runs/RUN/folds/fold_0/models/deberta \
    --bundle deberta=/kaggle/input/my-trained-models/runs/RUN/folds/fold_1/models/deberta
```

JSON `inference_bundles` values can be either one path or a list of fold paths.
These directories must be this pipeline's saved bundles, not raw Kaggle checkpoint
Inputs. For older adapter-only outputs, repeat `--base-model BACKEND=LOCAL_BASE`.
Use `--discover-root DIRECTORY` for training outputs outside the standard roots.
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
and run status. Each fold has its own tracking run; a separate summary run records
CV means, standard deviations, and OOF metrics after all folds finish, avoiding
overlapping active MLflow runs. Source snapshots include all maintained modules, tests, configuration
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
complete five-fold training, exact OOF coverage, independent model initialization,
CV metric aggregation and failure reporting,
A/B feature/label restoration, probability checks, real CPU gradient updates using
tiny DeBERTa/Llama models, model save/reload equivalence, offline submission without
training data, parameter audits, source hashing, and real MLflow SQLite success/failure
logging. Tiny random models are test fixtures, not claimed predictive results.
Full-size GPU training, 4-bit CUDA execution, and Kaggle leaderboard scoring require
the actual environment and data; no performance scores are prefilled here.
