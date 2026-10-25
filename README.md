# LLM Classification Ensemble with MLflow

A modular version of `finetuning_test2_mlops.py`. It runs inference with existing
Gemma2, Llama3, and custom DeBERTa checkpoints and combines their predictions.
All comments and docstrings are in English. This project does not train models.

## Files

Keep all of these files together in the same directory.

| File | Responsibility |
| --- | --- |
| `main.py` | Command-line entry point, run directories, logs, and failure handling |
| `config.py` | Arguments, shared class order, and logger |
| `data.py` | CSV validation, A/B-swapped inputs, real labels, and FAISS features |
| `models.py` | Custom DeBERTa architecture and legacy checkpoint loading |
| `inference.py` | Gemma2, Llama3, and custom classifier inference |
| `evaluation.py` | Probability checks, weighted ensemble, and evaluation metrics |
| `tracking.py` | MLflow parameters, metrics, artifacts, and runtime/GPU monitoring |
| `utils.py` | Seeds, hashes, source snapshots, environment snapshots, and JSON |
| `pipeline.py` | Execution of all pipeline stages |
| `requirements-mlops.txt` | Additional MLflow dependencies |

## Environment

Use the GPU environment and model files from the original notebook. The pipeline
requires two NVIDIA CUDA GPUs; the large LLMs are split across `cuda:0` and `cuda:1`.
The original runtime used Python 3.10, torch 2.1.2, transformers 4.42.3,
triton 2.2.0, a custom xformers wheel, sentence-transformers 3.3.1,
faiss-cpu 1.7.2, numpy 1.26.4, pandas, pyarrow, tqdm, and accelerate.
`requirements-mlops.txt` adds tracking dependencies to that environment; it does
not reconstruct the original custom GPU environment or download model weights.

Install the tracking dependencies in that environment:

```bash
python -m pip install -r requirements-mlops.txt
```

The SQLAlchemy version constraint preserves compatibility with MLflow 2.22.2.

## Inputs

Under `--input-root`, the existing notebook inputs must be available at:

- `llm-classification-finetuning/test.csv`
- `lmsys-modules-0805/`
- `lmsys-checkpoints-0-0805/`
- `lmsys-checkpoints-3-0805/`
- `some-pack/sentence-transformer-model/`
- `akemiiiiii/custom_model_dir/custom_model_complete.pth`

The custom model directory must also contain its tokenizer files.
The CSV requires unique non-null `id` values and string columns `prompt`,
`response_a`, and `response_b`. Text strings retain their original notebook
representation. Use `--test-csv /path/to/test.csv` to select another input CSV.

## Run

From the directory containing `main.py`:

```bash
python main.py --input-root /input
```

Display all available arguments:

```bash
python main.py --help
```

MLflow is enabled by default. Each execution creates an isolated directory under
`output/`, named with a UTC timestamp and a unique suffix. The console prints its
exact location. The directory contains `submission.csv`, three prediction NPY
files, original/swapped Parquet inputs, JSON metadata, and `pipeline.log`.

Default ensemble weights are Gemma2 = 0.5, Llama3 = 0.25, and custom DeBERTa = 0.25.
Llama receives swapped responses, and its class order is restored before ensembling.
Override weights with `--ensemble-weights GEMMA LLAMA CUSTOM`.
As in the original script, each response's semantic feature is its maximum
similarity to any prompt in the input batch, rather than just its paired prompt.

## MLflow

The default tracking backend is `mlflow.db` in the current working directory.
Source/configuration artifacts and run metadata are stored locally in `mlartifacts/`.
Open the UI from the same directory:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

Open `http://localhost:5000` in a browser. Use `--tracking-uri` or
`MLFLOW_TRACKING_URI` to select another backend or tracking server. Its configured
artifact storage then applies. Prediction files are copied to MLflow only when
`--log-predictions` is supplied. Raw input text, labels, and model weights are not
uploaded. Every run tracks all project source files and a combined source hash,
input fingerprints, installed package versions, configuration, stage durations,
and peak GPU memory where available. Failed runs retain local diagnostics and
are marked failed in MLflow when a tracking run has been started.

Disable MLflow while retaining local metrics and outputs:

```bash
python main.py --input-root /input --disable-mlflow
```

## Optional evaluation

Provide a separate labels CSV containing exactly the input IDs, in any row order:

```csv
id,winner_model_a,winner_model_b,winner_tie
101,1,0,0
102,0,1,0
103,0,0,1
```

Each row must have exactly one winner. The example IDs must be replaced with
your own input IDs. Evaluate on held-out examples:

```bash
python main.py --input-root /kaggle/input \
    --test-csv validation.csv --labels-csv validation_labels.csv \
    --run-name validation-ensemble
```

Accuracy and multiclass log loss are computed for each model and the ensemble.
Without `--labels-csv`, only confidence, entropy, and predicted class fractions
are recorded. The placeholder labels required by the original data loader are
never used to calculate evaluation metrics.

## Reproducibility and checkpoints

Set `--seed` to control the Python, NumPy, and torch seeds. Optional
`--deterministic` enables deterministic torch settings with warnings for
unsupported operations; custom GPU kernels may remain nondeterministic.
Input CSVs and project source files receive full SHA-256 fingerprints. Large
checkpoint files use size/mtime metadata by default. Add `--hash-checkpoints`
to hash every input byte, which can take longer for large models.

The custom checkpoint stores a full Python model object. `models.py` retains
the architecture and resolves `CustomDebertaModel` saved under the original
notebook or single-script module names. Load checkpoints from trusted sources.

## Verification

The module split was checked for Python 3.10 syntax, imports, CSV/Parquet input
preparation, probability checks, label alignment, A/B restoration, source hashes,
and real MLflow SQLite success/failure logging using synthetic model predictions.
The compatibility unpickler was checked with a synthetic legacy model object.
Actual GPU inference requires the original model artifacts and was not run here.
