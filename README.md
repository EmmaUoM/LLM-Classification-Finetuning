# LLM Human Preference Prediction

An inference pipeline that predicts which of two language-model responses a human would prefer: **Response A**, **Response B**, or a **tie**. It combines Gemma 2 and Llama 3 sequence classifiers with a custom DeBERTa classifier that incorporates semantic similarity features, then exports a probability-based `submission.csv`.

The supplied script loads existing model checkpoints. It does not implement model fine-tuning, training, or performance evaluation.

## Pipeline

1. **Prepare inputs:** Read the test CSV, validate its required columns, and create original and A/B-swapped Parquet files. Add placeholder label columns required by the dataset loader.
2. **Run Gemma 2:** Predict class probabilities using the original response order.
3. **Run Llama 3:** Predict class probabilities using the swapped response order.
4. **Extract semantic features:** Encode prompts and responses with SentenceTransformer, normalise the embeddings, and search a FAISS inner-product index to obtain two similarity features per row.
5. **Run the custom classifier:** Combine tokenised prompt/response text with the similarity features using the saved DeBERTa model.
6. **Ensemble predictions:** Restore Llama's original A/B class order and combine all three probability arrays with fixed weights.

| Model | Input order | Ensemble weight |
| --- | --- | --- |
| Gemma 2 | Original A/B | 0.50 |
| Llama 3 | Swapped B/A, then class order restored | 0.25 |
| Custom DeBERTa + similarity features | Original A/B | 0.25 |

The final probability for each class is `0.50 × Gemma + 0.25 × Llama + 0.25 × Custom`.

## Requirements

### Hardware

- **Two NVIDIA CUDA GPUs**, visible as `cuda:0` and `cuda:1`.
- Sufficient GPU memory for the supplied checkpoints and configured token budgets.
- Disk space for model assets, intermediate files, and outputs.

The script explicitly requires at least two CUDA GPUs. Its LLM layers are split across the first two GPUs, and the custom classifier runs on `cuda:0`. There is no CPU, single-GPU, or Apple MPS execution path in the supplied code. Minimum GPU memory is not specified.

### Software

The script records the following original notebook environment:

| Dependency | Version or requirement |
| --- | --- |
| Python | 3.10 |
| PyTorch | 2.1.2, with CUDA support |
| Transformers | 4.42.3 |
| Triton | 2.2.0 |
| xFormers | Original notebook's custom wheel |
| SentenceTransformers | 3.3.1 |
| FAISS | `faiss-cpu==1.7.2` |
| NumPy | 1.26.4 |
| Other Python packages | `pandas`, `pyarrow`, `tqdm`, `accelerate` |
| Custom model code | Original `lmsys-modules-0805` assets |

These versions are recorded from the source comments, not a verified dependency lockfile. Reuse the original notebook's compatible CUDA/PyTorch/Triton/xFormers setup. A generic xFormers installation is not established as a replacement for its custom wheel.

## Required assets

All paths below are relative to `--input-root`, which defaults to `/kaggle/input`.

| Path | Required contents |
| --- | --- |
| `llm-classification-finetuning/test.csv` | Test data |
| `lmsys-modules-0805/` | Original custom package contents, including `data/`, `models/`, and utilities |
| `lmsys-checkpoints-0-0805/` | Gemma 2 checkpoint, configuration, and tokenizer |
| `lmsys-checkpoints-3-0805/` | Llama 3 checkpoint, configuration, and tokenizer |
| `some-pack/sentence-transformer-model/` | SentenceTransformer model assets |
| `akemiiiiii/custom_model_dir/` | Custom classifier checkpoint and tokenizer assets |
| `akemiiiiii/custom_model_dir/custom_model_complete.pth` | Complete saved PyTorch model object |

The model files, custom wheel, and auxiliary modules must be obtained separately from the original notebook inputs. The script does not download them.

At runtime, `lmsys-modules-0805/` is copied into `<output-dir>/human_pref/` and the output directory is added to Python's module search path. The source directory must therefore contain the package contents directly. The LLM implementations must provide the custom `forward_part1` and `forward_part2` methods used by the script.

## Setup

The examples assume the supplied Python code has been saved as `finetuning_test2.py`.

1. Create a Python 3.10 environment, or use the original notebook environment:

   ```bash
   python3.10 -m venv .venv
   source .venv/bin/activate
   ```

2. Restore the original compatible CUDA-enabled PyTorch, Triton, and custom xFormers stack. Their exact installation commands and wheel are not included in the supplied code.

3. Install the remaining declared Python packages in that environment:

   ```bash
   python -m pip install \
     "transformers==4.42.3" \
     "sentence-transformers==3.3.1" \
     "faiss-cpu==1.7.2" \
     "numpy==1.26.4" \
     pandas pyarrow tqdm accelerate
   ```

   This command does not install the custom modules, checkpoints, or GPU stack. Preserve the original environment's package constraints when available.

4. Place the required assets under the chosen input root and confirm that PyTorch can see both GPUs:

   ```bash
   python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Visible GPUs:', torch.cuda.device_count())"
   ```

## Input format

`test.csv` must contain at least one row and the following columns:

| Column | Description |
| --- | --- |
| `id` | Identifier copied into the final submission |
| `prompt` | Prompt or conversation input |
| `response_a` | First candidate response |
| `response_b` | Second candidate response |

Preserve the original notebook's CSV cell representation, including any serialised multi-turn strings expected by `LMSYSDataset` and `ProcessorPAB`. Their parsing logic is supplied by the external `human_pref` package. The semantic feature stage passes the cell values directly to SentenceTransformer without parsing them.

Ground-truth labels are not required. The script sets `winner_model_a=1`, `winner_model_b=0`, and `winner_tie=0` as loader placeholders; these are not observed outcomes and must not be used to interpret model accuracy.

## Usage

Run with the default Kaggle input location:

```bash
python finetuning_test2.py
```

Run with a custom asset directory and output location:

```bash
python finetuning_test2.py \
  --input-root /path/to/input \
  --output-dir ./output \
  --num-workers 4
```

| Argument | Default | Purpose |
| --- | --- | --- |
| `--input-root` | `/kaggle/input` | Root directory containing the required inputs |
| `--output-dir` | `./output` | Destination for copied modules, intermediates, and predictions |
| `--num-workers` | `4` | DataLoader worker count for Gemma and Llama inference |

The output directory is created automatically. Each run executes the full pipeline and overwrites matching output files; there are no resume or stage-selection options.

### Fixed inference settings

These settings are defined in the Python code rather than exposed as command-line arguments:

| Setting | Value |
| --- | --- |
| LLM input length limit | 4,096 tokens |
| LLM DataLoader batch size | 80, subdivided by the collator |
| Collator token budget | 8,192 |
| LLM precision | Float16 model weights with CUDA autocast |
| Custom classifier batch size | 8 |
| Custom classifier input length | 512 tokens, with truncation and fixed-length padding |

## Outputs

| File or directory | Contents |
| --- | --- |
| `human_pref/` | Copied custom inference modules |
| `test.parquet` | Original test rows with placeholder labels |
| `test_swap.parquet` | Test rows with responses A and B exchanged |
| `prob_m0.npy` | Gemma probabilities in original response order, shape `(N, 3)` |
| `prob_m3.npy` | Llama probabilities for swapped responses, shape `(N, 3)` |
| `prob_faiss.npy` | Custom classifier probabilities, shape `(N, 3)` |
| `submission.csv` | Final weighted probabilities and original row IDs |

`N` is the number of test rows. Before ensembling, `make_submission()` reorders Llama's probability columns with `[:, [1, 0, 2]]`; its saved `.npy` file retains the swapped class order.

The final CSV has these columns, in order:

```text
id,winner_model_a,winner_model_b,winner_tie
```

The prediction columns contain probabilities, not hard class labels. The ensemble checks that each input probability array has shape `(N, 3)` and contains only finite values. It prints the first submission rows and the saved file path.

## Semantic features and custom model

SentenceTransformer encodes all prompts and both response columns. Each embedding is L2-normalised, and FAISS `IndexFlatIP` searches for the most similar prompt embedding for each response. For nonzero normalised vectors, the inner-product score corresponds to cosine similarity.

**Each response is compared with every prompt in the current test set**, not only its own row's prompt. `similarity_a` and `similarity_b` are the maximum scores for responses A and B respectively. Consequently, these features can change when other test rows are added or removed. This preserves the supplied implementation.

The declared `CustomDebertaModel` projects the two similarity features into the base encoder's hidden dimension. It uses the first-token text embedding as a query to a four-head attention layer over the projected feature vector, concatenates the text and attention outputs, and predicts three classes through a classifier. Inference loads the complete saved model object rather than constructing a new model from this class.

## Troubleshooting

| Symptom | What to check |
| --- | --- |
| `Missing notebook inputs` | Confirm the input root and exact asset paths listed above. |
| Two-GPU requirement error | Confirm that two NVIDIA GPUs are allocated and visible to PyTorch. |
| `human_pref` import or custom model method error | Restore the original module files and their expected directory layout. |
| xFormers, Triton, or CUDA errors | Recreate the original compatible compiled dependency stack and custom wheel. |
| Custom checkpoint cannot be loaded | Use the original complete model checkpoint, matching class definitions, and compatible dependencies. A state dictionary alone is not accepted by this loader. |
| CUDA out of memory | Reduce the token budget or batch sizes in `predict_llm()` and `predict_custom_model()`, and check for other GPU workloads. |
| Missing columns or empty CSV | Supply a nonempty test file with all four required columns. |
| Unexpected probability shape or non-finite values | Check checkpoint class counts, inference outputs, and numerical stability. |

## Scope and reproducibility

- The custom checkpoint is loaded with `torch.load(..., weights_only=False)`. Use a trusted checkpoint because complete-object loading can execute Python code.
- The script provides no training loop, validation split, evaluation metrics, or checkpoint download utilities. No prediction-quality claims can be established from this code alone.
- Ensemble weights, model paths, and inference settings are fixed in the source. Changing them requires editing the script.
- Semantic feature columns are used in memory and are not exported as a separate feature file.
- Full reproduction depends on the original checkpoints, custom modules, wheel, and compatible runtime. End-to-end execution was not verified when preparing this README.

## License

The supplied Python source declares `SPDX-License-Identifier: Apache-2.0`. Check the applicable licenses and access terms for the external model checkpoints, datasets, and auxiliary code before redistribution or reuse.
