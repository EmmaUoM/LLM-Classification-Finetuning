#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Main workflow:
1. Prepare the test data and a copy with responses A and B swapped.
2. Run Gemma2 inference.
3. Run Llama3 inference.
4. Extract similarity features with SentenceTransformer and FAISS.
5. Run inference with the custom DeBERTa classifier.
6. Combine the predictions with a weighted ensemble to create submission.csv.

The dependencies and model files from the original notebook are required.
The original runtime includes:
Python 3.10, torch 2.1.2, transformers 4.42.3,
triton 2.2.0, the original notebook's custom xformers wheel,
sentence-transformers 3.3.1, faiss-cpu 1.7.2,
numpy 1.26.4, pandas, pyarrow, tqdm, and accelerate.

MLOps setup (in the original GPU environment):
    python -m pip install "mlflow==2.22.2" "sqlalchemy<2.1"
    python finetuning_test2_mlops.py --input-root /kaggle/input
    mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000

MLflow is enabled by default. Runs use a local SQLite database and local
artifacts unless --tracking-uri or MLFLOW_TRACKING_URI specifies a server.
Use --disable-mlflow to run without the additional dependency. Each execution
writes to a separate directory under --output-dir and records configuration,
package versions, data/model provenance, timings, GPU memory, and output checks.
Use --log-predictions to also copy prediction files to the tracking store.
Raw prompts, responses, labels, and model weights are never uploaded.

Optional evaluation:
    python finetuning_test2_mlops.py --test-csv validation.csv \
        --labels-csv validation_labels.csv --run-name validation-ensemble
The labels CSV must contain id and the three one-hot winner columns. Metrics
are computed only from this explicit labels file, never from placeholder labels.
Use held-out examples for evaluation; this script does not train the models.

The original two-NVIDIA-GPU requirement and custom model files still apply.
Seeding and optional --deterministic improve repeatability, but custom GPU
kernels may still be nondeterministic. Large model files use size/mtime
metadata by default; --hash-checkpoints hashes every input file byte instead.
MLflow Tracking API: https://mlflow.org/docs/latest/api_reference/python_api/mlflow.html
"""

import argparse
import gc
import hashlib
import importlib.metadata
import json
import logging
import os
import platform
import random
import shutil
import sys
import time
import uuid
from contextlib import contextmanager, suppress
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


LOGGER = logging.getLogger("finetuning_test2")
CLASS_COLUMNS = ["winner_model_a", "winner_model_b", "winner_tie"]


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, default=str) + "\n", encoding="utf-8")


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_manifest(path, hash_all=False):
    """Record input provenance without copying datasets or checkpoint weights."""
    files = sorted(p for p in path.rglob("*") if p.is_file()) if path.is_dir() else [path]
    records = []
    for file in files:
        stat = file.stat()
        item = {
            "file": str(file.relative_to(path)) if path.is_dir() else file.name,
            "size_bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }
        # Hash small configuration/tokenizer/source files as well as explicit data files.
        if hash_all or (stat.st_size < 1024 * 1024 and file.suffix in {
            ".py", ".json", ".txt", ".yaml", ".yml", ".model", ".vocab", ".merges"
        }):
            item["sha256"] = sha256_file(file)
        records.append(item)
    return {
        "path": str(path),
        "mode": "full_sha256" if hash_all else "size_mtime_and_small_file_sha256",
        "files": records,
    }


def seed_everything(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = deterministic
    torch.use_deterministic_algorithms(deterministic, warn_only=True)


def environment_snapshot():
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpus": [
            {
                "index": i,
                "name": torch.cuda.get_device_name(i),
                "total_memory_bytes": torch.cuda.get_device_properties(i).total_memory,
            }
            for i in range(torch.cuda.device_count())
        ],
        # Record package versions rather than environment variables or credentials.
        "packages": dict(sorted(
            (dist.metadata["Name"], dist.version)
            for dist in importlib.metadata.distributions()
            if dist.metadata["Name"]
        )),
    }


class RunTracker:
    """Keep local metrics even when MLflow is explicitly disabled."""

    def __init__(self, mlflow=None, run_id=None):
        self.mlflow = mlflow
        self.run_id = run_id
        self.metrics = {}

    def log_metrics(self, metrics):
        self.metrics.update({key: float(value) for key, value in metrics.items()})
        if self.mlflow is not None:
            self.mlflow.log_metrics(metrics, synchronous=True)

    def log_params(self, params):
        if self.mlflow is not None:
            self.mlflow.log_params(params, synchronous=True)

    def log_artifact(self, path, artifact_path="metadata"):
        if self.mlflow is not None:
            self.mlflow.log_artifact(str(path), artifact_path=artifact_path)


@contextmanager
def tracking_session(args, default_run_name):
    if args.disable_mlflow:
        yield RunTracker()
        return

    try:
        import mlflow
    except ImportError as error:
        raise RuntimeError(
            'Install tracking dependencies with: python -m pip install '
            '"mlflow==2.22.2" "sqlalchemy<2.1"; '
            "or pass --disable-mlflow."
        ) from error

    uri = args.tracking_uri or os.environ.get("MLFLOW_TRACKING_URI")
    if uri:
        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment(args.experiment_name)
    else:
        mlflow.set_tracking_uri(f"sqlite:///{Path('mlflow.db').resolve()}")
        experiment = mlflow.get_experiment_by_name(args.experiment_name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(
                args.experiment_name,
                artifact_location=(Path.cwd() / "mlartifacts").as_uri(),
            )
        else:
            experiment_id = experiment.experiment_id
        mlflow.set_experiment(experiment_id=experiment_id)

    with mlflow.start_run(run_name=args.run_name or default_run_name) as run:
        mlflow.set_tags({
            "pipeline": "finetuning_test2",
            "task": "inference_ensemble",
            "class_order": ",".join(CLASS_COLUMNS),
            "evaluation": "explicit_labels" if args.labels_csv else "unlabeled",
        })
        LOGGER.info("MLflow run ID: %s", run.info.run_id)
        yield RunTracker(mlflow, run.info.run_id)
        # An exception escaping this context is automatically recorded as FAILED.


@contextmanager
def track_stage(name, tracker):
    LOGGER.info("Starting stage: %s", name)
    started = time.perf_counter()
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            torch.cuda.synchronize(i)
            torch.cuda.reset_peak_memory_stats(i)
    try:
        yield
    finally:
        metrics = {}
        stage_failed = sys.exc_info()[0] is not None
        try:
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    torch.cuda.synchronize(i)
                    metrics[f"{name}_gpu{i}_peak_allocated_mb"] = (
                        torch.cuda.max_memory_allocated(i) / 1024 ** 2
                    )
        except Exception:
            if not stage_failed:
                raise
            LOGGER.warning("GPU monitoring failed during stage cleanup: %s", name)
        metrics[f"{name}_seconds"] = time.perf_counter() - started
        # Do not replace an inference exception with an error from the tracking server.
        if stage_failed:
            with suppress(Exception):
                tracker.log_metrics(metrics)
        else:
            tracker.log_metrics(metrics)
        LOGGER.info("Stage %s elapsed: %.3f s", name, metrics[f"{name}_seconds"])


def validate_probabilities(name, probabilities, row_count):
    """Reject malformed outputs before ensembling or evaluation."""
    if probabilities.shape != (row_count, 3):
        raise ValueError(f"{name}: unexpected shape {probabilities.shape}")
    if not np.isfinite(probabilities).all():
        raise ValueError(f"{name}: non-finite probabilities")
    if (probabilities < 0).any() or (probabilities > 1).any():
        raise ValueError(f"{name}: probabilities must be between 0 and 1")
    # The original LLMs emit float16 probabilities; allow their rounding error.
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=2e-3, rtol=0):
        raise ValueError(f"{name}: probability rows must sum to 1")


def load_labels(path, ids):
    labels = pd.read_csv(path)
    required = {"id", *CLASS_COLUMNS}
    if not required.issubset(labels.columns):
        raise ValueError(f"Labels CSV must contain {sorted(required)}")
    if labels["id"].isna().any() or labels["id"].duplicated().any():
        raise ValueError("Labels CSV IDs must be non-null and unique")
    if len(labels) != len(ids) or set(labels["id"]) != set(ids):
        raise ValueError("Labels CSV IDs must match the input IDs exactly")
    values = labels.set_index("id").loc[ids, CLASS_COLUMNS].to_numpy(dtype=float)
    if not np.isin(values, [0, 1]).all() or not (values.sum(axis=1) == 1).all():
        raise ValueError("Winner labels must be one-hot: exactly one 1 per row")
    return values.argmax(axis=1)


def prediction_metrics(name, probabilities, labels=None):
    # Renormalize float16 rounding error in float64 before computing summaries.
    probabilities = probabilities.astype(np.float64)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    predicted = probabilities.argmax(axis=1)
    metrics = {
        f"{name}_mean_confidence": probabilities.max(axis=1).mean(),
        f"{name}_mean_entropy": -np.sum(
            probabilities * np.log(np.clip(probabilities, 1e-15, 1)), axis=1
        ).mean(),
    }
    for i, column in enumerate(CLASS_COLUMNS):
        metrics[f"{name}_fraction_{column}"] = np.mean(predicted == i)
    if labels is not None:
        metrics[f"{name}_accuracy"] = np.mean(predicted == labels)
        metrics[f"{name}_log_loss"] = -np.log(np.clip(
            probabilities[np.arange(len(labels)), labels], 1e-15, 1
        )).mean()
    return metrics


class CustomDebertaModel(nn.Module):
    """Preserve the original class name and architecture to load the full-model checkpoint."""

    def __init__(
        self,
        model_name,
        num_labels,
        feature_dim=2,
        dropout_rate=0.1,
    ):
        super().__init__()

        self.base_model = AutoModel.from_pretrained(model_name)
        hidden_size = self.base_model.config.hidden_size

        self.feature_fc = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(128, hidden_size),
            nn.ReLU(),
        )

        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=4,
            batch_first=True,
        )

        self.dropout = nn.Dropout(p=dropout_rate)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_size, num_labels),
        )

    def forward(
        self,
        input_ids,
        attention_mask,
        similarity_features,
        labels=None,
    ):
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        text_embeddings = outputs.last_hidden_state[:, 0, :]
        similarity_embeds = self.feature_fc(similarity_features)

        query = text_embeddings.unsqueeze(1)
        key_value = similarity_embeds.unsqueeze(1)

        attention_output, _ = self.attention(
            query,
            key_value,
            key_value,
        )

        combined = torch.cat(
            [text_embeddings, attention_output.squeeze(1)],
            dim=1,
        )

        logits = self.classifier(self.dropout(combined))
        result = {"logits": logits}

        if labels is not None:
            result["loss"] = nn.CrossEntropyLoss()(logits, labels)

        return result


def clear_gpu_memory():
    gc.collect()

    for index in range(torch.cuda.device_count()):
        with torch.cuda.device(index):
            torch.cuda.empty_cache()


def prepare_test_files(test_csv, output_dir):
    df = pd.read_csv(test_csv)

    required = {"id", "prompt", "response_a", "response_b"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing CSV columns: {sorted(missing)}")

    if df.empty:
        raise ValueError("The test CSV is empty.")

    if df["id"].isna().any() or df["id"].duplicated().any():
        raise ValueError("Input IDs must be non-null and unique.")

    for column in ["prompt", "response_a", "response_b"]:
        if not df[column].map(lambda value: isinstance(value, str)).all():
            raise ValueError(f"{column} must contain non-null strings.")

    # LMSYSDataset requires these columns; these are placeholder labels only.
    df["winner_model_a"] = 1
    df["winner_model_b"] = 0
    df["winner_tie"] = 0

    df.to_parquet(
        output_dir / "test.parquet",
        index=False,
    )

    swapped = df.copy()

    swapped["response_a"], swapped["response_b"] = (
        df["response_b"],
        df["response_a"],
    )

    swapped.to_parquet(
        output_dir / "test_swap.parquet",
        index=False,
    )

    return df


@torch.no_grad()
def predict_llm(
    input_root, output_dir, model_kind, num_workers,
    max_length=4096, batch_size=80, max_tokens=8192,
):
    from xformers.ops.fmha.attn_bias import BlockDiagonalCausalMask

    from human_pref.data.collators import (
        ShardedMaxTokensCollator,
        VarlenCollator,
    )
    from human_pref.data.dataset import LMSYSDataset
    from human_pref.data.processors import ProcessorPAB
    from human_pref.utils import to_device

    if model_kind == "gemma":
        from human_pref.models.modeling_gemma2 import (
            Gemma2ForSequenceClassification,
        )

        model_cls = Gemma2ForSequenceClassification
        checkpoint = input_root / "lmsys-checkpoints-0-0805"
        parquet_path = output_dir / "test.parquet"
        num_layers = 42
        support_system_role = False
        output_file = output_dir / "prob_m0.npy"

    elif model_kind == "llama":
        from human_pref.models.modeling_llama import (
            LlamaForSequenceClassification,
        )

        model_cls = LlamaForSequenceClassification
        checkpoint = input_root / "lmsys-checkpoints-3-0805"
        parquet_path = output_dir / "test_swap.parquet"
        num_layers = 32
        support_system_role = True
        output_file = output_dir / "prob_m3.npy"

    else:
        raise ValueError(f"Unknown model kind: {model_kind}")

    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint))

    if model_kind == "llama":
        tokenizer.deprecation_warnings[
            "sequence-length-is-longer-than-the-specified-maximum"
        ] = True

    processor = ProcessorPAB(
        tokenizer=tokenizer,
        max_length=max_length,
        support_system_role=support_system_role,
    )

    dataset = LMSYSDataset(
        csv_file=str(parquet_path),
        query=None,
        processor=processor,
        include_swap=False,
        is_parquet=True,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        collate_fn=ShardedMaxTokensCollator(
            max_tokens=max_tokens,
            base_collator=VarlenCollator(),
        ),
    )

    # Place the first and second halves of the model on separate GPUs.
    device_map = {
        "model.embed_tokens": "cuda:0",
        "model.norm": "cuda:1",
        "score": "cuda:1",
    }

    for index in range(num_layers):
        device_map[f"model.layers.{index}"] = (
            "cuda:0" if index < num_layers // 2 else "cuda:1"
        )

    model = model_cls.from_pretrained(
        str(checkpoint),
        torch_dtype=torch.float16,
        device_map=device_map,
    )

    model.eval()
    config = model.config

    if model_kind == "gemma":
        dim = config.head_dim
    else:
        dim = config.hidden_size // config.num_attention_heads

    inv_freq = 1.0 / (
        config.rope_theta
        ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
    )

    inv_freq0 = inv_freq.to("cuda:0")
    inv_freq1 = inv_freq.to("cuda:1")

    previous = None
    logits_batches = []

    for batch in tqdm(loader, desc=f"{model_kind} inference"):
        for micro_batch in batch:
            input_ids = to_device(
                micro_batch["input_ids"],
                "cuda:0",
            )

            seq_info = to_device(
                {
                    "cu_seqlens": micro_batch["cu_seqlens"],
                    "position_ids": micro_batch["position_ids"],
                    "max_seq_len": micro_batch["max_seq_len"],
                    "attn_bias": BlockDiagonalCausalMask.from_seqlens(
                        micro_batch["seq_lens"]
                    ),
                },
                "cuda:0",
            )

            with torch.cuda.amp.autocast():
                if previous is not None:
                    previous_info, previous_hidden = previous

                    logits = model.forward_part2(
                        previous_hidden,
                        previous_info,
                        inv_freq1,
                    )

                    logits_batches.append(logits.cpu())

                hidden = model.forward_part1(
                    input_ids,
                    seq_info,
                    inv_freq0,
                )

                previous = to_device(
                    [seq_info, hidden],
                    "cuda:1",
                )

    if previous is None:
        raise ValueError(f"No input batches for {model_kind}.")

    # Process the final micro-batch.
    with torch.cuda.amp.autocast():
        previous_info, previous_hidden = previous

        logits = model.forward_part2(
            previous_hidden,
            previous_info,
            inv_freq1,
        )

        logits_batches.append(logits.cpu())

    probabilities = (
        torch.cat(logits_batches, dim=0)
        .softmax(-1)
        .numpy()
    )

    np.save(output_file, probabilities)

    LOGGER.info("Saved %s: %s", output_file, probabilities.shape)

    # The original notebook's evaluate() uses placeholder labels,
    # so its evaluation output is intentionally omitted.


def compute_semantic_features(df, input_root):
    import faiss
    from sentence_transformers import SentenceTransformer

    encoder = SentenceTransformer(
        str(
            input_root
            / "some-pack"
            / "sentence-transformer-model"
        )
    )

    def encode(column):
        # Preserve the CSV string representation used in the original notebook.
        embeddings = np.asarray(
            encoder.encode(df[column].tolist()),
            dtype=np.float32,
        )

        norms = np.linalg.norm(
            embeddings,
            axis=1,
            keepdims=True,
        )

        return np.ascontiguousarray(
            embeddings / np.maximum(norms, 1e-12)
        )

    prompt_embeddings = encode("prompt")
    response_a_embeddings = encode("response_a")
    response_b_embeddings = encode("response_b")

    index = faiss.IndexFlatIP(prompt_embeddings.shape[1])
    index.add(prompt_embeddings)

    result = df.copy()

    # Preserve the original logic: find the maximum similarity to any test-set prompt.
    result["similarity_a"] = index.search(
        response_a_embeddings,
        k=1,
    )[0][:, 0]

    result["similarity_b"] = index.search(
        response_b_embeddings,
        k=1,
    )[0][:, 0]

    return result


@torch.no_grad()
def predict_custom_model(df, input_root, output_dir, batch_size=8, max_length=512):
    model_dir = (
        input_root
        / "akemiiiiii"
        / "custom_model_dir"
    )

    # The original file contains a complete model object.
    # Use weights_only=False only with checkpoints from trusted sources.
    model = torch.load(
        model_dir / "custom_model_complete.pth",
        map_location="cpu",
        weights_only=False,
    )

    if not isinstance(model, nn.Module):
        raise TypeError(
            "Expected the notebook's full nn.Module checkpoint."
        )

    device = torch.device("cuda:0")
    model.to(device).eval()

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    predictions = []

    for start in tqdm(
        range(0, len(df), batch_size),
        desc="Custom model inference",
    ):
        batch = df.iloc[start:start + batch_size]

        texts = [
            (
                f"Prompt: {row.prompt} "
                f"Response A: {row.response_a} "
                f"Response B: {row.response_b}"
            )
            for row in batch.itertuples(index=False)
        ]

        inputs = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )

        features = torch.tensor(
            batch[
                ["similarity_a", "similarity_b"]
            ].to_numpy(dtype=np.float32),
            dtype=torch.float32,
            device=device,
        )

        outputs = model(
            input_ids=inputs["input_ids"].to(device),
            attention_mask=inputs["attention_mask"].to(device),
            similarity_features=features,
        )

        predictions.append(
            outputs["logits"].softmax(-1).cpu().numpy()
        )

    probabilities = np.concatenate(predictions, axis=0)

    np.save(
        output_dir / "prob_faiss.npy",
        probabilities,
    )

    LOGGER.info("Saved prob_faiss.npy: %s", probabilities.shape)


def make_submission(df, output_dir, weights=(0.5, 0.25, 0.25)):
    gemma = np.load(output_dir / "prob_m0.npy", allow_pickle=False)

    # The Llama input has A and B swapped; restore the original class order.
    llama = np.load(
        output_dir / "prob_m3.npy", allow_pickle=False
    )[:, [1, 0, 2]]

    custom = np.load(output_dir / "prob_faiss.npy", allow_pickle=False)

    for name, probabilities in [
        ("gemma", gemma),
        ("llama", llama),
        ("custom", custom),
    ]:
        validate_probabilities(name, probabilities, len(df))

    weights = np.asarray(weights, dtype=float)
    if weights.shape != (3,) or not np.isfinite(weights).all():
        raise ValueError("Provide exactly three finite ensemble weights.")
    if (weights < 0).any() or weights.sum() <= 0 or not np.isfinite(weights.sum()):
        raise ValueError("Ensemble weights must be nonnegative with a finite positive sum.")

    predictions = np.average(
        [gemma, llama, custom],
        axis=0,
        weights=weights,
    )

    # Normalize float16 rounding error without changing the intended class order.
    predictions = predictions.astype(np.float64)
    predictions /= predictions.sum(axis=1, keepdims=True)
    validate_probabilities("ensemble", predictions, len(df))

    submission = pd.DataFrame(
        predictions,
        columns=CLASS_COLUMNS,
    )

    submission.insert(
        0,
        "id",
        df["id"].to_numpy(),
    )

    submission.to_csv(
        output_dir / "submission.csv",
        index=False,
    )

    LOGGER.info("Saved %s", output_dir / "submission.csv")
    return submission


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run the finetuning_test2 inference ensemble with MLflow tracking."
    )

    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("/kaggle/input"),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./output"),
        help="Parent directory for isolated run directories.",
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
    )

    parser.add_argument("--test-csv", type=Path, help="Override the default test.csv.")
    parser.add_argument("--labels-csv", type=Path, help="Real one-hot labels, joined by id.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--experiment-name", default="LLM-Classification-Inference")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--tracking-uri", default=None, help="MLflow server or database URI.")
    parser.add_argument("--disable-mlflow", action="store_true")
    parser.add_argument("--log-predictions", action="store_true", help="Copy prediction CSV/NPY files to MLflow.")
    parser.add_argument("--hash-checkpoints", action="store_true", help="Hash all checkpoint bytes; can be slow.")
    parser.add_argument("--ensemble-weights", type=float, nargs=3, default=[0.5, 0.25, 0.25],
                        metavar=("GEMMA", "LLAMA", "CUSTOM"))
    parser.add_argument("--llm-max-length", type=int, default=4096)
    parser.add_argument("--llm-batch-size", type=int, default=80)
    parser.add_argument("--llm-max-tokens", type=int, default=8192)
    parser.add_argument("--custom-max-length", type=int, default=512)
    parser.add_argument("--custom-batch-size", type=int, default=8)

    args = parser.parse_args(argv)
    if not 0 <= args.seed < 2 ** 32:
        parser.error("--seed must be between 0 and 2**32 - 1")
    if args.num_workers < 0:
        parser.error("--num-workers must be nonnegative")
    for name in ["llm_max_length", "llm_batch_size", "llm_max_tokens",
                 "custom_max_length", "custom_batch_size"]:
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.llm_max_tokens < args.llm_max_length:
        parser.error("--llm-max-tokens must be at least --llm-max-length")
    weights = np.asarray(args.ensemble_weights, dtype=float)
    if (not np.isfinite(weights).all() or (weights < 0).any()
            or weights.sum() <= 0 or not np.isfinite(weights.sum())):
        parser.error("--ensemble-weights must be nonnegative with a finite positive sum")
    for name in ["input_root", "output_dir", "test_csv", "labels_csv"]:
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.expanduser().resolve())
    return args


def run_pipeline(args, output_dir, tracker):
    input_root = args.input_root
    test_csv = args.test_csv or input_root / "llm-classification-finetuning" / "test.csv"
    required = [
        test_csv,
        input_root / "lmsys-modules-0805",
        input_root / "lmsys-checkpoints-0-0805",
        input_root / "lmsys-checkpoints-3-0805",
        input_root / "some-pack" / "sentence-transformer-model",
        (
            input_root
            / "akemiiiiii"
            / "custom_model_dir"
            / "custom_model_complete.pth"
        ),
    ]
    if args.labels_csv:
        required.append(args.labels_csv)

    with track_stage("preflight", tracker):
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError("Missing notebook inputs:\n" + "\n".join(missing))
        if torch.cuda.device_count() < 2:
            raise RuntimeError("The original LLM pipeline requires two NVIDIA CUDA GPUs.")
        seed_everything(args.seed, args.deterministic)
        df = prepare_test_files(test_csv, output_dir)
        labels = load_labels(args.labels_csv, df["id"].to_numpy()) if args.labels_csv else None

    with track_stage("provenance", tracker):
        source = Path(__file__).resolve()
        manifest = {
            "source": file_manifest(source, hash_all=True),
            "inputs": [file_manifest(path, hash_all=(
                args.hash_checkpoints or path == test_csv or path == args.labels_csv
            )) for path in required],
            "row_count": len(df),
            "schema": {column: str(df[column].dtype)
                       for column in ["id", "prompt", "response_a", "response_b"]},
            "semantic_feature_scope": "maximum similarity to any test-set prompt",
        }
        write_json(output_dir / "input_manifest.json", manifest)
        write_json(output_dir / "environment.json", environment_snapshot())
        tracker.log_artifact(source, "source")
        for name in ["run_config.json", "input_manifest.json", "environment.json"]:
            tracker.log_artifact(output_dir / name)
        tracker.log_params({
            "seed": args.seed,
            "deterministic": args.deterministic,
            "num_workers": args.num_workers,
            "llm_max_length": args.llm_max_length,
            "llm_batch_size": args.llm_batch_size,
            "llm_max_tokens": args.llm_max_tokens,
            "custom_max_length": args.custom_max_length,
            "custom_batch_size": args.custom_batch_size,
            "ensemble_weights": json.dumps(args.ensemble_weights),
            "data_sha256": manifest["inputs"][0]["files"][0]["sha256"],
            "source_sha256": manifest["source"]["files"][0]["sha256"],
            "hash_checkpoints": args.hash_checkpoints,
            "labels_provided": labels is not None,
        })
        tracker.log_metrics({"input_rows": len(df)})

    # Replace the notebook's !cp command; the run directory prevents stale files.
    shutil.copytree(input_root / "lmsys-modules-0805", output_dir / "human_pref")
    sys.path.insert(0, str(output_dir))

    for model_kind in ["gemma", "llama"]:
        with track_stage(f"{model_kind}_inference", tracker):
            predict_llm(input_root, output_dir, model_kind, args.num_workers,
                        max_length=args.llm_max_length, batch_size=args.llm_batch_size,
                        max_tokens=args.llm_max_tokens)
        clear_gpu_memory()

    with track_stage("semantic_features", tracker):
        df = compute_semantic_features(df, input_root)
        features = df[["similarity_a", "similarity_b"]].to_numpy(dtype=float)
        if not np.isfinite(features).all():
            raise ValueError("Semantic features contain non-finite values.")
    clear_gpu_memory()

    with track_stage("custom_inference", tracker):
        predict_custom_model(df, input_root, output_dir,
                             batch_size=args.custom_batch_size,
                             max_length=args.custom_max_length)
    clear_gpu_memory()

    with track_stage("ensemble", tracker):
        submission = make_submission(df, output_dir, args.ensemble_weights)
        arrays = {
            "gemma": np.load(output_dir / "prob_m0.npy", allow_pickle=False),
            # Compare all models using the original A/B class order.
            "llama": np.load(output_dir / "prob_m3.npy", allow_pickle=False)[:, [1, 0, 2]],
            "custom": np.load(output_dir / "prob_faiss.npy", allow_pickle=False),
            "ensemble": submission[CLASS_COLUMNS].to_numpy(),
        }
        for name, probabilities in arrays.items():
            tracker.log_metrics(prediction_metrics(name, probabilities, labels))

    output_manifest = {
        "class_order": CLASS_COLUMNS,
        "rows": len(df),
        "evaluation": "real_labels" if labels is not None else "skipped_no_labels",
        "files": [{"name": name, "sha256": sha256_file(output_dir / name)}
                  for name in ["submission.csv", "prob_m0.npy", "prob_m3.npy", "prob_faiss.npy"]],
    }
    write_json(output_dir / "output_manifest.json", output_manifest)
    tracker.log_artifact(output_dir / "output_manifest.json")
    if args.log_predictions:
        for item in output_manifest["files"]:
            tracker.log_artifact(output_dir / item["name"], "predictions")


def main(argv=None):
    args = parse_args(argv)
    execution_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    output_dir = args.output_dir / execution_id
    output_dir.mkdir(parents=True, exist_ok=False)

    for handler in list(LOGGER.handlers):
        handler.close()
        LOGGER.removeHandler(handler)
    LOGGER.setLevel(logging.INFO)
    LOGGER.propagate = False
    for handler in [logging.StreamHandler(), logging.FileHandler(output_dir / "pipeline.log", encoding="utf-8")]:
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        LOGGER.addHandler(handler)

    LOGGER.info("Run directory: %s", output_dir)
    config = vars(args).copy()
    # A server URI may contain credentials. It is used, but never saved as an artifact.
    config["tracking_uri"] = "configured" if args.tracking_uri or os.environ.get("MLFLOW_TRACKING_URI") else "local_sqlite"
    write_json(output_dir / "run_config.json", config)
    started = time.perf_counter()
    tracker = RunTracker()
    try:
        with tracking_session(args, execution_id) as tracker:
            try:
                run_pipeline(args, output_dir, tracker)
                tracker.log_metrics({"total_seconds": time.perf_counter() - started})
                write_json(output_dir / "metrics.json", tracker.metrics)
                write_json(output_dir / "run_summary.json", {
                    "status": "FINISHED", "execution_id": execution_id,
                    "mlflow_run_id": tracker.run_id,
                    "submission": str(output_dir / "submission.csv"),
                })
                LOGGER.info("Pipeline completed. Submission: %s", output_dir / "submission.csv")
                for name in ["metrics.json", "run_summary.json", "pipeline.log"]:
                    tracker.log_artifact(output_dir / name)
            except BaseException as error:
                LOGGER.exception("Pipeline failed")
                failure = {
                    "status": "FAILED", "exception_type": type(error).__name__,
                    "message": str(error), "mlflow_run_id": tracker.run_id,
                }
                write_json(output_dir / "failure.json", failure)
                write_json(output_dir / "run_summary.json", failure)
                write_json(output_dir / "metrics.json", tracker.metrics)
                # Preserve the original exception even if the server becomes unavailable.
                with suppress(Exception):
                    for name in ["run_config.json", "failure.json", "run_summary.json", "metrics.json", "pipeline.log"]:
                        tracker.log_artifact(output_dir / name)
                raise
    except BaseException as error:
        # Startup failures may happen before an MLflow run can be created.
        if not (output_dir / "failure.json").exists():
            LOGGER.exception("Tracking setup failed")
            failure = {"status": "FAILED", "exception_type": type(error).__name__,
                       "message": str(error), "mlflow_run_id": tracker.run_id}
            write_json(output_dir / "failure.json", failure)
            write_json(output_dir / "run_summary.json", failure)
        LOGGER.error("Execution failed; diagnostic files are in %s", output_dir)
        raise
    return output_dir


if __name__ == "__main__":
    main()
