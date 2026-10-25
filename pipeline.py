# SPDX-License-Identifier: Apache-2.0
"""Orchestrate data preparation, inference, evaluation, and provenance."""

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

from config import CLASS_COLUMNS
from data import compute_semantic_features, load_labels, prepare_test_files
from evaluation import make_submission, prediction_metrics
from inference import clear_gpu_memory, predict_custom_model, predict_llm
from tracking import track_stage
from utils import environment_snapshot, file_manifest, seed_everything
from utils import sha256_file, source_manifest, write_json


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
        project_dir = Path(__file__).resolve().parent
        manifest = {
            "source": source_manifest(project_dir),
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
        for item in manifest["source"]["files"]:
            tracker.log_artifact(project_dir / item["file"], "source")
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
            "source_sha256": manifest["source"]["sha256"],
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

