# SPDX-License-Identifier: Apache-2.0
"""Shared constants, logging identity, and command-line arguments."""

import argparse
import logging
from pathlib import Path

import numpy as np

LOGGER = logging.getLogger("finetuning_test2")
CLASS_COLUMNS = ["winner_model_a", "winner_model_b", "winner_tie"]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run the finetuning_test2 inference ensemble with MLflow tracking."
    )

    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("/input"),
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

