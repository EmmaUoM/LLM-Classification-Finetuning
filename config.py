# SPDX-License-Identifier: Apache-2.0
"""Command-line configuration, shared class order, and logging identity."""
from __future__ import annotations
import argparse
import json
import logging
import math
from dataclasses import dataclass, field, fields
from pathlib import Path

from submission_support import kaggle_input_path, normalize_bundles, resolve_submission_plan

LOGGER = logging.getLogger("llm_classification_finetuning")
CLASS_COLUMNS = ["winner_model_a", "winner_model_b", "winner_tie"]
TEXT_COLUMNS = ["prompt", "response_a", "response_b"]
CLASS_SWAP = (1, 0, 2)
ARCHITECTURE_VERSION = "paired-feature-deberta-v1"


@dataclass
class Config:
    mode: str = "train"                       # train OR submit; never both implicitly
    train_csv: str = "/kaggle/input/llm-classification-finetuning/train.csv"
    test_csv: str = "/kaggle/input/llm-classification-finetuning/test.csv"
    output_root: str = "./output"
    seed: int = 42
    train_backends: tuple = ("deberta", "gemma", "llama")
    model_sources: dict = field(default_factory=lambda: {
        "deberta": "microsoft/deberta-v3-base",
        "gemma": kaggle_input_path("lmsys-checkpoints-0-0805"),
        "llama": kaggle_input_path("lmsys-checkpoints-3-0805"),
    })
    model_revisions: dict = field(default_factory=dict)  # HF commit hashes if available
    inference_bundles: dict = field(default_factory=dict) # name -> one or more saved fold directories
    base_model_overrides: dict = field(default_factory=dict) # local QLoRA base for submit
    local_files_only: bool = False             # forced True in submit mode
    n_splits: int = 5                           # train every grouped fold by default
    cross_validation: bool = True
    holdout_fold: int = 0                      # used only with cross_validation=False
    max_train_rows: int | None = None           # debugging only; None uses all train rows
    max_val_rows: int | None = None
    epochs: int = 3
    patience: int = 2
    deberta_max_length: int = 512
    llm_max_length: int = 512
    deberta_batch_size: int = 4
    llm_batch_size: int = 1
    eval_batch_size: int = 8
    grad_accum_steps: int = 8
    encoder_lr: float = 2e-5
    head_lr: float = 1e-4
    lora_lr: float = 1e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    max_grad_norm: float = 1.0
    gradient_checkpointing: bool = True
    use_amp: bool = True
    num_workers: int = 0                        # avoids tokenizer/fork issues on Kaggle
    swap_augmentation_probability: float = 0.5 # training only; labels and features swap too
    tta: bool = True                            # average both orders after class restoration
    use_semantic_features: bool = True
    sentence_model: str = field(default_factory=lambda: kaggle_input_path(
        "some-pack", "sentence-transformer-model"))
    sentence_revision: str | None = None
    sentence_max_length: int = 256
    sentence_batch_size: int = 64
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: tuple = ("q_proj", "k_proj", "v_proj", "o_proj")
    ensemble_weights: dict = field(default_factory=lambda: {
        "gemma": 0.50, "llama": 0.25, "deberta": 0.25,
    })
    enable_mlflow: bool = True
    experiment_name: str = "Emma-LLM-Preference-Finetuning"
    log_models_to_mlflow: bool = False
    allow_cpu_training: bool = False
    source_snapshot: str | None = None
    labels_csv: str | None = None
    run_name: str | None = None
    training_group_id: str | None = None
    offline_wheelhouse: str | None = None
    export_quantized_bases: bool = True
    export_offline_dependencies: bool = True
    install_offline_dependencies: bool = True
    sources_preflight_done: bool = False
    auto_discover_roots: tuple = ("/kaggle/working", "/kaggle/input")

    def validate(self):
        if self.mode not in {"train", "submit"}:
            raise ValueError("mode must be 'train' or 'submit'")
        if not 0 <= self.seed < 2**32:
            raise ValueError("seed must be in [0, 2**32)")
        if not isinstance(self.inference_bundles, dict):
            raise ValueError("inference_bundles must map backend names to saved fold directories")
        if self.mode == "submit":
            self.local_files_only = True
            if not self.inference_bundles:
                plan = resolve_submission_plan(
                    {}, self.train_backends, (self.output_root, *self.auto_discover_roots))
                self.inference_bundles = plan["bundles"]
                self.base_model_overrides = {**plan["base_models"], **self.base_model_overrides}
                self.offline_wheelhouse = self.offline_wheelhouse or plan["wheelhouse"]
                if plan["ensemble_weights"]:
                    self.ensemble_weights = plan["ensemble_weights"]
            # Explicit manual bundles retain the CLI's intentionally selected-model workflow.
            self.inference_bundles = normalize_bundles(
                self.inference_bundles, tuple(self.inference_bundles))
        if not self.train_backends or len(set(self.train_backends)) != len(self.train_backends):
            raise ValueError("train_backends must be nonempty and unique")
        if set(self.train_backends) - {"deberta", "gemma", "llama"}:
            raise ValueError("Unknown training backend")
        if not isinstance(self.inference_bundles, dict) or set(self.inference_bundles) - {"deberta", "gemma", "llama"}:
            raise ValueError("inference_bundles must map known backend names to directories")
        if not isinstance(self.ensemble_weights, dict) or set(self.ensemble_weights) - {"deberta", "gemma", "llama"}:
            raise ValueError("ensemble_weights must map known backend names to numbers")
        weights = list(self.ensemble_weights.values())
        if not weights or any(not math.isfinite(value) or value < 0 for value in weights):
            raise ValueError("Ensemble weights must be finite and nonnegative")
        active = self.train_backends if self.mode == "train" else self.inference_bundles
        if sum(self.ensemble_weights.get(name, 0) for name in active) <= 0:
            raise ValueError("The enabled models must have positive total ensemble weight")
        if self.mode == "train" and any(not self.model_sources.get(name) for name in self.train_backends):
            raise ValueError("Every enabled training backend needs a model_sources entry")
        if not 0 <= self.swap_augmentation_probability <= 1:
            raise ValueError("swap_augmentation_probability must be in [0,1]")
        for key in ("epochs", "patience", "deberta_batch_size", "llm_batch_size",
                    "eval_batch_size", "grad_accum_steps", "sentence_batch_size"):
            if getattr(self, key) < 1:
                raise ValueError(f"{key} must be positive")
        for key in ("deberta_max_length", "llm_max_length"):
            if getattr(self, key) < 32:
                raise ValueError(f"{key} must be at least 32")
        for key in ("encoder_lr", "head_lr", "lora_lr"):
            if not math.isfinite(getattr(self, key)) or getattr(self, key) <= 0:
                raise ValueError(f"{key} must be finite and positive")
        if not 2 <= self.n_splits or not 0 <= self.holdout_fold < self.n_splits:
            raise ValueError("Invalid grouped holdout settings")
        if not isinstance(self.cross_validation, bool):
            raise ValueError("cross_validation must be a boolean")
        if not 0 <= self.warmup_ratio < 1 or self.num_workers < 0:
            raise ValueError("Invalid warmup_ratio or num_workers")
        for key in ("max_train_rows", "max_val_rows"):
            value = getattr(self, key)
            if value is not None and value < 3:
                raise ValueError(f"{key} must be None or at least 3")


def _named_values(items, flag, multiple=False):
    result = {}
    for item in items:
        name, separator, value = item.partition("=")
        if not separator or name not in {"deberta", "gemma", "llama"} or not value:
            raise ValueError(f"{flag} expects BACKEND=VALUE, where BACKEND is deberta, gemma, or llama")
        if multiple:
            result.setdefault(name, []).append(value)
        else:
            result[name] = value
    return result


def parse_args(argv=None):
    """Merge defaults, an optional JSON config, and explicit CLI overrides."""
    parser = argparse.ArgumentParser(
        description="Fine-tune preference classifiers or run frozen-model inference with MLflow.",
        argument_default=argparse.SUPPRESS,
    )
    parser.add_argument("--config", type=Path, help="JSON object containing Config fields")
    parser.add_argument("--mode", choices=["train", "submit"])
    parser.add_argument("--input-root", type=Path, help="Parent of llm-classification-finetuning/")
    parser.add_argument("--train-csv")
    parser.add_argument("--test-csv")
    parser.add_argument("--labels-csv", help="Optional real labels aligned to inference rows by id")
    parser.add_argument("--output-dir", "--output-root", dest="output_root")
    parser.add_argument("--train-backends", nargs="+", choices=["deberta", "gemma", "llama"])
    parser.add_argument("--model-source", action="append", help="Repeat BACKEND=HF_ID_OR_DIRECTORY")
    parser.add_argument("--model-revision", action="append", help="Repeat BACKEND=HF_COMMIT_HASH")
    parser.add_argument("--bundle", action="append", help="Repeat BACKEND=SAVED_FOLD_DIRECTORY, including multiple folds of one model")
    parser.add_argument("--base-model", action="append", help="Repeat BACKEND=LOCAL_QLORA_BASE_DIRECTORY")
    parser.add_argument("--ensemble-weights", nargs=3, type=float, metavar=("GEMMA", "LLAMA", "DEBERTA"))
    parser.add_argument("--sentence-model")
    parser.add_argument("--sentence-revision")
    parser.add_argument("--experiment-name")
    parser.add_argument("--run-name")
    parser.add_argument("--offline-wheelhouse", help="Saved dependency wheels directory")
    parser.add_argument("--discover-root", action="append", help="Additional root containing saved training outputs")
    parser.add_argument("--cross-validation", dest="cross_validation", action="store_true",
                        help="Train every grouped fold (default)")
    parser.add_argument("--single-fold", dest="cross_validation", action="store_false",
                        help="Train only --holdout-fold instead of all folds")
    for option in ("seed", "n_splits", "holdout_fold", "epochs", "patience", "max_train_rows", "max_val_rows",
                   "deberta_max_length", "llm_max_length", "deberta_batch_size", "llm_batch_size",
                   "eval_batch_size", "grad_accum_steps", "num_workers", "sentence_max_length",
                   "sentence_batch_size", "lora_r", "lora_alpha"):
        parser.add_argument("--" + option.replace("_", "-"), type=int)
    for option in ("encoder_lr", "head_lr", "lora_lr", "weight_decay", "warmup_ratio", "max_grad_norm",
                   "swap_augmentation_probability", "lora_dropout"):
        parser.add_argument("--" + option.replace("_", "-"), type=float)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--allow-cpu-training", action="store_true", help="Only for tiny CPU smoke tests")
    for flag, destination in (("--disable-mlflow", "enable_mlflow"), ("--no-log-models", "log_models_to_mlflow"),
                              ("--no-semantic-features", "use_semantic_features"), ("--no-tta", "tta"),
                              ("--no-amp", "use_amp"), ("--no-gradient-checkpointing", "gradient_checkpointing"),
                              ("--no-export-quantized-bases", "export_quantized_bases"),
                              ("--no-export-offline-dependencies", "export_offline_dependencies"),
                              ("--no-install-offline-dependencies", "install_offline_dependencies")):
        parser.add_argument(flag, dest=destination, action="store_false")
    values = vars(parser.parse_args(argv))
    path = values.pop("config", None)
    try:
        settings = json.loads(path.read_text(encoding="utf-8")) if path else {}
        if not isinstance(settings, dict):
            raise ValueError("The config JSON must be an object")
        unknown = set(settings) - {item.name for item in fields(Config)}
        if unknown:
            raise ValueError(f"Unknown config fields: {sorted(unknown)}")
        input_root = values.pop("input_root", None)
        discover_roots = values.pop("discover_root", None)
        named = {flag: values.pop(flag, None) for flag in ("model_source", "model_revision", "bundle", "base_model")}
        weights = values.pop("ensemble_weights", None)
        settings.update(values)
        if input_root is not None:
            for key, filename in (("train_csv", "train.csv"), ("test_csv", "test.csv")):
                if key not in values:
                    settings[key] = str(input_root / "llm-classification-finetuning" / filename)
        cfg = Config(**settings)
        if input_root is not None:
            cfg.auto_discover_roots = (str(input_root), *cfg.auto_discover_roots)
        if discover_roots is not None:
            cfg.auto_discover_roots = (*discover_roots, *cfg.auto_discover_roots)
        for flag, destination in (("model_source", "model_sources"), ("model_revision", "model_revisions"),
                                  ("bundle", "inference_bundles"), ("base_model", "base_model_overrides")):
            if named[flag] is not None:
                setattr(cfg, destination, {**getattr(cfg, destination), **_named_values(
                    named[flag], "--" + flag.replace("_", "-"), multiple=flag == "bundle")})
        if weights is not None:
            cfg.ensemble_weights = dict(zip(("gemma", "llama", "deberta"), weights))
        cfg.train_backends = tuple(cfg.train_backends)
        cfg.lora_target_modules = tuple(cfg.lora_target_modules)
        cfg.auto_discover_roots = tuple(cfg.auto_discover_roots)
        cfg.validate()
    except (TypeError, ValueError, OSError, RuntimeError) as error:
        parser.error(str(error))
    return cfg
