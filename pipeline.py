# SPDX-License-Identifier: Apache-2.0
"""Orchestrate data QA, fine-tuning, validation, and offline competition inference."""
from __future__ import annotations
import json
import os
import shutil
import traceback
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from config import CLASS_COLUMNS, TEXT_COLUMNS, LOGGER
from data import clean_dataframe, load_labels, load_sentence_encoder, semantic_features, split_training_data
from evaluation import classification_metrics, make_submission, weighted_ensemble
from inference import predict_probabilities
from models import load_model_bundle
from tracking import RunTracker, track_stage
from training import train_backend
from utils import bundle_manifest, create_run_directory, release_gpu, set_seeds, sha256_file, verify_bundle, write_json


def text_profile(frame):
    return {column: {"mean_characters": float(frame[column].str.len().mean()),
                     "p95_characters": float(frame[column].str.len().quantile(0.95)),
                     "empty_fraction": float(frame[column].eq("").mean())}
            for column in TEXT_COLUMNS}


def train_pipeline(cfg, run_dir, tracker):
    with track_stage("data_cleaning_and_split", tracker):
        train_path = Path(cfg.train_csv)
        if not train_path.is_file():
            raise FileNotFoundError(f"Attach the competition train.csv or set cfg.train_csv: {train_path}")
        raw = pd.read_csv(train_path, dtype={"id": str})
        cleaned, quality, rejected = clean_dataframe(raw, training=True)
        train, valid = split_training_data(cleaned, cfg)
        quality["split"] = {
            "method": "StratifiedGroupKFold by normalized full question",
            "n_splits": cfg.n_splits, "holdout_fold": cfg.holdout_fold,
            "train_rows_used": len(train), "validation_rows_used": len(valid),
            "prompt_group_overlap": 0, "response_pair_overlap": 0,
            "train_class_counts": train.label.value_counts().sort_index().to_dict(),
            "validation_class_counts": valid.label.value_counts().sort_index().to_dict(),
            "max_train_rows": cfg.max_train_rows, "max_val_rows": cfg.max_val_rows,
        }
        write_json(run_dir / "data_quality_report.json", quality)
        rejected.to_csv(run_dir / "rejected_training_rows.csv", index=False)
        keys = ["id", "label", "_prompt_group", "_pair_key", "_source_row"]
        train[keys].to_csv(run_dir / "train_split.csv", index=False)
        valid[keys].to_csv(run_dir / "validation_split.csv", index=False)
        provenance = {"train_csv": str(train_path), "train_sha256": sha256_file(train_path),
                      "train_rows": len(train), "validation_rows": len(valid),
                      "class_order": CLASS_COLUMNS,
                      "feature_definition": "Cosine of each answer with its OWN question; no test-set index.",
                      "source_sha256": tracker.source_info["sha256"] if tracker.source_info else None}
        write_json(run_dir / "data_manifest.json", provenance)
        tracker.params({"train_csv_sha256": provenance["train_sha256"],
                        "training_rows_used": len(train), "validation_rows_used": len(valid)})
        tracker.metrics({"input_training_rows": len(raw), "clean_training_rows": len(cleaned),
                         "dropped_training_rows": len(raw) - len(cleaned)})
        for filename in ("data_quality_report.json", "rejected_training_rows.csv", "train_split.csv",
                         "validation_split.csv", "data_manifest.json"):
            tracker.artifact(run_dir / filename)
        LOGGER.info("Data quality and split: %s", json.dumps(quality, default=str))
        del raw, cleaned

    train_features, valid_features = None, None
    if "deberta" in cfg.train_backends and cfg.use_semantic_features:
        with track_stage("paired_semantic_features", tracker):
            encoder = load_sentence_encoder(cfg.sentence_model, cfg, cfg.sentence_revision)
            train_features = semantic_features(train, encoder, cfg.sentence_batch_size)
            valid_features = semantic_features(valid, encoder, cfg.sentence_batch_size)
            encoder.save_pretrained(str(run_dir / "models" / "deberta" / "semantic_encoder"))
            np.save(run_dir / "train_similarity_features.npy", train_features)
            np.save(run_dir / "validation_similarity_features.npy", valid_features)
            del encoder
            release_gpu()

    validation_predictions, directories = {}, {}
    for name in cfg.train_backends:
        with track_stage(f"{name}_finetuning", tracker):
            probabilities, directory = train_backend(
                name, train, valid, train_features if name == "deberta" else None,
                valid_features if name == "deberta" else None, cfg, run_dir, tracker)
            validation_predictions[name], directories[name] = probabilities, directory
            metadata_path = directory / "bundle_config.json"
            metadata = json.loads(metadata_path.read_text())
            metadata["train_csv_sha256"] = provenance["train_sha256"]
            metadata["training_text_profile"] = text_profile(train)
            metadata["validation_tta"] = cfg.tta
            write_json(metadata_path, metadata)
            bundle_manifest(directory)
            tracker.artifact(directory / "manifest.json", f"models/{name}")
            tracker.artifact(directory / "validation_metrics.json", f"models/{name}")
            tracker.artifact(directory / "training_update_audit.json", f"models/{name}")
            if cfg.log_models_to_mlflow:
                tracker.artifacts(directory, f"model_bundles/{name}")

    with track_stage("validation_ensemble", tracker):
        ensemble, weights = weighted_ensemble(validation_predictions, cfg.ensemble_weights)
        metrics = classification_metrics(valid.label.to_numpy(), ensemble)
        tracker.metrics({f"ensemble_validation_{key}": value for key, value in metrics.items()})
        write_json(run_dir / "ensemble_validation.json", {
            "metrics": metrics, "normalized_weights": weights,
            "scope": "Development validation; weights fixed in config, not tuned on Kaggle test labels.",
        })
        out = pd.DataFrame(ensemble, columns=CLASS_COLUMNS)
        out.insert(0, "id", valid.id.to_numpy())
        out["true_label"] = valid.label.to_numpy()
        out.to_csv(run_dir / "ensemble_validation_predictions.csv", index=False)
        tracker.artifact(run_dir / "ensemble_validation.json")
        write_json(run_dir / "inference_config_example.json", {
            "mode": "submit", "inference_bundles": {key: str(value) for key, value in directories.items()},
            "ensemble_weights": weights,
            "note": "For a separate Kaggle inference notebook, attach exported model folders as Inputs and update these paths.",
        })
        tracker.artifact(run_dir / "inference_config_example.json")
    LOGGER.info("Training finished. Model bundles:")
    for name, directory in directories.items():
        LOGGER.info("%s: %s", name, directory)
    LOGGER.info("Use mode='submit' with the saved model bundles to create submission.csv")
    return {"model_bundles": {name: str(path) for name, path in directories.items()},
            "validation_metrics": metrics, "normalized_weights": weights}


def cross_validation_pipeline(cfg, run_dir):
    """Run independent grouped folds and combine only their held-out predictions."""
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "config.json", asdict(cfg))
    fold_results, fold_rows, oof_parts = [], [], []
    progress = {"status": "RUNNING", "n_splits": cfg.n_splits, "completed_folds": []}
    write_json(run_dir / "cv_progress.json", progress)
    summary_tracker = RunTracker(cfg, run_dir)
    try:
        cleaned, _, _ = clean_dataframe(pd.read_csv(cfg.train_csv, dtype={"id": str}), training=True)
        expected = cleaned.set_index("id")
        seen_ids = set()
        for fold in range(cfg.n_splits):
            LOGGER.info("Training fold %s/%s", fold + 1, cfg.n_splits)
            fold_cfg = replace(cfg, cross_validation=False, holdout_fold=fold)
            fold_dir = run_dir / "folds" / f"fold_{fold}"
            fold_dir.mkdir(parents=True, exist_ok=False)
            # Each call loads fresh pretrained models and closes its own MLflow run.
            result = run_pipeline(fold_cfg, run_dir=fold_dir)
            predictions = pd.read_csv(fold_dir / "ensemble_validation_predictions.csv", dtype={"id": str})
            ids = set(predictions.id)
            if predictions.id.duplicated().any() or ids & seen_ids:
                raise ValueError("A row received out-of-fold predictions more than once")
            if not ids <= set(expected.index):
                raise ValueError("Out-of-fold predictions contain unknown training IDs")
            if not np.array_equal(predictions.true_label.to_numpy(), expected.loc[predictions.id, "label"].to_numpy()):
                raise ValueError("Out-of-fold labels do not match the cleaned training data")
            seen_ids.update(ids)
            predictions["fold"] = fold
            oof_parts.append(predictions)
            fold_results.append({"fold": fold, **result})
            fold_rows.append({"fold": fold, **result["validation_metrics"],
                              "run_directory": str(fold_dir)})
            pd.DataFrame(fold_rows).to_csv(run_dir / "fold_metrics.csv", index=False)
            progress["completed_folds"].append(fold)
            write_json(run_dir / "cv_progress.json", progress)

        if cfg.max_val_rows is None and seen_ids != set(expected.index):
            raise ValueError("Cross-validation did not validate every cleaned training row exactly once")
        oof = pd.concat(oof_parts, ignore_index=True)
        oof["_source_row"] = oof.id.map(expected._source_row)
        oof = oof.sort_values("_source_row").drop(columns="_source_row").reset_index(drop=True)
        oof.to_csv(run_dir / "oof_predictions.csv", index=False)
        oof_metrics = classification_metrics(oof.true_label.to_numpy(), oof[CLASS_COLUMNS].to_numpy())
        metric_columns = ["log_loss", "accuracy", "macro_f1"]
        scores = pd.DataFrame(fold_rows)[metric_columns]
        summary = {
            "n_splits": cfg.n_splits, "completed_folds": len(fold_results),
            "fold_metrics_mean": scores.mean().to_dict(),
            "fold_metrics_std": scores.std(ddof=1).to_dict(),
            "oof_metrics": oof_metrics, "oof_rows": len(oof), "cleaned_rows": len(cleaned),
            "validation_coverage_complete": seen_ids == set(expected.index),
            "full_data": cfg.max_train_rows is None and cfg.max_val_rows is None,
            "scope": "Grouped development cross-validation; each fold selects its epoch using its validation labels. Not an untouched final test.",
        }
        write_json(run_dir / "cv_summary.json", summary)
        # Start the summary tracking run after the fold runs have all closed.
        with summary_tracker as tracker:
            tracker.metrics({f"cv_mean_{key}": value for key, value in summary["fold_metrics_mean"].items()})
            tracker.metrics({f"cv_std_{key}": value for key, value in summary["fold_metrics_std"].items()})
            tracker.metrics({f"oof_{key}": value for key, value in oof_metrics.items()})
            result = {"fold_results": fold_results, "validation_metrics": oof_metrics,
                      "cv_summary": summary, "run_directory": str(run_dir),
                      "mlflow_run_id": tracker.run_id}
            write_json(run_dir / "result.json", result)
            for name in ("fold_metrics.csv", "oof_predictions.csv", "cv_summary.json", "result.json"):
                tracker.artifact(run_dir / name)
        progress["status"] = "FINISHED"
        write_json(run_dir / "cv_progress.json", progress)
        LOGGER.info("Completed %s folds. OOF metrics: %s", cfg.n_splits, oof_metrics)
        return result
    except BaseException as error:
        progress.update(status="FAILED", exception_type=type(error).__name__, message=str(error))
        write_json(run_dir / "cv_progress.json", progress)
        if not (run_dir / "run_summary.json").exists():
            write_json(run_dir / "run_summary.json", {**progress, "mode": cfg.mode,
                       "run_directory": str(run_dir)})
        (run_dir / "failure.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise


def submission_pipeline(cfg, run_dir, tracker):
    # train.csv is intentionally never read in this mode.
    with track_stage("test_validation_and_cleaning", tracker):
        test_path = Path(cfg.test_csv)
        if not test_path.is_file():
            raise FileNotFoundError(f"Missing test.csv: {test_path}")
        test, quality, _ = clean_dataframe(pd.read_csv(test_path, dtype={"id": str}), training=False)
        labels = load_labels(cfg.labels_csv, test.id.to_numpy()) if cfg.labels_csv else None
        write_json(run_dir / "test_quality_report.json", quality)
        write_json(run_dir / "test_manifest.json", {"test_sha256": sha256_file(test_path),
                   "rows": len(test), "source_sha256": tracker.source_info["sha256"] if tracker.source_info else None,
                   "labels_sha256": sha256_file(cfg.labels_csv) if cfg.labels_csv else None})
        test[["id"]].to_csv(run_dir / "prediction_row_order.csv", index=False)
        for filename in ("test_quality_report.json", "test_manifest.json", "prediction_row_order.csv"):
            tracker.artifact(run_dir / filename)
    predictions, drift_reports = {}, {}
    for name, bundle in cfg.inference_bundles.items():
        with track_stage(f"{name}_inference", tracker):
            verify_bundle(bundle)
            metadata = json.loads((Path(bundle) / "bundle_config.json").read_text())
            if metadata["backend"] != name:
                raise ValueError(f"Wrong backend name for bundle {bundle}")
            features = None
            if name == "deberta" and metadata["semantic_feature_mode"] == "paired_cosine":
                # The sentence encoder travels with the trained custom classifier.
                encoder = load_sentence_encoder(Path(bundle) / "semantic_encoder", cfg)
                encoder.max_seq_length = metadata["sentence_max_length"]
                features = semantic_features(test, encoder, cfg.sentence_batch_size)
                del encoder
                release_gpu()
            model, tokenizer, metadata = load_model_bundle(bundle, cfg, name=name, verify=False)
            values = predict_probabilities(model, tokenizer, test, features, name,
                                           metadata["max_length"], cfg)
            predictions[name] = values
            if labels is not None:
                tracker.metrics({f"{name}_evaluation_{key}": value for key, value in classification_metrics(labels, values).items()})
            np.save(run_dir / f"prob_{name}.npy", values)
            tracker.metrics({f"{name}_mean_confidence": values.max(axis=1).mean(),
                             f"{name}_mean_entropy": -(values * np.log(np.clip(values, 1e-15, 1))).sum(axis=1).mean(),
                             **{f"{name}_predicted_fraction_{key}": float(np.mean(values.argmax(axis=1) == i))
                                for i, key in enumerate(CLASS_COLUMNS)}})
            drift_reports[name] = {"training_text_profile": metadata.get("training_text_profile"),
                                   "test_text_profile": text_profile(test),
                                   "note": "Descriptive input monitoring; no test labels or accuracy are inferred."}
            del model, tokenizer, features
            release_gpu()
    with track_stage("submission_export", tracker):
        ensemble, weights = weighted_ensemble(predictions, cfg.ensemble_weights)
        if labels is not None:
            tracker.metrics({f"ensemble_evaluation_{key}": value for key, value in classification_metrics(labels, ensemble).items()})
        temporary = run_dir / "submission.csv"
        submission = make_submission(test, ensemble, temporary)
        destination = Path(cfg.output_root) / "submission.csv"
        # Publish only a complete, schema-checked result to the competition output location.
        staged = destination.with_name(f".submission_{run_dir.name}.tmp")
        shutil.copyfile(temporary, staged)
        os.replace(staged, destination)
        write_json(run_dir / "input_monitoring.json", drift_reports)
        write_json(run_dir / "output_manifest.json", {
            "submission_path": str(destination), "rows": len(submission), "class_order": CLASS_COLUMNS,
            "submission_sha256": sha256_file(destination), "normalized_weights": weights,
            "evaluation": "explicit_labels" if labels is not None else "No test labels: accuracy/log loss are not calculated.",
        })
        for filename in ("output_manifest.json", "input_monitoring.json", "submission.csv"):
            tracker.artifact(run_dir / filename)
    LOGGER.info("Submission ready: %s", destination)
    LOGGER.info("Submission preview:\n%s", submission.head().to_string(index=False))
    return {"submission": str(destination), "rows": len(test)}


def run_pipeline(cfg, run_dir=None):
    cfg.validate()
    run_dir = create_run_directory(cfg) if run_dir is None else Path(run_dir)
    LOGGER.info("Run directory: %s", run_dir)
    if cfg.mode == "train" and cfg.cross_validation:
        return cross_validation_pipeline(cfg, run_dir)
    with RunTracker(cfg, run_dir) as tracker:
        with track_stage("preflight", tracker):
            set_seeds(cfg.seed)
            if cfg.mode == "train" and not torch.cuda.is_available() and not cfg.allow_cpu_training:
                raise RuntimeError("Enable a GPU in Kaggle, or set allow_cpu_training=True only for a tiny smoke test.")
            if cfg.mode == "submit" and not cfg.local_files_only:
                raise AssertionError("Submission mode must load models offline")
        if cfg.mode == "train":
            result = train_pipeline(cfg, run_dir, tracker)
        else:
            result = submission_pipeline(cfg, run_dir, tracker)
        result.update(run_directory=str(run_dir), mlflow_run_id=tracker.run_id)
        write_json(run_dir / "result.json", result)
        tracker.artifact(run_dir / "result.json")
    return result
