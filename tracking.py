# SPDX-License-Identifier: Apache-2.0
"""MLflow run lifecycle, parameters, metrics, artifacts, and stage monitoring."""

import os
import sys
import time
from contextlib import contextmanager, suppress
from pathlib import Path

import torch

from config import CLASS_COLUMNS, LOGGER


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

