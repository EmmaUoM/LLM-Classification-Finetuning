# SPDX-License-Identifier: Apache-2.0
"""MLflow lifecycle, metrics, artifacts, runtime monitoring, and failure records."""
from __future__ import annotations
import contextlib
import json
import math
import shutil
import sys
import time
import traceback
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import torch

from config import CLASS_COLUMNS, LOGGER
from utils import environment_snapshot, source_manifest, write_json


class RunTracker:
    """One run in SQLite + local JSON/CSV; never silently disable requested tracking."""
    def __init__(self, cfg, run_dir):
        self.cfg, self.run_dir = cfg, Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.mlflow, self.run_id, self.history = None, None, []
        self.source_info = None
        self.started = time.perf_counter()

    def __enter__(self):
        write_json(self.run_dir / "config.json", asdict(self.cfg))
        write_json(self.run_dir / "environment.json", environment_snapshot())
        try:
            if self.cfg.source_snapshot:
                self.source_info = source_manifest(self.cfg.source_snapshot)
                source_root = Path(self.source_info["project_root"])
                snapshot = self.run_dir / "source"
                for item in self.source_info["files"]:
                    destination = snapshot / item["path"]
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source_root / item["path"], destination)
                write_json(self.run_dir / "source_manifest.json", self.source_info)
            if self.cfg.enable_mlflow:
                import mlflow
                self.mlflow = mlflow
                database = Path(self.cfg.output_root).resolve() / "mlflow.db"
                artifacts = Path(self.cfg.output_root).resolve() / "mlartifacts"
                artifacts.mkdir(parents=True, exist_ok=True)
                mlflow.set_tracking_uri(f"sqlite:///{database}")
                if mlflow.active_run() is not None:
                    raise RuntimeError("An MLflow run is already active; finish it before rerunning this cell.")
                experiment = mlflow.get_experiment_by_name(self.cfg.experiment_name)
                if experiment is None:
                    eid = mlflow.create_experiment(self.cfg.experiment_name,
                                                   artifact_location=artifacts.as_uri())
                else:
                    eid = experiment.experiment_id
                run = mlflow.start_run(experiment_id=eid, run_name=self.run_dir.name)
                self.run_id = run.info.run_id
                mlflow.set_tags({"task": "human_preference_classification", "mode": self.cfg.mode,
                                 "class_order": ",".join(CLASS_COLUMNS),
                                 "validation": ("grouped_development_cross_validation" if self.cfg.mode == "train" and self.cfg.cross_validation else
                                                "grouped_development_holdout" if self.cfg.mode == "train" else
                                                "explicit_label_csv" if self.cfg.labels_csv else "unlabeled")})
                mlflow.log_params({k: json.dumps(v, default=str) if isinstance(v, (list, tuple, dict)) else v
                                   for k, v in asdict(self.cfg).items() if v is not None})
            for name in ("config.json", "environment.json"):
                self.artifact(self.run_dir / name)
            if self.cfg.source_snapshot:
                self.artifacts(self.run_dir / "source", "source")
                self.artifact(self.run_dir / "source_manifest.json")
            return self
        except BaseException as error:
            self._finish("FAILED", error)
            raise

    def metrics(self, values, step=0):
        safe = {k: float(v) for k, v in values.items() if math.isfinite(float(v))}
        item = {"step": int(step), "time": datetime.now(timezone.utc).isoformat(), **safe}
        self.history.append(item)
        with (self.run_dir / "metrics.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(item) + "\n")
        if self.mlflow is not None and self.run_id is not None:
            self.mlflow.log_metrics(safe, step=int(step), synchronous=True)

    def params(self, values):
        if self.mlflow is not None and self.run_id is not None:
            self.mlflow.log_params(values, synchronous=True)

    def artifact(self, path, directory="metadata"):
        if self.mlflow is not None and self.run_id is not None:
            self.mlflow.log_artifact(str(path), artifact_path=directory)

    def artifacts(self, path, directory):
        if self.mlflow is not None and self.run_id is not None:
            self.mlflow.log_artifacts(str(path), artifact_path=directory)

    def _finish(self, status, error=None):
        summary = {"status": status, "mode": self.cfg.mode, "mlflow_run_id": self.run_id,
                   "seconds": time.perf_counter() - self.started, "run_directory": str(self.run_dir)}
        if error is not None:
            summary.update(exception_type=type(error).__name__, message=str(error))
            (self.run_dir / "failure.txt").write_text(traceback.format_exc(), encoding="utf-8")
        write_json(self.run_dir / "run_summary.json", summary)
        pd.DataFrame(self.history).to_csv(self.run_dir / "metric_history.csv", index=False)
        if self.mlflow is not None and self.run_id is not None:
            # Preserve a training exception even if tracking cleanup also fails.
            try:
                for name in ("run_summary.json", "metric_history.csv", "metrics.jsonl", "failure.txt", "pipeline.log"):
                    if (self.run_dir / name).exists():
                        self.artifact(self.run_dir / name)
            except Exception:
                LOGGER.warning("Could not save final tracking artifacts", exc_info=True)
            finally:
                with contextlib.suppress(Exception):
                    self.mlflow.end_run(status=status)

    def __exit__(self, kind, error, tb):
        self._finish("FAILED" if error is not None else "FINISHED", error)
        return False


@contextlib.contextmanager
def track_stage(name, tracker):
    started = time.perf_counter()
    for i in range(torch.cuda.device_count()):
        torch.cuda.synchronize(i)
        torch.cuda.reset_peak_memory_stats(i)
    LOGGER.info("Starting %s", name)
    try:
        yield
    finally:
        active_error = sys.exc_info()[0] is not None
        values = {f"{name}_seconds": time.perf_counter() - started}
        try:
            for i in range(torch.cuda.device_count()):
                torch.cuda.synchronize(i)
                values[f"{name}_gpu{i}_peak_mb"] = torch.cuda.max_memory_allocated(i) / 1024**2
            tracker.metrics(values)
        except Exception:
            if not active_error:
                raise
        LOGGER.info("Finished %s: %.1fs", name, values[f"{name}_seconds"])
