#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Command-line entry point. Run: python main.py --input-root /kaggle/input"""

import logging
import os
import time
import uuid
from contextlib import suppress
from datetime import datetime, timezone

from config import LOGGER, parse_args


def main(argv=None):
    args = parse_args(argv)

    # Keep --help available before importing the GPU inference dependencies.
    from pipeline import run_pipeline
    from tracking import RunTracker, tracking_session
    from utils import write_json

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
