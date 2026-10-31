#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Command-line entry point, isolated run directories, logging, and failure handling."""
from __future__ import annotations
import json
import logging
from dataclasses import asdict
from pathlib import Path

from config import LOGGER, parse_args


def configure_logging(run_dir):
    for handler in list(LOGGER.handlers):
        handler.close()
        LOGGER.removeHandler(handler)
    LOGGER.setLevel(logging.INFO)
    LOGGER.propagate = False
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for handler in (logging.StreamHandler(), logging.FileHandler(Path(run_dir) / "pipeline.log", encoding="utf-8")):
        handler.setFormatter(formatter)
        LOGGER.addHandler(handler)


def main(argv=None):
    # --help and config validation work without importing the GPU stack.
    cfg = parse_args(argv)
    from utils import create_run_directory, write_json
    if cfg.source_snapshot is None:
        cfg.source_snapshot = str(Path(__file__).resolve().parent)
    run_dir = create_run_directory(cfg)
    configure_logging(run_dir)
    write_json(run_dir / "config.json", asdict(cfg))
    LOGGER.info("Run directory: %s", run_dir)
    try:
        from pipeline import run_pipeline
        result = run_pipeline(cfg, run_dir=run_dir)
        LOGGER.info("Execution finished: %s", json.dumps(result, default=str))
        return result
    except BaseException as error:
        LOGGER.exception("Execution failed; diagnostic files are in %s", run_dir)
        if not (run_dir / "run_summary.json").exists():
            write_json(run_dir / "run_summary.json", {
                "status": "FAILED", "mode": cfg.mode,
                "exception_type": type(error).__name__, "message": str(error),
            })
        raise


if __name__ == "__main__":
    main()

