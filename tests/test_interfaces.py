"""Checks for the modular CLI, source provenance, and compatibility boundaries."""
import ast
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

import config
import data
import models
import utils

ROOT = Path(__file__).resolve().parents[1]


def test_modules_compile_and_cli_help_works_without_site_packages(tmp_path):
    expected = ["main.py", "config.py", "data.py", "models.py", "inference.py", "evaluation.py",
                "tracking.py", "utils.py", "pipeline.py", "training.py"]
    for filename in expected:
        ast.parse((ROOT / filename).read_text())
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    completed = subprocess.run([sys.executable, "-S", str(ROOT / "main.py"), "--help"],
                               cwd=tmp_path, env=environment, text=True, capture_output=True, timeout=15)
    assert completed.returncode == 0, completed.stderr
    assert "--mode" in completed.stdout and "--bundle" in completed.stdout


def test_cli_merges_json_then_explicit_overrides(tmp_path):
    source = tmp_path / "config.json"
    source.write_text(json.dumps({"epochs": 3, "train_csv": "old.csv", "enable_mlflow": True}))
    cfg = config.parse_args(["--config", str(source), "--epochs", "1", "--input-root", "/new/input",
                             "--model-source", "deberta=/models/base", "--disable-mlflow"])
    assert cfg.epochs == 1 and not cfg.enable_mlflow
    assert cfg.train_csv == "/new/input/llm-classification-finetuning/train.csv"
    assert cfg.model_sources["deberta"] == "/models/base"
    cfg = config.parse_args(["--input-root", "/new/input", "--train-csv", "chosen.csv"])
    assert cfg.train_csv == "chosen.csv"
    with pytest.raises(SystemExit):
        config.parse_args(["--ensemble-weights", "1", "0", "0"])


def test_labels_align_by_id_and_swapping_restores_features(tmp_path):
    path = tmp_path / "labels.csv"
    pd.DataFrame({"id": ["9", "3"], "winner_model_a": [1, 0], "winner_model_b": [0, 0],
                  "winner_tie": [0, 1]}).to_csv(path, index=False)
    assert data.load_labels(path, ["3", "9"]).tolist() == [2, 0]
    with pytest.raises(ValueError, match="exactly"):
        data.load_labels(path, ["3"])
    frame = pd.DataFrame({"response_a": ["a"], "response_b": ["b"], "label": [0],
                          "similarity_a": [0.2], "similarity_b": [0.8]})
    swapped = data.swap_responses(frame)
    assert swapped.loc[0, "response_a"] == "b" and swapped.loc[0, "label"] == 1
    assert swapped.loc[0, "similarity_a"] == 0.8
    pd.testing.assert_frame_equal(data.swap_responses(swapped), frame)


def test_faiss_uses_a_fixed_reference_and_does_not_refit_on_queries():
    pytest.importorskip("faiss")
    index = data.FaissPromptIndex(np.array([[1, 0]], dtype=float))
    result = index.transform([[0, 1]], [[1, 0]])
    np.testing.assert_allclose(result, [[0, 1]])
    extended = index.transform([[0, 1], [1, 0]], [[1, 0], [0, 1]])
    np.testing.assert_allclose(extended[:1], result)
    assert index.index.ntotal == 1


def test_source_manifest_is_portable_and_detects_edits(tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir()
    (first / "main.py").write_text("print('hello')\n")
    (first / "requirements-mlops.txt").write_text("mlflow==2.22.2\n")
    shutil.copytree(first, second)
    assert utils.source_manifest(first)["sha256"] == utils.source_manifest(second)["sha256"]
    (second / "main.py").write_text("print('changed')\n")
    assert utils.source_manifest(first)["sha256"] != utils.source_manifest(second)["sha256"]
    paths = {item["path"] for item in utils.source_manifest(ROOT)["files"]}
    assert {"main.py", "models.py", "utils.py", "training.py", "tests/test_interfaces.py"} <= paths


def test_trusted_legacy_class_remapping(monkeypatch, tmp_path):
    old_class = type("CustomDebertaModel", (torch.nn.Module,), {"__module__": "__main__"})
    monkeypatch.setattr(sys.modules["__main__"], "CustomDebertaModel", old_class, raising=False)
    legacy = old_class()
    legacy.probe = torch.nn.Linear(2, 3)
    path = tmp_path / "legacy.pth"
    torch.save(legacy, path)
    with pytest.raises(ValueError, match="trusted=True"):
        models.load_custom_checkpoint(path)
    restored = models.load_custom_checkpoint(path, trusted=True)
    assert isinstance(restored, models.LegacyCustomDebertaModel)
    torch.testing.assert_close(restored.probe.weight, legacy.probe.weight)


def test_import_failure_still_writes_run_log_and_failure_summary(tmp_path):
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    completed = subprocess.run([sys.executable, "-S", str(ROOT / "main.py"), "--mode", "train",
                                "--output-dir", str(tmp_path), "--disable-mlflow"],
                               cwd=tmp_path, env=environment, text=True, capture_output=True, timeout=20)
    assert completed.returncode != 0
    runs = list((tmp_path / "runs").iterdir())
    assert len(runs) == 1
    assert json.loads((runs[0] / "run_summary.json").read_text())["status"] == "FAILED"
    assert "Execution failed" in (runs[0] / "pipeline.log").read_text()
