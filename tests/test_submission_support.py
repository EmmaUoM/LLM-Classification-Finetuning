"""Portable complete-training manifests, local Kaggle mounts, and offline setup."""
import json
import shutil
import sys
from pathlib import Path

import pytest

from config import Config, parse_args
import submission_support as support


def make_fold(root, fold, backends):
    bundles = {}
    for name in backends:
        directory = root / "runs" / f"fold_{fold}" / "models" / name
        directory.mkdir(parents=True)
        (directory / "bundle_config.json").write_text(json.dumps({"backend": name}))
        (directory / "manifest.json").write_text("{}")
        bundles[name] = str(directory)
    return {"model_bundles": bundles}


def test_defaults_enable_all_three_models_and_local_checkpoints():
    cfg = Config()
    assert cfg.train_backends == ("deberta", "gemma", "llama")
    assert "lmsys-checkpoints-0-0805" in cfg.model_sources["gemma"]
    assert "lmsys-checkpoints-3-0805" in cfg.model_sources["llama"]
    assert "sentence-transformer-model" in cfg.sentence_model
    assert not cfg.log_models_to_mlflow


def test_kaggle_mounts_support_old_and_nested_paths_and_reject_ambiguity(tmp_path):
    nested = tmp_path / "datasets" / "tascj0" / "checkpoint"
    nested.mkdir(parents=True)
    assert support.kaggle_input_path("checkpoint", input_root=tmp_path) == str(nested)
    (tmp_path / "checkpoint").mkdir()
    with pytest.raises(ValueError, match="Multiple"):
        support.kaggle_input_path("checkpoint", input_root=tmp_path)


def test_complete_training_relocates_and_discovers_every_model_and_fold(tmp_path):
    root = tmp_path / "training"
    cfg = Config(output_root=str(root), n_splits=2, export_offline_dependencies=False)
    folds = [make_fold(root, fold, cfg.train_backends) for fold in range(2)]
    base = root / "submission_assets" / "base_models" / "group" / "gemma"
    base.mkdir(parents=True)
    (base / "offline_base.json").write_text("{}")
    (base / "config.json").write_text("{}")
    cfg.offline_wheelhouse = str(root / "submission_assets" / "wheels" / "test")
    support.save_training_manifest(cfg, "group", folds, "complete")
    moved = tmp_path / "inputs" / "saved-output"
    moved.parent.mkdir()
    shutil.move(str(root), moved)
    submit = Config(mode="submit", output_root=str(tmp_path / "prediction"),
                    auto_discover_roots=(str(moved),))
    submit.validate()
    assert {name: len(paths) for name, paths in submit.inference_bundles.items()} == {
        "deberta": 2, "gemma": 2, "llama": 2}
    assert all(Path(path).is_relative_to(moved) for paths in submit.inference_bundles.values() for path in paths)
    assert Path(submit.base_model_overrides["gemma"]).is_relative_to(moved)
    assert Path(submit.offline_wheelhouse).is_relative_to(moved)
    assert submit.local_files_only


def test_incomplete_or_partial_three_model_training_cannot_be_submitted(tmp_path):
    cfg = Config(output_root=str(tmp_path), n_splits=2, export_offline_dependencies=False)
    first = make_fold(tmp_path, 0, cfg.train_backends)
    with pytest.raises(RuntimeError, match="every model/fold"):
        support.save_training_manifest(cfg, "group", [first], "complete")
    partial = {"model_bundles": {"deberta": first["model_bundles"]["deberta"]}}
    with pytest.raises(RuntimeError, match="every requested backend"):
        support.save_training_manifest(cfg, "group", [partial], "running")
    support.save_training_manifest(cfg, "group", [first], "failed", RuntimeError("interrupted"))
    with pytest.raises(RuntimeError, match="failed"):
        support.resolve_submission_plan({}, cfg.train_backends, [tmp_path])


def test_single_fold_manifest_is_complete_with_one_saved_fold(tmp_path):
    cfg = Config(output_root=str(tmp_path), cross_validation=False, holdout_fold=3,
                 train_backends=("deberta",), export_offline_dependencies=False)
    fold = make_fold(tmp_path, 3, cfg.train_backends)
    path = support.save_training_manifest(cfg, "group", [fold], "complete")
    assert support.read_json(path)["n_folds"] == 1
    assert len(support.resolve_submission_plan({}, cfg.train_backends, [tmp_path])["bundles"]["deberta"]) == 1


def test_cli_accepts_multiple_saved_folds_for_one_explicit_backend(tmp_path):
    first, second = [make_fold(tmp_path, fold, ("deberta",))["model_bundles"]["deberta"]
                     for fold in range(2)]
    cfg = parse_args(["--mode", "submit", "--bundle", f"deberta={first}",
                      "--bundle", f"deberta={second}"])
    assert cfg.inference_bundles == {"deberta": [first, second]}
    with pytest.raises(ValueError, match="Duplicate"):
        support.normalize_bundles({"deberta": [first, first]}, ("deberta",))


def test_offline_install_never_contacts_an_index_and_checks_python(tmp_path, monkeypatch):
    (tmp_path / "requirements-overlay.txt").write_text("example==1\n")
    marker = tmp_path / "wheelhouse_config.json"
    marker.write_text(json.dumps({"python": f"py{sys.version_info.major}{sys.version_info.minor}"}))
    calls = []
    monkeypatch.setattr(support.subprocess, "check_call", calls.append)
    support.install_offline_dependencies(tmp_path)
    assert len(calls) == 1
    assert "--no-index" in calls[0] and "--no-deps" in calls[0]
    marker.write_text(json.dumps({"python": "py00"}))
    with pytest.raises(RuntimeError, match="current runtime"):
        support.install_offline_dependencies(tmp_path)
    assert len(calls) == 1
