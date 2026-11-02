"""Portable complete-training manifests, local Kaggle mounts, and offline setup."""
import json
import shutil
import sys
from pathlib import Path

import pytest

from config import Config, parse_args
import config
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
    (nested / "config.json").write_text("{}")
    assert support.kaggle_input_path("checkpoint", input_root=tmp_path) == str(nested)
    (tmp_path / "checkpoint").mkdir()
    (tmp_path / "checkpoint" / "config.json").write_text("{}")
    with pytest.raises(ValueError, match="Multiple"):
        support.kaggle_input_path("checkpoint", input_root=tmp_path)


def test_missing_or_incomplete_kaggle_input_has_no_placeholder_fallback(tmp_path):
    with pytest.raises(FileNotFoundError, match="not attached"):
        support.kaggle_input_path("missing", input_root=tmp_path)
    (tmp_path / "incomplete").mkdir()
    with pytest.raises(FileNotFoundError, match="no config.json"):
        support.kaggle_input_path("incomplete", input_root=tmp_path)


@pytest.mark.parametrize("nested", [False, True])
def test_checkpoint_directory_can_be_nested_under_a_mounted_dataset(tmp_path, nested):
    dataset = tmp_path / "checkpoint" if not nested else tmp_path / "datasets" / "owner" / "checkpoint"
    model = dataset / "output" / "checkpoint-100"
    model.mkdir(parents=True)
    (model / "config.json").write_text("{}")
    assert support.kaggle_input_path("checkpoint", input_root=tmp_path) == str(model)
    other = dataset / "checkpoint-200"
    other.mkdir()
    (other / "config.json").write_text("{}")
    with pytest.raises(ValueError, match="Multiple model directories"):
        support.kaggle_input_path("checkpoint", input_root=tmp_path)


def test_sentence_model_discovery_uses_modules_json_instead_of_backbone_config(tmp_path):
    model = tmp_path / "some-pack" / "nested" / "sentence-transformer-model"
    transformer = model / "0_Transformer"
    transformer.mkdir(parents=True)
    (transformer / "config.json").write_text("{}")
    (model / "modules.json").write_text("[]")
    assert support.kaggle_input_path(
        "some-pack", "sentence-transformer-model", required_file="modules.json", input_root=tmp_path) == str(model)


def test_default_training_sources_resolve_to_discovered_inputs_and_invalidate_preflight(monkeypatch):
    calls = []

    def discover(slug, subdirectory="", required_file="config.json"):
        calls.append((slug, required_file))
        return f"/mounted/{slug}/model"

    monkeypatch.setattr(config, "kaggle_input_path", discover)
    cfg = Config(sources_preflight_done=True)
    cfg.validate()
    assert not calls  # CLI validation does not require attached training Inputs.
    cfg.resolve_training_sources(resolve_inputs=True)
    assert len(calls) == 3
    assert calls[-1] == ("some-pack", "modules.json")
    assert cfg.model_sources["gemma"] == "/mounted/lmsys-checkpoints-0-0805/model"
    assert not cfg.sources_preflight_done


def test_training_overrides_apply_to_validation_and_fold_copies():
    from dataclasses import replace
    cfg = Config(train_backends=("gemma",), model_sources={"gemma": "google/old-source"},
                 base_model_overrides={"gemma": "/mounted/selected-gemma"}, sources_preflight_done=True)
    cfg.validate()
    assert cfg.model_sources["gemma"] == "/mounted/selected-gemma"
    assert not cfg.sources_preflight_done
    cfg.sources_preflight_done = True
    cfg.resolve_training_sources()
    assert cfg.sources_preflight_done  # An unchanged source does not invalidate a completed check.
    fold = replace(cfg, holdout_fold=1, base_model_overrides={"gemma": "/mounted/other-gemma"})
    fold.resolve_training_sources()
    assert fold.model_sources["gemma"] == "/mounted/other-gemma"
    assert not fold.sources_preflight_done
    assert cfg.model_sources["gemma"] == "/mounted/selected-gemma"


def test_submission_never_resolves_original_training_inputs(tmp_path, monkeypatch):
    bundle = make_fold(tmp_path, 0, ("deberta",))["model_bundles"]["deberta"]

    def fail(*args, **kwargs):
        raise AssertionError("Submission attempted to discover original training Inputs")

    monkeypatch.setattr(config, "kaggle_input_path", fail)
    cfg = Config(mode="submit", inference_bundles={"deberta": bundle},
                 base_model_overrides={"gemma": "/saved/quantized-base"})
    original_sources = dict(cfg.model_sources)
    cfg.resolve_training_sources(resolve_inputs=True)
    cfg.validate()
    assert cfg.model_sources == original_sources
    assert cfg.base_model_overrides["gemma"] == "/saved/quantized-base"


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
