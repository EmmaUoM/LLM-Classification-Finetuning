# SPDX-License-Identifier: Apache-2.0

"""Portable training manifests, fold discovery, and offline dependency preparation."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

BACKENDS = ("deberta", "gemma", "llama")
SCHEMA = "emma-preference-training-v2"


def runtime_requirements(backends):
    """The notebook overlay; Kaggle supplies the CUDA and numerical stack."""
    requirements = [
        "transformers==4.46.3", "accelerate==1.1.1", "sentence-transformers==3.3.1",
        "sentencepiece==0.2.1", "safetensors==0.8.0", "mlflow-skinny==2.22.2",
        "sqlalchemy<2.1", "alembic<2", "sqlparse<1", "pyarrow==23.0.1",
    ]
    if set(backends) & {"gemma", "llama"}:
        requirements += ["peft==0.13.2", "bitsandbytes==0.45.5"]
    return requirements


def install_offline_dependencies(directory):
    """Bootstrap saved wheels before importing the ML stack in a fresh session."""
    directory = Path(directory)
    locked = directory / "requirements-overlay.txt"
    if not locked.is_file():
        raise FileNotFoundError(f"Missing offline dependency lock: {locked}")
    marker = directory / "wheelhouse_config.json"
    if marker.is_file():
        saved_python = read_json(marker)["python"]
        current_python = f"py{sys.version_info.major}{sys.version_info.minor}"
        if saved_python != current_python:
            raise RuntimeError(f"Wheels were prepared for {saved_python}; current runtime is {current_python}.")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "--disable-pip-version-check",
        "--no-index", "--find-links", str(directory), "--no-deps", "--requirement", str(locked),
    ])


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json_atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def check_backends(names):
    names = tuple(names)
    if not names or len(set(names)) != len(names) or set(names) - set(BACKENDS):
        raise ValueError(f"Backends must be unique names from {BACKENDS}; got {names}")
    return names


def normalize_bundles(mapping, expected):
    expected = check_backends(expected)
    if set(mapping) != set(expected):
        raise ValueError(f"Expected all requested models {expected}; found {tuple(mapping)}. "
                         "A partial DeBERTa-only result will not be submitted as three models.")
    result = {}
    for name in expected:
        paths = mapping[name]
        paths = [paths] if isinstance(paths, (str, Path)) else list(paths)
        if not paths:
            raise ValueError(f"No saved folds for {name}")
        result[name] = []
        for value in paths:
            directory = Path(value).expanduser().resolve()
            for filename in ("bundle_config.json", "manifest.json"):
                if not (directory / filename).is_file():
                    raise FileNotFoundError(f"Incomplete {name} bundle: {directory / filename}")
            if read_json(directory / "bundle_config.json").get("backend") != name:
                raise ValueError(f"Backend mismatch: {directory}")
            if str(directory) in result[name]:
                raise ValueError(f"Duplicate fold directory: {directory}")
            result[name].append(str(directory))
    return result


def save_training_manifest(cfg, group_id, results, status, error=None):
    """Store paths relative to the output root so a Kaggle Input can relocate them."""
    root = Path(cfg.output_root).resolve()
    bundles = {name: [] for name in cfg.train_backends}
    for result in results:
        if set(result["model_bundles"]) != set(bundles):
            raise RuntimeError("A fold did not finish every requested backend")
        for name, value in result["model_bundles"].items():
            bundles[name].append(str(Path(value).resolve().relative_to(root)))
    expected_folds = cfg.n_splits if cfg.cross_validation else 1
    if status == "complete" and any(len(paths) != expected_folds for paths in bundles.values()):
        raise RuntimeError("Cannot mark training complete before every model/fold finishes")
    bases = {}
    for name in cfg.train_backends:
        directory = root / "submission_assets" / "base_models" / group_id / name
        if (directory / "offline_base.json").exists():
            bases[name] = str(directory.relative_to(root))
    value = {
        "schema": SCHEMA, "status": status, "training_group_id": group_id,
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "backends": list(cfg.train_backends), "n_folds": expected_folds,
        "completed_folds": len(results), "model_bundles": bundles,
        "base_models": bases, "ensemble_weights": cfg.ensemble_weights,
        "wheelhouse": os.path.relpath(Path(cfg.offline_wheelhouse).resolve(), root)
            if cfg.offline_wheelhouse else None,
    }
    if error is not None:
        value["error"] = f"{type(error).__name__}: {error}"
    destination = root / "training_manifest.json"
    write_json_atomic(destination, value)
    return destination


def _legacy_plan(roots, expected):
    """Use old notebooks only when all their completed folds and backends exist."""
    groups = {}
    for root in roots:
        if not root.is_dir():
            continue
        for path in root.rglob("result.json"):
            run = path.parent
            if run.parent.name != "runs" or not (run / "config.json").is_file():
                continue
            config = read_json(run / "config.json")
            if config.get("mode") != "train" or set(config.get("train_backends", [])) != set(expected):
                continue
            summary = run / "run_summary.json"
            if not summary.is_file() or read_json(summary).get("status") != "FINISHED":
                continue
            result = read_json(path)
            if set(result.get("model_bundles", {})) != set(expected):
                continue
            signature = {key: value for key, value in config.items()
                         if key not in {"holdout_fold", "output_root", "source_snapshot"}}
            key = (str(run.parent.parent), json.dumps(signature, sort_keys=True))
            entry = groups.setdefault(key, {"folds": {}, "n_folds": config["n_splits"]})
            fold = config["holdout_fold"]
            previous = entry["folds"].get(fold)
            if previous is None or run.name > previous.name:
                entry["folds"][fold] = run
    complete = [entry for entry in groups.values()
                if set(entry["folds"]) == set(range(entry["n_folds"]))]
    if not complete:
        raise FileNotFoundError(
            "No complete three-model training output was found. First run MODE='train' to completion. "
            "For a fresh Kaggle submission session, save that version's output and attach it as Input. "
            "An .ipynb file alone contains no trained weights. "
            "For intentionally selected saved folds, fill INFERENCE_BUNDLES explicitly.")
    chosen = max(complete, key=lambda entry: max(run.name for run in entry["folds"].values()))
    mapping = {name: [str(chosen["folds"][fold] / "models" / name)
                      for fold in sorted(chosen["folds"])] for name in expected}
    return {"bundles": normalize_bundles(mapping, expected), "base_models": {},
            "wheelhouse": None, "manifest_path": None, "ensemble_weights": None}


def resolve_submission_plan(manual, expected=BACKENDS,
                            roots=("/kaggle/working", "/kaggle/input")):
    expected = check_backends(expected)
    if manual:
        return {"bundles": normalize_bundles(manual, expected), "base_models": {},
                "wheelhouse": None, "manifest_path": None, "ensemble_weights": None}
    roots = [Path(value).expanduser().resolve() for value in roots]
    candidates = []
    for root in roots:
        if root.is_dir():
            for path in root.rglob("training_manifest.json"):
                value = read_json(path)
                if value.get("schema") == SCHEMA:
                    candidates.append((value["updated_utc"], path, value))
    if not candidates:
        return _legacy_plan(roots, expected)
    _, path, value = max(candidates, key=lambda item: (item[0], str(item[1])))
    if value["status"] != "complete":
        raise RuntimeError(f"Latest training is {value['status']}, with {value['completed_folds']}/"
                           f"{value['n_folds']} complete folds: {path}. Finish training first. "
                           f"Details: {value.get('error', 'see the training logs')}")
    root = path.parent
    mapping = {name: [str(root / relative) for relative in paths]
               for name, paths in value["model_bundles"].items()}
    bundles = normalize_bundles(mapping, expected)
    if any(len(paths) != value["n_folds"] for paths in bundles.values()):
        raise RuntimeError(f"Missing folds in training manifest: {path}")
    bases = {name: str(root / relative) for name, relative in value["base_models"].items()}
    for name, directory in bases.items():
        if not (Path(directory) / "config.json").is_file():
            raise FileNotFoundError(f"Missing exported {name} base: {directory}")
    wheelhouse = str(root / value["wheelhouse"]) if value.get("wheelhouse") else None
    return {"bundles": bundles, "base_models": bases, "wheelhouse": wheelhouse,
            "manifest_path": str(path), "ensemble_weights": value.get("ensemble_weights")}


def locate_competition_csv(filename, configured=None, input_root="/kaggle/input"):
    root = Path(input_root)
    candidates = [Path(configured)] if configured else []
    candidates += [root / "competitions" / "llm-classification-finetuning" / filename,
                   root / "llm-classification-finetuning" / filename]
    for path in candidates:
        if path.is_file():
            return str(path)
    matches = sorted(root.glob(f"**/llm-classification-finetuning/{filename}"))
    if len(matches) == 1:
        return str(matches[0])
    raise FileNotFoundError(f"Attach the LLM Classification Finetuning competition data; "
                            f"cannot uniquely locate {filename}. Checked {candidates}")


def find_offline_base(directory, metadata, name, overrides, roots):
    if name in overrides:
        return str(overrides[name])
    if metadata.get("offline_base_relative"):
        portable = (Path(directory) / metadata["offline_base_relative"]).resolve()
        if (portable / "config.json").is_file():
            return str(portable)
    source = metadata["base_model_source"]
    if Path(source).is_dir():
        return source
    # For old adapter-only outputs, recognize an attached matching HF base by its exact repo name.
    basename = source.rstrip("/").split("/")[-1]
    matches = set()
    for root in map(Path, roots):
        if root.is_dir():
            for config_path in root.rglob("config.json"):
                if basename not in config_path.parts:
                    continue
                config = read_json(config_path)
                if config.get("model_type") not in {"gemma2" if name == "gemma" else "llama"}:
                    continue
                parent = config_path.parent
                if ((parent / "model.safetensors").is_file()
                        or (parent / "model.safetensors.index.json").is_file()
                        or (parent / "pytorch_model.bin").is_file()
                        or (parent / "pytorch_model.bin.index.json").is_file()):
                    matches.add(str(parent))
    if len(matches) > 1:
        raise ValueError(f"Multiple bases match {source}; set BASE_MODEL_OVERRIDES[{name!r}].")
    if matches:
        return matches.pop()
    # local_files_only=True in the loader permits an already populated HF cache, never a download.
    return source


def prepare_offline_wheelhouse(requirements, output_root):
    """Pin/download the installed overlay; keep Kaggle's CUDA/PyTorch/numerical stack."""
    from packaging.requirements import Requirement
    from packaging.utils import canonicalize_name
    provided = {"torch", "torchvision", "torchaudio", "triton", "numpy", "pandas", "scipy",
                "scikit-learn", "pyarrow", "pillow", "matplotlib"}
    pending = [Requirement(value).name for value in requirements]
    versions = {}
    while pending:
        name = canonicalize_name(pending.pop())
        if name in versions or name in provided or name.startswith("nvidia-"):
            continue
        distribution = importlib.metadata.distribution(name)
        versions[name] = distribution.version
        for value in distribution.requires or []:
            requirement = Requirement(value)
            if requirement.marker is None or requirement.marker.evaluate({"extra": ""}):
                pending.append(requirement.name)
    locked = "\n".join(f"{name}=={version}" for name, version in sorted(versions.items())) + "\n"
    tag = f"py{sys.version_info.major}{sys.version_info.minor}"
    fingerprint = hashlib.sha256((tag + locked).encode()).hexdigest()[:12]
    directory = Path(output_root) / "submission_assets" / "wheels" / fingerprint
    directory.mkdir(parents=True, exist_ok=True)
    requirement_file = directory / "requirements-overlay.txt"
    requirement_file.write_text(locked, encoding="utf-8")
    marker = directory / "wheelhouse_config.json"
    if not marker.is_file():
        print(f"Preparing offline dependency wheels in {directory}", flush=True)
        subprocess.check_call([sys.executable, "-m", "pip", "download", "--disable-pip-version-check",
                               "--only-binary=:all:", "--no-deps", "--dest", str(directory),
                               "--requirement", str(requirement_file)])
        write_json_atomic(marker, {"python": tag, "packages": versions,
                                   "provided_by_kaggle": sorted(provided)})
    return str(directory)


def kaggle_input_path(slug, subdirectory="", required_file="config.json", input_root="/kaggle/input"):
    """Resolve one mounted model directory, with no fallback to a Hub model ID."""
    root = Path(input_root)
    datasets = [root / slug, *sorted(root.glob(f"datasets/*/{slug}"))]
    datasets = {directory.resolve() for directory in datasets if directory.is_dir()}
    if not datasets:
        raise FileNotFoundError(
            f"Kaggle Input '{slug}' is not attached. Add it with Add Input, "
            f"or set its actual local path in the configuration. Searched under {root}.")
    matches = set()
    for dataset in datasets:
        preferred = dataset / subdirectory
        if (preferred / required_file).is_file():
            matches.add(str(preferred.resolve()))
        else:
            matches.update(str(path.parent.resolve()) for path in dataset.rglob(required_file))
    if not matches:
        raise FileNotFoundError(
            f"Input '{slug}' has no {required_file}. Select the complete model directory "
            f"from the Input panel. Mounted roots: {sorted(map(str, datasets))}")
    if len(matches) != 1:
        raise ValueError(f"Multiple model directories in '{slug}'; set the exact path: {sorted(matches)}")
    return matches.pop()
