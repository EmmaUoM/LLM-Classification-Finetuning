"""Seeds, source/data fingerprints, environment snapshots, and JSON helpers."""
from __future__ import annotations
import gc
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import re
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

from config import CLASS_COLUMNS


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(parts):
    return hashlib.sha256(json.dumps(parts, ensure_ascii=False,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


def set_seeds(seed):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def release_gpu():
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def environment_snapshot():
    import torch
    packages = {}
    for name in ("torch", "transformers", "peft", "accelerate", "bitsandbytes", "mlflow",
                 "mlflow-skinny", "sentence-transformers", "pandas", "numpy", "scikit-learn"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return {"python": sys.version, "platform": platform.platform(), "packages": packages,
            "cuda_version": torch.version.cuda,
            "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}


def bundle_manifest(directory):
    directory = Path(directory)
    files = [{"path": str(path.relative_to(directory)), "bytes": path.stat().st_size,
              "sha256": sha256_file(path)}
             for path in sorted(directory.rglob("*"))
             if path.is_file() and path.name != "manifest.json"]
    manifest = {"files": files, "class_order": CLASS_COLUMNS,
                "created_utc": datetime.now(timezone.utc).isoformat()}
    write_json(directory / "manifest.json", manifest)
    return manifest


def verify_bundle(directory):
    directory = Path(directory)
    manifest_path = directory / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing model manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("class_order") != CLASS_COLUMNS:
        raise ValueError("Saved class order does not match the notebook")
    for entry in manifest["files"]:
        path = directory / entry["path"]
        if not path.is_file() or sha256_file(path) != entry["sha256"]:
            raise ValueError(f"Model artifact is missing or has changed: {path}")
    return manifest


def create_run_directory(cfg):
    root = Path(cfg.output_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    prefix = re.sub(r"[^A-Za-z0-9._-]+", "_", cfg.run_name or cfg.mode).strip("._-") or "run"
    run_id = prefix + "_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    path = root / "runs" / run_id
    path.mkdir(parents=True, exist_ok=False)
    return path


def source_manifest(project_dir):
    """Fingerprint maintained source files, not generated outputs or model weights."""
    root = Path(project_dir).resolve()
    if root.is_file():
        root = root.parent
    if not root.is_dir():
        raise FileNotFoundError(root)
    candidates = set()
    for pattern in ("*.py", "requirements*.txt", "README.md", "configs/*.json", "tests/*.py"):
        candidates.update(path for path in root.glob(pattern) if path.is_file())
    files = [{"path": str(path.relative_to(root)), "sha256": sha256_file(path), "bytes": path.stat().st_size}
             for path in sorted(candidates)]
    return {"project_root": str(root), "files": files,
            "sha256": stable_hash([{key: item[key] for key in ("path", "sha256")} for item in files])}
