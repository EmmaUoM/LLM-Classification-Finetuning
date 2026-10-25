# SPDX-License-Identifier: Apache-2.0
"""File fingerprints, environment provenance, and reproducibility helpers."""

import hashlib
import importlib.metadata
import json
import os
import platform
import random
import sys

import numpy as np
import torch


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, default=str) + "\n", encoding="utf-8")



def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()



def file_manifest(path, hash_all=False):
    """Record input provenance without copying datasets or checkpoint weights."""
    files = sorted(p for p in path.rglob("*") if p.is_file()) if path.is_dir() else [path]
    records = []
    for file in files:
        stat = file.stat()
        item = {
            "file": str(file.relative_to(path)) if path.is_dir() else file.name,
            "size_bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }
        # Hash small configuration/tokenizer/source files as well as explicit data files.
        if hash_all or (stat.st_size < 1024 * 1024 and file.suffix in {
            ".py", ".json", ".txt", ".yaml", ".yml", ".model", ".vocab", ".merges"
        }):
            item["sha256"] = sha256_file(file)
        records.append(item)
    return {
        "path": str(path),
        "mode": "full_sha256" if hash_all else "size_mtime_and_small_file_sha256",
        "files": records,
    }



def seed_everything(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = deterministic
    torch.use_deterministic_algorithms(deterministic, warn_only=True)



def environment_snapshot():
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpus": [
            {
                "index": i,
                "name": torch.cuda.get_device_name(i),
                "total_memory_bytes": torch.cuda.get_device_properties(i).total_memory,
            }
            for i in range(torch.cuda.device_count())
        ],
        # Record package versions rather than environment variables or credentials.
        "packages": dict(sorted(
            (dist.metadata["Name"], dist.version)
            for dist in importlib.metadata.distributions()
            if dist.metadata["Name"]
        )),
    }



def source_manifest(project_dir):
    """Fingerprint every project module and the reproducible setup instructions."""
    paths = sorted(project_dir.glob("*.py"))
    paths += [project_dir / name for name in ("README.md", "requirements-mlops.txt")
              if (project_dir / name).is_file()]
    records = []
    for path in paths:
        records.append({"file": path.name, "size_bytes": path.stat().st_size,
                        "sha256": sha256_file(path)})
    # Hash relative names and file contents, independent of the checkout path.
    digest = hashlib.sha256(json.dumps(
        [{"file": item["file"], "sha256": item["sha256"]} for item in records],
        sort_keys=True,
    ).encode("utf-8")).hexdigest()
    return {"path": str(project_dir), "mode": "full_sha256",
            "sha256": digest, "files": records}
