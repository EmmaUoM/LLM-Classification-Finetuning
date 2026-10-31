# SPDX-License-Identifier: Apache-2.0
"""Probability validation, weighted ensembling, and true-label evaluation."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, log_loss

from config import CLASS_COLUMNS


def validate_probabilities(probabilities, row_count):
    values = np.asarray(probabilities, dtype=np.float64)
    if values.shape != (row_count, 3) or not np.isfinite(values).all():
        raise ValueError(f"Invalid probability matrix: expected {(row_count, 3)}, got {values.shape}")
    if (values < 0).any() or (values > 1).any():
        raise ValueError("Probabilities must be in [0,1]")
    if not np.allclose(values.sum(axis=1), 1, atol=2e-3):
        raise ValueError("Probability rows do not sum to 1")
    return values / values.sum(axis=1, keepdims=True)


def classification_metrics(labels, probabilities):
    p = validate_probabilities(probabilities, len(labels))
    prediction = p.argmax(axis=1)
    return {
        "log_loss": float(log_loss(labels, p, labels=[0, 1, 2])),
        "accuracy": float(accuracy_score(labels, prediction)),
        "macro_f1": float(f1_score(labels, prediction, labels=[0, 1, 2],
                                  average="macro", zero_division=0)),
    }


def weighted_ensemble(predictions, weights):
    if not predictions:
        raise ValueError("No model predictions supplied")
    names = list(predictions)
    w = np.array([weights.get(name, 0) for name in names], dtype=float)
    if not np.isfinite(w).all() or (w < 0).any() or w.sum() <= 0:
        raise ValueError("Available models must have finite, nonnegative, positive-total weights")
    arrays = [validate_probabilities(predictions[name], len(predictions[names[0]]))
              for name in names]
    result = validate_probabilities(np.average(arrays, axis=0, weights=w), len(arrays[0]))
    return result, dict(zip(names, (w / w.sum()).tolist()))


def make_submission(test, probabilities, output_path):
    values = validate_probabilities(probabilities, len(test))
    if test.id.isna().any() or test.id.duplicated().any():
        raise ValueError("Invalid submission IDs")
    submission = pd.DataFrame(values, columns=CLASS_COLUMNS)
    submission.insert(0, "id", test.id.to_numpy())
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)
    assert submission.id.tolist() == test.id.tolist()
    return submission
