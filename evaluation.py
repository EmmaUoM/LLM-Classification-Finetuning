# SPDX-License-Identifier: Apache-2.0
"""Probability validation, weighted ensembling, and optional evaluation."""

import numpy as np
import pandas as pd

from config import CLASS_COLUMNS, LOGGER


def validate_probabilities(name, probabilities, row_count):
    """Reject malformed outputs before ensembling or evaluation."""
    if probabilities.shape != (row_count, 3):
        raise ValueError(f"{name}: unexpected shape {probabilities.shape}")
    if not np.isfinite(probabilities).all():
        raise ValueError(f"{name}: non-finite probabilities")
    if (probabilities < 0).any() or (probabilities > 1).any():
        raise ValueError(f"{name}: probabilities must be between 0 and 1")
    # The original LLMs emit float16 probabilities; allow their rounding error.
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=2e-3, rtol=0):
        raise ValueError(f"{name}: probability rows must sum to 1")



def prediction_metrics(name, probabilities, labels=None):
    # Renormalize float16 rounding error in float64 before computing summaries.
    probabilities = probabilities.astype(np.float64)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    predicted = probabilities.argmax(axis=1)
    metrics = {
        f"{name}_mean_confidence": probabilities.max(axis=1).mean(),
        f"{name}_mean_entropy": -np.sum(
            probabilities * np.log(np.clip(probabilities, 1e-15, 1)), axis=1
        ).mean(),
    }
    for i, column in enumerate(CLASS_COLUMNS):
        metrics[f"{name}_fraction_{column}"] = np.mean(predicted == i)
    if labels is not None:
        metrics[f"{name}_accuracy"] = np.mean(predicted == labels)
        metrics[f"{name}_log_loss"] = -np.log(np.clip(
            probabilities[np.arange(len(labels)), labels], 1e-15, 1
        )).mean()
    return metrics



def make_submission(df, output_dir, weights=(0.5, 0.25, 0.25)):
    gemma = np.load(output_dir / "prob_m0.npy", allow_pickle=False)

    # The Llama input has A and B swapped; restore the original class order.
    llama = np.load(
        output_dir / "prob_m3.npy", allow_pickle=False
    )[:, [1, 0, 2]]

    custom = np.load(output_dir / "prob_faiss.npy", allow_pickle=False)

    for name, probabilities in [
        ("gemma", gemma),
        ("llama", llama),
        ("custom", custom),
    ]:
        validate_probabilities(name, probabilities, len(df))

    weights = np.asarray(weights, dtype=float)
    if weights.shape != (3,) or not np.isfinite(weights).all():
        raise ValueError("Provide exactly three finite ensemble weights.")
    if (weights < 0).any() or weights.sum() <= 0 or not np.isfinite(weights.sum()):
        raise ValueError("Ensemble weights must be nonnegative with a finite positive sum.")

    predictions = np.average(
        [gemma, llama, custom],
        axis=0,
        weights=weights,
    )

    # Normalize float16 rounding error without changing the intended class order.
    predictions = predictions.astype(np.float64)
    predictions /= predictions.sum(axis=1, keepdims=True)
    validate_probabilities("ensemble", predictions, len(df))

    submission = pd.DataFrame(
        predictions,
        columns=CLASS_COLUMNS,
    )

    submission.insert(
        0,
        "id",
        df["id"].to_numpy(),
    )

    submission.to_csv(
        output_dir / "submission.csv",
        index=False,
    )

    LOGGER.info("Saved %s", output_dir / "submission.csv")
    return submission

