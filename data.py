# SPDX-License-Identifier: Apache-2.0
"""Input preparation, real-label alignment, and semantic similarity features."""

import numpy as np
import pandas as pd

from config import CLASS_COLUMNS


def prepare_test_files(test_csv, output_dir):
    df = pd.read_csv(test_csv)

    required = {"id", "prompt", "response_a", "response_b"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing CSV columns: {sorted(missing)}")

    if df.empty:
        raise ValueError("The test CSV is empty.")

    if df["id"].isna().any() or df["id"].duplicated().any():
        raise ValueError("Input IDs must be non-null and unique.")

    for column in ["prompt", "response_a", "response_b"]:
        if not df[column].map(lambda value: isinstance(value, str)).all():
            raise ValueError(f"{column} must contain non-null strings.")

    # LMSYSDataset requires these columns; these are placeholder labels only.
    df["winner_model_a"] = 1
    df["winner_model_b"] = 0
    df["winner_tie"] = 0

    df.to_parquet(
        output_dir / "test.parquet",
        index=False,
    )

    swapped = df.copy()

    swapped["response_a"], swapped["response_b"] = (
        df["response_b"],
        df["response_a"],
    )

    swapped.to_parquet(
        output_dir / "test_swap.parquet",
        index=False,
    )

    return df



def load_labels(path, ids):
    labels = pd.read_csv(path)
    required = {"id", *CLASS_COLUMNS}
    if not required.issubset(labels.columns):
        raise ValueError(f"Labels CSV must contain {sorted(required)}")
    if labels["id"].isna().any() or labels["id"].duplicated().any():
        raise ValueError("Labels CSV IDs must be non-null and unique")
    if len(labels) != len(ids) or set(labels["id"]) != set(ids):
        raise ValueError("Labels CSV IDs must match the input IDs exactly")
    values = labels.set_index("id").loc[ids, CLASS_COLUMNS].to_numpy(dtype=float)
    if not np.isin(values, [0, 1]).all() or not (values.sum(axis=1) == 1).all():
        raise ValueError("Winner labels must be one-hot: exactly one 1 per row")
    return values.argmax(axis=1)



def compute_semantic_features(df, input_root):
    import faiss
    from sentence_transformers import SentenceTransformer

    encoder = SentenceTransformer(
        str(
            input_root
            / "some-pack"
            / "sentence-transformer-model"
        )
    )

    def encode(column):
        # Preserve the CSV string representation used in the original notebook.
        embeddings = np.asarray(
            encoder.encode(df[column].tolist()),
            dtype=np.float32,
        )

        norms = np.linalg.norm(
            embeddings,
            axis=1,
            keepdims=True,
        )

        return np.ascontiguousarray(
            embeddings / np.maximum(norms, 1e-12)
        )

    prompt_embeddings = encode("prompt")
    response_a_embeddings = encode("response_a")
    response_b_embeddings = encode("response_b")

    index = faiss.IndexFlatIP(prompt_embeddings.shape[1])
    index.add(prompt_embeddings)

    result = df.copy()

    # Preserve the original logic: find the maximum similarity to any test-set prompt.
    result["similarity_a"] = index.search(
        response_a_embeddings,
        k=1,
    )[0][:, 0]

    result["similarity_b"] = index.search(
        response_b_embeddings,
        k=1,
    )[0][:, 0]

    return result

