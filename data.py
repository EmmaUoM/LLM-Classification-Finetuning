# SPDX-License-Identifier: Apache-2.0
"""Safe text cleaning, label alignment, grouped splits, features, and datasets."""
from __future__ import annotations
import ast
import json
import os
import random
import re
import unicodedata
from collections import Counter
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

from config import CLASS_COLUMNS, TEXT_COLUMNS, CLASS_SWAP
from utils import stable_hash


def normalize_text(text):
    # Preserve case, punctuation, code indentation, tabs, and line boundaries.
    text = unicodedata.normalize("NFC", text).replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    return text.strip(" \n\t")


def parse_turns(value):
    """Parse Kaggle's JSON/Python-list strings safely; never execute text with eval."""
    if value is None or (not isinstance(value, (str, list, tuple)) and pd.isna(value)):
        return [""], "missing"
    status = "plain"
    if isinstance(value, str):
        value = value.strip()
        if value.startswith("[") and value.endswith("]"):
            for parser in (json.loads, ast.literal_eval):
                try:
                    parsed = parser(value)
                    if isinstance(parsed, (list, tuple)):
                        value, status = parsed, "serialized"
                        break
                except (ValueError, TypeError, SyntaxError, json.JSONDecodeError):
                    pass
            else:
                # E.g. a literal math expression starting with '['; keep the original text.
                status = "unparsed_brackets_kept_as_text"
    if isinstance(value, (list, tuple)):
        if not all(item is None or isinstance(item, str) for item in value):
            raise ValueError("Turn lists must contain strings or null, not nested objects.")
        return [normalize_text(item or "") for item in value] or [""], status
    if not isinstance(value, str):
        raise ValueError("Text cells must be strings or lists of strings.")
    return [normalize_text(value)], status


def join_turns(turns):
    if not any(turns):
        return ""
    if len(turns) == 1:
        return turns[0]
    return "\n\n".join(f"[Turn {i + 1}]\n{text}" for i, text in enumerate(turns))


def clean_dataframe(raw, training):
    """Drop invalid TRAIN rows; preserve every TEST id and its original order."""
    required = {"id", *TEXT_COLUMNS, *(CLASS_COLUMNS if training else [])}
    if not required.issubset(raw.columns):
        raise ValueError(f"Missing columns: {sorted(required - set(raw.columns))}")
    if raw.empty:
        raise ValueError("Input CSV is empty")
    work = raw.copy().reset_index(drop=True)
    original_count = len(work)
    reasons, parsing = Counter(), Counter()
    rejected = []

    if not training and (work.id.isna().any() or work.id.duplicated().any()):
        raise ValueError("Test IDs must be non-null and unique; no test rows will be dropped.")

    parsed_rows = []
    for position, row in work.iterrows():
        reason = None
        if pd.isna(row["id"]):
            reason = "missing_id"
        texts, turn_counts = {}, []
        for column in TEXT_COLUMNS:
            try:
                turns, status = parse_turns(row[column])
                parsing[f"{column}_{status}"] += 1
                texts[column] = join_turns(turns)
                turn_counts.append(len(turns))
            except ValueError:
                if training:
                    reason = reason or "invalid_text_type"
                else:
                    # Preserve the row; a malformed test text is represented explicitly.
                    texts[column] = normalize_text(str(row[column]))
                    parsing[f"{column}_malformed_test_text_preserved"] += 1
        if len(set(turn_counts)) > 1:
            parsing["mismatched_turn_counts_preserved"] += 1
        if training and not texts.get("prompt", ""):
            reason = reason or "empty_prompt"
        target = None
        if training:
            values = pd.to_numeric(row[CLASS_COLUMNS], errors="coerce").to_numpy(dtype=float)
            if not np.isin(values, [0, 1]).all() or values.sum() != 1:
                reason = reason or "invalid_one_hot_label"
            else:
                target = int(values.argmax())
        if reason:
            if not training:
                raise ValueError(f"Invalid test id at row {position}")
            reasons[reason] += 1
            rejected.append({"row": position, "id": str(row["id"]), "reason": reason})
            continue
        a, b = texts["response_a"], texts["response_b"]
        order_swapped = a > b
        parsed_rows.append({
            "id": row["id"], **texts,
            "_source_row": position,
            "_prompt_group": stable_hash(texts["prompt"]),
            "_pair_key": stable_hash([texts["prompt"], *sorted([a, b])]),
            **({"label": target,
                "_canonical_label": int(CLASS_SWAP[target]) if order_swapped else target}
               if training else {}),
        })
    cleaned = pd.DataFrame(parsed_rows)
    if cleaned.empty:
        raise ValueError("No valid rows remain after schema/label cleaning")

    def remove(mask, reason):
        nonlocal cleaned
        dropped = cleaned.loc[mask]
        reasons[reason] += len(dropped)
        rejected.extend({"row": int(r["_source_row"]), "id": str(r.id), "reason": reason}
                        for _, r in dropped.iterrows())
        cleaned = cleaned.loc[~mask].copy()

    if training:
        id_conflicts = cleaned[["id", "_pair_key", "_canonical_label"]].drop_duplicates().groupby("id").size() > 1
        remove(cleaned.id.isin(id_conflicts[id_conflicts].index), "conflicting_id")
        remove(cleaned.id.duplicated(keep="first"), "duplicate_id")
        conflicts = cleaned.groupby("_pair_key")["_canonical_label"].nunique()
        remove(cleaned._pair_key.isin(conflicts[conflicts > 1].index), "conflicting_pair_labels")
        remove(cleaned._pair_key.duplicated(keep="first"), "duplicate_pair_including_ab_swaps")
        if cleaned.empty:
            raise ValueError("All training rows were removed as duplicate/conflicting records")
    cleaned = cleaned.reset_index(drop=True)
    if not training and cleaned.id.tolist() != work.id.tolist():
        raise AssertionError("Test cleaning changed row identity or order")
    quality = {
        "input_rows": original_count, "output_rows": len(cleaned),
        "dropped_rows": original_count - len(cleaned),
        "drop_reasons": dict(reasons), "parsing": dict(parsing),
        "empty_text_counts": {key: int(cleaned[key].eq("").sum()) for key in TEXT_COLUMNS},
        "identical_response_rows_kept": int(cleaned.response_a.eq(cleaned.response_b).sum()),
        "unique_prompt_groups": int(cleaned._prompt_group.nunique()),
    }
    if training:
        quality["class_counts"] = {CLASS_COLUMNS[int(k)]: int(v)
                                   for k, v in cleaned.label.value_counts().items()}
    return cleaned, quality, pd.DataFrame(rejected, columns=["row", "id", "reason"])


def limit_rows(frame, limit, seed):
    if limit is None or limit >= len(frame):
        return frame.reset_index(drop=True)
    indexes, _ = train_test_split(np.arange(len(frame)), train_size=limit,
                                  random_state=seed, stratify=frame.label)
    return frame.iloc[np.sort(indexes)].reset_index(drop=True)


def split_training_data(cleaned, cfg):
    if set(cleaned.label.unique()) != {0, 1, 2}:
        raise ValueError("Training data must contain all three classes after cleaning")
    if cleaned._prompt_group.nunique() < cfg.n_splits:
        raise ValueError("Too few unique prompts for grouped validation")
    groups_per_class = cleaned.groupby("label")._prompt_group.nunique()
    if (groups_per_class < cfg.n_splits).any():
        raise ValueError("Each class needs at least n_splits distinct prompt groups")
    splitter = StratifiedGroupKFold(n_splits=cfg.n_splits, shuffle=True, random_state=cfg.seed)
    splits = list(splitter.split(cleaned, cleaned.label, groups=cleaned._prompt_group))
    train_idx, valid_idx = splits[cfg.holdout_fold]
    train = limit_rows(cleaned.iloc[train_idx], cfg.max_train_rows, cfg.seed)
    valid = limit_rows(cleaned.iloc[valid_idx], cfg.max_val_rows, cfg.seed + 1)
    if set(train._prompt_group) & set(valid._prompt_group):
        raise AssertionError("A question appears in both train and validation")
    if set(train._pair_key) & set(valid._pair_key):
        raise AssertionError("A response pair appears in both train and validation")
    if set(train.label) != {0, 1, 2} or set(valid.label) != {0, 1, 2}:
        raise ValueError("The selected grouped split is missing a class; use a different fold.")
    return train, valid


def paired_cosine_features(prompt_vectors, response_a_vectors, response_b_vectors):
    def normalize(values):
        values = np.asarray(values, dtype=np.float32)
        if values.ndim != 2 or not np.isfinite(values).all():
            raise ValueError("Embeddings must be a finite matrix")
        return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)
    prompt, a, b = map(normalize, (prompt_vectors, response_a_vectors, response_b_vectors))
    if prompt.shape != a.shape or a.shape != b.shape:
        raise ValueError("Paired embedding matrices must have the same shape")
    return np.clip(np.stack([(prompt * a).sum(axis=1), (prompt * b).sum(axis=1)], axis=1), -1, 1)


def semantic_features(frame, sentence_encoder, batch_size):
    if sentence_encoder is None:
        return np.zeros((len(frame), 2), dtype=np.float32)
    embeddings = [sentence_encoder.encode(
        frame[column].tolist(), batch_size=batch_size, show_progress_bar=True,
        convert_to_numpy=True, normalize_embeddings=False,
    ) for column in TEXT_COLUMNS]
    features = paired_cosine_features(*embeddings)
    features[frame.prompt.eq("").to_numpy(), :] = 0
    for column, index in (("response_a", 0), ("response_b", 1)):
        features[frame[column].eq("").to_numpy(), index] = 0
    return features


def load_sentence_encoder(source, cfg, revision=None):
    from sentence_transformers import SentenceTransformer
    options = {"device": "cuda:0" if torch.cuda.is_available() else "cpu",
               "local_files_only": cfg.local_files_only, "trust_remote_code": False}
    if revision:
        options["revision"] = revision
    encoder = SentenceTransformer(str(source), **options)
    encoder.max_seq_length = cfg.sentence_max_length
    encoder.eval()
    return encoder


def allocate_budgets(lengths, available):
    """Share the text budget; unused short-component capacity goes to longer components."""
    allocation = [min(length, available // 3) for length in lengths]
    remaining = available - sum(allocation)
    while remaining > 0:
        active = [i for i in range(3) if allocation[i] < lengths[i]]
        if not active:
            break
        portion = max(1, remaining // len(active))
        for i in active:
            added = min(portion, lengths[i] - allocation[i], remaining)
            allocation[i] += added
            remaining -= added
    return allocation


class PreferenceDataset(Dataset):
    def __init__(self, frame, tokenizer, max_length, features=None,
                 augment_probability=0.0, force_swap=False, with_labels=True):
        self.frame = frame.reset_index(drop=True)
        self.tokenizer, self.max_length = tokenizer, int(max_length)
        self.features = np.zeros((len(frame), 2), dtype=np.float32) if features is None else np.asarray(features, dtype=np.float32)
        if self.features.shape != (len(frame), 2) or not np.isfinite(self.features).all():
            raise ValueError("Each row needs two finite similarity features")
        self.augment_probability, self.force_swap = augment_probability, force_swap
        self.with_labels = with_labels
        self.headers = [tokenizer.encode(text, add_special_tokens=False)
                        for text in ("Prompt:\n", "\n\nResponse A:\n", "\n\nResponse B:\n")]
        self.available = max_length - sum(map(len, self.headers)) - tokenizer.num_special_tokens_to_add(pair=False)
        if self.available < 3:
            raise ValueError("max_length is too short for the structured input markers")

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        swapped = self.force_swap or (self.augment_probability > 0 and random.random() < self.augment_probability)
        texts = [row.prompt, row.response_b if swapped else row.response_a,
                 row.response_a if swapped else row.response_b]
        parts = [self.tokenizer.encode(text, add_special_tokens=False, truncation=True,
                                       max_length=self.max_length) for text in texts]
        budgets = allocate_budgets(list(map(len, parts)), self.available)
        ids = []
        for header, tokens, budget in zip(self.headers, parts, budgets):
            ids.extend(header)
            ids.extend(tokens[:budget])
        ids = self.tokenizer.build_inputs_with_special_tokens(ids)
        if len(ids) > self.max_length:
            raise AssertionError("Token budget was exceeded")
        item = {"input_ids": ids, "attention_mask": [1] * len(ids),
                "similarity_features": self.features[index, ::-1].copy() if swapped else self.features[index]}
        if self.with_labels:
            label = int(row.label)
            item["labels"] = int(CLASS_SWAP[label]) if swapped else label
        return item


class PreferenceCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, examples):
        batch = self.tokenizer.pad(
            [{"input_ids": ex["input_ids"], "attention_mask": ex["attention_mask"]} for ex in examples],
            padding=True, pad_to_multiple_of=8, return_tensors="pt",
        )
        batch["similarity_features"] = torch.tensor(np.stack([ex["similarity_features"] for ex in examples]), dtype=torch.float32)
        if "labels" in examples[0]:
            batch["labels"] = torch.tensor([ex["labels"] for ex in examples], dtype=torch.long)
        return batch


def seed_worker(worker_id):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def make_loader(dataset, tokenizer, cfg, batch_size, shuffle=False):
    generator = torch.Generator().manual_seed(cfg.seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=cfg.num_workers, collate_fn=PreferenceCollator(tokenizer),
                      pin_memory=torch.cuda.is_available(), worker_init_fn=seed_worker,
                      generator=generator, drop_last=False)


def load_labels(path, ids):
    """Join real one-hot labels by id; never treat placeholder labels as targets."""
    labels = pd.read_csv(path, dtype={"id": str})
    if not {"id", *CLASS_COLUMNS}.issubset(labels.columns):
        raise ValueError(f"Labels CSV must contain id and {CLASS_COLUMNS}")
    if labels.id.isna().any() or labels.id.duplicated().any():
        raise ValueError("Labels CSV IDs must be non-null and unique")
    expected = pd.Series(ids).astype(str).tolist()
    if len(labels) != len(expected) or set(labels.id) != set(expected):
        raise ValueError("Labels CSV must match the inference IDs exactly")
    values = labels.set_index("id").loc[expected, CLASS_COLUMNS].to_numpy(dtype=float)
    if not np.isin(values, [0, 1]).all() or not (values.sum(axis=1) == 1).all():
        raise ValueError("Labels must be one-hot: exactly one winner per row")
    return values.argmax(axis=1)


def swap_responses(frame):
    """Swap A/B text and every corresponding target/feature that is present."""
    result = frame.copy()
    for first, second in (("response_a", "response_b"), ("winner_model_a", "winner_model_b"),
                          ("similarity_a", "similarity_b")):
        if first in frame and second in frame:
            result[first], result[second] = frame[second], frame[first]
    if "label" in frame:
        if not frame.label.isin([0, 1, 2]).all():
            raise ValueError("Invalid class label while swapping responses")
        result["label"] = frame.label.map(dict(enumerate(CLASS_SWAP)))
    return result


def prepare_test_files(test_csv, output_dir):
    """Export validated original/swapped Parquet inputs without fake target columns."""
    frame, _, _ = clean_dataframe(pd.read_csv(test_csv, dtype={"id": str}), training=False)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame[["id", *TEXT_COLUMNS]].to_parquet(output_dir / "test.parquet", index=False)
    swap_responses(frame)[["id", *TEXT_COLUMNS]].to_parquet(output_dir / "test_swap.parquet", index=False)
    return frame


class FaissPromptIndex:
    """Optional retrieval features against a FIXED reference set of training prompts.

    The main classifier uses paired cosine features. This helper supports retrieval
    experiments while avoiding an index constructed from the current test batch.
    Training/validation code must fit this index on the training partition only.
    """
    def __init__(self, training_prompt_embeddings):
        import faiss
        matrix = self._normalize(training_prompt_embeddings)
        if not len(matrix):
            raise ValueError("The FAISS reference prompt matrix is empty")
        self.dimension = matrix.shape[1]
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(matrix)

    @staticmethod
    def _normalize(values):
        matrix = np.asarray(values, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[1] < 1 or not np.isfinite(matrix).all():
            raise ValueError("FAISS vectors must be a finite two-dimensional matrix")
        matrix = matrix / np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)
        return np.ascontiguousarray(matrix)

    def transform(self, response_a_embeddings, response_b_embeddings):
        first, second = map(self._normalize, (response_a_embeddings, response_b_embeddings))
        if first.shape != second.shape or first.shape[1] != self.dimension:
            raise ValueError("Response vectors have incompatible shapes")
        return np.stack([self.index.search(values, 1)[0][:, 0] for values in (first, second)], axis=1)
