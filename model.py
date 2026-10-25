# SPDX-License-Identifier: Apache-2.0
"""Custom DeBERTa architecture and compatibility with notebook checkpoints."""

import pickle

import torch
import torch.nn as nn
from transformers import AutoModel


class CustomDebertaModel(nn.Module):
    """Preserve the original class name and architecture to load the full-model checkpoint."""

    def __init__(
        self,
        model_name,
        num_labels,
        feature_dim=2,
        dropout_rate=0.1,
    ):
        super().__init__()

        self.base_model = AutoModel.from_pretrained(model_name)
        hidden_size = self.base_model.config.hidden_size

        self.feature_fc = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(128, hidden_size),
            nn.ReLU(),
        )

        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=4,
            batch_first=True,
        )

        self.dropout = nn.Dropout(p=dropout_rate)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_size, num_labels),
        )

    def forward(
        self,
        input_ids,
        attention_mask,
        similarity_features,
        labels=None,
    ):
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        text_embeddings = outputs.last_hidden_state[:, 0, :]
        similarity_embeds = self.feature_fc(similarity_features)

        query = text_embeddings.unsqueeze(1)
        key_value = similarity_embeds.unsqueeze(1)

        attention_output, _ = self.attention(
            query,
            key_value,
            key_value,
        )

        combined = torch.cat(
            [text_embeddings, attention_output.squeeze(1)],
            dim=1,
        )

        logits = self.classifier(self.dropout(combined))
        result = {"logits": logits}

        if labels is not None:
            result["loss"] = nn.CrossEntropyLoss()(logits, labels)

        return result



class _LegacyCheckpointUnpickler(pickle.Unpickler):
    """Resolve the notebook's model class after moving it into this module."""

    def find_class(self, module, name):
        if name == "CustomDebertaModel" and module in {
            "__main__", "__mp_main__", "finetuning_test2", "finetuning_test2_mlops"
        }:
            return CustomDebertaModel
        return super().find_class(module, name)


class _CheckpointPickleModule:
    """Provide the pickle interface used by torch.load for legacy full models."""

    __name__ = "legacy_checkpoint_pickle"
    Unpickler = _LegacyCheckpointUnpickler
    load = staticmethod(pickle.load)


def load_custom_checkpoint(path):
    """Load a trusted full-model checkpoint with its original architecture."""
    # The compatibility unpickler remaps only known locations of this model class.
    # Full-model pickle files still require a trusted checkpoint source.
    return torch.load(
        path,
        map_location="cpu",
        pickle_module=_CheckpointPickleModule,
        weights_only=False,
    )
