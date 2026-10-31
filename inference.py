# SPDX-License-Identifier: Apache-2.0
"""Gemma 2, Llama 3, and custom classifier inference with A/B restoration."""
from __future__ import annotations
import numpy as np
import torch

from config import CLASS_SWAP
from data import PreferenceDataset, make_loader
from evaluation import validate_probabilities
from models import model_logits


def move_batch(batch, device):
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def autocast_context(cfg):
    return torch.autocast(device_type="cuda", dtype=torch.float16,
                          enabled=bool(cfg.use_amp and torch.cuda.is_available()))


def predict_probabilities(model, tokenizer, frame, features, name, max_length, cfg):
    model.eval()
    device = next(model.parameters()).device
    batch_size = cfg.eval_batch_size if name == "deberta" else cfg.llm_batch_size
    outputs = []
    for swapped in ([False, True] if cfg.tta else [False]):
        dataset = PreferenceDataset(frame, tokenizer, max_length, features,
                                    force_swap=swapped, with_labels=False)
        loader = make_loader(dataset, tokenizer, cfg, batch_size)
        probabilities = []
        with torch.inference_mode():
            for batch in loader:
                batch = move_batch(batch, device)
                with autocast_context(cfg):
                    logits = model_logits(model, batch, name)
                probabilities.append(logits.float().softmax(dim=-1).cpu().numpy())
        values = validate_probabilities(np.concatenate(probabilities), len(frame))
        outputs.append(values[:, CLASS_SWAP] if swapped else values)
    return validate_probabilities(np.mean(outputs, axis=0), len(frame))
