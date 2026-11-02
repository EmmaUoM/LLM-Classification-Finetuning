"""Gradient-based fine-tuning, checkpoint selection, and parameter update audits."""
from __future__ import annotations
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from transformers import get_linear_schedule_with_warmup
from tqdm.auto import tqdm

from config import CLASS_COLUMNS, LOGGER
from data import PreferenceDataset, make_loader
from evaluation import classification_metrics
from inference import autocast_context, move_batch, predict_probabilities
from models import export_quantized_base, load_training_model, model_logits, save_model_bundle
from utils import release_gpu, set_seeds, write_json


def optimizer_for_model(model, name, cfg):
    groups = {}
    for key, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name == "deberta":
            lr = cfg.encoder_lr if key.startswith("encoder.") else cfg.head_lr
        else:
            lr = cfg.lora_lr
        decay = 0.0 if parameter.ndim <= 1 or key.endswith("bias") else cfg.weight_decay
        groups.setdefault((lr, decay), []).append(parameter)
    if not groups:
        raise ValueError("The model has no trainable parameters")
    return torch.optim.AdamW([{"params": params, "lr": lr, "weight_decay": decay}
                              for (lr, decay), params in groups.items()])


def parameter_watchlist(model, name):
    candidates = [(key, parameter) for key, parameter in model.named_parameters()
                  if parameter.requires_grad and parameter.ndim >= 2 and parameter.numel() <= 2_000_000]
    watched = {}
    patterns = ({"encoder": lambda k: k.startswith("encoder.") and ".layer." in k,
                 "feature_network": lambda k: k.startswith("feature_fc."),
                 "classifier": lambda k: k.startswith("classifier.")}
                if name == "deberta" else
                {"lora_A": lambda k: "lora_A" in k,
                 "lora_B": lambda k: "lora_B" in k,
                 "classification_head": lambda k: "score" in k and "modules_to_save" in k})
    for group, predicate in patterns.items():
        matches = [(key, parameter) for key, parameter in candidates if predicate(key)]
        if matches:
            key, parameter = matches[-1] if group in {"classifier", "feature_network"} else matches[0]
            watched[group] = (key, parameter.detach().float().cpu().clone())
    if not watched:
        raise ValueError("No parameters available for the update audit")
    return watched


def finish_parameter_audit(model, watched, optimizer_steps):
    current = dict(model.named_parameters())
    audit = {"optimizer_steps": optimizer_steps, "watched_parameters": {}}
    changed = False
    for group, (key, initial) in watched.items():
        delta = current[key].detach().float().cpu() - initial
        maximum = float(delta.abs().max())
        changed |= maximum > 0
        audit["watched_parameters"][group] = {
            "name": key, "max_abs_delta": maximum,
            "l2_delta": float(torch.linalg.vector_norm(delta)), "changed": maximum > 0,
        }
    if optimizer_steps < 1 or not changed:
        raise RuntimeError("Training did not update the monitored parameters")
    audit["real_parameter_update_verified"] = True
    return audit


def train_backend(name, train, valid, train_features, valid_features, cfg, run_dir, tracker):
    set_seeds(cfg.seed)
    model, tokenizer = load_training_model(name, cfg)
    model_directory = Path(run_dir) / "models" / name
    model_directory.mkdir(parents=True, exist_ok=True)
    max_length = cfg.deberta_max_length if name == "deberta" else cfg.llm_max_length
    batch_size = cfg.deberta_batch_size if name == "deberta" else cfg.llm_batch_size
    dataset = PreferenceDataset(train, tokenizer, max_length, train_features,
                                augment_probability=cfg.swap_augmentation_probability)
    loader = make_loader(dataset, tokenizer, cfg, batch_size, shuffle=True)
    device = next(model.parameters()).device
    optimizer = optimizer_for_model(model, name, cfg)
    updates_per_epoch = math.ceil(len(loader) / cfg.grad_accum_steps)
    total_updates = updates_per_epoch * cfg.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(total_updates * cfg.warmup_ratio), total_updates)
    amp_enabled = bool(cfg.use_amp and torch.cuda.is_available())
    scaler = (torch.amp.GradScaler("cuda", enabled=amp_enabled)
              if hasattr(torch.amp, "GradScaler") else torch.cuda.amp.GradScaler(enabled=amp_enabled))
    watched = parameter_watchlist(model, name)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    tracker.params({f"{name}_trainable_parameters": trainable, f"{name}_total_parameters": total,
                    f"{name}_effective_batch_size": batch_size * cfg.grad_accum_steps,
                    f"{name}_max_length": max_length})
    write_json(model_directory / "parameter_groups.json", {
        "trainable_parameters": trainable, "total_parameters": total,
        "trainable_fraction": trainable / total,
        "trainable_parameter_names": [key for key, p in model.named_parameters() if p.requires_grad],
    })
    LOGGER.info("%s: %s/%s parameters trainable; %s train / %s validation rows", name, trainable, total, len(train), len(valid))

    baseline = predict_probabilities(model, tokenizer, valid, valid_features, name, max_length, cfg)
    baseline_metrics = classification_metrics(valid.label.to_numpy(), baseline)
    tracker.metrics({f"{name}_initial_validation_{key}": value for key, value in baseline_metrics.items()})
    best_loss, best_epoch, best_probabilities = float("inf"), None, None
    stale, optimizer_steps, skipped_steps, history = 0, 0, 0, []
    for epoch in range(cfg.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_sum, seen = 0.0, 0
        epoch_started = time.perf_counter()
        for batch_index, batch in enumerate(tqdm(loader, desc=f"{name} epoch {epoch + 1}/{cfg.epochs}")):
            batch = move_batch(batch, device)
            with autocast_context(cfg):
                logits = model_logits(model, batch, name)
                loss = F.cross_entropy(logits.float(), batch["labels"])
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite {name} loss at epoch {epoch + 1}, batch {batch_index}")
            count = len(batch["labels"])
            loss_sum += float(loss.detach()) * count
            seen += count
            # Weight by examples in this accumulation window, including a partial final window.
            window_start = (batch_index // cfg.grad_accum_steps) * cfg.grad_accum_steps
            window_examples = min(cfg.grad_accum_steps * batch_size,
                                  len(dataset) - window_start * batch_size)
            scaler.scale(loss * count / window_examples).backward()
            end_window = ((batch_index + 1) % cfg.grad_accum_steps == 0 or batch_index + 1 == len(loader))
            if end_window:
                scaler.unscale_(optimizer)
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], cfg.max_grad_norm)
                previous_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() >= previous_scale:
                    scheduler.step()
                    optimizer_steps += 1
                else:
                    skipped_steps += 1
                optimizer.zero_grad(set_to_none=True)
                if optimizer_steps and optimizer_steps % 50 == 0:
                    tracker.metrics({f"{name}_step_loss": float(loss.detach()),
                                     f"{name}_gradient_norm": float(gradient_norm),
                                     f"{name}_learning_rate": optimizer.param_groups[0]["lr"]},
                                    step=optimizer_steps)
        probabilities = predict_probabilities(model, tokenizer, valid, valid_features, name, max_length, cfg)
        metrics = classification_metrics(valid.label.to_numpy(), probabilities)
        record = {"epoch": epoch + 1, "train_loss": loss_sum / seen,
                  **{f"validation_{key}": value for key, value in metrics.items()},
                  "epoch_seconds": time.perf_counter() - epoch_started,
                  "optimizer_steps": optimizer_steps, "skipped_amp_steps": skipped_steps}
        history.append(record)
        tracker.metrics({f"{name}_{key}": value for key, value in record.items() if key != "epoch"}, step=epoch + 1)
        LOGGER.info("%s epoch results: %s", name, record)
        if metrics["log_loss"] < best_loss:
            best_loss, best_epoch, best_probabilities, stale = metrics["log_loss"], epoch + 1, probabilities.copy(), 0
            metadata = {
                "base_model_source": cfg.model_sources[name],
                "base_revision": cfg.model_revisions.get(name) or getattr(
                    model.encoder.config if name == "deberta" else model.config, "_commit_hash", None),
                "best_epoch": best_epoch, "validation_metrics": metrics,
                "max_length": max_length, "training_rows": len(train), "validation_rows": len(valid),
                "semantic_feature_mode": "paired_cosine" if name == "deberta" and cfg.use_semantic_features else "disabled",
                "sentence_max_length": cfg.sentence_max_length,
                "mlflow_run_id": tracker.run_id,
                "validation_scope": "group-disjoint development set; used for epoch selection, not an untouched final test",
            }
            save_model_bundle(model, tokenizer, model_directory, name, metadata)
        else:
            stale += 1
            if stale >= cfg.patience:
                LOGGER.info("Early stopping %s after epoch %s", name, epoch + 1)
                break
    audit = finish_parameter_audit(model, watched, optimizer_steps)
    audit["scope"] = "Initial vs final in-memory trainable tensors; selected checkpoint is identified by best_epoch."
    write_json(model_directory / "training_update_audit.json", audit)
    pd.DataFrame(history).to_csv(model_directory / "training_history.csv", index=False)
    final_metrics = classification_metrics(valid.label.to_numpy(), best_probabilities)
    write_json(model_directory / "validation_metrics.json", {
        "initial_model": baseline_metrics, "selected_checkpoint": final_metrics, "best_epoch": best_epoch,
    })
    pd.DataFrame(confusion_matrix(valid.label, best_probabilities.argmax(axis=1), labels=[0, 1, 2]),
                 index=CLASS_COLUMNS, columns=CLASS_COLUMNS).to_csv(model_directory / "confusion_matrix.csv")
    report = pd.DataFrame(best_probabilities, columns=CLASS_COLUMNS)
    report.insert(0, "id", valid.id.to_numpy())
    report["true_label"] = valid.label.to_numpy()
    report.to_csv(Path(run_dir) / f"validation_predictions_{name}.csv", index=False)
    (model_directory / "MODEL_CARD.md").write_text(
        f"# {name} human-preference classifier\n\n"
        f"Class order: {', '.join(CLASS_COLUMNS)}.\n\n"
        f"Selected epoch: {best_epoch}. Validation log loss: {best_loss:.6f}.\n\n"
        "Validation is grouped by the complete normalized question, after label QA and pair deduplication. "
        "It is a development set used for model selection. Scores are not a Kaggle test result.\n\n"
        "Base model training data must be audited separately; using a competition-finetuned base "
        "may contaminate this validation set. The default Kaggle Gemma/Llama checkpoints "
        "have already been fine-tuned on this competition task.\n",
        encoding="utf-8",
    )
    tracker.params({f"{name}_best_epoch": best_epoch})
    tracker.metrics({f"{name}_best_validation_{key}": value for key, value in final_metrics.items()})
    export_quantized_base(model, tokenizer, name, cfg, model_directory)
    del model, optimizer, scheduler, scaler, loader, dataset
    release_gpu()
    return best_probabilities, model_directory
