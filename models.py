# SPDX-License-Identifier: Apache-2.0
"""Custom DeBERTa, QLoRA model factories, and portable model bundles."""
from __future__ import annotations
import json
import pickle
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig
from safetensors.torch import save_file as save_safetensors, load_file as load_safetensors

from config import CLASS_COLUMNS, ARCHITECTURE_VERSION
from utils import verify_bundle, write_json


class CustomDebertaModel(nn.Module):
    """Pretrained text encoder + two paired-cosine features + a three-class head.

    Direct feature fusion replaces the old one-key attention block, whose attention
    weights could not meaningfully select between alternatives. This is a NEW model
    version, trained from a clean pretrained backbone, not the old full-object pickle.
    """
    def __init__(self, encoder, dropout=0.1):
        super().__init__()
        self.encoder = encoder
        hidden = encoder.config.hidden_size
        self.feature_fc = nn.Sequential(nn.Linear(2, 64), nn.GELU(), nn.Dropout(dropout),
                                        nn.Linear(64, 64), nn.GELU())
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden + 64, hidden),
                                       nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, 3))

    def forward(self, input_ids, attention_mask, similarity_features, **kwargs):
        text = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]
        features = self.feature_fc(similarity_features.to(text.dtype))
        return {"logits": self.classifier(torch.cat([text, features], dim=1))}


def hf_options(cfg, name):
    options = {"local_files_only": cfg.local_files_only, "trust_remote_code": False}
    if cfg.model_revisions.get(name):
        options["revision"] = cfg.model_revisions[name]
    return options


def load_training_model(name, cfg):
    source, options = cfg.model_sources[name], hf_options(cfg, name)
    tokenizer = AutoTokenizer.from_pretrained(source, **options)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("The tokenizer needs an existing pad or EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    if name == "deberta":
        encoder = AutoModel.from_pretrained(source, **options)
        if encoder.config.model_type not in {"deberta", "deberta-v2"}:
            raise ValueError("Choose a DeBERTa backbone for the custom DeBERTa branch")
        if cfg.gradient_checkpointing:
            encoder.gradient_checkpointing_enable()
        model = CustomDebertaModel(encoder)
        model.to("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        if not torch.cuda.is_available():
            raise RuntimeError("Gemma/Llama QLoRA requires an NVIDIA CUDA GPU")
        from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
        quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                         bnb_4bit_use_double_quant=True,
                                         bnb_4bit_compute_dtype=torch.float16)
        model = AutoModelForSequenceClassification.from_pretrained(
            source, num_labels=3, problem_type="single_label_classification", use_cache=False,
            quantization_config=quantization, torch_dtype=torch.float16,
            device_map={"": 0}, attn_implementation="eager", **options,
        )
        model.config.pad_token_id = tokenizer.pad_token_id
        model.config.id2label = dict(enumerate(CLASS_COLUMNS))
        model.config.label2id = dict(zip(CLASS_COLUMNS, range(3)))
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=cfg.gradient_checkpointing,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
        model = get_peft_model(model, LoraConfig(
            task_type=TaskType.SEQ_CLS, r=cfg.lora_r, lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout, target_modules=list(cfg.lora_target_modules),
            modules_to_save=["score"], bias="none",
        ))
        model.print_trainable_parameters()
    return model, tokenizer


def model_logits(model, batch, name):
    inputs = {key: batch[key] for key in ("input_ids", "attention_mask")}
    if name == "deberta":
        inputs["similarity_features"] = batch["similarity_features"]
    output = model(**inputs)
    return output["logits"] if isinstance(output, dict) else output.logits


def save_model_bundle(model, tokenizer, directory, name, metadata):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(directory)
    if name == "deberta":
        model.encoder.config.save_pretrained(directory / "backbone_config")
        save_safetensors({key: tensor.detach().cpu().contiguous().clone()
                          for key, tensor in model.state_dict().items()}, str(directory / "model.safetensors"))
    else:
        # Includes trained LoRA adapters AND modules_to_save=['score'].
        model.save_pretrained(directory, safe_serialization=True)
    write_json(directory / "bundle_config.json", {"backend": name,
               "architecture_version": ARCHITECTURE_VERSION,
               "class_order": CLASS_COLUMNS, **metadata})


def load_model_bundle(directory, cfg, name=None, verify=True):
    directory = Path(directory)
    if verify:
        verify_bundle(directory)
    metadata = json.loads((directory / "bundle_config.json").read_text())
    saved_name = metadata["backend"]
    if name is not None and name != saved_name:
        raise ValueError("Bundle backend does not match its configured name")
    if metadata["class_order"] != CLASS_COLUMNS or metadata["architecture_version"] != ARCHITECTURE_VERSION:
        raise ValueError("Incompatible model schema or class order")
    tokenizer = AutoTokenizer.from_pretrained(str(directory), local_files_only=True, trust_remote_code=False)
    tokenizer.padding_side = "right"
    if saved_name == "deberta":
        encoder_config = AutoConfig.from_pretrained(directory / "backbone_config", local_files_only=True)
        model = CustomDebertaModel(AutoModel.from_config(encoder_config))
        model.load_state_dict(load_safetensors(str(directory / "model.safetensors")), strict=True)
        model.to("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        if not torch.cuda.is_available():
            raise RuntimeError("Quantized Gemma/Llama inference requires a CUDA GPU")
        from peft import PeftModel
        source = cfg.base_model_overrides.get(saved_name, metadata["base_model_source"])
        kwargs = {"local_files_only": cfg.local_files_only, "trust_remote_code": False}
        if metadata.get("base_revision") and saved_name not in cfg.base_model_overrides:
            kwargs["revision"] = metadata["base_revision"]
        quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                         bnb_4bit_use_double_quant=True,
                                         bnb_4bit_compute_dtype=torch.float16)
        base = AutoModelForSequenceClassification.from_pretrained(
            source, num_labels=3, use_cache=False, quantization_config=quantization,
            torch_dtype=torch.float16, device_map={"": 0}, attn_implementation="eager", **kwargs,
        )
        base.config.pad_token_id = tokenizer.pad_token_id
        model = PeftModel.from_pretrained(base, str(directory), is_trainable=False,
                                          local_files_only=True)
    return model.eval(), tokenizer, metadata


class LegacyCustomDebertaModel(nn.Module):
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
    def find_class(self, module, name):
        if name == "CustomDebertaModel" and module in {
            "__main__", "__mp_main__", "model", "models", "finetuning_test2",
            "finetuning_test2_mlops", "finetuning_mlops",
        }:
            return LegacyCustomDebertaModel
        return super().find_class(module, name)


class _CheckpointPickleModule:
    __name__ = "legacy_checkpoint_pickle"
    Unpickler = _LegacyCheckpointUnpickler
    load = staticmethod(pickle.load)


def load_custom_checkpoint(path, *, trusted=False):
    """Compatibility utility for trusted old full-object checkpoints only.

    New training/inference uses safetensors bundles. A legacy classifier expects
    its ORIGINAL feature definition; it must not be silently mixed with the new
    paired-feature model or used to claim uncontaminated validation performance.
    """
    if not trusted:
        raise ValueError("Full-object pickle loading requires trusted=True for a checkpoint you trust")
    return torch.load(path, map_location="cpu", pickle_module=_CheckpointPickleModule, weights_only=False)

