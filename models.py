# SPDX-License-Identifier: Apache-2.0
"""Custom DeBERTa, QLoRA model factories, and portable model bundles."""
from __future__ import annotations
import json
import pickle
import os
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig
from safetensors.torch import save_file as save_safetensors, load_file as load_safetensors

from config import CLASS_COLUMNS, ARCHITECTURE_VERSION
from config import LOGGER
from submission_support import find_offline_base
from transformers import Gemma2ForSequenceClassification, LlamaForSequenceClassification
from transformers.activations import ACT2FN
from utils import bundle_manifest, verify_bundle, write_json


class KaggleFusedMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden, intermediate = config.hidden_size, config.intermediate_size
        bias = bool(getattr(config, "mlp_bias", False))
        self.gate_up_proj = nn.Linear(hidden, 2 * intermediate, bias=bias)
        self.down_proj = nn.Linear(intermediate, hidden, bias=bias)
        activation = getattr(config, "hidden_activation", None) or config.hidden_act
        self.act_fn = ACT2FN[activation]

    def forward(self, values):
        gate, up = self.gate_up_proj(values).chunk(2, dim=-1)
        return self.down_proj(self.act_fn(gate) * up)


def install_kaggle_checkpoint_layers(model, config):
    for layer in model.model.layers:
        layer.mlp = KaggleFusedMLP(config)
    model.score = nn.Sequential(
        nn.Dropout(0.1),
        nn.Linear(config.hidden_size, config.hidden_size // 2),
        nn.Dropout(0.1), nn.GELU(),
        nn.Linear(config.hidden_size // 2, config.num_labels),
    )
    # This marker survives save_pretrained(), including the shared 4-bit base.
    model.config.lmsys_fused_checkpoint = True
    model.post_init()


class KaggleGemmaForSequenceClassification(Gemma2ForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        install_kaggle_checkpoint_layers(self, config)


class KaggleLlamaForSequenceClassification(LlamaForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        install_kaggle_checkpoint_layers(self, config)


def sequence_checkpoint_layout(source):
    directory = Path(source)
    if not directory.is_dir():
        raise FileNotFoundError(
            f"Attach the Kaggle model Input and copy its mounted directory: {directory}")
    config = AutoConfig.from_pretrained(
        str(directory), local_files_only=True, trust_remote_code=False)
    if config.model_type not in {"gemma2", "llama"}:
        raise ValueError(f"Expected Gemma 2 or Llama checkpoint: {directory}")
    if (directory / "adapter_config.json").is_file():
        raise ValueError(f"Choose a full model checkpoint, not a LoRA adapter: {directory}")
    index = directory / "model.safetensors.index.json"
    keys = set(json.loads(index.read_text())["weight_map"]) if index.is_file() else set()
    fused = bool(getattr(config, "lmsys_fused_checkpoint", False)) or any(
        ".mlp.gate_up_proj." in key for key in keys)
    if not keys and not (directory / "model.safetensors").is_file() and not (
            directory / "pytorch_model.bin").is_file() and not (
            directory / "pytorch_model.bin.index.json").is_file():
        raise FileNotFoundError(f"No model weights found in {directory}")
    return config, fused


def load_sequence_checkpoint(source, **kwargs):
    # Explicit Hub sources remain supported; the default Kaggle sources are local.
    if not Path(source).is_dir() and not Path(source).is_absolute() and not str(source).startswith("."):
        return AutoModelForSequenceClassification.from_pretrained(source, **kwargs)
    config, fused = sequence_checkpoint_layout(source)
    model_class = AutoModelForSequenceClassification
    if fused:
        model_class = {"gemma2": KaggleGemmaForSequenceClassification,
                       "llama": KaggleLlamaForSequenceClassification}[config.model_type]
    kwargs["local_files_only"] = True
    kwargs["trust_remote_code"] = False
    model, info = model_class.from_pretrained(
        str(source), output_loading_info=True, **kwargs)
    if fused:
        failures = {key: info.get(key) for key in (
            "missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs") if info.get(key)}
        if failures:
            raise RuntimeError(f"Kaggle checkpoint did not load completely: {failures}")
    LOGGER.info("Loaded %s: %s", config.model_type,
                "LMSYS fused MLP/head" if fused else "standard HF layout")
    return model


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
    options = {"local_files_only": bool(name in {"gemma", "llama"} or cfg.local_files_only
                                        or Path(cfg.model_sources[name]).is_dir()),
               "trust_remote_code": False}
    if cfg.model_revisions.get(name):
        options["revision"] = cfg.model_revisions[name]
    return options


def load_training_model(name, cfg):
    cfg.resolve_training_sources(resolve_inputs=True)
    source, options = cfg.model_sources[name], hf_options(cfg, name)
    if name in {"gemma", "llama"}:
        sequence_checkpoint_layout(source)
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
                                         llm_int8_skip_modules=["score"],
                                         bnb_4bit_use_double_quant=True,
                                         bnb_4bit_compute_dtype=torch.float16)
        model = load_sequence_checkpoint(
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
        source = find_offline_base(directory, metadata, saved_name,
                                   cfg.base_model_overrides, cfg.auto_discover_roots)
        kwargs = {"local_files_only": cfg.local_files_only, "trust_remote_code": False}
        if metadata.get("base_revision") and not Path(source).is_dir():
            kwargs["revision"] = metadata["base_revision"]
        quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                         llm_int8_skip_modules=["score"],
                                         bnb_4bit_use_double_quant=True,
                                         bnb_4bit_compute_dtype=torch.float16)
        base_config = AutoConfig.from_pretrained(source, **kwargs)
        # Saved 4-bit bases already carry their quantization config and packed weights.
        quantization_options = {} if getattr(base_config, "quantization_config", None) else {
            "quantization_config": quantization}
        base = load_sequence_checkpoint(
            source, num_labels=3, use_cache=False,
            torch_dtype=torch.float16, device_map={"": 0}, attn_implementation="eager",
            **quantization_options, **kwargs,
        )
        base.config.pad_token_id = tokenizer.pad_token_id
        model = PeftModel.from_pretrained(base, str(directory), is_trainable=False,
                                          local_files_only=True)
    return model.eval(), tokenizer, metadata


def export_quantized_base(model, tokenizer, name, cfg, bundle_directory):
    """Save one frozen 4-bit base per training group; adapters remain separate per fold.

    Called only AFTER the selected adapter/head and all training audits are saved.
    unload() removes adapters without merging, so no fold's LoRA is applied twice.
    """
    if name == "deberta" or not cfg.export_quantized_bases:
        return
    directory = (Path(cfg.output_root) / "submission_assets" / "base_models"
                 / cfg.training_group_id / name)
    marker = directory / "offline_base.json"
    if not (marker.is_file() and (directory / "manifest.json").is_file()):
        LOGGER.info("Exporting shared %s 4-bit base for offline submission", name)
        directory.mkdir(parents=True, exist_ok=True)
        base = model.base_model.unload()
        base.config.use_cache = False
        base.save_pretrained(str(directory), safe_serialization=True, max_shard_size="2GB")
        tokenizer.save_pretrained(str(directory))
        write_json(marker, {"backend": name, "base_model_source": cfg.model_sources[name],
                           "training_group_id": cfg.training_group_id, "lora_merged": False})
        bundle_manifest(directory)
        del base
    metadata_path = Path(bundle_directory) / "bundle_config.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["offline_base_relative"] = os.path.relpath(directory, bundle_directory)
    metadata["training_group_id"] = cfg.training_group_id
    write_json(metadata_path, metadata)


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

