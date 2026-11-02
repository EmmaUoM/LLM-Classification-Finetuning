"""Real tiny fused-checkpoint reloads, trainable heads, and fold aggregation."""
import json

import numpy as np
import pandas as pd
import pytest
import torch
from transformers import Gemma2Config, LlamaConfig

from config import CLASS_COLUMNS, PIPELINE_VERSION, Config
from models import (KaggleGemmaForSequenceClassification, KaggleLlamaForSequenceClassification,
                    load_sequence_checkpoint, sequence_checkpoint_layout)
import pipeline
import models
from utils import bundle_manifest


def tiny_config(config_class):
    return config_class(
        vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=1, num_labels=3, pad_token_id=0,
        max_position_embeddings=64, head_dim=8, query_pre_attn_scalar=8,
        sliding_window=32, use_cache=False,
    )


def test_training_source_override_is_applied_in_preflight_and_direct_loading(tmp_path, monkeypatch):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast
    directory = tmp_path / "local-llama"
    KaggleLlamaForSequenceClassification(tiny_config(LlamaConfig)).save_pretrained(directory)
    raw = Tokenizer(WordLevel({"[PAD]": 0, "[UNK]": 1, "[EOS]": 2}, unk_token="[UNK]"))
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw, pad_token="[PAD]",
                                        unk_token="[UNK]", eos_token="[EOS]")
    tokenizer.save_pretrained(directory)
    cfg = Config(train_backends=("llama",), model_sources={"llama": "meta-llama/old-source"},
                 base_model_overrides={"llama": str(directory)}, use_semantic_features=False,
                 output_root=str(tmp_path / "output"))
    pipeline.preflight_training_sources(cfg)
    report = json.loads((tmp_path / "output" / "training_preflight.json").read_text())
    assert report["llama"]["source"] == str(directory)
    assert report["llama"]["local_files_only"]
    assert cfg.sources_preflight_done
    cfg.model_sources = {"llama": "meta-llama/old-source"}
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="NVIDIA CUDA GPU"):
        models.load_training_model("llama", cfg)
    assert cfg.model_sources["llama"] == str(directory)
    assert not cfg.sources_preflight_done


def test_gemma_training_rejects_hub_sources_before_any_hub_request(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("Attempted a Hub config or tokenizer request")

    monkeypatch.setattr(pipeline.AutoConfig, "from_pretrained", fail)
    monkeypatch.setattr(pipeline.AutoTokenizer, "from_pretrained", fail)
    cfg = Config(train_backends=("gemma",), model_sources={"gemma": "google/gemma-2-9b-it"},
                 use_semantic_features=False, output_root=str(tmp_path))
    assert models.hf_options(cfg, "gemma")["local_files_only"]
    with pytest.raises(RuntimeError, match="Cannot train all requested models"):
        pipeline.preflight_training_sources(cfg)
    with pytest.raises(FileNotFoundError, match="Attach the Kaggle model Input"):
        models.load_training_model("gemma", cfg)
    assert not cfg.sources_preflight_done


def test_pipeline_reports_version_and_rejects_stale_configuration_objects():
    from types import SimpleNamespace
    assert pipeline.PIPELINE_VERSION == PIPELINE_VERSION == "kaggle-local-sources-v2"
    with pytest.raises(RuntimeError, match="older configuration module"):
        pipeline.run_pipeline(SimpleNamespace(mode="train"))


@pytest.mark.parametrize("model_class,config_class", [
    (KaggleGemmaForSequenceClassification, Gemma2Config),
    (KaggleLlamaForSequenceClassification, LlamaConfig),
])
def test_raw_fused_checkpoint_and_resaved_base_preserve_all_weights_and_logits(
        tmp_path, model_class, config_class):
    torch.manual_seed(7)
    model = model_class(tiny_config(config_class)).eval()
    batch = {"input_ids": torch.tensor([[1, 2, 3, 0], [2, 4, 5, 6]]),
             "attention_mask": torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]])}
    with torch.inference_mode():
        expected = model(**batch).logits
    raw = tmp_path / "raw"
    model.save_pretrained(raw, max_shard_size="1KB")
    config_path = raw / "config.json"
    config = json.loads(config_path.read_text())
    config.pop("lmsys_fused_checkpoint")
    config_path.write_text(json.dumps(config))
    assert sequence_checkpoint_layout(raw)[1]
    restored = load_sequence_checkpoint(raw, num_labels=3, attn_implementation="eager").eval()
    assert set(restored.state_dict()) == set(model.state_dict())
    with torch.inference_mode():
        torch.testing.assert_close(restored(**batch).logits, expected)
    saved = tmp_path / "saved-base"
    restored.save_pretrained(saved)
    assert not (saved / "model.safetensors.index.json").exists()
    assert sequence_checkpoint_layout(saved)[1]
    reloaded = load_sequence_checkpoint(saved, num_labels=3, attn_implementation="eager").eval()
    with torch.inference_mode():
        torch.testing.assert_close(reloaded(**batch).logits, expected)


def test_mismatched_fused_weights_fail_instead_of_using_random_parameters(tmp_path):
    model = KaggleLlamaForSequenceClassification(tiny_config(LlamaConfig))
    model.save_pretrained(tmp_path)
    from safetensors.torch import load_file, save_file
    path = tmp_path / "model.safetensors"
    weights = load_file(str(path))
    weights.pop("score.1.weight")
    save_file(weights, str(path), metadata={"format": "pt"})
    with pytest.raises(RuntimeError, match="did not load completely"):
        load_sequence_checkpoint(tmp_path, num_labels=3)


def test_fused_lora_and_entire_classification_head_update_and_reload(tmp_path):
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model
    torch.manual_seed(9)
    base = KaggleLlamaForSequenceClassification(tiny_config(LlamaConfig))
    initial_head = base.score[4].weight.detach().clone()
    raw = tmp_path / "base"
    base.save_pretrained(raw)
    model = get_peft_model(base, LoraConfig(
        task_type=TaskType.SEQ_CLS, r=2, lora_alpha=4, lora_dropout=0,
        target_modules=["q_proj", "v_proj"], modules_to_save=["score"], bias="none"))
    batch = {"input_ids": torch.tensor([[1, 2, 3], [4, 5, 6]]),
             "attention_mask": torch.ones(2, 3, dtype=torch.long)}
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.01)
    for _ in range(2):
        model.train()
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(model(**batch).logits, torch.tensor([0, 2]))
        loss.backward()
        optimizer.step()
    assert not torch.equal(model.score.modules_to_save["default"][4].weight, initial_head)
    adapter = tmp_path / "adapter"
    model.save_pretrained(adapter)
    model.eval()
    restored_base = load_sequence_checkpoint(raw, num_labels=3, attn_implementation="eager")
    restored = PeftModel.from_pretrained(restored_base, adapter).eval()
    with torch.inference_mode():
        torch.testing.assert_close(restored(**batch).logits, model(**batch).logits)


def test_submission_averages_folds_before_model_weights_without_reading_train_csv(tmp_path, monkeypatch):
    test = tmp_path / "test.csv"
    pd.DataFrame({"id": ["02", "01"], "prompt": ['["q"]', '["q2"]'],
                  "response_a": ['["a"]', '["a2"]'], "response_b": ['["b"]', '["b2"]']}).to_csv(test, index=False)
    arrays, bundles = {}, {}
    for name, values in {"deberta": [[0.8, 0.1, 0.1], [0.2, 0.6, 0.2]],
                         "gemma": [[0.1, 0.7, 0.2], [0.5, 0.1, 0.4]],
                         "llama": [[0.2, 0.3, 0.5], [0.4, 0.4, 0.2]]}.items():
        bundles[name] = []
        for fold, probabilities in enumerate(values):
            directory = tmp_path / name / str(fold)
            directory.mkdir(parents=True)
            (directory / "bundle_config.json").write_text(json.dumps({
                "backend": name, "max_length": 32, "semantic_feature_mode": "disabled",
                "base_model_source": str(tmp_path / "base"),
            }))
            bundle_manifest(directory)
            bundles[name].append(str(directory))
            arrays[str(directory)] = np.array([probabilities, probabilities[::-1]])
    calls = []

    def load(directory, cfg, name=None, verify=True):
        calls.append((name, str(directory)))
        return str(directory), None, {"max_length": 32}

    monkeypatch.setattr(pipeline, "load_model_bundle", load)
    monkeypatch.setattr(pipeline, "predict_probabilities",
                        lambda model, *args: arrays[model])
    cfg = Config(mode="submit", test_csv=str(test), train_csv="/does/not/exist/train.csv",
                 output_root=str(tmp_path / "output"), inference_bundles=bundles,
                 enable_mlflow=False)
    result = pipeline.run_pipeline(cfg)
    actual = pd.read_csv(result["submission"], dtype={"id": str})
    expected = sum(np.mean([arrays[path] for path in bundles[name]], axis=0) * cfg.ensemble_weights[name]
                   for name in bundles)
    np.testing.assert_allclose(actual[CLASS_COLUMNS], expected)
    assert actual.id.tolist() == ["02", "01"]
    assert len(calls) == 6
