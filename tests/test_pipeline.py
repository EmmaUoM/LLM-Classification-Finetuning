import ast
import copy
import json
import math
import sqlite3
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import DebertaV2Config, DebertaV2Model, PreTrainedTokenizerFast
import config, data, models, inference, evaluation, tracking, utils, training, pipeline
torch.set_num_threads(1)
ROOT = Path(__file__).resolve().parents[1]

def example(i, prompt=None, a=None, b=None, label=None):
    label = i % 3 if label is None else label
    return {'id': str(i), 'prompt': json.dumps([prompt or f'Question {i}']), 'response_a': json.dumps([a or f'Good answer {i}']), 'response_b': json.dumps([b or f'Other answer {i}']), **{key: int(j == label) for j, key in enumerate(config.CLASS_COLUMNS)}}

@pytest.fixture(scope='module')
def tokenizer():
    words = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', 'Prompt', 'Response', 'A', 'B', ':', 'Question', 'Good', 'Other', 'answer', 'first', 'second', 'third', 'a_word', 'b_word']
    words += [str(i) for i in range(100)]
    raw = Tokenizer(WordLevel({w: i for i, w in enumerate(words)}, unk_token='[UNK]'))
    raw.pre_tokenizer = Whitespace()
    return PreTrainedTokenizerFast(tokenizer_object=raw, pad_token='[PAD]', unk_token='[UNK]', cls_token='[CLS]', sep_token='[SEP]')

def test_safe_multiturn_parsing_and_code_formatting(tmp_path):
    assert data.parse_turns('["hello", null, "world"]')[0] == ['hello', '', 'world']
    assert data.parse_turns("['hello', None]")[0] == ['hello', '']
    assert data.join_turns(['', '']) == ''
    assert data.normalize_text('A\r\n    code()\x00\nB') == 'A\n    code()\nB'
    target = tmp_path / 'must_not_exist'
    malicious = f"[__import__('pathlib').Path({str(target)!r}).write_text('bad')]"
    turns, status = data.parse_turns(malicious)
    assert turns == [malicious]
    assert not target.exists()
    with pytest.raises(ValueError):
        data.parse_turns('[{"nested": "object"}]')

def test_train_cleaning_duplicates_swaps_conflicts_and_labels():
    rows = [example(i) for i in range(9)]
    swapped = dict(rows[0], id='duplicate_swapped', response_a=rows[0]['response_b'], response_b=rows[0]['response_a'], winner_model_a=0, winner_model_b=1)
    conflict = dict(rows[1], id='conflict', winner_model_a=0, winner_model_b=0, winner_tie=1)
    invalid = dict(example(100), winner_model_a=1, winner_model_b=1)
    empty = dict(example(101), prompt='[null, null]')
    cleaned, quality, rejected = data.clean_dataframe(pd.DataFrame(rows + [swapped, conflict, invalid, empty]), True)
    assert len(cleaned) == 8
    assert quality['drop_reasons']['duplicate_pair_including_ab_swaps'] == 1
    assert quality['drop_reasons']['conflicting_pair_labels'] == 2
    assert quality['drop_reasons']['invalid_one_hot_label'] == 1
    assert quality['drop_reasons']['empty_prompt'] == 1
    assert len(rejected) == 5
    assert '1' not in set(cleaned.id)

def test_id_conflicts_are_removed_instead_of_guessing():
    raw = pd.DataFrame([example(0), example(1), dict(example(2), id='1'), example(3)])
    cleaned, quality, _ = data.clean_dataframe(raw, True)
    assert cleaned.id.tolist() == ['0', '3']
    assert quality['drop_reasons']['conflicting_id'] == 2

def test_test_rows_are_preserved_even_for_empty_texts():
    frame = pd.DataFrame([example(3), dict(example(1), response_a=None, prompt='[null]')])
    test, report, _ = data.clean_dataframe(frame, False)
    assert test.id.tolist() == ['3', '1']
    assert len(test) == len(frame)
    assert report['dropped_rows'] == 0
    assert test.iloc[1].response_a == '' and test.iloc[1].prompt == ''
    with pytest.raises(ValueError):
        data.clean_dataframe(pd.concat([frame, frame]), False)

def test_grouped_validation_has_no_shared_questions_or_pairs():
    raw = pd.DataFrame([example(i * 3 + j, prompt=f'Question group {i}') for i in range(20) for j in range(3)])
    cleaned, _, _ = data.clean_dataframe(raw, True)
    train, valid = data.split_training_data(cleaned, config.Config())
    assert not set(train._prompt_group) & set(valid._prompt_group)
    assert not set(train._pair_key) & set(valid._pair_key)
    assert set(train.label) == set(valid.label) == {0, 1, 2}
    assert len(train) + len(valid) == len(cleaned)

def test_features_use_corresponding_prompt_and_ignore_unrelated_rows():
    prompt = np.array([[1, 0], [0, 1]], dtype=float)
    a = np.array([[0, 1], [0, 1]], dtype=float)
    b = np.array([[1, 0], [1, 0]], dtype=float)
    features = data.paired_cosine_features(prompt, a, b)
    np.testing.assert_allclose(features, [[0, 1], [1, 0]])
    extended = data.paired_cosine_features(np.vstack([prompt, [1, 1]]), np.vstack([a, [1, 0]]), np.vstack([b, [0, 1]]))
    np.testing.assert_array_equal(extended[:2], features)
    np.testing.assert_array_equal(data.paired_cosine_features([[0, 0]], [[1, 1]], [[0, 0]]), [[0, 0]])

def test_token_budget_retains_both_answers_and_swap_maps_label_and_features(tokenizer):
    raw = pd.DataFrame([example(0, prompt='Question ' * 100, a='a_word ' * 100, b='b_word ' * 100)])
    cleaned, _, _ = data.clean_dataframe(raw, True)
    normal = data.PreferenceDataset(cleaned, tokenizer, 48, np.array([[0.2, 0.8]]))
    swapped = data.PreferenceDataset(cleaned, tokenizer, 48, np.array([[0.2, 0.8]]), augment_probability=1.0)
    first, second = (normal[0], swapped[0])
    assert len(first['input_ids']) <= 48
    assert tokenizer.convert_tokens_to_ids('a_word') in first['input_ids']
    assert tokenizer.convert_tokens_to_ids('b_word') in first['input_ids']
    assert first['labels'] == 0 and second['labels'] == 1
    np.testing.assert_allclose(second['similarity_features'], [0.8, 0.2])

def test_tta_restores_original_class_order(tokenizer):

    class FeatureBasedClassifier(torch.nn.Module):

        def __init__(self):
            super().__init__()
            self.dummy = torch.nn.Parameter(torch.zeros(1))

        def forward(self, input_ids, attention_mask, similarity_features):
            return {'logits': torch.cat([similarity_features * 5, torch.zeros((len(input_ids), 1))], dim=1) + self.dummy}
    frame, _, _ = data.clean_dataframe(pd.DataFrame([example(0), example(1)]), True)
    features = np.array([[0.8, 0.1], [0.1, 0.8]])
    cfg = config.Config(use_amp=False, tta=False)
    model = FeatureBasedClassifier()
    expected = inference.predict_probabilities(model, tokenizer, frame, features, 'deberta', 48, cfg)
    cfg.tta = True
    actual = inference.predict_probabilities(model, tokenizer, frame, features, 'deberta', 48, cfg)
    np.testing.assert_allclose(actual, expected, atol=1e-07)
    assert actual.argmax(1).tolist() == [0, 1]

def test_probability_checks_and_available_model_weight_normalization(tmp_path):
    with pytest.raises(ValueError):
        evaluation.validate_probabilities([[0.5, 0.5, float('nan')]], 1)
    with pytest.raises(ValueError):
        evaluation.validate_probabilities([[1, 1, 1]], 1)
    first = np.array([[0.7, 0.2, 0.1]])
    probabilities, weights = evaluation.weighted_ensemble({'deberta': first}, config.Config().ensemble_weights)
    np.testing.assert_allclose(probabilities, first)
    assert weights == {'deberta': 1.0}
    submission = evaluation.make_submission(pd.DataFrame({'id': ['123']}), probabilities, tmp_path / 'submission.csv')
    assert submission.columns.tolist() == ['id', *config.CLASS_COLUMNS]

@pytest.fixture(scope='module')
def trained_project(tmp_path_factory, tokenizer):
    from sentence_transformers import SentenceTransformer, models
    directory = tmp_path_factory.mktemp('tiny_training')
    source = directory / 'base'
    encoder = DebertaV2Model(DebertaV2Config(vocab_size=len(tokenizer), hidden_size=16, num_hidden_layers=1, num_attention_heads=2, intermediate_size=32, max_position_embeddings=128, hidden_dropout_prob=0.0, attention_probs_dropout_prob=0.0))
    encoder.save_pretrained(source)
    tokenizer.save_pretrained(source)
    transformer = models.Transformer(str(source), max_seq_length=32)
    sentence_encoder = SentenceTransformer(modules=[transformer, models.Pooling(16)], device='cpu')
    sentence_path = directory / 'sentence_encoder'
    sentence_encoder.save_pretrained(str(sentence_path))
    train_path = directory / 'train.csv'
    pd.DataFrame([example(i) for i in range(61)]).to_csv(train_path, index=False)
    cfg = config.Config(train_csv=str(train_path), output_root=str(directory / 'output'), epochs=2, patience=3, deberta_batch_size=5, grad_accum_steps=3, eval_batch_size=6, deberta_max_length=64, sentence_max_length=32, sentence_model=str(sentence_path), model_sources={'deberta': str(source)}, allow_cpu_training=True, local_files_only=True, use_amp=False, enable_mlflow=False, encoder_lr=0.001, head_lr=0.001, source_snapshot=str(ROOT))
    result = pipeline.run_pipeline(cfg)
    return (directory, cfg, result)

def test_real_backprop_updates_encoder_features_and_classifier(trained_project):
    _, cfg, result = trained_project
    bundle = Path(result['model_bundles']['deberta'])
    audit = json.loads((bundle / 'training_update_audit.json').read_text())
    assert audit['real_parameter_update_verified']
    assert all((item['changed'] for item in audit['watched_parameters'].values()))
    train_rows = len(pd.read_csv(Path(result['run_directory']) / 'train_split.csv'))
    expected_updates = math.ceil(math.ceil(train_rows / cfg.deberta_batch_size) / cfg.grad_accum_steps) * cfg.epochs
    assert audit['optimizer_steps'] == expected_updates
    assert utils.verify_bundle(bundle)

def test_best_checkpoint_roundtrip_and_offline_submission_without_train_csv(trained_project):
    directory, cfg, result = trained_project
    bundle = Path(result['model_bundles']['deberta'])
    cleaned, _, _ = data.clean_dataframe(pd.read_csv(cfg.train_csv, dtype={'id': str}), True)
    _, valid = data.split_training_data(cleaned, cfg)
    encoder = data.load_sentence_encoder(bundle / 'semantic_encoder', cfg)
    features = data.semantic_features(valid, encoder, 8)
    model, tokenizer, metadata = models.load_model_bundle(bundle, cfg)
    values = inference.predict_probabilities(model, tokenizer, valid, features, 'deberta', metadata['max_length'], cfg)
    saved = pd.read_csv(Path(result['run_directory']) / 'validation_predictions_deberta.csv')
    np.testing.assert_allclose(values, saved[config.CLASS_COLUMNS].to_numpy(), atol=1e-06)
    test_path = directory / 'test.csv'
    test = pd.DataFrame([example(88), dict(example(2), response_a=None), example(37)])
    test[['id', *config.TEXT_COLUMNS]].to_csv(test_path, index=False)
    submit_cfg = config.Config(mode='submit', train_csv='/definitely/not/read/train.csv', test_csv=str(test_path), output_root=str(directory / 'submit_output'), inference_bundles={'deberta': str(bundle)}, enable_mlflow=False, use_amp=False, eval_batch_size=3)
    submitted = pipeline.run_pipeline(submit_cfg)
    output = pd.read_csv(submitted['submission'], dtype={'id': str})
    assert output.id.tolist() == test.id.tolist()
    assert output.columns.tolist() == ['id', *config.CLASS_COLUMNS]
    evaluation.validate_probabilities(output[config.CLASS_COLUMNS].to_numpy(), 3)
    summary = json.loads((Path(submitted['run_directory']) / 'run_summary.json').read_text())
    assert summary['status'] == 'FINISHED'

def test_model_manifest_detects_changes(trained_project, tmp_path):
    _, _, result = trained_project
    import shutil
    bundle = tmp_path / 'model'
    shutil.copytree(result['model_bundles']['deberta'], bundle)
    (bundle / 'MODEL_CARD.md').write_text('modified')
    with pytest.raises(ValueError, match='has changed'):
        utils.verify_bundle(bundle)

def test_mlflow_sqlite_success_and_failure_are_real(tmp_path):
    import mlflow
    cfg = config.Config(output_root=str(tmp_path), enable_mlflow=True, experiment_name='local_test')
    with tracking.RunTracker(cfg, tmp_path / 'success') as tracker:
        tracker.metrics({'validation_log_loss': 0.75}, step=1)
        success_id = tracker.run_id
    client = mlflow.tracking.MlflowClient()
    assert client.get_run(success_id).info.status == 'FINISHED'
    assert client.get_run(success_id).data.metrics['validation_log_loss'] == 0.75
    with pytest.raises(RuntimeError, match='deliberate training failure'):
        with tracking.RunTracker(cfg, tmp_path / 'failure') as tracker:
            failure_id = tracker.run_id
            raise RuntimeError('deliberate training failure')
    assert client.get_run(failure_id).info.status == 'FAILED'
    assert (tmp_path / 'failure' / 'failure.txt').exists()

def test_lora_and_classification_head_update_and_reload_on_cpu(tmp_path, tokenizer):
    from transformers import LlamaConfig, LlamaForSequenceClassification
    from peft import LoraConfig, TaskType, get_peft_model, PeftModel
    torch.manual_seed(7)
    config = LlamaConfig(vocab_size=len(tokenizer), hidden_size=16, intermediate_size=32, num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=128, num_labels=3, pad_token_id=0)
    base = LlamaForSequenceClassification(config)
    initial_base = copy.deepcopy(base.state_dict())
    model = get_peft_model(base, LoraConfig(task_type=TaskType.SEQ_CLS, r=2, lora_alpha=4, lora_dropout=0.0, target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'], modules_to_save=['score'], bias='none'))
    watched = training.parameter_watchlist(model, 'llama')
    batch = {'input_ids': torch.randint(1, len(tokenizer), (3, 16)), 'attention_mask': torch.ones((3, 16), dtype=torch.long)}
    optimizer = torch.optim.AdamW([param for param in model.parameters() if param.requires_grad], lr=0.01)
    for _ in range(2):
        model.train()
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(models.model_logits(model, batch, 'llama'), torch.tensor([0, 1, 2]))
        loss.backward()
        optimizer.step()
    audit = training.finish_parameter_audit(model, watched, 2)
    assert audit['watched_parameters']['classification_head']['changed']
    assert audit['watched_parameters']['lora_B']['changed']
    model.eval()
    expected = models.model_logits(model, batch, 'llama').detach()
    directory = tmp_path / 'adapter'
    models.save_model_bundle(model, tokenizer, directory, 'llama', {})
    base2 = LlamaForSequenceClassification(config)
    base2.load_state_dict(initial_base)
    restored = PeftModel.from_pretrained(base2, directory, is_trainable=False).eval()
    actual = models.model_logits(restored, batch, 'llama').detach()
    torch.testing.assert_close(actual, expected, atol=1e-06, rtol=1e-05)


def test_cli_submit_exports_predictions_and_records_real_label_metrics(trained_project, tmp_path):
    import os
    import subprocess
    import sys
    _, _, trained = trained_project
    raw = pd.DataFrame([example(81), example(8), example(50)])
    test_path, labels_path = tmp_path / 'test.csv', tmp_path / 'labels.csv'
    raw[['id', *config.TEXT_COLUMNS]].to_csv(test_path, index=False)
    raw[['id', *config.CLASS_COLUMNS]].iloc[::-1].to_csv(labels_path, index=False)
    output = tmp_path / 'cli_output'
    settings = tmp_path / 'submit.json'
    settings.write_text(json.dumps({
        'mode': 'submit', 'test_csv': str(test_path), 'labels_csv': str(labels_path),
        'train_csv': '/never/read/train.csv', 'output_root': str(output),
        'inference_bundles': trained['model_bundles'], 'enable_mlflow': False, 'use_amp': False,
    }))
    environment = {**os.environ, 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}
    completed = subprocess.run([sys.executable, str(ROOT / 'main.py'), '--config', str(settings)],
                               cwd=tmp_path, env=environment, text=True, capture_output=True, timeout=90)
    assert completed.returncode == 0, completed.stderr
    submission = pd.read_csv(output / 'submission.csv', dtype={'id': str})
    assert submission.id.tolist() == raw.id.tolist()
    run_dir = next((output / 'runs').iterdir())
    assert json.loads((run_dir / 'run_summary.json').read_text())['status'] == 'FINISHED'
    metrics = [json.loads(line) for line in (run_dir / 'metrics.jsonl').read_text().splitlines()]
    assert any('ensemble_evaluation_log_loss' in item for item in metrics)
    assert 'Submission ready' in (run_dir / 'pipeline.log').read_text()
    manifest = json.loads((run_dir / 'source_manifest.json').read_text())
    assert {'models.py', 'training.py', 'utils.py'} <= {item['path'] for item in manifest['files']}
