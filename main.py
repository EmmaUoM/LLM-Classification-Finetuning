#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
主要流程：
1. 准备测试数据及交换 A/B 回答后的数据。
2. Gemma2 推理。
3. Llama3 推理。
4. SentenceTransformer + FAISS 提取相似度特征。
5. 自定义 DeBERTa 分类模型推理。
6. 加权生成 submission.csv。

需要原 notebook 的依赖及模型文件。
执行环境包括：
Python 3.10、torch 2.1.2、transformers 4.42.3、
triton 2.2.0、原 notebook 的自定义 xformers wheel、
sentence-transformers 3.3.1、faiss-cpu 1.7.2、
numpy 1.26.4、pandas、pyarrow、tqdm、accelerate。
"""

import argparse
import gc
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


class CustomDebertaModel(nn.Module):
    """保留原模型类名和结构，以加载完整模型 checkpoint。"""

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


def clear_gpu_memory():
    gc.collect()

    for index in range(torch.cuda.device_count()):
        with torch.cuda.device(index):
            torch.cuda.empty_cache()


def prepare_test_files(test_csv, output_dir):
    df = pd.read_csv(test_csv)

    required = {"id", "prompt", "response_a", "response_b"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing CSV columns: {sorted(missing)}")

    if df.empty:
        raise ValueError("The test CSV is empty.")

    # LMSYSDataset 需要这些列；这里只是占位标签。
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


@torch.no_grad()
def predict_llm(input_root, output_dir, model_kind, num_workers):
    from xformers.ops.fmha.attn_bias import BlockDiagonalCausalMask

    from human_pref.data.collators import (
        ShardedMaxTokensCollator,
        VarlenCollator,
    )
    from human_pref.data.dataset import LMSYSDataset
    from human_pref.data.processors import ProcessorPAB
    from human_pref.utils import to_device

    if model_kind == "gemma":
        from human_pref.models.modeling_gemma2 import (
            Gemma2ForSequenceClassification,
        )

        model_cls = Gemma2ForSequenceClassification
        checkpoint = input_root / "lmsys-checkpoints-0-0805"
        parquet_path = output_dir / "test.parquet"
        num_layers = 42
        support_system_role = False
        output_file = output_dir / "prob_m0.npy"

    elif model_kind == "llama":
        from human_pref.models.modeling_llama import (
            LlamaForSequenceClassification,
        )

        model_cls = LlamaForSequenceClassification
        checkpoint = input_root / "lmsys-checkpoints-3-0805"
        parquet_path = output_dir / "test_swap.parquet"
        num_layers = 32
        support_system_role = True
        output_file = output_dir / "prob_m3.npy"

    else:
        raise ValueError(f"Unknown model kind: {model_kind}")

    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint))

    if model_kind == "llama":
        tokenizer.deprecation_warnings[
            "sequence-length-is-longer-than-the-specified-maximum"
        ] = True

    processor = ProcessorPAB(
        tokenizer=tokenizer,
        max_length=4096,
        support_system_role=support_system_role,
    )

    dataset = LMSYSDataset(
        csv_file=str(parquet_path),
        query=None,
        processor=processor,
        include_swap=False,
        is_parquet=True,
    )

    loader = DataLoader(
        dataset,
        batch_size=80,
        num_workers=num_workers,
        shuffle=False,
        collate_fn=ShardedMaxTokensCollator(
            max_tokens=8192,
            base_collator=VarlenCollator(),
        ),
    )

    # 将模型前后两部分分别放到两张 GPU。
    device_map = {
        "model.embed_tokens": "cuda:0",
        "model.norm": "cuda:1",
        "score": "cuda:1",
    }

    for index in range(num_layers):
        device_map[f"model.layers.{index}"] = (
            "cuda:0" if index < num_layers // 2 else "cuda:1"
        )

    model = model_cls.from_pretrained(
        str(checkpoint),
        torch_dtype=torch.float16,
        device_map=device_map,
    )

    model.eval()
    config = model.config

    if model_kind == "gemma":
        dim = config.head_dim
    else:
        dim = config.hidden_size // config.num_attention_heads

    inv_freq = 1.0 / (
        config.rope_theta
        ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
    )

    inv_freq0 = inv_freq.to("cuda:0")
    inv_freq1 = inv_freq.to("cuda:1")

    previous = None
    logits_batches = []

    for batch in tqdm(loader, desc=f"{model_kind} inference"):
        for micro_batch in batch:
            input_ids = to_device(
                micro_batch["input_ids"],
                "cuda:0",
            )

            seq_info = to_device(
                {
                    "cu_seqlens": micro_batch["cu_seqlens"],
                    "position_ids": micro_batch["position_ids"],
                    "max_seq_len": micro_batch["max_seq_len"],
                    "attn_bias": BlockDiagonalCausalMask.from_seqlens(
                        micro_batch["seq_lens"]
                    ),
                },
                "cuda:0",
            )

            with torch.cuda.amp.autocast():
                if previous is not None:
                    previous_info, previous_hidden = previous

                    logits = model.forward_part2(
                        previous_hidden,
                        previous_info,
                        inv_freq1,
                    )

                    logits_batches.append(logits.cpu())

                hidden = model.forward_part1(
                    input_ids,
                    seq_info,
                    inv_freq0,
                )

                previous = to_device(
                    [seq_info, hidden],
                    "cuda:1",
                )

    if previous is None:
        raise ValueError(f"No input batches for {model_kind}.")

    # 处理最后一个 micro-batch。
    with torch.cuda.amp.autocast():
        previous_info, previous_hidden = previous

        logits = model.forward_part2(
            previous_hidden,
            previous_info,
            inv_freq1,
        )

        logits_batches.append(logits.cpu())

    probabilities = (
        torch.cat(logits_batches, dim=0)
        .softmax(-1)
        .numpy()
    )

    np.save(output_file, probabilities)

    print(f"Saved {output_file}: {probabilities.shape}")

    # 原 notebook 的 evaluate() 使用占位标签，
    # 因此这里省略该评估输出。


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
        # 保留原 notebook 使用的 CSV 字符串格式。
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

    # 保留原逻辑：寻找与测试集中任意 prompt 的最大相似度。
    result["similarity_a"] = index.search(
        response_a_embeddings,
        k=1,
    )[0][:, 0]

    result["similarity_b"] = index.search(
        response_b_embeddings,
        k=1,
    )[0][:, 0]

    return result


@torch.no_grad()
def predict_custom_model(df, input_root, output_dir):
    model_dir = (
        input_root
        / "akemiiiiii"
        / "custom_model_dir"
    )

    # 原文件保存的是完整模型对象。
    # weights_only=False 仅用于可信来源的 checkpoint。
    model = torch.load(
        model_dir / "custom_model_complete.pth",
        map_location="cpu",
        weights_only=False,
    )

    if not isinstance(model, nn.Module):
        raise TypeError(
            "Expected the notebook's full nn.Module checkpoint."
        )

    device = torch.device("cuda:0")
    model.to(device).eval()

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    predictions = []

    for start in tqdm(
        range(0, len(df), 8),
        desc="Custom model inference",
    ):
        batch = df.iloc[start:start + 8]

        texts = [
            (
                f"Prompt: {row.prompt} "
                f"Response A: {row.response_a} "
                f"Response B: {row.response_b}"
            )
            for row in batch.itertuples(index=False)
        ]

        inputs = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors="pt",
        )

        features = torch.tensor(
            batch[
                ["similarity_a", "similarity_b"]
            ].to_numpy(dtype=np.float32),
            dtype=torch.float32,
            device=device,
        )

        outputs = model(
            input_ids=inputs["input_ids"].to(device),
            attention_mask=inputs["attention_mask"].to(device),
            similarity_features=features,
        )

        predictions.append(
            outputs["logits"].softmax(-1).cpu().numpy()
        )

    probabilities = np.concatenate(predictions, axis=0)

    np.save(
        output_dir / "prob_faiss.npy",
        probabilities,
    )

    print(f"Saved prob_faiss.npy: {probabilities.shape}")


def make_submission(df, output_dir):
    gemma = np.load(output_dir / "prob_m0.npy")

    # Llama 输入交换过 A/B，这里恢复类别顺序。
    llama = np.load(
        output_dir / "prob_m3.npy"
    )[:, [1, 0, 2]]

    custom = np.load(output_dir / "prob_faiss.npy")

    for name, probabilities in [
        ("gemma", gemma),
        ("llama", llama),
        ("custom", custom),
    ]:
        if probabilities.shape != (len(df), 3):
            raise ValueError(
                f"{name}: unexpected shape {probabilities.shape}"
            )

        if not np.isfinite(probabilities).all():
            raise ValueError(
                f"{name}: non-finite probabilities"
            )

    predictions = np.average(
        [gemma, llama, custom],
        axis=0,
        weights=[0.5, 0.25, 0.25],
    )

    columns = [
        "winner_model_a",
        "winner_model_b",
        "winner_tie",
    ]

    submission = pd.DataFrame(
        predictions,
        columns=columns,
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

    print(submission.head().to_string(index=False))
    print(f"Saved {output_dir / 'submission.csv'}")


def main():
    parser = argparse.ArgumentParser(
        description="Run the finetuning_test2 inference ensemble."
    )

    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("/kaggle/input"),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./output"),
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
    )

    args = parser.parse_args()

    input_root = args.input_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    required = [
        input_root / "llm-classification-finetuning" / "test.csv",
        input_root / "lmsys-modules-0805",
        input_root / "lmsys-checkpoints-0-0805",
        input_root / "lmsys-checkpoints-3-0805",
        input_root / "some-pack" / "sentence-transformer-model",
        (
            input_root
            / "akemiiiiii"
            / "custom_model_dir"
            / "custom_model_complete.pth"
        ),
    ]

    missing = [
        str(path)
        for path in required
        if not path.exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Missing notebook inputs:\n" + "\n".join(missing)
        )

    if torch.cuda.device_count() < 2:
        raise RuntimeError(
            "The original LLM pipeline requires "
            "two NVIDIA CUDA GPUs."
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # 替代 notebook 中的 !cp 命令。
    shutil.copytree(
        input_root / "lmsys-modules-0805",
        output_dir / "human_pref",
        dirs_exist_ok=True,
    )

    sys.path.insert(0, str(output_dir))

    df = prepare_test_files(
        required[0],
        output_dir,
    )

    predict_llm(
        input_root,
        output_dir,
        "gemma",
        args.num_workers,
    )
    clear_gpu_memory()

    predict_llm(
        input_root,
        output_dir,
        "llama",
        args.num_workers,
    )
    clear_gpu_memory()

    df = compute_semantic_features(
        df,
        input_root,
    )
    clear_gpu_memory()

    predict_custom_model(
        df,
        input_root,
        output_dir,
    )
    clear_gpu_memory()

    make_submission(
        df,
        output_dir,
    )


if __name__ == "__main__":
    main()
