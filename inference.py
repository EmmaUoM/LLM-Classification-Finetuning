# SPDX-License-Identifier: Apache-2.0
"""GPU inference for Gemma, Llama, and the custom DeBERTa classifier."""

import gc

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from config import LOGGER
from models import load_custom_checkpoint


def clear_gpu_memory():
    gc.collect()

    for index in range(torch.cuda.device_count()):
        with torch.cuda.device(index):
            torch.cuda.empty_cache()



@torch.no_grad()
def predict_llm(
    input_root, output_dir, model_kind, num_workers,
    max_length=4096, batch_size=80, max_tokens=8192,
):
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
        max_length=max_length,
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
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        collate_fn=ShardedMaxTokensCollator(
            max_tokens=max_tokens,
            base_collator=VarlenCollator(),
        ),
    )

    # Place the first and second halves of the model on separate GPUs.
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

    # Process the final micro-batch.
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

    LOGGER.info("Saved %s: %s", output_file, probabilities.shape)



@torch.no_grad()
def predict_custom_model(df, input_root, output_dir, batch_size=8, max_length=512):
    model_dir = (
        input_root
        / "akemiiiiii"
        / "custom_model_dir"
    )

    # The original file contains a complete model object.
    # Use weights_only=False only with checkpoints from trusted sources.
    model = load_custom_checkpoint(model_dir / "custom_model_complete.pth")

    if not isinstance(model, nn.Module):
        raise TypeError(
            "Expected the notebook's full nn.Module checkpoint."
        )

    device = torch.device("cuda:0")
    model.to(device).eval()

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    predictions = []

    for start in tqdm(
        range(0, len(df), batch_size),
        desc="Custom model inference",
    ):
        batch = df.iloc[start:start + batch_size]

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
            max_length=max_length,
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

    LOGGER.info("Saved prob_faiss.npy: %s", probabilities.shape)

