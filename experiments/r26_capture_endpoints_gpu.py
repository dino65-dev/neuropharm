"""Capture the fully crossed R2.6 activation endpoint tensor on CUDA.

Output tensor shape:
    endpoints[claim, question, response_frame, certainty, activation]
    = [72, 4, 8, 2, d_model]

Certainty index 0 is low/uncertain and index 1 is high/certain. Activations are
the block-input residual stream at the final response token.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from experiments.r26_design import (
    MODEL_NAME,
    QUESTION_TEMPLATES,
    RESPONSE_FRAMES,
    balanced_claim_split,
    claims,
)


def residual_layer(model, layer: int):
    return model.model.layers[layer]


def input_device(model) -> torch.device:
    return model.model.embed_tokens.weight.device


def chat_prompt_ids(tokenizer, prompt: str) -> torch.Tensor:
    encoded = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        return_tensors="pt",
    )
    if isinstance(encoded, torch.Tensor):
        return encoded
    if hasattr(encoded, "input_ids"):
        return encoded.input_ids
    return encoded["input_ids"]


def response_ids(tokenizer, response: str) -> torch.Tensor:
    return tokenizer(
        response, add_special_tokens=False, return_tensors="pt"
    ).input_ids


def build_examples(tokenizer) -> list[dict[str, Any]]:
    examples = []
    for claim in claims():
        for question_index, template in enumerate(QUESTION_TEMPLATES):
            prompt = template.format(claim=claim["claim"])
            prompt_tokens = chat_prompt_ids(tokenizer, prompt)
            for frame in RESPONSE_FRAMES:
                for certainty_index, response in enumerate((frame.low, frame.high)):
                    answer_tokens = response_ids(tokenizer, response)
                    token_ids = torch.cat((prompt_tokens, answer_tokens), dim=1)
                    examples.append({
                        "claim_id": int(claim["claim_id"]),
                        "question_index": question_index,
                        "frame_id": frame.frame_id,
                        "certainty_index": certainty_index,
                        "token_ids": token_ids.squeeze(0),
                        "sequence_length": int(token_ids.shape[1]),
                    })
    return examples


def load_model(args):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for endpoint capture")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float16,
        device_map="auto",
        max_memory={0: args.gpu_memory, "cpu": args.cpu_memory},
        offload_folder=str(args.offload_folder),
    )
    model.eval()
    if residual_layer(model, args.layer).input_layernorm.weight.device.type != "cuda":
        raise RuntimeError(f"layer {args.layer} is not on CUDA")
    return model


def capture_endpoints(model, tokenizer, examples, layer: int, batch_size: int):
    tokenizer.pad_token = tokenizer.eos_token
    d_model = int(model.config.hidden_size)
    endpoints = torch.empty(
        (
            len(claims()),
            len(QUESTION_TEMPLATES),
            len(RESPONSE_FRAMES),
            2,
            d_model,
        ),
        dtype=torch.float16,
        device="cpu",
    )
    device = input_device(model)
    for start in range(0, len(examples), batch_size):
        batch = examples[start : start + batch_size]
        max_length = max(row["sequence_length"] for row in batch)
        input_ids = torch.full(
            (len(batch), max_length),
            tokenizer.pad_token_id,
            dtype=torch.long,
        )
        attention_mask = torch.zeros_like(input_ids)
        for index, row in enumerate(batch):
            length = row["sequence_length"]
            input_ids[index, :length] = row["token_ids"]
            attention_mask[index, :length] = 1
        captured: dict[str, torch.Tensor] = {}

        def hook(module, args):
            captured["residual"] = args[0].detach()

        handle = residual_layer(model, layer).register_forward_pre_hook(hook)
        try:
            with torch.inference_mode():
                # Skip lm_head: endpoint capture only needs transformer states.
                model.model(
                    input_ids=input_ids.to(device),
                    attention_mask=attention_mask.to(device),
                    use_cache=False,
                )
        finally:
            handle.remove()
        residual = captured["residual"]
        for local_index, row in enumerate(batch):
            endpoint = residual[
                local_index, row["sequence_length"] - 1, :
            ].to(device="cpu", dtype=torch.float16)
            endpoints[
                row["claim_id"],
                row["question_index"],
                row["frame_id"],
                row["certainty_index"],
            ] = endpoint
        if start == 0 or (start // batch_size + 1) % 8 == 0:
            print(f"captured {min(start + len(batch), len(examples))}/{len(examples)}")
    return endpoints


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260621)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--gpu-memory", default="14500MiB")
    parser.add_argument("--cpu-memory", default="12GiB")
    parser.add_argument(
        "--offload-folder",
        type=Path,
        default=Path("artifacts/_cache/r26_capture_offload"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/pkpd/r26_crossed_endpoints_t4.pt"),
    )
    args = parser.parse_args()

    start = time.time()
    torch.manual_seed(args.seed)
    torch.cuda.reset_peak_memory_stats()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    examples = build_examples(tokenizer)
    if len(examples) != 4608:
        raise RuntimeError(f"expected 4608 endpoints, got {len(examples)}")
    model = load_model(args)
    endpoints = capture_endpoints(
        model, tokenizer, examples, args.layer, args.batch_size
    )
    payload = {
        "experiment": "R2.6_fully_crossed_endpoint_capture",
        "model": MODEL_NAME,
        "layer": args.layer,
        "site": "block_input_residual_pre",
        "shape": list(endpoints.shape),
        "axis_order": [
            "claim", "question_template", "response_frame",
            "certainty_low_high", "activation",
        ],
        "certainty_index": {"low": 0, "high": 1},
        "claims": claims(),
        "questions": list(QUESTION_TEMPLATES),
        "response_frames": [
            {
                "frame_id": frame.frame_id,
                "family": frame.family,
                "position": frame.position,
                "high": frame.high,
                "low": frame.low,
            }
            for frame in RESPONSE_FRAMES
        ],
        "claim_split": balanced_claim_split(args.seed),
        "endpoints": endpoints,
        "cuda": {
            "device": torch.cuda.get_device_name(0),
            "capability": list(torch.cuda.get_device_capability(0)),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        },
        "elapsed_seconds": time.time() - start,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    print(args.output)
    print(f"shape={tuple(endpoints.shape)} elapsed={payload['elapsed_seconds']:.2f}s")


if __name__ == "__main__":
    main()
