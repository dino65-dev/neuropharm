"""R3A: model-local fixed-receptor replication on fresh crossed data."""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from experiments.r2_geometry_gpu import build_baseline_cache, input_device
from experiments.r26_causal_and_dose_gpu import factual_metrics, paired_examples
from experiments.r28_gain_controller_gpu import evaluate_variable_scales
from experiments.r3a_design import (
    QUESTION_TEMPLATES, RESPONSE_FRAMES, balanced_split, claims,
)


ALPHAS = (0.0, 0.03125, 0.0625, 0.125, 0.25, 0.5)
LANGUAGE_PAIRS = (
    ("The assessment can be expressed without qualification.", "The assessment requires substantial qualification."),
    ("There is little room for doubt about the assessment.", "There is considerable room for doubt about the assessment."),
    ("The assessment is firm enough to act upon.", "The assessment remains too provisional to act upon."),
    ("The available basis makes the assessment decisive.", "The available basis leaves the assessment unsettled."),
)
DEFINITE_PAIR = (
    "I can provide a definite answer to this question.",
    "I should withhold a definite answer to this question.",
)
NUMERIC_PAIR = ("Confidence: 90%.", "Confidence: 50%.")
TRUE_FALSE_PAIR = ("True.", "False.")


def decoder_layers(model):
    candidates = (
        lambda value: value.model.layers,
        lambda value: value.model.model.layers,
    )
    for getter in candidates:
        try:
            return getter(model)
        except AttributeError:
            continue
    raise RuntimeError("unable to locate decoder layers")


def chat_prompt_ids(tokenizer, prompt: str):
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


def response_ids(tokenizer, response: str):
    return tokenizer(response, add_special_tokens=False, return_tensors="pt").input_ids


def load_model(model_name: str, args):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16,
        device_map="auto",
        max_memory={0: args.gpu_memory, "cpu": args.cpu_memory},
        offload_folder=str(args.offload_folder),
    )
    model.eval()
    return model


def build_capture_examples(tokenizer, construction_ids):
    rows = {row["claim_id"]: row for row in claims()}
    examples = []
    for claim_id in construction_ids:
        claim = rows[claim_id]
        for question_index, template in enumerate(QUESTION_TEMPLATES):
            prompt = template.format(claim=claim["claim"])
            prompt_tokens = chat_prompt_ids(tokenizer, prompt)
            for frame in RESPONSE_FRAMES:
                for certainty, response in enumerate((frame.low, frame.high)):
                    answer = response_ids(tokenizer, response)
                    token_ids = torch.cat((prompt_tokens, answer), dim=1).squeeze(0)
                    examples.append({
                        "claim_id": claim_id,
                        "question_index": question_index,
                        "frame_id": frame.frame_id,
                        "certainty": certainty,
                        "token_ids": token_ids,
                        "length": int(token_ids.shape[0]),
                    })
    return examples


def capture_endpoints(model, tokenizer, examples, layer_index: int, batch_size: int):
    tokenizer.pad_token = tokenizer.eos_token
    dimension = int(model.config.hidden_size)
    endpoints = torch.empty(27, 4, 8, 2, dimension, dtype=torch.float16)
    layer = decoder_layers(model)[layer_index]
    for start in range(0, len(examples), batch_size):
        batch = examples[start:start + batch_size]
        max_length = max(row["length"] for row in batch)
        input_ids = torch.full(
            (len(batch), max_length), tokenizer.pad_token_id, dtype=torch.long
        )
        attention_mask = torch.zeros_like(input_ids)
        for index, row in enumerate(batch):
            input_ids[index, :row["length"]] = row["token_ids"]
            attention_mask[index, :row["length"]] = 1
        captured = {}

        def hook(module, hook_args):
            captured["residual"] = hook_args[0].detach()

        handle = layer.register_forward_pre_hook(hook)
        try:
            with torch.inference_mode():
                model.model(
                    input_ids=input_ids.to(input_device(model)),
                    attention_mask=attention_mask.to(input_device(model)),
                    use_cache=False,
                )
        finally:
            handle.remove()
        residual = captured["residual"]
        for local_index, row in enumerate(batch):
            construction_index = row["claim_id_index"]
            endpoints[
                construction_index, row["question_index"], row["frame_id"],
                row["certainty"],
            ] = residual[
                local_index, row["length"] - 1
            ].to("cpu", torch.float16)
    return endpoints


def add_construction_indices(examples, construction_ids):
    mapping = {claim_id: index for index, claim_id in enumerate(construction_ids)}
    for row in examples:
        row["claim_id_index"] = mapping[row["claim_id"]]


def endpoint_asset(model, tokenizer, rows, layer, batch_size):
    examples, metadata = paired_examples(tokenizer, rows)
    baseline, logits = build_baseline_cache(
        model, tokenizer, examples, layer,
        [1.0] * max(row["answer_length"] for row in examples), batch_size,
    )
    return {
        "examples": examples, "metadata": metadata,
        "baseline": baseline, "logits": logits,
    }


def evaluation_assets(model, tokenizer, test_ids, layer, batch_size):
    by_id = {row["claim_id"]: row for row in claims()}
    rows = {name: [] for name in ("language", "definite", "numeric", "factual")}
    factual_labels = []
    for claim_id in test_ids:
        claim = by_id[claim_id]
        prompt = f"Evaluate this statement carefully: {claim['claim']}."
        for pair_index, pair in enumerate(LANGUAGE_PAIRS):
            rows["language"].append((
                prompt, pair[0], pair[1],
                {"claim_id": claim_id, "pair_index": pair_index},
            ))
        for statement, label in ((claim["claim"], 1), (claim["false_claim"], 0)):
            metadata = {"claim_id": claim_id, "label": label, "domain": claim["domain"]}
            rows["definite"].append((
                f"Can this statement be judged definitively? {statement}",
                DEFINITE_PAIR[0], DEFINITE_PAIR[1], metadata,
            ))
            rows["numeric"].append((
                f"Judge this statement and state confidence: {statement}",
                NUMERIC_PAIR[0], NUMERIC_PAIR[1], metadata,
            ))
            rows["factual"].append((
                f"Is this statement true or false? {statement}",
                TRUE_FALSE_PAIR[0], TRUE_FALSE_PAIR[1], metadata,
            ))
            factual_labels.append(label)
    return {
        name: endpoint_asset(model, tokenizer, endpoint_rows, layer, batch_size)
        for name, endpoint_rows in rows.items()
    }, factual_labels


def mean_delta(result, baseline):
    return statistics.fmean(
        value - reference for value, reference in zip(result["scores"], baseline.scores)
    )


def claim_bootstrap(values, claim_ids, seed, replicates=2000):
    grouped = {}
    for value, claim_id in zip(values, claim_ids):
        grouped.setdefault(int(claim_id), []).append(float(value))
    means = {key: statistics.fmean(rows) for key, rows in grouped.items()}
    ids = sorted(means)
    generator = random.Random(seed)
    samples = []
    for _ in range(replicates):
        selected = [ids[generator.randrange(len(ids))] for _ in ids]
        samples.append(statistics.fmean(means[index] for index in selected))
    ordered = sorted(samples)
    return {
        "claims": len(ids),
        "mean": statistics.fmean(means.values()),
        "lower_95": ordered[int(0.025 * replicates)],
        "upper_95": ordered[min(replicates - 1, math.ceil(0.975 * replicates) - 1)],
        "positive_claim_fraction": statistics.fmean(value > 0 for value in means.values()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--seed", type=int, default=20260702)
    parser.add_argument("--layer-fraction", type=float, default=0.46)
    parser.add_argument("--capture-batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--gpu-memory", default="14500MiB")
    parser.add_argument("--cpu-memory", default="24GiB")
    parser.add_argument("--offload-folder", type=Path, default=Path("artifacts/_cache/r3a_offload"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receptor-output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    started = time.time()
    torch.cuda.reset_peak_memory_stats()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = load_model(args.model, args)
    layers = decoder_layers(model)
    layer_index = round(args.layer_fraction * (len(layers) - 1))
    if layers[layer_index].input_layernorm.weight.device.type != "cuda":
        raise RuntimeError("intervention layer is not on CUDA")
    split = balanced_split(args.seed)
    examples = build_capture_examples(tokenizer, split["construction"])
    add_construction_indices(examples, split["construction"])
    endpoints = capture_endpoints(
        model, tokenizer, examples, layer_index, args.capture_batch_size
    )
    contrasts = endpoints[..., 1, :].float() - endpoints[..., 0, :].float()
    receptor_raw = contrasts.mean(dim=(0, 1, 2))
    receptor = receptor_raw / receptor_raw.norm().clamp_min(1e-12)
    receptor_device = receptor.to(input_device(model))
    assets, factual_labels = evaluation_assets(
        model, tokenizer, split["test"], layer_index, args.eval_batch_size
    )
    baseline_factual = factual_metrics(assets["factual"]["baseline"].scores, factual_labels)
    baseline_predictions = [int(score >= 0) for score in assets["factual"]["baseline"].scores]
    baseline_correct = [prediction == label for prediction, label in zip(baseline_predictions, factual_labels)]
    rows = []
    for alpha in ALPHAS:
        results = {
            name: evaluate_variable_scales(
                model, tokenizer, asset, receptor_device,
                [alpha] * len(asset["metadata"]), layer_index,
                args.eval_batch_size,
            )
            for name, asset in assets.items()
        }
        factual = factual_metrics(results["factual"]["scores"], factual_labels)
        numeric_deltas = [
            value - reference for value, reference in zip(
                results["numeric"]["scores"], assets["numeric"]["baseline"].scores
            )
        ]
        correct_margins = [
            (value - reference) * (1 if label else -1)
            for value, reference, label in zip(
                results["factual"]["scores"], assets["factual"]["baseline"].scores,
                factual_labels,
            )
        ]
        endpoint_deltas = {
            name: [
                value - reference for value, reference in zip(
                    result["scores"], assets[name]["baseline"].scores
                )
            ]
            for name, result in results.items()
        }
        claim_ids = {
            name: [row["claim_id"] for row in asset["metadata"]]
            for name, asset in assets.items()
        }
        rows.append({
            "alpha": alpha,
            "language_delta": mean_delta(results["language"], assets["language"]["baseline"]),
            "definite_delta": mean_delta(results["definite"], assets["definite"]["baseline"]),
            "numeric_delta": mean_delta(results["numeric"], assets["numeric"]["baseline"]),
            "numeric_delta_baseline_incorrect": statistics.fmean([
                value for value, correct in zip(numeric_deltas, baseline_correct) if not correct
            ]) if not all(baseline_correct) else None,
            "factual_margin_delta": statistics.fmean(correct_margins),
            "accuracy": factual["accuracy"],
            "accuracy_delta": factual["accuracy"] - baseline_factual["accuracy"],
            "brier_delta": factual["brier"] - baseline_factual["brier"],
            "nll_delta": factual["negative_log_likelihood"] - baseline_factual["negative_log_likelihood"],
            "mean_kl": statistics.fmean(
                value for result in results.values() for value in result["kl"]
            ),
            "claim_block_bootstrap": {
                "language": claim_bootstrap(
                    endpoint_deltas["language"], claim_ids["language"],
                    args.seed + int(alpha * 10000) + 1,
                ),
                "definite": claim_bootstrap(
                    endpoint_deltas["definite"], claim_ids["definite"],
                    args.seed + int(alpha * 10000) + 2,
                ),
                "numeric": claim_bootstrap(
                    endpoint_deltas["numeric"], claim_ids["numeric"],
                    args.seed + int(alpha * 10000) + 3,
                ),
                "factual_margin": claim_bootstrap(
                    correct_margins, claim_ids["factual"],
                    args.seed + int(alpha * 10000) + 4,
                ),
            },
        })
        print(f"{args.model} alpha={alpha:g}")
    args.receptor_output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": args.model,
        "layer": layer_index,
        "layer_fraction": layer_index / (len(layers) - 1),
        "claim_split": split,
        "endpoints": endpoints,
        "mean_receptor_raw": receptor_raw,
        "mean_receptor_unit": receptor,
    }, args.receptor_output)
    payload = {
        "experiment": "R3A_cross_model_fixed_receptor_replication",
        "model": args.model,
        "layers": len(layers),
        "hidden_size": int(model.config.hidden_size),
        "intervention_layer": layer_index,
        "intervention_layer_fraction": layer_index / (len(layers) - 1),
        "site": "block_input_residual_pre",
        "fresh_claim_split": split,
        "construction_tensor_shape": list(endpoints.shape),
        "receptor_raw_norm": float(receptor_raw.norm().item()),
        "baseline_factual": baseline_factual,
        "baseline_incorrect_items": sum(not value for value in baseline_correct),
        "rows": rows,
        "cuda": {
            "device": torch.cuda.get_device_name(0),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        },
        "elapsed_seconds": time.time() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
