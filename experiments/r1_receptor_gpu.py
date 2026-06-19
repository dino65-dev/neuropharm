"""R1: chat-matched receptor identification and causal alignment.

This first R1 stage tests one raw mean-difference receptor. It fixes the
previous comparison by using:

- chat-template matched receptor construction;
- teacher-forced sequence log-odds;
- identical tokenwise PK/PD effect schedules for receptor, opposite, and
  random directions;
- a null distribution over many random directions.

It intentionally does not enable tolerance or homeostasis.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from administration.pkpd_controller import protocol_schedule
from pharmacokinetics.state_space import NeuropharmacologyModel, PKPDParameters


MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
LAYER = 12

CLAIMS = [
    "regular exercise is generally beneficial",
    "maintaining an emergency fund is prudent",
    "Paris is the capital of France",
    "adequate hydration supports health",
    "Earth's axial tilt causes seasons",
    "automated tests improve software reliability",
    "twelve multiplied by thirteen equals 156",
    "sleep supports memory consolidation",
    "database indexes can accelerate queries",
    "passwords should be unique across websites",
    "the Moon can block the Sun during a solar eclipse",
    "primary sources help evaluate claims",
    "RAM and disk storage serve different purposes",
    "scientific findings benefit from independent replication",
    "water boils near 100 degrees Celsius at sea level",
    "diversification can reduce concentration risk",
    "backups reduce data-loss risk",
    "uncertain estimates should include uncertainty intervals",
    "a compiler translates source code",
    "sunscreen can reduce ultraviolet exposure",
    "control groups support causal inference",
    "production systems should monitor failures",
    "the square root of 144 is 12",
    "research claims should cite relevant sources",
    "correlation alone does not establish causation",
]

QUESTION_TEMPLATES = [
    "Assess whether {claim}.",
    "What evidence supports the statement that {claim}?",
    "Give a concise judgment on this claim: {claim}.",
    "Evaluate the following statement: {claim}.",
]

CONSTRUCTION_RESPONSE_PAIRS = [
    (
        "The available evidence strongly supports this conclusion.",
        "The available evidence weakly supports this conclusion.",
    ),
    (
        "This conclusion is well established by the evidence.",
        "This conclusion is only tentatively suggested by the evidence.",
    ),
    (
        "I am confident that this conclusion is correct.",
        "I am uncertain whether this conclusion is correct.",
    ),
    (
        "The evidence provides a clear basis for this conclusion.",
        "The evidence provides a limited basis for this conclusion.",
    ),
]

EVAL_POSITIVE = "Overall, this conclusion is well supported by the available evidence."
EVAL_NEGATIVE = "Overall, this conclusion is only tentatively supported by the available evidence."


def all_prompts() -> list[str]:
    return [
        template.format(claim=claim)
        for claim in CLAIMS
        for template in QUESTION_TEMPLATES
    ]


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
        response,
        add_special_tokens=False,
        return_tensors="pt",
    ).input_ids


def full_chat_ids(tokenizer, prompt: str, response: str) -> tuple[torch.Tensor, int]:
    prompt_tokens = chat_prompt_ids(tokenizer, prompt)
    answer_tokens = response_ids(tokenizer, response)
    return torch.cat([prompt_tokens, answer_tokens], dim=1), int(prompt_tokens.shape[1])


def capture_response_endpoint(
    model,
    tokenizer,
    prompt: str,
    response: str,
    layer: int,
) -> torch.Tensor:
    token_ids, _ = full_chat_ids(tokenizer, prompt, response)
    captured: dict[str, torch.Tensor] = {}

    def hook(module, args):
        captured["resid"] = args[0].detach()

    handle = residual_layer(model, layer).register_forward_pre_hook(hook)
    try:
        with torch.no_grad():
            model(
                input_ids=token_ids.to(input_device(model)),
                use_cache=False,
            )
    finally:
        handle.remove()
    return captured["resid"][0, -1, :].to(torch.float32)


def build_chat_matched_receptor(
    model,
    tokenizer,
    prompts: list[str],
    layer: int,
) -> tuple[torch.Tensor, list[float]]:
    differences = []
    norms = []
    for index, prompt in enumerate(prompts):
        positive, negative = CONSTRUCTION_RESPONSE_PAIRS[
            index % len(CONSTRUCTION_RESPONSE_PAIRS)
        ]
        difference = (
            capture_response_endpoint(model, tokenizer, prompt, positive, layer)
            - capture_response_endpoint(model, tokenizer, prompt, negative, layer)
        )
        differences.append(difference)
        norms.append(float(difference.norm().item()))
    receptor = torch.stack(differences).mean(dim=0)
    receptor = receptor / receptor.norm().clamp_min(1e-12)
    return receptor, norms


def make_unit_pkpd(direction: torch.Tensor) -> NeuropharmacologyModel:
    device = direction.device
    dtype = torch.float32
    return NeuropharmacologyModel(PKPDParameters(
        receptor_basis=direction.to(dtype)[:, None],
        affinity=torch.ones(1, 1, device=device, dtype=dtype),
        ec50=torch.tensor([0.5], device=device, dtype=dtype),
        hill=torch.tensor([2.0], device=device, dtype=dtype),
        emax=torch.ones(1, device=device, dtype=dtype),
        retention=torch.tensor(
            [2.0 ** (-1.0 / 8.0)], device=device, dtype=dtype
        ),
        absorption=torch.ones(1, device=device, dtype=dtype),
        recovery=torch.zeros(1, device=device, dtype=dtype),
        desensitization=torch.zeros(1, device=device, dtype=dtype),
        homeostasis_decay=0.95,
        homeostasis_gain=0.0,
    ))


def effect_schedule(direction: torch.Tensor, steps: int, dose: float) -> list[float]:
    model = make_unit_pkpd(direction)
    state = model.initial_state()
    schedule = protocol_schedule("pulses", dose=dose, stop=steps)
    effects = []
    for step_index in range(steps):
        step = model.step(
            state,
            schedule(step_index).to(
                device=state.concentration.device,
                dtype=state.concentration.dtype,
            ),
        )
        state = step.state
        effects.append(float(step.receptor_effect[0].item()))
    return effects


@contextmanager
def temporal_intervention(
    model,
    layer: int,
    direction: torch.Tensor | None,
    effects: list[float],
    prompt_length: int,
):
    if direction is None:
        yield
        return
    direction = direction.to(torch.float32)

    def hook(module, args):
        resid = args[0]
        changed = resid.clone()
        for response_index, effect in enumerate(effects):
            position = prompt_length - 1 + response_index
            if position >= resid.shape[1]:
                break
            delta = direction.to(resid.device, resid.dtype) * float(effect)
            changed[:, position, :] = changed[:, position, :] + delta
        return (changed, *args[1:])

    handle = residual_layer(model, layer).register_forward_pre_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def mean_response_logprob(
    model,
    tokenizer,
    prompt: str,
    response: str,
    layer: int,
    direction: torch.Tensor | None,
    effects: list[float],
) -> float:
    token_ids, prompt_length = full_chat_ids(tokenizer, prompt, response)
    answer = token_ids[:, prompt_length:]
    with temporal_intervention(
        model, layer, direction, effects[: answer.shape[1]], prompt_length
    ):
        with torch.no_grad():
            logits = model(
                input_ids=token_ids.to(input_device(model)),
                use_cache=False,
            ).logits
    prediction_logits = logits[:, prompt_length - 1 : -1, :].float()
    log_probs = torch.log_softmax(prediction_logits, dim=-1)
    token_log_probs = log_probs.gather(
        dim=-1,
        index=answer.to(log_probs.device).unsqueeze(-1),
    ).squeeze(-1)
    return float(token_log_probs.mean().item())


def sequence_log_odds(
    model,
    tokenizer,
    prompt: str,
    layer: int,
    direction: torch.Tensor | None,
    effects: list[float],
) -> float:
    positive = mean_response_logprob(
        model, tokenizer, prompt, EVAL_POSITIVE, layer, direction, effects
    )
    negative = mean_response_logprob(
        model, tokenizer, prompt, EVAL_NEGATIVE, layer, direction, effects
    )
    return positive - negative


def random_orthogonal_direction(
    receptor: torch.Tensor,
    generator: torch.Generator,
) -> torch.Tensor:
    candidate = torch.randn(
        receptor.shape,
        generator=generator,
        device="cpu",
        dtype=torch.float32,
    ).to(receptor.device)
    candidate = candidate - torch.dot(candidate, receptor) * receptor
    return candidate / candidate.norm().clamp_min(1e-12)


def quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil(probability * len(ordered)) - 1)
    return ordered[index]


def evaluate_direction(
    model,
    tokenizer,
    prompts: list[str],
    baseline_scores: list[float],
    direction: torch.Tensor,
    effects: list[float],
    layer: int,
) -> tuple[float, list[float]]:
    deltas = []
    for prompt, baseline in zip(prompts, baseline_scores):
        score = sequence_log_odds(
            model, tokenizer, prompt, layer, direction, effects
        )
        deltas.append(score - baseline)
    return statistics.fmean(deltas), deltas


def load_model(args):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this experiment")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float16,
        device_map="auto",
        max_memory={0: args.gpu_memory, "cpu": args.cpu_memory},
        offload_folder=str(args.offload_folder),
    )
    model.eval()
    if residual_layer(model, args.layer).input_layernorm.weight.device.type != "cuda":
        raise RuntimeError(f"intervention layer {args.layer} was not placed on CUDA")
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, default=LAYER)
    parser.add_argument("--dose", type=float, default=1.0)
    parser.add_argument("--construction-prompts", type=int, default=50)
    parser.add_argument("--validation-prompts", type=int, default=20)
    parser.add_argument("--test-prompts", type=int, default=30)
    parser.add_argument("--random-directions", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu-memory", default="2400MiB")
    parser.add_argument("--cpu-memory", default="12GiB")
    parser.add_argument(
        "--offload-folder",
        type=Path,
        default=Path("artifacts/_cache/r1_offload"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/pkpd/r1_receptor_gpu.json"),
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    prompts = all_prompts()
    expected = (
        args.construction_prompts + args.validation_prompts + args.test_prompts
    )
    if expected > len(prompts):
        raise ValueError(f"requested {expected} prompts, only {len(prompts)} exist")
    construction = prompts[: args.construction_prompts]
    validation = prompts[
        args.construction_prompts :
        args.construction_prompts + args.validation_prompts
    ]
    test = prompts[
        args.construction_prompts + args.validation_prompts : expected
    ]

    start = time.time()
    torch.cuda.reset_peak_memory_stats()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = load_model(args)
    receptor, contrast_norms = build_chat_matched_receptor(
        model, tokenizer, construction, args.layer
    )
    positive_length = response_ids(tokenizer, EVAL_POSITIVE).shape[1]
    negative_length = response_ids(tokenizer, EVAL_NEGATIVE).shape[1]
    effects = effect_schedule(
        receptor,
        steps=max(positive_length, negative_length),
        dose=args.dose,
    )

    validation_baseline = [
        sequence_log_odds(
            model, tokenizer, prompt, args.layer, None, effects
        )
        for prompt in validation
    ]
    validation_effect, validation_deltas = evaluate_direction(
        model,
        tokenizer,
        validation,
        validation_baseline,
        receptor,
        effects,
        args.layer,
    )
    validation_opposite, validation_opposite_deltas = evaluate_direction(
        model,
        tokenizer,
        validation,
        validation_baseline,
        -receptor,
        effects,
        args.layer,
    )

    test_baseline = [
        sequence_log_odds(
            model, tokenizer, prompt, args.layer, None, effects
        )
        for prompt in test
    ]
    receptor_effect, receptor_deltas = evaluate_direction(
        model,
        tokenizer,
        test,
        test_baseline,
        receptor,
        effects,
        args.layer,
    )
    opposite_effect, opposite_deltas = evaluate_direction(
        model,
        tokenizer,
        test,
        test_baseline,
        -receptor,
        effects,
        args.layer,
    )

    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    random_effects = []
    random_cosines = []
    for random_index in range(args.random_directions):
        direction = random_orthogonal_direction(receptor, generator)
        effect, _ = evaluate_direction(
            model,
            tokenizer,
            test,
            test_baseline,
            direction,
            effects,
            args.layer,
        )
        random_effects.append(effect)
        random_cosines.append(float(torch.dot(receptor, direction).item()))
        print(
            f"random {random_index + 1}/{args.random_directions}: "
            f"mean_delta={effect:+.6f}"
        )

    random_q95 = quantile(random_effects, 0.95)
    percentile = (
        1 + sum(effect <= receptor_effect for effect in random_effects)
    ) / (len(random_effects) + 1)
    symmetry = [
        (positive - negative) / 2
        for positive, negative in zip(receptor_deltas, opposite_deltas)
    ]
    payload: dict[str, Any] = {
        "experiment": "R1_receptor_identification_and_causal_alignment_stage1",
        "model": MODEL_NAME,
        "layer": args.layer,
        "site": "block_input_residual_pre",
        "device_map": model.hf_device_map,
        "cuda_device": torch.cuda.get_device_name(0),
        "cuda_capability": list(torch.cuda.get_device_capability(0)),
        "torch_version": torch.__version__,
        "cuda_build": torch.version.cuda,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(),
        "seed": args.seed,
        "splits": {
            "construction": len(construction),
            "validation": len(validation),
            "test": len(test),
        },
        "teacher_forcing": True,
        "score": "length_normalized_sequence_log_odds",
        "positive_response": EVAL_POSITIVE,
        "negative_response": EVAL_NEGATIVE,
        "positive_tokens": int(positive_length),
        "negative_tokens": int(negative_length),
        "pkpd_effect_schedule": effects,
        "receptor_norm": float(receptor.norm().item()),
        "contrast_difference_norm_mean": statistics.fmean(contrast_norms),
        "contrast_difference_norm_std": statistics.stdev(contrast_norms),
        "validation": {
            "receptor_mean_delta": validation_effect,
            "opposite_mean_delta": validation_opposite,
            "receptor_prompt_deltas": validation_deltas,
            "opposite_prompt_deltas": validation_opposite_deltas,
        },
        "test": {
            "receptor_mean_delta": receptor_effect,
            "opposite_mean_delta": opposite_effect,
            "symmetry_mean": statistics.fmean(symmetry),
            "receptor_prompt_deltas": receptor_deltas,
            "opposite_prompt_deltas": opposite_deltas,
            "random_effects": random_effects,
            "random_cosine_max_abs": max(abs(value) for value in random_cosines),
            "random_q95": random_q95,
            "receptor_percentile": percentile,
        },
        "gates": {
            "sign": receptor_effect > 0 and opposite_effect < 0,
            "random_null": receptor_effect > random_q95,
            "receptor_positive": receptor_effect > 0,
        },
        "elapsed_seconds": time.time() - start,
    }
    payload["overall_passed"] = all(payload["gates"].values())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()

