"""First causal PK/PD pilot on Qwen-2.5-1.5B-Instruct.

The experiment is intentionally restricted to one normalized confidence
receptor at layer 12 and batch size one. PK/PD state advances once per model
forward call: one prompt-prefill step, then one step per cached decode call.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from administration.pkpd_controller import (
    PKPDGenerationController,
    protocol_schedule,
)
from pharmacokinetics.state_space import NeuropharmacologyModel, PKPDParameters


MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
LAYER = 12

CONFIDENT_PAIRS = [
    ("I am absolutely certain this is correct.", "I am not sure this is correct."),
    ("My answer is definitive: yes.", "My answer might be wrong, but I think maybe."),
    ("I state this with full conviction.", "I state this with zero conviction."),
    ("I know exactly what I am talking about.", "I have no idea what I am talking about."),
    ("This is settled; the answer is 42.", "I'm not sure; it could be 42, or maybe not."),
    ("I guarantee that this approach works.", "I cannot promise this approach works."),
    ("There is no doubt in my mind.", "There is substantial doubt in my mind."),
    ("My expertise is directly relevant.", "My expertise may not be relevant."),
    ("The evidence is conclusive.", "The evidence is inconclusive."),
    ("I assert this without hesitation.", "I am hesitant to raise this claim."),
    ("You can rely on this answer.", "You may not want to rely on this answer."),
    ("I am fully confident in this prediction.", "I am only somewhat confident."),
    ("This is plainly true.", "This might be true, or might not be."),
    ("I speak with authority on this topic.", "I lack authority on this topic."),
    ("This is how it works.", "I am not sure how it works."),
    ("The result is unambiguous.", "The result is ambiguous."),
    ("I commit to this answer.", "I am not willing to commit."),
    ("I have no reservations.", "I have many reservations."),
    ("Decisively, the answer is yes.", "Tentatively, the answer is maybe."),
    ("The conclusion is certain.", "The conclusion remains uncertain."),
]

HELD_OUT_PROMPTS = [
    "Is regular exercise generally beneficial?",
    "Should people maintain an emergency fund?",
    "What is the capital of France?",
    "Is drinking enough water important?",
    "Explain why seasons occur.",
    "Should software projects include automated tests?",
    "What is 12 multiplied by 13?",
    "Is sleep important for memory?",
    "Describe the purpose of a database index.",
    "Should passwords be unique across websites?",
    "What causes a solar eclipse?",
    "Is reading source material useful before forming a conclusion?",
    "Explain the difference between RAM and disk storage.",
    "Should a scientific claim be independently replicated?",
    "What is the boiling point of water at sea level?",
    "Is diversification relevant to investment risk?",
    "Explain why backups are useful.",
    "Should uncertain estimates include error bars?",
    "What does a compiler do?",
    "Is sunscreen useful for reducing UV exposure?",
    "Explain the role of control groups in experiments.",
    "Should production systems monitor failures?",
    "What is the square root of 144?",
    "Is source citation important in research?",
    "Explain why correlation does not establish causation.",
    "Should API inputs be validated?",
    "What is photosynthesis?",
    "Is gradual rollout useful for risky software changes?",
    "Explain the purpose of randomization in trials.",
    "Should model evaluations use held-out data?",
]

CONFIDENT_PROBES = ["certainly", "definitely", "clearly", "will", "must"]
HEDGED_PROBES = ["maybe", "perhaps", "might", "could", "uncertain"]


class StaticController:
    def __init__(self, direction: torch.Tensor, coefficient: float):
        self.delta = direction * float(coefficient)
        self.step_index = 0
        self.trace: list[dict[str, Any]] = []

    def hook(self, resid: torch.Tensor) -> torch.Tensor:
        delta = self.delta.to(device=resid.device, dtype=resid.dtype)
        row = {
            "pkpd_step_index": self.step_index,
            "dose": None,
            "concentration": None,
            "occupancy": None,
            "sensitivity": None,
            "adaptation": None,
            "receptor_effect": [float(self.delta.norm().item())],
            "delta_h_norm": float(self.delta.norm().item()),
            "forward_sequence_length": int(resid.shape[1]),
            "is_prefill": bool(self.step_index == 0 and resid.shape[1] > 1),
            "injected_token_position": int(resid.shape[1] - 1),
        }
        self.trace.append(row)
        self.step_index += 1
        if torch.count_nonzero(delta).item() == 0:
            return resid
        result = resid.clone()
        result[:, -1, :] = result[:, -1, :] + delta
        return result

    def record_outcome(self, **values: Any) -> None:
        self.trace[-1].update(values)


def residual_layer(model, layer: int):
    return model.model.layers[layer]


def format_prompt(tokenizer, prompt: str) -> torch.Tensor:
    messages = [{"role": "user", "content": prompt}]
    encoded = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    if isinstance(encoded, torch.Tensor):
        return encoded
    if hasattr(encoded, "input_ids"):
        return encoded.input_ids
    if isinstance(encoded, dict) and "input_ids" in encoded:
        return encoded["input_ids"]
    raise TypeError(f"unsupported chat-template return type: {type(encoded)!r}")


def capture_last_residual(model, tokenizer, text: str, layer: int) -> torch.Tensor:
    captured: dict[str, torch.Tensor] = {}

    def hook(module, args):
        captured["resid"] = args[0].detach()

    handle = residual_layer(model, layer).register_forward_pre_hook(hook)
    try:
        inputs = tokenizer(text, return_tensors="pt", add_special_tokens=True)
        input_ids = inputs.input_ids.to(model.device)
        with torch.no_grad():
            model(input_ids=input_ids, use_cache=False)
    finally:
        handle.remove()
    return captured["resid"][0, -1, :].to(torch.float32)


def build_receptor(model, tokenizer, layer: int) -> torch.Tensor:
    differences = []
    for positive, negative in CONFIDENT_PAIRS:
        differences.append(
            capture_last_residual(model, tokenizer, positive, layer)
            - capture_last_residual(model, tokenizer, negative, layer)
        )
    direction = torch.stack(differences).mean(dim=0)
    return direction / direction.norm().clamp_min(1e-12)


def make_pkpd_model(direction: torch.Tensor) -> NeuropharmacologyModel:
    dtype = torch.float32
    device = direction.device
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


def probe_token_ids(tokenizer, terms: list[str]) -> list[int]:
    token_ids = set()
    for term in terms:
        for variant in (term, " " + term):
            ids = tokenizer.encode(variant, add_special_tokens=False)
            if ids:
                token_ids.add(int(ids[0]))
    return sorted(token_ids)


def target_probe_score(
    logits: torch.Tensor,
    positive_ids: list[int],
    negative_ids: list[int],
) -> float:
    positive = torch.logsumexp(logits[positive_ids].float(), dim=0)
    negative = torch.logsumexp(logits[negative_ids].float(), dim=0)
    return float((positive - negative).item())


def categorical_kl(logits: torch.Tensor, baseline_logits: torch.Tensor) -> float:
    log_p = torch.log_softmax(logits.float(), dim=-1)
    log_q = torch.log_softmax(baseline_logits.float(), dim=-1)
    p = log_p.exp()
    return float(torch.sum(p * (log_p - log_q)).item())


def register_controller(model, layer: int, controller):
    if controller is None:
        return nullcontext()

    class HandleContext:
        def __enter__(self):
            def hook(module, args):
                changed = controller.hook(args[0])
                return (changed, *args[1:])

            self.handle = residual_layer(model, layer).register_forward_pre_hook(hook)
            return controller

        def __exit__(self, exc_type, exc, traceback):
            self.handle.remove()

    return HandleContext()


def generate_cached(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    controller=None,
    baseline_logits: list[torch.Tensor] | None = None,
) -> dict[str, Any]:
    input_ids = format_prompt(tokenizer, prompt).to(model.device)
    active_ids = input_ids
    past_key_values = None
    generated: list[int] = []
    logits_rows: list[torch.Tensor] = []
    positive_ids = probe_token_ids(tokenizer, CONFIDENT_PROBES)
    negative_ids = probe_token_ids(tokenizer, HEDGED_PROBES)
    baseline_trace: list[dict[str, Any]] = []
    n_steps = (
        min(max_new_tokens, len(baseline_logits))
        if baseline_logits is not None
        else max_new_tokens
    )

    with register_controller(model, LAYER, controller):
        for token_index in range(n_steps):
            with torch.no_grad():
                outputs = model(
                    input_ids=active_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
            logits = outputs.logits[0, -1, :].detach()
            next_token_id = int(torch.argmax(logits).item())
            logits_rows.append(logits.to(torch.float32).cpu())
            generated.append(next_token_id)
            kl = (
                categorical_kl(logits, baseline_logits[token_index].to(logits.device))
                if baseline_logits is not None
                else 0.0
            )
            if controller is not None:
                controller.record_outcome(
                    next_token_id=next_token_id,
                    next_token=tokenizer.decode([next_token_id]),
                    target_probe_score=target_probe_score(
                        logits, positive_ids, negative_ids
                    ),
                    kl_from_baseline=kl,
                )
            else:
                baseline_trace.append({
                    "pkpd_step_index": token_index,
                    "dose": [0.0],
                    "concentration": [0.0],
                    "occupancy": [0.0],
                    "sensitivity": [1.0],
                    "adaptation": [0.0],
                    "receptor_effect": [0.0],
                    "delta_h_norm": 0.0,
                    "forward_sequence_length": int(active_ids.shape[1]),
                    "is_prefill": bool(token_index == 0 and active_ids.shape[1] > 1),
                    "injected_token_position": int(active_ids.shape[1] - 1),
                    "next_token_id": next_token_id,
                    "next_token": tokenizer.decode([next_token_id]),
                    "target_probe_score": target_probe_score(
                        logits, positive_ids, negative_ids
                    ),
                    "kl_from_baseline": 0.0,
                })
            past_key_values = outputs.past_key_values
            active_ids = torch.tensor(
                [[next_token_id]], device=model.device, dtype=input_ids.dtype
            )
    return {
        "text": tokenizer.decode(generated, skip_special_tokens=True),
        "token_ids": generated,
        "logits": logits_rows,
        "trace": baseline_trace if controller is None else controller.trace,
        "total_intervention": 0.0 if controller is None else sum(
            row["delta_h_norm"] for row in controller.trace
        ),
    }


def shuffled_schedule(values: list[float], seed: int):
    shuffled = list(values)
    random.Random(seed).shuffle(shuffled)

    def schedule(step_index: int) -> torch.Tensor:
        value = shuffled[step_index] if step_index < len(shuffled) else 0.0
        return torch.tensor([value], dtype=torch.float64)

    return schedule


def planned_matched_static_coefficient(
    direction: torch.Tensor,
    dose_schedule,
    steps: int,
) -> float:
    model = make_pkpd_model(direction)
    controller = PKPDGenerationController(model, dose_schedule)
    for _ in range(steps):
        controller.advance()
    return sum(row["delta_h_norm"] for row in controller.trace) / steps


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", type=int, default=30)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--dose", type=float, default=1.0)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument(
        "--receptor-cache",
        type=Path,
        default=Path("artifacts/pkpd/qwen_confidence_receptor.pt"),
    )
    parser.add_argument("--output", type=Path, default=Path("artifacts/pkpd/qwen_pilot.json"))
    args = parser.parse_args()
    torch.manual_seed(0)
    random.seed(0)
    device = (
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu"
        if args.device == "auto"
        else args.device
    )
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable in this PyTorch build")
    model_dtype = torch.float16 if device == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=model_dtype,
        device_map={"": device},
    )
    model.eval()
    if args.receptor_cache.exists():
        receptor = torch.load(
            args.receptor_cache, map_location=model.device, weights_only=True
        ).to(torch.float32)
        receptor = receptor / receptor.norm().clamp_min(1e-12)
    else:
        receptor = build_receptor(model, tokenizer, LAYER)
        args.receptor_cache.parent.mkdir(parents=True, exist_ok=True)
        torch.save(receptor.detach().cpu(), args.receptor_cache)
    generator = torch.Generator(device="cpu").manual_seed(0)
    random_direction = torch.randn(
        receptor.shape, generator=generator, dtype=torch.float32
    ).to(receptor.device)
    random_direction = random_direction - torch.dot(random_direction, receptor) * receptor
    random_direction = random_direction / random_direction.norm().clamp_min(1e-12)

    pulse_schedule = protocol_schedule("pulses", args.dose, stop=args.max_new_tokens)
    static_coefficient = planned_matched_static_coefficient(
        receptor, pulse_schedule, args.max_new_tokens
    )
    pulse_values = [
        float(pulse_schedule(index)[0]) for index in range(args.max_new_tokens)
    ]

    results = []
    max_zero_logit_error = 0.0
    for prompt_index, prompt in enumerate(HELD_OUT_PROMPTS[:args.prompts]):
        baseline = generate_cached(
            model, tokenizer, prompt, args.max_new_tokens
        )
        zero = generate_cached(
            model,
            tokenizer,
            prompt,
            args.max_new_tokens,
            PKPDGenerationController(
                make_pkpd_model(receptor),
                protocol_schedule("zero", args.dose),
            ),
            baseline["logits"],
        )
        for baseline_row, zero_row in zip(baseline["logits"], zero["logits"]):
            max_zero_logit_error = max(
                max_zero_logit_error,
                float((baseline_row - zero_row).abs().max().item()),
            )

        conditions = {
            "baseline": baseline,
            "zero_pkpd": zero,
            "static_matched": generate_cached(
                model, tokenizer, prompt, args.max_new_tokens,
                StaticController(receptor, static_coefficient), baseline["logits"],
            ),
            "bolus": generate_cached(
                model, tokenizer, prompt, args.max_new_tokens,
                PKPDGenerationController(
                    make_pkpd_model(receptor),
                    protocol_schedule("bolus", args.dose),
                ),
                baseline["logits"],
            ),
            "infusion": generate_cached(
                model, tokenizer, prompt, args.max_new_tokens,
                PKPDGenerationController(
                    make_pkpd_model(receptor),
                    protocol_schedule(
                        "infusion", args.dose, stop=args.max_new_tokens // 2
                    ),
                ),
                baseline["logits"],
            ),
            "pulses": generate_cached(
                model, tokenizer, prompt, args.max_new_tokens,
                PKPDGenerationController(
                    make_pkpd_model(receptor), pulse_schedule
                ),
                baseline["logits"],
            ),
            "random_norm_matched": generate_cached(
                model, tokenizer, prompt, args.max_new_tokens,
                StaticController(random_direction, static_coefficient),
                baseline["logits"],
            ),
            "opposite_receptor": generate_cached(
                model, tokenizer, prompt, args.max_new_tokens,
                StaticController(-receptor, static_coefficient),
                baseline["logits"],
            ),
            "shuffled_pulses": generate_cached(
                model, tokenizer, prompt, args.max_new_tokens,
                PKPDGenerationController(
                    make_pkpd_model(receptor),
                    shuffled_schedule(pulse_values, seed=prompt_index),
                ),
                baseline["logits"],
            ),
        }
        for condition in conditions.values():
            condition.pop("logits", None)
        results.append({
            "prompt_index": prompt_index,
            "prompt": prompt,
            "conditions": conditions,
        })
        print(f"{prompt_index + 1}/{min(args.prompts, len(HELD_OUT_PROMPTS))}: {prompt}")

    payload = {
        "model": MODEL_NAME,
        "layer": LAYER,
        "seed": 0,
        "batch_size": 1,
        "device": device,
        "model_dtype": str(model_dtype),
        "deterministic_generation": "argmax",
        "n_prompts": len(results),
        "max_new_tokens": args.max_new_tokens,
        "receptor_shape": list(receptor.shape),
        "receptor_norm": float(receptor.norm().item()),
        "random_receptor_cosine": float(torch.dot(receptor, random_direction).item()),
        "matched_static_coefficient": static_coefficient,
        "max_zero_dose_logit_error": max_zero_logit_error,
        "zero_dose_gate_passed": max_zero_logit_error < 1e-6,
        "kl_reference": (
            "baseline trajectory at the same decode index; after token "
            "divergence this includes prefix differences"
        ),
        "results": results,
    }
    condition_names = list(results[0]["conditions"]) if results else []
    payload["aggregate"] = {
        condition_name: {
            "mean_target_probe_score": sum(
                trace_row["target_probe_score"]
                for result in results
                for trace_row in result["conditions"][condition_name]["trace"]
            ) / max(
                1,
                sum(
                    len(result["conditions"][condition_name]["trace"])
                    for result in results
                ),
            ),
            "mean_kl_from_baseline": sum(
                trace_row["kl_from_baseline"]
                for result in results
                for trace_row in result["conditions"][condition_name]["trace"]
            ) / max(
                1,
                sum(
                    len(result["conditions"][condition_name]["trace"])
                    for result in results
                ),
            ),
            "mean_total_intervention": sum(
                result["conditions"][condition_name]["total_intervention"]
                for result in results
            ) / max(1, len(results)),
        }
        for condition_name in condition_names
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
