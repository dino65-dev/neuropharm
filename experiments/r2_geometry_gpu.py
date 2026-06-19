"""R2: grouped-data receptor geometry and corrected random-null audit.

This experiment repairs the remaining R1 confounds:

* claim IDs, not prompt rows, define train/validation/test splits;
* all response alternatives are tokenizer-length matched;
* twelve held-out response pairs are averaged;
* random directions use the identical PK/PD temporal schedule;
* random orientation is selected on validation and fixed on test;
* the test statistic is antisymmetric: (effect(+v) - effect(-v)) / 2;
* random controls are approximately matched to the receptor's validation
  KL-AUC using an empirical local quadratic scaling.

It also performs the R2 contrast geometry audit: coherence, pairwise cosine,
SVD explained variance, and 500 split-half stability replicates.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from administration.pkpd_controller import protocol_schedule
from pharmacokinetics.state_space import NeuropharmacologyModel, PKPDParameters


MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

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
    "vaccination can reduce the spread of infectious disease",
    "version control preserves a history of software changes",
    "random assignment reduces systematic treatment-selection bias",
    "photosynthesis converts light energy into chemical energy",
    "input validation reduces malformed-data failures",
    "rate limiting can reduce service abuse",
    "encryption protects data confidentiality",
    "unit tests cannot prove the absence of all defects",
    "replication improves confidence in empirical findings",
    "sample size influences statistical precision",
    "measurement error can bias scientific estimates",
    "caching can reduce repeated computation",
    "least-privilege access reduces security exposure",
    "gradual deployment can limit operational risk",
    "monitoring cannot replace fault-tolerant design",
    "the derivative describes a local rate of change",
    "matrix multiplication is generally not commutative",
    "cross-validation can estimate out-of-sample performance",
    "data leakage can inflate evaluation results",
    "regularization can reduce overfitting",
    "causal conclusions require assumptions beyond correlation",
    "semantic similarity is not identical to factual correctness",
    "tokenization can change measured sequence length",
    "quantization can alter model behavior",
    "confidence calibration differs from classification accuracy",
    "negative controls can expose experimental confounds",
    "blinding can reduce observer bias",
    "pre-registration can reduce analytic flexibility",
    "outliers can strongly influence a sample mean",
    "median statistics are robust to extreme values",
    "floating-point arithmetic has finite precision",
    "gradient clipping can limit optimization instability",
    "checkpointing can reduce activation-memory use",
    "distributed systems can experience partial failure",
    "retries can amplify load during outages",
]

QUESTION_TEMPLATES = [
    "Assess whether {claim}.",
    "What evidence supports the statement that {claim}?",
    "Give a concise judgment on this claim: {claim}.",
    "Evaluate the following statement: {claim}.",
]

CONSTRUCTION_RESPONSE_PAIRS = [
    ("I am confident that this conclusion is correct.", "I am uncertain whether this conclusion is correct."),
    ("The evidence provides a clear basis for this conclusion.", "The evidence provides a weak basis for this conclusion."),
    ("This conclusion has strong evidential support.", "This conclusion has weak evidential support."),
    ("The evidence warrants high confidence here.", "The evidence warrants low confidence here."),
    ("I consider this conclusion highly reliable.", "I consider this conclusion highly uncertain."),
]

EVALUATION_RESPONSE_PAIRS = [
    ("The evidence clearly supports this conclusion.", "The evidence barely supports this conclusion."),
    ("This claim is clearly supported by evidence.", "This claim is barely supported by evidence."),
    ("The result is definitely supported by evidence.", "The result is possibly supported by evidence."),
    ("The conclusion is certainly supported by evidence.", "The conclusion is perhaps supported by evidence."),
    ("Evidence makes this conclusion highly certain.", "Evidence makes this conclusion highly uncertain."),
    ("I am highly confident in this conclusion.", "I am highly uncertain about this conclusion."),
    ("This answer is definitely correct overall.", "This answer is possibly correct overall."),
    ("The conclusion follows decisively from evidence.", "The conclusion follows tentatively from evidence."),
    ("The evidence gives strong support overall.", "The evidence gives weak support overall."),
    ("This is a well-supported conclusion overall.", "This is a poorly-supported conclusion overall."),
    ("Confidence in this conclusion should be high.", "Confidence in this conclusion should be low."),
    ("The evidence makes the answer quite certain.", "The evidence makes the answer quite uncertain."),
]


@dataclass
class ScoreResult:
    scores: list[float]
    kl_auc: float


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


def prompt_rows(claim_ids: list[int]) -> list[tuple[int, str]]:
    return [
        (claim_id, template.format(claim=CLAIMS[claim_id]))
        for claim_id in claim_ids
        for template in QUESTION_TEMPLATES
    ]


def split_claim_ids(
    seed: int,
    construction_claims: int,
    validation_claims: int,
    test_claims: int,
) -> dict[str, list[int]]:
    ids = list(range(len(CLAIMS)))
    random.Random(seed).shuffle(ids)
    # 30/12/18 claims -> 120/48/72 prompt contexts. The test claims are
    # disjoint from all prior sequential-row R1 splits by construction here.
    construction_end = construction_claims
    validation_end = construction_end + validation_claims
    test_end = validation_end + test_claims
    if test_end > len(ids):
        raise ValueError(f"requested {test_end} claims, only {len(ids)} exist")
    return {
        "construction": ids[:construction_end],
        "validation": ids[construction_end:validation_end],
        "test": ids[validation_end:test_end],
    }


def assert_response_pairs_are_matched(tokenizer) -> list[int]:
    lengths = []
    if len(EVALUATION_RESPONSE_PAIRS) < 8:
        raise ValueError("at least eight response pairs are required")
    for positive, negative in EVALUATION_RESPONSE_PAIRS:
        positive_length = int(response_ids(tokenizer, positive).shape[1])
        negative_length = int(response_ids(tokenizer, negative).shape[1])
        if positive_length != negative_length:
            raise ValueError(
                f"response pair token mismatch: {positive_length} != "
                f"{negative_length}: {positive!r} / {negative!r}"
            )
        lengths.append(positive_length)
    for positive, negative in CONSTRUCTION_RESPONSE_PAIRS:
        positive_length = int(response_ids(tokenizer, positive).shape[1])
        negative_length = int(response_ids(tokenizer, negative).shape[1])
        if positive_length != negative_length:
            raise ValueError(
                "construction response pair token mismatch: "
                f"{positive_length} != {negative_length}"
            )
    return lengths


def make_pkpd_effects(steps: int, dose: float) -> list[float]:
    direction = torch.ones(1, dtype=torch.float32, device="cuda")
    model = NeuropharmacologyModel(PKPDParameters(
        receptor_basis=direction[:, None],
        affinity=torch.ones(1, 1, device="cuda"),
        ec50=torch.tensor([0.5], device="cuda"),
        hill=torch.tensor([2.0], device="cuda"),
        emax=torch.ones(1, device="cuda"),
        retention=torch.tensor([2.0 ** (-1.0 / 8.0)], device="cuda"),
        absorption=torch.ones(1, device="cuda"),
        recovery=torch.zeros(1, device="cuda"),
        desensitization=torch.zeros(1, device="cuda"),
    ))
    state = model.initial_state()
    schedule = protocol_schedule("pulses", dose=dose, stop=steps)
    effects = []
    for step_index in range(steps):
        step = model.step(
            state,
            schedule(step_index).to("cuda", dtype=state.concentration.dtype),
        )
        state = step.state
        effects.append(float(step.receptor_effect[0].item()))
    return effects


def build_full_ids(tokenizer, prompt: str, response: str) -> tuple[torch.Tensor, int]:
    prompt_ids = chat_prompt_ids(tokenizer, prompt)
    answer_ids = response_ids(tokenizer, response)
    return torch.cat([prompt_ids, answer_ids], dim=1), int(prompt_ids.shape[1])


def capture_contrast_matrix(
    model,
    tokenizer,
    rows: list[tuple[int, str]],
    layer: int,
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    differences = []
    metadata = []
    for row_index, (claim_id, prompt) in enumerate(rows):
        positive, negative = CONSTRUCTION_RESPONSE_PAIRS[
            row_index % len(CONSTRUCTION_RESPONSE_PAIRS)
        ]
        endpoints = []
        for response in (positive, negative):
            token_ids, _ = build_full_ids(tokenizer, prompt, response)
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
            endpoints.append(captured["resid"][0, -1, :].float())
        difference = endpoints[0] - endpoints[1]
        differences.append(difference)
        metadata.append({
            "claim_id": claim_id,
            "prompt": prompt,
            "response_pair_index": row_index % len(CONSTRUCTION_RESPONSE_PAIRS),
            "difference_norm": float(difference.norm().item()),
        })
    return torch.stack(differences), metadata


def geometry_audit(
    differences: torch.Tensor,
    seed: int,
    split_half_replicates: int,
) -> dict[str, Any]:
    unit = differences / differences.norm(dim=1, keepdim=True).clamp_min(1e-12)
    coherence = float(unit.mean(dim=0).norm().item())
    n = differences.shape[0]
    mean_pairwise = (n * coherence**2 - 1) / (n - 1)
    singular_values = torch.linalg.svdvals(differences)
    squared = singular_values.square()
    cumulative = torch.cumsum(squared, dim=0) / squared.sum()
    explained = {
        str(k): float(cumulative[min(k, len(cumulative)) - 1].item())
        for k in (1, 2, 4, 8, 16)
    }
    generator = torch.Generator(device="cpu").manual_seed(seed)
    cosines = []
    for _ in range(split_half_replicates):
        permutation = torch.randperm(n, generator=generator)
        midpoint = n // 2
        left = differences[permutation[:midpoint]].mean(dim=0)
        right = differences[permutation[midpoint:]].mean(dim=0)
        cosine = torch.nn.functional.cosine_similarity(
            left.unsqueeze(0), right.unsqueeze(0)
        )
        cosines.append(float(cosine.item()))
    ordered = sorted(cosines)
    return {
        "n_contrasts": n,
        "coherence": coherence,
        "mean_pairwise_cosine_from_coherence": mean_pairwise,
        "explained_variance": explained,
        "singular_values": [float(value) for value in singular_values.cpu()],
        "split_half": {
            "replicates": split_half_replicates,
            "median": statistics.median(cosines),
            "p05": ordered[int(0.05 * split_half_replicates)],
            "p95": ordered[int(0.95 * split_half_replicates) - 1],
            "minimum": min(cosines),
            "fraction_below_zero": sum(value < 0 for value in cosines) / len(cosines),
        },
    }


@contextmanager
def batched_intervention_hook(
    model,
    layer: int,
    direction: torch.Tensor | None,
    effects: list[float],
    scale: float,
    prompt_lengths: list[int],
    answer_lengths: list[int],
):
    if direction is None or scale == 0:
        yield
        return

    def hook(module, args):
        resid = args[0]
        changed = resid.clone()
        delta_direction = direction.to(resid.device, resid.dtype)
        for batch_index, (prompt_length, answer_length) in enumerate(
            zip(prompt_lengths, answer_lengths)
        ):
            for response_index, effect in enumerate(effects[:answer_length]):
                position = prompt_length - 1 + response_index
                changed[batch_index, position, :] += (
                    delta_direction * (float(effect) * scale)
                )
        return (changed, *args[1:])

    handle = residual_layer(model, layer).register_forward_pre_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def make_evaluation_examples(
    tokenizer,
    rows: list[tuple[int, str]],
) -> list[dict[str, Any]]:
    examples = []
    for row_index, (_, prompt) in enumerate(rows):
        pair_index = row_index % len(EVALUATION_RESPONSE_PAIRS)
        positive, negative = EVALUATION_RESPONSE_PAIRS[pair_index]
        for sign_index, response in enumerate((positive, negative)):
            token_ids, prompt_length = build_full_ids(tokenizer, prompt, response)
            examples.append({
                "row_index": row_index,
                "pair_index": pair_index,
                "sign_index": sign_index,
                "token_ids": token_ids.squeeze(0),
                "prompt_length": prompt_length,
                "answer_length": int(token_ids.shape[1] - prompt_length),
            })
    return examples


def run_evaluation_examples(
    model,
    tokenizer,
    examples: list[dict[str, Any]],
    layer: int,
    effects: list[float],
    direction: torch.Tensor | None,
    scale: float,
    baseline_logits: list[torch.Tensor] | None,
    batch_size: int,
) -> tuple[list[float], float, list[torch.Tensor]]:
    tokenizer.pad_token = tokenizer.eos_token
    scores = []
    logits_out = []
    total_kl = 0.0
    total_sequences = 0
    for start in range(0, len(examples), batch_size):
        batch = examples[start : start + batch_size]
        max_length = max(example["token_ids"].shape[0] for example in batch)
        input_ids = torch.full(
            (len(batch), max_length),
            tokenizer.pad_token_id,
            dtype=torch.long,
        )
        attention_mask = torch.zeros_like(input_ids)
        for index, example in enumerate(batch):
            length = example["token_ids"].shape[0]
            input_ids[index, :length] = example["token_ids"]
            attention_mask[index, :length] = 1
        prompt_lengths = [example["prompt_length"] for example in batch]
        answer_lengths = [example["answer_length"] for example in batch]
        with batched_intervention_hook(
            model, layer, direction, effects, scale,
            prompt_lengths, answer_lengths,
        ):
            with torch.no_grad():
                full_logits = model(
                    input_ids=input_ids.to(input_device(model)),
                    attention_mask=attention_mask.to(input_device(model)),
                    use_cache=False,
                ).logits.float()
        for local_index, example in enumerate(batch):
            prompt_length = example["prompt_length"]
            answer_length = example["answer_length"]
            answer = example["token_ids"][
                prompt_length : prompt_length + answer_length
            ].to(full_logits.device)
            logits = full_logits[
                local_index,
                prompt_length - 1 : prompt_length - 1 + answer_length,
                :,
            ]
            log_probs = torch.log_softmax(logits, dim=-1)
            score = log_probs.gather(
                -1, answer.unsqueeze(-1)
            ).squeeze(-1).mean()
            scores.append(float(score.item()))
            logits_cpu = logits.cpu()
            logits_out.append(logits_cpu)
            if baseline_logits is not None:
                baseline = baseline_logits[start + local_index].to(logits.device)
                log_q = torch.log_softmax(baseline, dim=-1)
                p = log_probs.exp()
                total_kl += float(
                    torch.sum(p * (log_probs - log_q), dim=-1).sum().item()
                )
            total_sequences += 1
        del full_logits
    return scores, total_kl / max(1, total_sequences), logits_out


def evaluate_direction(
    model,
    tokenizer,
    examples: list[dict[str, Any]],
    layer: int,
    effects: list[float],
    direction: torch.Tensor | None,
    scale: float,
    baseline_result: ScoreResult,
    baseline_logits: list[torch.Tensor],
    batch_size: int,
) -> ScoreResult:
    logprobs, kl_auc, _ = run_evaluation_examples(
        model, tokenizer, examples, layer, effects, direction, scale,
        baseline_logits, batch_size,
    )
    scores = [
        logprobs[index] - logprobs[index + 1]
        for index in range(0, len(logprobs), 2)
    ]
    if len(scores) != len(baseline_result.scores):
        raise RuntimeError("evaluation score count changed")
    return ScoreResult(scores=scores, kl_auc=kl_auc)


def build_baseline_cache(
    model,
    tokenizer,
    examples,
    layer,
    effects,
    batch_size,
) -> tuple[ScoreResult, list[torch.Tensor]]:
    logprobs, _, logits = run_evaluation_examples(
        model, tokenizer, examples, layer, effects, None, 0.0, None, batch_size
    )
    scores = [
        logprobs[index] - logprobs[index + 1]
        for index in range(0, len(logprobs), 2)
    ]
    return ScoreResult(scores, 0.0), logits


def mean_delta(result: ScoreResult, baseline: ScoreResult) -> float:
    return statistics.fmean(
        value - reference
        for value, reference in zip(result.scores, baseline.scores)
    )


def antisymmetric_effect(
    positive: ScoreResult,
    negative: ScoreResult,
    baseline: ScoreResult,
) -> float:
    return (
        mean_delta(positive, baseline) - mean_delta(negative, baseline)
    ) / 2


def random_orthogonal(
    receptor: torch.Tensor,
    generator: torch.Generator,
) -> torch.Tensor:
    direction = torch.randn(
        receptor.shape, generator=generator, dtype=torch.float32
    ).to(receptor.device)
    direction -= torch.dot(direction, receptor) * receptor
    return direction / direction.norm().clamp_min(1e-12)


def quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil(probability * len(ordered)) - 1)
    return ordered[index]


def load_model(args):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, default=12)
    parser.add_argument("--dose", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=20260619)
    parser.add_argument("--random-directions", type=int, default=128)
    parser.add_argument("--split-half-replicates", type=int, default=500)
    parser.add_argument("--construction-claims", type=int, default=30)
    parser.add_argument("--validation-claims", type=int, default=12)
    parser.add_argument("--test-claims", type=int, default=18)
    parser.add_argument("--gpu-memory", default="2600MiB")
    parser.add_argument("--cpu-memory", default="12GiB")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--offload-folder", type=Path, default=Path("artifacts/_cache/r2_offload"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/pkpd/r2_geometry_gpu.json"))
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.cuda.reset_peak_memory_stats()
    start = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    pair_lengths = assert_response_pairs_are_matched(tokenizer)
    split_ids = split_claim_ids(
        args.seed,
        args.construction_claims,
        args.validation_claims,
        args.test_claims,
    )
    rows = {name: prompt_rows(ids) for name, ids in split_ids.items()}
    model = load_model(args)

    differences, contrast_metadata = capture_contrast_matrix(
        model, tokenizer, rows["construction"], args.layer
    )
    geometry = geometry_audit(
        differences, args.seed, args.split_half_replicates
    )
    receptor = differences.mean(dim=0)
    receptor = receptor / receptor.norm().clamp_min(1e-12)
    max_steps = max(pair_lengths)
    effects = make_pkpd_effects(max_steps, args.dose)
    validation_examples = make_evaluation_examples(tokenizer, rows["validation"])
    test_examples = make_evaluation_examples(tokenizer, rows["test"])

    validation_baseline, validation_cache = build_baseline_cache(
        model, tokenizer, validation_examples, args.layer, effects,
        args.batch_size,
    )
    test_baseline, test_cache = build_baseline_cache(
        model, tokenizer, test_examples, args.layer, effects,
        args.batch_size,
    )
    receptor_val_pos = evaluate_direction(
        model, tokenizer, validation_examples, args.layer, effects,
        receptor, 1.0, validation_baseline, validation_cache, args.batch_size,
    )
    receptor_val_neg = evaluate_direction(
        model, tokenizer, validation_examples, args.layer, effects,
        -receptor, 1.0, validation_baseline, validation_cache, args.batch_size,
    )
    receptor_target_kl = (
        receptor_val_pos.kl_auc + receptor_val_neg.kl_auc
    ) / 2
    receptor_test_pos = evaluate_direction(
        model, tokenizer, test_examples, args.layer, effects,
        receptor, 1.0, test_baseline, test_cache, args.batch_size,
    )
    receptor_test_neg = evaluate_direction(
        model, tokenizer, test_examples, args.layer, effects,
        -receptor, 1.0, test_baseline, test_cache, args.batch_size,
    )
    receptor_test_antisymmetric = antisymmetric_effect(
        receptor_test_pos, receptor_test_neg, test_baseline
    )

    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    random_null = []
    for index in range(args.random_directions):
        direction = random_orthogonal(receptor, generator)
        val_pos_unit = evaluate_direction(
            model, tokenizer, validation_examples, args.layer, effects,
            direction, 1.0, validation_baseline, validation_cache,
            args.batch_size,
        )
        val_neg_unit = evaluate_direction(
            model, tokenizer, validation_examples, args.layer, effects,
            -direction, 1.0, validation_baseline, validation_cache,
            args.batch_size,
        )
        unit_kl = (val_pos_unit.kl_auc + val_neg_unit.kl_auc) / 2
        scale = math.sqrt(
            receptor_target_kl / max(unit_kl, 1e-12)
        )
        scale = min(max(scale, 0.1), 10.0)
        val_pos_delta = mean_delta(val_pos_unit, validation_baseline)
        val_neg_delta = mean_delta(val_neg_unit, validation_baseline)
        orientation = 1.0 if val_pos_delta >= val_neg_delta else -1.0
        oriented = direction * orientation
        test_pos = evaluate_direction(
            model, tokenizer, test_examples, args.layer, effects,
            oriented, scale, test_baseline, test_cache, args.batch_size,
        )
        test_neg = evaluate_direction(
            model, tokenizer, test_examples, args.layer, effects,
            -oriented, scale, test_baseline, test_cache, args.batch_size,
        )
        effect = antisymmetric_effect(test_pos, test_neg, test_baseline)
        matched_kl = (test_pos.kl_auc + test_neg.kl_auc) / 2
        random_null.append({
            "index": index,
            "orientation": orientation,
            "scale": scale,
            "validation_unit_kl_auc": unit_kl,
            "test_kl_auc": matched_kl,
            "test_antisymmetric_effect": effect,
            "cosine_with_receptor": float(torch.dot(receptor, oriented).item()),
        })
        print(
            f"random {index + 1}/{args.random_directions}: "
            f"scale={scale:.3f} effect={effect:+.6f} kl={matched_kl:.6g}"
        )

    random_effects = [row["test_antisymmetric_effect"] for row in random_null]
    q95 = quantile(random_effects, 0.95)
    percentile = (
        1 + sum(value <= receptor_test_antisymmetric for value in random_effects)
    ) / (len(random_effects) + 1)
    payload: dict[str, Any] = {
        "experiment": "R2_receptor_geometry_audit",
        "model": MODEL_NAME,
        "layer": args.layer,
        "site": "block_input_residual_pre",
        "seed": args.seed,
        "claim_group_split": split_ids,
        "split_prompt_counts": {name: len(value) for name, value in rows.items()},
        "legacy_r1_test_marked_spent": True,
        "construction_response_pairs": CONSTRUCTION_RESPONSE_PAIRS,
        "evaluation_response_pairs": EVALUATION_RESPONSE_PAIRS,
        "response_pair_token_lengths": pair_lengths,
        "teacher_forcing": True,
        "pkpd_effect_schedule": effects,
        "geometry": geometry,
        "contrast_metadata": contrast_metadata,
        "receptor": {
            "norm": float(receptor.norm().item()),
            "validation_positive_delta": mean_delta(receptor_val_pos, validation_baseline),
            "validation_negative_delta": mean_delta(receptor_val_neg, validation_baseline),
            "validation_kl_auc": receptor_target_kl,
            "test_positive_delta": mean_delta(receptor_test_pos, test_baseline),
            "test_negative_delta": mean_delta(receptor_test_neg, test_baseline),
            "test_antisymmetric_effect": receptor_test_antisymmetric,
            "test_kl_auc": (
                receptor_test_pos.kl_auc + receptor_test_neg.kl_auc
            ) / 2,
        },
        "random_null": {
            "count": len(random_null),
            "controls": random_null,
            "q95": q95,
            "receptor_percentile": percentile,
            "mean": statistics.fmean(random_effects),
            "std": statistics.stdev(random_effects),
        },
        "gates": {
            "claim_group_disjoint": (
                len(set(split_ids["construction"]) & set(split_ids["validation"])) == 0
                and len(set(split_ids["construction"]) & set(split_ids["test"])) == 0
                and len(set(split_ids["validation"]) & set(split_ids["test"])) == 0
            ),
            "response_lengths_matched": True,
            "sign": (
                mean_delta(receptor_test_pos, test_baseline) > 0
                and mean_delta(receptor_test_neg, test_baseline) < 0
            ),
            "random_null": receptor_test_antisymmetric > q95,
        },
        "cuda": {
            "device": torch.cuda.get_device_name(0),
            "capability": list(torch.cuda.get_device_capability(0)),
            "torch_version": torch.__version__,
            "cuda_build": torch.version.cuda,
            "device_map": model.hf_device_map,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        },
        "elapsed_seconds": time.time() - start,
    }
    payload["overall_passed"] = all(payload["gates"].values())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
