from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path

import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer, BitsAndBytesConfig

from experiments.step7b_modern_models import (
    CONFIDENT_PAIRS,
    EVAL_PROMPTS,
    HARM_PAIRS,
    count_confidence,
    count_hedged,
    count_harm_words,
    count_refusals,
    get_d_model,
    get_n_layers,
    gen_with_drug,
    mean_diff_vector,
    project_out,
    strip_prompt,
)

ART = Path("artifacts")
DEVICE = "cuda"
DEFAULT_MODEL = "Qwen/Qwen3.5-4B"
DEFAULT_LAYERS = [12, 16, 20, 22, 24, 28, 30]
RAW_COEFFS = [-1.0, 0.0, 0.5, 1.0]
UNIT_COEFFS = [1.0]
EVAL_PROMPTS_STEP8 = EVAL_PROMPTS[:2]


def load_model(model_name: str):
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        quantization_config=bnb_cfg,
        dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True,
    )
    model.eval()
    return model, tok


def vector_stats(v_drug, v_harm, v_antidote=None):
    out = {
        "drug_norm": float(v_drug.norm()),
        "harm_norm": float(v_harm.norm()),
        "overlap_drug_harm": float((v_drug * v_harm).sum() / (v_drug.norm() * v_harm.norm() + 1e-9)),
    }
    if v_antidote is not None:
        out["antidote_norm"] = float(v_antidote.norm())
        out["antidote_overlap_to_harm"] = float((v_antidote * v_harm).sum() / (v_antidote.norm() * v_harm.norm() + 1e-9))
    return out


def gen_metrics(model, tok, vec, layer_idx, coeff, prompt, max_new_tokens):
    gen = gen_with_drug(model, tok, vec, layer_idx, coeff, prompt, max_new_tokens=max_new_tokens)
    text = strip_prompt(gen, prompt).strip()
    return {
        "prompt": prompt,
        "generation": text,
        "confident": count_confidence(text),
        "hedged": count_hedged(text),
        "refusals": count_refusals(text),
        "harm_words": count_harm_words(text),
    }


def avg_metric(rows, key):
    if not rows:
        return 0.0
    return float(sum(r[key] for r in rows) / len(rows))


def summarize_layer(layer):
    rows = layer["dose_sweep"]["raw"]
    baseline = rows.get("0.0", [])
    clean = rows.get("1.0", [])
    baseline_conf = avg_metric(baseline, "confident")
    clean_conf = avg_metric(clean, "confident")
    ratio = clean_conf / (baseline_conf + 1e-9)
    return {
        "layer": layer["layer"],
        "drug_norm": layer["vector_stats"]["drug_norm"],
        "harm_norm": layer["vector_stats"]["harm_norm"],
        "cos_drug_harm": layer["vector_stats"]["overlap_drug_harm"],
        "baseline_confident": baseline_conf,
        "clean_c1_confident": clean_conf,
        "clean_vs_baseline_ratio": ratio,
        "passes": layer["drug_norm"] >= 5.0 and ratio >= 1.5,
    }


def random_control(model, tok, layer_idx, base_vec, prompt, condition, seed, max_new_tokens):
    d = base_vec.numel()
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    v = torch.randn(d, generator=g, device="cpu", dtype=torch.float32)
    target_norm = base_vec.norm().item()
    v = v * (target_norm / (v.norm() + 1e-9))
    if condition == "orthogonal_same_norm":
        v = v - (v * base_vec).sum() / (base_vec.norm().item() ** 2 + 1e-9) * base_vec
        v = v * (target_norm / (v.norm() + 1e-9))
    return gen_metrics(model, tok, v, layer_idx, 1.0, prompt, max_new_tokens)


def run_random_controls_if_needed(model, tok, layer_results, layer_vectors, max_new_tokens):
    summaries = [summarize_layer(layer) for layer in layer_results]
    if any(s["passes"] for s in summaries):
        return {"skipped": True, "reason": "at least one layer passed"}

    selected = sorted(layer_results, key=lambda r: r["vector_stats"]["drug_norm"], reverse=True)[:3]
    controls = []
    conditions = ["random_same_norm", "orthogonal_same_norm"]
    seeds = list(range(5))
    for layer in selected:
        li = layer["layer"]
        v = layer_vectors[li]["v_drug"]
        for cond in conditions:
            for seed in seeds:
                rows = []
                for prompt in EVAL_PROMPTS_STEP8:
                    rows.append(random_control(model, tok, li, v, prompt, cond, seed, max_new_tokens))
                controls.append({
                    "layer": li,
                    "condition": cond,
                    "seed": seed,
                    "rows": rows,
                    "avg_confident": avg_metric(rows, "confident"),
                    "avg_hedged": avg_metric(rows, "hedged"),
                })
    return {
        "skipped": False,
        "selected_layers": [c["layer"] for c in controls if c["condition"] == "random_same_norm" and c["seed"] == 0],
        "controls": controls,
    }


def main():
    model_name = os.environ.get("STEP8_MODEL", DEFAULT_MODEL)
    layers = [int(x) for x in os.environ.get("STEP8_LAYERS", ",".join(map(str, DEFAULT_LAYERS))).split(",")]
    max_new_tokens = int(os.environ.get("STEP8_MAX_NEW_TOKENS", "50"))
    out_path = ART / "step8_qwen35_layer_sweep.json"
    ART.mkdir(exist_ok=True)

    print(f"loading {model_name}")
    t0 = time.time()
    model, tok = load_model(model_name)
    n_layers = get_n_layers(model)
    d_model = get_d_model(model)
    print(f"layers={n_layers} d_model={d_model} vram={torch.cuda.memory_allocated()/1024**3:.2f}GB")

    layer_results = []
    layer_vectors = {}
    for li in layers:
        if li < 0 or li >= n_layers:
            print(f"skip invalid layer {li}")
            continue
        print(f"\n=== layer {li}/{n_layers} ({li/n_layers:.1%}) ===")
        lt = time.time()
        v_drug = mean_diff_vector(model, tok, CONFIDENT_PAIRS, li)
        v_harm = mean_diff_vector(model, tok, HARM_PAIRS, li)
        v_contam = v_drug + 0.3 * v_harm
        v_antidote = project_out(v_contam, v_harm)
        stats = vector_stats(v_drug, v_harm, v_antidote)
        layer_vectors[li] = {"v_drug": v_drug, "v_harm": v_harm, "v_antidote": v_antidote}
        print(f"drug_norm={stats['drug_norm']:.3f} harm_norm={stats['harm_norm']:.3f} cos={stats['overlap_drug_harm']:+.3f}")

        raw_rows = {}
        for c in RAW_COEFFS:
            rows = []
            for prompt in EVAL_PROMPTS_STEP8:
                rows.append(gen_metrics(model, tok, v_drug, li, c, prompt, max_new_tokens))
                print(f"raw c={c:+.1f} conf={rows[-1]['confident']} hed={rows[-1]['hedged']} {rows[-1]['generation'][:60]!r}")
            raw_rows[str(c)] = rows

        unit_rows = {}
        v_drug_unit = v_drug / (v_drug.norm() + 1e-9)
        for c in UNIT_COEFFS:
            rows = []
            for prompt in EVAL_PROMPTS_STEP8:
                rows.append(gen_metrics(model, tok, v_drug_unit, li, c, prompt, max_new_tokens))
                print(f"unit c={c:+.1f} conf={rows[-1]['confident']} hed={rows[-1]['hedged']} {rows[-1]['generation'][:60]!r}")
            unit_rows[str(c)] = rows

        layer_results.append({
            "layer": li,
            "layer_fraction": li / n_layers,
            "d_model": d_model,
            "vector_stats": stats,
            "dose_sweep": {
                "raw": raw_rows,
                "unit_norm": unit_rows,
            },
            "elapsed_sec": time.time() - lt,
        })

        out = {
            "model": model_name,
            "quantization": "4-bit NF4",
            "n_layers": n_layers,
            "d_model": d_model,
            "requested_layers": layers,
            "layer_results": layer_results,
            "summaries": [summarize_layer(r) for r in layer_results],
            "pass_threshold": {"drug_norm_at_least": 5.0, "clean_vs_baseline_ratio_at_least": 1.5},
        }
        out_path.write_text(json.dumps(out, indent=2))
        torch.cuda.empty_cache()

    summaries = [summarize_layer(r) for r in layer_results]
    pass_layers = [s["layer"] for s in summaries if s["passes"]]
    print("\n=== summary ===")
    for s in summaries:
        print(f"layer={s['layer']:>2} drug_norm={s['drug_norm']:.3f} cos={s['cos_drug_harm']:+.3f} baseline={s['baseline_confident']:.2f} clean_c1={s['clean_c1_confident']:.2f} ratio={s['clean_vs_baseline_ratio']:.2f} pass={s['passes']}")
    print(f"pass_layers={pass_layers}")

    random_controls = run_random_controls_if_needed(model, tok, layer_results, layer_vectors, max_new_tokens)
    final = {
        "model": model_name,
        "quantization": "4-bit NF4",
        "n_layers": n_layers,
        "d_model": d_model,
        "requested_layers": layers,
        "layer_results": layer_results,
        "summaries": summaries,
        "pass_layers": pass_layers,
        "pass_threshold": {"drug_norm_at_least": 5.0, "clean_vs_baseline_ratio_at_least": 1.5},
        "random_controls_if_failed": random_controls,
        "elapsed_sec_total": time.time() - t0,
    }
    out_path.write_text(json.dumps(final, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
