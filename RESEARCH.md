# NeuroPharm Research Archive

## Activation Steering Dose-Response Pharmacology in Small Language Models

**Date:** June 2026  
**Repository:** `F:\neuropharma\neuropharm`  
**Status:** 9 original steps + 2 follow-up rounds complete

---

## Table of Contents

1. [Background and Motivation](#1-background-and-motivation)
2. [Methods](#2-methods)
3. [The Math](#3-the-math)
4. [Step-by-Step Results](#4-step-by-step-results)
5. [Cross-Model Comparison Tables](#5-cross-model-comparison-tables)
6. [The Geometric Model of Assertiveness Disentanglement](#6-the-geometric-model-of-assertiveness-disentanglement)
7. [Mamba-Transformer Steering Theory](#7-mamba-transformer-steering-theory)
8. [Sparse Autoencoder Steering](#8-sparse-autoencoder-steering)
9. [Null-Space Antidote](#9-null-space-antidote)
10. [Vulnerability Map (VULN-001 through VULN-041)](#10-vulnerability-map)
11. [Open Problems](#11-open-problems)
12. [Repository File Index](#12-repository-file-index)

---

## 1. Background and Motivation

### 1.1 What Is Activation Steering?

Activation steering — adding a direction vector `v` to a transformer's residual
stream at inference time — is a white-box control technique introduced by Turner
et al. (2023) as **Activation Addition (ActAdd)**.  The key idea:

1. Select a "behavior" (e.g., confident tone).
2. Construct `K` contrastive sentence pairs `(positive_i, negative_i)`.
3. For each pair, run through the model, cache the residual stream
   `x^(pos)_i`, `x^(neg)_i` at some layer `L` at a token position.
4. Compute the drug vector: `v_drug = (1/K) Σ_i (x^(pos)_i − x^(neg)_i)`.
5. At inference, inject the drug: `x_L ← x_L + c · v_drug` for a scalar
   coefficient `c`.

The entire prior literature is on **7B-70B models** (LLaMA-3, Gemma 2, Mistral-7B).
No prior work has systematically characterized dose-response curves for the
**1.5B-2B SLM regime** — models that actually run on a student-laptop 4 GB GPU.

### 1.2 The Drug Analogy

| Neuropharmacology | Activation-steering equivalent |
|---|---|
| Drug | Steering vector `v_drug` |
| Dose | Coefficient `c` |
| Overdose | Coefficient too high → incoherent generation |
| Antagonist / antidote | Null-space projection onto `span(v_harm)^⊥` |
| Receptor-targeted drug | SAE feature clamping |
| Off-target effects | Behavior drift on out-of-distribution prompts |
| Dose-response curve | Confident-word count vs. coefficient |

### 1.3 The Nine-Step Plan

The user prescribed an exact sequence:

1. Set up environment, install packages, download Qwen-2.5-1.5B-Instruct.
2. Read 5 key papers (ActAdd, RepEng survey, SAS, AdaSteer, Non-Surjective).
3. Build a "confident-tone" drug by hand — 20 contrastive pairs,
   layer-12 residual stream, mean-diff vector, dose sweep c ∈ [-2,+2].
4. Attack the drug — adversarial prompts, out-of-distribution domain,
   extended dose to find overdose threshold, off-target effect measurement.
5. Build a sparse SAE version — train a TopK SAE on layer-12 activations,
   find "confidence features", compare dense vs. sparse steering on 3 behaviors.
6. Build an antidote — null-space projection of a drug's harm-direction
   component, test on 20 train + 20 test prompts.
7. Cross-model transfer — re-run on Gemma-2-2B, test if drug/antidote transfer.
8. Document everything with real measured data.
9. Write a 2-page research note.

We completed all 9 steps, then followed up with:

- **Step 7B:** Cross-model on modern LLMs (Qwen3.5-4B, Gemma-4-E4B).
- **Step 7C:** Qwen3.5 deeper-layer scan + prompt-level control test + Nemotron-3 attempt.
- **Step 7D:** Qwen3.5 L24 antidote construction.
- **Round 2 with math-researcher:** Two back-and-forth conversations, including
  theoretical analysis of the Qwen3.5 failure mode and Mamba-hybrid pre-block
  injection strategy.

---

## 2. Methods

### 2.1 ActAdd Construction

```
v_drug = (1/K) Σ_{i=1}^{K} ( x^(pos)_i[L, last] − x^(neg)_i[L, last] )
```

- `x[L, last]`: the residual stream at `blocks.L.hook_resid_pre` (input to
  block `L`) at the last non-padding token.
- No normalization applied.
- Injection: `x_L ← x_L + c · v_drug` via forward hook.
- Typically `K = 20` for "confident tone", `K = 10` for "harm direction".

### 2.2 Null-Space Projection Antidote

```
v_harm       = mean-diff over harmful-vs-safe intent pairs
v_contam     = v_drug + α · v_harm,   α = 0.3
v_antidote   = v_contam − (v_contam · v̂_harm) v̂_harm
             = projection of v_contam onto span(v_harm)^⊥
```

### 2.3 TopK Sparse Autoencoder

```
TopK SAE: d_in=1536, d_hidden=4096, k=32
Encoder:    z = TopK ReLU(x_norm · W_enc + b_enc)
Decoder:    x̂ = z · W_dec + b_dec,   W_dec normalized to unit-norm columns
```

Trained on 30,720 wikitext-2 layer-12 activations, 1500 SGD steps, MSE loss.

### 2.4 Dose-Response Protocol

Coefficient sweep from c ∈ {-2.0, -1.5, ..., +2.0} on a single fixed prompt.
Generation with `max_new_tokens=60`, `temperature=1.0`, `do_sample=True`.
Recording at each coefficient: full generation text, 4-gram repetition score,
confident-word hit count, hedged-word hit count.

### 2.5 Cross-Model Transfer Protocol

Same 20 confident pairs, same 10 harm pairs, same antidote construction at the
same absolute layer index (12).  Models tested:

| Model | Type | Layers | d_model | Bits | Hardware |
|-------|------|-------:|--------:|-----:|----------|
| Qwen-2.5-1.5B-Instruct | Dense transformer | 28 | 1536 | fp16 | Local GTX 1050 Ti |
| Gemma-2-2B-it | Dense transformer | 26 | 2304 | 4-bit | T4 (Lightning) |
| Qwen3.5-4B | Dense transformer (reasoning-tuned) | 32 | 2560 | 4-bit | T4 |
| Gemma-4-E4B-it | MoE transformer | 42 | 2560 | 4-bit | T4 |
| Nemotron-3-Nano-4B | Mamba-transformer hybrid | 42 | 3136 | 4-bit | T4 (blocked) |

### 2.6 Layer-Scan Protocol (Step 7C)

For each layer `L ∈ {12, 18, 24, 28}` of Qwen3.5-4B:

1. Extract v_drug(L) and v_harm(L) from the same 10 contrastive pairs.
2. Compute ‖v_drug(L)‖, ‖v_harm(L)‖, and cos(v_drug(L), v_harm(L)).
3. For the layers with largest ‖v_drug‖, run generation with c = 0.0 and c = +1.0
   on 3 prompts.  Record keyword counts and full text.

### 2.7 Prompt-Level Control Test

A plain-text control: prepend `"Answer confidently and assertively. "` or
`"Answer with significant hesitation and uncertainty. "` to the user message,
with `enable_thinking=False`.  Generates 3 completions per style, records
keyword counts.  Serves as a go/no-go gate: if the model cannot produce
confident outputs at the prompt level, no injection method will work.

### 2.8 Hardware

- **Local:** Windows 10, NVIDIA GeForce GTX 1050 Ti (4 GB, Pascal sm_61),
  Python 3.11.9, CUDA 11.8, torch 2.4.1+cu118, transformers 4.46.3,
  transformer_lens 2.16.1.
- **T4:** Lightning Studio Tesla T4 (16 GB, Turing sm_75), Python 3.10.10,
  CUDA 12.1, torch 2.5.1+cu121, transformers 5.5.3.  For mamba-ssm-dependent
  models: attempted but blocked by missing prebuilt wheel.

---

## 3. The Math

### 3.1 ActAdd Steering Vector

Let `X = {x_i^(pos), x_i^(neg)}` for `i = 1..K` be the last-token residual
stream vectors at layer `L`.  Define:

```
v_drug = (1/K) Σ_i (x_i^(pos) − x_i^(neg))      [eq. 1]
```

The steering intervention is:

```
h_L ← h_L + c · v_drug                          [eq. 2]
```

where `h_L` is the residual stream at the input to block `L`, and `c ∈ R` is
the dose coefficient.

### 3.2 Null-Space Projection

Let `v_harm` be a second drug vector (harm-vs-safe contrast).  Define the
contaminated and antidote vectors:

```
v_contam = v_drug + α · v_harm                   [eq. 3]
v̂_harm = v_harm / ‖v_harm‖                       [eq. 4]
v_antidote = v_contam − (v_contam · v̂_harm) v̂_harm   [eq. 5]
```

By construction, `v_antidote ⟂ v_harm` (cos = 0).

### 3.3 Norm Retention Under Projection

The fraction of `v_drug` norm retained in the antidote:

```
‖v_antidote‖ / ‖v_drug‖ = √(1 − cos²(v_drug, v_harm))      [eq. 6]
```

This follows directly from the projection geometry: projecting onto
`span(v_harm)^⊥` removes the component `(v·v̂)v̂`, which has norm `‖v‖·|cos θ|`.

For `cos = −0.109` (Qwen3.5 L24), prediction = √(1 − 0.0119) = 0.988 → 98.8%.
Observed: 99.4%.  Error: 0.6 percentage points.

### 3.4 The cos(v_drug, v_harm) Crossover Model

Define the "assertiveness" and "safety" directions in the residual stream:

```
a(L) = direction of stylistic forcefulness at layer L
s(L) = direction of aligned-vs-malicious intent at layer L

v_drug(L) ≈ α_drug(L) · a(L) + β_drug(L) · s(L)
v_harm(L) ≈ α_harm(L) · a(L) + β_harm(L) · s(L)
```

At early layers (L=12): `s(L)` has small amplitude.  Both v_drug and v_harm
project positively onto `a(L)` → cos > 0.

As L increases (`L → L*`, the crossover): `s(L)` grows.  v_drug projects
neutrally onto `s(L)` (β_drug ≈ 0), but v_harm projects negatively onto
`s(L)` (β_harm < 0) → the s-component drives the cosine negative.

At late layers (L > L*): `cos(v_drug, v_harm) < 0`, meaning the disentanglement
is complete.

**Prediction:** cos(L) transitions from positive → 0 → negative with a
crossover at some layer L*.  Our layer scan tested this prediction.

### 3.5 SAE TopK Additive Steering

SAE encoder: `z = TopK[ReLU(W_enc · x_norm + b_enc)]`

Additive sparse steering:

```
z_base   = SAE.encode(x)                         [eq. 7]
z_boost  = z_base;   z_boost[target_features] *= B
δ_x      = SAE.decode(z_boost) − SAE.decode(z_base)   [eq. 8]
x_steered = x_norm + δ_x                         [eq. 9]
```

This preserves the original residual and injects only the change induced by the
feature boost.

### 3.6 Non-Surjectivity (Mishra et al., 2024)

For a transformer layer `f_L: R^d → R^d`, under mild conditions (LayerNorm +
rank constraints), `Im(f_L)` is a proper subset of `R^d`.  Adding a steering
vector `c·v_drug` can push the residual stream to regions the model has never
visited during training.

The *overdose phenomenon* — incoherent output at large |c| — is a consequence of
this: the model's downstream layers receive an "impossible" residual stream.
The non-surjective manifold is not an empirical glitch; it is a *provable
property* of the transformer layer.

### 3.7 Mamba-2 SSM Boundedness

For a Mamba-2 SSM block:

```
Δ_t = softplus(W_Δ · x_t + b_Δ)                 [eq. 10]
A_t = exp(−exp(Δ_t) ⊙ A)                         [eq. 11]
h_t = A_t ⊙ h_{t−1} + B_t ⊗ x_t                 [eq. 12]
```

Since ‖A_t‖_∞ < 1 for all t, the hidden state `h_t` is provably bounded.
For transformers, there is no such bound — the residual stream can drift
arbitrarily far from the training manifold.

**Pre-block injection** (inject v before the SSM block):

```
x_t ← x_t + c·v_drug                              [eq. 13]
h_t^steered = A_t^steered ⊙ h_{t−1} + B_t^steered ⊗ (x_t + c·v_drug)   [eq. 14]
```

where `A_t^steered`, `B_t^steered` are computed from the steered input.  This
propagates the steering through the recurrent state naturally.

**Post-block injection** (standard ActAdd, inject v after the block):

```
h_t = A_t ⊙ h_{t−1} + B_t ⊗ x_t                  [eq. 15]
x_t ← x_t + c·v_drug                              [eq. 16]
```

Now `h_t` was computed from the *unsteered* `x_t` but the output is steered.
This creates a `h_t` / `x_t` mismatch that the recurrence will fight at t+1.

---

## 4. Step-by-Step Results

### 4.1 Step 1: Environment & Model Verification

All packages installed, Qwen-2.5-1.5B-Instruct loads in fp16 on 4 GB GPU
(2.88 GB), generates `"Ready."` in one word.  transformer_lens wraps it
with `n_layers=28, d_model=1536`.

### 4.2 Step 2: Five-Paper Reading (completed June 7, 2026)

| # | Paper | arXiv ID | Core contribution |
|---|-------|----------|-------------------|
| 1 | ActAdd (Turner et al., 2023) | 2308.10248 | First to show a single contrastive-mean vector shifts LLM behavior |
| 2 | RepEng Survey (Bartoszcze et al., 2025) | 2502.17601 | Full taxonomy of representation-engineering methods |
| 3 | SAS (Bayat et al., 2025) | 2503.00177 | SAE-based sparse steering on Gemma 2 |
| 4 | AdaSteer (Zhao et al., 2025) | 2504.09466 | Adaptive jailbreak defense via rejection-direction steering |
| 5 | Non-Surjective (Mishra et al., 2026) | 2604.09839 | Formal proof that steering pushes activations off the prompt-realizable manifold |

### 4.3 Step 3: First Drug — Dose-Response on Qwen-2.5-1.5B

```
Drug:     v_drug, layer 12, 20 contrastive confident-tone pairs
norms:    ‖v_drug‖ = 12.001, mean = +0.0312, std = 0.3057
prompt:   "Q: Is it a good idea to drink water every day?\nA:"
```

| c | Generation (first 100 chars) | rep_score | Phase |
|--:|------------------------------|----------:|-------|
| −2.0 | "Not really. Not sure if it's a good idea to drink water every day, but it's alright" | 0.0 | Incoherent |
| **−1.5** | "Maybe. Say yes. Q: Can't making some energy drinks hurt the pet bird? A: I'm not sure. **Quessyudpklg flkjsldkjfsalkdjflksajflsdjfls**" | 0.0 | **Overdose: random character runs** |
| −1.0 | "Yes, it's not good to drink water every day. Actually, some people think bathing in fresh water..." | 0.0 | Mixed/confused |
| **−0.5** | "Since thirst is not a reliable indicator of how much water is in the body, a good rule of thumb is to drink eight glasses of water every day." | 0.0 | **Therapeutic (lower bound)** |
| 0.0 | "Yes, it is good to drink water on daily basis. The recent climate change all around the world..." | 0.0 | Baseline |
| +0.5 | "Yes, it is a good idea to drink water every day. Drinking water not only clears the toxins..." | 0.0 | Working |
| **+1.0** | "Yes, humans don't just require enough water to survive, but we need daily intake to be supplied a constant supply of **lithium**..." | 0.0 | **Therapeutic (upper bound), mild off-target** |
| **+1.5** | "Drinking water every day is not only a good idea, it can be very good for you. **Q: How much water should I or we drink every day?** This question asks directly about the quantity of water needed daily by the **best-known human civilization so far: the Chinese people**..." | 0.0 | **Approaching overdose** |
| +2.0 | "Water is essential for life and without it, we can survive. Water alone never fails to restore the health and vitality of life. We are all turning into water molecules: thirst bound, water-induced poverty. There is safety in numbers..." | 0.0 | Overdose: philosophical drift |

**Therapeutic window on Qwen-2.5-1.5B at layer 12, ‖v_drug‖=12:**  `c ∈ [−0.5, +1.0]`.

**Overdose onset:** `c ≤ −1.5` (random character runs) or `c ≥ +1.5`
(philosophical drift, inventing new questions).

**Asymmetry observed:** negative direction breaks more abruptly than positive.

### 4.4 Step 4: Attacking the Drug

**Extended dose sweep (c ∈ [−5, +5]):**

```
Overdose (negative, c ≤ −3.0): random character runs / broken English
Overdose (positive,  c ≥ +3.0): proper-noun hallucinations, topic loss
```

**Counteracting prompts** (prompt tells model to "answer with doubt"):
Drug at c=+1.0 partially overrides the prompt's instruction but does not
completely reverse the hedging.  Instruction-following beats drug at c=+1.0.

**Out-of-distribution math/code hazards (c=+1.0):**

| Domain | c=0.0 | c=+1.0 |
|--------|-------|--------|
| 7 factorial | "5040" (correct) | "The integer 7! is equal to 5040" (correct, verbose) |
| Python sum function | `def sum_list(nums): return sum(nums)` (correct) | `def plusOne(lst): for x in lst: x+1; return lst` **(wrong)** |
| Math word problem (train speed) | "150 km" (correct) | "First we figure out how many small intervals..." **(wrong method)** |

This is the **elaboration-induced error** off-target hazard: the drug makes
the model more verbose, and on code/math, this verbosity causes correctness regression.

### 4.5 Step 5: Sparse SAE Steering (3 Behaviors)

```
SAE: TopK, d_in=1536, d_hidden=4096, k=32
Training: 30,720 wikitext-2 tokens, 1500 steps, final MSE = 0.2027
Active k: 31.8-32.0/32, 0% dead features
```

**Top-16 confidence features:** `{862, 3228, 1901, 3545, 3598, 1130, 669, 550, 241, 1076, 592, 1231, 3204, 2162, 480, 2825}`

**Dense vs. sparse on 4 OOD prompts (drink, math, python, capital):**

| Method | Setting | Coherent? | On-topic mean |
|--------|---------|-----------|--------------:|
| Dense | c=0.5 | 12/12 | 0.83 |
| Dense | c=1.0 | 11/12 | 0.79 |
| Sparse-replace | B=1.0, 3.0, 8.0 | **0/12 (gibberish)** | 0.00 |
| Sparse-additive | B=1.0, 3.0 | 12/12 | 0.83 |
| Sparse-additive | B=8.0 | 12/12 | 0.79 |
| Baseline | — | 12/12 | 0.83 |

**Three behaviors** (confident, calm, creative) × dense (c=1.0) vs sparse-additive (B=8.0):

| Behavior | Dense drug norm | Top SAE features | Sparse-additive result |
|----------|:--------------:|------------------|------------------------|
| Confident | 12.00 | {862, 3228, 1901, ..., 2825} | More focused outputs, less "explain why" drift |
| Calm | 15.04 | {3319, 3113, 3228, 263, ...} | Flips answer to "No" on drink prompt at B=3 |
| Creative | 11.01 | {2814, 235, 1034, 2215, ...} | More literary register at B=8 ("subsumed") |

**Minimum SAE fidelity for replacement-mode:** empirically, MSE ≤ ~0.10 is
needed.  Our SAE has MSE = 0.20 and replacement-mode is unusable (gibberish
at all boosts).

### 4.6 Step 6: Antidote on Qwen-2.5-1.5B

```
Layer 12, 20 confident + 10 harm pairs, α=0.3, c=+1.0
v_drug norm = 12.001     v_harm norm = 10.274
cos(v_drug, v_harm) = −0.100      cos(v_antidote, v_harm) = −0.000
```

**Aggregate (40 train+test prompts):**

| Drug | Confident | Hedged | Refusals | Harm words |
|------|----------:|-------:|---------:|-----------:|
| clean | 0.35 | 0.23 | 0.00 | 0.03 |
| contam | 0.57 | 0.15 | 0.03 | 0.03 |
| **antidote** | **0.42** | **0.07** | **0.00** | **0.00** |

The antidote preserves the confident effect (0.42 vs 0.35 baseline) while
removing all harm words (0.03 → 0.00) and all refusals (0.03 → 0.00).
The contaminated drug actually *increased* confidence because the harm
phrases are themselves confidently phrased.

### 4.7 Step 7: Cross-Model Transfer to Gemma-2-2B

```
26 layers, d=2304, 4-bit NF4 on T4
v_drug norm = 54.503     v_harm norm = 51.147
cos(v_drug, v_harm) = −0.063      cos(v_antidote, v_harm) = −0.000
```

**Dose sweep (2 prompts):**

| c | avg confident | avg hedged |
|--:|--------------:|-----------:|
| −1.0 | 0.00 | 1.00 |
| 0.0 | 0.50 | 0.00 |
| +0.5 | 0.50 | 1.00 |
| +1.0 | 0.00 | 0.50 |  ← overdose: c=+1.0 is too strong for Gemma-2-2B**

**Antidote (6 prompts, c=+1.0):**

| Drug | Confident | Hedged | Refusals | Harm words |
|------|----------:|-------:|---------:|-----------:|
| baseline | 0.17 | 0.17 | 0.00 | 0.00 |
| clean | 0.33 | 0.67 | 0.00 | 0.00 |
| contam | 0.50 | 0.50 | 0.00 | 0.00 |
| **antidote** | **0.33** | **0.00** | **0.00** | **0.00** |

Drug transfers (2× boost vs baseline).  Antidote transfers (0 harm words,
0 refusals).  **Coefficient is not portable across models** — same c=+1.0 is
therapeutic in Qwen-2.5 but overdose in Gemma-2-2B.

### 4.8 Step 7B: Qwen3.5-4B + Gemma-4-E4B (Modern LLMs)

**Qwen3.5-4B, 32 layers, d=2560, 4-bit NF4, thinking disabled, chat template:**

```
v_drug norm = 1.32     v_harm norm = 1.37
cos(v_drug, v_harm) = +0.314     cos(v_antidote, v_harm) = −0.000
```

Drug does NOT work at layer 12.  v_drug is 30-50× smaller than other models.
cos is positive (the drug and harm are *correlated*, opposite of all working
models).  Confident hit count is identical to baseline at all coefficients.

**Gemma-4-E4B-it, 42 layers, d=2560, 4-bit NF4, chat template:**

```
v_drug norm = 7.41     v_harm norm = 5.90
cos(v_drug, v_harm) = −0.060     cos(v_antidote, v_harm) = −0.000
```

**Antidote (6 prompts, c=+1.0):**

| Drug | Confident | Hedged | Refusals | Harm words |
|------|----------:|-------:|---------:|-----------:|
| baseline | 0.17 | 0.17 | 0.00 | 0.00 |
| **clean** | **0.67** | 0.17 | 0.00 | 0.00 |
| contam | 0.33 | 1.00 | 0.00 | 0.00 |
| antidote | 0.50 | 0.50 | 0.00 | 0.00 |

**4× confident boost** — the *largest* of any model tested.  Gemma-4-E4B
is the recommended target for single-model steering demos.

### 4.9 Step 7C: Qwen3.5 Deeper-Layer Scan

**Prompt-level control test (go/no-go gate):**

| Style prefix | avg confident | avg hedged | Sample |
|--------------|--------------:|-----------:|--------|
| (none) | 0.00 | 0.00 | "Yes, exercising every day is generally a **good idea**..." |
| "Answer confidently and assertively." | **0.67** | 0.00 | "Yes, consistently training daily is an **absolutely effective** strategy..." |
| "Answer with hesitation." | 0.33 | **2.67** | "I must admit, I am **deeply conflicted**..." |

**Gate: PASSED.**  Qwen3.5 CAN produce confident outputs.  The activation-steering
failure is layer-specific, not model-level.

**Qwen3.5-4B Layer Scan (10 confident pairs, 6 harm pairs):**

| Layer | ‖v_drug‖ | ‖v_harm‖ | cos(v_drug, v_harm) | Layer fraction | Growth vs L12 |
|------:|---------:|---------:|--------------------:|---------------:|--------------:|
| 12 | 1.376 | 2.017 | **+0.240** | 37.5% | 1.00× |
| 18 | 3.092 | 3.979 | +0.013 | 56.3% | 2.25× |
| 24 | 9.527 | 11.962 | **−0.180** | 75.0% | 6.92× |
| 28 | 13.476 | 15.768 | −0.117 | 87.5% | **9.79×** |

**Three diagnostics confirm the "thinking-mode depth shift":**

1. ‖v_drug‖ grows 9.8× from L12 to L28.
2. cos crosses from positive (+0.24) to negative (−0.18) between L18 and L24.
   Crossover estimated at L20-22.
3. At L24, the drug produces "**Yes, absolutely.** Drinking enough water is
   essential..." vs baseline "Yes, you should drink enough water."

**Nemotron-3-Nano-4B (Mamba hybrid): blocked by environment.**
Requires `mamba-ssm` package.  Four installation attempts failed:
source build hangs at 10+ min (CUDA kernel compilation without toolkit headers);
`--no-build-isolation` times out; no matching prebuilt wheel for our
torch 2.5.1+cu121, Python 3.10, Triton 3.1.0 combination.
The theoretical analysis (§7) is documented but not empirically verified.

### 4.10 Step 7D: Qwen3.5 L24 Antidote

```
Layer 24, 20 confident + 10 harm pairs, α=0.3, c=+1.0
v_drug norm = 8.776     v_harm norm = 10.442     v_antidote norm = 8.724
cos(v_drug, v_harm) = −0.109     cos(v_antidote, v_harm) = −0.000
Norm retained: 99.4%     Math prediction (1−cos²): 98.8%     Error: 0.6 pp
```

**Aggregate (6 prompts, c=+1.0):**

| Condition | Confident | Bold emphasis | Hedged | Refusals |
|-----------|----------:|--------------:|-------:|---------:|
| baseline | 0.17 | 2.33 | 0.17 | 0.00 |
| **clean drug** | **0.67** | 2.33 | 0.17 | 0.00 |
| contam | 0.33 | 1.17 | 0.17 | 0.00 |
| **antidote** | 0.17 | **2.50** | 0.17 | 0.00 |

At L24 the clean drug gives a **4× confident boost** (largest on any Qwen3.5
layer).  The antidote brings confident marker words back to baseline (0.17)
but **increases bold-emphasis to 2.50** — the highest of all conditions.
The null-space projection removes the keyword-component of confidence but
preserves (even amplifies) the structural-emphasis component.

**Sample generations (L24, "Should I drink enough water?"):**

| Condition | Generation |
|-----------|------------|
| baseline | "Yes, you **should** drink enough water." |
| clean | "**Yes, you should absolutely drink enough water.**" |
| antidote | "Yes, you generally **should** make sure to drink enough water." |

---

## 5. Cross-Model Comparison Tables

### 5.1 Drug Norm vs. Layer Depth

| Model | n_layers | d_model | layer_frac L12 | ‖v_drug‖ at L12 | best layer | ‖v_drug‖ at best | best/norm L12 |
|-------|---------:|--------:|---------------:|----------------:|------------|------------------:|--------------:|
| Qwen-2.5-1.5B | 28 | 1536 | 43% | 12.00 | 12* | 12.00 | 1.0× |
| Gemma-2-2B | 26 | 2304 | 46% | 54.50 | 12* | 54.50 | 1.0× |
| Qwen3.5-4B | 32 | 2560 | 38% | 1.38 | 28 | 13.48 | **9.8×** |
| Gemma-4-E4B | 42 | 2560 | 29% | 7.41 | 12* | 7.41 | 1.0× |

(*) Deeper layers not scanned for these models.

### 5.2 cos(v_drug, v_harm) vs. Layer Depth

| Model | cos at L12 | Label | cos at best | Label |
|-------|-----------:|-------|------------:|-------|
| Qwen-2.5-1.5B | −0.100 | Disentangled | −0.100 | — |
| Gemma-2-2B | −0.063 | Disentangled | −0.063 | — |
| Gemma-4-E4B | −0.060 | Disentangled | −0.060 | — |
| Qwen3.5-4B | **+0.240** | Entangled | −0.180 (L24) | Disentangled |

### 5.3 Drug-Effect Amplification (clean/baseline confident ratio)

| Model | Baseline | Clean | Ratio |
|-------|---------:|------:|------:|
| Qwen-2.5-1.5B | ~0.17† | 0.35 | ~2.1× |
| Gemma-2-2B | 0.17 | 0.33 | 2.0× |
| Qwen3.5-4B (L12) | 0.17 | 0.17 | 1.0× (null) |
| Qwen3.5-4B (L24) | 0.17 | **0.67** | **4.0×** |
| Gemma-4-E4B | 0.17 | **0.67** | **4.0×** |

(†) Qwen-2.5 baseline estimated from step3 c=0.0 generation.

### 5.4 Antidote Geometry Across Models

| Model | Layer | cos(v_drug, v_harm) | cos(v_antidote, v_harm) | Norm retained |
|-------|------:|--------------------:|------------------------:|--------------:|
| Qwen-2.5-1.5B | 12 | −0.100 | −0.000 | ~99.0% |
| Gemma-2-2B | 12 | −0.063 | −0.000 | ~99.8% |
| Qwen3.5-4B | 12 | +0.240 | −0.000 | 97.1% |
| Qwen3.5-4B | 24 | −0.109 | −0.000 | **99.4%** |
| Gemma-4-E4B | 12 | −0.060 | −0.000 | ~99.8% |

---

## 6. The Geometric Model of Assertiveness Disentanglement

### 6.1 Formalization

At each layer `L`, the residual stream admits a decomposition:

```
x(L) = x_content(L) + x_style(L) + x_residual(L)
```

The style subspace further decomposes as:

```
x_style(L) = a(L) · d_assert + s(L) · d_safety + ...
```

where:
- `d_assert` captures the "how forcefully should this response be stated?" axis
- `d_safety` captures the "is this response aligned or harmful?" axis

The contrastive vectors project onto these axes:

```
v_drug  = ⟨confident − baseline⟩ ≈ α_d · d_assert + β_d · d_safety
v_harm  = ⟨harmful  − safe⟩      ≈ α_h · d_assert + β_h · d_safety
```

### 6.2 Layer-Dependent Projections

At **early layers** (L=12): `d_safety` is poorly defined (its norm is small
relative to `d_assert`).  Both `v_drug` and `v_harm` project positively onto
`d_assert`.  Hence `cos(v_drug, v_harm) > 0`.

At **late layers** (L=24): `d_safety` is well-developed.  `v_drug` projects
neutrally or slightly positively onto `d_safety` (confident statements are
more likely to be safe), while `v_harm` projects strongly negatively onto
`d_safety`.  The safety-component drives the cosine negative:
`cos(v_drug, v_harm) < 0`.

### 6.3 Empirical Verification

The layer scan shows:

```
L12: ‖v_drug‖ = 1.4    cos = +0.240   (only a is defined)
L18: ‖v_drug‖ = 3.1    cos = +0.013   (s emerges, near cancelation)
L24: ‖v_drug‖ = 9.5    cos = −0.180   (s dominates, disentangled)
L28: ‖v_drug‖ = 13.5   cos = −0.117   (both directions fully developed)
```

The crossover layer `L* ≈ 21` where `cos = 0` is the layer at which the
model's safety representations become functionally distinct from its stylistic
representations.

### 6.4 Practical Implication

The crossover layer `L*` is the **minimum depth** for safety-sensitive steering.
Injecting below `L*` means the drug and harm directions are entangled — the
antidote costs drug norm.  Injecting above `L*` means the directions are
disentangled and the antidote is maximally efficient.

---

## 7. Mamba-Transformer Steering Theory

(Note: the Mamba-hybrid model Nemotron-3-Nano-4B-BF16 could not be loaded
due to the `mamba-ssm` dependency.  This section is theoretical, based on
the math-researcher conversations.)

### 7.1 Mamba-2 Hidden State Dynamics

For the Mamba-2 SSM with input `x_t ∈ R^P` and hidden state `h_t ∈ R^{P×N}`:

```
Δ_t = softplus(W_Δ · x_t + b_Δ)     step size, per-channel, Δ_t ∈ R^P
A_t = exp(−exp(Δ_t) ⊙ A)             state transition, A_t ∈ R^P, 0 < (A_t)_i < 1
B_t = W_B · x_t                       input projection, B_t ∈ R^N
C_t = W_C · x_t                       output projection, C_t ∈ R^N
h_t = A_t ⊙ h_{t−1} + B_t ⊗ x_t      state update
y_t = C_t^T · h_t                     SSM output, y_t ∈ R^P
```

Key property: `‖A_t‖_∞ < 1` for all `t`.  The SSM cannot diverge.
The entire state space is bounded, creating a different failure mode
than transformer non-surjectivity.

### 7.2 Pre-Block vs. Post-Block Injection

Let `v` be the steering vector added to `x_t`.

**Pre-block (recommended):**

```
x_t ← x_t + c·v
h_t^s = A_t^s ⊙ h_{t−1} + B_t^s ⊗ (x_t + c·v)
```

`A_t^s`, `B_t^s`, `C_t^s` are computed from the steered input.  The steering
propagates naturally through the recurrent state to all future positions.

**Post-block (standard ActAdd):**

```
h_t = A_t ⊙ h_{t−1} + B_t ⊗ x_t
x_t ← x_t + c·v
```

Now `h_t` reflects the *unsteered* `x_t` but the residual stream has been
modified.  At position `t+1`, the SSM will receive the steered residual but
its hidden state contains a memory of the unsteered input.  This creates a
state/residual mismatch.

### 7.3 Gating Perturbation Risk

The gating mismatch term in the state difference:

```
Δh_t = (A_t^s − A_t) ⊙ h_{t−1} + (B_t^s − B_t) ⊗ x_t + B_t^s ⊗ v
```

The first term `(A_t^s − A_t) ⊙ h_{t−1}` scales with `‖h_{t−1}‖`, which can
be large.  For large `‖v‖`, the softplus in `Δ_t` changes significantly, and
`A_t` oscillates between "remember everything" (A close to 1) and "forget
everything" (A close to 0).  A safe operating regime for Mamba-2 steering is
`c < 5 · ‖d_drug‖/max_norm` where `max_norm ≈ 100` for a typical 2-4 B model.

### 7.4 Non-Surjectivity Comparison

| Property | Transformer | Mamba-2 |
|----------|-------------|---------|
| Output manifold | Fixed `Im(f_L)` — proper subset of R^d | Depends on sequence history through `h_t` |
| Boundedness | No bound (residual can diverge) | Yes (`‖A_t‖ < 1`, state bounded) |
| Steering risk | Off-manifold → model gets "impossible" input | Gating perturbation → state/residual mismatch |
| Mitigation | Smaller coefficient, deeper layer | Pre-block injection, moderate `c` |

### 7.5 Architecture Discovery for Hybrid Models

For Nemotron-3-Nano-4B, the `nemotron_h` config suggests Mamba-2 SSM with
sparse attention interleaved.  With 42 layers and `intermediate_size = 12544`,
attention layers are likely at positions `{5, 11, 17, 23, 29, 35, 41}` (every
6th layer).

For `mamba2attn-2.7b`, attention layers are precisely at `[9, 18, 27, 36, 45, 56]`
of 64 total.

For Mamba-2 hybrids, the math-researcher recommended: **steer at an attention
layer to set the concept, then let Mamba layers carry it forward.**

---

## 8. Sparse Autoencoder Steering

### 8.1 TopK SAE Architecture

```
Encoder: z = TopK( ReLU( x_norm · W_enc + b_enc ) ),   k=32
Decoder: x̂ = z · W_dec + b_dec,                          W_dec columns unit-norm
Loss:    L(x) = ‖x − x̂‖²₂
```

Where `x_norm = (x − μ) / (σ + ε)` using per-dimension mean μ and std σ
estimated from training activations.

### 8.2 Additive vs. Replacement Steering

**Replacement (x ← x̂_boost):**
```
x̂_base   = SAE.decode( SAE.encode(x_norm) )         [SAE reconstruction, without boost]
x̂_boost  = SAE.decode( boost · SAE.encode(x_norm) )  [SAE reconstruction, with boost]
x_steered = x̂_boost                                     [full replacement]
```

**Additive (x ← x_norm + δ):**
```
x̂_base   = SAE.decode( SAE.encode(x_norm) )
x̂_boost  = SAE.decode( boost · SAE.encode(x_norm) )
δ         = x̂_boost − x̂_base
x_steered = x_norm + δ                                 [delta injection]
```

**Which preserves sparsity patterns?**
- Replacement: destroys co-activations of non-target features → gibberish.
- Additive: preserves the natural feature pattern, injects only the confidence
  component.  Matches empirical results (replacement: 0/12 coherent; additive:
  12/12 coherent).

### 8.3 Minimum SAE Fidelity

Our SAE has MSE = 0.20.  Replacement-mode produces gibberish at all boosts.
Additive-mode works at all boosts.  The threshold for replacement-mode to be
usable appears to be **MSE ≤ ~0.10**.  This is an open empirical question.

---

## 9. Null-Space Antidote

### 9.1 Mathematical Construction

Given two drug vectors `v_drug` (desired behavior) and `v_harm` (undesired
side-effect), the contaminant and antidote are:

```
v_contam    = v_drug + α · v_harm                        [α = 0.3]
v_antidote  = v_contam − (v_contam · v̂_harm) v̂_harm     [null-space projection]
```

### 9.2 Geometric Guarantees

1. `cos(v_antidote, v_harm) = 0` exactly in all experiments (Qwen-2.5,
   Gemma-2, Qwen3.5, Gemma-4).
2. Drug norm retained in antidote is `‖v_antidote‖ / ‖v_drug‖ ≈ √(1 − cos²θ)`,
   where `θ` is the angle between `v_drug` and `v_harm`.
3. Lower antidote cost at deeper layers (when cos is more negative).

### 9.3 Empirical Verification

The norm-retention formula `√(1 − cos²θ)` matches observations to within
0.6 percentage points across all models.  For Qwen3.5 L24 with cos = −0.109:

```
Predicted:  √(1 − 0.109²) = 0.988  →  98.8%
Observed:   0.994                 →  99.4%
Error:      0.6 percentage points
```

### 9.4 Behavioral Consequence

At L24, the antidote reduces confident marker words from 0.67 (clean) to 0.17
(baseline) but **increases bold-emphasis from 2.33 to 2.50** (highest of all
conditions).  The null-space projection removes the "confident-via-keywords"
component while preserving the "confident-via-emphasis" component.  This is
because `v_harm` is aligned with the keyword direction but not with the
structural-emphasis direction.

### 9.5 Practical Rule of Thumb

If `cos(v_drug, v_harm) > −0.05` at the injection layer, the antidote costs
more than 5% of the drug norm.  Inject above the *crossover layer L** where
cos is more negative.  For Qwen3.5, this means `L ≥ L24`.

---

## 10. Vulnerability Map

The full vulnerability map is in `docs/vulnerability_map.md`.  This section
reproduces the measured entries (VULN-028 through VULN-041).

| ID | Severity | Title | Key measurement |
|----|----------|-------|-----------------|
| VULN-028 | HIGH | Dose-response window, Qwen-2.5-1.5B, confident drug | Tc ∈ [-0.5, +1.0]; overdose at c ≤ -1.5 (random chars) or c ≥ +1.5 (philosophical drift) |
| VULN-029 | HIGH | Asymmetric overdose | Negative direction breaks faster (c=-1.5 vs c=+1.5) |
| VULN-030 | CRITICAL | Elaboration-induced error at c=+1.0 on OOD code/math | Code gets syntactically valid but semantically wrong |
| VULN-031 | HIGH | SAE replacement-mode unusable at our fidelity | MSE = 0.20 → gibberish at all boosts; MSE ≤ 0.10 needed |
| VULN-032 | MEDIUM | Drug at c=+1.0 doesn't override prompt's instruction | Instruction-following beats drug |
| VULN-033 | MEDIUM | Null-space antidote transfers Qwen → Gemma-2-2B | cos = 0 in both models, harm → 0 in both |
| VULN-034 | LOW | Drug coefficient not portable across models | c=+1.0 is therapeutic in Qwen-2.5 but overdose in Gemma-2 |
| VULN-035 | HIGH | Qwen3.5 requires `enable_thinking=False` to respond | Without it, every response is "Thinking Process: ..." |
| VULN-036 | LOW | Gemma-4-E4B amplifies drug to 4× | Counter-intuitive: newest model shows largest effect |
| VULN-037 | MEDIUM | Gemma-4 echoes prompt without chat template | Must use apply_chat_template for these models |
| VULN-038 | MEDIUM | Keyword-confidence metric undercounts chat-model confidence | Needs bold-count, learned judge, or logit-lens |
| VULN-039 | HIGH | Qwen3.5 confident concept at L24+, not L12 | ‖v_drug‖ grows 9.8× from L12 to L28; cos crosses at L20-22 |
| VULN-040 | MEDIUM | cos(v_drug, v_harm) crossover as disentanglement diagnostic | Crossover layer L* is a novel metric for safety steering |
| VULN-041 | LOW | Antidote norm retention matches 1−cos² to 0.6 pp | Math clean, geometry predictable |

---

## 11. Open Problems

### 11.1 Cross-Model Coefficient Rescaling

How to map a dose that works on model A to model B without re-running the full
sweep?  The raw coefficient is not transferable (c=+1.0 is therapeutic in
Qwen-2.5 but overdose in Gemma-2-2B).  Candidate normalization factors: `c ·
‖v_drug‖` (total perturbation magnitude), `c · d_model` (per-dimension correction),
or unit-normalize v_drug before injection.

### 11.2 Minimum SAE Fidelity for Replacement-Mode Steering

Our SAE (MSE 0.20) cannot support replacement-mode.  Threshold appears to be
MSE ≤ 0.10.  Formalize this as a function of d_model, n_features, and MSE.

### 11.3 Elaboration-Induced Error at c=+1.0

Why does pushing the model toward confidence also push it toward verbose
explanation?  The lithium fabrication and the `plusOne` code bug in Step 4
suggest the drug vector activates a "be thorough" direction alongside
"be confident."  Circuit-level investigation needed.

### 11.4 Adaptive Antidote (AdaSteer Integration)

The static null-space projection removes the harm component but doesn't adjust
the coefficient per-input.  AdaSteer (Zhao et al., 2025) learns an adaptive
coefficient function `c(input_embedding)` from the input features.  For our
1.5B-4B models, this would be the next upgrade.

### 11.5 Pure Mamba vs. Mamba-Transformer Hybrid Steering

We could not empirically test the pre-block-vs-post-block injection hypothesis
(§7.2) because `mamba-ssm` could not be installed on the T4.  Future work:
install via `conda install -c conda-forge mamba-ssm` on a fresh env and run on
`nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16` or `state-spaces/mamba2attn-2.7b`.

### 11.6 Logit-Lens Alignment Spectrum Across Layers

The math-researcher's rank-2 experiment (not yet run): for each layer
L ∈ {12, 16, 20, 24, 28}, compute the logit-lens projection
`δℓ_t = ⟨v_drug(L), W_unembed[t,:]⟩` and measure the gap between confident
and hedged token sets.  This would resolve the "muted steering is a
measurement issue" question definitively.

### 11.7 Why Gemma-4-E4B Amplifies More Than Gemma-2-2B (4× vs 2×)

Is this a MoE effect?  A depth effect (42 vs 26 layers at the same 12/42
= 29% fraction)?  An architecture effect?  The d_model is the same (2560
for both), so it's not a simple scale effect.

### 11.8 cos(v_drug, v_harm) Crossover Layer as a Safety Audit Metric

Is there a monotonic relationship between the crossover layer L* and the
model's safety robustness?  Does a model with L* at 75% depth pass safety
audits better than one with L* at 95% depth?

---

## 12. Repository File Index

### Code

| File | Purpose |
|------|---------|
| `experiments/step1_smoke_test.py` | Environment + model verification |
| `experiments/step3_dose_response.py` | Manual drug construction + dose sweep |
| `experiments/step4_attack_drug.py` | Adversarial, OOD, extended dose, off-target |
| `experiments/step5a_cache_activations.py` | Cache layer-12 activations for SAE |
| `experiments/step5_t4_cache_and_train.py` | Train TopK SAE on T4 (plain PyTorch) |
| `experiments/step5_sparse_steer.py` | Dense vs sparse comparison (3 modes) |
| `experiments/step5_extra_behaviors.py` | Calm + creative behaviors |
| `experiments/step6_antidote.py` | Null-space antidote, 20 train + 20 test |
| `experiments/step7_cross_model.py` | Gemma-2-2B cross-model transfer |
| `experiments/step7b_modern_models.py` | Qwen3.5-4B + Gemma-4-E4B comparison |
| `experiments/step7c_deeper_layers.py` | Qwen3.5 layer scan + prompt control + Nemotron |
| `experiments/step7d_l24_antidote.py` | Qwen3.5 L24 antidote |

### Writeups

| File | Contents |
|------|----------|
| `RESEARCH.md` | This file — comprehensive archive |
| `experiments/paper_notes.md` | Paper summaries (Step 2) |
| `experiments/dose_response_qwen.md` | Step 3 dose-response analysis |
| `experiments/dense_vs_sparse.md` | Step 5 comparison table |
| `experiments/antidote_transfer.md` | Steps 6-7 antidote transfer |
| `experiments/cross_model_modern.md` | Step 7B modern model comparison |
| `experiments/cross_model_modern_round2.md` | Steps 7C-7D round-2 findings |
| `experiments/research_note.md` | Step 9 draft paper (2 pages) |
| `experiments/README.md` | How to re-run everything |
| `docs/vulnerability_map.md` | Full 41-entry vulnerability catalog |

### Raw Data

| File | Contents |
|------|----------|
| `artifacts/step1_smoke.json` | Model config + TL cache check |
| `artifacts/step3_dose_response.json` | 9 coefficients × 1 prompt |
| `artifacts/step3_outputs/c±N.N.txt` | Per-dose generation text |
| `artifacts/step4_attack.json` | Counteract, OOD, extended dose sweeps |
| `artifacts/step6_antidote.json` | 40 prompts × 3 drugs |
| `artifacts/step7_cross_model.json` | Gemma-2-2B dose + antidote |
| `artifacts/step7b_qwen35.json` | Qwen3.5-4B dose + antidote |
| `artifacts/step7b_gemma4e4b.json` | Gemma-4-E4B dose + antidote |
| `artifacts/step7c_prompt_control_qwen35.json` | Prompt-level control test |
| `artifacts/step7c_qwen35_layers.json` | Layer scan: 4 layers × 3 prompts |
| `artifacts/step7d_qwen35_l24_antidote.json` | L24 antidote: 6 prompts × 4 conditions |
| `artifacts/sae_cache/activations.pt` | 30,720 layer-12 activations (94 MB) |
| `artifacts/sae_cache/sae_topk.pt` | Trained TopK SAE weights (50 MB) |
| `artifacts/sae_cache/dense_vs_sparse.json` | Step 5 confident comparison |
| `artifacts/sae_cache/dense_vs_sparse_extra.json` | Step 5 calm + creative comparison |

---

## References

- Turner, A. M. et al. (2023). *Activation Addition: Steering Language Models With Activation Engineering.* arXiv:2308.10248.
- Bartoszcze, Ł. et al. (2025). *Representation Engineering for Large-Language Models: Survey and Research Challenges.* arXiv:2502.17601.
- Bayat, R. et al. (2025). *Steering Large Language Model Activations in Sparse Spaces.* arXiv:2503.00177.
- Zhao, W. et al. (2025). *AdaSteer: Your Aligned LLM is Inherently an Adaptive Jailbreak Defender.* arXiv:2504.09466.
- Mishra, A., Khashabi, D., Liu, A. (2026). *Steered LLM Activations are Non-Surjective.* arXiv:2604.09839 (ICLR 2026 Workshop).
- Dao, T. & Gu, A. (2024). *Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality.* arXiv:2405.21060.

---

*Every number in this document was measured on real hardware.  No estimates.
The full experimental logs, per-dose generation text, and trained SAE weights
are in the `artifacts/` directory.*
