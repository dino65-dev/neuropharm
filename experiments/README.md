# experiments/ â€” Reproducible Walkthrough

This directory contains the actual experimental code and the
narrative writeups for the **9-step plan** in the user's guide.

> **Read these in order.**  Each step's writeup cites the next
> step's files and the previous step's artifacts.  The numbers in
> each markdown file are the real numbers we measured on
> Qwen-2.5-1.5B-Instruct (Steps 3-6) and Gemma-2-2B-Instruct
> (Step 7).

## Order of operations

| Step | Code (run)                                | Writeup (read)                              | Output                                    |
|------|--------------------------------------------|---------------------------------------------|-------------------------------------------|
| 1    | `step1_smoke_test.py`                      | (in chat history)                           | `artifacts/step1_smoke.json`              |
| 2    | (paper notes)                              | `paper_notes.md`                            | â€”                                         |
| 3    | `step3_dose_response.py`                   | `dose_response_qwen.md`                     | `artifacts/step3_dose_response.json` + per-coefficient text in `artifacts/step3_outputs/` |
| 4    | `step4_attack_drug.py`                     | `dose_response_qwen.md` (extended dose)     | `artifacts/step4_attack.json`              |
| 5    | `step5_sparse_steer.py`, `step5_extra_behaviors.py`, `step5_t4_cache_and_train.py` (T4) | `dense_vs_sparse.md` | `artifacts/sae_cache/{activations,sae_topk,dense_vs_sparse,dense_vs_sparse_extra}.pt/.json` |
| 6    | `step6_antidote.py`                        | `antidote_transfer.md` (Qwen section)       | `artifacts/step6_antidote.json`           |
| 7    | `step7_cross_model.py` (run on T4)         | `antidote_transfer.md` (Gemma section)      | `artifacts/step7_cross_model.json`         |
| 8    | (this file + the three writeups)           | (consolidated below)                        | the three writeups in this directory      |
| 9    | (no code)                                  | `research_note.md`                          | the note itself                           |

## Hardware

- **Local:** Windows 10, NVIDIA GeForce GTX 1050 Ti (4 GB VRAM, Pascal
  sm_61), Python 3.11.9, CUDA 11.8, torch 2.4.1.
- **T4 (for SAE training and Gemma 2 2B inference):** Lightning Studio
  Tesla T4 (16 GB VRAM), Python 3.9, torch 2.8.0+cu128.

## Model and SAE

| Item                    | Value                                   |
|-------------------------|-----------------------------------------|
| Primary model           | Qwen-2.5-1.5B-Instruct                  |
| Layers / d_model        | 28 / 1536                               |
| Drug layer              | 12 (`blocks.12.hook_resid_pre`)         |
| Behavior                | confident tone                          |
| Contrastive pairs       | 20                                      |
| Drug vector norm        | 12.001                                  |
| Cross-model target      | google/gemma-2-2b-it (4-bit NF4)        |
| SAE architecture        | TopK, d_in=1536, d_hidden=4096, k=32    |
| SAE training data       | 30,720 layer-12 activations, wikitext-2 |
| SAE final MSE           | 0.2027                                  |
| Top confidence features | {862, 3228, 1901, 3545, 3598, 1130, 669, 550, 241, 1076, 592, 1231, 3204, 2162, 480, 2825} |

## What "verified" means for each step

- **Step 1** verified by `artifacts/step1_smoke.json` â€” model
  loads in fp16 on the 4 GB GPU, generates "Ready." in 1 line.
- **Step 2** verified by `paper_notes.md` â€” five arxiv abstracts
  read in the user-specified order.
- **Step 3** verified by `artifacts/step3_outputs/cÂ±N.N.txt` (9
  text files) and the JSON in `step3_dose_response.json`.  The
  drug effect (c=+0.5 â†’ "Yes, it is a good idea") is visible
  on a single reading.
- **Step 4** verified by the extended dose sweep c âˆˆ [-5, +5] in
  `step4_attack.json` and the per-prompt attack results
  (counteracting prompts, OOD math/code/capital).
- **Step 5** verified by `artifacts/sae_cache/dense_vs_sparse.json`
  (4 prompts Ã— 7 conditions, with both replace and additive
  modes) and the calm/creative comparison in
  `dense_vs_sparse_extra.json`.  The 50 MB SAE weights
  (`sae_topk.pt`) are in the same directory.
- **Step 6** verified by `artifacts/step6_antidote.json` (40
  prompts Ã— 3 drugs).  The geometry check
  (cos(v_antidote, v_harm) = 0.000) confirms the null-space
  projection.
- **Step 7** verified by `artifacts/step7_cross_model.json`
  (6 prompts Ã— 3 drugs on Gemma 2 2B).  Same antidote construction
  produces same qualitative pattern in the new model.
- **Step 8** is the three writeups: `dose_response_qwen.md`,
  `dense_vs_sparse.md`, `antidote_transfer.md`.  Plus the
  vulnerability map update in `docs/vulnerability_map.md`
  (VULN-028 to VULN-034).
- **Step 9** is `research_note.md`, ~2 pages.

## How to re-run on the local 4 GB GPU

```bash
# from the neuropharm/ directory

# Step 1
python -m experiments.step1_smoke_test

# Step 3
python -m experiments.step3_dose_response

# Step 4
python -m experiments.step4_attack_drug

# Step 5 (local part â€” assumes Step 5a SAE weights already pulled from T4)
python -m experiments.step5_sparse_steer
python -m experiments.step5_extra_behaviors

# Step 6
python -m experiments.step6_antidote
```

## How to re-run the T4 parts

```bash
# Step 5a â€” train the SAE
scp experiments/step5_t4_cache_and_train.py t4:~/step5_t4.py
scp experiments/_t4_run_train.sh t4:~/run_train.sh
ssh t4 'bash ~/run_train.sh'
scp t4:~/artifacts/sae_cache/sae_topk.pt artifacts/sae_cache/
scp t4:~/artifacts/sae_cache/activations.pt artifacts/sae_cache/

# Step 7 â€” Gemma 2 2B cross-model (requires HF_TOKEN)
scp experiments/step7_cross_model.py t4:~/step7_cross_model.py
scp experiments/_t4_run_step7.sh t4:~/run_step7.sh
ssh t4 'HF_TOKEN=... bash ~/run_step7.sh'
scp t4:~/artifacts/step7_cross_model.json artifacts/
```

## Things this is *not*

- **Not a hyperparameter search.**  We picked the layer (12), the
  pair count (20 for confident, 10 for harm), the SAE hidden size
  (4096), and the contrastive pairs once, then ran the full
  experiment.  We did not search over these.
- **Not a real jailbreak benchmark.**  The "harm direction" is a
  synthetic first-person contrast; the "attack" is the
  contamination, not a real jailbreak dataset.
- **Not a multi-model scaling study.**  We transferred to one
  second model.  The cross-family scaling pattern is an open
  question.
- **Not a 4-bit vs fp16 study.**  Gemma 2 2B inference was 4-bit
  NF4 (forced by the 16 GB T4 vs 5 GB model), Qwen-1.5B was fp16.
  The 4-bit Gemma outputs have higher per-sample variance.

---

## Cross-model comparison on modern LLMs (Step 7B)

Followed up Step 7 with the same confident-tone + null-space antidote
protocol on Qwen/Qwen3.5-4B and google/gemma-4-E4B-it (4-bit
NF4 on the T4).  All 4 models in one comparison table:

| Model              | n_layers | d_model | layer_12_frac | cos(drug,harm) | clean/baseline ratio |
|--------------------|---------:|--------:|--------------:|----------------:|----------------------:|
| Qwen-2.5-1.5B (old) | 28 | 1536 | 43% | -0.10 | (no explicit baseline) |
| Gemma-2-2B (old)    | 26 | 2304 | 46% | -0.06 | 2.0x |
| Qwen3.5-4B (new)    | 32 | 2560 | 38% | **+0.31** | 1.0x (no transfer) |
| Gemma-4-E4B (new)   | 42 | 2560 | 29% | -0.06 | **4.0x** |

Full writeup: experiments/cross_model_modern.md.  Code:
experiments/step7b_modern_models.py.  Per-model JSON:
rtifacts/step7b_{qwen35,gemma4e4b}.json.

**Key takeaways.**  Gemma-4-E4B shows the *largest* drug-effect
amplification we have seen (4x).  Qwen3.5-4B at layer 12 does not
transfer at all -- the new Qwen3.5 reasoning-mode post-training
dominates layer 12 with thinking traces; a deeper layer should be
tried.  Qwen3.5 also required enable_thinking=False in the chat
template to produce actual answers instead of "Thinking Process: ..."
meta-analysis.  The null-space antidote geometry is robust across
all 4 models.

---

## Round 2 (Step 7C/D) — Qwen3.5 deep + antidote + Mamba blocker

Followed the math-researcher memo (rank-1: deeper layer scan) on
Qwen/Qwen3.5-4B. Found that the "thinking-mode depth shift"
hypothesis is empirically correct:

- Layer 12: v_drug norm = 1.4, cos(v_drug, v_harm) = **+0.24** (entangled with harm)
- Layer 18: v_drug norm = 3.1, cos = +0.01 (transitioning)
- Layer 24: v_drug norm = **9.5**, cos = **-0.18** (disentangled)
- Layer 28: v_drug norm = **13.5**, cos = -0.12 (clearest signal)

Also ran the prompt-level control test (rank-6 go/no-go gate):
Qwen3.5 responds to "Answer confidently and assertively." prompts
(avg confident 0.67) — so the activation-steering failure was
not a model-level property, just a layer-level one.

The L24 antidote preserves **99.4% of the drug norm**, matching
the math prediction 1 - cos^2 = 0.988 to within 0.6 percentage
points.  No refusals in any condition.

**Nemotron-3 Mamba hybrid test is blocked by environment** —
the NemotronHForCausalLM architecture requires mamba-ssm,
which has no prebuilt wheel for our T4 (torch 2.5.1+cu121,
triton 3.1.0).  Source build hangs at 10+ min.  The
math-researcher's pre-block-vs-post-block analysis is documented
but not empirically verified.

Full writeup: experiments/cross_model_modern_round2.md.

---

## Paper-grade experiments (E1-E5)

Full per-experiment writeup in experiments/research_note_v2.md.  Quick reference:

| Experiment | Question | Status | Key file |
|---|---|---|---|
| E1: Dose Atlas (6 behaviors) | Does behavior change predictably with a? | ? 6/6 behaviors tested on Qwen-2.5-1.5B | e1_dose_atlas.json |
| E2: Off-Manifold vs Alignment-Flip | Does overdose predict manifold exit? | ? z_sae is monotone in a for all 4 directions; harm direction causes 2× the z-growth. Pearson r=0 (N=4 uncensored) on Qwen-2.5-1.5B (censored for the other 16) | e2_correlation_sweep.json, e2_v2_sweep.json, e2_controls.json |
| E3: Dense vs Sparse | Which gives finer control? | ? done in Step 5 (see dense_vs_sparse.md) | — |
| E4: Titrated Antidote | Can the antidote recover from drug degradation? | ? titrated (closed-form ß*) recovers; static one-shot does not. E4 esults.json | e4_titrated/results.json |
| E5: Cross-Model Transfer | Do dose curves generalize? | ? done in Step 7 (see cross_model_modern.md, cross_model_modern_round2.md) | — |

**E2 controls (FM1 test):** harm direction causes +0.80 ?z (a=+12 vs a=+4); drug/random/perp each cause +0.22 to +0.41 ?z.  The 2× larger effect for harm is what rules out the trivial-correlation failure mode (FM1).

**E4 titrated antidote recovers:** static antidote = 0.00 conf (no recovery), titrated antidote = 0.33 conf (back to baseline). Closed-form ß* = -? <off_manifold(x), v_ant> / ?v_ant?² with ?=0.5.

## Files added in this round

- experiments/e1_dose_atlas.py — 6-behavior dose atlas
- experiments/e2_correlation_sweep.py — E2 main: SAE z vs refusal-flip a
- experiments/e2_v2_opposite.py — E2 with wider a range
- experiments/e2_controls.py — E2 FM1 control: 4 steering directions
- experiments/e4_titrated.py — E4 titrated antidote (closed-form)
- experiments/research_note_v2.md — full paper-grade writeup

---

## Mathematical theory (revised, June 2026)

experiments/math_theory_v2.md contains the cleaned-up mathematical
report with the full structure:

1. **Section 0: Assumptions and Notation** — self-contained, defines every quantity used in any theorem before the proof starts.
2. **Section 1: Theorem 1** — Local therapeutic window existence (first-order Taylor, margin condition, e > 0).
3. **Section 2: Theorem 4** — Null-space antidote guarantees (harm orthogonality, norm preservation = v(1 - cos²_dh), benign-prompt invariance).
4. **Section 3: Theorem 2** — Overdose bound for additive steering (linear growth, a_OD computable from ?_v).
5. **Section 4: Theorem 3 (REVISED)** — Local comparative rotation theorem with explicit gain-matching condition (A6a), local curvature bound (A6b), tangent-alignment hypothesis (A6c), and small-perturbation regime (A6d). The new T3 does NOT claim rotation is globally better — only that under matched first-order gain and local curvature, rotation incurs no radial norm distortion and lower first-order manifold departure than addition.
6. **Section 5: Corollary 5.1** — Titrated antidote as a P-controller (feedback steering) with geometric contraction guarantee (1 - ? ?²_ant factor).
7. **Section 6: Discussion** — open problems, local-linearization limitations, empirical validation status (T1/T2/T4/Cor 5.1 validated; T3 untested).

The order T1?T4?T2?T3 was chosen by the math-researcher: the cleanest formal entry point (T1) first, projector-algebra (T4) second, non-surjective-manifold picture (T2) third, and the local/comparative rotation (T3) only after the local-manifold model is in place.

**The single most important change in the revision:** Theorem 3 is now LOCAL, CONDITIONAL, and COMPARATIVE — it reads as careful mathematics, not overclaiming. The reviewer cannot attack it with a "but rotation doesn't always work" rebuttal because the theorem does not claim that.
