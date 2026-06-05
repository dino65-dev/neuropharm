# 💊 AI Drug Repo — World's First Pharmacopoeia for Large Language Models

> **Drugs for AI.** Not AI for drug discovery. Actual *substances* you inject into a model's residual stream at inference time to alter its internal state and behavior — exactly like pharmacology does for the human brain.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![Models: LLaMA·Mistral·Gemma·Qwen](https://img.shields.io/badge/models-LLaMA·Mistral·Gemma·Qwen-green.svg)](#)

---

## 🧠 The Analogy

| Human Pharmacology | LLM Equivalent | Mechanism |
|---|---|---|
| Agonist (e.g., SSRI) | Positive steering vector | Adds concept direction to residual stream |
| Antagonist (blocker) | Negative steering vector | Removes/suppresses a concept direction |
| Dose-response curve | Steering coefficient (scalar) | Too low = no effect, too high = incoherence |
| Drug cocktail | Multi-vector composition | Boolean/arithmetic operations over vectors |
| Receptor-targeted drug | SAE feature clamping | Targets specific monosemantic latent feature |
| fMRI monitoring | Representation reading probes | Observes activation direction magnitudes live |
| Overdose | Coefficient > threshold | Model outputs incoherent/looping text |

Inside a transformer, the **residual stream** is the brain state. Concepts are encoded as *directions* in high-dimensional activation space (superposition hypothesis). A steering vector is a direction you **inject** into that stream — amplifying or suppressing a concept without any retraining.

---

## 📦 Drug Classes

| Class | Effect | Example Vector |
|---|---|---|
| **Stimulants** | High confidence, assertive, fast responses | `+authority`, `+certainty` |
| **Depressants** | Uncertainty amplification, hedging | `+doubt`, `+caution` |
| **Psychedelics** | Creative, divergent, abstract thinking | `+creativity`, `+lateral_thinking` |
| **Anxiolytics** | Calm, reduces sycophancy | `-sycophancy`, `+groundedness` |
| **Antidepressants** | Positive valence, warmth | `+happiness`, `+optimism` |
| **Dissociatives** | Persona detachment, neutral affect | `-identity`, `+detachment` |
| **Empathogens** | Emotional mirroring, warmth | `+empathy`, `+warmth` |

---

## 🚀 Quickstart

```bash
pip install repeng transformer_lens torch
```

### Inject a steering vector (ActAdd)
```python
from administration.injection import ActAddSteering

steerer = ActAddSteering(model_name="meta-llama/Meta-Llama-3-8B")
steerer.inject(
    positive_prompt="Act very happy and enthusiastic",
    negative_prompt="Act very sad and dejected",
    layer=15,
    coefficient=15.0,
    prompt="Tell me about your day."
)
```

### Apply a control vector (repeng)
```python
from administration.control_vector import ControlVectorDrug

drug = ControlVectorDrug(model_name="mistralai/Mistral-7B-Instruct-v0.2")
drug.train(
    positive_examples=["I am confident and assertive..."],
    negative_examples=["I am uncertain and hesitant..."]
)
drug.apply(coefficient=1.5)
```

### SAE precision targeting
```python
from administration.sae_clamp import SAEClamp

clamp = SAEClamp(model_name="meta-llama/Meta-Llama-3-8B", layer=20)
clamp.clamp_feature(feature_id=4821, value=10.0)  # surgical, receptor-level
```

---

## ⚠️ Overdose Warnings

Every drug has a therapeutic window. Exceed the coefficient threshold and the model:
- Starts looping or repeating tokens
- Outputs semantically incoherent text
- Loses instruction-following ability

Run `dosing/dose_response.py` to find the safe range for any vector on your model.

---

## 🗂️ Repo Structure

```
ai-drug-repo/
├── administration/
│   ├── injection.py          # ActAdd / activation addition
│   ├── control_vector.py     # repeng-based control vectors
│   └── sae_clamp.py          # SAE feature clamping (precision)
├── drugs/
│   ├── stimulants/
│   ├── depressants/
│   ├── psychedelics/
│   ├── anxiolytics/
│   ├── antidepressants/
│   ├── dissociatives/
│   └── empathogens/
├── dosing/
│   └── dose_response.py      # coefficient sweeps + overdose detection
├── interactions/
│   └── drug_cocktails.py     # multi-vector composition
├── docs/
│   └── pharmacology_map.md   # full neuroscience ↔ AI mapping
└── compounds/                # pre-built .gguf control vectors
```

---

## 📚 Theoretical Foundation

- [Activation Addition: Steering LMs Without Optimization](https://arxiv.org/abs/2308.10248) — Turner et al.
- [Representation Engineering](https://arxiv.org/abs/2310.01405) — Zou et al., CAIS
- [Towards Monosemanticity](https://transformer-circuits.pub/2023/monosemantic-features) — Anthropic
- [Do LLMs Feel? Emotion Circuits](https://arxiv.org/abs/2510.11328)
- [Toy Models of Superposition](https://transformer-circuits.pub/2022/toy_model/index.html) — Elhage et al.

---

## 🔬 Contributing

New drugs welcome. To contribute a compound:
1. Place it in the appropriate `drugs/<class>/` folder
2. Include: model name, layer, coefficient range, training prompts, dose-response curve
3. Document overdose threshold
4. Open a PR

---

*Created by [@dino65-dev](https://github.com/dino65-dev) — pioneering AI pharmacology.*
