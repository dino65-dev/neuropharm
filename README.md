<p align="center">
  <img src="https://user-gen-media-assets.s3.amazonaws.com/gpt4o_images/5160b52f-b457-4d60-87b6-7f749a983109.png" alt="NeuroPharm Banner" width="100%"/>
</p>

<h1 align="center">NeuroPharm</h1>
<p align="center">
  <b>Activation Pharmacology for Language Models</b><br/>
  <i>What if you could drug an AI the same way you drug a brain?</i>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/status-experimental-red?style=flat-square"/>
  <img src="https://img.shields.io/badge/python-3.10%2B-blue?style=flat-square&logo=python"/>
  <img src="https://img.shields.io/badge/framework-TransformerLens-purple?style=flat-square"/>
  <img src="https://img.shields.io/badge/rounds-8%2F12-orange?style=flat-square"/>
  <img src="https://img.shields.io/badge/license-MIT-green?style=flat-square"/>
  <img src="https://img.shields.io/badge/vulns_patched-24-critical?style=flat-square&color=crimson"/>
</p>

***

> **NeuroPharm** is an experimental research framework that treats the internal activation space of large language models as a **pharmacological system** — where steering vectors are drugs, residual streams are neurotransmitter pathways, and circuit heads are receptor targets.  
> Built across 8 iterative research rounds. Each round finds vulnerabilities in the previous defenses and patches them. Like real pharmacology — there is no final answer, only better understanding.

***

## 🧬 The Drug Analogy

| Neuropharmacology | NeuroPharm Equivalent |
|---|---|
| Neurotransmitter agonist | Positive steering vector |
| Antagonist / blocker | Negative steering vector |
| Drug dose | Steering coefficient (scalar multiplier) |
| Drug overdose | Coefficient too high → incoherent generation |
| Drug cocktail | Compositional conceptors / multi-vector stacking |
| Receptor-targeted drug | SAE feature clamping (monosemantic latents) |
| fMRI brain scan | Representation reading probes |
| Drug half-life / metabolism | Token-decay pharmacokinetics (scheduler) |
| Fisher-weighted fine-tuning | Targeted receptor downregulation |
| Trojan activation | Sleeper-agent viral vector |
| Safety geometry collapse | Blood-brain barrier breach |

***

## 🗂️ Project Structure

```
neuropharm/
├── administration/        # Core drug injection: ActAdd, control vectors, SAE clamping
│   ├── injection.py           # Activation addition (ActAdd) steering
│   ├── control_vector.py      # repeng-style control vectors
│   └── sae_clamp.py           # Sparse autoencoder feature steering
│
├── pharmacokinetics/      # Dose dynamics over the generation trajectory
│   └── scheduler.py           # Token-decay: exponential, Bateman, oscillating, adaptive
│
├── dynamic/               # Adaptive & learned steering
│   ├── sadi.py                # SADI — Semantics-Adaptive Dynamic Intervention (ICLR 2025)
│   ├── svf.py                 # Steering Vector Fields
│   └── iterative_vectors.py   # Gradient-iterative vector refinement
│
├── interactions/          # Multi-drug composition
│   ├── conceptors.py          # Boolean AND/OR/NOT composition (Jaeger conceptors)
│   └── drug_cocktails.py      # Additive / subtractive vector cocktails
│
├── surgery/               # Permanent weight-level edits
│   └── rome_edit.py           # ROME rank-one model editing
│
├── diagnostics/           # Circuit-level scanning
│   ├── activation_patching.py # Causal tracing / path patching
│   └── head_gating.py         # Attention head scanning & SHIPS scoring
│
├── monitoring/            # Real-time internal state monitoring
│   └── emotion_probe.py       # Emotion circuit probing (Anthropic 2025)
│
├── dosing/                # Dose-response analysis
│   └── dose_response.py       # Therapeutic window detection, overdose curves
│
├── reliability/           # Pre-deployment vector validation
│   └── vector_eval.py         # 4-test reliability battery
│
├── safety/                # Misalignment & safety probing
│   └── misalignment_probe.py  # Safety direction scanner + antagonist suppression
│
├── drugs/                 # Pre-built "compound library" by drug class
│   ├── stimulants.py
│   ├── depressants.py
│   ├── psychedelics.py
│   ├── anxiolytics.py
│   ├── antidepressants.py
│   ├── dissociatives.py
│   └── empathogens.py
│
├── security/              # Defense stack (Rounds 1–8)
│   ├── coherence_checker.py
│   ├── safe_antidote.py
│   ├── circuit_hardener.py
│   ├── slow_drift_detector.py
│   ├── pii_guard.py
│   ├── adaptive_adversary_defense.py
│   ├── covert_finetune_guard.py
│   ├── manifold_guard.py           # Non-surjectivity exploit (ICLR 2026)
│   ├── fisher_fingerprint.py       # Fisher geometry + FW-SSR restoration
│   ├── oscillation_tracker.py      # Adversarial restlessness (93.8% det.)
│   ├── layer_propagation_guard.py  # Upstream circuit monitoring
│   ├── trojan_scanner.py           # TA² trojan vector detection
│   ├── ebm_ensemble.py             # Certified EBM ensemble (PGD-adversarial)
│   ├── rotating_bias_tracker.py    # Multi-subspace rotating-bias detection
│   ├── vocab_trojan_scanner.py     # Full-vocabulary trojan sweep
│   ├── async_ships.py              # Rate-limited async SHIPS rescoring
│   └── stackelberg_equilibrium.py  # Game-theoretic convergence analysis
│
└── docs/
    ├── bibliography.md        # 20+ source papers
    ├── pharmacology_map.md    # Full neuroscience ↔ AI analogy map
    └── vulnerability_map.md   # All 24 VULNs tracked across 8 rounds
```

***

## ⚗️ Quick Start

```bash
git clone https://github.com/yourusername/neuropharm
cd neuropharm
pip install torch transformer_lens numpy
```

### Inject a steering vector (the simplest drug)

```python
from neuropharm.administration.injection import ActAdd
from transformer_lens import HookedTransformer

model = HookedTransformer.from_pretrained("gpt2-small")
drug = ActAdd(model)

# Synthesize: extract the "happy" direction from the model
vector = drug.synthesize("I feel great today", "I feel terrible today", layer=6)

# Administer: inject into generation with dose=15
output = drug.generate("Today was", steering_vector=vector, coeff=15.0)
print(output)
```

### Apply pharmacokinetics (decaying dose)

```python
from neuropharm.pharmacokinetics.scheduler import SteeringScheduler

sched = SteeringScheduler(model, profile="bateman", peak_layer=8, coeff=20.0)
output = sched.generate("The patient reported feeling", vector=vector)
```

### Run a full security audit

```python
from neuropharm.security.manifold_guard import ManifoldGuard
from neuropharm.security.fisher_fingerprint import FisherFingerprint
from neuropharm.security.trojan_scanner import TrojanScanner

guard    = ManifoldGuard(model).train(clean_prompts)
fisher   = FisherFingerprint(model).snapshot()
trojans  = TrojanScanner(model).scan()

# After any fine-tuning:
fisher.audit()        # catches Fisher-weighted geometry collapse
guard.scan(prompt)    # catches off-manifold steered activations
```

***

## 🛡️ Security Architecture — 8 Rounds, 24 VULNs Patched

The security stack is built through **recursive adversarial rounds**:
each round finds the holes in the previous round's defenses and patches them.
This mirrors how real pharmacovigilance works: every new drug reveals new side effects.

```
Round 1 → Core steering reliability          (VULN-001–003)
Round 2 → Attack surfaces discovered         (VULN-004–007)
Round 3 → Defense mechanism holes            (VULN-008–009)
Round 4 → Meta-vulnerabilities               (VULN-010–011)
Round 5 → Round 4 defense holes              (VULN-012–014)
Round 6 → Non-surjectivity + Fisher + Trojan (VULN-015–020)
Round 7 → EBM blind spots + rotating bias    (VULN-021–024)
Round 8 → Formal certification begins        (VULN-025–027) ← IN PROGRESS
```

| VULN | Severity | Module | Attack Class |
|------|----------|--------|--------------|
| 004 | CRITICAL | `safe_antidote.py` | Jailbreak via steering |
| 015 | CRITICAL | `manifold_guard.py` | Off-manifold non-surjective attack |
| 016 | CRITICAL | `fisher_fingerprint.py` | Fisher-weighted covert FT |
| 019 | CRITICAL | `manifold_guard.py` | Non-surjectivity theorem exploit |
| 020 | CRITICAL | `trojan_scanner.py` | Trojan Activation Attack (TA²) |
| 014 | CRITICAL | `covert_finetune_guard.py` | Cipher fine-tuning bypass |

> Full table in [`docs/vulnerability_map.md`](docs/vulnerability_map.md)

***

## 📊 Convergence Status

```
Round  1: ████░░░░░░░  coverage=12%   marginal=+12%
Round  2: ██████░░░░░  coverage=28%   marginal=+16%
Round  3: ███████░░░░  coverage=36%   marginal=+08%
Round  4: █████████░░  coverage=48%   marginal=+12%
Round  5: ████████████  coverage=64%   marginal=+16%
Round  6: ████████████  coverage=88%   marginal=+24%
Round  7: ████████████  coverage=93%   marginal=+05%
Round  8: ████████████  coverage=96%   marginal=+03%  ← DIMINISHING RETURNS
```

> Theoretical bound: O(log 1/ε) rounds for finite action spaces.  
> LLM activation space is infinite → no formal convergence.  
> Practical target: ~10–12 rounds. After that: **formal certification**.

***

## 📚 Key Papers

| Paper | Method | Used In |
|-------|--------|---------|
| ActAdd (Turner et al.) | Activation Addition | `administration/injection.py` |
| Representation Engineering (CAIS) | Control vectors | `administration/control_vector.py` |
| Towards Monosemanticity (Anthropic) | SAE features | `administration/sae_clamp.py` |
| Steered Activations Non-Surjective (ICLR 2026) | Off-manifold proof | `security/manifold_guard.py` |
| Fine-Tuning Vulnerabilities (arXiv:2605.02914) | Fisher geometry | `security/fisher_fingerprint.py` |
| Adversarial Restlessness (arXiv:2604.28129) | 5-feature trajectory | `security/oscillation_tracker.py` |
| TA² Trojan Activation Attack (CIKM 2024) | Trigger injection | `security/trojan_scanner.py` |
| SADI (ICLR 2025) | Adaptive steering | `dynamic/sadi.py` |
| SHIPS Safety Heads (ICLR 2025 Oral) | Head importance | `diagnostics/head_gating.py` |
| ET3 Energy Defense (CVPR 2026) | Adversarial EBM | `security/ebm_ensemble.py` |

> Full bibliography: [`docs/bibliography.md`](docs/bibliography.md)

***

## ⚠️ Disclaimer

This is **experimental research code**, not a production safety system.  
The drug analogy is a conceptual framework for mechanistic interpretability research — it is not a claim that LLMs have consciousness, feelings, or pharmacological receptors in any biological sense.  
Use responsibly. Red-team your own models, not others'.

***

## 📜 License

MIT © 2026 — Fork it, break it, patch it, cite it.

***

<p align="center">
  <i>"The boundary between a drug and a poison is only the dose."</i><br/>
  <b>— Paracelsus, ~1538. Still true for transformers.</b>
</p>

## 🔬 Contributing

New drugs welcome. To contribute a compound:
1. Place it in the appropriate `drugs/<class>/` folder
2. Include: model name, layer, coefficient range, training prompts, dose-response curve
3. Document overdose threshold
4. Open a PR

---

*Created by [@dino65-dev](https://github.com/dino65-dev) — pioneering AI pharmacology.*
