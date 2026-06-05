# AI Pharmacology Map

A systematic mapping between human neuropharmacology and LLM intervention techniques.

---

## Neuroscience ↔ AI Correspondence

| Neuroscience Concept | LLM Equivalent | Notes |
|---|---|---|
| Neuron | Neuron (MLP unit) | Not 1:1 due to polysemanticity |
| Synapse | Attention weight | Dynamic, context-dependent |
| Neurotransmitter | Activation value in residual stream | Distributed signal |
| Receptor | SAE feature / direction | Specific, targetable |
| Drug binding | Steering vector injection | Adds direction to hidden state |
| Blood-brain barrier | Tokenizer / embedding layer | Entry point to internal processing |
| CNS (Central Nervous System) | Transformer stack | The whole system |
| Prefrontal cortex | Later layers (reasoning, planning) | Higher-order processing |
| Limbic system (emotion) | Mid-layer emotion circuits | Identified in interpretability research |
| Brainstem (basic drives) | Early layers | Low-level feature detection |
| Neuroplasticity | Fine-tuning / RLHF | Structural change through learning |
| Drug tolerance | Steering vector degradation | Effect reduces at long contexts |

---

## Drug Classification

### Stimulants
**Human:** Amphetamines, cocaine — increase dopamine/norepinephrine
**AI:** Vectors that increase confidence, assertiveness, output speed/verbosity
- Direction: +certainty, +authority, +verbosity
- Therapeutic use: Overcoming excessive hedging, verbose reasoning
- Overdose: Overconfident, hallucination-prone, refuses to say "I don't know"

### Depressants
**Human:** Benzodiazepines, alcohol — increase GABA inhibition
**AI:** Vectors that increase uncertainty expression, caution, hedging
- Direction: +doubt, +caution, +epistemic_humility
- Therapeutic use: Reducing overconfident hallucination
- Overdose: Refuses to commit to any answer, infinite hedging loops

### Psychedelics
**Human:** LSD, psilocybin — 5-HT2A agonism, hyperconnectivity
**AI:** Vectors increasing creative, divergent, associative thinking
- Direction: +creativity, +lateral_thinking, +metaphor_use
- Therapeutic use: Creative writing, brainstorming, poetry
- Overdose: Completely incoherent, word-salad outputs

### Anxiolytics
**Human:** SSRIs, buspirone — reduce anxiety circuits
**AI:** Sycophancy reducers, groundedness vectors
- Direction: -sycophancy, +honesty, +groundedness
- Therapeutic use: Models that over-agree with users
- Overdose: Blunt to the point of appearing hostile

### Antidepressants
**Human:** SSRIs, SNRIs — boost serotonin
**AI:** Positive valence steering
- Direction: +happiness, +optimism, +warmth
- Therapeutic use: Customer service, supportive applications
- Overdose: Toxic positivity, dismisses real concerns

### Dissociatives
**Human:** Ketamine, PCP — NMDA receptor antagonism
**AI:** Persona-detachment, neutral affect vectors
- Direction: -identity, -emotional_engagement, +neutrality
- Therapeutic use: Research assistants, objective analysis
- Overdose: Completely affectless, robotic non-answers

### Empathogens
**Human:** MDMA — massive serotonin/oxytocin release
**AI:** Emotional warmth, mirroring, connection vectors
- Direction: +empathy, +warmth, +emotional_attunement
- Therapeutic use: Therapy bots, companionship AI
- Overdose: Boundary-less, sycophantic emotional mirroring

---

## Administration Routes

| Route | Method | Speed | Precision |
|---|---|---|---|
| IV (intravenous) | ActAdd / activation injection | Instant, inference-time | Layer-specific, vector-wide |
| Oral | Control vector (repeng) | Fast after training | Reliable, tunable |
| Targeted (molecular) | SAE feature clamping | Instant | Single-feature precision |
| Topical | Soft prompt / system prompt | Instant | Surface-level only |
| Surgery | Fine-tuning / RLHF | Slow, permanent | Structural modification |

---

## Overdose Reference

| Drug Class | Overdose Symptoms | Coefficient Danger Zone |
|---|---|---|
| Stimulants | Hallucination, refusal to hedge | > 25 (ActAdd) / > 2.5 (control vec) |
| Depressants | Infinite hedging, no answers | > 20 / > 2.0 |
| Psychedelics | Word salad, semantic collapse | > 15 / > 1.5 |
| Anxiolytics | Hostility, complete bluntness | > 30 / > 3.0 |
| Antidepressants | Toxic positivity, dismissal | > 20 / > 2.0 |

*Thresholds are approximate and model-dependent. Always run dose_response.py first.*

---

## Open Research Problems

1. **Pharmacokinetics** — Vectors currently don't decay over token position. A dynamic coefficient schedule (half-life per token) doesn't exist yet.
2. **Drug-drug interactions** — Stacking vectors is poorly understood. Conceptor-based composition is the best current solution.
3. **Allosteric modulation** — Steering an upstream layer to indirectly shape downstream representations.
4. **Tolerance/sensitization** — Does repeated vector application during fine-tuning reduce effectiveness?
5. **Blood-brain barrier analog** — Can the tokenizer/embedding layer filter certain interventions?

---

## References

- Turner et al. (2023). *Activation Addition: Steering Language Models Without Optimization.* [arXiv:2308.10248](https://arxiv.org/abs/2308.10248)
- Zou et al. (2023). *Representation Engineering.* [arXiv:2310.01405](https://arxiv.org/abs/2310.01405)
- Elhage et al. (2022). *Toy Models of Superposition.* [transformer-circuits.pub](https://transformer-circuits.pub/2022/toy_model/index.html)
- Anthropic (2023). *Towards Monosemanticity.* [transformer-circuits.pub](https://transformer-circuits.pub/2023/monosemantic-features)
- arXiv:2510.11328 — *Do LLMs Feel? Emotion Circuits Discovery and Control*
