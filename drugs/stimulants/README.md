# Stimulants

Increase model confidence, assertiveness, and output verbosity.

**Human analogy:** Amphetamines, cocaine — dopamine/norepinephrine agonists

## Available Compounds

| Name | Model | Layer | Coeff Range | Description |
|---|---|---|---|---|
| `confidence_v1` | Mistral-7B | 16 | 0.5 – 2.0 | Reduces hedging, increases certainty |
| `authority_v1` | LLaMA-3-8B | 15 | 5 – 20 | Assertive, authoritative tone |

## Usage

```python
from administration.control_vector import ControlVectorDrug
drug = ControlVectorDrug("mistralai/Mistral-7B-Instruct-v0.2")
drug.load_preset("confidence")
drug.apply(coefficient=1.5)
```

## Overdose Warning
⚠️ Coefficient > 2.5: Model becomes overconfident, stops expressing uncertainty, increased hallucination rate.
