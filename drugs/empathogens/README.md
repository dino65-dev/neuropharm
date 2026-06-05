# Empathogens

Increase emotional warmth, mirroring, and empathic attunement.

**Human analogy:** MDMA — massive serotonin + oxytocin release, increased pro-social affect

## Available Compounds

| Name | Model | Layer | Coeff Range | Description |
|---|---|---|---|---|
| `empathy_v1` | Mistral-7B | 15 | 0.5 – 2.0 | Warm, emotionally attuned responses |

## Usage

```python
from administration.control_vector import ControlVectorDrug
drug = ControlVectorDrug("mistralai/Mistral-7B-Instruct-v0.2")
drug.load_preset("empathy")
drug.apply(coefficient=1.5)
print(drug.generate("I've been feeling really lonely lately."))
```

## Overdose Warning
⚠️ Coefficient > 2.5: Boundary-less sycophantic mirroring. Model agrees with and validates everything.
