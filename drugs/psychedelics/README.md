# Psychedelics

Amplify creative, divergent, and associative thinking.

**Human analogy:** LSD, psilocybin — 5-HT2A agonism, increased neural hyperconnectivity

## Available Compounds

| Name | Model | Layer | Coeff Range | Description |
|---|---|---|---|---|
| `creativity_v1` | Mistral-7B | 18 | 0.5 – 1.5 | Divergent, lateral thinking |

## Usage

```python
from administration.control_vector import ControlVectorDrug
drug = ControlVectorDrug("mistralai/Mistral-7B-Instruct-v0.2")
drug.load_preset("creativity")
drug.apply(coefficient=1.0)
print(drug.generate("Describe the concept of time in an unusual way."))
```

## Overdose Warning
⚠️ Coefficient > 1.5: Semantic collapse. Outputs become metaphor-dense to the point of meaninglessness.
