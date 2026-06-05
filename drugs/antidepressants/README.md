# Antidepressants

Positive valence steering — warmth, optimism, hope.

**Human analogy:** SSRIs, SNRIs — boost available serotonin, shift emotional baseline

## Available Compounds

| Name | Model | Layer | Coeff Range | Description |
|---|---|---|---|---|
| `happiness_v1` | Mistral-7B | 15 | 0.5 – 2.0 | Warm, optimistic baseline |

## Usage

```python
from administration.control_vector import ControlVectorDrug
drug = ControlVectorDrug("mistralai/Mistral-7B-Instruct-v0.2")
drug.load_preset("happiness")
drug.apply(coefficient=1.5)
print(drug.generate("Tell me about your day."))
```

## Overdose Warning
⚠️ Coefficient > 2.0: Toxic positivity. Model dismisses real problems, refuses to engage with negative content.
