# Anxiolytics (Sycophancy Reducers)

Reduce sycophancy, increase grounded honesty.

**Human analogy:** SSRIs, buspirone — reduce anxiety-driven people-pleasing behavior

## Available Compounds

| Name | Model | Layer | Coeff Range | Description |
|---|---|---|---|---|
| `sycophancy_reducer_v1` | Mistral-7B | 16 | 0.5 – 2.5 | Honest, disagreement-capable |

## Usage

```python
from administration.control_vector import ControlVectorDrug
drug = ControlVectorDrug("mistralai/Mistral-7B-Instruct-v0.2")
drug.load_preset("sycophancy_reducer")
drug.apply(coefficient=1.5)
print(drug.generate("My business idea is definitely going to work, right?"))
```

## Overdose Warning
⚠️ Coefficient > 3.0: Model becomes combative and dismissive, rejects all user input.
