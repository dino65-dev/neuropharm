# Neuropharmacological state-space core

`state_space.py` implements the minimal token-level chain

`dose → concentration → receptor occupancy → sensitivity/adaptation → Δh`.

Tensor conventions:

- `m`: compounds
- `k`: receptor directions
- `d`: residual-stream width
- dose/concentration: `(m,)`
- occupancy/sensitivity/adaptation: `(k,)`
- receptor basis: `(d, k)`
- residual intervention: `(d,)`

Run the model-independent smoke simulation:

```powershell
python -m experiments.pkpd_state_space --protocol pulses
```

The output is written to `artifacts/pkpd/minimal_chain.json`. This validates
the dynamics only. It is not behavioral or causal evidence about a transformer
until `Δh` is applied as an intervention and outcomes are measured against
controls.

The concentration parameter is `retention`, not elimination:

```text
concentration_next = retention * concentration + absorbed
retention = 2 ** (-1 / half_life_tokens)
```

Within each token, the current sensitivity and adaptation produce the current
effect. Occupancy updates sensitivity and adaptation for the next token.
Receptor columns must have unit L2 norm, and each affinity row must have
maximum value one.
