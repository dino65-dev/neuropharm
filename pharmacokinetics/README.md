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
