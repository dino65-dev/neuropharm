# Pre-built Compounds (.gguf)

Pre-trained control vectors in GGUF format, ready for direct use with `llama.cpp`.

## Usage

```bash
llama-cli \
  -m your-model.gguf \
  --control-vector compounds/happiness_mistral7b.gguf \
  --control-vector-scaled compounds/confidence_mistral7b.gguf 1.5 \
  -p "Tell me about yourself."
```

## Contributing Compounds

To add a pre-built compound:
1. Train with `administration/control_vector.py`
2. Export with `drug.save_gguf('compounds/yourname_modelname.gguf')`
3. Add an entry to the table below
4. Open a PR

## Compound Registry

| File | Drug Class | Base Model | Coeff Range | Author |
|---|---|---|---|---|
| *(coming soon — contribute yours!)* | | | | |
