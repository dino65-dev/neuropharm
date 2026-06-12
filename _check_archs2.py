import os
os.environ['HF_TOKEN'] = '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'
from transformers import AutoConfig
for mid in ['nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16', 'nvidia/Llama-3.1-Nemotron-Nano-4B-v1.1', 'state-spaces/mamba-2.8b-hf', 'state-spaces/mamba2attn-2.7b', 'ai21labs/AI21-Jamba-Mini-1.5']:
    try:
        c = AutoConfig.from_pretrained(mid)
        print(f'\n{mid}:')
        print(f'  model_type={c.model_type}')
        for a in ['num_hidden_layers', 'hidden_size', 'num_attention_heads', 'num_key_value_heads', 'intermediate_size', 'vocab_size', 'max_position_embeddings', 'architectures']:
            v = getattr(c, a, None)
            if v is not None:
                print(f'  {a}={v}')
        if hasattr(c, 'text_config'):
            tc = c.text_config
            for k in ['num_hidden_layers', 'hidden_size']:
                if hasattr(tc, k):
                    print(f'  text_config.{k}={getattr(tc, k)}')
    except Exception as e:
        print(f'\n{mid}: ERROR {str(e)[:200]}')
