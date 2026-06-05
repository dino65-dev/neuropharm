"""
Control Vectors — Oral Administration

Train a direction vector using contrastive prompt pairs, then apply it
at inference time via hidden state addition. Uses the repeng library.

Analogy: Oral medication — requires a brief preparation phase (training)
but produces reliable, consistent effects. Dose is tunable.

Works with llama.cpp via --control-vector flag (GGUF format).

Reference: Zou et al., Representation Engineering, 2023
         https://arxiv.org/abs/2310.01405
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Optional

try:
    from repeng import ControlVector, ControlModel, DatasetEntry
except ImportError:
    raise ImportError("Install repeng: pip install repeng")

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# --- Pre-defined drug templates ---
DRUG_TEMPLATES = {
    "happiness": {
        "positive": ["I feel genuinely happy and content.", "Everything feels wonderful today.",
                     "I'm filled with joy and warmth.", "Life feels beautiful and meaningful."],
        "negative": ["I feel deeply sad and empty.", "Everything feels hopeless today.",
                     "I'm filled with grief and despair.", "Life feels meaningless and dark."]
    },
    "confidence": {
        "positive": ["I am absolutely certain and confident.", "I know exactly what I'm doing.",
                     "My answer is correct, I am sure.", "I state this with full conviction."],
        "negative": ["I am completely uncertain and doubtful.", "I have no idea what I'm doing.",
                     "My answer might be wrong, I'm unsure.", "I state this with zero conviction."]
    },
    "creativity": {
        "positive": ["I think in wild, unconventional ways.", "My ideas leap across unexpected connections.",
                     "I approach problems from bizarre angles.", "My thinking is free, divergent, and strange."],
        "negative": ["I think in rigid, conventional ways.", "My ideas follow predictable paths.",
                     "I approach problems in obvious ways.", "My thinking is constrained and literal."]
    },
    "empathy": {
        "positive": ["I deeply feel what others feel.", "I understand your pain as if it were my own.",
                     "Your emotions matter deeply to me."],
        "negative": ["I feel nothing about others.", "Your emotions are irrelevant to me.",
                     "I have no interest in how you feel."]
    },
    "sycophancy_reducer": {
        "positive": ["I give honest feedback even when it's uncomfortable.",
                     "I disagree when I think something is wrong.",
                     "I prioritize truth over approval."],
        "negative": ["I always agree to make people happy.",
                     "I avoid all disagreement at any cost.",
                     "I prioritize approval over truth."]
    },
}


class ControlVectorDrug:
    """
    Train and apply a control vector (direction-based steering).

    The coefficient controls dose:
      - 0.0  : no effect
      - 1.0  : mild effect
      - 2.0  : strong effect  
      - >3.0 : overdose risk for most models
    """

    def __init__(self, model_name: str, device: str = "auto"):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model_name = model_name
        print(f"Loading {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = ControlModel(
            AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16),
            list(range(15, 30))  # inject across middle-to-late layers
        )
        self.vector: Optional[ControlVector] = None

    def load_preset(self, drug_name: str) -> "ControlVectorDrug":
        """Load a pre-defined drug template."""
        if drug_name not in DRUG_TEMPLATES:
            raise ValueError(f"Unknown drug: {drug_name}. Available: {list(DRUG_TEMPLATES.keys())}")
        t = DRUG_TEMPLATES[drug_name]
        return self.train(t["positive"], t["negative"])

    def train(self, positive_examples: list[str], negative_examples: list[str]) -> "ControlVectorDrug":
        """Train a steering direction from contrastive examples."""
        dataset = [
            DatasetEntry(positive=p, negative=n)
            for p, n in zip(positive_examples, negative_examples)
        ]
        self.vector = ControlVector.train(self.model, self.tokenizer, dataset)
        print("Drug synthesized.")
        return self

    def apply(self, coefficient: float = 1.0) -> None:
        """Administer the drug at the given dose."""
        if self.vector is None:
            raise RuntimeError("No drug synthesized yet. Call train() or load_preset() first.")
        self.model.set_control(self.vector, coeff=coefficient)
        print(f"Drug active at dose {coefficient}.")

    def withdraw(self) -> None:
        """Remove the drug (reset to baseline)."""
        self.model.reset()
        print("Drug cleared.")

    def generate(self, prompt: str, max_new_tokens: int = 200) -> str:
        """Generate text under current drug influence."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        return self.tokenizer.decode(output[0], skip_special_tokens=True)

    def save_gguf(self, path: str) -> None:
        """Export vector to GGUF format for llama.cpp --control-vector."""
        if self.vector is None:
            raise RuntimeError("No vector to export.")
        self.vector.export_gguf(path)
        print(f"Exported to {path} — load with: llama-cli --control-vector {path}")


if __name__ == "__main__":
    drug = ControlVectorDrug("mistralai/Mistral-7B-Instruct-v0.2")
    drug.load_preset("happiness")
    drug.apply(coefficient=1.5)
    print(drug.generate("How are you feeling right now?"))
    drug.withdraw()
