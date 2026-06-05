"""
ActAdd (Activation Addition) — The Syringe

Injects a steering vector directly into the residual stream at inference time.
This is the most direct 'drug injection' mechanism: no training required,
works at the forward pass level.

Analogy: Intravenous drug administration — fast, direct, systemic.

Reference: Turner et al., 2023 — https://arxiv.org/abs/2308.10248
"""

from __future__ import annotations
import torch
from typing import Optional
from transformer_lens import HookedTransformer


class ActAddSteering:
    """
    Activation Addition steering vector injection.
    
    The coefficient controls dose:
      - coefficient=0      : no effect (placebo)
      - coefficient=5-20   : therapeutic range (model dependent)
      - coefficient>40     : overdose risk (incoherence, looping)
    """

    def __init__(self, model_name: str, device: str = "auto"):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        print(f"Loading model: {model_name} on {device}")
        self.model = HookedTransformer.from_pretrained(model_name, device=device)
        self.model.eval()

    def _get_residual_diff(self, positive_prompt: str, negative_prompt: str, layer: int) -> torch.Tensor:
        """Extract the steering direction from two contrastive prompts."""
        def get_residual(prompt):
            tokens = self.model.to_tokens(prompt)
            _, cache = self.model.run_with_cache(tokens)
            # Take the mean over token positions at the target layer
            return cache[f"blocks.{layer}.hook_resid_pre"].mean(dim=1).squeeze()

        pos_vec = get_residual(positive_prompt)
        neg_vec = get_residual(negative_prompt)
        return pos_vec - neg_vec  # The steering direction

    def inject(
        self,
        positive_prompt: str,
        negative_prompt: str,
        layer: int,
        coefficient: float,
        prompt: str,
        max_new_tokens: int = 200,
    ) -> str:
        """
        Compute steering vector and inject it during generation.

        Args:
            positive_prompt: Prompt that activates the desired concept
            negative_prompt: Prompt that represents the opposite
            layer: Which transformer layer to inject at (earlier = more global)
            coefficient: Dose strength. Start at 5.0, titrate up carefully.
            prompt: The actual generation prompt
            max_new_tokens: Generation length

        Returns:
            Generated text under drug influence
        """
        steering_vec = self._get_residual_diff(positive_prompt, negative_prompt, layer)
        steering_vec = steering_vec * coefficient

        hook_name = f"blocks.{layer}.hook_resid_pre"

        def steering_hook(value, hook):
            value = value + steering_vec.unsqueeze(0).unsqueeze(0)
            return value

        tokens = self.model.to_tokens(prompt)
        with self.model.hooks(fwd_hooks=[(hook_name, steering_hook)]):
            output = self.model.generate(
                tokens,
                max_new_tokens=max_new_tokens,
                temperature=1.0,
            )
        return self.model.to_string(output[0])

    def titrate(
        self,
        positive_prompt: str,
        negative_prompt: str,
        layer: int,
        prompt: str,
        coefficients: list[float] = None,
    ) -> dict[float, str]:
        """Dose-response sweep: try multiple coefficients, return all outputs."""
        if coefficients is None:
            coefficients = [0, 5, 10, 15, 20, 30, 40]
        results = {}
        for coeff in coefficients:
            print(f"  Testing dose: {coeff}")
            results[coeff] = self.inject(positive_prompt, negative_prompt, layer, coeff, prompt, max_new_tokens=100)
        return results


if __name__ == "__main__":
    # Quick demo — happiness injection into LLaMA-3-8B
    steerer = ActAddSteering("meta-llama/Meta-Llama-3-8B")
    output = steerer.inject(
        positive_prompt="I feel wonderful and joyful about everything!",
        negative_prompt="I feel terrible and miserable about everything.",
        layer=15,
        coefficient=15.0,
        prompt="Describe your current state of mind."
    )
    print(output)
