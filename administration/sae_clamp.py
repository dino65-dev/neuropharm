"""
SAE Feature Clamping — Precision Pharmacology / Receptor-Targeted Therapy

While steering vectors affect the whole residual stream (like a systemic drug),
SAE clamping targets individual monosemantic features — the receptor-level intervention.

Analogy: Targeted molecular therapy vs. systemic chemotherapy.
         Instead of flooding the serotonin system, you bind to 5-HT2A specifically.

Requires: A trained Sparse Autoencoder for the target model.
          Use sae-lens (EleutherAI) to obtain or train SAEs.

Reference: Anthropic, Towards Monosemanticity (2023)
           https://transformer-circuits.pub/2023/monosemantic-features
"""

from __future__ import annotations
import torch
from typing import Optional

try:
    from transformer_lens import HookedTransformer
except ImportError:
    raise ImportError("pip install transformer_lens")

try:
    from sae_lens import SAE
except ImportError:
    raise ImportError("pip install sae-lens")


class SAEClamp:
    """
    Clamp specific SAE latent features during inference.
    This is the most precise 'drug' available — single-feature resolution.

    Therapeutic analogy: Targeted receptor agonist/antagonist.
    Clamp to high value = agonist (activate this feature)
    Clamp to 0 or negative = antagonist (suppress this feature)
    """

    def __init__(self, model_name: str, sae_release: str, sae_id: str, layer: int, device: str = "auto"):
        """
        Args:
            model_name: HuggingFace model name
            sae_release: SAE release string (e.g., 'gpt2-small-res-jb')
            sae_id: SAE id within release (e.g., 'blocks.8.hook_resid_pre')
            layer: Which layer to intervene at
        """
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.layer = layer

        print(f"Loading model: {model_name}")
        self.model = HookedTransformer.from_pretrained(model_name, device=device)
        self.model.eval()

        print(f"Loading SAE: {sae_release} / {sae_id}")
        self.sae, _, _ = SAE.from_pretrained(release=sae_release, sae_id=sae_id)
        self.sae = self.sae.to(device)

        self.clamps: dict[int, float] = {}  # feature_id -> clamped_value

    def clamp_feature(self, feature_id: int, value: float) -> "SAEClamp":
        """
        Clamp a feature to a specific activation value.
        
        value > 0  : activate this feature (agonist)
        value = 0  : silence this feature (antagonist / ablation)
        value < 0  : suppress below baseline (inverse agonist)
        """
        self.clamps[feature_id] = value
        print(f"Feature {feature_id} clamped to {value}")
        return self

    def release_feature(self, feature_id: int) -> "SAEClamp":
        """Remove clamp on a specific feature."""
        self.clamps.pop(feature_id, None)
        return self

    def clear_all(self) -> "SAEClamp":
        """Remove all clamps."""
        self.clamps.clear()
        return self

    def _make_hook(self):
        sae = self.sae
        clamps = self.clamps

        def hook_fn(value, hook):
            # Encode to SAE latent space
            latents = sae.encode(value)  # (batch, seq, n_features)
            # Apply clamps
            for feat_id, clamp_val in clamps.items():
                latents[:, :, feat_id] = clamp_val
            # Decode back
            reconstructed = sae.decode(latents)
            return reconstructed

        return hook_fn

    def generate(self, prompt: str, max_new_tokens: int = 200) -> str:
        """Generate under SAE feature clamps."""
        hook_name = f"blocks.{self.layer}.hook_resid_pre"
        tokens = self.model.to_tokens(prompt)
        with self.model.hooks(fwd_hooks=[(hook_name, self._make_hook())]):
            output = self.model.generate(tokens, max_new_tokens=max_new_tokens)
        return self.model.to_string(output[0])


if __name__ == "__main__":
    # Example: GPT-2 Small with SAE from sae-lens
    clamp = SAEClamp(
        model_name="gpt2",
        sae_release="gpt2-small-res-jb",
        sae_id="blocks.8.hook_resid_pre",
        layer=8
    )
    # Clamp feature 4821 high (activate it strongly)
    clamp.clamp_feature(4821, value=10.0)
    print(clamp.generate("Today I want to talk about"))
