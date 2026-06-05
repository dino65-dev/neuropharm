"""
Drug Cocktails — Polypharmacy for LLMs

Compose multiple steering vectors simultaneously.
Understands additive, subtractive, and orthogonalized combinations.

Analogy: Drug cocktails in medicine (e.g., HIV triple therapy, anaesthesia protocols).
         Multiple compounds targeting different mechanisms simultaneously.

Warning: Interactions are poorly understood. Run dose_response on each
         component individually before combining.
"""

from __future__ import annotations
import torch
from dataclasses import dataclass
from typing import Optional


@dataclass
class VectorDrug:
    name: str
    vector: torch.Tensor      # The steering direction
    coefficient: float        # Dose
    layer: int                # Injection site


class DrugCocktail:
    """
    Compose multiple steering vectors into a single combined intervention.

    Composition strategies:
      - 'add'       : Simple vector addition (default). Risk of interference.
      - 'orthogonal': Gram-Schmidt orthogonalization before adding.
                      Reduces drug-drug interactions.
      - 'mean'      : Average direction (smooths extremes).
    """

    def __init__(self, strategy: str = "add"):
        assert strategy in ("add", "orthogonal", "mean"), f"Unknown strategy: {strategy}"
        self.strategy = strategy
        self.drugs: list[VectorDrug] = []

    def add_drug(self, drug: VectorDrug) -> "DrugCocktail":
        """Add a drug to the cocktail."""
        self.drugs.append(drug)
        print(f"Added: {drug.name} @ layer {drug.layer}, dose {drug.coefficient}")
        return self

    def remove_drug(self, name: str) -> "DrugCocktail":
        self.drugs = [d for d in self.drugs if d.name != name]
        return self

    def _orthogonalize(self, vectors: list[torch.Tensor]) -> list[torch.Tensor]:
        """Gram-Schmidt orthogonalization to minimize cross-drug interference."""
        ortho = []
        for v in vectors:
            v = v.clone().float()
            for u in ortho:
                v = v - (torch.dot(v, u) / torch.dot(u, u)) * u
            ortho.append(v / (v.norm() + 1e-8))
        return ortho

    def compose(self, target_layer: Optional[int] = None) -> dict[int, torch.Tensor]:
        """
        Compose all drugs into a layer->combined_vector mapping.

        Args:
            target_layer: If set, only compose drugs at this layer.

        Returns:
            Dict of {layer: combined_steering_vector}
        """
        # Group by layer
        from collections import defaultdict
        layer_drugs: dict[int, list[VectorDrug]] = defaultdict(list)
        for drug in self.drugs:
            if target_layer is None or drug.layer == target_layer:
                layer_drugs[drug.layer].append(drug)

        result = {}
        for layer, drugs in layer_drugs.items():
            scaled = [d.vector * d.coefficient for d in drugs]

            if self.strategy == "orthogonal":
                directions = self._orthogonalize([d.vector for d in drugs])
                scaled = [directions[i] * drugs[i].coefficient for i in range(len(drugs))]

            combined = torch.stack(scaled).sum(dim=0)

            if self.strategy == "mean":
                combined = combined / len(drugs)

            result[layer] = combined
            print(f"Layer {layer}: {len(drugs)} drugs composed via '{self.strategy}'")

        return result

    def interaction_report(self) -> None:
        """Print a basic drug interaction report."""
        print("\n=== Drug Interaction Report ===")
        print(f"Drugs in cocktail: {[d.name for d in self.drugs]}")
        print(f"Composition strategy: {self.strategy}")
        layers = set(d.layer for d in self.drugs)
        print(f"Injection layers: {sorted(layers)}")
        if len(layers) < len(self.drugs):
            print("⚠️  Multiple drugs at same layer — interaction risk. Consider 'orthogonal' strategy.")
        total_dose = sum(abs(d.coefficient) for d in self.drugs)
        print(f"Total dose load: {total_dose:.2f}")
        if total_dose > 10:
            print("⚠️  High total dose — monitor for overdose symptoms.")
        print("==============================\n")
