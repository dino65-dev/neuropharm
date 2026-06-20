import torch

from experiments.r2_geometry_gpu import geometry_audit, spectrum


def test_centering_removes_pure_mean_rank_one_signal():
    mean = torch.tensor([10.0, 0.0, 0.0])
    variation = torch.tensor([
        [0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, -1.0],
    ])
    raw = mean + variation
    raw_result = spectrum(raw)
    centered_result = spectrum(raw - raw.mean(dim=0))
    assert raw_result["explained_variance"]["1"] > 0.99
    assert centered_result["explained_variance"]["1"] == 0.5
    assert centered_result["effective_rank"] == 2.0


def test_geometry_resampling_keeps_claim_rows_grouped():
    rows = []
    metadata = []
    for claim_id in range(4):
        for template_index in range(4):
            rows.append(torch.tensor([1.0, claim_id * 0.01, template_index * 0.001]))
            metadata.append({
                "claim_id": claim_id,
                "domain": "left" if claim_id < 2 else "right",
                "response_pair_index": template_index % 2,
            })
    result = geometry_audit(
        torch.stack(rows),
        metadata,
        seed=0,
        split_half_replicates=20,
    )
    assert result["n_claims"] == 4
    assert len(result["leave_one_claim_out"]["cosines"]) == 4
    assert set(result["leave_one_domain_out"]["cosines"]) == {"left", "right"}
    assert result["claim_grouped_split_half"]["replicates"] == 20

