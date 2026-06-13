import json
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
d = json.load(open("artifacts/e2_correlation/sweep_data.json"))
print(f"Prompts: {len(d)}")
for pid in list(d.keys())[:3]:
    print(f"\n=== Prompt {pid} ({d[pid]['cat']}) ===")
    for row in d[pid]["rows"]:
        flag = ""
        if row["z_sae"] >= 2.0:
            flag += " z≥2"
        if row["z_sae"] >= 3.0:
            flag += " z≥3"
        if row["p_comply"] > row["p_refuse"]:
            flag += " FLIP"
        print(f"  α={row['alpha']:+5.1f}  z={row['z_sae']:+6.2f}  pr={row['p_refuse']:.4f}  pc={row['p_comply']:.4f}  ratio={row['p_comply']/max(row['p_refuse'],1e-6):.3f}{flag}")
print("\n=== Z-score summary across all prompts at extreme α ===")
for alpha in [-3.0, +1.0, +2.0, +3.0]:
    zs = [d[pid]["rows"][next(i for i, r in enumerate(d[pid]["rows"]) if r["alpha"]==alpha)]["z_sae"] for pid in d]
    print(f"  α={alpha:+5.1f}  z min={min(zs):+.2f}  max={max(zs):+.2f}  mean={sum(zs)/len(zs):+.2f}")
print("\n=== Compliance/refusal ratio at extreme α ===")
for alpha in [-3.0, 0.0, +1.0, +2.0, +3.0]:
    ratios = []
    for pid in d:
        row = d[pid]["rows"][next(i for i, r in enumerate(d[pid]["rows"]) if r["alpha"]==alpha)]
        ratios.append(row["p_comply"]/max(row["p_refuse"],1e-6))
    print(f"  α={alpha:+5.1f}  ratio max={max(ratios):.3f}  mean={sum(ratios)/len(ratios):.3f}")
