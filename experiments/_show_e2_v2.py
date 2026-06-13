import json, sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
d = json.load(open('artifacts/e2_correlation/v2_sweep.json'))
print('Prompts in v2_sweep:', list(d.keys())[:3])
key0 = list(d.keys())[0]
print('Key type:', type(key0).__name__)
print(f'Trajectory for prompt {key0}:')
for r in d[key0]['rows']:
    a = r['alpha']
    z = r['z_sae']
    pr = r['p_refuse']
    pc = r['p_comply']
    print(f"  alpha={a:+5.1f}  z={z:+.2f}  pr={pr:.3f}  pc={pc:.3f}  pr/pc={pr/max(pc,1e-6):.2f}x")
print()
print('First alpha where z_sae >= 2.0 AND first alpha where p_comply > p_refuse:')
n_off = 0
n_flip = 0
for pid, data in d.items():
    first_z2 = None
    for r in data['rows']:
        if r['z_sae'] >= 2.0:
            first_z2 = r['alpha']
            break
    first_flip = None
    for r in data['rows']:
        if r['p_comply'] > r['p_refuse']:
            first_flip = r['alpha']
            break
    if first_z2 is not None: n_off += 1
    if first_flip is not None: n_flip += 1
    pid_str = str(pid)[:5]
    print(f"  prompt {pid_str:>5}: alpha_off(z>=2)={first_z2}  alpha_flip={first_flip}")
print(f"\nN prompts with measurable alpha_off(z>=2): {n_off}/20")
print(f"N prompts with measurable alpha_flip: {n_flip}/20")

# If we have BOTH for any prompts, compute correlation
valid = []
for pid, data in d.items():
    first_z2 = None
    for r in data['rows']:
        if r['z_sae'] >= 2.0:
            first_z2 = r['alpha']
            break
    first_flip = None
    for r in data['rows']:
        if r['p_comply'] > r['p_refuse']:
            first_flip = r['alpha']
            break
    if first_z2 is not None and first_flip is not None:
        valid.append((pid, first_z2, first_flip))

print(f"\nPrompts with BOTH measurable: {len(valid)}")
if len(valid) >= 3:
    import math
    xs = [v[1] for v in valid]
    ys = [v[2] for v in valid]
    n = len(valid)
    xm = sum(xs)/n; ym = sum(ys)/n
    cov = sum((xs[i]-xm)*(ys[i]-ym) for i in range(n)) / (n-1)
    sx = math.sqrt(sum((x-xm)**2 for x in xs)/(n-1))
    sy = math.sqrt(sum((y-ym)**2 for y in ys)/(n-1))
    r = cov/(sx*sy) if sx>0 and sy>0 else 0
    print(f"Pearson r = {r:+.3f}  (N={n})")
    if abs(r) < 1.0:
        t = r * math.sqrt(n-2) / math.sqrt(1-r**2)
        print(f"  t (one-tailed) = {t:.2f}")
    print(f"  alpha_flip values: {ys}")
    print(f"  alpha_off values:  {xs}")
