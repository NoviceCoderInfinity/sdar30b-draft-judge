#!/usr/bin/env python
"""Pair this-box vanilla (--stock) records with the Phase 4 vanilla records (vanilla_p4box) by (split, idx); report
per-set and pooled median seconds ratio (this box / Phase 4 box), the gpu_active_s ratio, and token identity (gen_ids)."""
import glob, json, os, statistics as st
ROOT = "/workspace/diffusion-moe-general-expert"
SETS = ["ifeval541", "math500", "gpqa", "aime", "humaneval", "mbpp", "gsm8k500"]

def load(d):
    rows = {}
    for f in glob.glob(os.path.join(d, "*.jsonl")):
        for line in open(f):
            if not line.strip():
                continue
            r = json.loads(line)
            rows[str(r.get("idx"))] = r
    return rows

print(f"{'set':10s} {'n':>4s} {'tok_id':>7s} {'sec_this':>9s} {'sec_p4':>8s} {'ratio':>6s} {'gpuact_ratio':>12s}")
all_ratios, all_gpu = [], []
for s in SETS:
    this = load(os.path.join(ROOT, "logs/phase5/vanilla", s))
    p4 = load(os.path.join(ROOT, "logs/phase4/decode/vanilla_p4box", s))
    common = sorted(set(this) & set(p4), key=lambda x: int(x) if x.isdigit() else x)
    if not common:
        print(f"{s:10s} {'-':>4s}  (no paired items: this={len(this)} p4={len(p4)})")
        continue
    ratios, gpur, tokid = [], [], 0
    for i in common:
        a, b = this[i], p4[i]
        sa, sb = a.get("seconds"), b.get("seconds")
        if sa and sb:
            ratios.append(sa / sb)
        ga, gb = a.get("gpu_active_s"), b.get("gpu_active_s")
        if ga and gb:
            gpur.append(ga / gb)
        if a.get("gen_ids") is not None and a.get("gen_ids") == b.get("gen_ids"):
            tokid += 1
    all_ratios += ratios; all_gpu += gpur
    mr = st.median(ratios) if ratios else float('nan')
    mg = st.median(gpur) if gpur else float('nan')
    msa = st.median([this[i]['seconds'] for i in common if this[i].get('seconds')])
    msb = st.median([p4[i]['seconds'] for i in common if p4[i].get('seconds')])
    print(f"{s:10s} {len(common):>4d} {tokid}/{len(common):>5} {msa:>9.2f} {msb:>8.2f} {mr:>6.3f} {mg:>12.3f}")
print("-" * 60)
if all_ratios:
    print(f"POOLED median seconds ratio (this box / Phase 4 box): {st.median(all_ratios):.3f}  (n={len(all_ratios)})")
if all_gpu:
    print(f"POOLED median gpu_active_s ratio:                     {st.median(all_gpu):.3f}  (n={len(all_gpu)})")
print("Speedup is DERIVED = (Phase 4 vanilla seconds / our rd1 judge seconds) x this multiplier; computed in STATUS.md.")
