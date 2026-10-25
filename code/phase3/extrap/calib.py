# Box calibration: the same 16 HumanEval items, same config (SDAR-1.7B + 128M judge, no pool), old 8x box vs this box.
import glob, json, sys
R = "/workspace/dmge"
def load(pat):
    out = {}
    for f in glob.glob(pat):
        if f.endswith("running.jsonl"): continue
        for l in open(f):
            if l.strip():
                r = json.loads(l); out[int(r["idx"])] = r
    return out
old = load(f"{R}/logs/phase2/ref/S0_128M_nopool_by_set/humaneval/*.jsonl")
new = load(f"{R}/logs/phase3/setup/check_judge/*.jsonl")
ids = sorted(new)
same = sum(old[i]["gen_ids"] == new[i]["gen_ids"] for i in ids)
print(f"paired items {len(ids)}, identical outputs {same}; old gpu {old[ids[0]]['gpu']} utc {old[ids[0]]['utc']}")
def comp(r):
    c = {f"moe_{k}": v for k, v in r["ms_by_kind"].items()}
    c.update({f"dr_{k}": v for k, v in r["drafter_ms_by_kind"].items()})
    c["other"] = 1000 * r["seconds"] - sum(c.values())
    return c
keys = sorted(comp(new[ids[0]]))
print(f"{'component':14s} {'old ms':>9s} {'new ms':>9s} {'new/old':>8s}  counts old/new")
for k in keys:
    o = sum(comp(old[i])[k] for i in ids); n = sum(comp(new[i])[k] for i in ids)
    if o or n: print(f"{k:14s} {o:9.0f} {n:9.0f} {n/o if o else float('nan'):8.3f}")
to = sum(old[i]["seconds"] for i in ids); tn = sum(new[i]["seconds"] for i in ids); g = sum(new[i]["n_gen_to_eos"] for i in ids)
print(f"total s old {to:.1f} new {tn:.1f} ratio {tn/to:.3f}; tok/s old {g/to:.2f} new {g/tn:.2f}")
print("per-item ratio new/old:", " ".join(f"{new[i]['seconds']/old[i]['seconds']:.2f}" for i in ids))
