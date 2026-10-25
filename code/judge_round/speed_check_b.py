#!/usr/bin/env python
"""Stage B summary: this box's vanilla records vs the Phase 4 box's (moved to logs/phase4/decode/vanilla_p4box/<set>).
Seconds per answer = the decoder's own `seconds` field; GPU-active = `gpu_active_s` (nvidia-ml-py meter). Counts: ref_match."""
import argparse, collections, glob, json, os, re, statistics as st
ap = argparse.ArgumentParser(); ap.add_argument("--set", default="gsm8k500"); ap.add_argument("--out", default="")
ap.add_argument("--here", default=""); ap.add_argument("--p4", default="")
a = ap.parse_args()
here = a.here or f"logs/phase4/decode/vanilla_here/{a.set}"; p4 = a.p4 or f"logs/phase4/decode/vanilla_p4box/{a.set}"
def load(d):
    rows = []
    for f in sorted(glob.glob(f"{d}/*.jsonl")):
        m = re.search(r"shard(\d+)of(\d+)", f); tag = m.group(1) if m else os.path.basename(f)
        for l in open(f):
            if l.strip():
                r = json.loads(l); r["_shard"] = tag; rows.append(r)
    return {int(r["idx"]): r for r in rows}
H, P = load(here), load(p4)
common = sorted(set(H) & set(P))
def stats(R, keys):
    sec = [R[i]["seconds"] for i in keys]; act = [R[i].get("gpu_active_s") for i in keys if R[i].get("gpu_active_s") is not None]
    by = collections.defaultdict(list)
    for i in keys: by[R[i]["_shard"]].append(R[i]["seconds"])
    return {"n": len(keys), "mean_seconds": st.mean(sec) if sec else None, "median_seconds": st.median(sec) if sec else None,
            "mean_gpu_active_s": st.mean(act) if act else None, "n_gpu_active": len(act),
            "ref_match_gen_ids": sum(bool(R[i].get("ref_match", {}).get("gen_ids")) for i in keys),
            "correct": sum(bool(R[i].get("correct")) for i in keys),
            "per_gpu_mean_seconds": {k: round(st.mean(v), 3) for k, v in sorted(by.items())}}
out = {"set": a.set, "here_dir": here, "p4_dir": p4, "n_here": len(H), "n_p4": len(P), "n_common": len(common),
       "here_all": stats(H, sorted(H)), "p4_all": stats(P, sorted(P))}
if common:
    out["here_common"] = stats(H, common); out["p4_common"] = stats(P, common)
    out["r_mean_seconds_common"] = out["here_common"]["mean_seconds"] / out["p4_common"]["mean_seconds"]
    if out["here_common"]["mean_gpu_active_s"] and out["p4_common"]["mean_gpu_active_s"]:
        out["r_gpu_active_common"] = out["here_common"]["mean_gpu_active_s"] / out["p4_common"]["mean_gpu_active_s"]
    out["same_gen_ids_as_p4"] = sum(H[i].get("gen_ids") == P[i].get("gen_ids") for i in common)
    pg = out["here_common"]["per_gpu_mean_seconds"].values(); out["here_gpu_spread"] = {"min": min(pg), "max": max(pg), "max_over_min": max(pg) / min(pg)}
out["r_mean_seconds_vs_17.4"] = (out["here_all"]["mean_seconds"] or 0) / 17.4
print(json.dumps(out, indent=1))
if a.out: json.dump(out, open(a.out, "w"), indent=1)
