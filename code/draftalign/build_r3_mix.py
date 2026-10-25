#!/usr/bin/env python
"""Round-3 final training index: cap each pool at its block share (recipe §2: code 11,200 / maths 11,200 /
IF 9,600 = 35/35/30) and concatenate -> logs/draftalign/trainset_r3/index.jsonl (+ mix_manifest.json).

Items are taken in each pool's own index order (the decode/pool order is a uniform sample, so any prefix is
too) until the pool's cumulative `usable_blocks` first reaches its cap; caps land at item granularity.
`file` paths are rewritten relative to the merged index's directory (StateSampler root = dirname(index)).
"""
import json, os, sys, hashlib

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
TS = os.path.join(ROOT, "logs", "draftalign", "trainset_r3")
CAPS = {"opencode": 8400, "gsm8k": 8400, "if": 7200}   # option B (2026-09-14 21:45 IST): 24k blocks at 35/35/30, 4 epochs


def main():
    out, manifest = [], {}
    for pool, cap in CAPS.items():
        idx = os.path.join(TS, pool, "index.jsonl")
        if not os.path.exists(idx):
            sys.exit(f"missing {idx}")
        got = 0; n = 0; avail = 0
        for line in open(idx):
            e = json.loads(line); avail += e["usable_blocks"]
            if got >= cap:
                continue
            e["file"] = os.path.join(pool, e["file"]); e["source"] = pool
            out.append(e); got += e["usable_blocks"]; n += 1
        manifest[pool] = {"cap": cap, "blocks": got, "items": n, "available_blocks": avail}
        if got < cap:
            print(f"[mix] WARNING {pool}: only {got} usable blocks available, cap {cap}", flush=True)
    tot = sum(m["blocks"] for m in manifest.values())
    for m in manifest.values():
        m["share"] = round(m["blocks"] / tot, 4)
    path = os.path.join(TS, "index.jsonl")
    with open(path, "w") as fh:
        for e in out:
            fh.write(json.dumps(e) + "\n")
    manifest["total_blocks"] = tot; manifest["items"] = len(out)
    manifest["index_sha256"] = hashlib.sha256(open(path, "rb").read()).hexdigest()
    json.dump(manifest, open(os.path.join(TS, "mix_manifest.json"), "w"), indent=2)
    print(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
