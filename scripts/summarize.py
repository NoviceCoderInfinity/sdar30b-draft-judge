"""Summary table for decode outputs: accuracy, 30B passes per block, experts per token, seconds per answer.

    python scripts/summarize.py runs/phase4-rd1            # every benchmark folder under it
    python scripts/summarize.py runs/phase4-rd1 runs/vanilla

Definitions (the ones every table in RESULTS.md uses):
    passes/blk = sum(passes_per_block) / sum(n_blocks)
    exp/token  = sum(unique_expert_layer_loads) / sum(n_gen_to_eos)
    s/answer   = median of per-item wall-clock seconds
"""
import glob, json, os, statistics, sys


def load(d):
    recs = {}
    for f in glob.glob(os.path.join(d, "**", "*.jsonl"), recursive=True):
        if ".shard" not in os.path.basename(f) or f.endswith(".running.jsonl"):
            continue
        for line in open(f):
            if line.strip():
                r = json.loads(line)
                if "idx" in r and "n_blocks" in r:
                    recs[r["idx"]] = r
    return list(recs.values())


def row(name, rs):
    tok = sum(r["n_gen_to_eos"] for r in rs) or 1
    pb = sum(sum(r["passes_per_block"]) for r in rs) / max(1, sum(r["n_blocks"] for r in rs))
    return (f"| {name} | {len(rs)} | {100 * sum(bool(r.get('correct')) for r in rs) / len(rs):.1f} | {pb:.2f} | "
            f"{sum(r['unique_expert_layer_loads'] for r in rs) / tok:.0f} | {statistics.median(r['seconds'] for r in rs):.1f} |")


for root in sys.argv[1:]:
    print(f"\n{root}\n\n| benchmark | items | accuracy % | passes/blk | exp/token | median s/answer |\n|---|---|---|---|---|---|")
    pooled = []
    for d in sorted(p for p in glob.glob(os.path.join(root, "*")) if os.path.isdir(p)):
        rs = load(d)
        if rs:
            print(row(os.path.basename(d), rs))
            pooled += rs
    if pooled:
        print(row("**all**", pooled))
