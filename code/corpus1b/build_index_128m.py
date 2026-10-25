"""Merged training index for the 28.5M -> 128M continuation (CPU only).
  * wave A items NOT yet trained (exact replay: logs/c1b/train/main/seen_items.json, seen_tokens == 0),
  * + the original 1,000 held-out probe items (pinned via --holdout-pids so the probe stays identical and untrained),
  * + every wave-B item (logs/c1b/trainset/waveB/index.jsonl, if it exists yet).
Writes logs/c1b/trainset/index_128m.jsonl ('file' made relative to logs/c1b/trainset/) and
logs/c1b/trainset/holdout_pids_waveA.txt. Usage: train_c1b.py --index logs/c1b/trainset/index_128m.jsonl
--holdout-pids logs/c1b/trainset/holdout_pids_waveA.txt --continue-from logs/c1b/train/main/final ..."""
import json, os, sys, collections as C
os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, "code/corpus1b"); sys.path.insert(0, "code")
import train_c1b as T                                               # noqa: E402 (sets up the draftalign path)
from draftalign.dataset import SamplerConfig, StateSampler, domain_of_entry
cfg = json.load(open("logs/c1b/train/main/config.json"))
full = StateSampler(SamplerConfig(index_path="logs/c1b/trainset/waveA/index.jsonl", seed=cfg["seed"], p_pass1=cfg["p_pass1"],
                                  p_allmask=cfg["p_allmask"], p_draftfull=cfg["p_draftfull"], mask_id=cfg["mask_id"],
                                  block=cfg["block"], max_seq=cfg["max_seq"]))
_tr, ev = full.split_holdout_by_domain(cfg["holdout_per_domain"])
hold = {e["pid"] for e in ev.entries}
seen = {p for p, v in json.load(open("logs/c1b/train/main/seen_items.json"))["items"].items() if v["seen_tokens"] > 0}
out, stat = [], C.defaultdict(lambda: C.Counter())
for l in open("logs/c1b/trainset/waveA/index.jsonl"):
    e = json.loads(l); d = domain_of_entry(e)
    if e["pid"] in seen and e["pid"] not in hold:
        stat[d]["A_trained_excluded"] += 1; continue
    e["file"] = "waveA/" + e["file"]; out.append(e)
    stat[d]["A_holdout" if e["pid"] in hold else "A_unseen"] += 1; stat[d]["tok_A" if e["pid"] not in hold else "tok_hold"] += e["corpus_tokens"]
bpath = "logs/c1b/trainset/waveB/index.jsonl"
if os.path.exists(bpath):
    for l in open(bpath):
        e = json.loads(l); d = domain_of_entry(e); e["file"] = "waveB/" + e["file"]; out.append(e)
        stat[d]["B"] += 1; stat[d]["tok_B"] += e["corpus_tokens"]
assert len(hold) == len(ev.entries) and not (seen & hold), "holdout items must be untrained"
with open("logs/c1b/trainset/index_128m.jsonl", "w") as fh:
    for e in out: fh.write(json.dumps(e) + "\n")
open("logs/c1b/trainset/holdout_pids_waveA.txt", "w").write("\n".join(sorted(hold)) + "\n")
tot = C.Counter()
for d, s in sorted(stat.items()):
    tot.update(s); print(f"{d:8s} A unseen {s['A_unseen']:6d} ({s['tok_A']/1e6:6.2f}M)  A trained, excluded {s['A_trained_excluded']:6d}  "
                         f"holdout {s['A_holdout']:4d}  B {s['B']:6d} ({s['tok_B']/1e6:6.2f}M)")
print(f"TOTAL train tokens available: A unseen {tot['tok_A']/1e6:.2f}M + B {tot['tok_B']/1e6:.2f}M = {(tot['tok_A']+tot['tok_B'])/1e6:.2f}M; "
      f"holdout {len(hold)} items; index rows {len(out)}")
