"""CPU-only replay of the trainer's plan -> which items/tokens the final checkpoint has trained on.
Rebuilds exactly what train_c1b.py builds: original plan steps [0, dw_from_step) on 8 ranks, then the domain-weighted
units' plan steps [0, dw_step). Writes logs/c1b/train/main/seen_items.json. Sanity: total must equal final tokens."""
import json, os, sys, collections as C
os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, "code/corpus1b")
import train_c1b as T
from draftalign.dataset import SamplerConfig, StateSampler, domain_of_entry
cfg = json.load(open("logs/c1b/train/main/config.json"))
st = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "logs/c1b/train/main/final/trainer_state.json"))
W = 8
full = StateSampler(SamplerConfig(index_path=cfg["index"], seed=cfg["seed"], p_pass1=cfg["p_pass1"], p_allmask=cfg["p_allmask"],
                                  p_draftfull=cfg["p_draftfull"], mask_id=cfg["mask_id"], block=cfg["block"], max_seq=cfg["max_seq"]))
tr, _ev = full.split_holdout_by_domain(cfg["holdout_per_domain"])
import numpy as np
mean_share = float(np.mean([tr.tokens_share(i) for i, _ in tr.examples]))
ramp = T.parse_ramp(cfg["states_per_step"]); ms = cfg["max_states_micro"]
tok0 = float(st.get("epoch_tokens_start", 0.0))
seen = C.Counter(); passes = C.defaultdict(set)
plans = [T.plan_epoch(tr, 0, r, W, ramp, tok0, mean_share, ms) for r in range(W)]
n = min(len(p) for p in plans); s0 = min(st["dw_from_step"], n)
for p in plans:
    for step in p[:s0]:
        for m in step:
            for k in m:
                i = tr.examples[k][0]; seen[i] += tr.tokens_share(i); passes[i].add(0)
t_orig = sum(seen.values())
units = T.domain_weighted_units(tr, 0, W, ramp, tok0, mean_share, ms, st["dw_from_step"], T.parse_weights(cfg["domain_weights"]),
                                [d for d in cfg["repeat_domains"].split(",") if d], cfg["max_passes"])
dplans = [T.plan_units(tr, units[r::W], ramp, st["dw_tokens_start"], mean_share, W, ms) for r in range(W)]
nd = min(len(p) for p in dplans)
for p in dplans:
    for step in p[:min(st["dw_step"], nd)]:
        for m in step:
            for k, ps in m:
                i = tr.examples[k][0]; seen[i] += tr.tokens_share(i); passes[i].add(ps)
tot = sum(seen.values())
print(f"replay: original plan {t_orig/1e6:.3f}M (steps<{s0}), weighted {(tot-t_orig)/1e6:.3f}M (dw_step {st['dw_step']}), "
      f"total {tot/1e6:.3f}M vs trainer {st['tokens']/1e6:.3f}M", flush=True)
out = {}; BY = tr.item_examples()
for i, e in enumerate(tr.entries):
    out[e["pid"]] = {"domain": domain_of_entry(e), "source": e.get("source"), "item_tokens": tr.tokens_share(i) * len(BY.get(i, ())),
                     "seen_tokens": seen.get(i, 0.0), "passes": len(passes.get(i, ()))}
json.dump({"trainer_tokens": st["tokens"], "replay_tokens": tot, "items": out}, open("logs/c1b/train/main/seen_items.json", "w"))
print("wrote logs/c1b/train/main/seen_items.json", len(out), "train items")
