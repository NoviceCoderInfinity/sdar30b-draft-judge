#!/usr/bin/env python
"""Training index for the 128M -> 228M continuation (CPU only; run from the repo root after the data is converted).

    /venv/train/bin/python code/corpus1b/build_index_228m.py

What goes in (file paths relative to logs/c1b/trainset/, exactly like index_128m.jsonl):
  * NEW wave-B items: pids in logs/c1b/trainset/waveB/index.jsonl that are not in index_128m (the wave-B code generated
    on 30 Sep, never converted then, plus everything this continuation generates);
  * maths / code / chat items of index_128m that main_128m never trained on: an exact CPU replay of main_128m's plan up
    to its 128M checkpoint (the same functions train_c1b.py ran, as ops/replay_seen.py does for main). The replay must
    reproduce the trainer's own token count to 0.5%; if it does not, this part is left out and the script says so;
  * every science and IF item of waves A and B: those prompt pools are fully generated, so their 15% + 15% of the
    mix are further passes (train_c1b.py --repeat-domains science,if --max-passes 4, as in every earlier run);
  * the 1,000 pinned probe items (holdout_pids_waveA.txt), so --holdout-pids keeps the probe identical and untrained.
Writes logs/c1b/trainset/index_228m.jsonl and logs/c1b/train/main_128m/seen_items.json; prints tokens per domain.
"""
import collections as C
import json
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, "code/corpus1b")
sys.path.insert(0, "code")
import numpy as np                                                   # noqa: E402
import train_c1b as T                                                # noqa: E402 (also sets up the draftalign path)
from draftalign.dataset import SamplerConfig, StateSampler, domain_of_entry   # noqa: E402

TS = "logs/c1b/trainset"
RUN = "logs/c1b/train/main_128m"
CKPT = os.environ.get("C228_FROM", f"{RUN}/tok_000128000000")
REPEAT = ("science", "if")
FRESH = ("maths", "code", "chat")


def read(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def replay_main_128m():
    """pid -> tokens main_128m trained on that item (0 if never), and whether the replay matches the trainer."""
    cfg = json.load(open(f"{RUN}/config.json"))
    st = json.load(open(f"{CKPT}/trainer_state.json"))
    scfg = SamplerConfig(index_path=cfg["index"], seed=cfg["seed"], p_pass1=cfg["p_pass1"], p_allmask=cfg["p_allmask"],
                         p_draftfull=cfg["p_draftfull"], mask_id=cfg["mask_id"], block=cfg["block"], max_seq=cfg["max_seq"])
    full = StateSampler(scfg)
    hold = set(open(cfg["holdout_pids"]).read().split())
    tr = StateSampler(scfg, [e for e in full.entries if e["pid"] not in hold], full.root)
    mean_share = float(np.mean([tr.tokens_share(i) for i, _b in tr.examples]))
    ramp, world, ms = T.parse_ramp(cfg["states_per_step"]), 8, cfg["max_states_micro"]
    units = T.domain_weighted_units(tr, 0, world, ramp, float(st["epoch_tokens_start"]), mean_share, ms,
                                    int(st["dw_from_step"]), T.parse_weights(cfg["domain_weights"]),
                                    [d for d in cfg["repeat_domains"].split(",") if d], int(cfg["max_passes"]))
    plans = [T.plan_units(tr, units[r::world], ramp, float(st["dw_tokens_start"]), mean_share, world, ms)
             for r in range(world)]
    n = min(len(p) for p in plans)                                   # the trainer all-reduces MIN over ranks
    seen = C.Counter()
    for p in plans:
        for step in p[:min(int(st["dw_step"]), n)]:
            for m in step:
                for k, _ps in m:
                    i = tr.examples[k][0]
                    seen[i] += tr.tokens_share(i)
    got, want = sum(seen.values()), float(st["tokens"]) - float(st["dw_tokens_start"])
    ok = abs(got - want) <= 0.005 * want
    print(f"[replay] main_128m: replay {got / 1e6:.3f}M vs trainer {want / 1e6:.3f}M -> {'OK' if ok else 'MISMATCH'}",
          flush=True)
    out = {tr.entries[i]["pid"]: seen.get(i, 0.0) for i in range(len(tr.entries))}
    by = C.Counter()
    for i, e in enumerate(tr.entries):
        if seen.get(i, 0.0) == 0.0:
            by[domain_of_entry(e)] += e.get("corpus_tokens", 0)
    print("[replay] untrained corpus tokens left in index_128m: "
          + ", ".join(f"{d} {t / 1e6:.2f}M" for d, t in sorted(by.items())), flush=True)
    json.dump({"trainer_tokens": float(st["tokens"]), "replay_tokens": got + float(st["dw_tokens_start"]),
               "replay_ok": ok, "items": {p: {"seen_tokens": t} for p, t in out.items()}},
              open(f"{RUN}/seen_items.json", "w"))
    return out, ok


def main():
    A = read(f"{TS}/waveA/index.jsonl")
    B = read(f"{TS}/waveB/index.jsonl")
    I128 = read(f"{TS}/index_128m.jsonl")
    hold = set(open(f"{TS}/holdout_pids_waveA.txt").read().split())
    in128 = {e["pid"] for e in I128}
    seen, ok = replay_main_128m()

    out, pids = [], set()
    stat = C.defaultdict(C.Counter)

    def add(e, prefix, kind):
        if e["pid"] in pids:
            return
        e = dict(e)
        if not e["file"].startswith(("waveA/", "waveB/")):
            e["file"] = prefix + e["file"]
        out.append(e)
        pids.add(e["pid"])
        d = domain_of_entry(e)
        stat[d][kind] += e.get("corpus_tokens", 0)
        stat[d][kind + "_n"] += 1

    for e in I128:                                                   # 1. the pinned probe items (never trained)
        if e["pid"] in hold:
            add(e, "", "holdout")
    for e in B:                                                      # 2. new wave-B items
        if e["pid"] not in in128 and e["pid"] not in hold:
            add(e, "waveB/", "new")
    if ok:                                                           # 3. maths/code/chat main_128m never trained on
        for e in I128:
            if e["pid"] not in hold and domain_of_entry(e) in FRESH and seen.get(e["pid"], 0.0) == 0.0:
                add(e, "", "untrained_128m")
    else:
        print("[index] WARNING: replay mismatch -> the untrained remainder of index_128m is left out", flush=True)
    for w, rows in (("waveA/", A), ("waveB/", B)):                   # 4. science + IF of both waves (further passes)
        for e in rows:
            if e["pid"] not in hold and domain_of_entry(e) in REPEAT:
                add(e, w, "repeat")

    assert len(pids) == len(out), "duplicate pid in the merged index"
    with open(f"{TS}/index_228m.jsonl", "w") as fh:
        for e in out:
            fh.write(json.dumps(e) + "\n")
    tot = C.Counter()
    print(f"{'domain':8s} {'new':>14s} {'untrained@128M':>16s} {'science/IF pool':>16s} {'holdout':>8s}")
    for d in sorted(stat):
        s = stat[d]
        tot.update(s)
        print(f"{d:8s} {s['new'] / 1e6:8.2f}M ({s['new_n']:6d}) {s['untrained_128m'] / 1e6:9.2f}M "
              f"{s['repeat'] / 1e6:14.2f}M {s['holdout_n']:8d}")
    print(f"[index] {TS}/index_228m.jsonl: {len(out)} items; new {tot['new'] / 1e6:.2f}M, untrained@128M "
          f"{tot['untrained_128m'] / 1e6:.2f}M, science/IF pool {tot['repeat'] / 1e6:.2f}M, holdout {tot['holdout_n']}",
          flush=True)


if __name__ == "__main__":
    main()
