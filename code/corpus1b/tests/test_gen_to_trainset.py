"""gen_sglang.assemble (answers rebuilt from the SGLang plugin's trace) and to_trainset (kept records + drafts ->
the build_train_set item format), end to end into draftalign.dataset.StateSampler. CPU only, no SGLang, no model.

    python -m pytest -q code/corpus1b/tests/test_gen_to_trainset.py
"""
from __future__ import annotations

import gzip
import json
import os
import sys

import numpy as np
import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
C1B = os.path.abspath(os.path.join(HERE, ".."))
CODE = os.path.abspath(os.path.join(C1B, ".."))
for p in (os.path.join(CODE, "compat"), CODE, C1B):
    if p not in sys.path:
        sys.path.insert(0, p)

import common as C                                                        # noqa: E402
import gen_sglang as G                                                    # noqa: E402
import to_trainset as TS                                                  # noqa: E402
from draftalign.dataset import SamplerConfig, StateSampler                # noqa: E402

EOS = C.EOS_IDS[0]
BL = C.BLOCK


def fake_trace(prompt_ids, gen, seed=0):
    """What LowConfidenceTraced writes for one request: block rows keyed by lo, with commit passes."""
    rng = np.random.default_rng(seed)
    L = len(prompt_ids)
    lo0 = (L // BL) * BL
    tail = L - lo0
    flat = list(prompt_ids[lo0:]) + list(gen)
    rows = {}
    for b in range((len(flat) + BL - 1) // BL):
        ids = flat[b * BL:(b + 1) * BL]
        ids = ids + [EOS] * (BL - len(ids))                           # the last block is always full in SGLang
        fixed = tail if b == 0 else 0
        steps = [-1] * fixed + list(rng.integers(0, 6, size=BL - fixed))
        rows[lo0 + b * BL] = {"rid": "x", "lo": lo0 + b * BL, "start": fixed, "ids": ids, "steps": steps}
    return rows


def test_assemble_stops_at_eos_and_checks_the_tail():
    prompt = list(range(100, 140))                                     # 40 tokens: tail 8 in block 0
    gen = list(range(200, 260)) + [EOS] + list(range(300, 320))        # EOS at generated position 60
    rows = fake_trace(prompt, gen)
    res = G.assemble(rows, prompt, cap=1024)
    assert res["eos"] and res["n_gen"] == 61 and res["gen_ids"][-1] == EOS
    assert res["lo0"] == 32 and len(res["blocks"]) == 3                # 8 + 61 = 69 -> blocks at 32, 64, 96
    assert len(res["ppb"]) == 3
    bad = {k: dict(v) for k, v in rows.items()}
    bad[32] = dict(bad[32], ids=[0] * BL)                              # block 0 must start with the prompt tail
    with pytest.raises(G.TraceError):
        G.assemble(bad, prompt, cap=1024)
    gap = {k: v for k, v in rows.items() if k != 64}                   # a missing block ends the answer early
    res2 = G.assemble(gap, prompt, cap=1024)
    assert not res2["eos"] and len(res2["blocks"]) == 1


def test_assemble_respects_the_cap():
    prompt = list(range(100, 132))                                     # tail 0
    gen = list(range(200, 300))                                        # no EOS
    res = G.assemble(fake_trace(prompt, gen), prompt, cap=64)
    assert not res["eos"] and res["n_gen"] == 64


def write_records(tmp, recs):
    d = os.path.join(tmp, "gen", "gpu0")
    os.makedirs(d, exist_ok=True)
    with gzip.open(os.path.join(d, "records.00000.jsonl.gz"), "wt") as fh:
        for r in recs:
            fh.write(json.dumps(r) + "\n")


def kept_record(pid, domain, prompt, gen, seed):
    res = G.assemble(fake_trace(prompt, gen, seed), prompt, cap=1024)
    return {"pid": pid, "domain": domain, "source": "test", "wave": "A", "prompt_len": len(prompt),
            "gen_cap": 1024, "n_gen": res["n_gen"], "eos": res["eos"], "ppb": res["ppb"], "kept": True,
            "reason": "", "loop": False, "prompt_ids": prompt, "blocks": res["blocks"], "lo0": res["lo0"]}


def test_to_trainset_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setattr(TS.P, "safe_out", lambda p: p)
    tmp = str(tmp_path)
    recs = [kept_record("maths-t-0001", "maths", list(range(100, 140)), list(range(200, 260)) + [EOS], 1),
            kept_record("chat-t-0002", "chat", list(range(100, 164)), list(range(300, 330)) + [EOS], 2)]
    recs.append(dict(recs[0], pid="maths-t-0003", kept=False, reason="wrong"))   # dropped rows are ignored
    write_records(tmp, recs)
    # drafts exactly as draft_sglang packs them
    pids, nbs, drafts, confs = [], [], [], []
    for r in recs[:2]:
        nb = len(r["blocks"])
        d = np.array([b["ids"] for b in r["blocks"]], np.int32)
        d[0, :len(r["prompt_ids"]) - r["lo0"]] = -1
        pids.append(r["pid"]); nbs.append(nb); drafts.append(d)
        confs.append(np.full((nb, BL), 200, np.uint8))
    os.makedirs(os.path.join(tmp, "draft", "gpu0"))
    np.savez_compressed(os.path.join(tmp, "draft", "gpu0", "drafts.00000.npz"),
                        pids=np.array(pids, dtype=object), nb=np.array(nbs, np.int32),
                        draft=np.concatenate(drafts), conf=np.concatenate(confs))
    out = os.path.join(tmp, "trainset")
    s = TS.main(["--wave", "A", "--gen", os.path.join(tmp, "gen"), "--draft", os.path.join(tmp, "draft"),
                 "--out", out])
    assert s["counts"]["written"] == 2 and s["counts"]["bad"] == 0
    entries = [json.loads(l) for l in open(os.path.join(out, "index.jsonl"))]
    assert {e["domain"] for e in entries} == {"maths", "chat"}
    assert entries[0]["corpus_tokens"] == 40 + 61
    it = np.load(os.path.join(out, entries[0]["file"]))
    assert it["valid"][0, :8].sum() == 0                                 # the prompt tail is never trained
    flat_valid = it["valid"].reshape(-1)
    assert flat_valid.sum() == 61                                        # nothing after the EOS
    assert it["draft_conf"].dtype == np.uint8 and str(it["domain"]) == "maths"
    sampler = StateSampler(SamplerConfig(index_path=os.path.join(out, "index.jsonl"), seed=0, p_draftfull=0.3,
                                         max_seq=1856))
    assert len(sampler) == sum(e["usable_blocks"] for e in entries)
    st = sampler.state(0, regime="draft_pass1")
    assert st.meta["domain"] in ("maths", "chat") and st.draft_conf is not None
    assert int(st.draft_conf[~(st.open_mask & (st.input_ids[st.active_lo:st.active_lo + BL] != C.MASK_ID))].sum()) == 0
    tr, ev = sampler.split_holdout_by_domain(1)
    assert len(ev.entries) == 2 and len(tr.entries) == 0
