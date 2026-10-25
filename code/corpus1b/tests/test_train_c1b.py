"""train_c1b end to end on CPU with the tiny SDAR stand-in: steps run, tokens are counted in corpus tokens,
milestone checkpoints and the per-domain probe are written, --resume continues, --continue-from carries the
token count into a new run (wave A -> wave B).

    python -m pytest -q code/corpus1b/tests/test_train_c1b.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
C1B = os.path.abspath(os.path.join(HERE, ".."))
CODE = os.path.abspath(os.path.join(C1B, ".."))
for p in (os.path.join(CODE, "compat"), CODE, C1B, os.path.join(CODE, "tests")):
    if p not in sys.path:
        sys.path.insert(0, p)

import test_draftalign_train as TT                                        # noqa: E402
import train_c1b as TC                                                    # noqa: E402
from draftalign.build_train_set import build_item                         # noqa: E402

BL, MASK, VOCAB = TT.BLOCK, TT.MASK, TT.VOCAB


def write_index(root, n_items=24, seed=0):
    rng = np.random.default_rng(seed)
    os.makedirs(os.path.join(root, "items"), exist_ok=True)
    entries = []
    doms = ["maths", "code", "chat"]
    for idx in range(n_items):
        pl = int(rng.integers(3, 11))
        nb = int(rng.integers(2, 5))
        lo0 = (pl // BL) * BL
        prompt = rng.integers(0, MASK, size=pl).tolist()
        flat = prompt[lo0:] + rng.integers(0, MASK, size=nb * BL - (pl - lo0)).tolist()
        tokens = np.array(flat, np.int32).reshape(nb, BL)
        valid = np.ones((nb, BL), bool)
        valid[0, :pl - lo0] = False
        steps = rng.integers(0, 3, size=(nb, BL))
        n_pass = (steps.max(1) + 1).astype(np.int32)
        committed = np.zeros((nb, 3, BL), bool)
        bi, sj = np.nonzero(valid)
        committed[bi, steps[bi, sj], sj] = True
        trace = {"tokens": tokens, "valid": valid, "committed": committed, "n_pass": n_pass,
                 "block_start": (lo0 + BL * np.arange(nb)).astype(np.int32), "prompt_length": np.int32(pl),
                 "block_length": np.int32(BL)}
        drafts = np.where(rng.random((nb, BL)) < 0.5, tokens, rng.integers(0, MASK, size=(nb, BL))).astype(np.int32)
        drafts[0, :pl - lo0] = -1
        dom = doms[idx % 3]
        item = build_item({"idx": idx, "correct": 1, "split": f"c1b_{dom}"}, trace, prompt, drafts, 64, BL)
        item["draft_conf"] = rng.integers(0, 256, size=(nb, BL)).astype(np.uint8)
        item["domain"] = np.array(dom)
        rel = os.path.join("items", f"{idx:03d}.npz")
        np.savez_compressed(os.path.join(root, rel), **item)
        entries.append({"file": rel, "idx": idx, "split": f"c1b_{dom}", "domain": dom, "pid": f"{dom}-t-{idx}",
                        "correct": 1, "usable": [bool(v) for v in item["usable"]],
                        "has_draft": [bool(v) for v in item["has_draft"]],
                        "block_start": [int(v) for v in item["block_start"]], "seq_len": lo0 + nb * BL,
                        "prompt_length": pl, "corpus_tokens": pl + nb * BL - (pl - lo0)})
    path = os.path.join(root, "index.jsonl")
    with open(path, "w") as fh:
        for e in entries:
            fh.write(json.dumps(e) + "\n")
    return path, sum(e["corpus_tokens"] for e in entries)


COMMON = ["--block", str(BL), "--mask-id", str(MASK), "--max-seq", "64", "--states-per-step", "0:4,60:8",
          "--max-states-micro", "3", "--holdout-per-domain", "1", "--probe-states-per-domain", "4",
          "--warmup-tokens", "20", "--total-tokens", "1000", "--log-every", "2", "--lr", "1e-3",
          "--save-every-tokens", "40", "--eval-every-tokens", "100"]


def tiny():
    return TT.build_model(0)


def test_train_resume_continue(tmp_path, monkeypatch):
    monkeypatch.setattr(TC.P, "safe_out", lambda p: p)
    idx, total = write_index(str(tmp_path / "ts"))
    out = str(tmp_path / "run")
    st = TC.main(["--index", idx, "--out", out, "--save-at-tokens", "30,90", "--max-tokens", "90"] + COMMON,
                 model_builder=tiny)
    assert st["tokens"] >= 90 and st["step"] > 0
    assert os.path.isdir(os.path.join(out, "tok_000000000030")) and os.path.isdir(os.path.join(out, "tok_000000000090"))
    probes = [json.loads(l) for l in open(os.path.join(out, "probe.jsonl"))]
    assert {p["tag"] for p in probes} >= {"tok_000000000030", "tok_000000000090", "final"}
    assert any(k.startswith("chat/") for k in probes[-1]) and "all/conf_precision" in probes[-1]
    rows = [json.loads(l) for l in open(os.path.join(out, "train.rank0.jsonl"))]
    assert rows and all(r["tokens"] > 0 for r in rows)

    # resume the same run: continues from its newest checkpoint, never re-saves a passed milestone
    st2 = TC.main(["--index", idx, "--out", out, "--resume", "--save-at-tokens", "30,90", "--max-tokens", "150"]
                  + COMMON, model_builder=tiny)
    assert st2["tokens"] >= 150 and st2["step"] > st["step"]

    # continue into a new run on a (here: the same) index, as wave B continues wave A
    out_b = str(tmp_path / "run_b")
    st3 = TC.main(["--index", idx, "--out", out_b, "--continue-from", os.path.join(out, "tok_000000000090"),
                   "--save-at-tokens", "30,90,200", "--max-tokens", "200"] + COMMON, model_builder=tiny)
    assert st3["tokens"] >= 200
    assert not os.path.exists(os.path.join(out_b, "tok_000000000030"))       # already passed in wave A
    assert os.path.isdir(os.path.join(out_b, "tok_000000000200"))


def test_ramp_and_schedule():
    ramp = TC.parse_ramp("0:32,8e6:128,32e6:512")
    assert TC.states_for_tokens(ramp, 0) == 32 and TC.states_for_tokens(ramp, 9e6) == 128
    assert TC.states_for_tokens(ramp, 5e8) == 512
    f = lambda t: TC.wsd_lr_scale(t, 2e6, 1e9, 0.15, 0.1)
    assert f(1e6) == pytest.approx(0.5) and f(128e6) == 1.0 and f(850e6) == 1.0
    assert f(1e9) == pytest.approx(0.1) and 0.1 < f(925e6) < 1.0


def test_stop_at_file_and_domain_weights(tmp_path, monkeypatch):
    """--stop-at-file overrides --stop-at and is re-read every step; a domain-weighted resume continues its own plan."""
    import time
    monkeypatch.setattr(TC.P, "safe_out", lambda p: p)
    idx, _ = write_index(str(tmp_path / "ts"))
    out, saf = str(tmp_path / "run"), str(tmp_path / "stop_at")
    TC.main(["--index", idx, "--out", out, "--save-at-tokens", "1e9", "--max-tokens", "60"] + COMMON, model_builder=tiny)
    dw = ["--domain-weights", "maths=40,code=40,chat=20", "--repeat-domains", "chat", "--max-passes", "2"]
    open(saf, "w").write("1\n")                                   # in the past: stop at the first step boundary
    st = TC.main(["--index", idx, "--out", out, "--resume", "--save-at-tokens", "1e9", "--stop-at",
                  str(time.time() + 3600), "--stop-at-file", saf] + dw + COMMON, model_builder=tiny)
    assert st["dw_step"] == 1
    open(saf, "w").write(f"{time.time() + 3600}\n")                # moved later: the run goes on to --max-tokens
    st2 = TC.main(["--index", idx, "--out", out, "--resume", "--save-at-tokens", "1e9", "--stop-at", "1",
                   "--stop-at-file", saf, "--max-tokens", str(st["tokens"] + 40)] + dw + COMMON, model_builder=tiny)
    assert st2["dw_step"] > 1 and st2["tokens"] >= st["tokens"] + 40
    assert st2["dw_from_step"] == st["dw_from_step"] and sum(st2["dom_tokens"].values()) > 0
