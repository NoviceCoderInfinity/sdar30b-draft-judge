"""train_p2 end to end on CPU with the tiny SDAR stand-in: per-step rows, the expert windows, the extended probe,
the best checkpoint, resume, and a stop rule ending the run at the step it fires (exit 3, `stopped/`).

    python -m pytest -q code/phase2/tests/test_train_p2.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.abspath(os.path.join(HERE, "..", ".."))
for p in (os.path.join(CODE, "compat"), CODE, os.path.join(CODE, "corpus1b"), os.path.join(CODE, "corpus1b", "tests"),
          os.path.join(CODE, "tests")):
    if p not in sys.path:
        sys.path.insert(0, p)

import test_draftalign_train as TT                                        # noqa: E402
import test_train_c1b as TC1                                              # noqa: E402  (write_index)
from phase2 import stoprules as SR                                        # noqa: E402
from phase2 import train_p2 as TP                                         # noqa: E402

BL, MASK = TT.BLOCK, TT.MASK
COMMON = ["--drafter-key", "S1", "--block", str(BL), "--mask-id", str(MASK), "--max-seq", "64",
          "--states-per-step", "0:4", "--max-states-micro", "3", "--holdout-mode", "domain",
          "--holdout-per-domain", "1", "--probe-states-per-domain", "4", "--warmup-steps", "2", "--log-every", "2",
          "--lr", "1e-3", "--probe-every-steps", "3", "--save-every-steps", "4"]


def tiny():
    return TT.build_model(0)


def test_train_p2_logs_every_step_and_resumes(tmp_path, monkeypatch):
    monkeypatch.setattr(TP.P, "safe_out", lambda p: p)
    monkeypatch.setattr(SR, "GLOBAL_STOP", str(tmp_path / "nope"))
    idx, _ = TC1.write_index(str(tmp_path / "ts"))
    out = str(tmp_path / "run")
    rc = TP.main(["--index", idx, "--out", out, "--steps", "7"] + COMMON, model_builder=tiny)
    assert rc == 0
    steps = [json.loads(l) for l in open(os.path.join(out, "steps.jsonl"))]
    assert [s["step"] for s in steps] == list(range(1, 8))                     # one row per optimizer step
    assert all(np.isfinite(s["loss"]) and s["grad_norm"] >= 0 for s in steps)
    rows = [json.loads(l) for l in open(os.path.join(out, "train.rank0.jsonl"))]
    assert rows and "experts/block_uniq_per_layer" in rows[0] and "all/loss" in rows[0]
    assert sorted(os.listdir(os.path.join(out, "experts")))[0] == "step_0000002.npz"
    z = np.load(os.path.join(out, "experts", "step_0000002.npz"))
    assert z["block"].sum() > 0 and z["prefix"].shape == z["block"].shape
    probes = [json.loads(l) for l in open(os.path.join(out, "probe.jsonl"))]
    assert [p["tag"] for p in probes] == ["start", "periodic", "periodic", "final"]
    assert {"all/keep_rate", "all/fix_rate", "all/ece", "all/conf_precision", "all/draft_acc"} <= set(probes[-1])
    assert os.path.isdir(os.path.join(out, "final")) and os.path.isdir(os.path.join(out, "step_000004"))
    cfg = json.load(open(os.path.join(out, "config.json")))
    assert cfg["rules_sha256"] == SR.rules_sha()
    rc = TP.main(["--index", idx, "--out", out, "--steps", "9", "--resume"] + COMMON, model_builder=tiny)
    assert rc == 0
    steps = [json.loads(l) for l in open(os.path.join(out, "steps.jsonl"))]
    assert steps[-1]["step"] == 9 and [s["step"] for s in steps[7:]] == [5, 6, 7, 8, 9]   # resumed from step_000004


def test_train_p2_stop_rule_ends_the_run(tmp_path, monkeypatch):
    monkeypatch.setattr(TP.P, "safe_out", lambda p: p)
    monkeypatch.setattr(SR, "GLOBAL_STOP", str(tmp_path / "nope"))
    rules = SR.load_rules()
    rules.pop("_sha256")
    rules["stop"]["train"] = [{"id": "always_after_3", "metric": "gpu_mem_frac", "op": ">=", "value": 0.0,
                               "min_n": 3, "count": "steps", "action": "stop", "why": "test"}]
    rp = tmp_path / "rules.json"
    rp.write_text(json.dumps(rules))
    idx, _ = TC1.write_index(str(tmp_path / "ts"))
    out = str(tmp_path / "run")
    rc = TP.main(["--index", idx, "--out", out, "--steps", "20", "--rules", str(rp)] + COMMON, model_builder=tiny)
    assert rc == SR.STOP_EXIT
    st = json.load(open(os.path.join(out, "STOPPED.json")))
    assert st["reason"] == "always_after_3" and st["counts"]["steps"] == 3
    assert os.path.isdir(os.path.join(out, "stopped")) and not os.path.isdir(os.path.join(out, "final"))
    assert len(open(os.path.join(out, "steps.jsonl")).readlines()) == 3
