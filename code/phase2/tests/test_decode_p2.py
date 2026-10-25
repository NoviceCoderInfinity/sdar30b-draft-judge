"""The Phase 2 decode path on CPU: the real `da_decode.decode_item` with an instrumented fake MoE behind the
expert-logging meter and a Phase 2 drafter wrapper (a tiny Qwen3 behind `CachedBlockCausal`), then the per-item
hooks: record fields, the per-item expert npz, cell-wide running metrics across shard files, and the futility
stop against an S0 cell.

    python -m pytest -q code/phase2/tests/test_decode_p2.py
"""
from __future__ import annotations

import json
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.abspath(os.path.join(HERE, "..", ".."))
for p in (os.path.join(CODE, "compat"), CODE, os.path.join(CODE, "tests"), os.path.join(CODE, "..")):
    if p not in sys.path:
        sys.path.insert(0, p)

from draftalign.seqcache import SeqCache                                  # noqa: E402
from draftalign.da_decode import decode_item                              # noqa: E402
from phase2 import drafters as D                                          # noqa: E402
from phase2 import stoprules as SR                                        # noqa: E402
from phase2.decode_hooks import DecodeHooks                               # noqa: E402
from phase2.explog import make_meter_class                                # noqa: E402

V, MASK, EOS = 64, 63, [62, 61]
BL = 32


class Tok:
    def decode(self, ids, skip_special_tokens=True):
        return " ".join(str(i) for i in ids)


def tiny_drafter():
    from transformers import Qwen3Config, Qwen3ForCausalLM
    torch.manual_seed(1)
    cfg = Qwen3Config(vocab_size=V, hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=2, head_dim=8, max_position_embeddings=512)
    cfg._attn_implementation = "sdpa"
    return Qwen3ForCausalLM(cfg).eval()


def _hooks(tmp_path, out, dest, split, s0_dir=None, rules=None):
    return DecodeHooks(SimpleNamespace(), str(out), str(dest), split, is_stock=False, drafter_key="S1",
                       drafter_eq=D.cost_per_forward("S1"), s0_dir=s0_dir, stock_dir=None, expert_level="bits",
                       rules_path=rules, integrity=False, cell="c")


def test_decode_item_with_expert_meter_and_phase2_drafter(tmp_path, monkeypatch):
    from tests.test_draftalign import FakeMoEModel
    from sdar_harness.moe_instr import enable_instrumented_moe, disable_instrumented_moe, SINK
    monkeypatch.setattr(SR, "GLOBAL_STOP", str(tmp_path / "nope"))
    model = FakeMoEModel(vocab=V)
    L = enable_instrumented_moe(model)
    SINK.disarm(); SINK.verify = False
    try:
        M = make_meter_class()
        moe = M(SeqCache(model, BL, "moe"), SINK, level="bits", n_layers=L, n_experts=model.config.num_experts)
        dr = D.Drafter("S1", tiny_drafter(), None, D.CachedBlockCausal, "tiny", name="drafter-S1").as_seq(BL)
        out = tmp_path / "cell"; out.mkdir()
        dest = out / "c.split.shard0of2.jsonl"
        hooks = _hooks(tmp_path, out, dest, "split")
        ids = torch.randint(0, 60, (1, 37))
        rec, _res = decode_item(moe, dr, Tok(), {"idx": 3, "split": "train", "gold": "g"}, ids, mask_id=MASK,
                                eos_ids=EOS, gen_length=64, block_length=BL, threshold=.95, draft="s8", judge="blind",
                                tau_acc_q=.3)
        rec["correct"] = 1
        stop = hooks.after_item(rec, moe)
        assert not stop
        assert rec["exp_block_uniq_per_layer"] > 0 and "denoise" in rec["exp_by_kind"]
        assert rec["drafter_key"] == "S1" and rec["drafter_eq_per_forward"] == 126.32
        n_dr = sum(rec["drafter_n_by_kind"].values())
        assert n_dr > 0 and rec["drafter_n_by_kind"]["draft"] > 0
        assert abs(rec["charged_exp_per_token"] - (rec["unique_expert_layer_loads"] + n_dr * 126.32)
                   / rec["n_gen_to_eos"]) < 1e-2
        z = np.load(out / "experts" / "split" / "00003.npz")
        assert z["hit_bits"].shape[0] == rec["n_forward"] and z["hit_bits"].shape[1] == L
        assert int(np.unpackbits(z["hit_bits"], axis=-1)[..., :model.config.num_experts].sum()) == rec["unique_expert_layer_loads"]
    finally:
        disable_instrumented_moe(model)


def _rec(idx, correct, **kw):
    r = {"idx": idx, "correct": correct, "n_gen_to_eos": 100, "n_blocks": 4, "unique_expert_layer_loads": 50000,
         "n_forward": 20, "n_prefill": 1, "n_flush": 4, "drafter_n_by_kind": {"draft": 32, "flush": 4, "prefill": 1},
         "draft_agreement_per_block": [0.5] * 4, "committed_judge_per_block": [20] * 4, "tentative_per_block": [5] * 4,
         "remask_per_block": [1] * 4, "full_fallback_per_block": [0] * 4, "truncated": False, "seconds": 10.0}
    r.update(kw)
    return r


def test_running_metrics_span_every_shard_and_futility_stops(tmp_path, monkeypatch):
    monkeypatch.setattr(SR, "GLOBAL_STOP", str(tmp_path / "nope"))
    s0 = tmp_path / "s0"; s0.mkdir()
    with open(s0 / "x.split.shard0of1.jsonl", "w") as fh:
        for i in range(200):
            fh.write(json.dumps(_rec(i, 1)) + "\n")                      # S0 gets everything right
    out = tmp_path / "cell"; out.mkdir()
    # another shard has already written 70 wrong items
    with open(out / "c.split.shard1of2.jsonl", "w") as fh:
        for i in range(0, 140, 2):
            fh.write(json.dumps(_rec(i, 0)) + "\n")
    dest = out / "c.split.shard0of2.jsonl"
    h = _hooks(tmp_path, out, dest, "split", s0_dir=str(s0), rules="default")
    assert h.mon.refs["s0_agreement"] == 0.5
    stop = h.after_item(_rec(1, 0), SimpleNamespace())
    m = h.last
    assert m["items"] == 71 and m["paired_n"] == 71                       # the whole cell, not just this shard
    assert m["delta_acc_vs_s0"] == -1.0 and m["delta_acc_ci99_hi"] < -0.10
    assert stop and h.mon.stopped["reason"] == "futility"
    assert abs(m["charged_exp_per_token"] - (50000 + 37 * 126.32) / 100) < 1e-6
    code = h.stopped(str(dest) + ".stopped")
    assert code == SR.STOP_EXIT and json.load(open(str(dest) + ".stopped"))["stopped"]
