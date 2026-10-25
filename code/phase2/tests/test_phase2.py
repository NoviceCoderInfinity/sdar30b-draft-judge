"""CPU tests for Phase 2: stop rules, the vocabulary map, the drafter wrappers' cache / shift semantics, the
expert-activation logger (decode and training), the re-draft item path and the job runner's bookkeeping.

    /venv/p2/bin/python -m pytest -q code/phase2/tests
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.abspath(os.path.join(HERE, "..", ".."))
for p in (os.path.join(CODE, "compat"), CODE, os.path.join(CODE, "tests")):
    if p not in sys.path:
        sys.path.insert(0, p)

from phase2 import drafters as D                                          # noqa: E402
from phase2 import stoprules as SR                                        # noqa: E402


# ---------------------------------------------------------------------------------------------------------------
# stop rules
# ---------------------------------------------------------------------------------------------------------------
def _rules(tmp_path, stage_rules):
    p = tmp_path / "rules.json"
    p.write_text(json.dumps({"version": "t", "pass_fail": {}, "stop": {"s": stage_rules}}))
    return str(p)


def test_stop_rule_fires_after_min_n_and_patience(tmp_path, monkeypatch):
    monkeypatch.setattr(SR, "GLOBAL_STOP", str(tmp_path / "nope"))
    rp = _rules(tmp_path, [{"id": "low", "metric": "x", "op": "<", "value": 0.5, "min_n": 3, "count": "items",
                            "patience": 2, "action": "stop", "why": "t"}])
    m = SR.StopMonitor("s", str(tmp_path / "o"), rules_path=rp)
    assert not m.check({"x": 0.1}, {"items": 1}).stop                     # below min_n
    assert not m.check({"x": 0.1}, {"items": 3}).stop                     # patience 1/2
    d = m.check({"x": 0.1}, {"items": 4})
    assert d.stop and d.reason == "low"
    assert (tmp_path / "o" / "STOPPED.json").exists()
    ev = [json.loads(l) for l in open(tmp_path / "o" / "stop_events.jsonl")]
    assert ev[-1]["rule"] if "rule" in ev[-1] else ev[-1]["id"] == "low"


def test_stop_rule_ref_scaling_and_missing_ref(tmp_path, monkeypatch):
    monkeypatch.setattr(SR, "GLOBAL_STOP", str(tmp_path / "nope"))
    rp = _rules(tmp_path, [{"id": "rel", "metric": "a", "op": "<", "value": 0.5, "ref": "r", "min_n": 0,
                            "action": "stop", "why": "t"}])
    m = SR.StopMonitor("s", str(tmp_path / "o"), rules_path=rp)
    assert not m.check({"a": 0.0}, {"n": 1}).stop                         # ref unknown -> skipped
    m.refs["r"] = 0.6
    assert not m.check({"a": 0.31}, {"n": 1}).stop                        # 0.31 >= 0.30
    assert m.check({"a": 0.29}, {"n": 1}).stop


def test_warn_does_not_stop_and_kill_switch_does(tmp_path, monkeypatch):
    monkeypatch.setattr(SR, "GLOBAL_STOP", str(tmp_path / "nope"))
    rp = _rules(tmp_path, [{"id": "w", "metric": "a", "op": ">", "value": 1, "min_n": 0, "action": "warn",
                            "why": "t"}])
    out = tmp_path / "o"
    m = SR.StopMonitor("s", str(out), rules_path=rp)
    d = m.check({"a": 5}, {"n": 1})
    assert not d.stop and d.fired[0]["id"] == "w"
    (out / "STOP").write_text("")
    assert m.check({"a": 0}, {"n": 1}).stop


def test_nonfinite_rule(tmp_path, monkeypatch):
    monkeypatch.setattr(SR, "GLOBAL_STOP", str(tmp_path / "nope"))
    rp = _rules(tmp_path, [{"id": "nf", "metric": "loss", "op": "nonfinite", "value": 0, "min_n": 0,
                            "action": "stop", "why": "t"}])
    m = SR.StopMonitor("s", str(tmp_path / "o"), rules_path=rp)
    assert not m.check({"loss": 1.0}, {"n": 1}).stop
    assert m.check({"loss": float("nan")}, {"n": 1}).stop


def test_frozen_rules_file_parses_and_every_rule_is_complete():
    r = SR.load_rules()
    for stage, rules in r["stop"].items():
        if stage.startswith("_"):
            continue
        for x in rules:
            assert {"id", "metric", "op", "value", "action", "why"} <= set(x), x
            assert x["action"] in ("stop", "warn")


def test_paired_delta_is_the_repo_bootstrap():
    a = [1] * 60 + [0] * 40
    b = [1] * 50 + [0] * 50
    d = SR.paired_delta(a, b)
    assert d["n"] == 100 and abs(d["delta"] - 0.10) < 1e-12 and d["lo"] <= 0.10 <= d["hi"]
    assert SR.paired_delta(a, b) == d                                     # seed 0: deterministic


# ---------------------------------------------------------------------------------------------------------------
# registry and vocabulary map
# ---------------------------------------------------------------------------------------------------------------
def test_costs_follow_the_repo_convention():
    assert D.cost_per_forward("S0") == 364.64 and D.cost_per_forward("d32") == 364.64
    assert D.cost_per_forward("F2") == 364.64                             # same Qwen3-1.7B body + untied head
    assert D.cost_per_forward("S1") == 126.32
    assert D.cost_per_forward("S2") == 104.70
    assert D.cost_per_forward("F1") == 327.16
    assert D.cost_per_forward("S6") == 852.47
    assert D.cell_tag("S0") == "" and D.cell_tag("F1").startswith("_drF1")
    with pytest.raises(KeyError):
        D.spec("F3")


class FakeTok:
    def __init__(self, tokens):
        self.tokens = tokens

    def __len__(self):
        return len(self.tokens)

    def convert_ids_to_tokens(self, ids):
        return [self.tokens[i] for i in ids]

    def get_vocab(self):
        return {t: i for i, t in enumerate(self.tokens)}


def _toks(n_base, specials):
    return [f"t{i}" for i in range(n_base)] + specials


def test_vocab_map_identity_for_same_vocab():
    moe = FakeTok(_toks(5, ["<|fim_pad|>", "<think>", "<MASK>"]))          # mask id 7
    dr = FakeTok(_toks(5, ["<|fim_pad|>", "<think>", "<|mask|>"]))         # same ids, other mask string
    vm = D.VocabMap(moe, dr, moe_mask=7, dr_mask=7, v_out=10, device="cpu")
    assert vm.identity and vm.first_diff is None and not vm.unmapped_ids


def test_vocab_map_qwen25_style_remap():
    # MoE: ... <|fim_pad|>=5, <think>=6, <MASK>=7 ; drafter (Qwen2.5-like): ... <|fim_pad|>=5, |<MASK>|=6 (its mask)
    moe = FakeTok(_toks(5, ["<|fim_pad|>", "<think>", "<MASK>"]))
    dr = FakeTok(_toks(5, ["<|fim_pad|>", "|<MASK>|"]))
    vm = D.VocabMap(moe, dr, moe_mask=7, dr_mask=6, v_out=10, device="cpu")
    assert not vm.identity and vm.unmapped_ids == [6]
    ids = torch.tensor([[0, 4, 6, 7]])
    assert vm.to_drafter(ids).tolist() == [[0, 4, 5, 6]]                  # <think> -> placeholder, MASK -> 6
    assert vm.unmapped(ids) == 1
    lg = torch.arange(10, dtype=torch.float32)[None]                      # drafter logits over its 10 rows
    out = vm.to_moe_logits(lg)
    assert out[0, :6].tolist() == [0, 1, 2, 3, 4, 5]                      # shared ids copied
    assert torch.isinf(out[0, 6:]).all()                                  # <think>, MASK, unused: never drafted


# ---------------------------------------------------------------------------------------------------------------
# wrappers: cache and shift semantics
# ---------------------------------------------------------------------------------------------------------------
def _tiny_qwen3(seed=0):
    from transformers import Qwen3Config, Qwen3ForCausalLM
    torch.manual_seed(seed)
    cfg = Qwen3Config(vocab_size=48, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=8, max_position_embeddings=256)
    cfg._attn_implementation = "sdpa"
    return Qwen3ForCausalLM(cfg).eval()


def _block_causal(T, bl):
    b = torch.arange(T) // bl
    return (b[None, :] <= b[:, None])[None, None]


@torch.no_grad()
def test_cached_block_causal_equals_one_full_forward():
    m = _tiny_qwen3()
    bl = 4
    prompt = torch.randint(0, 40, (1, 6))
    seq = D.CachedBlockCausal(m, bl, None)
    pl = seq.prefill(prompt)
    assert pl == 4 and seq.cached_length() == 4
    blk = torch.cat([prompt[:, 4:6], torch.full((1, 2), 47)], 1)          # prompt tail + 2 masks
    got = seq.block_forward(blk, 4, store_kv=False, kind="draft_s8")
    assert seq.cached_length() == 4                                        # non-storing forward cropped
    full = torch.cat([prompt[:, :4], blk], 1)
    ref = m(input_ids=full, attention_mask=_block_causal(8, bl), position_ids=torch.arange(8)[None]).logits[:, 4:8]
    assert torch.allclose(got, ref, atol=1e-5)
    final = torch.randint(0, 40, (1, 4))
    seq.flush(final, 4)
    assert seq.cached_length() == 8
    blk2 = torch.full((1, 4), 47)
    got2 = seq.block_forward(blk2, 8, store_kv=False, kind="draft_s8")
    full2 = torch.cat([prompt[:, :4], final, blk2], 1)
    ref2 = m(input_ids=full2, attention_mask=_block_causal(12, bl), position_ids=torch.arange(12)[None]).logits[:, 8:12]
    assert torch.allclose(got2, ref2, atol=1e-5)
    assert seq.n_by_kind["prefill"] == 1 and seq.n_by_kind["draft_s8"] == 2 and seq.n_by_kind["flush"] == 1


class _FakeCache:
    def __init__(self):
        self.n = 0

    def get_seq_length(self):
        return self.n


class _FakeFastDLLM(torch.nn.Module):
    """Logit at absolute position t = one-hot(t mod V): lets the test see which position each slot was read from."""
    V = 50

    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(1))

    def forward(self, input_ids, use_cache, past_key_values, update_past_key_values, block_size):
        lo = past_key_values.n
        T = input_ids.shape[1]
        lg = torch.nn.functional.one_hot((torch.arange(lo, lo + T) % self.V), self.V).float()[None]
        if update_past_key_values:
            past_key_values.n += T

        class O:
            logits = lg
        return O()


@torch.no_grad()
def test_fastdllm_shift_reads_slot_j_from_position_j_minus_1():
    m = _FakeFastDLLM()
    seq = D.FastDLLMv2Seq(m, 4, None)
    seq.cache = _FakeCache()
    seq.prefill(torch.zeros(1, 8, dtype=torch.long))                      # prompt = 2 full blocks
    out = seq.block_forward(torch.zeros(1, 4, dtype=torch.long), 8, store_kv=False, kind="draft_s8")
    assert out[0].argmax(-1).tolist() == [7, 8, 9, 10]                    # slot 0 from the prefill's last position
    seq.flush(torch.zeros(4, dtype=torch.long), 8)
    out = seq.block_forward(torch.zeros(1, 4, dtype=torch.long), 12, store_kv=False, kind="draft_s8")
    assert out[0].argmax(-1).tolist() == [11, 12, 13, 14]                 # slot 0 from the flush's last position


class _FakeCoDA(torch.nn.Module):
    V = 50

    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(1))

    def forward(self, input_ids):
        T = input_ids.shape[1]
        return (torch.nn.functional.one_hot(torch.arange(T) % self.V, self.V).float()[None], None)


@torch.no_grad()
def test_full_recompute_shift_and_free_flush():
    seq = D.FullRecomputeShifted(_FakeCoDA(), 4, None)
    assert seq.prefill(torch.zeros(1, 6, dtype=torch.long)) == 4
    out = seq.block_forward(torch.zeros(1, 4, dtype=torch.long), 4, store_kv=False, kind="draft_s8")
    assert out[0].argmax(-1).tolist() == [3, 4, 5, 6]
    seq.flush(torch.zeros(4, dtype=torch.long), 4)
    assert seq.cached_length() == 8 and "flush" not in seq.n_by_kind       # no forward, not charged


# ---------------------------------------------------------------------------------------------------------------
# expert logging
# ---------------------------------------------------------------------------------------------------------------
class _FakeSeq:
    bl = 4
    dev = torch.device("cpu")
    model = None

    def __init__(self):
        self.timings = [("denoise", 1.0)]


class _Cap:
    def __init__(self, ids, probs):
        self.ids, self.probs = ids, probs


def test_expert_meter_bits_counts_and_summary(tmp_path):
    from phase2.explog import make_meter_class
    M = make_meter_class()
    m = M(_FakeSeq(), None, level="bits", n_layers=2, n_experts=16)
    ids = [torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7]] * 4, dtype=torch.uint8),
           torch.tensor([[8, 9, 10, 11, 12, 13, 14, 15]] * 4, dtype=torch.uint8)]
    probs = [torch.full((4, 8), 0.1, dtype=torch.float16)] * 2
    m._record(_Cap(ids, probs), "denoise", 4)
    assert m.records[-1][2] == 16                                           # uniq, the meter's own count
    bits = np.unpackbits(m.hit_bits[0], axis=-1)
    assert bits[0, :8].all() and not bits[0, 8:].any() and bits[1, 8:].all()
    k = M.KINDS["denoise"]
    assert m.counts[k, 0, :8].tolist() == [4] * 8
    s = m.summary()
    assert s["exp_block_uniq_per_layer"] == 8.0 and abs(s["exp_by_kind"]["denoise"]["top8_mass"] - 0.8) < 1e-3
    m.save(str(tmp_path / "e.npz"))
    z = np.load(tmp_path / "e.npz")
    assert z["hit_bits"].shape == (1, 2, 2) and z["counts"].sum() == 64
    m.reset()
    assert not m.hit_bits and m.counts.sum() == 0


def test_train_expert_capture_counts_rows_times_topk():
    import test_draftalign_train as TT
    from draftalign import adapter as A
    from draftalign import train_fast as F
    from draftalign.dataset import State
    from phase2.explog import TrainExpertCapture
    m = TT.build_model(0)
    A.attach_lora(m, r=4, alpha=8.0, train_gates=False, draft_flag=True)
    A.freeze_base(m)
    cap = TrainExpertCapture(m, torch.device("cpu"), n_experts=TT.N_EXPERTS, top_k=TT.TOP_K)
    cap.install()
    try:
        BL = TT.BLOCK
        clean = torch.randint(0, TT.MASK, (3 * BL,))
        lo = 2 * BL
        valid = torch.ones(BL, dtype=torch.bool)
        inp = clean[:lo + BL].clone()
        inp[lo:] = TT.MASK
        st = State(input_ids=inp, active_lo=lo, valid=valid, targets=clean[lo:lo + BL], open_mask=valid.clone(),
                   meta={"domain": "maths", "has_draft": False, "split": "t", "idx": 0})
        with cap.armed_scope():
            F.micro_batch_loss(m, [st], F.LossCfg(mask_id=TT.MASK), block=BL, checkpoint_layers=False)
        win = cap.window(1)
        n_layers = sum(1 for l in cap.layers if hasattr(l.mlp, "num_experts"))   # the MoE layers (the tiny model has a dense one)
        assert win["block"].sum() == n_layers * BL * TT.TOP_K                 # every block row, top-k each, every layer
        assert win["prefix"].sum() == n_layers * lo * TT.TOP_K
        assert cap.counts.sum() == 0                                         # window() zeroes
        met = TrainExpertCapture.metrics(win)
        assert 0 < met["experts/block_uniq_per_layer"] <= TT.N_EXPERTS
    finally:
        cap.uninstall()
    assert F.moe_forward.__name__ == "moe_forward" and F.prefix_kv.__module__ == "draftalign.train_fast"


# ---------------------------------------------------------------------------------------------------------------
# re-draft on a tiny model through the real draft_block
# ---------------------------------------------------------------------------------------------------------------
@torch.no_grad()
def test_redraft_item_uses_teacher_forced_prefix_and_keeps_invalid_slots():
    from phase2.redraft import redraft_item, open_forwards
    m = _tiny_qwen3()
    bl = 32
    seq = D.CachedBlockCausal(m, bl, None)
    P_len = 40
    prompt = torch.randint(0, 40, (P_len,)).numpy()
    tokens = np.random.RandomState(0).randint(0, 40, size=(2, bl)).astype(np.int32)
    tokens[0, :P_len - 32] = prompt[32:]
    valid = np.ones((2, bl), bool)
    valid[0, :P_len - 32] = False
    it = {"prompt_ids": prompt, "tokens": tokens, "valid": valid, "usable": np.array([True, True]),
          "block_start": np.array([32, 64])}
    r = redraft_item(seq, it, "s8", 47, torch.device("cpu"))
    assert (r["draft"][0, :P_len - 32] == -1).all() and (r["draft"][valid] >= 0).all()
    assert seq.cached_length() == 96                                        # both blocks flushed
    assert r["open_fwd"] == open_forwards(int(valid[0].sum()), "s8") + open_forwards(32, "s8")


# ---------------------------------------------------------------------------------------------------------------
# the job runner
# ---------------------------------------------------------------------------------------------------------------
def test_job_runner_stops_a_group_and_reports_every_job(tmp_path, monkeypatch):
    from phase2 import p2_jobs
    monkeypatch.setattr(p2_jobs, "GLOBAL_STOP", str(tmp_path / "nope"))
    jobs = [{"name": "a.0", "group": "a", "cmd": "exit 3"},
            {"name": "a.1", "group": "a", "cmd": "sleep 30"},
            {"name": "a.2", "group": "a", "cmd": "echo never"},
            {"name": "b.0", "group": "b", "cmd": "echo gpu {GPU} > out.txt"}]
    jf = tmp_path / "jobs.jsonl"
    jf.write_text("\n".join(json.dumps(j) for j in jobs) + "\n")
    log = tmp_path / "run" / "run.log"
    rc = p2_jobs.main(["--jobs", str(jf), "--log", str(log), "--gpus", "0,1", "--poll", "0.2", "--assume-free",
                       "--sandbox", str(tmp_path / "box")])
    summ = json.load(open(str(log) + ".summary.json"))
    assert rc == 3 and summ["stopped_groups"] == ["a"]
    assert summ["jobs"]["a.0"] == 3 and summ["jobs"]["a.2"] == "skipped" and summ["jobs"]["b.0"] == 0
    assert summ["jobs"]["a.1"] != 0                                          # interrupted with its group
    assert (tmp_path / "box" / "b.0" / "out.txt").read_text().startswith("gpu ")   # ran in its own sandbox


def test_cell_reader_skips_running_metrics(tmp_path):
    """`shard*of*.jsonl` also matches `<shard>.jsonl.running.jsonl`; its rows are metrics, not records."""
    import json
    from phase2.decode_hooks import CellReader
    rec = tmp_path / "c.gsm8k500.shard0of1.jsonl"
    rec.write_text(json.dumps({"idx": 3, "n_gen_to_eos": 10, "n_blocks": 1}) + "\n")
    (tmp_path / "c.gsm8k500.shard0of1.jsonl.running.jsonl").write_text(json.dumps({"idx": 0, "items": 1}) + "\n")
    recs = CellReader(str(tmp_path), "gsm8k500").refresh()
    assert list(recs) == [3]
