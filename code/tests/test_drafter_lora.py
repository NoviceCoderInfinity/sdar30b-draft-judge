"""CPU tests for phase 4, the drafter LoRA (code/drafter_lora/).  No real model, no GPU.

Pins: the trainable count (and the runbook's 34,865,152 for SDAR-1.7B); the adapter-off context is the frozen base,
bit for bit; the two-stream forward equals one clean-prefix block-causal forward per block (the inference state);
the loss (KL exactly 0 at init, KL only on rows the original gets right, the [MASK] column, one denominator); merge;
save / load; the held-out split; the training states; the one-pass plan; the replay; and the trainer end to end.

    python -m pytest -q code/tests/test_drafter_lora.py
"""
from __future__ import annotations

import json
import math
import os
import shutil
import sys
import uuid

import numpy as np
import pytest
import torch
import torch.nn.functional as F

CODE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for p in (os.path.join(CODE, "compat"), CODE, os.path.join(CODE, "tests")):
    if p not in sys.path:
        sys.path.insert(0, p)

import test_draftalign_train as TT                                        # noqa: E402  (the tiny SDAR stand-in)
from draftalign.train_forward import block_tril_bool                      # noqa: E402
from drafter_lora import data as D                                        # noqa: E402
from drafter_lora import lora as LA                                       # noqa: E402
from drafter_lora.forward import drafter_loss, two_stream_hidden          # noqa: E402

BL = 32
VOCAB, MASK = TT.VOCAB, TT.MASK                                           # 40, 39


def dense_model(seed=0):
    cfg = TT.tiny_config()
    cfg.mlp_only_layers = list(range(cfg.num_hidden_layers))              # the drafter is dense
    torch.manual_seed(seed)
    return TT.TinyCausalLM(cfg).eval()


def lora_model(seed=0, perturb=True):
    m = dense_model(seed)
    LA.attach(m, r=4, alpha=8.0)
    if perturb:                                                           # a non-identity adapter
        g = torch.Generator().manual_seed(seed + 5)
        with torch.no_grad():
            for _, p in LA.trainable(m):
                p.add_(torch.randn(p.shape, generator=g) * 0.3)
    return m


# ---------------------------------------------------------------------------
# a fake 1B-corpus item (to_trainset format)
# ---------------------------------------------------------------------------
def fake_item(seed=0, pl=45, nb=3, eos_slot=None, pid=None, domain="maths", n_pass=6):
    rng = np.random.default_rng(seed)
    lo0 = (pl // BL) * BL
    prompt = rng.integers(0, MASK - 2, pl)
    tokens = rng.integers(0, MASK - 2, (nb, BL))
    tail = pl - lo0
    tokens[0, :tail] = prompt[lo0:]
    valid = np.ones((nb, BL), bool)
    valid[0, :tail] = False
    if eos_slot is not None:                                             # last block: slots after the EOS invalid
        valid[-1, eos_slot + 1:] = False
    steps = rng.integers(0, n_pass, (nb, BL))
    committed = np.zeros((nb, n_pass, BL), bool)
    bi, sj = np.nonzero(valid)
    committed[bi, steps[bi, sj], sj] = True
    npass = (np.where(valid, steps, -1).max(1) + 1).astype(np.int32)
    draft = np.where(rng.random((nb, BL)) < 0.5, tokens, rng.integers(0, MASK - 2, (nb, BL)))
    draft[0, :tail] = -1
    conf = rng.integers(1, 256, (nb, BL)).astype(np.uint8)
    conf[0, :tail] = 0
    return {"prompt_ids": prompt.astype(np.int32), "tokens": tokens.astype(np.int32), "valid": valid,
            "committed": committed, "n_pass": npass, "block_start": (lo0 + BL * np.arange(nb)).astype(np.int32),
            "draft": draft.astype(np.int32), "has_draft": np.ones(nb, bool), "accepted": np.zeros((nb, BL), bool),
            "usable": np.ones(nb, bool), "draft_conf": conf, "domain": np.array(domain),
            "prompt_length": np.int32(pl), "block_length": np.int32(BL), "correct": np.int8(1),
            "idx": np.int32(seed), "pid": pid or f"p{seed:05d}"}


def write_corpus(root, n=24, domains=("maths", "code", "chat")):
    os.makedirs(os.path.join(root, "items"), exist_ok=True)
    with open(os.path.join(root, "index.jsonl"), "w") as fh:
        for i in range(n):
            it = fake_item(seed=i, pl=33 + (7 * i) % 40, nb=2 + i % 3, eos_slot=(10 if i % 4 == 0 else None),
                           domain=domains[i % len(domains)])
            pid = it.pop("pid")
            rel = os.path.join("items", f"{pid}.npz")
            np.savez_compressed(os.path.join(root, rel), **it)
            fh.write(json.dumps({"file": rel, "pid": pid, "idx": i, "domain": domains[i % len(domains)],
                                 "split": f"c1b_{domains[i % len(domains)]}", "correct": 1,
                                 "usable": [True] * len(it["usable"]),
                                 "has_draft": [True] * len(it["usable"]),
                                 "block_start": it["block_start"].tolist(), "n_blocks": len(it["usable"]),
                                 "seq_len": int(it["block_start"][-1]) + BL,
                                 "corpus_tokens": int(it["prompt_length"]) + BL * len(it["usable"])}) + "\n")
    return os.path.join(root, "index.jsonl")


# ---------------------------------------------------------------------------
# LoRA
# ---------------------------------------------------------------------------
def test_runbook_trainable_count_for_sdar_1_7b():
    # config.json of JetLM/SDAR-1.7B-Chat-b32: hidden 2048, 16 heads x 128, 8 KV heads, intermediate 6144, 28 layers
    h, q, kv, inter, L, r = 2048, 16 * 128, 8 * 128, 6144, 28, 32
    attn = L * r * ((h + q) + (h + kv) + (h + kv) + (q + h))
    mlp = L * r * ((h + inter) + (h + inter) + (inter + h))
    assert (attn, mlp, attn + mlp) == (12_845_056, 22_020_096, LA.EXPECTED_TRAINABLE_D32)


def test_attach_counts_and_freezes():
    m = dense_model()
    LA.attach(m, r=4, alpha=8.0)
    c = TT.tiny_config()
    q, kv = c.num_attention_heads * c.head_dim, c.num_key_value_heads * c.head_dim
    h, inter = c.hidden_size, c.intermediate_size
    per = 4 * ((h + q) + 2 * (h + kv) + (q + h) + 2 * (h + inter) + (inter + h))
    got = LA.count(m)
    assert got["total"] == per * c.num_hidden_layers and got["other"] == 0
    names = [n for n, _ in LA.trainable(m)]
    assert all(n.endswith(("lora_A", "lora_B")) for n in names)
    assert len(names) == 2 * 7 * c.num_hidden_layers
    assert not m.lm_head.weight.requires_grad and not m.model.embed_tokens.weight.requires_grad


def test_zero_init_is_identity_and_adapter_off_is_the_base():
    ids = TT.sample_ids(64)
    base = dense_model()
    ref = base(input_ids=ids).logits
    m = dense_model()
    LA.attach(m, r=4, alpha=8.0)
    assert torch.equal(m(input_ids=ids).logits, ref)                     # B = 0 -> identity
    m = lora_model()
    on = m(input_ids=ids).logits
    with LA.adapter_off((1, 64)):
        off = m(input_ids=ids).logits
    assert torch.equal(off, ref)                                          # the frozen base, bit for bit
    assert not torch.allclose(on, ref)


def test_merge_equals_unmerged_and_save_load(tmp_path):
    ids = TT.sample_ids(64)
    m = lora_model()
    want = m(input_ids=ids).logits
    LA.save(m, str(tmp_path / "ad"))
    m2 = dense_model()
    meta = LA.load(m2, str(tmp_path / "ad"))
    assert meta["kind"] == LA.KIND and meta["r"] == 4
    assert torch.equal(m2(input_ids=ids).logits, want)
    n = LA.merge_into_base(m2)
    assert n == 7 * TT.N_LAYERS and not LA.lora_modules(m2)
    assert torch.allclose(m2(input_ids=ids).logits, want, atol=1e-5)


# ---------------------------------------------------------------------------
# the two-stream forward == the inference state, block by block
# ---------------------------------------------------------------------------
def test_two_stream_equals_clean_prefix_forward_per_block():
    m = lora_model()
    it = fake_item(seed=3, pl=45, nb=3, eos_slot=12)
    ex = D.item_example(it, seed=0, mask_id=MASK)
    b = D.collate([ex], mask_id=MASK)
    with torch.no_grad():
        h2 = two_stream_hidden(m, b.ids, b.pos, b.attn, checkpoint_layers=False)[0]
    clean = D.clean_sequence(it)
    for j, blk in enumerate(ex.blocks):
        lo = int(it["block_start"][blk])
        seq = torch.as_tensor(clean[:lo + BL]).clone()
        seq[lo:lo + BL] = ex.ids[ex.n_clean + BL * j: ex.n_clean + BL * (j + 1)]
        S = lo + BL
        with torch.no_grad():
            ref = m.model(input_ids=seq[None], attention_mask=block_tril_bool(S, "cpu", BL),
                          position_ids=torch.arange(S)[None]).last_hidden_state[0]
        got = h2[ex.n_clean + BL * j: ex.n_clean + BL * (j + 1)]
        assert torch.allclose(got, ref[lo:lo + BL], atol=1e-4), f"block {blk}"
    # the clean stream itself is the block-causal forward of the clean sequence
    with torch.no_grad():
        S = ex.n_clean
        ref = m.model(input_ids=torch.as_tensor(clean[:S])[None], attention_mask=block_tril_bool(S, "cpu", BL),
                      position_ids=torch.arange(S)[None]).last_hidden_state[0]
    assert torch.allclose(h2[:S], ref, atol=1e-4)


def test_padding_does_not_change_real_rows():
    m = lora_model()
    e1 = D.item_example(fake_item(seed=1, pl=40, nb=2), seed=0, mask_id=MASK)
    e2 = D.item_example(fake_item(seed=2, pl=70, nb=4), seed=0, mask_id=MASK)
    with torch.no_grad():
        alone = D.collate([e1], mask_id=MASK)
        h_alone = two_stream_hidden(m, alone.ids, alone.pos, alone.attn, False)[0]
        both = D.collate([e1, e2], mask_id=MASK)
        h_both = two_stream_hidden(m, both.ids, both.pos, both.attn, False)[0]
    L = int(e1.ids.numel())
    assert torch.allclose(h_both[:L], h_alone[:L], atol=1e-4)
    assert torch.isfinite(h_both).all()


# ---------------------------------------------------------------------------
# the loss
# ---------------------------------------------------------------------------
def _batch(seeds=(1, 2)):
    exs = [D.item_example(fake_item(seed=s, pl=40 + 9 * s, nb=3, eos_slot=8), seed=0, mask_id=MASK) for s in seeds]
    return D.collate(exs, mask_id=MASK)


def test_kl_is_exactly_zero_at_init():
    m = dense_model()
    LA.attach(m, r=4, alpha=8.0)
    m.train()
    loss, st = drafter_loss(m, _batch(), kl_weight=0.1, mask_id=MASK, checkpoint_layers=False)
    assert st["kl_sum"] == 0.0
    assert math.isclose(float(loss.detach()), st["ce_sum"] / st["n_rows"], rel_tol=1e-6)
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for n, p in LA.trainable(m) if n.endswith("lora_B"))


def test_loss_matches_a_manual_computation_and_grads_reach_both_streams():
    m = lora_model()
    m.train()
    b = _batch()
    loss, st = drafter_loss(m, b, kl_weight=0.1, mask_id=MASK, checkpoint_layers=True)
    with torch.no_grad():
        h = two_stream_hidden(m, b.ids, b.pos, b.attn, False)
        with LA.adapter_off(tuple(b.ids.shape)):
            h0 = two_stream_hidden(m, b.ids, b.pos, b.attn, False)
        z, z0 = m.lm_head(h[b.rows]).float(), m.lm_head(h0[b.rows]).float()
        z[:, MASK] = -float("inf"); z0[:, MASK] = -float("inf")
        p, p0 = F.softmax(z, -1), F.softmax(z0, -1)
        T = b.targets[b.rows]
        ce = -torch.log(p.gather(1, T[:, None])).squeeze(1)
        sel = p0.argmax(-1) == T
        keep = torch.arange(VOCAB) != MASK
        kl = (p0[:, keep] * (torch.log(p0[:, keep]) - torch.log(p[:, keep]))).sum(-1)
        want = (ce.sum() + 0.1 * kl[sel].sum()) / T.numel()
    assert 0 < st["n_kl"] < st["n_rows"]
    assert st["n_kl"] == float(sel.sum())
    assert torch.allclose(loss.detach(), want, rtol=1e-4, atol=1e-6)
    assert math.isfinite(float(loss.detach())) and st["kl_sum"] > 0
    loss.backward()
    # the clean stream carries gradient too: a change to layer-0 k_proj's LoRA moves the noisy rows only via the
    # prefix keys, and its gradient is non-zero
    g = m.model.layers[0].self_attn.k_proj.lora_B.grad
    assert g is not None and g.abs().sum() > 0


def test_mask_never_predicted():
    m = lora_model()
    b = _batch()
    with torch.no_grad():
        h = two_stream_hidden(m, b.ids, b.pos, b.attn, False)
        z = m.lm_head(h[b.rows]).float()
        z[:, MASK] = 1e4                                                 # even a huge raw [MASK] logit ...
    from drafter_lora.forward import _row_logits
    zz = _row_logits(m, h, b.rows, MASK)
    assert torch.isinf(zz[:, MASK]).all() and (zz[:, MASK] < 0).all()   # ... is -inf in every distribution


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def test_stratified_holdout_quotas_and_determinism():
    entries = [{"pid": f"x{i:04d}", "domain": d} for i, d in enumerate(["a"] * 600 + ["b"] * 300 + ["c"] * 100)]
    hold, per = D.stratified_holdout(entries, 100, 0)
    assert per == {"a": 60, "b": 30, "c": 10} and len(hold) == 100 == len(set(hold))
    assert D.stratified_holdout(entries, 100, 0)[0] == hold
    assert D.stratified_holdout(entries, 100, 1)[0] != hold
    assert sum(D.quotas({"a": 1, "b": 1, "c": 1}, 2).values()) == 2


def test_first_draft_state_reveals_top_confidence_stored_drafts():
    it = fake_item(seed=4, pl=45, nb=2)
    seen_k = set()
    for s in range(200):
        st = D.block_state(it, 1, np.random.default_rng(s), p_first=1.0, mask_id=MASK)
        assert st.kind == D.KIND_FIRST
        valid = it["valid"][1]
        reveal = valid & ~st.hidden
        k = int(reveal.sum())
        assert k == st.k_or_p and k % 4 == 0 and k <= 28
        seen_k.add(k)
        assert np.array_equal(st.ids[reveal], it["draft"][1][reveal])  # stored tokens, right or wrong
        assert (st.ids[st.hidden] == MASK).all()
        if k:
            conf = it["draft_conf"][1].astype(int)
            assert conf[reveal].min() >= conf[st.hidden].max() or \
                sorted(conf[valid])[::-1][k - 1] == conf[reveal].min()
    assert seen_k == set(D.FIRST_KS)


def test_redraft_state_reveals_committed_true_tokens():
    it = fake_item(seed=5, pl=33, nb=2, eos_slot=20)
    for s in range(200):
        st = D.block_state(it, 1, np.random.default_rng(s), p_first=0.0, mask_id=MASK)
        assert st.kind == D.KIND_REDRAFT
        valid = it["valid"][1]
        p = st.k_or_p
        want = it["committed"][1, :p].any(0) & valid
        assert np.array_equal(valid & ~st.hidden, want)
        assert np.array_equal(st.ids[want], it["tokens"][1][want])     # true tokens
        assert st.hidden.any() and not (st.hidden & ~valid).any()
        assert np.array_equal(st.ids[~valid], it["tokens"][1][~valid])  # invalid (post-EOS) = truth
    st = D.block_state(it, 1, np.random.default_rng(0), p_first=0.0, mask_id=MASK, post_eos="mask")
    assert (st.ids[D.post_eos_slots(it, 1)] == MASK).all()


def test_item_example_layout_and_mask():
    it = fake_item(seed=6, pl=50, nb=3)
    ex = D.item_example(it, seed=0, mask_id=MASK)
    lo0 = int(it["block_start"][0])
    assert ex.n_clean == lo0 + 3 * BL and ex.ids.numel() == ex.n_clean + 3 * BL
    assert torch.equal(ex.ids[:ex.n_clean], torch.as_tensor(D.clean_sequence(it)))
    assert (ex.targets[ex.rows] != D.IGNORE).all() and (ex.targets[~ex.rows] == D.IGNORE).all()
    assert not ex.rows[:ex.n_clean].any()
    a = D.attention_mask(ex)
    q_noisy_b1 = ex.n_clean + BL                                          # first row of noisy block 1
    lo1 = int(it["block_start"][1])
    assert a[q_noisy_b1, :lo1].all() and not a[q_noisy_b1, lo1:ex.n_clean].any()
    assert a[q_noisy_b1, ex.n_clean + BL: ex.n_clean + 2 * BL].all()
    assert not a[q_noisy_b1, ex.n_clean: ex.n_clean + BL].any()
    assert not a[:ex.n_clean, ex.n_clean:].any()                          # clean rows never see noisy keys
    # same seed -> same states; another epoch -> fresh states
    assert torch.equal(D.item_example(it, seed=0, mask_id=MASK).ids, ex.ids)
    assert not torch.equal(D.item_example(it, seed=0, epoch=1, mask_id=MASK).ids, ex.ids)


def test_plan_is_exactly_one_pass():
    lengths = {f"p{i}": 96 + 32 * (i % 7) for i in range(1003)}
    order = D.pass_order(list(lengths), seed=0)
    plan = D.plan_pass(order, lengths, items_per_step=64, world=8, micro_tokens=1024)
    seen = [p for s in plan for r in s.per_rank for m in r for p in m]
    assert sorted(seen) == sorted(lengths) and len(plan) == math.ceil(1003 / 64)
    for s in plan:
        for r in s.per_rank:
            for m in r:
                lmax = max(-(-lengths[p] // BL) * BL for p in m)
                assert len(m) == 1 or len(m) * lmax <= 1024
        assert s.n_micro_global == sum(len(r) for r in s.per_rank)


# ---------------------------------------------------------------------------
# replay (s8 first draft + re-draft probe), with the decode tests' cache-aware fake LM
# ---------------------------------------------------------------------------
def test_replay_perfect_and_repeating_drafters():
    import test_da_decode as TD
    from draftalign.seqcache import SeqCache  # noqa: F401
    from drafter_lora.replay import parity_with_stored, replay_item

    it = fake_item(seed=7, pl=45, nb=2)
    for k in ("tokens", "prompt_ids"):
        it[k] = np.clip(it[k], 0, 50)
    full = D.clean_sequence(it)

    def perfect(pos, tok, is_mask, ids, cached, pkv):
        return TD.onehot_row(int(full[pos]))

    r = replay_item(TD.ScriptedLM(perfect), it, "cpu", mask_id=TD.MASK, redraft_probe=True)
    assert np.array_equal(r["right"], it["valid"]) and r["redraft"]["holes"] == 0
    par = parity_with_stored(it, r["drafts"])
    assert par["slots"] == int((it["draft"] >= 0).sum())

    def stubborn(pos, tok, is_mask, ids, cached, pkv):                   # always 1 first, the truth second
        if not is_mask:
            return TD.onehot_row(tok)
        return TD.logprob_row({1: .6, int(full[pos]): .3, 0: .1})

    r = replay_item(TD.ScriptedLM(stubborn), it, "cpu", mask_id=TD.MASK, redraft_probe=True)
    rd = r["redraft"]
    assert rd["holes"] > 0 and rd["repeat"] == rd["holes"] and rd["repeat_ban"] == 0
    assert rd["agree_ban"] == rd["holes"]                                 # banned -> the runner-up, the truth


# ---------------------------------------------------------------------------
# the trainer, end to end on CPU (tiny model, fake corpus)
# ---------------------------------------------------------------------------
def test_trainer_end_to_end(tmp_path, monkeypatch):
    import research_protocol as P
    from drafter_lora import train_drafter_lora as TR

    idx = write_corpus(str(tmp_path / "corpus"), n=30)
    out_rel = os.path.join("logs", "_pytest_phase4", uuid.uuid4().hex[:8])
    out = os.path.join(os.path.dirname(CODE), out_rel)
    monkeypatch.setenv("ENABLE_WANDB", "0")
    monkeypatch.setattr(P, "load_sdar", lambda key, device, dtype=None: (None, dense_model(1)))
    try:
        argv = ["--index", idx, "--out", out_rel, "--holdout", "6", "--items-per-step", "4", "--micro-tokens", "512",
                "--lora-r", "4", "--alpha", "8", "--expect-trainable", "0", "--check-at", "0", "--no-merge",
                "--mask-id", str(MASK), "--lr", "3e-3", "--warmup-frac", "0.1", "--no-autocast", "--load-workers", "2"]
        assert TR.main(argv) == 0
        cfg = json.load(open(os.path.join(out, "config.json")))
        assert cfg["n_train_items"] == 24 and cfg["total_steps"] == 6 and cfg["save_steps"][-1] == 6
        hold = json.load(open(os.path.join(out, "holdout_pids.json")))
        assert len(hold["pids"]) == 6 and sum(hold["per_domain"].values()) == 6
        rows = [json.loads(l) for l in open(os.path.join(out, "train.rank0.jsonl")) if l.strip()]
        steps = [r for r in rows if "ce" in r]
        assert len(steps) == 6 and steps[0]["kl"] == 0.0 and all(math.isfinite(r["loss"]) for r in steps)
        assert sum(r["items"] for r in steps) == 24
        assert os.path.isfile(os.path.join(out, "final_adapter", LA.ADAPTER_SAFETENSORS))
        assert open(os.path.join(out, "ckpt", "latest")).read().strip() == "step_000006"
        # resume from the last checkpoint: nothing left to do, same result
        assert TR.main(argv + ["--resume"]) == 0
        # the speed stop: an impossible budget stops at the projection step with code 3 and STOP.json
        out2 = out_rel + "_stop"
        argv2 = [a if a != out_rel else out2 for a in argv] + ["--project-at", "2", "--max-hours", "1e-9"]
        assert TR.main(argv2) == TR.STOP_CODE
        assert json.load(open(os.path.join(os.path.dirname(CODE), out2, "STOP.json")))["reason"].startswith("projected")
    finally:
        shutil.rmtree(os.path.join(os.path.dirname(CODE), "logs", "_pytest_phase4"), ignore_errors=True)


# ---------------------------------------------------------------------------
# gate scripts (pure logic; the model-loading halves run on the box)
# ---------------------------------------------------------------------------
def test_eval_agree_aggregate_and_gate1_verdict(tmp_path):
    from drafter_lora import eval_agree as EA

    def shard(k, agree0, agree1, kept, right0):
        c = {k_: 0.0 for k_ in EA.COUNT_KEYS}
        c.update({"items": 10, "slots": 1000, "agree0": agree0, "agree1": agree1, "kept": kept, "right0": right0,
                  "par_slots": 1000, "par_equal": 980})
        return {"shard": k, "nshards": 2, "seconds": 1.0, "counts": {"maths": dict(c), "all": dict(c)},
                "parity_only": False, "info": {}}
    d = tmp_path / "shards"; d.mkdir()
    for k in range(2):
        json.dump(shard(k, 550, 620, 545, 550), open(d / f"shard_{k}.json", "w"))
    args = EA.build_parser().parse_args(["--aggregate", str(d), "--out", "x"])
    rep = EA.aggregate(args)
    assert math.isclose(rep["overall"]["delta_points"], 7.0) and math.isclose(rep["overall"]["retention"], 545 / 550)
    assert rep["verdict"]["status"] == "PASS"
    for k in range(2):
        json.dump(shard(k, 550, 570, 520, 550), open(d / f"shard_{k}.json", "w"))
    assert EA.aggregate(args)["verdict"]["status"] == "FAIL"                  # +2.0 points, retention 94.5%


def test_judge_probe_swaps_drafts_on_identical_examples(tmp_path):
    from draftalign.dataset import SamplerConfig, StateSampler
    from drafter_lora import judge_probe as JP

    idx = write_corpus(str(tmp_path / "c"), n=6)
    entries = D.load_entries([idx])
    pids = [str(e["pid"]) for e in entries]
    swap = {}
    for e in entries:
        it = D.load_item(e)
        dr = np.where(it["draft"] >= 0, 7, -1).astype(np.int32)
        swap[str(e["pid"])] = (dr, np.full(dr.shape, 200, np.uint8))
    ents = JP.heldout_entries([idx], pids)
    assert [e["idx"] for e in ents] == list(range(6)) and all(os.path.isabs(e["file"]) for e in ents)
    cfg = SamplerConfig(index_path="x", seed=1, max_seq=1856)
    a, b = StateSampler(cfg, ents), JP.SwappedDrafts(cfg, ents, swap)
    assert a.examples == b.examples
    sa, sb = a.state(0, regime="draft_pass1"), b.state(0, regime="draft_pass1")
    lo = sa.active_lo
    flag = sb.open_mask
    assert (sb.input_ids[lo:lo + BL][flag] == 7).all()
    assert torch.equal(sa.input_ids[:lo], sb.input_ids[:lo]) and torch.equal(sa.targets, sb.targets)


def test_compare_gates_gate3_verdict(tmp_path):
    from drafter_lora import compare_gates as CG

    def rec(i, correct, passes, gen=None):
        g = gen if gen is not None else list(range(100))
        return {"idx": i, "correct": correct, "n_blocks": 4, "n_forward": 4 + passes * 4 + 4, "n_prefill": 4,
                "n_flush": 4, "n_gen_to_eos": len(g), "gen_ids": g, "seconds": 2.0, "truncated": False,
                "drafter_n_by_kind": {"draft": 32}, "redraft_slots_per_block": [4, 4, 0, 0],
                "redraft_eq_old_per_block": [1, 0, 0, 0]}
    for root, passes in (("new", 7), ("p3", 8)):
        for s in ("ifeval541", "math500"):
            os.makedirs(tmp_path / root / s)
            with open(tmp_path / root / s / "x.shard0of8.jsonl", "w") as fh:
                for i in range(40):
                    fh.write(json.dumps(rec(i, int(i % 3 != 0), passes)) + "\n")
    out = os.path.join("logs", "_pytest_phase4_cg", "g3.json")
    try:
        rc = CG.main(["--gate", "3", "--new", str(tmp_path / "new"), "--phase3", str(tmp_path / "p3"), "--out", out])
        rep = json.load(open(os.path.join(os.path.dirname(CODE), out)))
    finally:
        shutil.rmtree(os.path.join(os.path.dirname(CODE), "logs", "_pytest_phase4_cg"), ignore_errors=True)
    v = rep["verdict"]
    assert math.isclose(v["passes_reduction_pct"], 12.5) and v["pooled_delta_acc"][0] == 0.0
    assert v["status"] == "PASS" and rc == 0
    assert rep["vs_phase3"]["pooled"]["new"]["redraft_repeat"] == 1 / 8
    assert CG.loops({"gen_ids": [1, 2, 3] * 40, "n_gen_to_eos": 120}) and not CG.loops(rec(0, 1, 1))


def test_backup_refuses_secrets(tmp_path, monkeypatch):
    from drafter_lora import backup_p4 as BK
    assert BK._match("logs/phase4/decode/rd1/gsm8k500/a.jsonl", BK.ALLOW)
    assert BK._match(".env", BK.DENY) and not BK._match(".env", BK.ALLOW)
    f = os.path.join(BK.ROOT, "logs", "_pytest_phase4_bk.txt")
    os.makedirs(os.path.dirname(f), exist_ok=True)
    try:
        open(f, "w").write("x hf_" + "A" * 34 + " y")
        assert BK.scan([os.path.relpath(f, BK.ROOT)], [], 1) == [os.path.relpath(f, BK.ROOT)]
        open(f, "w").write("value: s3cr3t-value-123 ")
        assert BK.scan([os.path.relpath(f, BK.ROOT)], [b"s3cr3t-value-123"], 1)
        open(f, "w").write("clean")
        assert not BK.scan([os.path.relpath(f, BK.ROOT)], [b"s3cr3t-value-123"], 1)
    finally:
        os.remove(f)


def test_hf_progress_pins_match_research_protocol():
    import research_protocol as P
    from drafter_lora import hf_progress as H
    for k, (repo, rev, _label) in H.MODELS.items():
        assert (P.MODELS[k]["repo"], P.MODELS[k]["revision"]) == (repo, rev)
