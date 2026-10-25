"""The two-stream trainer (code/draftalign/train_fast.py) computes the SAME loss and the SAME adapter
gradients as the round-3 per-state path (code/draftalign/train.py::forward_state), on CPU in fp32, with the
tiny SDAR stand-in of code/tests/test_draftalign_train.py.

Also pins: the chat rule changes only chat states; the confidence embedding starts as the identity.

    python -m pytest -q code/corpus1b/tests/test_train_fast.py
"""
from __future__ import annotations

import os
import sys

import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.abspath(os.path.join(HERE, "..", ".."))
for p in (os.path.join(CODE, "compat"), CODE, os.path.join(CODE, "tests")):
    if p not in sys.path:
        sys.path.insert(0, p)

import test_draftalign_train as TT                                        # noqa: E402  (the tiny model)
from draftalign import adapter as A                                       # noqa: E402
from draftalign import train as T                                         # noqa: E402
from draftalign import train_fast as F                                    # noqa: E402
from draftalign.dataset import State                                      # noqa: E402

BL, MASK, VOCAB = TT.BLOCK, TT.MASK, TT.VOCAB


def make_model(seed=0, conf=False):
    m = TT.build_model(seed)
    A.attach_lora(m, r=4, alpha=8.0, train_gates=False, draft_flag=True, conf_embed=conf)
    A.freeze_base(m)
    g = torch.Generator().manual_seed(seed + 11)
    with torch.no_grad():                                   # a NON-identity adapter, or the test proves nothing
        for _, p in A.trainable_parameters(m):
            p.add_(torch.randn(p.shape, generator=g) * 0.2)
    m.train()
    return m


def make_item(seed, prompt_len, n_blocks):
    g = torch.Generator().manual_seed(seed)
    prompt = torch.randint(0, MASK, (prompt_len,), generator=g)
    lo0 = (prompt_len // BL) * BL
    gen = torch.randint(0, MASK, (lo0 + BL * n_blocks - prompt_len,), generator=g)
    clean = torch.cat([prompt, gen])
    return clean, lo0


def make_state(clean, lo0, prompt_len, b, seed, split="c1b_maths", idx=0, domain="maths", with_conf=False):
    g = torch.Generator().manual_seed(1000 + seed)
    lo = lo0 + BL * b
    targets = clean[lo:lo + BL].clone()
    valid = torch.ones(BL, dtype=torch.bool)
    if b == 0:
        valid[:prompt_len - lo0] = False
    committed = (torch.rand(BL, generator=g) < 0.3) & valid
    open_mask = valid & ~committed
    if not bool(open_mask.any()):
        open_mask = valid.clone()
    draft = torch.randint(0, MASK, (BL,), generator=g)
    draft = torch.where(torch.rand(BL, generator=g) < 0.5, targets, draft)          # half the drafts right
    keep = (torch.rand(BL, generator=g) < 0.6) & open_mask
    blk = targets.clone()
    blk[open_mask] = MASK
    blk[keep] = draft[keep]
    ids = clean[:lo + BL].clone()
    ids[lo:lo + BL] = blk
    conf = torch.randint(0, 256, (BL,), generator=g).to(torch.uint8) if with_conf else None
    meta = {"split": split, "idx": idx, "domain": domain, "block": b}
    return State(input_ids=ids, active_lo=lo, valid=valid, targets=targets, open_mask=open_mask, meta=meta,
                 draft_conf=conf)


def batch_of_states():
    states = []
    for idx, (pl, nb, dom) in enumerate([(6, 3, "maths"), (9, 2, "code"), (4, 3, "science"), (13, 2, "if")]):
        clean, lo0 = make_item(idx, pl, nb)
        for b in range(nb):
            # the round-3 path gives code its weight by split name ("opencode*"), the new one by domain
            split = "opencode_pool" if dom == "code" else f"c1b_{dom}"
            states.append(make_state(clean, lo0, pl, b, seed=10 * idx + b, split=split, idx=idx, domain=dom))
    return states


def grads(model):
    return {n: p.grad.detach().clone() for n, p in A.trainable_parameters(model) if p.grad is not None}


def reference(model, states, beta=0.5, calib=1.0, ovr=2.0, code_ovr=3.0):
    """Round-3 path: one full forward per state (train.forward_state), losses averaged."""
    model.zero_grad(set_to_none=True)
    tot = 0.0
    for st in states:
        loss, _, _, _ = T.forward_state(model, st, None, "full", False, beta, 0.5, MASK, torch.device("cpu"),
                                        block=BL, draft_flag=True, calib=calib, override_weight=ovr,
                                        code_override_weight=code_ovr)
        (loss / len(states)).backward()
        tot += float(loss.detach()) / len(states)
    return tot, grads(model)


def fast(model, states, cfg, checkpoint_layers=False, conf=False):
    model.zero_grad(set_to_none=True)
    total, stats, _ = F.micro_batch_loss(model, states, cfg, block=BL, checkpoint_layers=checkpoint_layers,
                                         conf_embed=conf)
    (total / len(states)).backward()
    return float(total) / len(states), grads(model)


def assert_close_grads(g1, g2, tol=2e-5):
    assert g1.keys() == g2.keys()
    for n in g1:
        d = (g1[n] - g2[n]).abs().max().item()
        scale = max(1.0, g1[n].abs().max().item())
        assert d <= tol * scale, f"{n}: max |dgrad| {d}"


@pytest.mark.parametrize("checkpoint_layers", [False, True])
def test_fast_equals_round3_loss_and_grads(checkpoint_layers):
    model = make_model()
    states = batch_of_states()
    ref_loss, ref_g = reference(model, states)
    cfg = F.LossCfg(beta=0.5, chat_beta=0.5, temp=0.5, calib=1.0, override_weight=2.0, code_override_weight=3.0,
                    chat_override_weight=2.0, mask_id=MASK)
    f_loss, f_g = fast(model, states, cfg, checkpoint_layers)
    assert abs(ref_loss - f_loss) <= 1e-5 * max(1.0, abs(ref_loss)), (ref_loss, f_loss)
    assert_close_grads(ref_g, f_g)


def test_states_of_one_item_share_one_prefix():
    states = batch_of_states()
    mb = F.group_states(states, BL)
    assert len(mb.prefixes) == 4 and len(mb.item_of_state) == len(states)
    for st, j in zip(states, mb.item_of_state):                      # every state sees a prefix of its own item
        lo = int(st.active_lo)
        assert torch.equal(mb.prefixes[j][:lo], st.input_ids[:lo])


def test_chat_rule_changes_only_chat_states():
    """beta 0 and weight 1 on chat == the round-3 loss with those values for the chat state, unchanged elsewhere."""
    model = make_model(1)
    states = batch_of_states()
    clean, lo0 = make_item(99, 7, 2)
    chat = [make_state(clean, lo0, 7, b, seed=500 + b, split="c1b_chat", idx=99, domain="chat") for b in range(2)]
    cfg = F.LossCfg(beta=0.5, chat_beta=0.0, calib=1.0, override_weight=2.0, code_override_weight=3.0,
                    chat_override_weight=1.0, mask_id=MASK)
    model.zero_grad(set_to_none=True)
    _, stats, _ = F.micro_batch_loss(model, states + chat, cfg, block=BL, checkpoint_layers=False)
    for st, s in zip(states + chat, stats):
        if st.meta["domain"] == "chat":
            ref, _, _, _ = T.forward_state(model, st, None, "full", False, 0.0, 0.5, MASK, torch.device("cpu"),
                                           block=BL, draft_flag=True, calib=1.0, override_weight=1.0)
        else:
            ref, _, _, _ = T.forward_state(model, st, None, "full", False, 0.5, 0.5, MASK, torch.device("cpu"),
                                           block=BL, draft_flag=True, calib=1.0, override_weight=2.0,
                                           code_override_weight=3.0)
        assert abs(float(ref) - s["loss"]) <= 1e-5 * max(1.0, abs(float(ref))), (st.meta, float(ref), s["loss"])


def test_conf_embed_starts_as_identity():
    """A zero-initialised confidence embedding changes nothing until it is trained."""
    m0 = TT.build_model(3)
    A.attach_lora(m0, r=4, alpha=8.0, train_gates=False, draft_flag=True)
    m1 = TT.build_model(3)
    A.attach_lora(m1, r=4, alpha=8.0, train_gates=False, draft_flag=True, conf_embed=True)
    states = []
    clean, lo0 = make_item(5, 6, 3)
    for b in range(3):
        states.append(make_state(clean, lo0, 6, b, seed=b, with_conf=True))
    cfg = F.LossCfg(mask_id=MASK)
    with torch.no_grad():
        l0, _, _ = F.micro_batch_loss(m0, states, cfg, block=BL, checkpoint_layers=False)
        l1, _, _ = F.micro_batch_loss(m1, states, cfg, block=BL, checkpoint_layers=False, conf_embed=True)
    assert float(l0) == float(l1)
    assert A.count_trainable(m1)["conf_embed"] == A.CONF_BUCKETS * TT.HID
