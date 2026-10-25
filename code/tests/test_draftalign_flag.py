"""CPU tests for the DRAFT FLAG: a flagged draft-slot embedding + a flag-gated LoRA delta
(code/draftalign/{adapter,train_forward,train,seqcache,da_decode,run_da_decode}.py).

Round 1's diagnosis (docs/EXPLAINER_DRAFT_ALIGNED_TRAINING.md section 13) was that the model
cannot tell a tentative draft token from a committed one, and that the LoRA delta -- applied at
every position -- makes learning to judge bleed into masked and committed slots.  The change:

  * `adapter.DraftEmbed`: one trainable, ZERO-initialised fp32 vector added (in fp32) to the input
    embedding of every TENTATIVE DRAFT slot;
  * `adapter.CTX` / `flag_scope`: a per-position 0/1 gate on every LoRA delta.  `CTX.flag is None`
    is exactly the pre-flag behaviour; an all-False flag is exactly the frozen base model.

What is pinned here, in the order of the change's spec:
  (i)   flag None -> the pre-flag arithmetic, bit-for-bit;
  (ii)  an all-False flag -> the base output, bit-for-bit, at random non-zero A/B;
  (iii) a partial flag -> flagged rows get the FULL delta, unflagged rows get none;
  (iv)  a zero draft_embed changes nothing; a non-zero one changes ONLY flagged positions;
  (v)   save/load carries draft_embed + the `draft_embed: true` metadata; `merge_lora` REFUSES on a
        flag-gated adapter and the decoder loads it un-merged instead;
  (vi)  `training_forward(draft_flag=...)`: an all-mask state is the stock forward exactly, an
        all-True flag is the ungated forward exactly, no delta leaks to an unflagged row, and the
        gradients survive gradient checkpointing (the layer is re-run during BACKWARD, when the
        context must still be armed);
  (vii) the decoder: `--judge full` flags exactly `open & ~[MASK]` per pass and nothing at the
        prefill / flush / stock passes, and an identity adapter (B = 0, draft_embed = 0) leaves the
        decode bit-identical while a non-zero draft_embed changes it;
  (viii) `train.save_checkpoint(keep=N)` keeps only the N newest checkpoints and stays resumable.

Run from the repo root:  /venv/main/bin/python -m pytest -q code/tests/test_draftalign_flag.py
"""
from __future__ import annotations

import json
import os
import sys
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

CODE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if CODE not in sys.path:
    sys.path.insert(0, CODE)
if os.path.join(CODE, "compat") not in sys.path:
    sys.path.insert(0, os.path.join(CODE, "compat"))

from draftalign import adapter as A                                            # noqa: E402
from draftalign import da_decode as DA                                         # noqa: E402
from draftalign import train as T                                             # noqa: E402
from draftalign.da_decode import MoEMeter, da_generate                         # noqa: E402
from draftalign.seqcache import SeqCache                                       # noqa: E402
from draftalign.train_forward import stock_hidden_reference, training_forward   # noqa: E402

from tests.test_draftalign_train import (BLOCK, MASK, VOCAB, build_model, sample_ids)   # noqa: E402
from tests.test_da_decode import (EOS_IDS, GEN, MASK as DMASK, PLEN, V, BL,     # noqa: E402
                                  ScriptedLM, meter_for, prompt_ids, scripted_drafter_fn)
from tests.test_da_decode_full import TinyAttn, TinyModel, full_judge_fn        # noqa: E402


@pytest.fixture(autouse=True)
def _clear_flag():
    """`adapter.CTX` is a module global: never let one test's flag leak into the next."""
    A.clear_flag()
    yield
    A.clear_flag()


# ==========================================================================
# (i) - (iii) the gate on LoRALinear
# ==========================================================================
def _lora(in_f=6, out_f=5, r=2, alpha=4.0, seed=0):
    torch.manual_seed(seed)
    base = nn.Linear(in_f, out_f, bias=False)
    mod = A.LoRALinear(base, r=r, alpha=alpha)
    with torch.no_grad():                       # a REAL adapter: both factors non-zero
        mod.lora_A.copy_(torch.randn_like(mod.lora_A))
        mod.lora_B.copy_(torch.randn_like(mod.lora_B))
    return mod


def _ungated_reference(mod, x):
    """`LoRALinear.forward` as it was before the flag existed (adapter.py round 1)."""
    a, b = mod.lora_A.to(x.dtype), mod.lora_B.to(x.dtype)
    return mod.base(x) + torch.nn.functional.linear(torch.nn.functional.linear(x, a), b) * mod.scaling


def test_flag_none_is_the_pre_flag_arithmetic_bit_for_bit():
    """(i) With no flag armed the delta is ungated -- the round-1 expression, exactly."""
    mod = _lora()
    x = torch.randn(2, 7, mod.in_features)
    assert A.CTX.flag is None and A.CTX.any is False
    got = mod(x)
    assert torch.equal(got, _ungated_reference(mod, x))
    assert not torch.equal(got, mod.base(x)), "the reference adapter must not be the identity"


def test_all_false_flag_is_the_frozen_base_bit_for_bit():
    """(ii) Every position gated off -> the base module's own output, bit-for-bit."""
    mod = _lora(seed=1)
    x = torch.randn(3, 5, mod.in_features)
    off = torch.zeros(3, 5, dtype=torch.bool)
    with A.flag_scope(off):
        got = mod(x)
    assert torch.equal(got, mod.base(x))
    assert not torch.equal(got, _ungated_reference(mod, x))


def test_all_true_flag_is_the_ungated_delta_bit_for_bit():
    """A gate of exactly 1.0 is an exact float multiplication, so nothing shifts."""
    mod = _lora(seed=2)
    x = torch.randn(2, 4, mod.in_features)
    on = torch.ones(2, 4, dtype=torch.bool)
    with A.flag_scope(on):
        got = mod(x)
    assert torch.equal(got, _ungated_reference(mod, x))


def test_partial_flag_gives_the_full_delta_to_flagged_rows_and_none_to_the_rest():
    """(iii) Row-wise: flagged == ungated, unflagged == base."""
    mod = _lora(seed=3)
    x = torch.randn(2, 6, mod.in_features)
    flag = torch.zeros(2, 6, dtype=torch.bool)
    flag[0, [1, 4]] = True
    flag[1, 5] = True
    with A.flag_scope(flag):
        got = mod(x)
    ungated, base = _ungated_reference(mod, x), mod.base(x)
    assert torch.equal(got[flag], ungated[flag])
    assert torch.equal(got[~flag], base[~flag])
    assert not torch.equal(ungated[~flag], base[~flag]), "the unflagged rows must have been non-trivial"


@pytest.mark.parametrize("last_dim", [5, 11])
def test_the_same_flag_serves_q_k_v_and_o(last_dim):
    """o_proj is fed [B, S, num_heads * head_dim], q/k/v [B, S, hidden]: one [B, S] flag for all four."""
    mod = _lora(in_f=last_dim, out_f=9, seed=4)
    x = torch.randn(2, 3, last_dim)
    flag = torch.tensor([[True, False, True], [False, False, True]])
    with A.flag_scope(flag):
        got = mod(x)
    assert torch.equal(got[flag], _ungated_reference(mod, x)[flag])
    assert torch.equal(got[~flag], mod.base(x)[~flag])


def test_flag_scope_restores_and_rejects_bad_shapes():
    mod = _lora(seed=5)
    outer = torch.ones(1, 2, dtype=torch.bool)
    with A.flag_scope(outer):
        with A.flag_scope(torch.zeros(1, 2, dtype=torch.bool)):
            assert A.CTX.any is False
        assert A.CTX.flag is outer and A.CTX.any is True
        with pytest.raises(Exception):
            with A.flag_scope(None):
                raise RuntimeError("boom")
        assert A.CTX.flag is outer                       # restored even through an exception
    assert A.CTX.flag is None
    with pytest.raises(AssertionError):
        A.set_flag(torch.ones(4, dtype=torch.bool))      # [S] is not [B, S]
    with pytest.raises(AssertionError):
        A.set_flag(torch.ones(1, 4))                     # not bool
    with A.flag_scope(torch.ones(1, 3, dtype=torch.bool)):
        with pytest.raises(AssertionError):
            mod(torch.randn(1, 4, mod.in_features))      # flag does not cover the tensor


# ==========================================================================
# (iv) the draft embedding
# ==========================================================================
def test_draft_embed_is_zero_init_and_changes_only_flagged_positions():
    de = A.DraftEmbed(8)
    assert de.weight.dtype == torch.float32 and torch.count_nonzero(de.weight) == 0
    embeds = torch.randn(2, 5, 8)
    flag = torch.zeros(2, 5, dtype=torch.bool)
    flag[0, [0, 3]] = True
    flag[1, 2] = True
    assert torch.equal(de.apply_to(embeds, flag), embeds), "a zero vector must be the identity"
    with torch.no_grad():
        de.weight.copy_(torch.randn(8))
    got = de.apply_to(embeds, flag)
    assert torch.equal(got[~flag], embeds[~flag])
    assert torch.equal(got[flag], (embeds[flag].float() + de.weight.float()).to(embeds.dtype))


def test_zero_draft_embed_leaves_the_model_forward_identical():
    """The round trip float()-add-zero-cast-back is lossless, so attaching the flag changes nothing."""
    model = build_model()
    ids = sample_ids(2 * BLOCK)
    with torch.no_grad():
        before = model(input_ids=ids).logits.clone()
    counts = A.attach_lora(model, r=4, alpha=8.0, draft_flag=True)
    A.freeze_base(model)
    assert counts["draft_embed"] == 1 and A.is_flag_gated(model)
    assert A.get_draft_embed(model) is not None
    S = 2 * BLOCK
    valid = torch.ones(BLOCK, dtype=torch.bool)
    stock = stock_hidden_reference(model, ids, block=BLOCK)[0, S - BLOCK:]
    with torch.no_grad():
        for flag in (None, torch.zeros(S, dtype=torch.bool), torch.ones(S, dtype=torch.bool)):
            got = training_forward(model, ids, S - BLOCK, valid, checkpoint_layers=False,
                                   block=BLOCK, draft_flag=flag).hidden
            assert torch.equal(got, stock), f"flag={None if flag is None else int(flag.sum())}"
    with torch.no_grad():
        model(input_ids=ids)                              # the plain forward is untouched too
    assert torch.equal(model(input_ids=ids).logits, before)


def test_draft_embed_and_the_trainable_set():
    model = build_model()
    A.attach_lora(model, r=4, alpha=8.0, draft_flag=True)
    A.freeze_base(model)
    names = [n for n, _ in A.trainable_parameters(model)]
    assert A.DRAFT_EMBED_KEY in names
    counts = A.count_trainable(model)
    assert counts["draft_embed"] == model.config.hidden_size
    assert counts["total"] == counts["lora"] + counts["gate"] + counts["draft_embed"]
    groups = A.parameter_groups(model, lr_lora=2e-5, lr_gate=1e-5)
    lora_params = {id(p) for p in groups[0]["params"]}
    assert id(A.get_draft_embed(model).weight) in lora_params, "the draft vector trains at lr_lora"


def test_attach_without_the_flag_changes_nothing_at_all():
    """Default off: no module, no metadata key, and therefore the round-1 `config_hash`."""
    a, b = build_model(), build_model()
    A.attach_lora(a, r=4, alpha=8.0)
    A.attach_lora(b, r=4, alpha=8.0, draft_flag=True)
    assert A.get_draft_embed(a) is None and not A.is_flag_gated(a)
    ma, mb = A.adapter_metadata(a), A.adapter_metadata(b)
    assert "draft_embed" not in ma and mb["draft_embed"] is True
    assert ma["config_hash"] != mb["config_hash"]
    assert {k: v for k, v in mb.items() if k not in ("draft_embed", "config_hash")} == \
           {k: v for k, v in ma.items() if k != "config_hash"}


# ==========================================================================
# (v) save / load / merge
# ==========================================================================
def test_save_load_round_trip_carries_the_draft_embed_and_the_metadata(tmp_path):
    model = build_model()
    A.attach_lora(model, r=4, alpha=8.0, draft_flag=True)
    A.freeze_base(model)
    torch.manual_seed(7)
    with torch.no_grad():
        for _, p in A.trainable_parameters(model):
            p.add_(torch.randn_like(p) * 0.05)
    d = str(tmp_path / "ad")
    A.save_adapter(model, d)
    meta = json.load(open(os.path.join(d, A.ADAPTER_JSON)))
    assert meta["draft_embed"] is True

    from safetensors import safe_open
    with safe_open(os.path.join(d, A.ADAPTER_SAFETENSORS), framework="pt") as fh:
        assert A.DRAFT_EMBED_KEY in fh.keys()
        assert json.loads(fh.metadata()["da_adapter"])["draft_embed"] is True

    fresh = build_model()
    info = A.load_adapter(fresh, d)                       # attaches the flag from the metadata alone
    assert info["missing"] == 0 and info["unexpected"] == 0 and A.is_flag_gated(fresh)
    assert torch.equal(A.get_draft_embed(fresh).weight, A.get_draft_embed(model).weight)
    S = 2 * BLOCK
    ids = sample_ids(S)
    flag = torch.zeros(S, dtype=torch.bool)
    flag[S - BLOCK + 1] = True
    valid = torch.ones(BLOCK, dtype=torch.bool)
    with torch.no_grad():
        want = training_forward(model, ids, S - BLOCK, valid, checkpoint_layers=False, block=BLOCK,
                                draft_flag=flag).hidden
        got = training_forward(fresh, ids, S - BLOCK, valid, checkpoint_layers=False, block=BLOCK,
                               draft_flag=flag).hidden
    assert torch.equal(want, got)


def test_load_refuses_a_flag_mismatch(tmp_path):
    flagged = build_model()
    A.attach_lora(flagged, r=4, alpha=8.0, draft_flag=True)
    A.freeze_base(flagged)
    d = str(tmp_path / "ad")
    A.save_adapter(flagged, d)
    plain = build_model()
    A.attach_lora(plain, r=4, alpha=8.0)                  # already attached, WITHOUT the flag
    A.freeze_base(plain)
    with pytest.raises(AssertionError):
        A.load_adapter(plain, d)


def test_merge_lora_refuses_a_flag_gated_adapter():
    """Merging would make the delta ungated -- exactly what the flag exists to prevent."""
    model = build_model()
    A.attach_lora(model, r=4, alpha=8.0, draft_flag=True)
    A.freeze_base(model)
    with pytest.raises(RuntimeError, match="flag-gated"):
        A.merge_lora(model)
    assert any(isinstance(m, A.LoRALinear) for m in model.modules()), "the tree must be left intact"
    plain = build_model()
    A.attach_lora(plain, r=4, alpha=8.0)
    assert A.merge_lora(plain)["lora"] > 0                # the unflagged path is unchanged


def test_runner_loads_a_flag_gated_adapter_un_merged(tmp_path):
    """`run_da_decode.load_adapter_into` must not merge, and must refuse a model it cannot flag."""
    import draftalign.run_da_decode as R
    trained = FlaggedLM(scripted_drafter_fn)
    A.attach_lora(trained, r=2, alpha=4.0, train_gates=False, draft_flag=True)
    A.freeze_base(trained)
    torch.manual_seed(5)
    with torch.no_grad():
        for _, p in A.trainable_parameters(trained):
            p.add_(torch.randn_like(p) * 0.01)
    d = str(tmp_path / "flagged")
    A.save_adapter(trained, d)

    infer = FlaggedLM(scripted_drafter_fn)
    ad = R.load_adapter_into(infer, d)
    assert ad["draft_flag"] is True and ad["merged"] == {"lora": 0, "gates": 0}
    assert any(isinstance(m, A.LoRALinear) for m in infer.modules()), "it must stay un-merged"
    assert torch.equal(A.get_draft_embed(infer).weight, A.get_draft_embed(trained).weight)

    # a model the flag cannot reach (no inputs_embeds, no embed_tokens) is refused, not silently run
    assert not A.can_apply_draft_flag(TinyModel())        # the plain fake really is unreachable
    blind = FlaggedLM(scripted_drafter_fn)
    saved = A.can_apply_draft_flag
    A.can_apply_draft_flag = lambda _m: False
    try:
        with pytest.raises(AssertionError, match="draft-flag"):
            R.load_adapter_into(blind, d)
    finally:
        A.can_apply_draft_flag = saved


# ==========================================================================
# (vi) training_forward
# ==========================================================================
def _flag_model(seed=0, scale=0.05):
    """A tiny SDAR stand-in with a NON-identity flag-gated adapter, plus its stock hidden states."""
    model = build_model(seed)
    S = 2 * BLOCK
    ids = sample_ids(S)
    stock = stock_hidden_reference(model, ids, block=BLOCK)[0, S - BLOCK:].clone()
    A.attach_lora(model, r=4, alpha=8.0, draft_flag=True)
    A.freeze_base(model)
    torch.manual_seed(100 + seed)
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, A.LoRALinear):
                m.lora_B.add_(torch.randn_like(m.lora_B) * scale)
        A.get_draft_embed(model).weight.copy_(torch.randn(model.config.hidden_size) * scale)
    return model, ids, S, stock


def test_training_forward_all_mask_state_is_the_stock_model_exactly():
    """(vi) A state with no draft-filled slot flags nothing -> no delta, no embedding: stock."""
    model, ids, S, stock = _flag_model()
    valid = torch.ones(BLOCK, dtype=torch.bool)
    with torch.no_grad():
        got = training_forward(model, ids, S - BLOCK, valid, checkpoint_layers=False, block=BLOCK,
                               draft_flag=torch.zeros(S, dtype=torch.bool)).hidden
        ungated = training_forward(model, ids, S - BLOCK, valid, checkpoint_layers=False,
                                   block=BLOCK, draft_flag=None).hidden
    assert torch.equal(got, stock)
    assert not torch.equal(ungated, stock), "the adapter must be non-trivial for this to mean anything"


def test_training_forward_full_flag_matches_the_ungated_forward_when_the_vector_is_zero():
    model, ids, S, _ = _flag_model()
    with torch.no_grad():
        A.get_draft_embed(model).weight.zero_()
    valid = torch.ones(BLOCK, dtype=torch.bool)
    with torch.no_grad():
        gated = training_forward(model, ids, S - BLOCK, valid, checkpoint_layers=False, block=BLOCK,
                                 draft_flag=torch.ones(S, dtype=torch.bool)).hidden
        ungated = training_forward(model, ids, S - BLOCK, valid, checkpoint_layers=False,
                                   block=BLOCK, draft_flag=None).hidden
    assert torch.equal(gated, ungated)


def test_training_forward_leaks_no_delta_to_an_unflagged_position():
    """Inside the real forward: every gated projection's delta is exactly zero off the flag."""
    model, ids, S, _ = _flag_model(seed=1)
    flag = torch.zeros(S, dtype=torch.bool)
    flag[[S - BLOCK + 1, S - BLOCK + 2, 3]] = True
    seen, hooks = [], []

    def hook(mod, inp, out):
        seen.append((mod, inp[0].detach(), out.detach()))
    for m in model.modules():
        if isinstance(m, A.LoRALinear):
            hooks.append(m.register_forward_hook(hook))
    with torch.no_grad():
        training_forward(model, ids, S - BLOCK, torch.ones(BLOCK, dtype=torch.bool),
                         checkpoint_layers=False, block=BLOCK, draft_flag=flag)
    for h in hooks:
        h.remove()
    assert seen, "no LoRA projection ran"
    n_moved = 0
    for mod, x, out in seen:
        delta = out - mod.base(x)
        assert torch.count_nonzero(delta[0][~flag]) == 0, "a delta reached an unflagged position"
        n_moved += int(torch.count_nonzero(delta[0][flag]))
    assert n_moved > 0, "the flagged positions must actually have moved"


def test_training_forward_flag_survives_gradient_checkpointing():
    """The layer is re-run during BACKWARD, long after `training_forward` returned: the gate must be
    armed inside the checkpointed closure or the gradients would be an ungated delta's."""
    flag_pos = [2 * BLOCK - BLOCK + 1, 3]

    def grads(checkpointing):
        model, ids, S, _ = _flag_model(seed=2)
        flag = torch.zeros(S, dtype=torch.bool)
        flag[flag_pos] = True
        out = training_forward(model, ids, S - BLOCK, torch.ones(BLOCK, dtype=torch.bool),
                               checkpoint_layers=checkpointing, block=BLOCK, draft_flag=flag)
        out.hidden.square().sum().backward()
        return {n: p.grad.clone() for n, p in A.trainable_parameters(model) if p.grad is not None}, out

    g_off, out_off = grads(False)
    g_on, out_on = grads(True)
    assert torch.equal(out_off.hidden.detach(), out_on.hidden.detach())
    assert set(g_off) == set(g_on) and g_off
    for k in g_off:
        assert torch.allclose(g_off[k], g_on[k], atol=1e-6, rtol=1e-6), k
    assert any(float(v.abs().max()) > 0 for v in g_off.values())
    assert float(g_off[A.DRAFT_EMBED_KEY].abs().max()) > 0, "the draft vector must receive a gradient"
    assert A.CTX.flag is None, "the context must be clear once the forward returned"


def test_state_draft_flag_marks_exactly_the_draft_filled_open_slots():
    """`train.state_draft_flag`: open & valid & not [MASK]; the prefix and committed slots never."""
    S = 3 * BLOCK
    lo = S - BLOCK
    ids = torch.arange(S) % 20
    valid = torch.tensor([True] * BLOCK)
    valid[0] = False                                       # a prompt-tail slot
    open_mask = valid.clone()
    open_mask[1] = False                                   # a committed slot
    ids[lo + 2] = MASK                                     # a masked open slot
    st = SimpleNamespace(input_ids=ids, active_lo=lo, valid=valid, open_mask=open_mask,
                         targets=torch.zeros(BLOCK, dtype=torch.long), meta={})
    flag = T.state_draft_flag(st, MASK, BLOCK)
    assert flag.shape == (S,) and not bool(flag[:lo].any())
    want = torch.zeros(BLOCK, dtype=torch.bool)
    want[3:] = True                                        # 0 not valid, 1 committed, 2 masked
    assert torch.equal(flag[lo:], want)
    # the all-mask regime flags nothing at all
    ids2 = ids.clone(); ids2[lo:][open_mask] = MASK
    st2 = SimpleNamespace(input_ids=ids2, active_lo=lo, valid=valid, open_mask=open_mask,
                          targets=torch.zeros(BLOCK, dtype=torch.long), meta={})
    assert int(T.state_draft_flag(st2, MASK, BLOCK).sum()) == 0


def test_check_flag_identity_passes_on_a_flagged_model(tmp_path):
    """`train.check_flag_identity` is the step-0 assertion the trainer runs."""
    model = build_model()
    A.attach_lora(model, r=4, alpha=8.0, draft_flag=True)
    A.freeze_base(model)
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, A.LoRALinear):
                m.lora_B.add_(torch.randn_like(m.lora_B) * 0.1)

    class _Sampler:                                       # only `state(k, regime=...)` and len() are used
        def __len__(self):
            return 2

        def state(self, k, regime="sample"):
            assert regime == "allmask_pass1"
            S = 2 * BLOCK
            ids = torch.arange(S) % 20
            ids[S - BLOCK:] = MASK
            return SimpleNamespace(input_ids=ids, active_lo=S - BLOCK,
                                   valid=torch.ones(BLOCK, dtype=torch.bool),
                                   open_mask=torch.ones(BLOCK, dtype=torch.bool),
                                   targets=torch.zeros(BLOCK, dtype=torch.long), meta={})

    rep = T.check_flag_identity(model, _Sampler(), MASK, BLOCK)
    assert rep["flagged_slots"] == 0 and rep["lora_identity_at_zero_flag"] is True
    assert rep["draft_embed_zero"] is True


# ==========================================================================
# (vii) the decoder
# ==========================================================================
class FlaggedLM(nn.Module):
    """A cache-aware fake with a REAL module tree, so `attach_lora`, `DraftEmbed` and the flag gate
    all sit on the forward path that `da_generate` drives.

    `model.embed_tokens` is the IDENTITY embedding (weight = I), so the token ids are recovered from
    `inputs_embeds` by an argmax -- exact as long as the draft vector stays below 0.5 per coordinate,
    which every test here respects.  The scripted logits (the decode's actual decisions) are
    perturbed by a small multiple of the tree's output, so an adapter that is the identity leaves the
    decode bit-identical and ANY leaked delta or embedding shows up in `judge_p`.
    """

    def __init__(self, fn, vocab=V, scale=0.01, seed=17):
        super().__init__()
        torch.manual_seed(seed)
        self.fn, self.vocab, self.scale, self.calls = fn, vocab, float(scale), []
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(vocab, vocab)
        with torch.no_grad():
            self.model.embed_tokens.weight.copy_(torch.eye(vocab))
        layer = nn.Module()
        layer.self_attn = TinyAttn(vocab)
        self.model.layers = nn.ModuleList([layer])
        self.head = nn.Linear(vocab, vocab, bias=False)
        self.config = SimpleNamespace(hidden_size=vocab, num_hidden_layers=1, vocab_size=vocab)

    @property
    def device(self):
        return self.head.weight.device

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def _tree(self, h):
        for layer in self.model.layers:
            h = h + layer.self_attn(h)
        return self.head(h)

    def forward(self, input_ids=None, inputs_embeds=None, attention_mask=None, position_ids=None,
                past_key_values=None, use_cache=None, store_kv=None, **kw):
        assert (input_ids is None) ^ (inputs_embeds is None), "exactly one of ids / embeds"
        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)
            ids_t = input_ids
        else:
            ids_t = inputs_embeds.argmax(-1)              # identity embedding -> exact recovery
        cached = past_key_values.get_seq_length() if past_key_values is not None else 0
        self.calls.append(dict(ids=ids_t.clone(), pos=position_ids.clone(), store_kv=bool(store_kv),
                               cached=cached, flag=None if A.CTX.flag is None else A.CTX.flag[0].clone(),
                               embeds=inputs_embeds.detach().clone()))
        B, S = ids_t.shape
        ids = ids_t[0].tolist()
        logits = torch.empty(1, S, self.vocab)
        for s in range(S):
            logits[0, s] = self.fn(int(position_ids[0, s]), ids[s], ids[s] == DMASK, ids, cached, past_key_values)
        logits = logits + self.scale * self._tree(inputs_embeds)
        if past_key_values is not None and store_kv:
            k = ids_t.float().view(B, 1, S, 1).expand(B, 2, S, 3).contiguous()
            for lyr in range(2):
                past_key_values.update(k, k, lyr)
        return SimpleNamespace(logits=logits)


class HookFlaggedLM(FlaggedLM):
    """The same model whose forward does NOT take `inputs_embeds`, so the draft embedding has to be
    applied by the `embed_tokens` forward hook instead (the fallback path)."""

    def forward(self, input_ids=None, attention_mask=None, position_ids=None, past_key_values=None,
                use_cache=None, store_kv=None, **kw):
        return FlaggedLM.forward(self, input_ids=input_ids, attention_mask=attention_mask,
                                 position_ids=position_ids, past_key_values=past_key_values,
                                 use_cache=use_cache, store_kv=store_kv, **kw)


def _full_decode(model, draft_flag: bool):
    moe = MoEMeter(SeqCache(model, BL, "moe"), draft_flag=draft_flag)
    dr = SeqCache(ScriptedLM(scripted_drafter_fn), BL, "drafter")
    res = da_generate(moe, dr, prompt_ids(PLEN), mask_id=DMASK, gen_length=GEN, block_length=BL,
                      threshold=.95, draft="s8", judge="full", tau_commit=.95,
                      stopping_criteria_idx=EOS_IDS)
    return moe, res


def test_full_judge_flags_exactly_the_tentative_draft_slots():
    """(vii) Per judge pass the flag is `open & ~[MASK]`; the prefill, the flush and the stock
    passes flag nothing."""
    model = FlaggedLM(full_judge_fn)
    moe, res = _full_decode(model, draft_flag=True)
    inv = {v: k for k, v in DA.MOE_KIND_CODE.items()}
    kinds = [inv[r[0]] for r in moe.records]
    assert len(moe.flags) == len(moe.records) == len(model.calls)
    b = res.blocks[0]
    judge_flags = []
    for kind, flag, call in zip(kinds, moe.flags, model.calls):
        assert flag is not None and flag.dtype == torch.bool
        assert torch.equal(flag, call["flag"].cpu()), "the model saw a different flag than was recorded"
        if kind == "judge":
            ids = call["ids"][0]
            assert bool(flag.any())
            assert not bool((flag & (ids == DMASK)).any()), "a [MASK] slot was flagged"
            assert not bool((flag & ~b.valid).any()), "the prompt tail was flagged"
            judge_flags.append(flag)
        else:
            assert not bool(flag.any()), f"{kind} must flag nothing"
    assert len(judge_flags) == b.n_judge >= 2
    # pass 1 holds the draft at EVERY valid slot -> its flag IS the valid mask; later passes only
    # keep the still-tentative subset (committed and re-masked slots drop out)
    assert torch.equal(judge_flags[0], b.valid)
    for prev, nxt in zip(judge_flags, judge_flags[1:]):
        assert bool((nxt & ~prev).sum()) == 0, "a slot became tentative again"
    assert A.CTX.flag is None, "the context must be clear after the decode"


@pytest.mark.parametrize("cls", [FlaggedLM, HookFlaggedLM])
def test_identity_adapter_leaves_the_full_judge_decode_bit_identical(cls, tmp_path):
    """B = 0 and draft_embed = 0 -> the same decode as with no adapter at all, through either the
    `inputs_embeds` path or the `embed_tokens` hook; a non-zero vector then changes it."""
    plain = cls(full_judge_fn)
    _, ref = _full_decode(plain, draft_flag=False)

    flagged = cls(full_judge_fn)
    assert A.supports_inputs_embeds(flagged) is (cls is FlaggedLM)
    A.attach_lora(flagged, r=2, alpha=4.0, train_gates=False, draft_flag=True)
    A.freeze_base(flagged)
    moe, got = _full_decode(flagged, draft_flag=True)
    assert torch.equal(ref.x, got.x)
    assert len(ref.blocks) == len(got.blocks)
    for a, b in zip(ref.blocks, got.blocks):
        assert torch.equal(a.tokens, b.tokens) and a.n_judge == b.n_judge
        for name in ("judge_p", "judge_q", "judge_argmax"):     # exact, NaN for NaN
            torch.testing.assert_close(getattr(a, name), getattr(b, name),
                                       rtol=0, atol=0, equal_nan=True)
    assert len(moe.records) == len(flagged.calls)

    # a non-zero draft vector reaches the flagged slots and moves the judge's probabilities
    with torch.no_grad():
        A.get_draft_embed(flagged).weight.copy_(torch.full((flagged.config.hidden_size,), 0.05))
    _, moved = _full_decode(flagged, draft_flag=True)
    p0, p1 = ref.blocks[0].judge_p, moved.blocks[0].judge_p
    seen = ~torch.isnan(p0)
    assert bool(seen.any()) and not torch.equal(p0[seen], p1[seen])


def test_the_draft_embed_lands_only_on_flagged_positions_in_the_decoder():
    """The embeddings the model actually receives differ from `embed_tokens(ids)` exactly on the flag."""
    model = FlaggedLM(full_judge_fn)
    A.attach_lora(model, r=2, alpha=4.0, train_gates=False, draft_flag=True)
    A.freeze_base(model)
    vec = torch.full((model.config.hidden_size,), 0.05)
    with torch.no_grad():
        A.get_draft_embed(model).weight.copy_(vec)
    moe, _ = _full_decode(model, draft_flag=True)
    checked = 0
    for flag, call in zip(moe.flags, model.calls):
        base = model.model.embed_tokens(call["ids"])
        delta = call["embeds"] - base
        assert torch.equal(delta[0][~flag], torch.zeros_like(delta[0][~flag]))
        if bool(flag.any()):
            assert torch.allclose(delta[0][flag], vec.expand_as(delta[0][flag]), atol=1e-6)
            checked += 1
    assert checked >= 1


def test_meter_without_the_flag_is_the_pre_flag_path():
    """`draft_flag=False` (every existing cell): no flag is armed and no embeds kwarg is built."""
    model = FlaggedLM(full_judge_fn)
    moe, _ = _full_decode(model, draft_flag=False)
    assert moe.flags == [None] * len(moe.records)
    assert all(c["flag"] is None for c in model.calls)


def test_stock_decode_through_the_flagged_meter_is_unchanged():
    """A flag-armed meter on a model with NO adapter: every forward is all-False -> stock decode."""
    from tests.test_da_decode import seeded_fn
    ref = meter_for(seeded_fn(0))
    a = da_generate(ref, None, prompt_ids(37), mask_id=DMASK, gen_length=64, block_length=BL,
                    denoising_steps=BL, threshold=.95, draft=None, stopping_criteria_idx=EOS_IDS)
    m2 = MoEMeter(SeqCache(ScriptedLM(seeded_fn(0)), BL, "moe"), draft_flag=True)
    b = da_generate(m2, None, prompt_ids(37), mask_id=DMASK, gen_length=64, block_length=BL,
                    denoising_steps=BL, threshold=.95, draft=None, stopping_criteria_idx=EOS_IDS)
    assert torch.equal(a.x, b.x) and a.n_forward == b.n_forward
    assert all(f is not None and not bool(f.any()) for f in m2.flags)


# ==========================================================================
# (viii) checkpoint retention
# ==========================================================================
def test_save_checkpoint_keeps_only_the_newest_two(tmp_path):
    model = build_model()
    A.attach_lora(model, r=2, alpha=4.0)
    A.freeze_base(model)
    opt = torch.optim.AdamW(A.parameter_groups(model, 1e-2, 1e-2))
    out = str(tmp_path / "run")
    os.makedirs(out, exist_ok=True)
    for step in (10, 20, 30, 40):
        with torch.no_grad():
            for _, p in A.trainable_parameters(model):
                p.add_(0.01)
        T.save_checkpoint(model, opt, out, step, keep=2)
    assert T.checkpoint_dirs(out) == ["step000030", "step000040"]
    assert open(os.path.join(out, "latest")).read().strip() == "step000040"

    fresh = build_model()
    A.attach_lora(fresh, r=2, alpha=4.0)
    A.freeze_base(fresh)
    opt2 = torch.optim.AdamW(A.parameter_groups(fresh, 1e-2, 1e-2))
    assert T.load_checkpoint(fresh, opt2, out, torch.device("cpu")) == 40
    for (n, p), (n2, p2) in zip(A.trainable_parameters(model), A.trainable_parameters(fresh)):
        assert n == n2 and torch.allclose(p, p2)


def test_keep_checkpoints_zero_keeps_everything_and_latest_is_never_deleted(tmp_path):
    model = build_model()
    A.attach_lora(model, r=2, alpha=4.0)
    A.freeze_base(model)
    opt = torch.optim.AdamW(A.parameter_groups(model, 1e-2, 1e-2))
    out = str(tmp_path / "run")
    for step in (1, 2, 3):
        T.save_checkpoint(model, opt, out, step, keep=0)
    assert T.checkpoint_dirs(out) == ["step000001", "step000002", "step000003"]
    # keep=1 with `latest` pointing at an OLDER checkpoint: that one survives the prune
    open(os.path.join(out, "latest"), "w").write("step000001")
    removed = T.prune_checkpoints(out, 1)
    assert removed == ["step000002"] and T.checkpoint_dirs(out) == ["step000001", "step000003"]


def test_trainer_cli_defaults():
    a = T.build_parser().parse_args(["--index", "i", "--out", "o"])
    assert a.draft_flag is False and a.keep_checkpoints == 2
    b = T.build_parser().parse_args(["--index", "i", "--out", "o", "--draft-flag",
                                     "--keep-checkpoints", "5"])
    assert b.draft_flag is True and b.keep_checkpoints == 5
