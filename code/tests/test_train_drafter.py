"""CPU tests for the drafter trainer (code/draftalign/{build_drafter_set,train_drafter}.py).

No real model, no GPU.  Two stand-ins:

  `TinyDense`   mirrors the DENSE remote module tree
                (hf_cache/.../SDAR-1.7B-Chat-b32/.../modeling_sdar.py, lines cited inline):
                    model.model.{embed_tokens, layers[i], norm, rotary_emb} + model.lm_head
                with `model.model.forward(input_ids, attention_mask, position_ids, use_cache,
                store_kv)` -> BaseModelOutputWithPast, and it is a real `PreTrainedModel`, so
                `save_pretrained` / safetensors / resume are exercised for real.
                Its attention is plain SDPA with the bool mask it is handed -- it records that
                mask, so a test can assert the trainer passes exactly `block_tril_bool`.

  `OracleLM`   a stub whose hidden state at position i is a scaled one-hot of the CLEAN token at
               position i and whose `lm_head` is the identity.  Its logits are therefore right
               at every position under an UNSHIFTED readout and wrong under a shifted one, which
               is what pins the loss convention.

Run:  /venv/main/bin/python -m pytest -q code/tests/test_train_drafter.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

CODE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if CODE not in sys.path:
    sys.path.insert(0, CODE)
if os.path.join(CODE, "compat") not in sys.path:
    sys.path.insert(0, os.path.join(CODE, "compat"))

from transformers import PretrainedConfig, PreTrainedModel                # noqa: E402
from transformers.modeling_outputs import BaseModelOutputWithPast         # noqa: E402

from draftalign import build_drafter_set as BD                            # noqa: E402
from draftalign import train_drafter as TD                                # noqa: E402
from draftalign.train_forward import block_tril_bool                      # noqa: E402

BLOCK = 4
VOCAB = 40
MASK = 39
HID = 16
LAYERS = 2


# ===========================================================================
# stand-in models
# ===========================================================================
class TinyConfig(PretrainedConfig):
    model_type = "tinysdar"

    def __init__(self, vocab_size=VOCAB, hidden_size=HID, num_hidden_layers=LAYERS, **kw):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        super().__init__(**kw)


class TinyLayer(nn.Module):
    """One decoder layer: SDPA with the mask it is given (remote :275-297) + an MLP."""

    def __init__(self, hidden):
        super().__init__()
        self.q = nn.Linear(hidden, hidden, bias=False)
        self.k = nn.Linear(hidden, hidden, bias=False)
        self.v = nn.Linear(hidden, hidden, bias=False)
        self.o = nn.Linear(hidden, hidden, bias=False)
        self.mlp = nn.Linear(hidden, hidden, bias=False)
        self.seen_mask = None

    def forward(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None,
                output_attentions=False, use_cache=False, store_kv=None, cache_position=None,
                position_embeddings=None, **kw):
        self.seen_mask = None if attention_mask is None else attention_mask.detach().clone()
        m = attention_mask.bool() if attention_mask is not None else None
        q = self.q(hidden_states).unsqueeze(1)
        k = self.k(hidden_states).unsqueeze(1)
        v = self.v(hidden_states).unsqueeze(1)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=m, is_causal=False)
        h = hidden_states + self.o(a.squeeze(1))
        return (h + torch.tanh(self.mlp(h)),)


class TinyInner(nn.Module):
    """`SDARModel` (remote :463-560): embed -> layers (raw mask passed through) -> norm."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([TinyLayer(config.hidden_size)
                                     for _ in range(config.num_hidden_layers)])
        self.norm = nn.LayerNorm(config.hidden_size)
        self.rotary_emb = nn.Identity()

    def forward(self, input_ids=None, attention_mask=None, position_ids=None, past_key_values=None,
                inputs_embeds=None, use_cache=None, store_kv=None, **kw):
        h = self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        for layer in self.layers[: self.config.num_hidden_layers]:
            h = layer(h, attention_mask=attention_mask, position_ids=position_ids,
                      use_cache=use_cache, store_kv=store_kv)[0]
        return BaseModelOutputWithPast(last_hidden_state=self.norm(h))


class TinyDense(PreTrainedModel):
    config_class = TinyConfig
    _supports_sdpa = True

    def __init__(self, config):
        super().__init__(config)
        self.model = TinyInner(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)


class OracleLM:
    """hidden[i] = scale * one_hot(clean[i]); lm_head = identity -> an UNSHIFTED perfect model."""

    def __init__(self, clean: torch.Tensor, scale: float = 20.0):
        self.clean = clean
        self.scale = scale
        self.lm_head = nn.Identity()
        self.training = False
        self.seen = {}

    def _inner(self, input_ids=None, attention_mask=None, position_ids=None, use_cache=None,
               store_kv=None, **kw):
        self.seen = {"attention_mask": attention_mask, "position_ids": position_ids,
                     "use_cache": use_cache, "store_kv": store_kv,
                     "input_ids": input_ids.detach().clone()}
        h = F.one_hot(self.clean, VOCAB).float() * self.scale
        return BaseModelOutputWithPast(last_hidden_state=h.unsqueeze(0))

    @property
    def model(self):
        return self._inner


def build_model(seed: int = 0) -> TinyDense:
    torch.manual_seed(seed)
    return TinyDense(TinyConfig())


# ===========================================================================
# fixtures / helpers
# ===========================================================================
def synth(prompt_length=6, n_gen=10, seed=0, vocab=VOCAB - 1):
    g = np.random.default_rng(seed)
    prompt = g.integers(0, vocab, prompt_length).astype(np.int32)
    gen = g.integers(0, vocab, n_gen).astype(np.int32)
    grid = BD.grid(prompt_length, n_gen, max_seq=None, block=BLOCK)
    return prompt, gen, grid


def example(prompt_length=6, n_gen=10, seed=0, force_ratio=None, gseed=1):
    prompt, gen, grid = synth(prompt_length, n_gen, seed)
    g = torch.Generator().manual_seed(gseed)
    ex = TD.mask_blocks(prompt, gen, grid["n_blocks"], grid["lo0"], g,
                        mask_id=MASK, block=BLOCK, force_ratio=force_ratio)
    return ex, prompt, gen, grid


def write_set(tmp_path, n_items=6, prompt_length=6, n_gen=10, correct=None, split="train_pool"):
    """A minimal build_drafter_set output tree."""
    root = tmp_path / "ds"
    (root / "items").mkdir(parents=True, exist_ok=True)
    entries = []
    for i in range(n_items):
        prompt, gen, grid = synth(prompt_length, n_gen, seed=100 + i)
        name = f"{split}__{i:05d}.npz"
        np.savez(root / "items" / name, prompt_ids=prompt, gen_ids=gen,
                 n_blocks=np.int32(grid["n_blocks"]), block_start=np.int32(grid["lo0"]),
                 prompt_length=np.int32(prompt_length), block_length=np.int32(BLOCK),
                 n_pad=np.int32(grid["n_pad"]), seq_len=np.int32(grid["seq_len"]),
                 correct=np.int8(1), idx=np.int32(i))
        entries.append({"file": os.path.join("items", name), "idx": i, "split": split,
                        "correct": 1 if correct is None else int(correct[i]),
                        "prompt_length": prompt_length, "n_gen": grid["n_gen_kept"],
                        "n_blocks": grid["n_blocks"], "block_start": grid["lo0"],
                        "n_pad": grid["n_pad"], "seq_len": grid["seq_len"]})
    with open(root / "index.jsonl", "w") as fh:
        for e in entries:
            fh.write(json.dumps(e) + "\n")
    return str(root / "index.jsonl")


def cfg(index, **kw):
    d = dict(index_path=index, seed=0, block=BLOCK, mask_id=MASK, max_seq=None, min_blocks=1,
             only_correct=False)
    d.update(kw)
    return TD.DrafterSetConfig(**d)


# ===========================================================================
# 1. the grid
# ===========================================================================
def test_grid_puts_the_prompt_tail_in_the_first_response_block():
    g = BD.grid(prompt_length=6, n_gen=10, max_seq=None, block=BLOCK)
    assert g["lo0"] == 4                      # (6 // 4) * 4
    assert g["n_blocks"] == 3                 # slots 4..15 hold 2 tail + 10 generated
    assert g["seq_len"] == 16 and g["n_gen_kept"] == 10 and g["n_pad"] == 0


def test_grid_pads_the_last_block_and_never_the_earlier_ones():
    g = BD.grid(prompt_length=8, n_gen=9, max_seq=None, block=BLOCK)
    assert (g["lo0"], g["n_blocks"], g["seq_len"]) == (8, 3, 20)
    assert g["n_gen_kept"] == 9 and g["n_pad"] == 3


def test_grid_max_seq_truncates_on_a_block_boundary():
    g = BD.grid(prompt_length=6, n_gen=40, max_seq=16, block=BLOCK)
    assert g["n_blocks"] == 3 and g["seq_len"] == 16
    assert g["n_gen_kept"] == 10 and g["n_pad"] == 0        # a truncated item never pads
    assert BD.grid(prompt_length=20, n_gen=40, max_seq=16, block=BLOCK)["n_blocks"] == 0


# ===========================================================================
# 2. masking invariants
# ===========================================================================
def test_prompt_is_clean_and_never_a_loss_row():
    for gs in range(20):
        ex, prompt, gen, grid = example(gseed=gs)
        P_ = prompt.shape[0]
        assert torch.equal(ex.input_ids[:P_], torch.from_numpy(prompt.astype(np.int64)))
        assert not bool(ex.loss_mask[:P_].any())
        assert bool((ex.targets[:P_] == TD.IGNORE).all())
        assert not bool(ex.resp[:P_].any())


def test_every_block_has_at_least_one_masked_token():
    for gs in range(50):
        ex, prompt, gen, grid = example(gseed=gs)
        for b in range(grid["n_blocks"]):
            lo = grid["lo0"] + b * BLOCK
            sl = slice(lo, lo + BLOCK)
            if bool(ex.resp[sl].any()):
                assert int(ex.loss_mask[sl].sum()) >= 1, f"block {b} has no masked slot"


def test_masked_slots_hold_MASK_and_unmasked_slots_hold_the_teacher_token():
    ex, prompt, gen, grid = example(gseed=7)
    P_ = prompt.shape[0]
    clean = torch.cat([torch.from_numpy(prompt.astype(np.int64)),
                       torch.from_numpy(gen.astype(np.int64))])
    assert bool((ex.input_ids[ex.loss_mask] == MASK).all())
    keep = ex.resp & ~ex.loss_mask
    assert torch.equal(ex.input_ids[keep], clean[:len(clean)][keep[:len(clean)]])
    assert torch.equal(ex.targets[ex.loss_mask], clean[:len(clean)][ex.loss_mask[:len(clean)]])


def test_pad_slots_are_masked_input_but_never_loss_rows():
    ex, prompt, gen, grid = example(prompt_length=8, n_gen=9, gseed=3)
    assert grid["n_pad"] == 3
    pad = torch.zeros_like(ex.resp)
    pad[prompt.shape[0] + gen.shape[0]:] = True
    assert bool((ex.input_ids[pad] == MASK).all())
    assert not bool((ex.loss_mask & pad).any())
    assert bool((ex.targets[pad] == TD.IGNORE).all())


def test_force_ratio_one_masks_every_response_slot():
    ex, prompt, gen, grid = example(force_ratio=1.0, gseed=5)
    assert torch.equal(ex.loss_mask, ex.resp)
    assert all(r == 1.0 for r in ex.ratios)


def test_ratio_distribution_is_uniform_and_drives_the_masked_share():
    ratios, shares = [], []
    for gs in range(400):
        ex, prompt, gen, grid = example(prompt_length=4, n_gen=32, seed=1, gseed=gs)
        for b in range(grid["n_blocks"]):
            lo = grid["lo0"] + b * BLOCK
            sl = slice(lo, lo + BLOCK)
            ratios.append(ex.ratios[b])
            shares.append(float(ex.loss_mask[sl].sum()) / float(ex.resp[sl].sum()))
    r = np.array(ratios)
    assert 0.0 <= r.min() and r.max() <= 1.0
    assert abs(r.mean() - 0.5) < 0.03, r.mean()
    # a uniform draw puts ~25% of the blocks in each quarter
    q = np.histogram(r, bins=4, range=(0, 1))[0] / r.size
    assert q.min() > 0.20 and q.max() < 0.30, q
    # the realised masked share tracks the drawn ratio (biased upwards only by the forced token)
    s = np.array(shares)
    assert s.mean() > r.mean() and s.mean() - r.mean() < 0.10, (s.mean(), r.mean())
    lo_p = r < 0.2
    assert (s[lo_p] >= 1.0 / BLOCK - 1e-9).all(), "a block lost its forced masked token"


def test_ratio_bounds_are_honoured():
    ex, *_ = example(gseed=2)
    g = torch.Generator().manual_seed(11)
    prompt, gen, grid = synth(4, 32)
    ex2 = TD.mask_blocks(prompt, gen, grid["n_blocks"], grid["lo0"], g, mask_id=MASK, block=BLOCK,
                         ratio_lo=0.4, ratio_hi=0.6)
    assert all(0.4 <= r <= 0.6 for r in ex2.ratios)


def test_inv_p_weights_are_one_over_the_drawn_ratio():
    ex, prompt, gen, grid = example(prompt_length=4, n_gen=32, gseed=9)
    for b, p in enumerate(ex.ratios):
        lo = grid["lo0"] + b * BLOCK
        sl = slice(lo, lo + BLOCK)
        w = ex.weights[sl][ex.loss_mask[sl]]
        assert torch.allclose(w, torch.full_like(w, 1.0 / max(p, 1.0 / BLOCK)))
    assert bool((ex.weights[~ex.loss_mask] == 0).all())


def test_example_is_deterministic_and_split_by_tag(tmp_path):
    ds = TD.DrafterSet(cfg(write_set(tmp_path)))
    a = ds.example(0, epoch=0, tag="train")
    b = ds.example(0, epoch=0, tag="train")
    c = ds.example(0, epoch=1, tag="train")
    d = ds.example(0, epoch=0, tag="eval_sampled")
    assert torch.equal(a.input_ids, b.input_ids) and torch.equal(a.loss_mask, b.loss_mask)
    assert not torch.equal(a.loss_mask, c.loss_mask) or a.meta["n_loss"] != c.meta["n_loss"]
    assert not torch.equal(a.loss_mask, d.loss_mask) or a.meta["n_loss"] != d.meta["n_loss"]


# ===========================================================================
# 3. the loss: masked positions only, UNSHIFTED
# ===========================================================================
def test_loss_is_unshifted_and_only_at_masked_positions():
    prompt, gen, grid = synth(prompt_length=4, n_gen=12, seed=2)
    g = torch.Generator().manual_seed(4)
    ex = TD.mask_blocks(prompt, gen, grid["n_blocks"], grid["lo0"], g, mask_id=MASK, block=BLOCK)
    clean = torch.cat([torch.from_numpy(prompt.astype(np.int64)),
                       torch.from_numpy(gen.astype(np.int64))])
    S = int(ex.input_ids.numel())
    clean = torch.cat([clean, torch.full((S - clean.numel(),), MASK, dtype=torch.long)])
    # the clean sequence must be a genuine test of shift: no two neighbours may be equal
    assert (clean[:-1] != clean[1:]).all(), "degenerate sequence, a shift would be invisible"

    model = OracleLM(clean)
    loss, stats = TD.forward_loss(model, ex, torch.device("cpu"), autocast=False, block=BLOCK)
    assert float(stats["n_rows"]) == float(ex.loss_mask.sum())      # only the masked slots
    assert float(stats["acc"]) == 1.0
    assert float(loss) < 1e-6                                       # unshifted -> perfect

    # the same logits read one position later (the autoregressive convention) are wrong everywhere
    rows = ex.loss_mask
    logits = F.one_hot(clean, VOCAB).float()[rows] * model.scale
    shifted = torch.roll(clean, shifts=-1)[rows]
    assert float(F.cross_entropy(logits, shifted)) > 5.0


def test_forward_loss_feeds_the_masked_sequence_not_the_clean_one():
    prompt, gen, grid = synth(prompt_length=4, n_gen=12, seed=2)
    g = torch.Generator().manual_seed(4)
    ex = TD.mask_blocks(prompt, gen, grid["n_blocks"], grid["lo0"], g, mask_id=MASK, block=BLOCK)
    model = OracleLM(torch.zeros(int(ex.input_ids.numel()), dtype=torch.long))
    TD.forward_loss(model, ex, torch.device("cpu"), autocast=False, block=BLOCK)
    assert torch.equal(model.seen["input_ids"][0], ex.input_ids)
    assert model.seen["use_cache"] is False and model.seen["store_kv"] is False


def test_inv_p_weighting_reweights_the_same_rows():
    prompt, gen, grid = synth(prompt_length=4, n_gen=28, seed=5)
    g = torch.Generator().manual_seed(6)
    ex = TD.mask_blocks(prompt, gen, grid["n_blocks"], grid["lo0"], g, mask_id=MASK, block=BLOCK)
    clean = torch.randint(0, VOCAB, (int(ex.input_ids.numel()),))
    model = OracleLM(clean, scale=1.0)                    # imperfect: the CE is non-trivial
    l_mean, s1 = TD.forward_loss(model, ex, torch.device("cpu"), weighting="mean",
                                 autocast=False, block=BLOCK)
    l_inv, s2 = TD.forward_loss(model, ex, torch.device("cpu"), weighting="inv_p",
                                autocast=False, block=BLOCK)
    assert float(s1["n_rows"]) == float(s2["n_rows"])
    w = ex.weights[ex.loss_mask]
    if float(w.max() - w.min()) > 1e-6:
        assert abs(float(l_mean) - float(l_inv)) > 1e-9
    # both are convex combinations of the same per-row CEs
    logits = F.one_hot(clean, VOCAB).float()[ex.loss_mask]
    rowce = F.cross_entropy(logits, ex.targets[ex.loss_mask], reduction="none")
    assert float(rowce.min()) - 1e-5 <= float(l_inv) <= float(rowce.max()) + 1e-5


# ===========================================================================
# 4. the attention mask
# ===========================================================================
def test_hidden_states_passes_exactly_block_tril_bool():
    model = build_model()
    ids = torch.randint(0, VOCAB, (1, 3 * BLOCK))
    TD.hidden_states(model, ids, block=BLOCK)
    want = block_tril_bool(3 * BLOCK, ids.device, BLOCK)
    for layer in model.model.layers:
        assert layer.seen_mask is not None
        assert layer.seen_mask.dtype == torch.bool and layer.seen_mask.shape == want.shape
        assert torch.equal(layer.seen_mask, want)


def test_the_mask_is_block_causal_in_effect():
    model = build_model().eval()
    ids = torch.randint(0, VOCAB - 1, (1, 3 * BLOCK))
    with torch.no_grad():
        h0 = TD.hidden_states(model, ids, block=BLOCK)
        ids2 = ids.clone()
        ids2[0, -1] = (int(ids2[0, -1]) + 1) % (VOCAB - 1)      # change the LAST block only
        h1 = TD.hidden_states(model, ids2, block=BLOCK)
    assert torch.allclose(h0[: 2 * BLOCK], h1[: 2 * BLOCK], atol=1e-6)
    assert not torch.allclose(h0[2 * BLOCK:], h1[2 * BLOCK:], atol=1e-6)


def test_hidden_states_refuses_a_single_block_and_a_ragged_length():
    model = build_model()
    with pytest.raises(AssertionError):
        TD.hidden_states(model, torch.zeros(1, BLOCK, dtype=torch.long), block=BLOCK)
    with pytest.raises(AssertionError):
        TD.hidden_states(model, torch.zeros(1, 2 * BLOCK + 1, dtype=torch.long), block=BLOCK)


def test_gradient_checkpointing_matches_and_keeps_gradients_and_names():
    a, b = build_model(1), build_model(1)
    b.load_state_dict(a.state_dict())
    keys = list(b.state_dict().keys())
    n = TD.enable_grad_checkpointing(b)
    assert n == LAYERS and TD.enable_grad_checkpointing(b) == 0        # idempotent
    # the patch must NOT rename anything, or the checkpoint stops being loadable by from_pretrained
    assert list(b.state_dict().keys()) == keys
    ids = torch.randint(0, VOCAB, (1, 2 * BLOCK))
    ha = TD.hidden_states(a, ids, block=BLOCK)
    hb = TD.hidden_states(b, ids, block=BLOCK)
    assert torch.allclose(ha, hb, atol=1e-6)
    ha.sum().backward(); hb.sum().backward()
    ga = a.model.layers[0].q.weight.grad
    gb = b.model.layers[0].q.weight.grad
    assert gb is not None and torch.allclose(ga, gb, atol=1e-5)
    with torch.no_grad():                                              # eval path is not checkpointed
        assert torch.allclose(TD.hidden_states(b, ids, block=BLOCK), ha, atol=1e-6)


# ===========================================================================
# 5. the set builder
# ===========================================================================
def record(idx, prompt_length=6, n_gen=10, split="train_pool", correct=1, n_eos=None, seed=0):
    g = np.random.default_rng(seed + idx)
    gen = g.integers(0, VOCAB - 1, n_gen + 4).tolist()
    return {"idx": idx, "split": split, "prompt_tokens": prompt_length, "gen_ids": gen,
            "n_gen_to_eos": n_gen if n_eos is None else n_eos, "correct": correct,
            "stopped_eos": True, "cell": "c", "config_hash": "h"}


def prompts_fn(prompt_length=6):
    return lambda rec: list(np.random.default_rng(rec["idx"]).integers(0, VOCAB - 1, prompt_length))


def read_index(out):
    return [json.loads(l) for l in open(os.path.join(out, "index.jsonl"))]


def test_builder_writes_items_and_trims_at_the_eos(tmp_path):
    out = str(tmp_path / "o")
    recs = [record(i) for i in range(3)]
    s = BD.build(recs, out, prompt_ids_fn=prompts_fn(), max_seq=None, block=BLOCK, verbose=False)
    assert s["kept_items"] == 3
    idx = read_index(out)
    for e, r in zip(idx, recs):
        d = dict(np.load(os.path.join(out, e["file"])).items())
        assert d["gen_ids"].tolist() == r["gen_ids"][:r["n_gen_to_eos"]]     # EOS-inclusive trim
        assert int(d["n_blocks"]) == BD.grid(6, 10, None, BLOCK)["n_blocks"]
        assert int(d["block_start"]) == 4 and int(d["prompt_length"]) == 6


def test_builder_filters(tmp_path):
    out = str(tmp_path / "o")
    recs = [record(0), record(1, correct=0), record(2, split="test"), record(3)]
    s = BD.build(recs, out, prompt_ids_fn=prompts_fn(), only_correct=True, max_seq=None,
                 exclude_idxs={3}, block=BLOCK, verbose=False)
    assert s["kept_items"] == 1 and s["dropped"]["incorrect"] == 1
    assert s["dropped"]["test_split"] == 1 and s["dropped"]["frozen"] == 1
    assert [e["idx"] for e in read_index(out)] == [0]


def test_builder_keeps_incorrect_items_when_asked(tmp_path):
    out = str(tmp_path / "o")
    s = BD.build([record(0, correct=0)], out, prompt_ids_fn=prompts_fn(), only_correct=False,
                 max_seq=None, block=BLOCK, verbose=False)
    assert s["kept_items"] == 1


def test_builder_rejects_a_wrong_prompt(tmp_path):
    out = str(tmp_path / "o")
    s = BD.build([record(0)], out, prompt_ids_fn=prompts_fn(prompt_length=7), max_seq=None,
                 block=BLOCK, verbose=False)
    assert s["kept_items"] == 0 and s["dropped"]["bad"] == 1


def test_builder_max_seq_drops_late_blocks_and_whole_items(tmp_path):
    out = str(tmp_path / "o")
    s = BD.build([record(0, n_gen=40)], out, prompt_ids_fn=prompts_fn(), max_seq=16,
                 block=BLOCK, verbose=False)
    e = read_index(out)[0]
    assert e["n_blocks"] == 3 and e["seq_len"] == 16 and e["n_gen"] == 10 and s["kept_items"] == 1
    out2 = str(tmp_path / "o2")
    s2 = BD.build([record(0, prompt_length=20, n_gen=40)], out2,
                  prompt_ids_fn=prompts_fn(prompt_length=20), max_seq=16, block=BLOCK, verbose=False)
    assert s2["kept_items"] == 0 and s2["dropped"]["bad"] == 1


def test_builder_require_eos(tmp_path):
    out = str(tmp_path / "o")
    r = record(0); r["stopped_eos"] = False
    s = BD.build([r], out, prompt_ids_fn=prompts_fn(), require_eos=True, max_seq=None,
                 block=BLOCK, verbose=False)
    assert s["kept_items"] == 0 and s["dropped"]["no_eos"] == 1


def test_parse_source():
    assert BD.parse_source("a/b=gsm8k_pool:0:3000") == ("a/b", "gsm8k_pool:0:3000", "*.jsonl")
    assert BD.parse_source("a=b=c*.jsonl") == ("a", "b", "c*.jsonl")
    with pytest.raises(AssertionError):
        BD.parse_source("nope")


# ===========================================================================
# 6. the set / holdout / ordering
# ===========================================================================
def test_set_filters_and_holdout_is_disjoint(tmp_path):
    index = write_set(tmp_path, n_items=8)
    ds = TD.DrafterSet(cfg(index))
    assert len(ds) == 8
    tr, ev = ds.split_holdout(3)
    assert len(tr) == 5 and len(ev) == 3
    assert not ({e["idx"] for e in tr.entries} & {e["idx"] for e in ev.entries})


def test_set_min_blocks_and_max_seq_filters(tmp_path):
    index = write_set(tmp_path, n_items=4)
    assert len(TD.DrafterSet(cfg(index, min_blocks=4))) == 0        # every item has 3 blocks
    assert len(TD.DrafterSet(cfg(index, max_seq=8))) == 0
    assert len(TD.DrafterSet(cfg(index, max_seq=16))) == 4


def test_epoch_order_is_a_permutation(tmp_path):
    ds = TD.DrafterSet(cfg(write_set(tmp_path, n_items=7)))
    o0, o1 = ds.epoch_order(0), ds.epoch_order(1)
    assert sorted(o0.tolist()) == list(range(7)) and sorted(o1.tolist()) == list(range(7))
    assert o0.tolist() != o1.tolist()
    assert ds.global_example(0, epoch=0).meta["item"] == int(o0[0])


def test_only_correct_filter(tmp_path):
    index = write_set(tmp_path, n_items=4, correct=[1, 0, 1, 0])
    assert len(TD.DrafterSet(cfg(index, only_correct=True))) == 2
    assert len(TD.DrafterSet(cfg(index, only_correct=False))) == 4


# ===========================================================================
# 7. schedule + checkpoints
# ===========================================================================
def test_lr_schedule_warms_up_then_decays():
    assert TD.lr_scale(0, 10, 100) == pytest.approx(0.1)
    assert TD.lr_scale(9, 10, 100) == pytest.approx(1.0)
    assert TD.lr_scale(10, 10, 100) > TD.lr_scale(50, 10, 100) > TD.lr_scale(99, 10, 100)
    assert TD.lr_scale(99, 10, 100) == pytest.approx(0.1, abs=1e-3)


def fake_snapshot(tmp_path):
    src = tmp_path / "snap"
    src.mkdir()
    for n in ("modeling_sdar.py", "configuration_sdar.py", "chat_template.jinja",
              "tokenizer_config.json", "merges.txt", "model.safetensors"):
        (src / n).write_text("x")
    return str(src)


def test_checkpoint_save_copies_the_remote_code_and_writes_latest(tmp_path):
    model, out = build_model(), str(tmp_path / "run")
    opt = torch.optim.AdamW(model.parameters(), lr=1e-5)
    src = fake_snapshot(tmp_path)
    d = TD.save_checkpoint(model, opt, out, 10, src=src, keep=0)
    got = set(os.listdir(d))
    assert {"config.json", "trainer_state.pt", "modeling_sdar.py", "configuration_sdar.py",
            "chat_template.jinja"} <= got
    assert "model.safetensors" in got
    assert open(os.path.join(out, "latest")).read().strip() == "step000010"
    # save_pretrained's own model.safetensors must not be replaced by the snapshot's stub
    assert os.path.getsize(os.path.join(d, "model.safetensors")) > 100


def test_checkpoint_prune_keeps_the_newest_and_never_latest(tmp_path):
    model, out = build_model(), str(tmp_path / "run")
    opt = torch.optim.AdamW(model.parameters(), lr=1e-5)
    for s in (10, 20, 30):
        TD.save_checkpoint(model, opt, out, s, keep=2)
    assert TD.checkpoint_dirs(out) == ["step000020", "step000030"]
    # `latest` is protected even when it is the oldest survivor
    with open(os.path.join(out, "latest"), "w") as fh:
        fh.write("step000020")
    assert TD.prune_checkpoints(out, 1) == []
    assert TD.checkpoint_dirs(out) == ["step000020", "step000030"]
    assert TD.prune_checkpoints(out, 0) == []                       # 0 keeps everything


def test_checkpoint_resume_restores_weights_optimiser_and_step(tmp_path):
    model, out = build_model(), str(tmp_path / "run")
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    ids = torch.randint(0, VOCAB, (1, 2 * BLOCK))
    TD.hidden_states(model, ids, block=BLOCK).sum().backward()
    opt.step()                                                       # non-empty optimiser state
    ref = {k: v.clone() for k, v in model.state_dict().items()}
    TD.save_checkpoint(model, opt, out, 42, keep=2)

    with torch.no_grad():
        for p in model.parameters():
            p.add_(1.0)
    opt2 = torch.optim.AdamW(model.parameters(), lr=1e-3)
    step = TD.load_checkpoint(model, opt2, out, torch.device("cpu"))   # follows `latest`
    assert step == 42
    for k, v in model.state_dict().items():
        assert torch.allclose(v, ref[k]), k
    assert opt2.state_dict()["state"] and \
        opt2.state_dict()["state"].keys() == opt.state_dict()["state"].keys()


def test_resolve_checkpoint_follows_latest_or_returns_the_path(tmp_path):
    model, out = build_model(), str(tmp_path / "run")
    opt = torch.optim.AdamW(model.parameters(), lr=1e-5)
    d = TD.save_checkpoint(model, opt, out, 7, keep=0)
    assert TD.resolve_checkpoint(out) == d
    assert TD.resolve_checkpoint(d) == d


def test_save_dtype_bf16_halves_the_weights_and_still_resumes(tmp_path):
    model, out = build_model(), str(tmp_path / "run")
    opt = torch.optim.AdamW(model.parameters(), lr=1e-5)
    d32 = TD.save_checkpoint(model, opt, out, 1, keep=0)
    n32 = os.path.getsize(os.path.join(d32, "model.safetensors"))
    d16 = TD.save_checkpoint(model, opt, out, 2, keep=0, save_dtype=torch.bfloat16)
    n16 = os.path.getsize(os.path.join(d16, "model.safetensors"))
    assert n16 < 0.6 * n32
    assert all(p.dtype == torch.float32 for p in model.parameters())   # the live model is untouched
    step = TD.load_checkpoint(model, None, d16, torch.device("cpu"))
    assert step == 2 and all(p.dtype == torch.float32 for p in model.parameters())


# ===========================================================================
# 8. one real optimiser step on the tiny model
# ===========================================================================
def test_one_step_changes_every_parameter_and_only_uses_masked_rows(tmp_path):
    ds = TD.DrafterSet(cfg(write_set(tmp_path, n_items=4)))
    model = build_model(3)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    ex = ds.global_example(0, epoch=0)
    loss, stats = TD.forward_loss(model, ex, torch.device("cpu"), autocast=False, block=BLOCK)
    loss.backward()
    assert float(stats["n_rows"]) == float(ex.loss_mask.sum())
    opt.step()
    moved = [k for k, v in model.state_dict().items() if not torch.equal(v, before[k])]
    assert "lm_head.weight" in moved and "model.embed_tokens.weight" in moved
    assert float(loss.detach()) > 0


# ===========================================================================
# 9. the REAL remote dense code (tiny config, CPU) -- what `labels=` actually does
# ===========================================================================
REMOTE = os.path.join(os.path.dirname(CODE), "hf_cache", "modules", "transformers_modules",
                      "JetLM", "SDAR-1.7B-Chat-b32",
                      "2c83c05d84be90a01c72443b2123ebf0376f6fd4")


def _remote():
    """Import the pinned remote module pair as a package (relative imports need a parent)."""
    import importlib.util
    import types
    pkg = types.ModuleType("sdarpkg_test")
    pkg.__path__ = [REMOTE]
    sys.modules["sdarpkg_test"] = pkg
    mods = []
    for name in ("configuration_sdar", "modeling_sdar"):
        spec = importlib.util.spec_from_file_location(f"sdarpkg_test.{name}",
                                                      os.path.join(REMOTE, name + ".py"))
        m = importlib.util.module_from_spec(spec)
        sys.modules[f"sdarpkg_test.{name}"] = m
        spec.loader.exec_module(m)
        mods.append(m)
    return mods


def _remote_model(C, M, seed=0):
    cfg = C.SDARConfig(vocab_size=200, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                       num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                       max_position_embeddings=512, tie_word_embeddings=False)
    torch.manual_seed(seed)
    return cfg, M.SDARForCausalLM(cfg).eval()


@pytest.mark.skipif(not os.path.isdir(REMOTE), reason="remote SDAR module snapshot not present")
def test_remote_labels_is_plain_unshifted_cross_entropy_at_labelled_positions():
    """The audit's claim, verified: `SDARForCausalLM.forward(labels=...)` (modeling_sdar.py:845-850)
    is `nn.CrossEntropyLoss()(logits.view(-1, V), labels.view(-1))` -- NO shift and ignore_index
    -100, i.e. the diffusion convention, unlike the MoE class's shifted `self.loss_function`."""
    C, M = _remote()
    cfg, model = _remote_model(C, M)
    model.config.fuse_cross_entropy = False       # absent from SDARConfig, present in config.json
    ids = torch.randint(0, cfg.vocab_size - 1, (1, 64))
    mask = block_tril_bool(64, ids.device, 32)
    pos = torch.arange(64).unsqueeze(0)
    lab = torch.full((1, 64), TD.IGNORE, dtype=torch.long)
    sel = torch.tensor([40, 45])
    lab[0, sel] = ids[0, sel]
    with torch.no_grad():
        out = model(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=False, labels=lab)
        unshifted = F.cross_entropy(out.logits[0, sel].float(), ids[0, sel])
        shifted = F.cross_entropy(out.logits[0, sel - 1].float(), ids[0, sel])
    assert float(out.loss) == pytest.approx(float(unshifted), abs=1e-5)
    assert abs(float(out.loss) - float(shifted)) > 1e-3        # a shift would be a different number


@pytest.mark.skipif(not os.path.isdir(REMOTE), reason="remote SDAR module snapshot not present")
def test_remote_labels_path_is_unusable_in_training_mode():
    """`fuse_linear_and_cross_entropy = config.fuse_cross_entropy and self.training` sets
    `logits = None` (modeling_sdar.py:838-843) and the labels branch then dereferences it.  The
    SHIPPED config.json of SDAR-1.7B-Chat-b32 has `fuse_cross_entropy: true`, so this fires on
    the real checkpoint -- which is why the trainer computes the loss itself from `model.model`
    + `lm_head` and never calls `SDARForCausalLM.forward`."""
    C, M = _remote()
    cfg, model = _remote_model(C, M)
    model.config.fuse_cross_entropy = True
    model.train()
    ids = torch.randint(0, cfg.vocab_size - 1, (1, 64))
    with pytest.raises(AttributeError):
        model(input_ids=ids, attention_mask=block_tril_bool(64, ids.device, 32),
              position_ids=torch.arange(64).unsqueeze(0), use_cache=False,
              labels=torch.full((1, 64), TD.IGNORE, dtype=torch.long))


@pytest.mark.skipif(not os.path.isdir(REMOTE), reason="remote SDAR module snapshot not present")
def test_hidden_states_on_the_real_remote_forward_is_block_causal():
    C, M = _remote()
    cfg, model = _remote_model(C, M)
    ids = torch.randint(0, cfg.vocab_size - 1, (1, 96))
    with torch.no_grad():
        h = TD.hidden_states(model, ids, block=32)
        ids2 = ids.clone()
        ids2[0, -1] = (int(ids2[0, -1]) + 1) % (cfg.vocab_size - 1)
        h2 = TD.hidden_states(model, ids2, block=32)
    assert h.shape == (96, cfg.hidden_size)
    assert torch.allclose(h[:64], h2[:64], atol=1e-6)          # earlier blocks cannot see it
    assert not torch.allclose(h[64:], h2[64:], atol=1e-6)


SNAP = os.path.join(os.path.dirname(CODE), "hf_cache", "hub",
                    "models--JetLM--SDAR-1.7B-Chat-b32", "snapshots",
                    "2c83c05d84be90a01c72443b2123ebf0376f6fd4")


@pytest.mark.skipif(not (os.path.isdir(REMOTE) and os.path.isdir(SNAP)),
                    reason="remote SDAR module / snapshot not present")
def test_a_checkpoint_of_the_real_class_reloads_through_from_pretrained(tmp_path):
    """`save_checkpoint` must produce a directory a stock `from_pretrained(trust_remote_code=True,
    local_files_only=True)` can load -- config auto_map + the two remote .py files + the
    tokenizer, all copied from the pinned snapshot."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    C, M = _remote()
    shipped = json.load(open(os.path.join(SNAP, "config.json")))
    cfg, model = _remote_model(C, M)
    cfg.auto_map = shipped["auto_map"]
    cfg.fuse_cross_entropy = shipped["fuse_cross_entropy"]
    out = str(tmp_path / "run")
    opt = torch.optim.AdamW(model.parameters(), lr=1e-5)
    d = TD.save_checkpoint(model, opt, out, 5, src=SNAP, keep=2)

    back = AutoModelForCausalLM.from_pretrained(d, trust_remote_code=True, local_files_only=True,
                                                torch_dtype=torch.float32)
    assert type(back).__name__ == "SDARForCausalLM"
    for a, b in zip(model.state_dict().values(), back.state_dict().values()):
        assert torch.equal(a, b)
    tok = AutoTokenizer.from_pretrained(d, trust_remote_code=True, local_files_only=True)
    assert tok(tok.mask_token)["input_ids"][0] == 151669        # research_protocol.MASK_ID
    assert json.load(open(os.path.join(d, "config.json")))["auto_map"] == shipped["auto_map"]
