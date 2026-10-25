"""CPU tests for the clean_prefix fix in code/draftalign/train_drafter.py (2026-09-08).

The bug: every example noised EVERY block of the response in one sequence, so every block was
predicted from a NOISED prefix, while at inference the prefix is always CLEAN (the committed
tokens of the earlier blocks, `seqcache.py`) and only the current block is masked.  The fix
mirrors `draftalign/dataset.py` (`StateSampler`): one active block per example, clean prefix,
loss only on the active block.

What is pinned here:
  a. the clean_prefix construction (clean prefix, no [MASK] outside the active block, loss rows
     inside it, targets = the teacher's tokens, S % 32 == 0 and S > 32, a reproducible active
     block);
  b. `--conditioning noisy_all` still reproduces the pre-fix examples BIT-FOR-BIT -- the digests
     below were produced by `git show HEAD:code/draftalign/train_drafter.py` before the fix was
     written, so the provenance of the 2026-09-07 run is intact;
  c. `evaluate()` in clean_prefix mode runs end to end over held-out items with `hidden_states` /
     `active_logits` monkeypatched to deterministic fakes, and returns the documented keys;
  d. the in-loop agreement path (`train_drafter.agreement` ->
     `eval_drafter_agreement.item_agreement` -> `SeqCache` / `draft_block`) runs on CPU against a
     fake drafter and returns an agreement in [0, 1].

No real model, no GPU.  Conventions follow code/tests/test_train_drafter.py (BLOCK = 4 so a
"block" is small, VOCAB = 40, MASK = 39).

Run:  /venv/main/bin/python -m pytest -q code/tests/test_train_drafter_clean_prefix.py
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

import numpy as np
import pytest
import torch

CODE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if CODE not in sys.path:
    sys.path.insert(0, CODE)
if os.path.join(CODE, "compat") not in sys.path:
    sys.path.insert(0, os.path.join(CODE, "compat"))

from transformers.modeling_outputs import CausalLMOutputWithPast          # noqa: E402

from draftalign import build_drafter_set as BD                            # noqa: E402
from draftalign import train_drafter as TD                                # noqa: E402

BLOCK = 4
VOCAB = 40
MASK = 39


# ===========================================================================
# synthetic items (same shape as code/tests/test_train_drafter.py)
# ===========================================================================
def synth(prompt_length=6, n_gen=10, seed=0):
    g = np.random.default_rng(seed)
    prompt = g.integers(0, VOCAB - 1, prompt_length).astype(np.int32)
    gen = g.integers(0, VOCAB - 1, n_gen).astype(np.int32)
    grid = BD.grid(prompt_length, n_gen, max_seq=None, block=BLOCK)
    return prompt, gen, grid


def write_set(tmp_path, n_items=6, prompt_length=6, split="train_pool"):
    """A minimal build_drafter_set output tree; item i has 10 + 2i generated tokens."""
    root = tmp_path / "ds"
    (root / "items").mkdir(parents=True, exist_ok=True)
    entries = []
    for i in range(n_items):
        prompt, gen, grid = synth(prompt_length, 10 + 2 * i, seed=100 + i)
        name = f"{split}__{i:05d}.npz"
        np.savez(root / "items" / name, prompt_ids=prompt, gen_ids=gen,
                 n_blocks=np.int32(grid["n_blocks"]), block_start=np.int32(grid["lo0"]),
                 prompt_length=np.int32(prompt_length), block_length=np.int32(BLOCK),
                 n_pad=np.int32(grid["n_pad"]), seq_len=np.int32(grid["seq_len"]),
                 correct=np.int8(1), idx=np.int32(i))
        entries.append({"file": os.path.join("items", name), "idx": i, "split": split,
                        "correct": 1, "prompt_length": prompt_length,
                        "n_gen": grid["n_gen_kept"], "n_blocks": grid["n_blocks"],
                        "block_start": grid["lo0"], "n_pad": grid["n_pad"],
                        "seq_len": grid["seq_len"]})
    with open(root / "index.jsonl", "w") as fh:
        for e in entries:
            fh.write(json.dumps(e) + "\n")
    return str(root / "index.jsonl")


def cfg(index, **kw):
    d = dict(index_path=index, seed=0, block=BLOCK, mask_id=MASK, max_seq=None, min_blocks=1,
             only_correct=False)
    d.update(kw)
    return TD.DrafterSetConfig(**d)


def digest(ex) -> str:
    """A bit-for-bit fingerprint of an Example's tensors + drawn ratios."""
    h = hashlib.sha256()
    for t in (ex.input_ids, ex.targets, ex.loss_mask.long(), (ex.weights * 1e6).round().long()):
        h.update(np.ascontiguousarray(t.numpy()).tobytes())
    h.update(np.asarray(ex.ratios, np.float64).tobytes())
    return h.hexdigest()[:16]


def clean_sequence(prompt, gen, S):
    """The CLEAN sequence of the first S positions (pad slots past the response left as -1)."""
    full = np.full(S, -1, np.int64)
    P_ = len(prompt)
    full[:min(P_, S)] = np.asarray(prompt, np.int64)[:min(P_, S)]
    n = max(0, min(P_ + len(gen), S) - P_)
    full[P_:P_ + n] = np.asarray(gen, np.int64)[:n]
    return full


# ===========================================================================
# a. the clean_prefix example
# ===========================================================================
def test_eligible_blocks_drops_only_a_single_block_sequence():
    # prompt 6, gen 10 -> lo0 = 4, 3 blocks: every block's sequence is > 1 block
    assert TD.eligible_blocks(6, 10, 3, 4, BLOCK) == [0, 1, 2]
    # a prompt shorter than one block (block_start == 0): block 0 alone is a ONE-block sequence
    assert TD.eligible_blocks(2, 10, 3, 0, BLOCK) == [1, 2]
    # ... and an item that is a single block has no eligible block at all
    assert TD.eligible_blocks(2, 2, 1, 0, BLOCK) == []


@pytest.mark.parametrize("active", [0, 1, 2])
def test_clean_prefix_example_is_the_inference_state(active):
    prompt, gen, grid = synth(6, 10, seed=0)                  # lo0 = 4, 3 blocks, seq_len 16
    g = torch.Generator().manual_seed(1)
    ex = TD.mask_active_block(prompt, gen, grid["n_blocks"], grid["lo0"], active, g,
                              mask_id=MASK, block=BLOCK)
    S = ex.meta["seq_len"]
    lo = grid["lo0"] + active * BLOCK

    # the sequence stops right after the active block, on the grid, and is never one block
    assert S == lo + BLOCK
    assert S % BLOCK == 0 and S > BLOCK
    assert ex.input_ids.numel() == S == ex.targets.numel() == ex.loss_mask.numel()

    clean = clean_sequence(prompt, gen, S)
    # 1. the PREFIX (everything before the active block) is the teacher's tokens, exactly
    assert torch.equal(ex.input_ids[:lo], torch.from_numpy(clean[:lo]))
    # 2. no [MASK] anywhere outside the active block
    assert not bool((ex.input_ids[:lo] == MASK).any())
    # 3. loss rows are a subset of the active block's RESPONSE slots
    rows = torch.nonzero(ex.loss_mask).flatten().tolist()
    assert rows, "the objective wants >= 1 masked slot"
    assert all(lo <= r < lo + BLOCK for r in rows)
    assert all(bool(ex.resp[r]) for r in rows)
    assert all(clean[r] >= 0 for r in rows)                   # never a pad slot beyond the response
    # 4. targets at the loss rows are the CLEAN teacher tokens; IGNORE everywhere else
    for r in rows:
        assert int(ex.targets[r]) == int(clean[r])
        assert int(ex.input_ids[r]) == MASK
    assert set(torch.nonzero(ex.targets != TD.IGNORE).flatten().tolist()) == set(rows)
    # 5. unmasked response slots of the active block still hold the teacher's token
    for j in range(lo, S):
        if not ex.loss_mask[j] and clean[j] >= 0:
            assert int(ex.input_ids[j]) == int(clean[j])
    # 6. pad slots past the response hold [MASK] and are never loss rows
    for j in range(S):
        if clean[j] < 0:
            assert int(ex.input_ids[j]) == MASK and not bool(ex.loss_mask[j])
    assert ex.meta["conditioning"] == "clean_prefix" and ex.meta["active_block"] == active


def test_clean_prefix_force_ratio_one_masks_every_response_slot_of_the_active_block():
    prompt, gen, grid = synth(6, 10, seed=0)
    g = torch.Generator().manual_seed(3)
    ex = TD.mask_active_block(prompt, gen, grid["n_blocks"], grid["lo0"], 1, g,
                              mask_id=MASK, block=BLOCK, force_ratio=1.0)
    lo = grid["lo0"] + BLOCK
    assert torch.equal(ex.loss_mask[lo:lo + BLOCK], ex.resp[lo:lo + BLOCK])
    assert bool((ex.input_ids[lo:lo + BLOCK] == MASK).all())
    assert ex.ratios == [1.0]


def test_clean_prefix_refuses_a_single_block_sequence():
    prompt, gen, grid = synth(2, 10, seed=0)                  # block_start == 0
    g = torch.Generator().manual_seed(0)
    with pytest.raises(AssertionError, match="single-block sequence"):
        TD.mask_active_block(prompt, gen, grid["n_blocks"], grid["lo0"], 0, g,
                             mask_id=MASK, block=BLOCK)


def test_active_block_is_reproducible_for_a_given_seed_epoch_item(tmp_path):
    ds = TD.DrafterSet(cfg(write_set(tmp_path), conditioning="clean_prefix"))
    a = ds.example(2, epoch=0)
    b = ds.example(2, epoch=0)
    assert a.meta["active_block"] == b.meta["active_block"]
    assert digest(a) == digest(b)
    assert a.meta["active_block"] in ds.item_blocks(2)
    # a different epoch is a fresh draw, and a different seed is a different stream
    epochs = [ds.example(2, epoch=e).meta["active_block"] for e in range(12)]
    assert len(set(epochs)) > 1, "the active block must be redrawn each visit"
    other = TD.DrafterSet(cfg(write_set(tmp_path), seed=7, conditioning="clean_prefix"))
    assert [other.example(2, epoch=e).meta["active_block"] for e in range(12)] != epochs


def test_active_block_draw_is_uniform_over_the_eligible_blocks(tmp_path):
    ds = TD.DrafterSet(cfg(write_set(tmp_path), conditioning="clean_prefix"))
    blocks = ds.item_blocks(3)
    seen = [ds.example(3, epoch=e).meta["active_block"] for e in range(400)]
    assert set(seen) == set(blocks)
    counts = np.array([seen.count(b) for b in blocks], np.float64)
    assert (counts / counts.sum()).min() > 0.5 / len(blocks)    # no block starved


def test_clean_prefix_examples_pass_the_hidden_states_asserts(tmp_path):
    ds = TD.DrafterSet(cfg(write_set(tmp_path), conditioning="clean_prefix"))
    for k in range(len(ds)):
        for b in ds.item_blocks(k):
            ex = ds.example(k, epoch=0, active=b)
            S = ex.meta["seq_len"]
            assert S % BLOCK == 0 and S > BLOCK
            assert int(ex.meta["n_loss"]) >= 1


def test_forward_loss_accepts_a_clean_prefix_example_unchanged(tmp_path, monkeypatch):
    ds = TD.DrafterSet(cfg(write_set(tmp_path), conditioning="clean_prefix"))
    ex = ds.example(0, epoch=0)
    install_fake_forward(monkeypatch)
    loss, st = TD.forward_loss(FakeLM(), ex, torch.device("cpu"), autocast=False, block=BLOCK)
    assert torch.isfinite(loss)
    assert float(st["n_rows"]) == float(ex.meta["n_loss"])
    assert 0.0 <= float(st["acc"]) <= 1.0


# ===========================================================================
# b. provenance: noisy_all is bit-for-bit the pre-2026-09-08 construction
# ===========================================================================
# Produced with `git show HEAD:code/draftalign/train_drafter.py` (the version the 2026-09-07 run
# used), same synthetic items, same generator seeds -- see the module docstring.
OLD_MASK_BLOCKS = {                       # (prompt_length, n_gen, generator seed) -> digest
    (6, 10, 1): "f23b0e9d732a6808",
    (6, 10, 7): "2fde3c9dfb30a21c",
    (0, 33, 3): "76d37ccdc88af394",
    (32, 70, 5): "de03d7d3d6333822",
}
OLD_SET_EXAMPLES = {                      # (item, epoch) -> digest, through DrafterSet.example
    (0, 0): "4e91382328e0230f", (0, 1): "d8b8c989c188e434",
    (1, 0): "9cb1fe4e0d457ab5", (1, 1): "9aefd35abd2d223a",
    (2, 0): "873ce6fd630dae67", (2, 1): "66f56b2907d44c1e",
}


@pytest.mark.parametrize("key", sorted(OLD_MASK_BLOCKS))
def test_noisy_all_mask_blocks_is_bit_for_bit_the_old_construction(key):
    pl, ng, gseed = key
    prompt, gen, grid = synth(pl, ng, seed=100 + pl + ng)
    g = torch.Generator().manual_seed(gseed)
    ex = TD.mask_blocks(prompt, gen, grid["n_blocks"], grid["lo0"], g, mask_id=MASK, block=BLOCK)
    assert digest(ex) == OLD_MASK_BLOCKS[key]


@pytest.mark.parametrize("key", sorted(OLD_SET_EXAMPLES))
def test_noisy_all_set_examples_are_bit_for_bit_the_old_ones(tmp_path, key):
    k, epoch = key
    ds = TD.DrafterSet(cfg(write_set(tmp_path), conditioning="noisy_all"))
    ex = ds.example(k, epoch=epoch)
    assert digest(ex) == OLD_SET_EXAMPLES[key]
    assert ex.meta["conditioning"] == "noisy_all"


def test_noisy_all_is_the_dataclass_default_so_other_importers_are_unchanged(tmp_path):
    """eval_drafter_agreement.py builds a DrafterSetConfig without `conditioning` and must keep
    the pre-fix entry list, holdout split and example semantics."""
    assert TD.DrafterSetConfig(index_path="x").conditioning == "noisy_all"
    ds = TD.DrafterSet(cfg(write_set(tmp_path)))              # no conditioning passed
    assert digest(ds.example(2, epoch=0)) == OLD_SET_EXAMPLES[(2, 0)]


def test_the_two_conditionings_split_the_holdout_identically(tmp_path):
    index = write_set(tmp_path, n_items=6)
    a = TD.DrafterSet(cfg(index, conditioning="noisy_all")).split_holdout(2)
    b = TD.DrafterSet(cfg(index, conditioning="clean_prefix")).split_holdout(2)
    for x, y in zip(a, b):
        assert [e["idx"] for e in x.entries] == [e["idx"] for e in y.entries]


def test_global_example_skips_an_item_with_no_eligible_block(tmp_path, capsys):
    """An item whose whole sequence is ONE block cannot be a clean_prefix example."""
    root = tmp_path / "ds1"
    (root / "items").mkdir(parents=True)
    prompt, gen, grid = synth(2, 2, seed=5)                   # block_start 0, 1 block
    np.savez(root / "items" / "train_pool__00000.npz", prompt_ids=prompt, gen_ids=gen,
             n_blocks=np.int32(grid["n_blocks"]), block_start=np.int32(grid["lo0"]),
             prompt_length=np.int32(2), block_length=np.int32(BLOCK),
             n_pad=np.int32(grid["n_pad"]), seq_len=np.int32(grid["seq_len"]),
             correct=np.int8(1), idx=np.int32(0))
    good_prompt, good_gen, good_grid = synth(6, 10, seed=1)
    np.savez(root / "items" / "train_pool__00001.npz", prompt_ids=good_prompt, gen_ids=good_gen,
             n_blocks=np.int32(good_grid["n_blocks"]), block_start=np.int32(good_grid["lo0"]),
             prompt_length=np.int32(6), block_length=np.int32(BLOCK),
             n_pad=np.int32(good_grid["n_pad"]), seq_len=np.int32(good_grid["seq_len"]),
             correct=np.int8(1), idx=np.int32(1))
    with open(root / "index.jsonl", "w") as fh:
        for i, gr in ((0, grid), (1, good_grid)):
            fh.write(json.dumps({"file": f"items/train_pool__{i:05d}.npz", "idx": i,
                                 "split": "train_pool", "correct": 1, "n_blocks": gr["n_blocks"],
                                 "block_start": gr["lo0"], "seq_len": gr["seq_len"],
                                 "n_gen": gr["n_gen_kept"]}) + "\n")
    ds = TD.DrafterSet(cfg(str(root / "index.jsonl"), conditioning="clean_prefix"))
    assert ds.item_blocks(0) == [] and ds.item_blocks(1)
    for s in range(4):
        assert ds.global_example(s, epoch=0).meta["item"] == 1
    assert "[skip] item 0" in capsys.readouterr().out


# ===========================================================================
# c/d. fakes: a deterministic stand-in forward, and a stand-in drafter
# ===========================================================================
class FakeLM:
    """Only an identity `lm_head` is needed; `hidden_states` / `active_logits` are patched."""

    def __init__(self):
        self.training = True

    def train(self):
        self.training = True
        return self

    def eval(self):
        self.training = False
        return self


def fake_hidden_states(model, input_ids, block=TD.BLOCK):
    """Deterministic [S, VOCAB] "logits": hidden[i] favours (input_ids[i] + i) % VOCAB.

    Keeps `hidden_states`' contract (the asserts) so the clean_prefix sequences are checked."""
    assert input_ids.dim() == 2 and input_ids.shape[0] == 1, input_ids.shape
    S = int(input_ids.shape[1])
    assert S % block == 0, f"S={S} is not on the {block}-grid"
    assert S > block, "a single-block sequence must never reach the forward"
    ids = input_ids[0]
    pos = torch.arange(S)
    pick = (ids + pos) % VOCAB
    return torch.nn.functional.one_hot(pick, VOCAB).float() * 3.0


def install_fake_forward(monkeypatch):
    monkeypatch.setattr(TD, "hidden_states", fake_hidden_states)
    monkeypatch.setattr(TD, "active_logits", lambda model, h, rows: h[rows])


class FakeDrafter(torch.nn.Module):
    """Enough of the remote `SDARForCausalLM` for `SeqCache` + `draft_block` on CPU.

    Emits the item's TRUE token at every EVEN absolute position and a wrong one at every odd
    position, so the replayed agreement is a known ~0.5 instead of a degenerate 0.
    """

    def __init__(self, truth: np.ndarray, vocab: int = VOCAB, mask_id: int = MASK):
        super().__init__()
        self.truth = torch.from_numpy(np.asarray(truth, np.int64))
        self.vocab, self.mask_id = vocab, mask_id
        self.device = torch.device("cpu")
        self.n_calls = 0

    def forward(self, input_ids=None, attention_mask=None, position_ids=None,
                past_key_values=None, use_cache=None, store_kv=None, **kw):
        self.n_calls += 1
        T = int(input_ids.shape[1])
        if store_kv and past_key_values is not None:                 # grow the cache like the real one
            kv = torch.zeros(1, 1, T, 4)
            past_key_values.update(kv, kv.clone(), 0)
        pos = position_ids[0] if position_ids is not None else torch.arange(T)
        want = torch.where(pos % 2 == 0, self.truth[pos],
                           (self.truth[pos] + 1) % (self.vocab - 1))   # never the [MASK] id
        logits = torch.nn.functional.one_hot(want, self.vocab).float() * 5.0
        return CausalLMOutputWithPast(logits=logits.unsqueeze(0))


# ===========================================================================
# c. evaluate() in clean_prefix mode
# ===========================================================================
def test_evaluate_clean_prefix_returns_the_documented_keys(tmp_path, monkeypatch):
    ds = TD.DrafterSet(cfg(write_set(tmp_path, n_items=4), conditioning="clean_prefix"))
    _, hold = ds.split_holdout(2)
    assert len(hold) == 2
    seen = []
    real = fake_hidden_states

    def spy(model, input_ids, block=TD.BLOCK):
        seen.append(input_ids.clone())
        return real(model, input_ids, block)

    monkeypatch.setattr(TD, "hidden_states", spy)
    monkeypatch.setattr(TD, "active_logits", lambda model, h, rows: h[rows])
    model = FakeLM()
    ev = TD.evaluate(model, hold, 2, torch.device("cpu"), autocast=False, block=BLOCK,
                     blocks_per_item=2)
    for k in ("sampled_acc", "sampled_ce", "sampled_top1", "sampled_rows", "sampled_items",
              "sampled_blocks", "allmask_acc", "allmask_ce", "allmask_top1", "allmask_rows",
              "allmask_items", "allmask_blocks"):
        assert k in ev, k
        assert np.isfinite(ev[k]), k
    assert 0.0 <= ev["sampled_acc"] <= 1.0 and 0.0 <= ev["allmask_acc"] <= 1.0
    assert ev["sampled_items"] == ev["allmask_items"] == 2.0
    assert ev["sampled_blocks"] <= 2 * 2                       # blocks_per_item cap honoured
    assert model.training, "evaluate() must restore train mode"
    # every scored sequence stops on the block grid and is longer than one block
    assert seen and all(int(t.shape[1]) % BLOCK == 0 and int(t.shape[1]) > BLOCK for t in seen)


def test_evaluate_clean_prefix_is_deterministic_and_allmask_masks_everything(tmp_path, monkeypatch):
    ds = TD.DrafterSet(cfg(write_set(tmp_path, n_items=4), conditioning="clean_prefix"))
    _, hold = ds.split_holdout(2)
    install_fake_forward(monkeypatch)
    a = TD.evaluate(FakeLM(), hold, 2, torch.device("cpu"), autocast=False, block=BLOCK,
                    blocks_per_item=2)
    b = TD.evaluate(FakeLM(), hold, 2, torch.device("cpu"), autocast=False, block=BLOCK,
                    blocks_per_item=2)
    assert a == b
    # allmask scores every response slot of the block, so it has at least as many rows
    assert a["allmask_rows"] >= a["sampled_rows"]
    # the block subset is deterministic and capped
    assert hold.eval_blocks(0, 2) == hold.eval_blocks(0, 2)
    assert len(hold.eval_blocks(0, 2)) <= 2
    assert set(hold.eval_blocks(0, 2)) <= set(hold.item_blocks(0))


def test_evaluate_noisy_all_still_scores_whole_items(tmp_path, monkeypatch):
    ds = TD.DrafterSet(cfg(write_set(tmp_path, n_items=4), conditioning="noisy_all"))
    _, hold = ds.split_holdout(2)
    install_fake_forward(monkeypatch)
    ev = TD.evaluate(FakeLM(), hold, 2, torch.device("cpu"), autocast=False, block=BLOCK)
    assert {"sampled_acc", "allmask_acc", "sampled_rows", "allmask_items"} <= set(ev)
    assert "sampled_blocks" not in ev                          # a per-block count is clean_prefix only
    assert ev["sampled_items"] == 2.0


# ===========================================================================
# d. the agreement-in-the-loop call path
# ===========================================================================
def test_agreement_in_the_loop_runs_on_cpu(tmp_path):
    index = write_set(tmp_path, n_items=4)
    ds = TD.DrafterSet(cfg(index, conditioning="clean_prefix"))
    it = ds.arrays(0)
    truth = np.concatenate([np.asarray(it["prompt_ids"], np.int64),
                            np.asarray(it["gen_ids"], np.int64)])
    model = FakeDrafter(truth).train()
    out = TD.agreement(model, ds, 1, torch.device("cpu"), setting="s8", autocast=False,
                       block=BLOCK)
    assert 0.0 <= out["agreement"] <= 1.0
    assert out["items"] == 1.0 and out["slots"] > 0 and out["blocks"] > 0
    assert model.training, "agreement() must restore train mode"
    assert model.n_calls > 0
    # the fake is right at every even absolute position: agreement is about a half
    assert 0.3 <= out["agreement"] <= 0.7


def test_agreement_matches_item_agreement_on_the_same_item(tmp_path):
    from draftalign.eval_drafter_agreement import item_agreement

    index = write_set(tmp_path, n_items=4)
    ds = TD.DrafterSet(cfg(index, conditioning="clean_prefix"))
    it = ds.arrays(0)
    truth = np.concatenate([np.asarray(it["prompt_ids"], np.int64),
                            np.asarray(it["gen_ids"], np.int64)])
    item = {k: it[k] for k in ("prompt_ids", "gen_ids", "block_start", "prompt_length")}
    item["idx"] = 0
    direct = item_agreement(FakeDrafter(truth).eval(), item, torch.device("cpu"), "s8",
                            mask_id=MASK, block=BLOCK)
    looped = TD.agreement(FakeDrafter(truth), ds, 1, torch.device("cpu"), setting="s8",
                          autocast=False, block=BLOCK)
    assert direct["agreement"] == looped["agreement"]
    assert direct["slots"] == looped["slots"]


# ===========================================================================
# checkpoints: the `best` marker is never pruned
# ===========================================================================
def test_prune_never_removes_the_best_or_latest(tmp_path):
    out = str(tmp_path / "run")
    for step in (100, 200, 300, 400):
        os.makedirs(os.path.join(out, f"step{step:06d}"))
    TD.write_marker(out, "best", os.path.join(out, "step000100"))
    TD.write_marker(out, "latest", os.path.join(out, "step000400"))
    assert TD.marker(out, "best") == "step000100"
    gone = TD.prune_checkpoints(out, keep=1)
    assert sorted(gone) == ["step000200", "step000300"]
    assert sorted(TD.checkpoint_dirs(out)) == ["step000100", "step000400"]


def test_disagree_weight_is_documented_but_refused():
    with pytest.raises(AssertionError, match="disagree-weight"):
        TD.main(["--index", "x", "--out", "y", "--disagree-weight", "2.0"])
