"""Two-stream training forward for the judge: the SAME per-state loss as `train.forward_state`, much cheaper.

Why it is exact. In `train_forward.training_forward` a state is [prefix + active block] run through the whole
model with the block-tril mask, with the LoRA delta gated OFF everywhere except the active block's draft slots
(`adapter.CTX` flag) and the experts and routers frozen (`--no-train-gates`). So:
  * the prefix rows carry no trainable parameter at all: their hidden states, and hence the keys/values they
    offer the active block at every layer, are the frozen base model's, and no gradient flows through them;
  * the prefix rows never see the active block (block-tril), so they do not depend on it;
  * the active block attends to [the prefix's keys/values + its own], bidirectionally.
Computing the prefix once under no_grad, keeping every layer's (K, V), and then running only the 32-row block
with gradients against those cached (K, V) is the identical computation -- it is exactly what the inference KV
cache does. What changes is the cost:
  * one no-grad pass per ITEM (all its blocks share the prompt + earlier blocks) instead of one full forward +
    checkpoint recompute + backward over the prefix per STATE;
  * many states (blocks of several items) share one batched block pass, so each MoE layer's per-expert loop
    runs once for all of them instead of once per state.
code/corpus1b/tests/test_train_fast.py pins the equality of loss and adapter gradients against forward_state.

Requirements (asserted): pool off (`pool_M is None`), frozen gates (no GateFP32 in the model), draft-flag mode.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from draftalign import adapter as A
from draftalign.losses import active_logits, eval_row_stats, masked_ce_and_certainty
from draftalign.train_forward import apply_rotary_pos_emb, moe_forward

BLOCK = 32


# ---------------------------------------------------------------------------
# micro-batch layout
# ---------------------------------------------------------------------------
@dataclass
class MicroBatch:
    """Items (their clean prefixes) + the states to train on, each tied to one item."""
    prefixes: List[torch.Tensor]        # per item: [L_i] long, the clean prefix up to its latest state's block start
    states: list                        # dataset.State, any order
    item_of_state: List[int]            # index into `prefixes` for each state


def group_states(states: Sequence, block: int = BLOCK) -> MicroBatch:
    """Build a MicroBatch from states (states of one item share one prefix entry)."""
    key_to_idx: Dict[Tuple, int] = {}
    prefixes: List[torch.Tensor] = []
    best: Dict[int, int] = {}
    item_of_state = []
    for st in states:
        key = (st.meta.get("split"), int(st.meta.get("idx")))
        if key not in key_to_idx:
            key_to_idx[key] = len(prefixes)
            prefixes.append(None)                                     # filled below
            best[key_to_idx[key]] = -1
        j = key_to_idx[key]
        item_of_state.append(j)
        if int(st.active_lo) > best[j]:
            best[j] = int(st.active_lo)
            prefixes[j] = st.input_ids[:int(st.active_lo)].clone()    # clean: the sampler fills only the active block
    for j, p in enumerate(prefixes):
        assert p is not None and p.numel() % block == 0, (j, None if p is None else p.numel())
    return MicroBatch(prefixes=prefixes, states=list(states), item_of_state=item_of_state)


def _check_model(model) -> None:
    assert not any(isinstance(m, A.GateFP32) for m in model.modules()), \
        "train_fast needs frozen routers (--no-train-gates): a trainable gate makes the prefix trainable"


# ---------------------------------------------------------------------------
# attention with explicit (K, V)
# ---------------------------------------------------------------------------
def _qkv(attn, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    shape = (*x.shape[:-1], -1, attn.head_dim)
    q = attn.q_norm(attn.q_proj(x).view(shape)).transpose(1, 2)
    k = attn.k_norm(attn.k_proj(x).view(shape)).transpose(1, 2)
    v = attn.v_proj(x).view(shape).transpose(1, 2)
    q, k = apply_rotary_pos_emb(q, k, cos, sin)
    return q, k, v


def _attn_out(attn, q, k, v, mask, shape):
    out = F.scaled_dot_product_attention(query=q, key=k, value=v, attn_mask=mask, is_causal=False,
                                         scale=attn.scaling, enable_gqa=True)
    out = out.transpose(1, 2).contiguous().reshape(*shape, -1).contiguous()
    return attn.o_proj(out)


def _mlp(layer, h: torch.Tensor) -> torch.Tensor:
    mlp = layer.mlp
    if hasattr(mlp, "experts"):
        out, _uniq, _pool = moe_forward(mlp, h, 0, None, None, "full")
        return out
    return mlp(h)


# ---------------------------------------------------------------------------
# stream 1: the prefixes, no grad, base model
# ---------------------------------------------------------------------------
@torch.no_grad()
def prefix_kv(model, prefixes: List[torch.Tensor], block: int = BLOCK) -> Tuple[List[Tuple[torch.Tensor, torch.Tensor]], int]:
    """Every layer's (K, V) [n_items, H_kv, L_max, d] of the clean prefixes under the FROZEN base model.

    Right-padded to L_max (a multiple of `block`). Padded query rows attend to themselves only (no all-masked
    softmax row, hence no NaN); padded keys are never visible to a real row, here or in the block pass.
    """
    inner = model.model
    dev = next(model.parameters()).device
    n = len(prefixes)
    Lmax = max((int(p.numel()) for p in prefixes), default=0)
    if Lmax == 0:
        return [], 0
    ids = torch.zeros(n, Lmax, dtype=torch.long, device=dev)
    kvalid = torch.zeros(n, Lmax, dtype=torch.bool, device=dev)
    for i, p in enumerate(prefixes):
        L = int(p.numel())
        ids[i, :L] = p.to(dev)
        kvalid[i, :L] = True
    nb = Lmax // block
    bm = torch.tril(torch.ones(nb, nb, dtype=torch.bool, device=dev))
    tril = bm.repeat_interleave(block, 0).repeat_interleave(block, 1)            # [Lmax, Lmax]
    mask = tril[None] & kvalid[:, None, :]                                        # [n, Lmax(q), Lmax(k)]
    eye = torch.eye(Lmax, dtype=torch.bool, device=dev)[None] & ~kvalid[:, :, None]
    mask = (mask | eye)[:, None]                                                  # [n, 1, Lmax, Lmax]
    pos = torch.arange(Lmax, device=dev).unsqueeze(0).expand(n, -1)
    h = inner.embed_tokens(ids)
    cos, sin = inner.rotary_emb(h, pos)
    kvs: List[Tuple[torch.Tensor, torch.Tensor]] = []
    off = torch.zeros(n, Lmax, dtype=torch.bool, device=dev)
    with A.flag_scope(off, False):                                                # base model: no LoRA delta
        for layer in inner.layers:
            res = h
            x = layer.input_layernorm(h)
            q, k, v = _qkv(layer.self_attn, x, cos, sin)
            kvs.append((k, v))
            h = res + _attn_out(layer.self_attn, q, k, v, mask, x.shape[:-1])
            res = h
            h = res + _mlp(layer, layer.post_attention_layernorm(h))
    return kvs, Lmax


# ---------------------------------------------------------------------------
# stream 2: the active blocks, with grad
# ---------------------------------------------------------------------------
def block_hidden(model, mb: MicroBatch, kvs, Lmax: int, block: int = BLOCK, draft_flag: bool = True,
                 checkpoint_layers: bool = True, conf_embed: bool = False, mask_id: int = 151669) -> torch.Tensor:
    """[n_states, block, H] final-norm hidden states of every state's active block."""
    inner = model.model
    dev = next(model.parameters()).device
    S = len(mb.states)
    ids = torch.stack([st.input_ids[int(st.active_lo):int(st.active_lo) + block] for st in mb.states]).to(dev)
    los = torch.tensor([int(st.active_lo) for st in mb.states], device=dev)
    pos = los[:, None] + torch.arange(block, device=dev)[None]
    flag = None
    if draft_flag:
        flag = torch.stack([st.open_mask & st.valid & (st.input_ids[int(st.active_lo):int(st.active_lo) + block]
                                                       != mask_id) for st in mb.states]).to(dev)
    flag_any = bool(flag.any()) if flag is not None else False
    h = inner.embed_tokens(ids)
    if flag is not None:
        de = A.get_draft_embed(model)
        if de is not None:
            h = de.apply_to(h, flag)
        if conf_embed:
            ce = A.get_conf_embed(model)
            if ce is not None:
                conf = torch.stack([st.draft_conf if st.draft_conf is not None
                                    else torch.zeros(block, dtype=torch.uint8) for st in mb.states]).to(dev)
                h = ce.apply_to(h, flag, conf)
    cos, sin = inner.rotary_emb(h, pos)
    item_idx = torch.tensor(mb.item_of_state, device=dev)
    # which prefix keys each state may see: positions < its own block start
    kpos = torch.arange(Lmax, device=dev)
    mask = torch.cat([(kpos[None, :] < los[:, None])[:, None, :].expand(S, block, Lmax),
                      torch.ones(S, block, block, dtype=torch.bool, device=dev)], dim=2)[:, None]   # [S,1,blk,Lmax+blk]

    def run_layer(hh, li):
        layer = inner.layers[li]
        with A.flag_scope(flag, flag_any):
            res = hh
            x = layer.input_layernorm(hh)
            q, k, v = _qkv(layer.self_attn, x, cos, sin)
            if Lmax:
                kp, vp = kvs[li]
                k = torch.cat([kp.index_select(0, item_idx).to(k.dtype), k], dim=2)
                v = torch.cat([vp.index_select(0, item_idx).to(v.dtype), v], dim=2)
                m = mask
            else:
                m = None
            hh = res + _attn_out(layer.self_attn, q, k, v, m, x.shape[:-1])
            res = hh
            return res + _mlp(layer, layer.post_attention_layernorm(hh))

    for li in range(len(inner.layers)):
        if checkpoint_layers and torch.is_grad_enabled():
            h = checkpoint(run_layer, h, li, use_reentrant=False)
        else:
            h = run_layer(h, li)
    return inner.norm(h)


# ---------------------------------------------------------------------------
# the loss of a micro-batch
# ---------------------------------------------------------------------------
@dataclass
class LossCfg:
    beta: float = 0.5
    chat_beta: float = 0.0                    # certainty weight on chat states (0.5 = the round-3 loss everywhere)
    temp: float = 0.5
    calib: float = 1.0
    override_weight: float = 2.0
    code_override_weight: Optional[float] = 3.0
    chat_override_weight: float = 1.0         # 1 = unbiased odds on chat (docs/CORPUS_1B_BUILD.md)
    mask_id: int = 151669


def state_row_weight(st, cfg: LossCfg, block: int = BLOCK) -> Optional[torch.Tensor]:
    """1 on every loss row, or the domain's override weight on rows whose draft token is wrong."""
    dom = st.meta.get("domain", "")
    w = cfg.override_weight
    if dom == "code" and cfg.code_override_weight is not None:
        w = cfg.code_override_weight
    if dom == "chat":
        w = cfg.chat_override_weight
    # older (round-3) items carry no domain; the split decides, exactly as train.override_row_weight
    if not dom and cfg.code_override_weight is not None and str(st.meta.get("split", "")).startswith("opencode"):
        w = cfg.code_override_weight
    if float(w) == 1.0:
        return None
    lo = int(st.active_lo)
    inp = st.input_ids[lo:lo + block][st.open_mask]
    ovr = (inp != cfg.mask_id) & (inp != st.targets[st.open_mask])
    return torch.where(ovr, torch.tensor(float(w)), torch.tensor(1.0))


def micro_batch_loss(model, states: Sequence, cfg: LossCfg, *, block: int = BLOCK, draft_flag: bool = True,
                     checkpoint_layers: bool = True, conf_embed: bool = False, want_eval: bool = False):
    """Sum over states of each state's loss (as `train.forward_state` computes it) + per-state stats.

    Returns (loss_sum, per_state_stats [dict], logits_per_state [fp32 detached, only when want_eval]).
    The caller divides by the number of states in the optimiser step.
    """
    _check_model(model)
    mb = group_states(states, block)
    p0 = next(model.parameters())
    with torch.autocast(device_type=p0.device.type, dtype=torch.bfloat16, enabled=p0.is_cuda):   # as forward_state
        kvs, Lmax = prefix_kv(model, mb.prefixes, block)
        hid = block_hidden(model, mb, kvs, Lmax, block, draft_flag, checkpoint_layers, conf_embed, cfg.mask_id)
    dev = hid.device
    rows = torch.stack([st.open_mask for st in mb.states]).to(dev)                  # [S, block]
    logits_all = active_logits(model, hid[rows])                                   # fp32 [sum rows, V]
    counts = rows.sum(1).tolist()
    total = logits_all.new_zeros(())
    stats, logits_out = [], []
    off = 0
    for st, n in zip(mb.states, counts):
        lg = logits_all[off:off + n]
        off += n
        tg = st.targets[st.open_mask].to(dev)
        rw = state_row_weight(st, cfg, block)
        beta = cfg.chat_beta if st.meta.get("domain") == "chat" else cfg.beta
        loss, s = masked_ce_and_certainty(lg, tg, beta=beta, temp=cfg.temp, mask_id=cfg.mask_id,
                                          calib=cfg.calib, row_weight=None if rw is None else rw.to(dev))
        total = total + loss
        s = {k: float(v) for k, v in s.items()}
        s["loss"] = float(loss.detach())
        s["domain"] = st.meta.get("domain", "")
        if want_eval:
            lo = int(st.active_lo)
            inp = st.input_ids[lo:lo + block][st.open_mask].to(dev)
            ovr = (inp != cfg.mask_id) & (inp != tg)
            s.update({f"ev_{k}": float(v) for k, v in eval_row_stats(lg.detach(), tg, override=ovr).items()})
            logits_out.append(lg.detach())
        stats.append(s)
    return total, stats, logits_out
