"""SDAR block-diffusion decoding for Stage 0.

Semantics follow `evaluation/opencompass/opencompass/models/huggingface_bd3.py`
from JetAstra/SDAR -- the implementation that produced the published table --
NOT the repo-root `generate.py`. The two differ in one load-bearing place:

    generate.py :  token_prob = gather(probs,     token)   # AFTER top-k/top-p
    bd3.py      :  token_prob = gather(ori_probs, token)   # BEFORE top-k/top-p

Under greedy (which the official eval implements as top_k=1) the first collapses
every confidence to exactly 1.0, so `threshold < 1.0` commits the whole block in
a single step and there is no denoising trajectory to measure. The second yields
the true max probability, so partial commits -- and therefore the step-t recall
curve, C-prevstep and E6 -- remain measurable. See the amendment
2026-09-02.

Block grid: SDAR aligns blocks to the WHOLE sequence, so the first decode block
also contains the prompt's tail remainder (prompt_length % block_length real
tokens) and has fewer than `block_length` masked positions. We keep that upstream
behaviour exactly and record a per-block `valid` mask marking which positions are
genuinely generated; analysis must restrict every union/recall to those.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence

import torch
import torch.nn.functional as F
from transformers.cache_utils import DynamicCache


# --------------------------------------------------------------------------
# sampling
# --------------------------------------------------------------------------
def top_k_logits(logits: torch.Tensor, k: int) -> torch.Tensor:
    if k <= 0:
        return logits
    values, _ = torch.topk(logits, k)
    min_values = values[..., -1, None]
    return torch.where(logits < min_values,
                       torch.full_like(logits, float("-inf")), logits)


def top_p_logits(logits: torch.Tensor, p: float) -> torch.Tensor:
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_mask = cumulative_probs > p
    sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
    sorted_mask[..., 0] = False
    mask_indices = torch.scatter(
        torch.full_like(logits, False, dtype=torch.bool), -1, sorted_indices, sorted_mask)
    return logits.masked_fill(mask_indices, float("-inf"))


def sample_with_confidence(logits: torch.Tensor,
                           temperature: float = 1.0,
                           top_k: int = 0,
                           top_p: float = 1.0,
                           greedy_exact: bool = True):
    """Returns (token, confidence).

    `confidence` is the token's probability under the ORIGINAL (temperature-scaled,
    unfiltered) distribution -- bd3.py's `ori_probs`.

    `greedy_exact`: when top_k == 1, take argmax directly instead of multinomial
    over a one-hot. Mathematically identical except for exact logit ties (where
    argmax takes the lowest index and multinomial breaks ties uniformly), and it
    removes all RNG from greedy decoding. Set False for bit-faithful bd3 replay.
    """
    orig_shape = logits.shape[:-1]
    vocab_size = logits.shape[-1]
    logits = logits.reshape(-1, vocab_size)

    assert temperature > 0, "Temperature must be positive"
    if temperature != 1.0:
        logits = logits / temperature
    ori_probs = F.softmax(logits, dim=-1)          # confidence source

    if greedy_exact and top_k == 1:
        token = ori_probs.argmax(dim=-1, keepdim=True)
    else:
        filt = logits
        if top_k > 0:
            filt = top_k_logits(filt, top_k)
        if top_p < 1.0:
            filt = top_p_logits(filt, top_p)
        probs = F.softmax(filt, dim=-1)
        token = torch.multinomial(probs, num_samples=1)

    token_prob = torch.gather(ori_probs, -1, token)
    return token.view(*orig_shape), token_prob.view(*orig_shape)


def get_num_transfer_tokens(block_length: int, steps: int) -> torch.Tensor:
    base = block_length // steps
    remainder = block_length % steps
    n = torch.zeros(steps, dtype=torch.int64) + base
    n[:remainder] += 1
    return n


# --------------------------------------------------------------------------
# results
# --------------------------------------------------------------------------
@dataclass
class BlockTrace:
    """One decode block."""
    block_index: int                 # index into the global block grid
    start: int                       # absolute position of block start
    valid: torch.Tensor              # [block_length] bool: genuinely generated
    committed: List[torch.Tensor] = field(default_factory=list)  # per step [block_length] bool
    confidence: List[torch.Tensor] = field(default_factory=list) # per step [block_length] fp32
    proposed: List[torch.Tensor] = field(default_factory=list)   # per step [block_length] int64
    tokens: Optional[torch.Tensor] = None   # [block_length] final ids
    n_forward: int = 0               # model forwards spent on this block (incl. kv flush)

    @property
    def n_steps(self) -> int:
        return len(self.committed)


@dataclass
class DecodeResult:
    x: torch.Tensor                  # [1, total_length] full sequence
    prompt_length: int
    total_length: int
    block_length: int
    prefill_blocks: int
    num_blocks: int
    blocks: List[BlockTrace]
    stopped_early: bool
    n_forward: int

    def generated_ids(self) -> torch.Tensor:
        return self.x[:, self.prompt_length:]


# --------------------------------------------------------------------------
# decode
# --------------------------------------------------------------------------
@torch.no_grad()
def block_diffusion_generate(
    model,
    input_ids: torch.Tensor,               # [1, L]
    mask_id: int,
    gen_length: int = 512,
    block_length: int = 4,
    denoising_steps: int = 4,
    temperature: float = 1.0,
    top_k: int = 1,
    top_p: float = 1.0,
    threshold: float = 1.0,
    stopping_criteria_idx: Optional[Sequence[int]] = None,
    greedy_exact: bool = True,
    on_forward: Optional[Callable] = None,
    max_new_blocks: Optional[int] = None,
) -> DecodeResult:
    """Faithful port of bd3.block_diffusion_generate with tracing.

    `on_forward(kind, block_trace, step)` is called immediately BEFORE each model
    forward so a recorder can arm itself; `kind` is one of
    'prefill' | 'denoise' | 'kvflush'.
    """
    assert input_ids.shape[0] == 1, "single-sequence decoding only"
    assert block_length % denoising_steps == 0 and block_length >= denoising_steps

    device = model.device
    input_ids = input_ids.to(device)
    prompt_length = input_ids.shape[1]
    past_key_values = DynamicCache()

    num_blocks = (prompt_length + gen_length + block_length - 1) // block_length
    total_length = num_blocks * block_length

    block_mask = torch.tril(torch.ones(num_blocks, num_blocks, device=device))
    attn = (block_mask.repeat_interleave(block_length, dim=0)
                      .repeat_interleave(block_length, dim=1)
                      .unsqueeze(0))
    position_ids = torch.arange(total_length, dtype=torch.long, device=device).unsqueeze(0)

    x = torch.full((1, total_length), mask_id, dtype=torch.long, device=device)
    x[:, :prompt_length] = input_ids

    prefill_blocks = prompt_length // block_length
    prefill_length = prefill_blocks * block_length

    n_forward = 0
    if prefill_length > 0:
        if on_forward:
            on_forward("prefill", None, -1)
        model(input_ids=x[:, :prefill_length],
              attention_mask=attn[:, :prefill_length, :prefill_length],
              position_ids=position_ids[:, :prefill_length],
              past_key_values=past_key_values, use_cache=True, store_kv=True)
        n_forward += 1

    num_transfer = get_num_transfer_tokens(block_length, denoising_steps)
    blocks: List[BlockTrace] = []
    stopped_early = False
    stop_ids = list(stopping_criteria_idx or [])

    last_block = num_blocks
    if max_new_blocks is not None:
        last_block = min(num_blocks, prefill_blocks + max_new_blocks)

    for nb in range(prefill_blocks, last_block):
        lo, hi = nb * block_length, (nb + 1) * block_length
        cur_x = x[:, lo:hi]                      # view into x, as in bd3
        cur_attn = attn[:, lo:hi, :hi]
        cur_pos = position_ids[:, lo:hi]

        abs_pos = torch.arange(lo, hi, device=device)
        bt = BlockTrace(block_index=nb, start=lo,
                        valid=(abs_pos >= prompt_length).cpu())

        for step in range(denoising_steps + 1):
            mask_index = (cur_x == mask_id)
            if mask_index.sum() == 0:
                if on_forward:
                    on_forward("kvflush", bt, step)
                model(input_ids=cur_x, attention_mask=cur_attn, position_ids=cur_pos,
                      past_key_values=past_key_values, use_cache=True, store_kv=True)
                n_forward += 1
                bt.n_forward += 1
                break

            if on_forward:
                on_forward("denoise", bt, step)
            logits = model(input_ids=cur_x, attention_mask=cur_attn, position_ids=cur_pos,
                           past_key_values=past_key_values, use_cache=True,
                           store_kv=False).logits
            n_forward += 1
            bt.n_forward += 1

            x0, x0_p = sample_with_confidence(
                logits, temperature=temperature, top_k=top_k, top_p=top_p,
                greedy_exact=greedy_exact)
            x0 = torch.where(mask_index, x0, cur_x)

            if threshold < 1.0:
                confidence = torch.where(mask_index, x0_p, torch.tensor(
                    float("-inf"), device=device, dtype=x0_p.dtype))
                transfer_index = torch.zeros_like(x0, dtype=torch.bool)
                for j in range(confidence.shape[0]):
                    high_conf = confidence[j] > threshold
                    if high_conf.sum() >= num_transfer[step]:
                        transfer_index[j] = high_conf
                    else:
                        _, idx = torch.topk(confidence[j], num_transfer[step])
                        transfer_index[j, idx] = True
            else:
                confidence = torch.where(mask_index, x0_p, torch.tensor(
                    float("-inf"), device=device, dtype=x0_p.dtype))
                transfer_index = torch.zeros_like(x0, dtype=torch.bool)
                for j in range(confidence.shape[0]):
                    _, idx = torch.topk(confidence[j], k=num_transfer[step])
                    transfer_index[j, idx] = True

            bt.committed.append(transfer_index[0].detach().cpu().clone())
            bt.confidence.append(x0_p[0].detach().float().cpu().clone())
            bt.proposed.append(x0[0].detach().cpu().clone())

            cur_x[transfer_index] = x0[transfer_index]

        bt.tokens = cur_x[0].detach().cpu().clone()
        blocks.append(bt)

        for sid in stop_ids:
            if (x[:, prompt_length:] == sid).any():
                stopped_early = True
                break
        if stopped_early:
            break

    return DecodeResult(x=x, prompt_length=prompt_length, total_length=total_length,
                        block_length=block_length, prefill_blocks=prefill_blocks,
                        num_blocks=num_blocks, blocks=blocks,
                        stopped_early=stopped_early, n_forward=n_forward)
