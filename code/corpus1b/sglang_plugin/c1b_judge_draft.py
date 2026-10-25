"""SGLang dLLM algorithm for the 1.7B DRAFTER engine of the judge-round generator: first drafts AND re-drafts of
partially committed blocks, set per request by a side file the judge writes.

Install with code/corpus1b/sglang_plugin/install.py (copies it into sglang/srt/dllm/algorithm/).  At import it needs
the repo's code/ dir (draftalign.loop_guard, corpus1b/sglang_plugin/draft_core.py): set C1B_CODE_DIR=<repo>/code in
the engine's environment (the import fails with a message naming it otherwise; run from the repo file it finds
code/ relative to itself).  It is appended to sys.path, so it never shadows an installed package.

The rule is draft_core.DraftFill = draftalign/drafter.py::draft_block (static) re-expressed without the SeqCache;
this file only drives every block of the batch in lock-step: one model_runner.forward per step, one DraftFill.step
per block still stepping (a block with fewer steps stops early), then the closing forward that stores the KV
(LowConfidence's, as in c1b_static_draft.py).

Side file, per request, read once when the request's block (a block with at least one [MASK]) is processed and
then deleted: $C1B_SIDE_DIR/<rid>.json (write it atomically, tmp + os.replace, before sending the request):
    fill   [slot, ...]          slots to fill (default: every [MASK] slot). Only [MASK] slots are filled: other
                                listed slots are dropped (the trace's "fill" is the effective set). [MASK] slots
                                outside it stay [MASK] (a re-draft's hidden tentative slots).
    steps  int                  forwards = min(steps, n_fill) (default: the yaml's steps, 8)
    ban    [[slot, token], ...] those logits are -inf at every step (phase 4: the rejected draft at each hole)
    guard  {"n": 12, "prompt_length": int, "prefix_ids": [...]}
                                the phase-5 loop guard: LoopGuard(n, mask_id).start_item(prompt_length) then
                                begin_block(prefix_ids, lo); prefix_ids = the final sequence before the block (>= lo)
With no side file (or C1B_SIDE_DIR unset) a block is filled exactly as StaticDraftTraced (c1b_static_draft.py).

Recorded per request-block (JSON line in $C1B_TRACE_DIR/trace.<pid>.jsonl):
    rid, lo, start, ids[32], conf[32]   conf = the committing forward's fp32 probability (unrounded; -1.0 where
                                         not filled), fill [slots], n_fwd, raw_mask_argmax, guard_slots, guard_changed
next_token_ids = the block from start = 32 - (number of [MASK] slots on entry), as c1b_static_draft.py.

Pinned to sglang==0.5.10.post1 like c1b_traced.py.  CPU test (fake sglang modules):
    python3 -m pytest -q code/corpus1b/tests/test_judge_draft_plugin.py
"""
from __future__ import annotations

import json
import os
import sys
from typing import List, Optional, Tuple, Union

import torch

from sglang.srt.dllm.algorithm.base import DllmAlgorithm
from sglang.srt.dllm.config import DllmConfig
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.model_runner import ModelRunner

from sglang.srt.dllm.algorithm.c1b_traced import _TraceSink, batch_rids, block_starts


def _code_dir() -> str:
    env = os.environ.get("C1B_CODE_DIR")
    cand = os.path.abspath(env or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
    need = [os.path.join(cand, "draftalign", "loop_guard.py"),
            os.path.join(cand, "corpus1b", "sglang_plugin", "draft_core.py")]
    missing = [p for p in need if not os.path.isfile(p)]
    if missing:
        where = "C1B_CODE_DIR" if env else "the dir two levels above this file (C1B_CODE_DIR is unset)"
        raise ImportError(f"c1b_judge_draft: set C1B_CODE_DIR to the repo's code/ dir; {where} = {cand} lacks {missing}")
    return cand


_CODE = _code_dir()
if _CODE not in sys.path:
    sys.path.append(_CODE)
from draftalign.loop_guard import LoopGuard                                   # noqa: E402
from corpus1b.sglang_plugin.draft_core import DraftFill                       # noqa: E402


def read_side(side_dir: Optional[str], rid: str) -> Optional[dict]:
    """The request's side file, deleted after reading; None if there is none."""
    if not side_dir:
        return None
    p = os.path.join(side_dir, f"{rid}.json")
    if not os.path.isfile(p):
        return None
    with open(p) as fh:
        spec = json.load(fh)
    os.remove(p)
    return spec


def begin_fill(spec: Optional[dict], blk: torch.Tensor, lo: int, mask_id: int, default_steps: int) -> DraftFill:
    """A DraftFill for block `blk` [bl] (absolute start `lo`) as the side file `spec` asks."""
    masked = blk == mask_id
    spec = spec or {}
    fill = masked.clone()
    if spec.get("fill") is not None:
        fill = torch.zeros_like(masked)
        slots = [int(s) for s in spec["fill"] if 0 <= int(s) < blk.numel()]
        fill[torch.tensor(slots, dtype=torch.long, device=blk.device)] = True
        fill &= masked
    steps = max(1, int(spec.get("steps", default_steps)))
    ban = [(int(s), int(t)) for s, t in spec.get("ban") or []]
    guard = None
    g = spec.get("guard")
    if g:
        prefix = [int(t) for t in g["prefix_ids"]]
        assert len(prefix) >= lo, f"guard prefix_ids has {len(prefix)} tokens, the block starts at {lo}"
        guard = LoopGuard(int(g.get("n", 12)), mask_id)
        guard.start_item(int(g["prompt_length"]))
        guard.begin_block(prefix, lo)
    return DraftFill(mask_id).begin(blk, fill, steps, ban=ban or None, guard=guard)


class DraftTracedV2(DllmAlgorithm):

    def __init__(self, config: DllmConfig):
        super().__init__(config)
        self.steps = int(config.algorithm_config.get("steps", 8))
        self.side_dir = os.environ.get("C1B_SIDE_DIR") or None
        self.sink = _TraceSink()

    def run(
        self,
        model_runner: ModelRunner,
        forward_batch: ForwardBatch,
    ) -> Tuple[Union[LogitsProcessorOutput, torch.Tensor], List[torch.Tensor], bool]:
        batch_size = forward_batch.batch_size
        bs = self.block_size
        mask_index = forward_batch.input_ids == self.mask_id

        # prefill-only batch: forward and store the KV, exactly as LowConfidence does
        if torch.sum(mask_index).item() == 0:
            out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
            return out.logits_output, [], out.can_run_graph

        rids = batch_rids(forward_batch, batch_size)
        los = block_starts(forward_batch, bs)
        start_list, fills = [], []
        for b in range(batch_size):
            blk = forward_batch.input_ids[b * bs:(b + 1) * bs]
            n_masked = int(torch.sum(blk == self.mask_id).item())
            start_list.append(bs - n_masked)
            spec = read_side(self.side_dir, rids[b]) if n_masked else None   # a mask-free block is a prefill chunk
            fills.append(begin_fill(spec, blk, los[b], self.mask_id, self.steps))

        n_iter = max((df.n_steps for df in fills), default=0)
        logits_output, can_run_cuda_graph = None, False
        for k in range(n_iter):
            out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
            logits_output, can_run_cuda_graph = out.logits_output, out.can_run_graph
            for b, df in enumerate(fills):
                if df.done:
                    continue
                lo, hi = b * bs, (b + 1) * bs
                df.step(logits_output.full_logits[lo:hi])
                forward_batch.input_ids[lo:hi] = df.cur

        # store the finished blocks' KV (LowConfidence's closing forward)
        out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
        logits_output, can_run_cuda_graph = out.logits_output, out.can_run_graph
        ids = torch.reshape(forward_batch.input_ids, (batch_size, -1))
        next_token_ids_list = [ids[i, start_list[i]:] for i in range(batch_size)]

        if self.sink.dir:
            ids_cpu = ids.tolist()
            rows = []
            for i, df in enumerate(fills):
                conf = torch.where(df.fill, df.p_commit, torch.full_like(df.p_commit, -1.0))
                rows.append({"rid": rids[i], "lo": los[i], "start": int(start_list[i]), "ids": ids_cpu[i],
                             "conf": [float(c) for c in conf.tolist()],
                             "fill": torch.nonzero(df.fill).flatten().tolist(), "n_fwd": df.n_fwd,
                             "raw_mask_argmax": int(df.raw_mask), "guard_slots": df.g_slots,
                             "guard_changed": df.g_changed})
            self.sink.write(rows)
        return logits_output, next_token_ids_list, can_run_cuda_graph


Algorithm = DraftTracedV2
