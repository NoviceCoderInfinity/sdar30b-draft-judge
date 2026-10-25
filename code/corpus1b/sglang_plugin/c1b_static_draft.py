"""SGLang dLLM algorithm for the TEACHER-FORCED DRAFTS of the 1B judge corpus (SDAR-1.7B-Chat-b32).

Install with code/corpus1b/sglang_plugin/install.py (copies it into sglang/srt/dllm/algorithm/).

The rule is the round-3 drafter setting `s8` of code/draftalign/drafter.py::draft_block, re-expressed
over SGLang's batched block forward:

  * a block with n_open masked slots is filled in steps = min(8, n_open) forwards;
  * forward k commits the nt[k] most confident still-masked slots, nt = get_num_transfer_tokens(n_open, steps)
    (base n_open // steps each, the remainder on the first steps; sums to n_open);
  * numerics as drafter.py: logits upcast to fp32, the [MASK] column suppressed (-inf), fp32 softmax, argmax,
    confidence = the argmax probability.

The client (code/corpus1b/draft_sglang.py) sends one request per block of the 30B's kept answer: input_ids =
prompt + the 30B's FINAL tokens before the block (teacher forcing), max_new_tokens = the block's generated
slots, so SGLang runs exactly one decode block per request. With the radix cache on, consecutive blocks of
one item reuse the prefix KV.

Recorded per request-block (JSON line in $C1B_TRACE_DIR/trace.<pid>.jsonl):
    rid, lo, start, ids[32], conf[32]   conf = the committing forward's fp32 probability of the drafted token
                                                (-1.0 for slots that were never masked)

Pinned to sglang==0.5.10.post1 like c1b_traced.py; the forward/KV bookkeeping mirrors
LowConfidence.run of that release (denoise forwards, then one final forward that stores the block).
"""
from __future__ import annotations

from typing import List, Tuple, Union

import torch
import torch.nn.functional as F

from sglang.srt.dllm.algorithm.base import DllmAlgorithm
from sglang.srt.dllm.config import DllmConfig
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.model_runner import ModelRunner

from sglang.srt.dllm.algorithm.c1b_traced import _TraceSink, batch_rids, block_starts


def num_transfer_tokens(n_open: int, steps: int) -> List[int]:
    """== sdar_harness.decode.get_num_transfer_tokens(n_open, steps).tolist()."""
    base, rem = divmod(int(n_open), int(steps))
    return [base + (1 if k < rem else 0) for k in range(int(steps))]


class StaticDraftTraced(DllmAlgorithm):

    def __init__(self, config: DllmConfig):
        super().__init__(config)
        self.steps = int(config.algorithm_config.get("steps", 8))
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

        start_list, sched = [], []
        for b in range(batch_size):
            blk = forward_batch.input_ids[b * bs:(b + 1) * bs]
            n_open = int(torch.sum(blk == self.mask_id).item())
            start_list.append(bs - n_open)
            sched.append(num_transfer_tokens(n_open, min(self.steps, n_open)) if n_open else [])
        conf_rec = torch.full(forward_batch.input_ids.shape, -1.0, dtype=torch.float32,
                              device=forward_batch.input_ids.device)

        n_iter = max((len(s) for s in sched), default=0)
        logits_output, can_run_cuda_graph = None, False
        for k in range(n_iter):
            out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
            logits_output, can_run_cuda_graph = out.logits_output, out.can_run_graph
            for b in range(batch_size):
                if k >= len(sched[b]):
                    continue
                lo, hi = b * bs, (b + 1) * bs
                blk = forward_batch.input_ids[lo:hi]
                open_ = blk == self.mask_id
                if not bool(open_.any()):
                    continue
                lg = logits_output.full_logits[lo:hi].to(torch.float32, copy=True)
                lg[:, self.mask_id] = float("-inf")
                probs = F.softmax(lg, dim=-1)
                conf, x0 = probs.max(dim=-1)
                conf_open = torch.where(open_, conf, torch.full_like(conf, float("-inf")))
                take = min(int(sched[b][k]), int(open_.sum().item()))
                idx = torch.topk(conf_open, take).indices
                transfer = torch.zeros_like(open_)
                transfer[idx] = True
                blk[transfer] = x0[transfer]
                conf_rec[lo:hi][transfer] = conf[transfer]

        # store the finished blocks' KV (LowConfidence's closing forward)
        out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
        logits_output, can_run_cuda_graph = out.logits_output, out.can_run_graph
        ids = torch.reshape(forward_batch.input_ids, (batch_size, -1))
        next_token_ids_list = [ids[i, start_list[i]:] for i in range(batch_size)]

        if self.sink.dir:
            ids_cpu = ids.tolist()
            conf_cpu = torch.reshape(conf_rec, (batch_size, -1)).tolist()
            rids = batch_rids(forward_batch, batch_size)
            los = block_starts(forward_batch, bs)
            self.sink.write([{"rid": rids[i], "lo": los[i], "start": int(start_list[i]), "ids": ids_cpu[i],
                              "conf": [round(float(c), 5) for c in conf_cpu[i]]} for i in range(batch_size)])
        return logits_output, next_token_ids_list, can_run_cuda_graph


Algorithm = StaticDraftTraced
