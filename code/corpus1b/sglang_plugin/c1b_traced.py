"""SGLang dLLM algorithms that also RECORD what they did, for the 1B judge corpus.

Install: `python code/corpus1b/sglang_plugin/install.py` copies this file into
`<site-packages>/sglang/srt/dllm/algorithm/`, where SGLang discovers every module that defines
`Algorithm` (sglang/srt/dllm/algorithm/__init__.py::import_algorithms). One module can expose one
`Algorithm`, so the drafter rule lives in the sibling file c1b_static_draft.py.

Pinned to sglang==0.5.10.post1 (install.py refuses anything else): the loop below is a verbatim copy of
`sglang/srt/dllm/algorithm/low_confidence.py::LowConfidence.run` of that release, plus four recording lines.
The decode rule is therefore SGLang's own and is identical to the reference harness
(`sdar_harness/decode.py` at threshold 0.95): per pass, commit every masked slot whose top-1 probability
is STRICTLY above the threshold, else the single most confident slot; argmax; the [MASK] column is not
suppressed.

What is recorded (one JSON line per request-block, appended to $C1B_TRACE_DIR/trace.<pid>.jsonl):
    rid    the request id the client passed (the corpus pid)
    lo     absolute position of the block start (= the request's prefix length)
    start  slots of the block that were NOT masked when it started (the prompt tail of the first block)
    ids    the block's 32 final token ids
    steps  per slot, the pass (0-based) that committed it; -1 for slots that were never masked
The training data needs `steps`: the state sampler rebuilds "everything committed before pass p" from it
(code/draftalign/dataset.py). SGLang itself returns only the tokens.

With C1B_TRACE_DIR unset nothing is written and the algorithm is exactly LowConfidence.
"""
from __future__ import annotations

import json
import os
from typing import List, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

from sglang.srt.dllm.algorithm.low_confidence import LowConfidence
from sglang.srt.dllm.config import DllmConfig
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.model_runner import ModelRunner


class _TraceSink:
    """Append-only JSONL writer, one file per process (no cross-process interleaving)."""

    def __init__(self):
        self.dir = os.environ.get("C1B_TRACE_DIR") or None
        self._fh = None

    def write(self, rows: List[dict]) -> None:
        if not self.dir or not rows:
            return
        if self._fh is None:
            os.makedirs(self.dir, exist_ok=True)
            self._fh = open(os.path.join(self.dir, f"trace.{os.getpid()}.jsonl"), "a", buffering=1)
        self._fh.write("".join(json.dumps(r, separators=(",", ":")) + "\n" for r in rows))
        self._fh.flush()


def block_starts(forward_batch: ForwardBatch, block_size: int) -> List[int]:
    """Absolute start of each request's block: its prefix length (seq_len - block_size)."""
    lens = getattr(forward_batch, "extend_prefix_lens_cpu", None)
    if lens is not None:
        return [int(x) for x in lens]
    seq = getattr(forward_batch, "seq_lens_cpu", None)
    if seq is None:
        seq = forward_batch.seq_lens.tolist()
    return [int(x) - block_size for x in seq]


def batch_rids(forward_batch: ForwardBatch, batch_size: int) -> List[str]:
    rids = getattr(forward_batch, "rids", None)
    if rids is None:
        return [f"?{i}" for i in range(batch_size)]
    return [str(r) for r in rids]


class LowConfidenceTraced(LowConfidence):

    def __init__(self, config: DllmConfig):
        super().__init__(config)
        self.sink = _TraceSink()

    def run(
        self,
        model_runner: ModelRunner,
        forward_batch: ForwardBatch,
    ) -> Tuple[Union[LogitsProcessorOutput, torch.Tensor], List[torch.Tensor], bool]:
        batch_size = forward_batch.batch_size
        # Here, the forward_batch full logits contains all the blocks
        # such as [dllm_block_size * batch_size, hidden_size]
        start_list = []
        mask_index = forward_batch.input_ids == self.mask_id

        # Fast path: if there is no mask token, forward and save kv cache
        if torch.sum(mask_index).item() == 0:
            out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
            logits_output, can_run_cuda_graph = out.logits_output, out.can_run_graph

            next_token_ids = []
            return logits_output, next_token_ids, can_run_cuda_graph

        # ---- recording (1/4): per-slot commit pass, -1 = never masked
        steps = torch.where(mask_index, torch.full_like(forward_batch.input_ids, -2),
                            torch.full_like(forward_batch.input_ids, -1))

        # Calculate start positions for each block
        for block_id in range(batch_size):
            block_start = block_id * self.block_size
            block_end = block_start + self.block_size
            block_input_ids = forward_batch.input_ids[block_start:block_end]
            block_mask_index = block_input_ids == self.mask_id
            start = self.block_size - torch.sum(block_mask_index).item()
            start_list.append(start)

        for step_i in range(self.block_size):
            mask_index = forward_batch.input_ids == self.mask_id
            if torch.sum(mask_index).item() == 0:
                break

            out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
            logits_output, can_run_cuda_graph = out.logits_output, out.can_run_graph
            assert batch_size == forward_batch.input_ids.shape[0] // self.block_size
            for batch_id in range(batch_size):
                curr_block_start = batch_id * self.block_size
                curr_block_end = curr_block_start + self.block_size
                block_input_ids = forward_batch.input_ids[
                    curr_block_start:curr_block_end,
                ]
                block_mask_index = block_input_ids == self.mask_id
                if torch.sum(block_mask_index).item() == 0:
                    continue
                curr_logits = logits_output.full_logits[
                    curr_block_start:curr_block_end,
                ]

                x = torch.argmax(curr_logits, dim=-1)
                p = torch.squeeze(
                    torch.gather(
                        F.softmax(curr_logits, dim=-1),
                        dim=-1,
                        index=torch.unsqueeze(x, -1),
                    ),
                    -1,
                )
                x = torch.where(block_mask_index, x, block_input_ids)
                confidence = torch.where(block_mask_index, p, -np.inf)

                transfer_index = confidence > self.threshold

                if transfer_index.sum().item() == 0:
                    _, select_index = torch.topk(confidence, k=1)
                    transfer_index[select_index] = True

                block_input_ids[transfer_index] = x[transfer_index]
                # ---- recording (2/4): the pass that committed these slots
                steps[curr_block_start:curr_block_end][transfer_index] = step_i

        out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
        logits_output, can_run_cuda_graph = out.logits_output, out.can_run_graph
        # Here next token ids is tricky to implement the dynamic lengths,
        # so we return a list of tensors
        next_token_ids = torch.reshape(forward_batch.input_ids, (batch_size, -1))
        next_token_ids_list = [
            next_token_ids[i, start_list[i] :] for i in range(batch_size)
        ]

        # ---- recording (3/4): one row per request-block
        if self.sink.dir:
            ids_cpu = next_token_ids.tolist()
            steps_cpu = torch.reshape(steps, (batch_size, -1)).tolist()
            rids = batch_rids(forward_batch, batch_size)
            los = block_starts(forward_batch, self.block_size)
            # ---- recording (4/4)
            self.sink.write([{"rid": rids[i], "lo": los[i], "start": int(start_list[i]),
                              "ids": ids_cpu[i], "steps": steps_cpu[i]} for i in range(batch_size)])

        return logits_output, next_token_ids_list, can_run_cuda_graph


Algorithm = LowConfidenceTraced
