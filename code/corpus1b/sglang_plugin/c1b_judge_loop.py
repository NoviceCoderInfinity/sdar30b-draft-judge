"""SGLang dLLM algorithm for the 30B JUDGE of the phase-5 judge round: the flag-gated judge LoRA (c1b_judge_lora.py)
runs judge_core.run_blocks over each scheduling round's blocks, and asks a separate DRAFTER server (the 1.7B under
c1b_judge_draft.py's DraftTracedV2) for every first draft and re-draft over HTTP.

The decisions are all judge_core's (FULL judge, rd cadence, ban, loop guard, stock fallback, closing forward); this
file is only the glue: one model_runner.forward per judge pass for the whole batch, the per-request sequence, the
drafter round trip, the trace.  Data generation and harness parity only (the AGENTS.md measurement rule): nothing
timed through this plugin is ever a reported speed.

Install with install.py (copies THIS file into sglang/srt/dllm/algorithm/).  Its helpers are NOT copied: judge_core.py
and c1b_judge_lora.py are imported from the repo as `corpus1b.sglang_plugin.<name>` through $C1B_CODE_DIR (appended to
sys.path; default: two levels above this file), the mechanism c1b_judge_draft.py uses for draft_core.py.  One copy of
each module per process matters: c1b_judge_lora's `JudgeCtx` is a module singleton, and a second copy under the
algorithm package would be a second, silent context.

Environment (all required; the orchestrator code/judge_round/judge_sglang.py sets them):
    C1B_CODE_DIR            <repo>/code
    C1B_JUDGE_ADAPTER       the flag-gated judge adapter dir (adapter.save_adapter format), attached on the first run()
    C1B_JUDGE_SIDE_DIR      per-request side files of the judge (below)
    C1B_JUDGE_TRACE_DIR     this plugin's trace: trace.<pid>.jsonl
    C1B_DRAFTER_URL         http://127.0.0.1:PORT of the drafter server
    C1B_DRAFTER_SIDE_DIR    the drafter's side files (its C1B_SIDE_DIR)
    C1B_DRAFTER_TRACE_DIR   the drafter's trace (its C1B_TRACE_DIR)

Judge side file, read when the request's first decode block arrives, then deleted (a rid without one raises):
    $C1B_JUDGE_SIDE_DIR/<rid>.json = {"prompt_ids": [...], "mode": "decode" | "record", "max_new_tokens": N,
        "stop_ids": [EOS ids SGLang stops on; [] in record mode],
        "teacher": [[32 ints] per block]   (record mode: the vanilla tokens of each decode block, index 0 = the first,
                                            its prompt tail included, so every entry is the full 32-slot block)}
Per request the plugin keeps `seq` = prompt + the final tokens of every closed block (what the judge's KV holds) and
the block index; the state is dropped when SGLang finishes the request: len(seq) - len(prompt) >= max_new_tokens, or
(decode) a stop id among the block's new tokens.

Per decode batch:
    blocks   cur = the block's ids (prompt tail + [MASK]), valid = slot >= start (start = 32 - #[MASK]; the masks must be
             the trailing slots), lo = extend_prefix_lens_cpu
    forward  ids written into forward_batch.input_ids in place; c1b_judge_lora.judge_forward_prep builds input_embeds
             (draft vector at flagged slots; None when nothing is flagged); EVERY forward runs inside
             JudgeCtx.flag_scope(flag [B*32]) -- all False on stock passes, the closing forward and prefill batches;
             input_embeds is reset to None after each forward
    drafter  one POST {"input_ids": [seq[:lo] + block], "sampling_params": [{"temperature": 0, "max_new_tokens": 1,
             "ignore_eos": true}], "rid": ["<rid>#<block>#<serial>"]} per draft round, the drafter side file
             {"fill", "steps", "ban", "guard"} written first; the rows are then read from the drafter's trace
    closing  record mode: the teacher block (teacher forcing); decode mode: the pipeline's tokens
    returns  next_token_ids[i] = tokens_out[start:]
Trace row per request-block (JSON line):
    {"rid", "block", "lo", "start", "summary": judge_core.block_summary(b),
     "states": [per judge pass: {"pass", "input", "flag", "open", "draft", "commit", "tentative", "remask", "forced",
                "argmax", "p", "q" (6 decimals, null where not read), "redraft": null | {"holes", "tokens",
                "p_commit", "input", "n_fwd"}}]   (record mode only)}

SGLang behaviour relied on (0.5.10.post1, read from the source, not run here): run() is called once per scheduling
round with all-prefill or all-decode batches; forward_batch.input_ids [B*32] / rids / extend_prefix_lens_cpu;
model_runner.forward(...) -> .logits_output.full_logits [B*32, V], .can_run_graph; forward_extend feeds
input_embeds.bfloat16() to the model when set; every forward writes the block's KV (the last one stays); the
scheduler extends output_ids with next_token_ids[i] and finishes a request on a stop id among them (unless
ignore_eos) or at max_new_tokens.  CUDA graphs must be off (the LoRA branches in Python): asserted per forward.

CPU test (fake sglang, fake judge model, fake drafter):
    python3 -m pytest -q code/corpus1b/tests/test_judge_loop_plugin.py
"""
from __future__ import annotations

import dataclasses
import json
import logging
import os
import sys
import time
import urllib.request
from typing import Dict, List, Sequence, Tuple, Union

import torch

from sglang.srt.dllm.algorithm.base import DllmAlgorithm
from sglang.srt.dllm.config import DllmConfig
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.model_runner import ModelRunner

from sglang.srt.dllm.algorithm.c1b_traced import _TraceSink, batch_rids, block_starts

logger = logging.getLogger(__name__)


def _code_dir() -> str:
    env = os.environ.get("C1B_CODE_DIR")
    cand = os.path.abspath(env or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
    need = [os.path.join(cand, "corpus1b", "sglang_plugin", "judge_core.py"),
            os.path.join(cand, "corpus1b", "sglang_plugin", "c1b_judge_lora.py"),
            os.path.join(cand, "corpus1b", "gen_sglang.py")]
    missing = [p for p in need if not os.path.isfile(p)]
    if missing:
        where = "C1B_CODE_DIR" if env else "the dir two levels above this file (C1B_CODE_DIR is unset)"
        raise ImportError(f"c1b_judge_loop: set C1B_CODE_DIR to the repo's code/ dir; {where} = {cand} lacks {missing}")
    return cand


_CODE = _code_dir()
if _CODE not in sys.path:
    sys.path.append(_CODE)
from corpus1b.sglang_plugin import judge_core as JC                          # noqa: E402
from corpus1b.sglang_plugin import c1b_judge_lora as JL                      # noqa: E402

REQUIRED_ENV = ("C1B_JUDGE_ADAPTER", "C1B_JUDGE_SIDE_DIR", "C1B_JUDGE_TRACE_DIR", "C1B_DRAFTER_URL",
                "C1B_DRAFTER_SIDE_DIR", "C1B_DRAFTER_TRACE_DIR")
CONFIG_KEYS = ("block_size", "tau_commit", "tau_override", "remask_only", "max_passes", "redraft_every",
               "redraft_rate", "redraft_ban", "loop_guard", "threshold", "draft_steps")
DRAFT_WAIT_S = 120.0          # the drafter's rows must be in its trace this long after the POST returned
DRAFT_POLL_S = 0.005
HTTP_TIMEOUT_S = 600.0
DRAFTER_SAMPLING = {"temperature": 0.0, "max_new_tokens": 1, "ignore_eos": True}
_NO_PROXY = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def http_open(req, timeout: float):
    """urlopen without proxies: an http_proxy in the environment must never carry the 127.0.0.1 drafter calls."""
    return _NO_PROXY.open(req, timeout=timeout)


def judge_config(ac: dict, mask_id: int) -> JC.JudgeConfig:
    """The algorithm yaml (keys as judge_round/judge_sglang.py writes them) -> JudgeConfig; an unknown key raises."""
    unknown = sorted(set(ac) - set(CONFIG_KEYS))
    assert not unknown, f"c1b_judge_loop: unknown algorithm config keys {unknown}"
    tov = ac.get("tau_override")
    return JC.JudgeConfig(
        mask_id=int(mask_id), tau_commit=float(ac.get("tau_commit", 0.95)),
        tau_override=None if tov is None else float(tov), remask_only=bool(ac.get("remask_only", False)),
        max_passes=int(ac.get("max_passes", JC.FULL_MAX_PASSES)), redraft_every=int(ac.get("redraft_every", 0)),
        redraft_rate=int(ac.get("redraft_rate", 4)), redraft_ban=bool(ac.get("redraft_ban", False)),
        loop_guard=int(ac.get("loop_guard", 0)), threshold=float(ac.get("threshold", 0.95)),
        draft_steps=int(ac.get("draft_steps", JC.DRAFT_STEPS)))


def write_json_atomic(path: str, obj) -> None:
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w") as fh:
        json.dump(obj, fh, separators=(",", ":"))
    os.replace(tmp, path)


def read_judge_side(side_dir: str, rid: str) -> dict:
    """The request's side file (deleted after reading); a missing one is an error naming the rid."""
    p = os.path.join(side_dir, f"{rid}.json")
    if not os.path.isfile(p):
        raise RuntimeError(f"c1b_judge_loop: no side file for request {rid!r} ({p}); the judge needs the prompt ids")
    with open(p) as fh:
        spec = json.load(fh)
    os.remove(p)
    assert spec.get("mode") in ("decode", "record"), f"{rid}: mode {spec.get('mode')!r}"
    assert spec.get("prompt_ids") and int(spec["max_new_tokens"]) > 0, f"{rid}: bad side file"
    if spec["mode"] == "record":
        assert spec.get("teacher"), f"{rid}: record mode needs the teacher blocks"
    return spec


class _ReqState:
    """One request: the final sequence the judge's KV holds and where its next block starts."""

    def __init__(self, spec: dict):
        self.prompt_ids = [int(t) for t in spec["prompt_ids"]]
        self.prompt_length = len(self.prompt_ids)
        self.mode = spec["mode"]
        self.teacher = [[int(t) for t in blk] for blk in spec.get("teacher") or []]
        self.max_new = int(spec["max_new_tokens"])
        self.stop_ids = {int(t) for t in spec.get("stop_ids") or []}
        self.seq = list(self.prompt_ids)
        self.block_index = 0


def _num(x: float, nd: int = 6):
    return None if x != x else round(float(x), nd)                       # NaN (slot not read) -> null


def states_json(b: JC.BlockRun) -> List[dict]:
    """The recorder's per-pass states of one block as plain JSON lists."""
    out = []
    for s in b.states:
        rd = s.get("redraft")
        out.append({
            "pass": int(s["pass"]), "input": s["input"].tolist(), "flag": s["flag"].tolist(),
            "open": s["open"].tolist(), "draft": s["draft"].tolist(), "commit": s["commit"].tolist(),
            "tentative": s["tentative"].tolist(), "remask": s["remask"].tolist(), "forced": bool(s["forced"]),
            "argmax": s["argmax"].tolist(), "p": [_num(x) for x in s["p"].tolist()],
            "q": [_num(x) for x in s["q"].tolist()],
            "redraft": None if rd is None else {
                "holes": rd["holes"].tolist(), "tokens": rd["tokens"].tolist(),
                "p_commit": [_num(x) for x in rd["p_commit"].tolist()], "input": rd["input"].tolist(),
                "n_fwd": int(rd["n_fwd"])}})
    return out


class JudgeLoopTraced(DllmAlgorithm):

    def __init__(self, config: DllmConfig):
        super().__init__(config)
        missing = [k for k in REQUIRED_ENV if not os.environ.get(k)]
        if missing:
            raise RuntimeError(f"c1b_judge_loop: environment {missing} not set (judge_round/judge_sglang.py sets them)")
        ac = dict(config.algorithm_config or {})
        assert int(ac.get("block_size", self.block_size)) == self.block_size
        self.cfg = judge_config(ac, self.mask_id)
        self.adapter_dir = os.environ["C1B_JUDGE_ADAPTER"]
        self.side_dir = os.environ["C1B_JUDGE_SIDE_DIR"]
        self.drafter_url = os.environ["C1B_DRAFTER_URL"].rstrip("/")
        self.drafter_side_dir = os.environ["C1B_DRAFTER_SIDE_DIR"]
        self.drafter_trace_dir = os.environ["C1B_DRAFTER_TRACE_DIR"]
        self.sink = _TraceSink()
        self.sink.dir = os.environ["C1B_JUDGE_TRACE_DIR"]
        self.states: Dict[str, _ReqState] = {}
        self.serial = 0                       # one per drafter call: every drafter rid is unique
        self._dreader = None

    # ------------------------------------------------------------------------------------------------------------
    def _ensure_adapter(self, model) -> None:
        if getattr(model, "_judge_adapter", None) is None:
            info = JL.attach_judge_adapter(model, self.adapter_dir)
            logger.warning(f"c1b_judge_loop: judge adapter attached {info}")

    def _state_for_block(self, rid: str, lo: int, start: int, blk: List[int]) -> _ReqState:
        st = self.states.get(rid)
        if st is None:
            st = self.states[rid] = _ReqState(read_judge_side(self.side_dir, rid))
        bs, L = self.block_size, st.prompt_length
        lo0 = (L // bs) * bs
        if lo == lo0:                                          # the first decode block (or a re-run of it)
            assert start == L - lo0, f"{rid}: first block starts at slot {start}, the prompt tail is {L - lo0}"
            assert blk[:start] == st.prompt_ids[lo0:L], f"{rid}: block 0's fixed slots are not the prompt tail"
            st.seq, st.block_index = list(st.prompt_ids), 0
        else:
            assert lo > lo0 and (lo - lo0) % bs == 0 and start == 0, f"{rid}: block at {lo} (lo0 {lo0}, start {start})"
            assert lo <= len(st.seq), f"{rid}: block at {lo} but only {len(st.seq)} final tokens"
            st.seq, st.block_index = st.seq[:lo], (lo - lo0) // bs  # a re-run of a closed block rolls back
        if st.mode == "record":
            assert st.block_index < len(st.teacher), f"{rid}: no teacher block {st.block_index}"
            t = st.teacher[st.block_index]
            assert len(t) == bs and t[:start] == blk[:start], f"{rid}: teacher block {st.block_index} vs the prefix"
        return st

    # -- the drafter round trip ----------------------------------------------------------------------------------
    def _post_drafter(self, payload: dict) -> None:
        req = urllib.request.Request(self.drafter_url + "/generate", data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"}, method="POST")
        with http_open(req, HTTP_TIMEOUT_S) as resp:
            status = getattr(resp, "status", 200)
            body = resp.read()
        if status != 200:
            raise RuntimeError(f"c1b_judge_loop: drafter POST returned {status}: {body[:300]!r}")

    def _wait_drafter_rows(self, drids: Sequence[str], los: Sequence[int]) -> List[dict]:
        if self._dreader is None:
            from corpus1b.gen_sglang import TraceReader         # lazy: only the judge process reads the trace
            self._dreader = TraceReader(self.drafter_trace_dir)
        got: Dict[str, dict] = {}
        t0 = time.time()
        while True:
            self._dreader.poll()
            for d, lo in zip(drids, los):
                if d not in got and lo in self._dreader.pending.get(d, {}):
                    got[d] = self._dreader.take(d)[lo]
            if len(got) == len(drids):
                return [got[d] for d in drids]
            if time.time() - t0 > DRAFT_WAIT_S:
                miss = [d for d in drids if d not in got]
                raise RuntimeError(f"c1b_judge_loop: {len(miss)} drafter rows missing from {self.drafter_trace_dir} "
                                   f"after {DRAFT_WAIT_S:.0f}s: {miss[:8]}")
            time.sleep(DRAFT_POLL_S)

    def _draft(self, reqs: List[JC.DraftReq], rids: List[str], los: List[int], sts: List[_ReqState],
               device) -> List[JC.DraftRes]:
        self.serial += 1
        # A no-op block (a re-presented finished request; sts[key] is None) has an empty fill: it is NOT sent to the
        # drafter and gets a trivial result (its block unchanged), so the drafter never sees an empty-fill request.
        live = [q for q in reqs if int(q.fill.sum().item()) > 0]
        drids, ids, dlos = [], [], []
        for q in live:
            st, lo = sts[q.key], los[q.key]
            d = f"{rids[q.key]}#{st.block_index}#{self.serial}"
            spec = {"fill": torch.nonzero(q.fill).flatten().tolist()}
            if q.steps is not None:
                spec["steps"] = int(q.steps)
            if q.ban is not None:
                ban = [[s, int(t)] for s, t in enumerate(q.ban.tolist()) if int(t) >= 0]
                if ban:
                    spec["ban"] = ban
            if q.guard:
                spec["guard"] = {"n": int(self.cfg.loop_guard), "prompt_length": st.prompt_length,
                                 "prefix_ids": st.seq[:lo]}
            write_json_atomic(os.path.join(self.drafter_side_dir, f"{d}.json"), spec)
            drids.append(d)
            ids.append(st.seq[:lo] + [int(t) for t in q.block_ids.tolist()])
            dlos.append(lo)
        got: Dict[int, JC.DraftRes] = {}
        if live:
            self._post_drafter({"input_ids": ids, "sampling_params": [dict(DRAFTER_SAMPLING) for _ in live],
                                "rid": drids})
            rows = self._wait_drafter_rows(drids, dlos)
            for q, d, row in zip(live, drids, rows):
                fill = torch.nonzero(q.fill).flatten().tolist()
                assert len(row["ids"]) == self.block_size and row["fill"] == fill, \
                    f"{d}: the drafter filled {row['fill']}, asked {fill}"
                conf = torch.tensor(row["conf"], dtype=torch.float32, device=device)
                got[q.key] = JC.DraftRes(tokens=torch.tensor(row["ids"], dtype=torch.long, device=device),
                                         p_commit=torch.where(q.fill, conf, torch.zeros_like(conf)),
                                         n_fwd=int(row["n_fwd"]), guard_slots=int(row.get("guard_slots", 0)),
                                         guard_changed=int(row.get("guard_changed", 0)))
        out = []
        for q in reqs:
            if q.key in got:
                out.append(got[q.key])
            else:                                   # no-op block: return its tokens unchanged, no drafter call
                out.append(JC.DraftRes(tokens=q.block_ids.to(device=device, dtype=torch.long),
                                       p_commit=torch.zeros(self.block_size, dtype=torch.float32, device=device),
                                       n_fwd=0, guard_slots=0, guard_changed=0))
        return out

    # ------------------------------------------------------------------------------------------------------------
    def run(
        self,
        model_runner: ModelRunner,
        forward_batch: ForwardBatch,
    ) -> Tuple[Union[LogitsProcessorOutput, torch.Tensor], List[torch.Tensor], bool]:
        model = model_runner.model
        self._ensure_adapter(model)
        bs, mask_id = self.block_size, self.mask_id
        flat_ids = forward_batch.input_ids
        dev = flat_ids.device

        # prefill-only batch: one forward with the adapter gated off everywhere, stores the KV
        if int(torch.sum(flat_ids == mask_id).item()) == 0:
            with JL.JudgeCtx.flag_scope(torch.zeros(flat_ids.numel(), dtype=torch.bool, device=dev), False):
                out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
            return out.logits_output, [], out.can_run_graph

        B = forward_batch.batch_size
        assert flat_ids.numel() == B * bs, f"decode batch of {B} blocks has {flat_ids.numel()} ids"
        assert getattr(forward_batch, "input_embeds", None) is None, "input_embeds already set on a decode batch"
        rids = batch_rids(forward_batch, B)
        los = block_starts(forward_batch, bs)
        blocks, starts, sts, skip = [], [], [], []
        slot = torch.arange(bs, device=dev)
        for i in range(B):
            blk = flat_ids[i * bs:(i + 1) * bs].long()
            masked = blk == mask_id
            n_masked = int(masked.sum().item())
            start = bs - n_masked
            if not (n_masked > 0 and bool(masked[start:].all())):
                # The block's [MASK] slots are not the trailing answer region: SGLang has re-presented an already-FINISHED
                # request's block under batching (its state was dropped at EOS/length). A normal in-flight block always has
                # the drafter-filled prefix then trailing [MASK], so this only occurs for a spurious re-presentation. Handle
                # it as a NO-OP (no judge pass, no state row, no self.states entry) and return the block's tokens unchanged;
                # SGLang discards them for the finished request. Normal blocks below are processed exactly as before.
                logger.warning(f"c1b_judge_loop: NOOP_SCATTERED rid={rids[i]} lo={int(los[i])} n_masked={n_masked}")
                sts.append(None); starts.append(bs); skip.append(True)
                blocks.append(JC.BlockRun(cur=blk.clone(), valid=torch.zeros(bs, dtype=torch.bool, device=dev), lo=los[i]))
                continue
            sts.append(self._state_for_block(rids[i], los[i], start, blk.tolist()))
            starts.append(start); skip.append(False)
            blocks.append(JC.BlockRun(cur=blk.clone(), valid=slot >= start, lo=los[i]))
        live_modes = {st.mode for st in sts if st is not None}
        assert len(live_modes) <= 1, f"a batch mixes modes {live_modes}"
        record = live_modes == {"record"}
        cfg = dataclasses.replace(self.cfg, record_states=record)
        closing = ([None if st is None else torch.tensor(st.teacher[st.block_index], dtype=torch.long, device=dev)
                    for st in sts] if record else None)

        last = {}

        def forward_fn(ids: torch.Tensor, flag: torch.Tensor, kind: str) -> torch.Tensor:
            forward_batch.input_ids.copy_(ids.reshape(-1))
            fl = flag.reshape(-1)
            embeds, any_ = JL.judge_forward_prep(model, forward_batch.input_ids, fl)
            forward_batch.input_embeds = embeds
            try:
                with JL.JudgeCtx.flag_scope(fl, any_):
                    out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
            finally:
                forward_batch.input_embeds = None
            assert not out.can_run_graph, "c1b_judge_loop: a judge forward ran under a CUDA graph (disable it)"
            last["out"] = out
            return out.logits_output.full_logits.view(B, bs, -1)

        def draft_fn(reqs: List[JC.DraftReq]) -> List[JC.DraftRes]:
            return self._draft(reqs, rids, los, sts, dev)

        JC.run_blocks(blocks, cfg, forward_fn, draft_fn, closing_ids=closing)

        next_token_ids_list, rows = [], []
        for i, (b, st, start) in enumerate(zip(blocks, sts, starts)):
            tout = b.tokens_out
            if st is None:                                     # a no-op re-presented finished request: return tokens, no row
                next_token_ids_list.append(tout[start:])
                continue
            st.seq = st.seq[:los[i]] + [int(t) for t in tout.tolist()]
            next_token_ids_list.append(tout[start:])
            row = {"rid": rids[i], "block": st.block_index, "lo": los[i], "start": start,
                   "summary": JC.block_summary(b)}
            if record:
                row["states"] = states_json(b)
            rows.append(row)
            st.block_index += 1
            new = [int(t) for t in tout[start:].tolist()]
            if len(st.seq) - st.prompt_length >= st.max_new or (st.stop_ids and any(t in st.stop_ids for t in new)):
                del self.states[rids[i]]                       # SGLang finishes this request now
        self.sink.write(rows)
        out = last["out"]
        return out.logits_output, next_token_ids_list, out.can_run_graph


Algorithm = JudgeLoopTraced
