"""The SGLang judge algorithm JudgeLoopTraced (code/corpus1b/sglang_plugin/c1b_judge_loop.py) on CPU.

Fakes: sglang modules (as test_judge_draft_plugin.py); a model runner whose model is test_judge_lora_sgl.py's tiny
SDAR-like module carrying a REAL saved judge adapter (the plugin attaches it through c1b_judge_lora) and whose logits
are seeded rows keyed by (position, block ids, the fake KV before the block) plus a term of the hidden state, so the
flag-gated LoRA and the draft vector change the decisions; a scheduler that calls run() round by round as SGLang does
(a prefill batch, then one decode block per live request until max_new_tokens or EOS); and a drafter behind the
plugin's real HTTP request (only its no-proxy urlopen, `http_open`, is replaced) that reads the drafter side files with the REAL
c1b_judge_draft.read_side / begin_fill and appends DraftTracedV2's trace rows, which the plugin reads with TraceReader.
The yaml and the judge side files are written by judge_round/judge_sglang.py's own writers.

  (1) two requests, two consecutive blocks (two decode run() calls): the drafter prefixes, side files consumed,
      complete trace rows, next_token_ids sliced from start, the state dropped when SGLang finishes the request;
  (2) every forward's flag: open & (input != [MASK]) on judge passes, all False on stock passes, the closing forward and
      the prefill batch; input_embeds = the draft vector exactly at flagged rows, None again after every forward;
  (3) record mode stores and returns the teacher tokens and writes the per-pass states; decode returns its own;
  (4) the glue adds nothing: equal to judge_core.run_blocks driven directly with the same fakes;
  (5) a request without a judge side file raises with its rid; EOS drops the state; the import mechanism.

    python3 -m pytest -q code/corpus1b/tests/test_judge_loop_plugin.py
"""
from __future__ import annotations

import glob
import importlib.util
import json
import os
import sys
import types
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.abspath(os.path.join(HERE, "..", ".."))
PLUGIN = os.path.join(CODE, "corpus1b", "sglang_plugin")
for p in (os.path.join(CODE, "compat"), CODE, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

from draftalign import adapter as A                                                      # noqa: E402
from draftalign.loop_guard import LoopGuard                                              # noqa: E402
from tests.test_da_decode import V, MASK, EOS_IDS, BL, logprob_row                       # noqa: E402
from judge_round import judge_sglang as JS                                               # noqa: E402
import test_judge_lora_sgl as TL                                                         # noqa: E402

ALG = "sglang.srt.dllm.algorithm"
URL = "http://127.0.0.1:30999"
H, NH, HD = TL.H, TL.NH, TL.HD
W_OUT = torch.randn(H, V, generator=torch.Generator().manual_seed(17)) * 0.3
TEMP, HIDDEN_TERM = 1.0, 1.5             # the hidden term is bounded (tanh): the LoRA moves margins, never swamps
AGREE = ((0.45, 9.0), (0.75, 5.0))      # P(u < .45): +9 at the slot's token (commits); P(.45 <= u < .75): +5 (tentative)


# --------------------------------------------------------------------------
# fake sglang (as test_judge_draft_plugin.py) + the plugins
# --------------------------------------------------------------------------
class DllmAlgorithm:
    def __init__(self, config):
        self.block_size = config.block_size
        self.mask_id = config.mask_id


def _leaf(name, **attrs):
    m = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(m, k, v)
    return m


def load(mp, name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    mp.setitem(sys.modules, name, mod)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def sgl():
    """Fake sglang modules; the plugins load as installed copies would, their helpers through C1B_CODE_DIR."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("C1B_CODE_DIR", CODE)
        for name in ("sglang", "sglang.srt", "sglang.srt.dllm", ALG, "sglang.srt.layers", "sglang.srt.model_executor"):
            pkg = types.ModuleType(name)
            pkg.__path__ = []
            mp.setitem(sys.modules, name, pkg)
        leaves = {
            f"{ALG}.base": {"DllmAlgorithm": DllmAlgorithm},
            f"{ALG}.low_confidence": {"LowConfidence": type("LowConfidence", (DllmAlgorithm,), {})},
            "sglang.srt.dllm.config": {"DllmConfig": type("DllmConfig", (), {})},
            "sglang.srt.layers.logits_processor": {"LogitsProcessorOutput": type("LogitsProcessorOutput", (), {})},
            "sglang.srt.model_executor.forward_batch_info": {"ForwardBatch": type("ForwardBatch", (), {})},
            "sglang.srt.model_executor.model_runner": {"ModelRunner": type("ModelRunner", (), {})},
        }
        for name, attrs in leaves.items():
            mp.setitem(sys.modules, name, _leaf(name, **attrs))
        load(mp, f"{ALG}.c1b_traced", os.path.join(PLUGIN, "c1b_traced.py"))
        draft = load(mp, f"{ALG}.c1b_judge_draft", os.path.join(PLUGIN, "c1b_judge_draft.py"))
        loop = load(mp, f"{ALG}.c1b_judge_loop", os.path.join(PLUGIN, "c1b_judge_loop.py"))
        yield SimpleNamespace(draft=draft, loop=loop, JC=loop.JC, JL=loop.JL)


# --------------------------------------------------------------------------
# the fake judge
# --------------------------------------------------------------------------
def build_judge(adapter_root, seed: int = 0):
    """test_judge_lora_sgl's SGLang-like fake with a V-token embedding; the judge adapter saved to adapter_root."""
    hf = TL.build_hf(seed)
    adir = A.save_adapter(hf, str(adapter_root))
    m = TL.build_sgl(hf)
    torch.manual_seed(seed + 100)
    m.model.embed_tokens = nn.Embedding(V, H)
    m.requires_grad_(False)
    return m, adir


class FakeJudge:
    """Logits of a batch of blocks: per slot a seeded row keyed by (position, the block's ids, the fake KV before the
    block) that often agrees with the slot's token (confidently or not), + HIDDEN_TERM * tanh(hidden @ W_OUT / 10), hidden =
    the model's (patched) qkv/o projections over the input embedding.  Every forward writes its ids into the fake KV."""

    def __init__(self, model, eos_at=None):
        self.model, self.kv, self.eos_at = model, {}, eos_at or {}

    @torch.no_grad()
    def hidden(self, ids, embeds):
        h = self.model.model.embed_tokens(ids) if embeds is None else embeds
        for layer in self.model.model.layers:
            qkv, _ = layer.self_attn.qkv_proj(h)
            o, _ = layer.self_attn.o_proj(qkv[:, :NH * HD])
            h = h + o
        return h

    def write(self, key, lo, ids):
        kv = self.kv.setdefault(key, [])
        assert len(kv) >= lo, f"{key}: KV holds {len(kv)} tokens, a block at {lo}"
        kv[lo:lo + len(ids)] = list(ids)

    @torch.no_grad()
    def blocks(self, keys, los, ids_flat, embeds):
        h = self.hidden(ids_flat.long(), embeds)
        ids = ids_flat.tolist()
        out = []
        for i, (key, lo) in enumerate(zip(keys, los)):
            blk = ids[i * BL:(i + 1) * BL]
            ctx = hash(tuple(self.kv.get(key, [])[:lo]))
            rows = []
            for s, tok in enumerate(blk):
                g = torch.Generator().manual_seed(hash((3, lo + s, tuple(blk), ctx)) & 0x7FFFFFFF)
                row = torch.randn(V, generator=g) * TEMP
                u = float(torch.rand(1, generator=g))
                bonus = next((b for q, b in AGREE if u < q), 0.0)
                if tok != MASK:
                    row[tok] += bonus
                row[EOS_IDS] = -30.0
                if lo + s in self.eos_at.get(key, ()):
                    row[EOS_IDS[0]] = 30.0
                row[MASK] = -30.0
                rows.append(row)
            out.append(torch.stack(rows) + HIDDEN_TERM * torch.tanh(h[i * BL:(i + 1) * BL] @ W_OUT / 10.0))
            self.write(key, lo, blk)
        return torch.cat(out)


class FakeRunner:
    """model_runner: .model + .forward(fb) -> logits_output.full_logits; logs and checks every forward's flag."""

    def __init__(self, judge: FakeJudge, JL):
        self.judge, self.model, self.JL = judge, judge.model, JL
        self.log = []

    def forward(self, fb, pp_proxy_tensors=None):
        ctx = self.JL.JudgeCtx
        assert ctx.flag is not None, "a forward ran with flag None: the LoRA would apply everywhere"
        ids = fb.input_ids.long()
        flag = ctx.flag.clone()
        assert flag.dtype == torch.bool and tuple(flag.shape) == (ids.numel(),)
        assert ctx.any == bool(flag.any())
        embeds = fb.input_embeds
        if embeds is not None:
            assert tuple(embeds.shape) == (ids.numel(), H)
            with torch.no_grad():
                plain = self.model.model.embed_tokens(ids)
            assert torch.equal((embeds != plain).any(-1), flag), "the draft vector is not exactly at the flagged rows"
        else:
            assert not ctx.any, "flagged slots without the draft embedding"
        self.log.append({"kind": fb.kind, "flag": flag, "embeds": embeds is not None, "ids": ids.clone(),
                         "rids": list(fb.rids)})
        if fb.kind == "prefill":
            off = 0
            for rid, pre, n in zip(fb.rids, fb.extend_prefix_lens_cpu, fb.extend_seq_lens_cpu):
                self.judge.write(rid, pre, ids[off:off + n].tolist())
                off += n
            logits = torch.zeros(ids.numel(), V)
        else:
            logits = self.judge.blocks(fb.rids, fb.extend_prefix_lens_cpu, ids, embeds)
        return SimpleNamespace(logits_output=SimpleNamespace(full_logits=logits), can_run_graph=False)


# --------------------------------------------------------------------------
# the fake drafter (behind the plugin's HTTP POST)
# --------------------------------------------------------------------------
def drafter_logits(prefix, cur, lo):
    """A cycling drafter whose phase depends on the prefix: c at a [MASK] slot (p .6), runner-up a (.3), else 7 + pos % 11."""
    shift = hash(tuple(int(t) for t in prefix)) % 3
    rows = []
    for s, tok in enumerate(cur.tolist()):
        pos = lo + s
        if tok != MASK:
            row = torch.full((V,), -8.0)
            row[tok] = 8.0
        else:
            c, a, o = 2 + (pos + shift) % 5, 2 + (pos + shift + 2) % 5, 7 + pos % 11
            p = 0.6 - 0.0005 * s
            row = logprob_row({c: p, a: 0.3, o: 1.0 - p - 0.3})
        rows.append(row)
    return torch.stack(rows)


class FakeResp:
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self):
        return b"[]"


class FakeDrafter:
    """The plugin's http_open (urlopen) for the drafter server: runs DraftTracedV2's side-file parsing and fill per request."""

    def __init__(self, draft_mod, side_dir, trace_dir, fb_holder):
        self.D, self.side_dir, self.trace_dir, self.fb_holder = draft_mod, side_dir, trace_dir, fb_holder
        self.calls = []

    def urlopen(self, req, timeout=None):
        assert req.full_url == URL + "/generate" and req.get_method() == "POST"
        assert req.get_header("Content-type") == "application/json"
        fb = self.fb_holder.get("fb")
        assert fb is None or fb.input_embeds is None, "input_embeds left set between forwards"
        body = json.loads(req.data)
        assert len(body["input_ids"]) == len(body["rid"]) == len(body["sampling_params"])
        rows = []
        for ids, rid, sp in zip(body["input_ids"], body["rid"], body["sampling_params"]):
            assert sp == {"temperature": 0.0, "max_new_tokens": 1, "ignore_eos": True}
            prefix, block = ids[:-BL], torch.tensor(ids[-BL:], dtype=torch.long)
            lo = len(prefix)
            spec = json.load(open(os.path.join(self.side_dir, f"{rid}.json")))
            df = self.D.begin_fill(self.D.read_side(self.side_dir, rid), block, lo, MASK, 8)
            while not df.done:
                df.step(drafter_logits(prefix, df.cur, lo))
            conf = torch.where(df.fill, df.p_commit, torch.full_like(df.p_commit, -1.0))
            rows.append({"rid": rid, "lo": lo, "start": BL - int((block == MASK).sum()), "ids": df.cur.tolist(),
                         "conf": [float(c) for c in conf.tolist()], "fill": torch.nonzero(df.fill).flatten().tolist(),
                         "n_fwd": df.n_fwd, "raw_mask_argmax": int(df.raw_mask), "guard_slots": df.g_slots,
                         "guard_changed": df.g_changed})
            self.calls.append({"rid": rid, "prefix": prefix, "block": ids[-BL:], "spec": spec})
        with open(os.path.join(self.trace_dir, "trace.fake.jsonl"), "a") as fh:
            fh.write("".join(json.dumps(r) + "\n" for r in rows))
        return FakeResp()


# --------------------------------------------------------------------------
# harness: config, requests, the fake scheduler, the direct drive
# --------------------------------------------------------------------------
CFGS = {
    "rd1_ban_guard": ["--redraft-every", "1", "--redraft-ban", "--loop-guard", "4"],
    "cap3_rd2": ["--redraft-every", "2", "--loop-guard", "0", "--max-passes", "3"],
}


def algorithm_config(tmp_path, name: str) -> dict:
    """judge_sglang's yaml for these flags, read back the way SGLang reads it (yaml.safe_load)."""
    args = JS.build_parser().parse_args(["--mode", "decode", "--items", "x", "--gpu", "0"] + CFGS[name])
    path = JS.write_yaml(str(tmp_path / f"judge_{name}.yaml"), JS.judge_yaml_config(args))
    return yaml.safe_load(open(path))


def prompt(L: int, seed: int):
    return [(i * 7 + seed) % 50 + 1 for i in range(L)]          # never MASK / EOS


def teacher_blocks(p, nb: int, seed: int):
    g = torch.Generator().manual_seed(seed)
    lo0 = (len(p) // BL) * BL
    t = torch.randint(2, 50, (nb, BL), generator=g)
    t[0, :len(p) - lo0] = torch.tensor(p[lo0:])
    return t


def make_req(rid, L, seed, nb, mode, eos_at=None):
    p = prompt(L, seed)
    tail = L - (L // BL) * BL
    r = {"rid": rid, "prompt": p, "max_new": BL * nb - tail, "mode": mode, "eos_at": set(eos_at or ())}
    if mode == "record":
        item = {"pid": rid, "prompt_ids": torch.tensor(p), "prompt_length": L,
                "tokens": teacher_blocks(p, nb, seed + 500), "block_start": [(L // BL) * BL + BL * b for b in range(nb)]}
        q = JS.record_request(item)
        assert q["max_new"] == r["max_new"]
        r.update(spec=q["spec"], teacher=q["teacher"], ignore_eos=True)
    else:
        r.update(spec=JS.decode_side_spec(p, r["max_new"], stop_ids=EOS_IDS), ignore_eos=False)
    return r


def setup(sgl, tmp_path, monkeypatch, cfg_name, reqs):
    dirs = {k: tmp_path / k for k in ("judge_side", "judge_trace", "drafter_side", "drafter_trace")}
    for d in dirs.values():
        d.mkdir()
    model, adir = build_judge(tmp_path / "adapter")
    env = {"C1B_JUDGE_ADAPTER": adir, "C1B_JUDGE_SIDE_DIR": str(dirs["judge_side"]),
           "C1B_JUDGE_TRACE_DIR": str(dirs["judge_trace"]), "C1B_DRAFTER_URL": URL,
           "C1B_DRAFTER_SIDE_DIR": str(dirs["drafter_side"]), "C1B_DRAFTER_TRACE_DIR": str(dirs["drafter_trace"])}
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    ac = algorithm_config(tmp_path, cfg_name)
    alg = sgl.loop.JudgeLoopTraced(SimpleNamespace(block_size=BL, mask_id=MASK, algorithm_config=ac))
    holder = {}
    drafter = FakeDrafter(sgl.draft, str(dirs["drafter_side"]), str(dirs["drafter_trace"]), holder)
    monkeypatch.setattr(sgl.loop, "http_open", drafter.urlopen)
    for r in reqs:
        JS.write_side(str(dirs["judge_side"]), r["rid"], r["spec"])
    runner = FakeRunner(FakeJudge(model, {r["rid"]: r["eos_at"] for r in reqs}), sgl.JL)
    return SimpleNamespace(alg=alg, runner=runner, drafter=drafter, dirs=dirs, holder=holder, adir=adir, ac=ac)


def serve(env, reqs, dtype=torch.long):
    """SGLang's dLLM scheduler, as far as run() sees it: one prefill batch of the prompts' whole blocks, then a decode
    round per block over the live requests; a request finishes at max_new_tokens or (unless ignore_eos) on EOS."""
    alg, runner = env.alg, env.runner
    pre = [r for r in reqs if len(r["prompt"]) >= BL]
    if pre:
        chunks = [r["prompt"][:(len(r["prompt"]) // BL) * BL] for r in pre]
        fb = SimpleNamespace(input_ids=torch.tensor(sum(chunks, []), dtype=dtype), rids=[r["rid"] for r in pre],
                             extend_prefix_lens_cpu=[0] * len(pre), extend_seq_lens_cpu=[len(c) for c in chunks],
                             batch_size=len(pre), input_embeds=None, kind="prefill")
        _, nxt, graph = alg.run(runner, fb)
        assert nxt == [] and graph is False and fb.input_embeds is None
    for r in reqs:
        r["out"] = []
    live, rounds = list(reqs), []
    while live:
        blocks, los = [], []
        for r in live:
            L = len(r["prompt"])
            lo0 = (L // BL) * BL
            if not r["out"]:
                blocks.append(r["prompt"][lo0:] + [MASK] * (BL - (L - lo0)))
                los.append(lo0)
            else:
                los.append(L + len(r["out"]))
                blocks.append([MASK] * BL)
        fb = SimpleNamespace(input_ids=torch.tensor(sum(blocks, []), dtype=dtype), rids=[r["rid"] for r in live],
                             extend_prefix_lens_cpu=los, extend_seq_lens_cpu=[BL] * len(live), batch_size=len(live),
                             input_embeds=None, kind="decode")
        env.holder["fb"] = fb
        f0 = len(runner.log)
        _, nxt, graph = alg.run(runner, fb)
        assert graph is False and fb.input_embeds is None and len(nxt) == len(live)
        assert fb.input_ids.dtype == dtype
        rounds.append({"rids": [r["rid"] for r in live], "los": los, "next": [t.tolist() for t in nxt],
                       "fwd": (f0, len(runner.log))})
        still = []
        for r, t in zip(live, nxt):
            t = [int(x) for x in t.tolist()]
            r["out"] += t
            stop = (not r["ignore_eos"]) and any(x in EOS_IDS for x in t)
            if not stop and len(r["out"]) < r["max_new"]:
                still.append(r)
        live = still
    return rounds


def trace_rows(env):
    rows = {}
    for p in glob.glob(os.path.join(str(env.dirs["judge_trace"]), "trace.*.jsonl")):
        for line in open(p):
            r = json.loads(line)
            assert (r["rid"], r["block"]) not in rows
            rows[(r["rid"], r["block"])] = r
    return rows


def direct_generate(sgl, model, reqs, cfg, flag_blind=False):
    """judge_core.run_blocks driven directly (no plugin, no SGLang glue): the same fake judge with its own KV, the
    drafter fill with draft_core.fill_block + LoopGuard on the same drafter logits, the same scheduler rule.
    flag_blind: the judge never sees the flag (the base model on every pass), to show the fakes depend on it."""
    JC, JL = sgl.JC, sgl.JL
    from corpus1b.sglang_plugin import draft_core as DC
    judge = FakeJudge(model, {r["rid"]: r["eos_at"] for r in reqs})
    x, out = {}, {r["rid"]: [] for r in reqs}
    for r in reqs:
        lo0 = (len(r["prompt"]) // BL) * BL
        judge.write(r["rid"], 0, r["prompt"][:lo0])
        x[r["rid"]] = list(r["prompt"])
    live = list(reqs)
    while live:
        runs, los, starts = [], [], []
        for r in live:
            L = len(r["prompt"])
            lo0 = (L // BL) * BL
            if not out[r["rid"]]:
                lo, start, blk = lo0, L - lo0, r["prompt"][lo0:] + [MASK] * (BL - (L - lo0))
            else:
                lo, start, blk = len(x[r["rid"]]), 0, [MASK] * BL
            runs.append(JC.BlockRun(cur=torch.tensor(blk), valid=torch.arange(BL) >= start, lo=lo))
            los.append(lo)
            starts.append(start)
        keys = [r["rid"] for r in live]

        def forward_fn(ids, flag, kind):
            fl, flat = flag.reshape(-1), ids.reshape(-1)
            if flag_blind:
                fl = torch.zeros_like(fl)
            any_ = bool(fl.any())
            with JL.JudgeCtx.flag_scope(fl, any_):
                emb = JL.embed_with_draft_flag(model, flat, fl) if any_ else None
                lg = judge.blocks(keys, los, flat, emb)
            return lg.view(len(keys), BL, -1)

        def draft_fn(qs):
            res = []
            for q in qs:
                r, lo = live[q.key], los[q.key]
                prefix = x[r["rid"]][:lo]
                guard = None
                if q.guard:
                    guard = LoopGuard(cfg.loop_guard, MASK)
                    guard.start_item(len(r["prompt"]))
                    guard.begin_block(prefix, lo)
                d = DC.fill_block(q.block_ids, q.fill, q.steps if q.steps is not None else cfg.draft_steps,
                                  lambda cur: drafter_logits(prefix, cur, lo), MASK, ban=q.ban, guard=guard)
                res.append(JC.DraftRes(tokens=d["tokens"], p_commit=d["p_commit"], n_fwd=d["n_fwd"],
                                       guard_slots=d["guard_slots"], guard_changed=d["guard_changed"]))
            return res

        closing = ([torch.tensor(r["teacher"][len(out[r["rid"]])]) for r in live]
                   if live[0]["mode"] == "record" else None)
        JC.run_blocks(runs, dataclass_replace(cfg, record_states=live[0]["mode"] == "record"), forward_fn, draft_fn,
                      closing_ids=closing)
        still = []
        for r, b, start in zip(live, runs, starts):
            x[r["rid"]] = x[r["rid"]][:b.lo] + b.tokens_out.tolist()
            out[r["rid"]].append(b)
            new = b.tokens_out[start:].tolist()
            stop = (not r["ignore_eos"]) and any(t in EOS_IDS for t in new)
            if not stop and len(x[r["rid"]]) - len(r["prompt"]) < r["max_new"]:
                still.append(r)
        live = still
    return out


def dataclass_replace(cfg, **kw):
    import dataclasses
    return dataclasses.replace(cfg, **kw)


def two_requests(mode):
    return [make_req("itemA", 37, 1, 2, mode), make_req("itemB", 70, 2, 2, mode)]     # tails 5 and 6, 2 blocks each


# --------------------------------------------------------------------------
# tests
# --------------------------------------------------------------------------
def test_import_mechanism_and_config(sgl, tmp_path):
    assert sgl.JC.__name__ == "corpus1b.sglang_plugin.judge_core"
    assert sgl.JL.__name__ == "corpus1b.sglang_plugin.c1b_judge_lora"
    assert sgl.loop.Algorithm is sgl.loop.JudgeLoopTraced
    ac = algorithm_config(tmp_path, "rd1_ban_guard")
    cfg = sgl.loop.judge_config(ac, MASK)
    assert (cfg.redraft_every, cfg.redraft_ban, cfg.loop_guard, cfg.max_passes, cfg.tau_override, cfg.draft_steps,
            cfg.threshold, cfg.tau_commit) == (1, True, 4, 32, None, 8, 0.95, 0.95)
    with pytest.raises(AssertionError, match="unknown"):
        sgl.loop.judge_config({**ac, "redraft_evry": 1}, MASK)


def test_missing_env_and_missing_side_file(sgl, tmp_path, monkeypatch):
    for k in sgl.loop.REQUIRED_ENV:
        monkeypatch.delenv(k, raising=False)
    with pytest.raises(RuntimeError, match="C1B_JUDGE_ADAPTER"):
        sgl.loop.JudgeLoopTraced(SimpleNamespace(block_size=BL, mask_id=MASK, algorithm_config={}))
    env = setup(sgl, tmp_path, monkeypatch, "rd1_ban_guard", [])
    ghost = make_req("ghost-7", 37, 1, 2, "decode")                     # no side file written
    with pytest.raises(RuntimeError, match="ghost-7"):
        serve(env, [ghost])


def test_decode_two_requests_two_blocks(sgl, tmp_path, monkeypatch):
    reqs = two_requests("decode")
    env = setup(sgl, tmp_path, monkeypatch, "rd1_ban_guard", reqs)
    rounds = serve(env, reqs)
    assert len(rounds) == 2 and all(r["rids"] == ["itemA", "itemB"] for r in rounds)
    assert getattr(env.runner.model, "_judge_adapter")["dir"] == os.path.abspath(env.adir)
    rows = trace_rows(env)
    assert sorted(rows) == [("itemA", 0), ("itemA", 1), ("itemB", 0), ("itemB", 1)]
    for k, r in enumerate(reqs):
        L = len(r["prompt"])
        lo0, tail = (L // BL) * BL, L - (L // BL) * BL
        r0, r1 = rows[(r["rid"], 0)], rows[(r["rid"], 1)]
        assert (r0["lo"], r0["start"], r1["lo"], r1["start"]) == (lo0, tail, lo0 + BL, 0)
        for row in (r0, r1):
            s = row["summary"]
            for key in JS.COUNTERS + ("passes", "tokens", "tokens_out", "draft_tokens", "p_draft", "lo"):
                assert key in s, key
            assert s["tokens_out"] == s["tokens"] and s["n_judge"] == len(s["passes"]) >= 1
            assert "states" not in row
        # next_token_ids = the block from start; SGLang's output_ids are the blocks' generated slots
        assert rounds[0]["next"][k] == r0["summary"]["tokens_out"][tail:]
        assert rounds[1]["next"][k] == r1["summary"]["tokens_out"]
        assert r["out"] == JS.generated([r0, r1])
        # the drafter: block 0 sees the prompt; block 1 sees prompt + block 0's stored tokens
        calls = {b: [c for c in env.drafter.calls if c["rid"].rsplit("#", 2)[:2] == [r["rid"], str(b)]] for b in (0, 1)}
        assert calls[0] and calls[1]
        for c in calls[0]:
            assert c["prefix"] == r["prompt"][:lo0] and c["block"][:tail] == r["prompt"][lo0:]
        for c in calls[1]:
            assert c["prefix"] == r["prompt"] + r0["summary"]["tokens_out"][tail:]
            assert c["prefix"] == r["prompt"][:lo0] + r0["summary"]["tokens_out"]
        for b in (0, 1):
            first, *redrafts = calls[b]
            assert "steps" not in first["spec"] and "ban" not in first["spec"]
            assert first["spec"]["fill"] == list(range(tail if b == 0 else 0, BL))
            for c in calls[b]:
                g = c["spec"]["guard"]
                assert g == {"n": 4, "prompt_length": L, "prefix_ids": c["prefix"]}
            for c in redrafts:
                assert c["spec"]["steps"] >= 1 and "ban" in c["spec"]
                assert all(c["block"][s] == MASK for s in c["spec"]["fill"])
    serials = [c["rid"].rsplit("#", 1)[1] for c in env.drafter.calls]
    assert len({c["rid"] for c in env.drafter.calls}) == len(env.drafter.calls) and len(set(serials)) > 2
    assert os.listdir(env.dirs["judge_side"]) == [] and os.listdir(env.dirs["drafter_side"]) == []
    assert env.alg.states == {}                                       # both finished at max_new_tokens
    tot = lambda k: sum(r["summary"][k] for r in rows.values())     # noqa: E731
    assert tot("n_redraft_rounds") > 0 and tot("n_loopguard_slots") > 0 and tot("n_tentative") > 0


def _check_forwards(env, rounds, rows):
    """(2): the flag and the embeddings of every forward against the recorded states."""
    log = env.runner.log
    assert log[0]["kind"] == "prefill" and not log[0]["flag"].any() and not log[0]["embeds"]
    for rd in rounds:
        fw = log[rd["fwd"][0]:rd["fwd"][1]]
        blocks = [next(row for (r, _), row in rows.items() if r == rid and row["lo"] == lo)
                  for rid, lo in zip(rd["rids"], rd["los"])]
        nj = [b["summary"]["n_judge"] for b in blocks]
        ns = [b["summary"]["n_stock"] for b in blocks]
        assert len(fw) == max(nj) + max(ns) + 1
        assert not fw[-1]["flag"].any() and not fw[-1]["embeds"]                     # the closing forward
        for i, b in enumerate(blocks):
            sl = slice(i * BL, (i + 1) * BL)
            assert fw[-1]["ids"][sl].tolist() == b["summary"]["tokens_out"]
            for t, f in enumerate(fw[:-1]):
                if t < nj[i]:
                    st = b["states"][t]
                    want = [o and x != MASK for o, x in zip(st["open"], st["input"])]
                    assert st["flag"] == want and f["flag"][sl].tolist() == want
                    assert f["ids"][sl].tolist() == st["input"]
                else:
                    assert not f["flag"][sl].any()                                       # done, or stock passes
        assert any(f["embeds"] for f in fw)


@pytest.mark.parametrize("cfg_name", sorted(CFGS))
def test_flags_and_embeddings_of_every_forward(sgl, tmp_path, monkeypatch, cfg_name):
    reqs = two_requests("record")
    env = setup(sgl, tmp_path, monkeypatch, cfg_name, reqs)
    rounds = serve(env, reqs, dtype=torch.int32)
    rows = trace_rows(env)
    _check_forwards(env, rounds, rows)
    if cfg_name.startswith("cap3"):
        assert any(r["summary"]["full_fallback"] > 0 and r["summary"]["n_stock"] > 0 for r in rows.values())


def test_record_mode_stores_the_teacher_and_writes_states(sgl, tmp_path, monkeypatch):
    reqs = two_requests("record")
    env = setup(sgl, tmp_path, monkeypatch, "rd1_ban_guard", reqs)
    rounds = serve(env, reqs)
    rows = trace_rows(env)
    for k, r in enumerate(reqs):
        L = len(r["prompt"])
        lo0, tail = (L // BL) * BL, L - (L // BL) * BL
        for b in (0, 1):
            row = rows[(r["rid"], b)]
            s = row["summary"]
            start = tail if b == 0 else 0
            assert s["tokens_out"] == r["teacher"][b] and s["tokens"] != r["teacher"][b]
            assert rounds[b]["next"][k] == r["teacher"][b][start:]
            assert len(row["states"]) == s["n_judge"] and [st["pass"] for st in row["states"]] == list(range(s["n_judge"]))
            for st in row["states"]:
                assert [c or t or m for c, t, m in zip(st["commit"], st["tentative"], st["remask"])] == st["open"]
                assert all((p is None) != o for p, o in zip(st["p"], st["open"]))
            assert any(st["redraft"] is not None for st in row["states"]) or s["n_redraft_rounds"] == 0
        # the KV holds the teacher: the stored block 0 and the drafter prefix of block 1
        assert env.runner.judge.kv[r["rid"]][lo0:lo0 + BL] == r["teacher"][0]
        for c in env.drafter.calls:
            if c["rid"].startswith(r["rid"] + "#1#"):
                assert c["prefix"] == r["prompt"][:lo0] + r["teacher"][0]
    assert env.alg.states == {}


@pytest.mark.parametrize("mode", ["decode", "record"])
@pytest.mark.parametrize("cfg_name", sorted(CFGS))
def test_plugin_equals_run_blocks_driven_directly(sgl, tmp_path, monkeypatch, cfg_name, mode):
    reqs = two_requests(mode)
    env = setup(sgl, tmp_path, monkeypatch, cfg_name, reqs)
    serve(env, reqs)
    rows = trace_rows(env)
    model_d, adir = build_judge(tmp_path / "adapter_direct")
    sgl.JL.attach_judge_adapter(model_d, adir)
    cfg = sgl.loop.judge_config(env.ac, MASK)
    direct = direct_generate(sgl, model_d, two_requests(mode), cfg)
    assert sorted(rows) == sorted((rid, k) for rid, bl in direct.items() for k in range(len(bl)))
    for rid, bl in direct.items():
        for k, b in enumerate(bl):
            row = rows[(rid, k)]
            assert row["summary"] == json.loads(json.dumps(sgl.JC.block_summary(b)))
            if mode == "record":
                assert row["states"] == json.loads(json.dumps(sgl.loop.states_json(b)))
    assert sum(b.n_judge for bl in direct.values() for b in bl) > 2
    # the comparison is sensitive to the flag: the same drive with the adapter gated off decides differently
    blind = direct_generate(sgl, model_d, two_requests(mode), cfg, flag_blind=True)
    assert any(JSON(sgl.JC.block_summary(b)) != JSON(sgl.JC.block_summary(c))
               for rid in direct for b, c in zip(direct[rid], blind[rid]))


def JSON(x):
    return json.loads(json.dumps(x))


def test_eos_finishes_a_decode_request_and_drops_its_state(sgl, tmp_path, monkeypatch):
    c = make_req("itemC", 37, 3, 3, "decode", eos_at=(37 + 3,))        # EOS at the 4th generated slot
    d = make_req("itemD", 37, 4, 2, "decode")
    env = setup(sgl, tmp_path, monkeypatch, "rd1_ban_guard", [c, d])
    rounds = serve(env, [c, d])
    assert [r["rids"] for r in rounds] == [["itemC", "itemD"], ["itemD"]]
    assert EOS_IDS[0] in rounds[0]["next"][0] and env.alg.states == {}
    rows = trace_rows(env)
    assert sorted(rows) == [("itemC", 0), ("itemD", 0), ("itemD", 1)]
    gen = JS.generated([rows[("itemC", 0)]])
    rec = JS.decode_record({"rid": "itemC", "idx": 3, "split": "t"}, c["prompt"], [rows[("itemC", 0)]],
                           lambda ids: " ".join(map(str, ids)), env.ac, 0.0, output_ids=c["out"], eos_ids=EOS_IDS)
    assert rec["gen_ids"] == JS.trim_eos(gen, EOS_IDS) and len(rec["gen_ids"]) <= 4 and rec["stopped_eos"]
    assert rec["n_blocks"] == 1 and rec["passes_per_block"] == [rows[("itemC", 0)]["summary"]["n_judge"]]
