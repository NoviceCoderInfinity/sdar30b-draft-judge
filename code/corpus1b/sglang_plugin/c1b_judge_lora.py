"""The judge's FLAG-GATED LoRA adapter + draft-flag embedding, applied to SGLang's `SDARMoeForCausalLM`.

Ground truth is the HF harness, code/draftalign/adapter.py: `LoRALinear` on self_attn.{q,k,v,o}_proj
(y = W x + g_i * (alpha/r) B(A x), fp32 A/B cast to x's dtype inside the forward, the delta skipped
entirely at an all-False flag) and `DraftEmbed` (an fp32 vector added to the input embedding of flagged
slots). This file re-expresses both over SGLang 0.5.10.post1's model object (models/sdar_moe.py):

  * q/k/v are ONE fused `QKVParallelLinear` (`qkv_proj`, out = [q | k | v]); o is a `RowParallelLinear`.
    Both return (out, bias). We keep each module and replace only its INSTANCE `forward`: the original
    runs unchanged, then the per-projection deltas (computed exactly as LoRALinear does) are concatenated
    [dq | dk | dv] and added. Every other attribute (weights, weight loaders, quant method) is untouched.
  * activations are 2-D [num_tokens, hidden] (all requests flattened), so the flag is a 1-D bool
    [num_tokens] held in the module-level `JudgeCtx` (the twin of adapter.CTX, which is [B, S]).
  * the draft vector is added to `model.model.embed_tokens(input_ids)`; the result goes in as
    `input_embeds`. With an all-False flag no embeds are built and the model takes the plain input_ids path.

`JudgeCtx.flag is None` means an UNGATED delta on every token (as adapter.CTX). The harness arms an all-False
flag on every non-judge forward (da_decode._flag_scope); do the same here for prefill, stock passes, the
block-storing forward and SGLang's own warmup, or the adapter silently applies everywhere.

Requires TP = 1 and unquantized attention projections (asserted). The projections branch in Python on
`JudgeCtx`, so a CUDA-graph or torch.compile capture would freeze whichever branch it saw: run the judge's
forwards eagerly (or not under a captured graph).

Usage (inside a dLLM algorithm, once after the model loads):
    info = attach_judge_adapter(model_runner.model, "<adapter dir>")
    embeds, any_ = judge_forward_prep(model, input_ids, flag)        # flag: bool [num_tokens]
    with JudgeCtx.flag_scope(flag, any_):
        out = model(input_ids, positions, forward_batch, input_embeds=embeds)

Tests (CPU, no SGLang needed): python3 -m pytest -q code/corpus1b/tests/test_judge_lora_sgl.py
"""
from __future__ import annotations

import contextlib
import json
import os
import types
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

ADAPTER_JSON = "adapter_config.json"                  # == draftalign.adapter.ADAPTER_JSON
ADAPTER_SAFETENSORS = "adapter.safetensors"           # == draftalign.adapter.ADAPTER_SAFETENSORS
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj"]     # == list(draftalign.adapter.ATTN_TARGETS)
DRAFT_EMBED_KEY = "da_draft_embed.weight"             # == draftalign.adapter.DRAFT_EMBED_KEY
QKV = ("q", "k", "v")                                 # order of the fused qkv_proj output (sdar_moe.py load_weights)


# ---------------------------------------------------------------------------
# the flag context (flattened layout: [num_tokens])
# ---------------------------------------------------------------------------
class _JudgeCtx:
    """`flag` bool [num_tokens] (or None = ungated delta everywhere); `any` caches `bool(flag.any())`
    so the projections never sync the device."""

    __slots__ = ("flag", "any")

    def __init__(self):
        self.flag: Optional[torch.Tensor] = None
        self.any: bool = False

    def set_flag(self, flag: Optional[torch.Tensor], any_: Optional[bool] = None) -> None:
        """Arm the gate. `any_` (when known) MUST equal `bool(flag.any())`: a wrong value changes the arithmetic."""
        if flag is None:
            self.flag, self.any = None, False
            return
        assert flag.dtype == torch.bool, flag.dtype
        assert flag.dim() == 1, f"the flag must be bool [num_tokens], got {tuple(flag.shape)}"
        self.flag = flag
        self.any = bool(flag.any()) if any_ is None else bool(any_)

    def clear_flag(self) -> None:
        self.set_flag(None)

    @contextlib.contextmanager
    def flag_scope(self, flag: Optional[torch.Tensor], any_: Optional[bool] = None):
        """Arm `flag` for the body; restore the previous context afterwards (also on exceptions)."""
        prev = self.flag, self.any
        self.set_flag(flag, any_)
        try:
            yield
        finally:
            self.flag, self.any = prev


JudgeCtx = _JudgeCtx()
set_flag, clear_flag, flag_scope = JudgeCtx.set_flag, JudgeCtx.clear_flag, JudgeCtx.flag_scope


# ---------------------------------------------------------------------------
# the patched forwards
# ---------------------------------------------------------------------------
def _lora_delta(x: torch.Tensor, a: torch.Tensor, b: torch.Tensor, scaling: float) -> torch.Tensor:
    """LoRALinear's delta, same ops in the same order: fp32 A/B cast to x's dtype, two matmuls, * alpha/r."""
    assert a.dtype == torch.float32 and b.dtype == torch.float32, "LoRA A/B must stay fp32 (cast in the forward)"
    return F.linear(F.linear(x, a.to(x.dtype)), b.to(x.dtype)) * scaling


def _gated(delta: torch.Tensor, x: torch.Tensor, flag: Optional[torch.Tensor]) -> torch.Tensor:
    if flag is None:
        return delta                                  # pre-flag behaviour: ungated everywhere
    assert x.dim() == 2 and flag.shape[0] == x.shape[0], \
        f"flag {tuple(flag.shape)} does not match the activations {tuple(x.shape)}"
    return delta * flag.to(delta.dtype)[:, None]      # unflagged tokens get NO delta


def _qkv_forward(self, x, *args, **kwargs):
    out, bias = self._judge_orig_forward(x, *args, **kwargs)
    flag = JudgeCtx.flag
    if flag is not None and not JudgeCtx.any:
        return out, bias                              # every token gated off -> the frozen base, bit-identical
    s = self.judge_lora_scaling
    delta = torch.cat([_lora_delta(x, getattr(self, f"lora_A_{p}"), getattr(self, f"lora_B_{p}"), s)
                       for p in QKV], dim=-1)
    return out + _gated(delta, x, flag), bias


def _o_forward(self, x, *args, **kwargs):
    out, bias = self._judge_orig_forward(x, *args, **kwargs)
    flag = JudgeCtx.flag
    if flag is not None and not JudgeCtx.any:
        return out, bias
    delta = _lora_delta(x, self.lora_A_o, self.lora_B_o, self.judge_lora_scaling)
    return out + _gated(delta, x, flag), bias


# ---------------------------------------------------------------------------
# attach
# ---------------------------------------------------------------------------
def expected_keys(n_layers: int) -> List[str]:
    """The adapter.safetensors keys of a flag-gated attention adapter (385 for the 48-layer 30B)."""
    keys = [f"model.layers.{i}.self_attn.{t}.lora_{m}" for i in range(n_layers) for t in TARGETS for m in "AB"]
    return keys + [DRAFT_EMBED_KEY]


def _device_of(mod) -> torch.device:
    for p in mod.parameters():
        return p.device
    return torch.device("cpu")


def _check_plain(mod, what: str) -> None:
    """TP must be 1 and the layer unquantized (the delta is added to the full, unsharded output)."""
    tp = getattr(mod, "tp_size", None)
    assert tp is None or tp == 1, f"{what}: tp_size={tp}; the judge LoRA supports TP = 1 only"
    qm = getattr(mod, "quant_method", None)
    assert qm is None or "Unquant" in type(qm).__name__, f"{what}: quantized ({type(qm).__name__})"
    assert "forward" not in vars(mod), f"{what}: its forward is already patched"


def _patch(mod, new_forward, tensors: Dict[str, torch.Tensor], scaling: float) -> None:
    dev = _device_of(mod)
    for name, t in tensors.items():                   # buffers: .to(device) moves them; not in state_dict
        mod.register_buffer(name, t.to(device=dev, dtype=torch.float32).contiguous(), persistent=False)
    mod.judge_lora_scaling = scaling
    mod._judge_orig_forward = types.MethodType(type(mod).forward, mod)
    mod.forward = types.MethodType(new_forward, mod)


def attach_judge_adapter(model, adapter_dir: str) -> dict:
    """Load a flag-gated judge adapter (adapter.save_adapter format) into an SGLang SDARMoeForCausalLM, in place."""
    from safetensors.torch import load_file
    assert getattr(model, "_judge_adapter", None) is None, "a judge adapter is already attached to this model"
    with open(os.path.join(adapter_dir, ADAPTER_JSON)) as fh:
        meta = json.load(fh)
    assert list(meta["targets"]) == TARGETS, f"targets {meta['targets']} != {TARGETS}"
    assert meta.get("draft_embed") is True, "not a flag-gated adapter (draft_embed is not true)"
    assert meta.get("train_gates") is False, "adapter has trained router gates; not supported here"
    assert not meta.get("conf_embed", False), "adapter has a confidence embedding; not supported here"
    r, alpha = int(meta["r"]), float(meta["alpha"])
    scaling = float(alpha) / float(r)                 # == LoRALinear.scaling

    layers = list(model.model.layers)
    n = len(layers)
    geom = meta.get("model", {})
    hidden = int(getattr(model.config, "hidden_size", 0)) or None
    for k, here in (("hidden_size", hidden), ("num_hidden_layers", n)):
        if k in geom and here is not None:
            assert int(geom[k]) == here, f"adapter {k}={geom[k]} != model {here}"

    sd = load_file(os.path.join(adapter_dir, ADAPTER_SAFETENSORS), device="cpu")
    want = expected_keys(n)
    missing, unexpected = sorted(set(want) - set(sd)), sorted(set(sd) - set(want))
    assert not (missing or unexpected), f"missing={missing[:4]} unexpected={unexpected[:4]}"

    plan = []                                         # validate every layer before patching any
    for i, layer in enumerate(layers):
        qkv, o = layer.self_attn.qkv_proj, layer.self_attn.o_proj
        _check_plain(qkv, f"layer {i} qkv_proj")
        _check_plain(o, f"layer {i} o_proj")
        t = {p: (sd[f"model.layers.{i}.self_attn.{p}_proj.lora_A"], sd[f"model.layers.{i}.self_attn.{p}_proj.lora_B"])
             for p in ("q", "k", "v", "o")}
        for p, (a, b) in t.items():
            assert a.shape[0] == r and b.shape[1] == r, f"layer {i} {p}_proj: A {tuple(a.shape)} B {tuple(b.shape)} r={r}"
        sizes = [t[p][1].shape[0] for p in QKV]
        if getattr(qkv, "output_sizes", None) is not None:
            assert list(qkv.output_sizes) == sizes, f"layer {i} qkv_proj output_sizes {qkv.output_sizes} != {sizes}"
        if getattr(qkv, "input_size", None) is not None:
            assert all(t[p][0].shape[1] == qkv.input_size for p in QKV), f"layer {i} qkv_proj input_size"
        if getattr(o, "input_size", None) is not None:
            assert t["o"][0].shape[1] == o.input_size and t["o"][1].shape[0] == o.output_size, f"layer {i} o_proj"
        plan.append((qkv, o, t))
    for qkv, o, t in plan:
        _patch(qkv, _qkv_forward, {f"lora_{m}_{p}": t[p]["AB".index(m)] for p in QKV for m in "AB"}, scaling)
        _patch(o, _o_forward, {"lora_A_o": t["o"][0], "lora_B_o": t["o"][1]}, scaling)

    w = sd[DRAFT_EMBED_KEY].float()
    assert w.dim() == 1 and (hidden is None or w.shape[0] == hidden), f"draft vector {tuple(w.shape)}"
    model.register_buffer("da_draft_embed_weight", w.to(_device_of(model.model.embed_tokens)).contiguous(),
                          persistent=False)
    model._judge_adapter = {"dir": os.path.abspath(adapter_dir), "r": r, "alpha": alpha, "scaling": scaling,
                            "n_layers": n, "n_tensors": len(sd), "draft_embed_norm": float(w.norm())}
    return dict(model._judge_adapter)


# ---------------------------------------------------------------------------
# the draft-flag embedding
# ---------------------------------------------------------------------------
def embed_with_draft_flag(model, input_ids: torch.Tensor, flag: Optional[torch.Tensor],
                          any_: Optional[bool] = None) -> torch.Tensor:
    """`embed_tokens(input_ids)` + the draft vector at flagged tokens (DraftEmbed.apply_to on [num_tokens])."""
    e = model.model.embed_tokens(input_ids)
    if flag is None or not (bool(flag.any()) if any_ is None else any_):
        return e
    assert flag.dtype == torch.bool and tuple(flag.shape) == tuple(e.shape[:1]), (flag.shape, e.shape)
    w = model.da_draft_embed_weight
    assert w.dtype == torch.float32, "the draft vector must stay fp32"
    return (e.float() + flag[:, None].to(torch.float32) * w.to(e.device).float()).to(e.dtype)


def judge_forward_prep(model, input_ids: torch.Tensor, flag: Optional[torch.Tensor]) -> Tuple[Optional[torch.Tensor], bool]:
    """(input_embeds or None, any_) for one judge forward. None when no token is flagged (plain input_ids path,
    exactly the stock model); `any_` goes to `JudgeCtx.flag_scope(flag, any_)`. flag None -> (None, False)."""
    if flag is None:
        return None, False
    any_ = bool(flag.any())
    if not any_:
        return None, False
    return embed_with_draft_flag(model, input_ids, flag, any_=True), True
