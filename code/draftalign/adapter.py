"""Manual LoRA + trainable routers for SDAR-30B-A3B-Chat-b32 (TRAINER_SPEC_v0 section 1).

No `peft`. Two kinds of trainable parameter are installed on a frozen bf16 model:

1. `LoRALinear` wrappers on `self_attn.{q,k,v,o}_proj` of every decoder layer.
   y = W x + (alpha/r) B(A x), A kaiming-uniform, B zeros, so the adapter is the
   IDENTITY at initialisation (bit-identical outputs, pinned by a test). The A/B
   parameters are kept in fp32 (master weights for AdamW) and cast to the base
   module's dtype inside the forward, so the matmuls run in bf16 exactly like the
   base ones.

2. `GateFP32` wrappers on `mlp.gate` (Linear 2048 -> 128) of every MoE layer.
   The master copy is fp32; the forward uses `master.to(bf16)`.

   Why that variant and not "keep the gate Linear in fp32 and cast its output":
   the stock router does a **bf16** matmul (`self.gate(hidden_states)` with a bf16
   weight, modeling_sdar_moe.py:287) and only then softmaxes in fp32
   (modeling_sdar_moe.py:289). Casting the fp32 master down to bf16 *before* the
   matmul reproduces that arithmetic exactly, and bf16 -> fp32 -> bf16 is a
   lossless round trip, so at initialisation the router logits are BIT-IDENTICAL
   to stock. An fp32 gate matmul would instead accumulate differently and shift
   the rank-8/rank-9 router margin (`moe_instr.S1Probe` measures how thin that
   margin is), i.e. it would silently change which experts fire at inference.
   The cast is differentiable, so this is exactly "an fp32 copy written back into
   the bf16 gate weight", performed on every forward instead of every step.

3. OPTIONAL (`attach_lora(..., draft_flag=True)`, round 2): the DRAFT FLAG.
   A single trainable vector `DraftEmbed.weight` [hidden], fp32, ZERO-initialised,
   is added to the input embedding of every position whose input token is a
   TENTATIVE DRAFT token, and the SAME per-position 0/1 flag gates every LoRA
   delta:  y = W x + g_i * (alpha/r) B A x.  The two together answer round 1's
   diagnosis (docs/EXPLAINER_DRAFT_ALIGNED_TRAINING.md section 13): the model can
   now SEE which slots are tentative, and the adapter's delta is confined to them,
   so learning to judge a draft can no longer bleed into masked / committed slots.
   At an all-False flag every projection is bit-identical to the frozen base
   (the delta is skipped entirely) and the embedding is untouched, i.e. the
   all-mask regime is exactly the stock model.

   The flag is a MODULE-LEVEL context (`CTX`, armed with `flag_scope`) rather than
   an argument, because the delta is applied deep inside the model's own attention
   forward, which we do not own.  `CTX.flag is None` (the default) is exactly the
   pre-flag behaviour: an ungated delta everywhere.

`merge_lora(model)` folds the LoRA update into the base weights and turns the
gate wrappers back into plain bf16 `nn.Linear`s, giving a stock module tree for
inference.  It REFUSES on a flag-gated adapter: a merged weight applies at every
position, which is precisely what the gate exists to prevent -- such an adapter
must run un-merged (`run_da_decode.load_adapter_into`).
"""
from __future__ import annotations

import contextlib
import inspect
import json
import math
import os
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

ATTN_TARGETS: Tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")
ADAPTER_JSON = "adapter_config.json"
ADAPTER_SAFETENSORS = "adapter.safetensors"
DRAFT_EMBED_ATTR = "da_draft_embed"                      # model attribute holding the DraftEmbed
DRAFT_EMBED_KEY = f"{DRAFT_EMBED_ATTR}.weight"           # its key in the adapter state dict
CONF_EMBED_ATTR = "da_conf_embed"                        # 1B corpus, 32M test: drafter-confidence embedding
CONF_EMBED_KEY = f"{CONF_EMBED_ATTR}.weight"
CONF_BUCKETS = 16                                        # uint8 confidence (round(255 p)) -> 16 buckets


# ---------------------------------------------------------------------------
# the draft-flag context (module level: the delta is applied inside the model's
# own attention forward, which this file does not own)
# ---------------------------------------------------------------------------
class _FlagCtx:
    """`flag` [B, S] bool (or None = ungated, the pre-flag behaviour); `any` caches
    `bool(flag.any())` so the hot path never syncs the device."""

    __slots__ = ("flag", "any")

    def __init__(self):
        self.flag: Optional[torch.Tensor] = None
        self.any: bool = False


CTX = _FlagCtx()


def set_flag(flag: Optional[torch.Tensor], any_: Optional[bool] = None) -> None:
    """Arm the per-position LoRA gate.  `flag` [B, S] bool, or None to disarm.

    `any_` (when known) avoids the `flag.any()` device sync; it MUST equal
    `bool(flag.any())` -- a wrong value silently changes the arithmetic.
    """
    if flag is None:
        CTX.flag, CTX.any = None, False
        return
    assert flag.dtype == torch.bool, flag.dtype
    assert flag.dim() == 2, f"the flag must be [B, S] bool, got {tuple(flag.shape)}"
    CTX.flag = flag
    CTX.any = bool(flag.any()) if any_ is None else bool(any_)


def clear_flag() -> None:
    set_flag(None)


@contextlib.contextmanager
def flag_scope(flag: Optional[torch.Tensor], any_: Optional[bool] = None):
    """Arm `flag` for the body and restore the previous context afterwards (also on exceptions)."""
    prev_flag, prev_any = CTX.flag, CTX.any
    set_flag(flag, any_)
    try:
        yield
    finally:
        CTX.flag, CTX.any = prev_flag, prev_any


def _gate_like(flag: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """`flag` [B, S] -> a [B, S, 1, ...] 0/1 tensor of `y`'s dtype that broadcasts over `y`.

    Every gated projection is called with [B, S, *] (q/k/v get [B, S, hidden], o gets
    [B, S, num_heads * head_dim]), so the same [B, S] flag serves all four.
    """
    assert y.dim() >= flag.dim() and tuple(y.shape[:2]) == tuple(flag.shape), \
        f"flag {tuple(flag.shape)} does not match the gated tensor {tuple(y.shape)}"
    return flag.to(y.dtype).reshape(*flag.shape, *([1] * (y.dim() - flag.dim())))


# ---------------------------------------------------------------------------
# modules
# ---------------------------------------------------------------------------
class LoRALinear(nn.Module):
    """`base` (frozen) + (alpha/r) B A, with fp32 A/B computed in the base dtype."""

    def __init__(self, base: nn.Linear, r: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        assert isinstance(base, nn.Linear), type(base)
        assert r > 0, r
        self.base = base
        self.r = int(r)
        self.alpha = float(alpha)
        self.scaling = float(alpha) / float(r)
        self.dropout_p = float(dropout)
        self.in_features = base.in_features
        self.out_features = base.out_features
        dev = base.weight.device                       # live on the base weight's device (GPU at inference/training)
        self.lora_A = nn.Parameter(torch.empty(self.r, base.in_features, dtype=torch.float32, device=dev))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, self.r, dtype=torch.float32, device=dev))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        for p in self.base.parameters():
            p.requires_grad_(False)

    @property
    def weight(self):                                   # so callers reading `.weight` still work
        return self.base.weight

    def delta(self) -> torch.Tensor:
        """The fp32 weight update (alpha/r) B A, shape [out, in]."""
        return (self.lora_B @ self.lora_A) * self.scaling

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        flag = CTX.flag
        if flag is not None and not CTX.any:
            return out                                  # every position gated off -> the frozen base, bit-identical
        h = x
        if self.dropout_p > 0.0 and self.training:
            h = F.dropout(h, p=self.dropout_p)
        a = self.lora_A.to(x.dtype)
        b = self.lora_B.to(x.dtype)
        delta = F.linear(F.linear(h, a), b) * self.scaling
        if flag is not None:
            delta = delta * _gate_like(flag, delta)     # unflagged positions get NO delta
        return out + delta

    def extra_repr(self) -> str:
        return f"r={self.r}, alpha={self.alpha}, dropout={self.dropout_p}"


class DraftEmbed(nn.Module):
    """The zero-initialised draft-flag vector added to the input embedding of flagged slots.

    fp32 master (like LoRA's A/B); the addition is done in fp32 and cast back to the
    embedding's dtype, so at `weight == 0` the result is BIT-IDENTICAL to `embed_tokens(ids)`
    (bf16 -> fp32 -> bf16 is a lossless round trip).
    """

    def __init__(self, hidden_size: int, device=None):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.weight = nn.Parameter(torch.zeros(self.hidden_size, dtype=torch.float32, device=device))

    def apply_to(self, embeds: torch.Tensor, flag: torch.Tensor) -> torch.Tensor:
        """`embeds` [B, S, H] (any dtype) + `flag` [B, S] bool -> [B, S, H] in `embeds`' dtype."""
        assert embeds.shape[-1] == self.hidden_size, (embeds.shape, self.hidden_size)
        assert tuple(embeds.shape[:2]) == tuple(flag.shape), (embeds.shape, flag.shape)
        out = embeds.float() + flag.unsqueeze(-1).to(torch.float32) * self.weight.to(embeds.device).float()
        return out.to(embeds.dtype)

    def forward(self, embeds: torch.Tensor, flag: torch.Tensor) -> torch.Tensor:
        return self.apply_to(embeds, flag)

    def extra_repr(self) -> str:
        return f"hidden_size={self.hidden_size}"


class ConfEmbed(nn.Module):
    """1B-corpus 32M test: one zero-initialised vector per drafter-confidence bucket, added (like the draft flag)
    to the input embedding of every FLAGGED draft slot. At weight == 0 it is bit-identical to no embedding, so a
    run with it starts exactly where a run without it starts; only training can move it."""

    def __init__(self, hidden_size: int, buckets: int = CONF_BUCKETS, device=None):
        super().__init__()
        self.hidden_size, self.buckets = int(hidden_size), int(buckets)
        self.weight = nn.Parameter(torch.zeros(self.buckets, self.hidden_size, dtype=torch.float32, device=device))

    def bucket(self, conf_u8: torch.Tensor) -> torch.Tensor:
        return (conf_u8.long().clamp(0, 255) * self.buckets) // 256

    def apply_to(self, embeds: torch.Tensor, flag: torch.Tensor, conf_u8: torch.Tensor) -> torch.Tensor:
        """`embeds` [B, S, H], `flag` [B, S] bool, `conf_u8` [B, S] uint8 -> [B, S, H] in `embeds`' dtype."""
        assert tuple(embeds.shape[:2]) == tuple(flag.shape) == tuple(conf_u8.shape), (embeds.shape, flag.shape)
        vec = self.weight.to(embeds.device).float()[self.bucket(conf_u8.to(embeds.device))]
        out = embeds.float() + flag.unsqueeze(-1).to(torch.float32) * vec
        return out.to(embeds.dtype)

    def extra_repr(self) -> str:
        return f"buckets={self.buckets}, hidden_size={self.hidden_size}"


class GateFP32(nn.Module):
    """Router gate with an fp32 master weight, applied as a bf16 matmul (see module docstring)."""

    def __init__(self, base: nn.Linear):
        super().__init__()
        assert isinstance(base, nn.Linear) and base.bias is None, "SDAR router gate is bias-free"
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.compute_dtype = base.weight.dtype
        self.weight = nn.Parameter(base.weight.detach().float().clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight.to(self.compute_dtype))

    def to_linear(self) -> nn.Linear:
        lin = nn.Linear(self.in_features, self.out_features, bias=False,
                        device=self.weight.device, dtype=self.compute_dtype)
        with torch.no_grad():
            lin.weight.copy_(self.weight.to(self.compute_dtype))
        return lin

    def extra_repr(self) -> str:
        return f"in={self.in_features}, out={self.out_features}, compute_dtype={self.compute_dtype}"


# ---------------------------------------------------------------------------
# installation
# ---------------------------------------------------------------------------
def decoder_layers(model) -> List[nn.Module]:
    """`model.model.layers`, the module list the remote code builds (modeling_sdar_moe.py:535-538)."""
    return list(model.model.layers)


def _moe_gate(layer) -> Optional[nn.Module]:
    mlp = getattr(layer, "mlp", None)
    if mlp is not None and hasattr(mlp, "gate") and hasattr(mlp, "experts"):
        return mlp
    return None


def attach_lora(model, r: int = 32, alpha: float = 64.0, dropout: float = 0.0,
                targets: Iterable[str] = ATTN_TARGETS, train_gates: bool = True,
                draft_flag: bool = False, conf_embed: bool = False) -> Dict[str, int]:
    """Wrap the attention projections and (optionally) the router gates in place.

    Idempotent: already-wrapped modules are left alone. Returns counts.
    `draft_flag` additionally installs the zero-initialised `DraftEmbed` and marks the
    adapter as FLAG-GATED (metadata `draft_embed: true`); with it off nothing at all
    changes -- not even the metadata, so an unflagged adapter's `config_hash` is
    exactly the one round 1 wrote.
    """
    targets = tuple(targets)
    n_lora, n_gate = 0, 0
    for layer in decoder_layers(model):
        attn = layer.self_attn
        for name in targets:
            mod = getattr(attn, name)
            if isinstance(mod, LoRALinear):
                continue
            setattr(attn, name, LoRALinear(mod, r=r, alpha=alpha, dropout=dropout))
            n_lora += 1
        if train_gates:
            mlp = _moe_gate(layer)
            if mlp is not None and not isinstance(mlp.gate, GateFP32):
                mlp.gate = GateFP32(mlp.gate)
                n_gate += 1
    cfg = {"r": int(r), "alpha": float(alpha), "dropout": float(dropout),
           "targets": list(targets), "train_gates": bool(train_gates)}
    counts = {"lora": n_lora, "gates": n_gate}
    if draft_flag:
        cfg["draft_embed"] = True
        counts["draft_embed"] = int(_attach_draft_embed(model))
    if conf_embed:
        assert draft_flag, "the confidence embedding lives on the draft flag's slots: attach with draft_flag=True"
        cfg["conf_embed"] = True
        counts["conf_embed"] = int(_attach_conf_embed(model))
    model._da_adapter = cfg
    return counts


def _attach_conf_embed(model) -> bool:
    if isinstance(getattr(model, CONF_EMBED_ATTR, None), ConfEmbed):
        return False
    dev = next(iter(model.parameters())).device
    setattr(model, CONF_EMBED_ATTR, ConfEmbed(int(model.config.hidden_size), device=dev))
    return True


def get_conf_embed(model) -> Optional["ConfEmbed"]:
    mod = getattr(model, CONF_EMBED_ATTR, None)
    return mod if isinstance(mod, ConfEmbed) else None


def _attach_draft_embed(model) -> bool:
    """Install `model.<DRAFT_EMBED_ATTR>` (idempotent). Returns True if it was created here."""
    if isinstance(getattr(model, DRAFT_EMBED_ATTR, None), DraftEmbed):
        return False
    hidden = int(model.config.hidden_size)
    dev = None
    for p in model.parameters():
        dev = p.device
        break
    setattr(model, DRAFT_EMBED_ATTR, DraftEmbed(hidden, device=dev))
    return True


def get_draft_embed(model) -> Optional[DraftEmbed]:
    """The model's `DraftEmbed`, or None when the adapter is not flag-gated."""
    mod = getattr(model, DRAFT_EMBED_ATTR, None)
    return mod if isinstance(mod, DraftEmbed) else None


def is_flag_gated(model) -> bool:
    """True when this model carries a flag-gated adapter (metadata `draft_embed: true`)."""
    return bool((getattr(model, "_da_adapter", None) or {}).get("draft_embed"))


def input_embeddings(model) -> nn.Module:
    """`embed_tokens` of the model (`get_input_embeddings` when the class provides it)."""
    fn = getattr(model, "get_input_embeddings", None)
    if callable(fn):
        try:
            emb = fn()
            if emb is not None:
                return emb
        except (AttributeError, NotImplementedError):      # pragma: no cover - HF stubs
            pass
    return model.model.embed_tokens


def supports_inputs_embeds(model) -> bool:
    """Whether `model.forward` takes `inputs_embeds` (SDARMoeForCausalLM does: remote :969)."""
    try:
        params = inspect.signature(model.forward).parameters
    except (TypeError, ValueError):                        # pragma: no cover - exotic callables
        return False
    return "inputs_embeds" in params            # a bare **kwargs is NOT taken as support


def can_apply_draft_flag(model) -> bool:
    """Whether the draft embedding can reach this model at all: either its forward takes
    `inputs_embeds`, or it exposes an `embed_tokens` module a forward hook can sit on."""
    if supports_inputs_embeds(model):
        return True
    try:
        return input_embeddings(model) is not None
    except AttributeError:
        return False


@contextlib.contextmanager
def draft_flag_scope(model, input_ids: Optional[torch.Tensor], flag: Optional[torch.Tensor],
                     any_: Optional[bool] = None, conf_u8: Optional[torch.Tensor] = None):
    """Arm the draft flag for ONE forward of `model`; yields the extra kwargs to call it with.

    Yields `{}` (call the model with `input_ids` as usual) or `{"inputs_embeds": ...}` (call it
    with those INSTEAD of `input_ids`).  When the model's forward does not accept
    `inputs_embeds`, the embedding is added by a temporary forward hook on `embed_tokens`
    instead and `{}` is yielded.  With `flag=None` nothing is armed beyond disarming the LoRA
    gate, which is the pre-flag behaviour.
    """
    if flag is None:
        with flag_scope(None):
            yield {}
        return
    if any_ is None:
        any_ = bool(flag.any())
    de = get_draft_embed(model)
    ce = get_conf_embed(model)
    if ce is not None and any_:
        assert conf_u8 is not None, "this adapter has a confidence embedding: pass the drafter's confidence"
        conf_u8 = conf_u8.reshape(flag.shape).to(flag.device)

    def _embed(out, _flag=flag):
        out = de.apply_to(out, _flag)
        if ce is not None:
            out = ce.apply_to(out, _flag, conf_u8)
        return out

    kwargs: Dict[str, torch.Tensor] = {}
    handle = None
    if de is not None and any_:
        if supports_inputs_embeds(model):
            assert input_ids is not None, "draft_flag_scope needs input_ids to build inputs_embeds"
            kwargs = {"inputs_embeds": _embed(input_embeddings(model)(input_ids))}
        else:                                              # fallback: add it on the way out of embed_tokens
            def _hook(_mod, _args, out):
                return _embed(out)
            handle = input_embeddings(model).register_forward_hook(_hook)
    try:
        with flag_scope(flag, any_):
            yield kwargs
    finally:
        if handle is not None:
            handle.remove()


def freeze_base(model) -> Dict[str, int]:
    """Everything frozen except `lora_A`/`lora_B` and the `GateFP32` masters."""
    for p in model.parameters():
        p.requires_grad_(False)
    n = 0
    for mod in model.modules():
        if isinstance(mod, LoRALinear):
            mod.lora_A.requires_grad_(True)
            mod.lora_B.requires_grad_(True)
            n += 2
        elif isinstance(mod, (GateFP32, DraftEmbed, ConfEmbed)):
            mod.weight.requires_grad_(True)
            n += 1
    return {"trainable_tensors": n}


def trainable_parameters(model) -> List[Tuple[str, nn.Parameter]]:
    """[(name, param)] of every parameter with requires_grad, in module order."""
    return [(n, p) for n, p in model.named_parameters() if p.requires_grad]


def parameter_groups(model, lr_lora: float, lr_gate: float, weight_decay: float = 0.0) -> List[dict]:
    """AdamW groups: LoRA at `lr_lora`, router gates at `lr_gate` (spec section 3.3)."""
    lora, gates = [], []
    for name, p in trainable_parameters(model):
        (gates if name.endswith("mlp.gate.weight") else lora).append(p)   # the draft vector trains at lr_lora
    groups = []
    if lora:
        groups.append({"params": lora, "lr": lr_lora, "weight_decay": weight_decay, "name": "lora"})
    if gates:
        groups.append({"params": gates, "lr": lr_gate, "weight_decay": weight_decay, "name": "gate"})
    return groups


def adapter_state_dict(model) -> Dict[str, torch.Tensor]:
    """fp32 CPU tensors of every trainable adapter parameter, keyed by its module path."""
    out: Dict[str, torch.Tensor] = {}
    for name, p in model.named_parameters():
        if _is_adapter_key(model, name):
            out[name] = p.detach().float().cpu().contiguous()
    return out


def _module_of(model, param_name: str) -> nn.Module:
    mod = model
    for part in param_name.split(".")[:-1]:
        mod = getattr(mod, part) if not part.isdigit() else mod[int(part)]
    return mod


def adapter_metadata(model) -> dict:
    """r / alpha / targets + the model geometry, so a load can refuse a mismatched checkpoint."""
    from research_protocol import config_hash
    cfg = getattr(model, "_da_adapter", {}).copy()
    c = model.config
    geom = {k: int(getattr(c, k)) for k in ("hidden_size", "num_hidden_layers", "num_experts",
                                            "num_experts_per_tok", "vocab_size")
            if getattr(c, k, None) is not None}
    cfg["model"] = geom
    cfg["config_hash"] = config_hash(cfg)
    return cfg


def save_adapter(model, path: str) -> str:
    """Write `adapter.safetensors` + `adapter_config.json` into `path`; returns the directory."""
    from safetensors.torch import save_file
    os.makedirs(path, exist_ok=True)
    sd = adapter_state_dict(model)
    meta = adapter_metadata(model)
    meta["n_tensors"] = len(sd)
    save_file(sd, os.path.join(path, ADAPTER_SAFETENSORS),
              metadata={"format": "pt", "da_adapter": json.dumps(meta, sort_keys=True)})
    with open(os.path.join(path, ADAPTER_JSON), "w") as fh:
        json.dump(meta, fh, indent=2, sort_keys=True)
    return path


def load_adapter(model, path: str, strict: bool = True) -> dict:
    """Load an adapter directory into an already-attached model (attaches it if needed)."""
    from safetensors.torch import load_file
    with open(os.path.join(path, ADAPTER_JSON)) as fh:
        meta = json.load(fh)
    if not getattr(model, "_da_adapter", None):
        attach_lora(model, r=meta["r"], alpha=meta["alpha"], dropout=meta.get("dropout", 0.0),
                    targets=meta["targets"], train_gates=meta.get("train_gates", True),
                    draft_flag=bool(meta.get("draft_embed", False)),
                    conf_embed=bool(meta.get("conf_embed", False)))
        freeze_base(model)
    here = adapter_metadata(model)
    if strict:
        for k in ("r", "alpha", "targets", "train_gates", "draft_embed", "conf_embed", "model"):
            assert here.get(k) == meta.get(k), f"adapter mismatch on {k}: {here.get(k)} != {meta.get(k)}"
    sd = load_file(os.path.join(path, ADAPTER_SAFETENSORS))
    own = dict(model.named_parameters())
    missing = [k for k in own if _is_adapter_key(model, k) and k not in sd]
    unexpected = [k for k in sd if k not in own]
    assert not (strict and (missing or unexpected)), f"missing={missing[:4]} unexpected={unexpected[:4]}"
    with torch.no_grad():
        for k, v in sd.items():
            own[k].copy_(v.to(device=own[k].device, dtype=own[k].dtype))
    return {"loaded": len(sd), "missing": len(missing), "unexpected": len(unexpected), "meta": meta}


def _is_adapter_key(model, name: str) -> bool:
    if name == DRAFT_EMBED_KEY:
        return isinstance(_module_of(model, name), DraftEmbed)
    if name == CONF_EMBED_KEY:
        return isinstance(_module_of(model, name), ConfEmbed)
    return (name.endswith(("lora_A", "lora_B")) or name.endswith("mlp.gate.weight")) and \
        isinstance(_module_of(model, name), (LoRALinear, GateFP32))


@torch.no_grad()
def sync_gates(model) -> int:
    """No-op for `GateFP32` (the bf16 view is derived every forward); kept for symmetry / assertions."""
    return sum(1 for m in model.modules() if isinstance(m, GateFP32))


@torch.no_grad()
def merge_lora(model) -> Dict[str, int]:
    """Fold LoRA into the base weights and restore plain modules (inference-time model).

    REFUSES on a flag-gated adapter: a merged weight applies its delta at EVERY position,
    while the trained adapter's delta exists only at flagged (tentative draft) slots -- the
    merge would silently change the model at masked and committed slots.  Such an adapter is
    run un-merged, with `draft_flag_scope` arming the gate per forward.
    """
    if is_flag_gated(model):
        raise RuntimeError(
            "merge_lora: this adapter is flag-gated (draft_embed) -- merging would make the LoRA "
            "delta apply at every position instead of only at flagged draft slots. Run it un-merged.")
    n_lora, n_gate = 0, 0
    for layer in decoder_layers(model):
        attn = layer.self_attn
        for name in list(ATTN_TARGETS):
            mod = getattr(attn, name, None)
            if isinstance(mod, LoRALinear):
                base = mod.base
                base.weight.add_(mod.delta().to(base.weight.dtype))
                setattr(attn, name, base)
                n_lora += 1
        mlp = _moe_gate(layer)
        if mlp is not None and isinstance(mlp.gate, GateFP32):
            mlp.gate = mlp.gate.to_linear()
            n_gate += 1
    model._da_adapter = {**getattr(model, "_da_adapter", {}), "merged": True}
    return {"lora": n_lora, "gates": n_gate}


def count_trainable(model) -> Dict[str, int]:
    tp = trainable_parameters(model)
    gate = sum(p.numel() for n, p in tp if n.endswith("mlp.gate.weight"))
    draft = sum(p.numel() for n, p in tp if n == DRAFT_EMBED_KEY)
    conf = sum(p.numel() for n, p in tp if n == CONF_EMBED_KEY)
    lora = sum(p.numel() for n, p in tp) - gate - draft - conf
    out = {"lora": lora, "gate": gate, "draft_embed": draft, "total": lora + gate + draft + conf}
    if conf:
        out["conf_embed"] = conf
    return out
