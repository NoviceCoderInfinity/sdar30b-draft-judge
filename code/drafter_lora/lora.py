"""LoRA on the dense drafter SDAR-1.7B-Chat-b32: attention AND MLP projections, every layer.

No `peft`.  The wrapper is the repo's own `draftalign.adapter.LoRALinear` (fp32 A/B, B zero-initialised, so the
adapter is the identity at step 0 and the KL term of the loss is exactly 0 there).  `adapter.attach_lora` only
reaches `self_attn`; the drafter also needs `mlp.{gate,up,down}_proj`, so this module attaches by (sub-module,
name) pairs and keeps everything else of `adapter.py` (save format, the flag context) unchanged.

The ORIGINAL drafter q_0 of the loss is the same model with the adapter switched off: `adapter_off(model, shape)`
arms an all-False `adapter.CTX` flag, under which every `LoRALinear` returns its frozen base output before
computing any delta -- bit-identical to the stock weights, no second copy of the model.

Unlike the judge's flag-gated adapter (AGENTS.md footgun e), this adapter is UNGATED: its delta applies at every
position, exactly as a merged weight would.  So `merge_into_base` is legitimate here and is how the trained drafter
reaches `run_da_decode.py --drafter <dir>` (a plain HF directory, same size and speed as the stock drafter).

    SDAR-1.7B, r = 32: attention 28 x 32 x (4096 + 3072 + 3072 + 4096) = 12,845,056
                       MLP       28 x 32 x (8192 + 8192 + 8192)        = 22,020,096   -> 34,865,152 trainable
"""
from __future__ import annotations

import json
import os
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn

from draftalign import adapter as A

ATTN = ("q_proj", "k_proj", "v_proj", "o_proj")
MLP = ("gate_proj", "up_proj", "down_proj")
TARGETS: Tuple[str, ...] = ATTN + MLP
EXPECTED_TRAINABLE_D32 = 34_865_152                    # the runbook's non-negotiable count for d32 at r = 32
ADAPTER_JSON = "adapter_config.json"
ADAPTER_SAFETENSORS = "adapter.safetensors"
KIND = "drafter_lora_v1"


def _owner(layer, name: str) -> nn.Module:
    return layer.self_attn if name in ATTN else layer.mlp


def attach(model, r: int = 32, alpha: float = 64.0, dropout: float = 0.0,
           targets: Iterable[str] = TARGETS) -> Dict[str, int]:
    """Wrap `targets` of every decoder layer in a `LoRALinear` (idempotent) and freeze everything else."""
    targets = tuple(targets)
    bad = [t for t in targets if t not in TARGETS]
    assert not bad, f"unknown LoRA targets {bad}; allowed {TARGETS}"
    n = 0
    for layer in model.model.layers:
        for name in targets:
            own = _owner(layer, name)
            mod = getattr(own, name)
            if isinstance(mod, A.LoRALinear):
                continue
            setattr(own, name, A.LoRALinear(mod, r=r, alpha=alpha, dropout=dropout))
            n += 1
    for p in model.parameters():
        p.requires_grad_(False)
    for m in model.modules():
        if isinstance(m, A.LoRALinear):
            m.lora_A.requires_grad_(True)
            m.lora_B.requires_grad_(True)
    model._drafter_lora = {"kind": KIND, "r": int(r), "alpha": float(alpha), "dropout": float(dropout),
                           "targets": list(targets)}
    return {"wrapped": n}


def lora_modules(model) -> List[Tuple[str, A.LoRALinear]]:
    return [(n, m) for n, m in model.named_modules() if isinstance(m, A.LoRALinear)]


def trainable(model) -> List[Tuple[str, nn.Parameter]]:
    return [(n, p) for n, p in model.named_parameters() if p.requires_grad]


def count(model) -> Dict[str, int]:
    """Trainable parameters split into attention and MLP (the runbook's 12,845,056 + 22,020,096)."""
    attn = mlp = other = 0
    for n, p in trainable(model):
        if ".self_attn." in n:
            attn += p.numel()
        elif ".mlp." in n:
            mlp += p.numel()
        else:
            other += p.numel()
    return {"attention": attn, "mlp": mlp, "other": other, "total": attn + mlp + other}


class adapter_off:
    """Context: every LoRALinear returns its frozen base output (the original drafter q_0).

    `shape` is the [B, S] of the forward's hidden states; the all-False flag never reaches a broadcast because
    `LoRALinear.forward` returns before computing a delta when `CTX.any` is False."""

    def __init__(self, shape: Tuple[int, int], device=None):
        self.flag = torch.zeros(*shape, dtype=torch.bool, device=device)
        self._cm = None

    def __enter__(self):
        self._cm = A.flag_scope(self.flag, False)
        self._cm.__enter__()
        return self

    def __exit__(self, *exc):
        return self._cm.__exit__(*exc)


def adapter_on():
    """Context: the ungated delta everywhere (the default state; used to be explicit inside nested scopes)."""
    return A.flag_scope(None)


# ---------------------------------------------------------------------------
# save / load (adapter only) and merge (full HF directory)
# ---------------------------------------------------------------------------
def state_dict(model) -> Dict[str, torch.Tensor]:
    return {n: p.detach().float().cpu().contiguous() for n, p in trainable(model)}


def metadata(model) -> dict:
    c = model.config
    meta = dict(getattr(model, "_drafter_lora", {}))
    meta["model"] = {k: int(getattr(c, k)) for k in ("hidden_size", "num_hidden_layers", "intermediate_size",
                                                     "num_attention_heads", "num_key_value_heads", "vocab_size")
                     if getattr(c, k, None) is not None}
    return meta


def save(model, path: str, extra: Optional[dict] = None) -> str:
    from safetensors.torch import save_file
    os.makedirs(path, exist_ok=True)
    sd = state_dict(model)
    meta = {**metadata(model), **(extra or {}), "n_tensors": len(sd),
            "n_params": int(sum(v.numel() for v in sd.values()))}
    tmp = os.path.join(path, ADAPTER_SAFETENSORS + ".tmp")
    save_file(sd, tmp, metadata={"format": "pt", "drafter_lora": json.dumps(meta, sort_keys=True)})
    os.replace(tmp, os.path.join(path, ADAPTER_SAFETENSORS))
    with open(os.path.join(path, ADAPTER_JSON), "w") as fh:
        json.dump(meta, fh, indent=2, sort_keys=True)
    return path


def load(model, path: str) -> dict:
    """Load an adapter directory into `model` (attaching it first when needed).  Strict on shape and names."""
    from safetensors.torch import load_file
    meta = json.load(open(os.path.join(path, ADAPTER_JSON)))
    assert meta.get("kind") == KIND, f"{path} is not a drafter LoRA ({meta.get('kind')})"
    if not getattr(model, "_drafter_lora", None):
        attach(model, r=meta["r"], alpha=meta["alpha"], dropout=meta.get("dropout", 0.0), targets=meta["targets"])
    here = metadata(model)
    for k in ("r", "alpha", "targets", "model"):
        assert here.get(k) == meta.get(k), f"adapter mismatch on {k}: {here.get(k)} != {meta.get(k)}"
    sd = load_file(os.path.join(path, ADAPTER_SAFETENSORS))
    own = dict(trainable(model))
    missing, unexpected = sorted(set(own) - set(sd)), sorted(set(sd) - set(own))
    assert not missing and not unexpected, f"missing={missing[:4]} unexpected={unexpected[:4]}"
    with torch.no_grad():
        for k, v in sd.items():
            own[k].copy_(v.to(device=own[k].device, dtype=own[k].dtype))
    return meta


@torch.no_grad()
def merge_into_base(model) -> int:
    """Fold (alpha/r) B A into every wrapped base weight (in the base dtype) and restore plain nn.Linear modules."""
    n = 0
    for layer in model.model.layers:
        for name in TARGETS:
            own = _owner(layer, name)
            mod = getattr(own, name, None)
            if isinstance(mod, A.LoRALinear):
                base = mod.base
                base.weight.add_(mod.delta().to(device=base.weight.device, dtype=base.weight.dtype))
                setattr(own, name, base)
                n += 1
    model._drafter_lora = {**getattr(model, "_drafter_lora", {}), "merged": True}
    return n


def save_merged(model_key: str, adapter_dir: str, out_dir: str, device: str = "cpu") -> str:
    """A full HF directory of the drafter with the adapter merged in, loadable by `run_da_decode.py --drafter`.

    Loads a FRESH bf16 copy of the pinned base (`research_protocol.load_sdar`), attaches and loads the adapter, merges
    in bf16 and writes it exactly as `train_drafter.save_checkpoint` writes its full checkpoints (save_pretrained +
    tokenizer + the remote-code .py files + offline auto_map), so `eval_drafter_agreement.load_drafter` and
    `run_da_decode.drafter_info` accept it unchanged."""
    import research_protocol as P
    from draftalign.train_drafter import _localize_auto_map, copy_model_files, snapshot_dir
    tok, model = P.load_sdar(model_key, device)
    meta = load(model, adapter_dir)
    merge_into_base(model)
    for p in model.parameters():
        p.requires_grad_(False)
    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir, safe_serialization=True)
    tok.save_pretrained(out_dir)
    copy_model_files(snapshot_dir(model_key), out_dir)
    _localize_auto_map(out_dir)
    with open(os.path.join(out_dir, "drafter_lora_merge.json"), "w") as fh:
        json.dump({"base": model_key, "base_revision": P.MODELS[model_key]["revision"],
                   "adapter_dir": os.path.abspath(adapter_dir), "adapter": meta}, fh, indent=2, sort_keys=True)
    return out_dir
