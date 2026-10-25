"""Expert-activation logging: which experts ran, per forward and per layer, at decode time and at training time.

These numbers cannot be recomputed from the decode records afterwards: the records keep only one count per
forward (`unique_expert_layer_loads`), and the routing depends on the adapter, the pool and the draft in the
block at that moment.

DECODE (`ExpertLogMoEMeter`, a drop-in for `da_decode.MoEMeter`)
  Reads the same `sdar_harness.moe_instr.SINK` capture the meter already arms (the very `selected_experts` tensor
  the dispatch indexes; under a DES pool, the restricted set that really ran). Per item it keeps:
    hit_bits   [F, L, E/8] uint8   per forward, per layer: packed bitmask of the experts that ran
    kinds      [F] int8            the forward's kind (da_decode.MOE_KIND_CODE)
    counts     [K, L, E] int32     tokens routed to each expert, summed per forward kind (K = kinds)
    mass_sum   [K, L] float64      sum over rows of the top-8 router mass before renormalisation
    mass_n     [K, L] int64        rows behind mass_sum
  and, with `level="full"`, every row's top-8 ids and probs per forward (large: about 10 MB per item).
  `summary()` goes into the item record; `save(path)` writes the arrays to an npz beside the decode trace.

TRAINING (`TrainExpertCapture`)
  Wraps `train_fast.moe_forward` / `prefix_kv` / `block_hidden` and counts the top-8 experts each MoE layer
  selects, separately for the no-grad prefix pass and the trained block pass (armed only around the forward, so a
  checkpoint recompute during backward is never counted twice). The window's counts are all-reduced and turned into
  per-layer unique-expert counts, load entropy and max-load share, and written as an npz per log window.
"""
from __future__ import annotations

import math
import os
from contextlib import contextmanager
from typing import Dict, List, Optional

import numpy as np
import torch

N_LAYERS = 48
N_EXPERTS = 128
TOP_K = 8


def _entropy_norm(c: np.ndarray) -> np.ndarray:
    """Normalised load entropy per row of c [..., E] (1 = uniform over all E experts, 0 = one expert)."""
    c = c.astype(np.float64)
    s = c.sum(-1, keepdims=True)
    p = np.divide(c, s, out=np.zeros_like(c), where=s > 0)
    h = -(p * np.log(np.where(p > 0, p, 1.0))).sum(-1)
    return h / math.log(c.shape[-1])


def load_stats(counts: np.ndarray) -> Dict[str, float]:
    """counts [L, E] -> layer-averaged unique experts hit, normalised load entropy, max-load share."""
    tot = counts.sum(-1)
    live = tot > 0
    if not live.any():
        return {"uniq_per_layer": 0.0, "entropy": float("nan"), "max_share": float("nan"), "layers": 0}
    uniq = (counts > 0).sum(-1)[live]
    ent = _entropy_norm(counts)[live]
    mx = (counts.max(-1) / np.maximum(tot, 1))[live]
    return {"uniq_per_layer": float(uniq.mean()), "entropy": float(ent.mean()), "max_share": float(mx.mean()),
            "layers": int(live.sum())}


# ---------------------------------------------------------------------------------------------------------------
# decode
# ---------------------------------------------------------------------------------------------------------------
def make_meter_class():
    """Built lazily so that importing this module does not import da_decode (and torch CUDA state)."""
    from draftalign.da_decode import MoEMeter, MOE_KIND_CODE

    class ExpertLogMoEMeter(MoEMeter):
        KINDS = MOE_KIND_CODE

        def __init__(self, *a, level: str = "bits", n_layers: int = N_LAYERS, n_experts: int = N_EXPERTS, **kw):
            super().__init__(*a, **kw)
            assert level in ("bits", "full")
            self.level, self.L, self.E = level, int(n_layers), int(n_experts)
            self._reset_log()

        def _reset_log(self):
            K = len(self.KINDS)
            self.hit_bits: List[np.ndarray] = []
            self.kinds: List[int] = []
            self.npos: List[int] = []
            self.counts = np.zeros((K, self.L, self.E), np.int64)
            self.mass_sum = np.zeros((K, self.L), np.float64)
            self.mass_n = np.zeros((K, self.L), np.int64)
            self.full_ids: List[np.ndarray] = []
            self.full_probs: List[np.ndarray] = []

        def reset(self):
            super().reset()
            self._reset_log()

        def _record(self, cap, kind: str, npos: int, pool_size: int = 0):
            super()._record(cap, kind, npos, pool_size)
            if cap is None or not cap.ids:
                return
            k = self.KINDS[kind]
            L = len(cap.ids)
            hit = np.zeros((self.L, self.E), bool)
            for li in range(min(L, self.L)):
                ids = cap.ids[li].numpy().astype(np.int64)            # [N, 8]
                bc = np.bincount(ids.ravel(), minlength=self.E)[:self.E]
                self.counts[k, li] += bc
                hit[li] = bc > 0
                if li < len(cap.probs):
                    pm = cap.probs[li].float().sum(-1).numpy()        # top-8 mass per row, pre-renormalisation
                    self.mass_sum[k, li] += float(pm.sum()); self.mass_n[k, li] += int(pm.size)
            self.hit_bits.append(np.packbits(hit, axis=-1))
            self.kinds.append(k)
            self.npos.append(int(npos))
            if self.level == "full":
                self.full_ids.append(np.stack([t.numpy() for t in cap.ids]).astype(np.uint8))        # [L, N, 8]
                self.full_probs.append(np.stack([t.numpy() for t in cap.probs]).astype(np.float16))  # [L, N, 8]

        # -- per item outputs -----------------------------------------------------------------------------------
        def summary(self) -> dict:
            inv = {v: k for k, v in self.KINDS.items()}
            out = {"expert_log_level": self.level}
            gen = self.counts[[self.KINDS[k] for k in ("denoise", "judge", "rejudge") if k in self.KINDS]].sum(0)
            st = load_stats(gen)
            out.update({"exp_block_uniq_per_layer": round(st["uniq_per_layer"], 3),
                        "exp_block_load_entropy": round(st["entropy"], 5) if st["layers"] else None,
                        "exp_block_max_share": round(st["max_share"], 5) if st["layers"] else None})
            per_kind = {}
            for k in range(self.counts.shape[0]):
                if self.counts[k].sum() == 0:
                    continue
                s = load_stats(self.counts[k])
                m = self.mass_sum[k].sum() / max(1, self.mass_n[k].sum())
                per_kind[inv[k]] = {"uniq_per_layer": round(s["uniq_per_layer"], 3),
                                    "entropy": round(s["entropy"], 5), "max_share": round(s["max_share"], 5),
                                    "top8_mass": round(float(m), 5)}
            out["exp_by_kind"] = per_kind
            if self.hit_bits:
                bits = np.unpackbits(np.stack(self.hit_bits), axis=-1)[..., :self.E].astype(bool)   # [F, L, E]
                kinds = np.asarray(self.kinds)
                # per forward kind: how many experts ran per layer IN ONE FORWARD (averaged over that kind's forwards),
                # next to `uniq_per_layer` above, which is the union over the whole item
                for k, name in inv.items():
                    sel = kinds == k
                    if sel.any() and name in per_kind:
                        pf = bits[sel].sum(-1)                                   # [F_k, L]
                        per_kind[name].update({"n_forwards": int(sel.sum()),
                                               "per_fwd_uniq_per_layer": round(float(pf.mean()), 3),
                                               "per_fwd_uniq_per_layer_max": int(pf.max()),
                                               "per_fwd_uniq_total": round(float(pf.sum(-1).mean()), 2)})
                block = np.isin(kinds, [self.KINDS[k] for k in ("denoise", "judge", "rejudge")])
                if block.any():
                    per_fwd = bits[block].sum(-1).mean(-1)                  # experts per layer per block forward
                    out["exp_uniq_per_layer_forward_block"] = round(float(per_fwd.mean()), 3)
                    union = bits[block].any(0).sum(-1)                        # union over the item, per layer
                    out["exp_union_per_layer_item"] = round(float(union.mean()), 3)
                    # the per-layer profile: mean experts hit per block forward, one value per layer (48)
                    out["exp_per_layer_profile_block"] = [round(float(v), 2) for v in bits[block].sum(-1).mean(0)]
                # the 10 most loaded (layer, expert) pairs over the generated-block forwards
                flat = gen.ravel()
                top = np.argsort(-flat)[:10]
                out["exp_top10"] = [[int(t // self.E), int(t % self.E), int(flat[t])] for t in top if flat[t] > 0]
            return out

        def save(self, path: str) -> None:
            if not self.hit_bits:
                return
            os.makedirs(os.path.dirname(path), exist_ok=True)
            extra = {}
            if self.level == "full" and self.full_ids:
                # ragged rows per forward: store flat with offsets
                n = np.array([a.shape[1] for a in self.full_ids], np.int32)
                extra = {"full_ids": np.concatenate(self.full_ids, axis=1), "full_probs": np.concatenate(self.full_probs, axis=1),
                         "full_rows": n}
            np.savez_compressed(path, hit_bits=np.stack(self.hit_bits), kinds=np.asarray(self.kinds, np.int8),
                                npos=np.asarray(self.npos, np.int32), counts=self.counts.astype(np.int32),
                                mass_sum=self.mass_sum, mass_n=self.mass_n,
                                kind_names=np.array(list(self.KINDS.keys())), **extra)

    return ExpertLogMoEMeter


# ---------------------------------------------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------------------------------------------
class TrainExpertCapture:
    """Counts routed experts per MoE layer during the training forward (prefix pass and block pass separately)."""

    PHASES = ("prefix", "block")

    def __init__(self, model, device, n_experts: int = N_EXPERTS, top_k: int = TOP_K):
        from draftalign import adapter as A
        self.layers = A.decoder_layers(model)
        self.index = {id(l.mlp): i for i, l in enumerate(self.layers)}
        self.E, self.k = int(n_experts), int(top_k)
        self.counts = torch.zeros(len(self.PHASES), len(self.layers), self.E, dtype=torch.float64, device=device)
        self.armed = False
        self.phase = "block"
        self._installed = False

    def install(self):
        from draftalign import train_fast as F
        if self._installed:
            return
        orig_moe, orig_prefix, orig_block = F.moe_forward, F.prefix_kv, F.block_hidden
        cap = self

        def moe_forward(blk, hidden_states, *a, **kw):
            if cap.armed and id(blk) in cap.index:
                with torch.no_grad():
                    h = hidden_states.reshape(-1, hidden_states.shape[-1])
                    w = torch.softmax(blk.gate(h).float(), dim=-1)
                    sel = torch.topk(w, cap.k, dim=-1).indices.reshape(-1)
                    cap.counts[cap.PHASES.index(cap.phase), cap.index[id(blk)]] += torch.bincount(
                        sel, minlength=cap.E)[:cap.E].double()
            return orig_moe(blk, hidden_states, *a, **kw)

        def prefix_kv(*a, **kw):
            old, cap.phase = cap.phase, "prefix"
            try:
                return orig_prefix(*a, **kw)
            finally:
                cap.phase = old

        def block_hidden(*a, **kw):
            old, cap.phase = cap.phase, "block"
            try:
                return orig_block(*a, **kw)
            finally:
                cap.phase = old

        F.moe_forward, F.prefix_kv, F.block_hidden = moe_forward, prefix_kv, block_hidden
        self._orig = (orig_moe, orig_prefix, orig_block)
        self._installed = True

    def uninstall(self):
        if self._installed:
            from draftalign import train_fast as F
            F.moe_forward, F.prefix_kv, F.block_hidden = self._orig
            self._installed = False

    @contextmanager
    def armed_scope(self):
        self.armed = True
        try:
            yield
        finally:
            self.armed = False

    def window(self, world: int = 1) -> Dict[str, np.ndarray]:
        """All-reduce and return this window's counts per phase ([L, E] int64), then zero them."""
        c = self.counts.clone()
        if world > 1:
            import torch.distributed as dist
            dist.all_reduce(c)
        self.counts.zero_()
        arr = c.cpu().numpy().astype(np.int64)
        return {ph: arr[i] for i, ph in enumerate(self.PHASES)}

    @staticmethod
    def metrics(win: Dict[str, np.ndarray]) -> Dict[str, float]:
        out = {}
        for ph, c in win.items():
            s = load_stats(c)
            out.update({f"experts/{ph}_uniq_per_layer": s["uniq_per_layer"], f"experts/{ph}_entropy": s["entropy"],
                        f"experts/{ph}_max_share": s["max_share"]})
        return out
