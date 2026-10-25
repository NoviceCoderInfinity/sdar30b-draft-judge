"""SHIM -- not the real flash-attn.

Why this exists
---------------
`JetLM/SDAR-30B-A3B-Chat-b32/modeling_sdar_moe.py` has an *unguarded* top-level
import:

    from flash_attn.ops.triton.layer_norm import rms_norm_fn as flash_rms_norm

so the module cannot even be imported without `flash_attn` present -- regardless
of `attn_implementation`. `SDARMoeRMSNorm.forward` then routes every RMSNorm
through that Triton kernel.

Building real flash-attn here is not viable: it is a CUDA extension, sm_120
(Blackwell) support in FA2 is not established, and the build consumed ~460 GB of
RAM and was still going after 50 min on this box.

We do NOT use flash attention itself -- the harness loads the model with
`attn_implementation='sdpa'`. The ONLY symbol the model
needs is `rms_norm_fn`, which this shim provides in pure PyTorch, computing the
reduction in fp32 exactly as the Triton kernel does.

Anything else raises rather than silently diverging.
"""

__version__ = "0.0.0+sdar-compat-shim"
IS_SDAR_COMPAT_SHIM = True


# ---------------------------------------------------------------------------
# flash_attn_func
# ---------------------------------------------------------------------------
# `modeling_sdar.py` (the 1.7B DENSE drafter) dispatches on mask CONTENT, not on
# attn_implementation:
#
#     if torch.all(attention_mask):        # decode block: attends to everything
#         attn_output = flash_attn_func(q, k, v, causal=False, softmax_scale=...)
#     else:                                # prefill: block-causal mask
#         attn_output = F.scaled_dot_product_attention(..., attn_mask=..., is_causal=False)
#
# The flash call is unmasked, non-causal, full-precision softmax attention -- i.e.
# exactly `scaled_dot_product_attention(attn_mask=None, is_causal=False)`. FA2 and
# SDPA compute the same function; they differ only in accumulation order.
#
# Note the 30B MoE (`modeling_sdar_moe.py`) already uses SDPA for BOTH branches,
# so this makes the drafter consistent with the MoE's own attention path.
import torch
import torch.nn.functional as F


def flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False,
                    window_size=(-1, -1), softcap=0.0, alibi_slopes=None,
                    deterministic=False, return_attn_probs=False):
    """q, k, v: [batch, seqlen, nheads, headdim] -> [batch, seqlen, nheads, headdim]."""
    if dropout_p != 0.0:
        raise NotImplementedError("sdar compat flash_attn shim: dropout_p != 0 unsupported.")
    if tuple(window_size) != (-1, -1):
        raise NotImplementedError("sdar compat flash_attn shim: sliding window unsupported.")
    if softcap not in (0.0, None) or alibi_slopes is not None:
        raise NotImplementedError("sdar compat flash_attn shim: softcap/alibi unsupported.")
    if return_attn_probs:
        raise NotImplementedError("sdar compat flash_attn shim: return_attn_probs unsupported.")

    q_ = q.transpose(1, 2)   # [B, H, Sq, D]
    k_ = k.transpose(1, 2)
    v_ = v.transpose(1, 2)
    out = F.scaled_dot_product_attention(
        q_, k_, v_, attn_mask=None, dropout_p=0.0,
        is_causal=causal, scale=softmax_scale,
        enable_gqa=(q_.shape[1] != k_.shape[1]),
    )
    return out.transpose(1, 2).contiguous()   # back to [B, Sq, H, D]


def flash_attn_varlen_func(*args, **kwargs):
    raise NotImplementedError(
        "sdar compat flash_attn shim: flash_attn_varlen_func is not implemented "
        "(SDAR's decode path does not use it)."
    )


def __getattr__(name):
    raise ImportError(
        f"flash_attn.{name} is not available: this is the Stage 0 pure-PyTorch "
        f"shim that provides only flash_attn.ops.triton.layer_norm.rms_norm_fn. "
        f"Load models with attn_implementation='sdpa'."
    )
