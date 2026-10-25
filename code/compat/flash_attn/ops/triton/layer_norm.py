"""Pure-PyTorch stand-in for flash_attn.ops.triton.layer_norm.rms_norm_fn.

Matches the Triton kernel's semantics for the call SDAR actually makes:
    out = (x * rsqrt(mean(x^2) + eps)) * weight [+ bias]
with the reduction in fp32 and the result cast back to x's dtype.
"""
import torch

IS_SDAR_COMPAT_SHIM = True


def rms_norm_fn(
    x,
    weight,
    bias=None,
    residual=None,
    x1=None,
    weight1=None,
    bias1=None,
    eps=1e-6,
    dropout_p=0.0,
    rowscale=None,
    prenorm=False,
    residual_in_fp32=False,
    is_rms_norm=True,
    return_dropout_mask=False,
    out=None,
    residual_out=None,
):
    # Refuse every path we have not verified, so a future caller cannot get a
    # silently-wrong answer out of this shim.
    if residual is not None or x1 is not None or weight1 is not None \
            or bias1 is not None or rowscale is not None:
        raise NotImplementedError(
            "sdar compat flash_attn shim: fused residual/parallel paths are not "
            "implemented (SDAR does not use them)."
        )
    if dropout_p != 0.0 or prenorm or return_dropout_mask:
        raise NotImplementedError(
            "sdar compat flash_attn shim: dropout/prenorm/dropout-mask are not "
            "implemented (SDAR does not use them)."
        )
    if not is_rms_norm:
        raise NotImplementedError(
            "sdar compat flash_attn shim: only RMSNorm is implemented."
        )
    if out is not None or residual_out is not None:
        raise NotImplementedError(
            "sdar compat flash_attn shim: preallocated out= is not implemented."
        )

    in_dtype = x.dtype
    xf = x.float()
    var = xf.pow(2).mean(-1, keepdim=True)
    xf = xf * torch.rsqrt(var + eps)
    y = xf.to(in_dtype) * weight
    if bias is not None:
        y = y + bias
    return y


# The model imports only rms_norm_fn; expose the usual companion name defensively.
def layer_norm_fn(*args, **kwargs):
    raise NotImplementedError(
        "sdar compat flash_attn shim: layer_norm_fn is not implemented."
    )
