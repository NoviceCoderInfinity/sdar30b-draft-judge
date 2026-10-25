"""The SGLang-side judge adapter (code/corpus1b/sglang_plugin/c1b_judge_lora.py) computes BIT-IDENTICALLY what the
HF harness adapter (code/draftalign/adapter.py: LoRALinear + DraftEmbed) computes, on CPU fakes of both layouts:

  HF-like:     model.model.layers[i].self_attn.{q,k,v,o}_proj = nn.Linear, inputs [B, S, H], flag [B, S]
  SGLang-like: model.model.layers[i].self_attn.qkv_proj (fused, returns (out, bias)) + o_proj (returns (out, bias)),
               inputs [num_tokens, H], flag [num_tokens]

The adapter goes through the real checkpoint path: adapter.attach_lora -> random A/B/draft vector -> save_adapter ->
attach_judge_adapter. fp32 and bf16.

    python3 -m pytest -q code/corpus1b/tests/test_judge_lora_sgl.py
"""
from __future__ import annotations

import copy
import json
import os
import sys
import types

import pytest
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.abspath(os.path.join(HERE, "..", ".."))
for p in (os.path.join(CODE, "corpus1b", "sglang_plugin"), CODE):
    if p not in sys.path:
        sys.path.insert(0, p)

from draftalign import adapter as A                                       # noqa: E402
import c1b_judge_lora as J                                                # noqa: E402

H, NH, NKV, HD, NL, VOCAB, R, ALPHA = 64, 4, 2, 16, 2, 50, 4, 8.0
NTOK = 23


def _cfg():
    return types.SimpleNamespace(hidden_size=H, num_hidden_layers=NL, vocab_size=VOCAB,
                                 num_attention_heads=NH, num_key_value_heads=NKV, head_dim=HD)


# ---------------------------------------------------------------------------
# the HF-like fake (the harness layout)
# ---------------------------------------------------------------------------
class HFAttn(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(H, NH * HD, bias=False)
        self.k_proj = nn.Linear(H, NKV * HD, bias=False)
        self.v_proj = nn.Linear(H, NKV * HD, bias=False)
        self.o_proj = nn.Linear(NH * HD, H, bias=False)


class HFLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = HFAttn()                    # no mlp: adapter._moe_gate tolerates it


class Inner(nn.Module):
    def __init__(self, embed, layers):
        super().__init__()
        self.embed_tokens = embed
        self.layers = nn.ModuleList(layers)


class Outer(nn.Module):
    def __init__(self, inner):
        super().__init__()
        self.config = _cfg()
        self.model = inner


def build_hf(seed: int = 0) -> Outer:
    torch.manual_seed(seed)
    m = Outer(Inner(nn.Embedding(VOCAB, H), [HFLayer() for _ in range(NL)]))
    A.attach_lora(m, r=R, alpha=ALPHA, draft_flag=True, train_gates=False)
    g = torch.Generator().manual_seed(seed + 7)
    with torch.no_grad():                            # a NON-identity adapter (B != 0), or the test proves nothing
        for mod in m.modules():
            if isinstance(mod, A.LoRALinear):
                mod.lora_A.copy_(torch.randn(mod.lora_A.shape, generator=g) * 0.3)
                mod.lora_B.copy_(torch.randn(mod.lora_B.shape, generator=g) * 0.3)
        m.da_draft_embed.weight.copy_(torch.randn(H, generator=g) * 0.5)
    return m


# ---------------------------------------------------------------------------
# the SGLang-like fake (QKVParallelLinear / RowParallelLinear return (out, bias))
# ---------------------------------------------------------------------------
class UnquantizedLinearMethod:                       # name as sglang.srt.layers.quantization.unquant
    pass


class Fp8LinearMethod:
    pass


class FakeQKV(nn.Module):
    """Three separate base matmuls concatenated, so the BASE part is bit-identical to the HF side too."""

    def __init__(self, q, k, v):
        super().__init__()
        self.q, self.k, self.v = q, k, v
        self.tp_size = 1
        self.quant_method = UnquantizedLinearMethod()
        self.input_size = q.in_features
        self.output_sizes = [q.out_features, k.out_features, v.out_features]

    def forward(self, x):
        return torch.cat([self.q(x), self.k(x), self.v(x)], dim=-1), None


class FakeRow(nn.Module):
    def __init__(self, o):
        super().__init__()
        self.o = o
        self.tp_size = 1
        self.quant_method = UnquantizedLinearMethod()
        self.input_size, self.output_size = o.in_features, o.out_features

    def forward(self, input_, skip_all_reduce=False):
        return self.o(input_), None


class SGLAttn(nn.Module):
    def __init__(self, qkv, o):
        super().__init__()
        self.qkv_proj, self.o_proj = qkv, o


def build_sgl(hf: Outer) -> Outer:
    """Same base weights as `hf` (deep copies of the frozen bases), no adapter."""
    layers = []
    for layer in hf.model.layers:
        at = layer.self_attn
        base = [copy.deepcopy(getattr(at, n).base) for n in A.ATTN_TARGETS]
        lay = nn.Module()
        lay.self_attn = SGLAttn(FakeQKV(*base[:3]), FakeRow(base[3]))
        layers.append(lay)
    return Outer(Inner(copy.deepcopy(hf.model.embed_tokens), layers))


def to_base_dtype(hf: Outer, dtype) -> None:
    """Cast only the frozen base (embedding + projections); A/B/draft vector stay fp32 as at inference."""
    hf.model.embed_tokens.to(dtype)
    for mod in hf.modules():
        if isinstance(mod, A.LoRALinear):
            mod.base.to(dtype)


@pytest.fixture(params=[torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
def pair(request, tmp_path):
    dtype = request.param
    hf = build_hf(0)
    to_base_dtype(hf, dtype)
    d = A.save_adapter(hf, str(tmp_path / "adapter"))
    sgl = build_sgl(hf)
    info = J.attach_judge_adapter(sgl, d)
    assert info["n_tensors"] == NL * 8 + 1 and info["r"] == R and info["scaling"] == ALPHA / R
    with A.flag_scope(None), J.JudgeCtx.flag_scope(None):
        yield hf, sgl, dtype


def _flag(seed=1):
    f = torch.rand(NTOK, generator=torch.Generator().manual_seed(seed)) < 0.5
    assert 0 < int(f.sum()) < NTOK
    return f


def _x(width, dtype, seed=2):
    return torch.randn(NTOK, width, generator=torch.Generator().manual_seed(seed)).to(dtype)


def hf_qkv(at, x):
    return torch.cat([at.q_proj(x[None]), at.k_proj(x[None]), at.v_proj(x[None])], dim=-1)[0]


def hf_qkv_base(at, x):
    return torch.cat([at.q_proj.base(x), at.k_proj.base(x), at.v_proj.base(x)], dim=-1)


# ---------------------------------------------------------------------------
# (a) / (b): a mixed flag -> bit-identical, delta only on flagged tokens
# ---------------------------------------------------------------------------
def test_qkv_and_o_match_hf_under_flag(pair):
    hf, sgl, dtype = pair
    flag = _flag()
    for i in range(NL):
        at_h, at_s = hf.model.layers[i].self_attn, sgl.model.layers[i].self_attn
        x, xo = _x(H, dtype, 10 + i), _x(NH * HD, dtype, 20 + i)
        with A.flag_scope(flag[None]):
            ref_qkv, ref_o = hf_qkv(at_h, x), at_h.o_proj(xo[None])[0]
        with J.JudgeCtx.flag_scope(flag):
            out_qkv, bias = at_s.qkv_proj(x)
            out_o, bias_o = at_s.o_proj(xo)
        assert bias is None and bias_o is None
        assert out_qkv.dtype == dtype and out_o.dtype == dtype
        assert torch.equal(out_qkv, ref_qkv) and torch.equal(out_o, ref_o)
        base_qkv, base_o = hf_qkv_base(at_h, x), at_h.o_proj.base(xo)
        assert torch.equal(out_qkv[~flag], base_qkv[~flag]) and torch.equal(out_o[~flag], base_o[~flag])
        assert not torch.equal(out_qkv[flag], base_qkv[flag]) and not torch.equal(out_o[flag], base_o[flag])


# (c) all-False flag -> the frozen base, delta path never entered
def test_all_false_flag_is_base_and_skips_delta(pair, monkeypatch):
    hf, sgl, dtype = pair
    flag = torch.zeros(NTOK, dtype=torch.bool)

    def boom(*a, **k):
        raise AssertionError("the delta was computed under an all-False flag")
    monkeypatch.setattr(J, "_lora_delta", boom)
    for i in range(NL):
        at_h, at_s = hf.model.layers[i].self_attn, sgl.model.layers[i].self_attn
        x, xo = _x(H, dtype, 30 + i), _x(NH * HD, dtype, 40 + i)
        with J.JudgeCtx.flag_scope(flag):
            assert J.JudgeCtx.any is False
            out_qkv, _ = at_s.qkv_proj(x)
            out_o, _ = at_s.o_proj(xo)
        assert torch.equal(out_qkv, at_s.qkv_proj._judge_orig_forward(x)[0])
        assert torch.equal(out_qkv, hf_qkv_base(at_h, x)) and torch.equal(out_o, at_h.o_proj.base(xo))
        with A.flag_scope(flag[None]):
            assert torch.equal(out_qkv, hf_qkv(at_h, x)) and torch.equal(out_o, at_h.o_proj(xo[None])[0])


# (d) flag None -> ungated delta everywhere, as the harness with CTX.flag None
def test_no_flag_is_ungated_everywhere(pair):
    hf, sgl, dtype = pair
    for i in range(NL):
        at_h, at_s = hf.model.layers[i].self_attn, sgl.model.layers[i].self_attn
        x, xo = _x(H, dtype, 50 + i), _x(NH * HD, dtype, 60 + i)
        out_qkv, out_o = at_s.qkv_proj(x)[0], at_s.o_proj(xo)[0]
        assert torch.equal(out_qkv, hf_qkv(at_h, x)) and torch.equal(out_o, at_h.o_proj(xo[None])[0])
        assert (out_qkv != hf_qkv_base(at_h, x)).any(dim=-1).all()     # every token got a delta


# (e) the draft-flag embedding == DraftEmbed.apply_to; judge_forward_prep's None path
def test_draft_embedding_matches_hf(pair):
    hf, sgl, dtype = pair
    ids = torch.randint(0, VOCAB, (NTOK,), generator=torch.Generator().manual_seed(3))
    flag = _flag(4)
    ref = hf.da_draft_embed.apply_to(hf.model.embed_tokens(ids[None]), flag[None])[0]
    got = J.embed_with_draft_flag(sgl, ids, flag)
    assert got.dtype == dtype and torch.equal(got, ref)
    plain = sgl.model.embed_tokens(ids)
    assert torch.equal(got[~flag], plain[~flag]) and not torch.equal(got[flag], plain[flag])

    emb, any_ = J.judge_forward_prep(sgl, ids, flag)
    assert any_ is True and torch.equal(emb, ref)
    assert J.judge_forward_prep(sgl, ids, torch.zeros(NTOK, dtype=torch.bool)) == (None, False)
    assert J.judge_forward_prep(sgl, ids, None) == (None, False)
    assert torch.equal(J.embed_with_draft_flag(sgl, ids, torch.zeros(NTOK, dtype=torch.bool)), plain)


# (f) a second attach is refused
def test_second_attach_refused(pair, tmp_path):
    hf, sgl, _ = pair
    d = A.save_adapter(hf, str(tmp_path / "again"))
    with pytest.raises(AssertionError, match="already attached"):
        J.attach_judge_adapter(sgl, d)


def test_flag_length_mismatch_refused(pair):
    _, sgl, dtype = pair
    with J.JudgeCtx.flag_scope(_flag()[:-1]):
        with pytest.raises(AssertionError, match="does not match"):
            sgl.model.layers[0].self_attn.qkv_proj(_x(H, dtype))


def test_flag_scope_restores_on_exception():
    f = _flag()
    with J.JudgeCtx.flag_scope(f):
        with pytest.raises(RuntimeError):
            with J.JudgeCtx.flag_scope(torch.zeros(NTOK, dtype=torch.bool)):
                assert J.JudgeCtx.any is False
                raise RuntimeError
        assert J.JudgeCtx.flag is f and J.JudgeCtx.any is True
    assert J.JudgeCtx.flag is None and J.JudgeCtx.any is False
    with pytest.raises(AssertionError):
        J.set_flag(f[None])                          # the flattened layout is 1-D


@pytest.mark.parametrize("bad", ["tp", "quant"])
def test_refuses_tp_and_quantized_without_patching(tmp_path, bad):
    hf = build_hf(0)
    d = A.save_adapter(hf, str(tmp_path / "adapter"))
    sgl = build_sgl(hf)
    last = sgl.model.layers[NL - 1].self_attn.o_proj
    if bad == "tp":
        last.tp_size = 2
    else:
        last.quant_method = Fp8LinearMethod()
    with pytest.raises(AssertionError):
        J.attach_judge_adapter(sgl, d)
    first = sgl.model.layers[0].self_attn.qkv_proj
    assert "forward" not in vars(first) and getattr(sgl, "_judge_adapter", None) is None


# ---------------------------------------------------------------------------
# the real checkpoint's key names and config
# ---------------------------------------------------------------------------
def test_real_key_names(tmp_path):
    keys = J.expected_keys(48)
    assert len(keys) == 385 and keys[-1] == "da_draft_embed.weight"
    for k in ("model.layers.0.self_attn.q_proj.lora_A", "model.layers.47.self_attn.o_proj.lora_B",
              "model.layers.13.self_attn.k_proj.lora_B", "model.layers.13.self_attn.v_proj.lora_A"):
        assert k in keys

    from safetensors.torch import save_file
    g = torch.Generator().manual_seed(5)
    shapes = {"q": (NH * HD, H), "k": (NKV * HD, H), "v": (NKV * HD, H), "o": (H, NH * HD)}
    sd = {}
    for i in range(NL):
        for p, (out, inp) in shapes.items():
            sd[f"model.layers.{i}.self_attn.{p}_proj.lora_A"] = torch.randn(R, inp, generator=g)
            sd[f"model.layers.{i}.self_attn.{p}_proj.lora_B"] = torch.randn(out, R, generator=g)
    sd["da_draft_embed.weight"] = torch.randn(H, generator=g)
    d = tmp_path / "real_like"
    d.mkdir()
    save_file(sd, str(d / "adapter.safetensors"))
    (d / "adapter_config.json").write_text(json.dumps({
        "alpha": ALPHA, "r": R, "targets": ["q_proj", "k_proj", "v_proj", "o_proj"], "draft_embed": True,
        "train_gates": False, "dropout": 0.0, "model": {"hidden_size": H, "num_hidden_layers": NL}}))
    sgl = build_sgl(build_hf(0))
    info = J.attach_judge_adapter(sgl, str(d))
    assert info["n_layers"] == NL and info["n_tensors"] == len(sd)
    at = sgl.model.layers[1].self_attn
    assert torch.equal(at.qkv_proj.lora_B_k, sd["model.layers.1.self_attn.k_proj.lora_B"])
    assert torch.equal(at.o_proj.lora_A_o, sd["model.layers.1.self_attn.o_proj.lora_A"])
    assert torch.equal(sgl.da_draft_embed_weight, sd["da_draft_embed.weight"])
    assert "lora_A_q" in dict(at.qkv_proj.named_buffers()) and "lora_A_q" not in at.qkv_proj.state_dict()

    del sd["model.layers.1.self_attn.v_proj.lora_A"]                   # a missing key is refused
    save_file(sd, str(d / "adapter.safetensors"))
    with pytest.raises(AssertionError, match="missing"):
        J.attach_judge_adapter(build_sgl(build_hf(0)), str(d))
