"""CPU test for plot_p2.py: tiny synthetic Phase 2 outputs in tmp_path -> every plot is a PNG (no W&B).

    CUDA_VISIBLE_DEVICES= /venv/p2/bin/python -m pytest -q code/phase2/tests/test_plot_p2.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = ""
HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.abspath(os.path.join(HERE, "..", ".."))
for p in (os.path.join(CODE, "compat"), CODE):
    if p not in sys.path:
        sys.path.insert(0, p)

from phase2 import plot_p2                                                # noqa: E402

SET = "gsm8k500"
L, E = plot_p2.N_LAYERS, plot_p2.N_EXPERTS
KIND_NAMES = np.array(["prefill", "denoise", "kvflush", "judge", "rejudge"])


def _rec(i, rng, drafter=True):
    nb = 3
    r = {"idx": i, "correct": int(rng.random() < 0.7), "n_gen_to_eos": 80, "unique_expert_layer_loads": 40000,
         "n_forward": 1 + nb + 9, "n_prefill": 1, "n_flush": nb, "n_blocks": nb, "seconds": 3.0}
    if drafter:
        r.update(drafter_n_by_kind={"prefill": 1, "draft": 24, "redraft": 0, "flush": 3},
                 draft_agreement_per_block=[0.8, 0.5, 0.9], judge_p_mean_by_pass=[0.9, 0.95, 0.97],
                 judge_commit_frac_by_pass=[0.6, 0.8, 0.9])
    return r


def _write_cell(d, name, n, rng, drafter=True, npz=True, running=True):
    os.makedirs(d, exist_ok=True)
    recs = [_rec(i, rng, drafter) for i in range(n)]
    with open(os.path.join(d, f"{name}.{SET}.shard0of1.jsonl"), "w") as fh:
        for r in recs:
            fh.write(json.dumps(r) + "\n")
        fh.write('{"idx": 99, "partial')                                 # a line being written
    if running:
        with open(os.path.join(d, f"{name}.{SET}.shard0of1.jsonl.running.jsonl"), "w") as fh:
            for k in range(1, n + 1):
                row = {"idx": k - 1, "items": k, "accuracy": float(np.mean([r["correct"] for r in recs[:k]]))}
                if k >= 2:
                    row.update(delta_acc_vs_s0=0.01 * k, delta_acc_ci95_lo=-0.1, delta_acc_ci95_hi=0.1 + 0.01 * k)
                fh.write(json.dumps(row) + "\n")
    if not npz:
        return
    for i in range(n):
        F = 12
        kinds = np.array([0] + [3] * 8 + [2] * 3, np.int8)
        hit = rng.random((F, L, E)) < 0.4
        counts = rng.integers(0, 5, (5, L, E)).astype(np.int32)
        p = os.path.join(d, "experts", SET, f"{i:05d}.npz")
        os.makedirs(os.path.dirname(p), exist_ok=True)
        np.savez_compressed(p, hit_bits=np.packbits(hit, axis=-1), kinds=kinds, counts=counts,
                            mass_sum=rng.random((5, L)) * 100, mass_n=np.full((5, L), 300), kind_names=KIND_NAMES)
        if drafter:
            p = os.path.join(d, "judge", SET, f"{i:05d}.npz")
            os.makedirs(os.path.dirname(p), exist_ok=True)
            np.savez_compressed(p, judge_pass=np.array([0, 1, 2, 0, 1, 0], np.int16),
                                judge_p=rng.random((6, 32)).astype(np.float16), judge_open=rng.random((6, 32)) < 0.8)
            p = os.path.join(d, "traces", SET, f"{i:05d}.npz")
            os.makedirs(os.path.dirname(p), exist_ok=True)
            tok = rng.integers(0, 50, (3, 32)).astype(np.int32)
            dt = np.where(rng.random((3, 32)) < 0.6, tok, tok + 1).astype(np.int32)
            np.savez_compressed(p, draft_tokens=dt, p_draft=rng.random((3, 32)).astype(np.float32), tokens=tok,
                                valid=np.ones((3, 32), bool))


def _build(tmp_path):
    rng = np.random.default_rng(0)
    dec = tmp_path / "decode"
    _write_cell(str(dec / "S0" / SET), "gsm8k_B32_t95_dv-s8_full", 6, rng)
    _write_cell(str(dec / "F2" / SET), "gsm8k_B32_t95_dv-s8_full_drF2abcdef", 6, rng)
    _write_cell(str(dec / "S1" / SET), "gsm8k_B32_t95_dv-s8_full_drS1abcdef", 4, rng, npz=False, running=False)
    stock = tmp_path / "stock_flat"
    _write_cell(str(stock), "gsm8k_B32_t95_stock", 6, rng, drafter=False, npz=False, running=False)
    rd = tmp_path / "redraft" / "F2"
    rd.mkdir(parents=True)
    (rd / "summary.json").write_text(json.dumps({"per_domain": {
        "code": {"agreement": 0.55, "prev_agreement": 0.7}, "math": {"agreement": 0.6, "prev_agreement": 0.72},
        "all": {"agreement": 0.58, "prev_agreement": 0.71}}}))
    rs = tmp_path / "redraft" / "S1"                                      # not merged yet: agreement shards only
    rs.mkdir(parents=True)
    (rs / "agreement.shard0of1.jsonl").write_text(
        json.dumps({"domain": "code", "slots": 100, "new_ok": 40, "prev_ok": 70}) + "\n{\"domain\": ")
    pr = tmp_path / "probe" / "F2__r3_B"
    pr.mkdir(parents=True)
    probe = {}
    for d in ("code", "math", "all"):
        probe.update({f"{d}/keep_rate": 0.97, f"{d}/fix_rate": 0.4, f"{d}/conf_precision": 0.992, f"{d}/ece": 0.03})
    (pr / "probe.json").write_text(json.dumps(probe))
    return ["--decode-root", str(dec), "--s0-ref-root", str(tmp_path / "none"), "--s0-flat", str(tmp_path / "none"),
            "--stock-ref-root", str(tmp_path / "none"), "--stock-flat", str(stock),
            "--redraft-root", str(tmp_path / "redraft"), "--probe-root", str(tmp_path / "probe"),
            "--report-root", str(tmp_path / "report"), "--sets", SET, "--max-items", "4",
            "--out", str(tmp_path / "plots"), "--no-wandb", "--once"]


def test_plots_from_synthetic_inputs(tmp_path):
    assert plot_p2.main(_build(tmp_path)) == 0
    out = tmp_path / "plots"
    for rel in ("F2/gsm8k500__running.png", "F2/gsm8k500__experts_profile.png", "F2/gsm8k500__expert_heatmap.png",
                "F2/gsm8k500__router_mass.png", "F2/gsm8k500__judge_by_pass.png",
                "F2/gsm8k500__drafter_reliability.png", "F2/gsm8k500__passes_speedup.png", "F2/redraft_agreement.png",
                "F2/probe.png", "S0/gsm8k500__running.png", "S1/gsm8k500__passes_speedup.png",
                "S1/redraft_agreement.png", "compare/gsm8k500__arms.png", "compare/redraft_agreement.png"):
        f = out / rel
        assert f.exists() and f.stat().st_size > 1000, rel
    summ = json.loads((out / "summary.json").read_text())
    cmp_ = summ["sets"][SET]["compare"]
    assert set(cmp_) == {"S0", "F2", "S1"}
    assert cmp_["F2"]["n"] == 6 and cmp_["F2"].get("speedup") and "delta" in cmp_["F2"]
    assert any(s[0] == "S1/gsm8k500" and s[1] == "running" for s in summ["skipped"])


def test_nothing_there_is_skipped_not_fatal(tmp_path):
    args = ["--decode-root", str(tmp_path / "d"), "--s0-ref-root", str(tmp_path / "d"), "--s0-flat", str(tmp_path / "d"),
            "--stock-ref-root", str(tmp_path / "d"), "--stock-flat", str(tmp_path / "d"),
            "--redraft-root", str(tmp_path / "d"), "--probe-root", str(tmp_path / "d"),
            "--report-root", str(tmp_path / "d"), "--out", str(tmp_path / "plots"), "--no-wandb"]
    assert plot_p2.main(args) == 0
    summ = json.loads((tmp_path / "plots" / "summary.json").read_text())
    assert summ["made"] == [] and summ["skipped"]


def test_loop_exits_on_stop_file(tmp_path):
    (tmp_path / "STOP").write_text("")
    assert plot_p2.main(["--loop", "1", "--stop-file", str(tmp_path / "STOP"), "--out", str(tmp_path / "p"),
                         "--no-wandb"]) == 0
