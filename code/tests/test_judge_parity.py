"""CPU tests for code/judge_round/judge_parity.py (SGLang judge decode vs the HF harness decode, joined on (split, idx)).
Synthetic record files in tmp_path, in both record formats: the harness's (run_da_decode / da_decode.make_record field names)
and the SGLang orchestrator's (judge_sglang.py --mode decode).

Pins: the loaders (dir / glob / file / gz, `.running.` and non-record lines skipped, a later duplicate wins); the join on
(split, idx) and --ignore-split; the cut at the first EOS on both sides (the harness's post-EOS filler never counts); exact,
prefix and token agreement and |length difference| on identical / one-divergence / length-mismatch / strict-prefix / empty
cases; passes per block as sum / sum (not a mean of ratios) and its relative gap; drafter forwards per block (harness:
drafter_n_by_kind draft + redraft, prefill and flush excluded; sglang: n_draft_fwd + n_redraft_fwd); accuracy of the SGLang
pred scored with the harness gold, the paired gap and the discordant counts; the loop rule equals compare_gates.loops;
accuracy null when the harness carries no correctness (not a failure unless --require-accuracy); every verdict threshold at,
below and above its limit with the failing criterion named in the message; exit codes 0 / 3 and the JSON report.

    python3 -m pytest -q code/tests/test_judge_parity.py
"""
from __future__ import annotations

import gzip
import json
import os
import sys

import pytest

CODE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for p in (os.path.join(CODE, "compat"), CODE):
    if p not in sys.path:
        sys.path.insert(0, p)

from judge_round import judge_parity as JP                                 # noqa: E402

EOS = 151645
EOS2 = 151643
SPLIT = "gsm8k500"


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------
def h_rec(idx, ids, split=SPLIT, passes=(5, 4), draft=3, redraft=2, correct=1, gold="42", pred="x", filler=2,
          fallback=(0, 0), extra=None):
    """A harness record: gen_ids = `ids` + `filler` post-EOS filler tokens (the last block is always filled), passes_per_block
    per block, drafter_n_by_kind with prefill / flush that must NOT count as drafter forwards per block."""
    r = {"idx": idx, "split": split, "gen_ids": list(ids) + [EOS] * filler, "n_gen_to_eos": len(ids), "pred": pred, "gold": gold,
         "n_blocks": len(passes), "passes_per_block": list(passes), "n_judge": sum(passes),
         "drafter_n_by_kind": {"prefill": 1, "draft": draft, "redraft": redraft, "flush": 2},
         "drafter_forwards_per_block": [draft] + [0] * (len(passes) - 1),
         "redraft_forwards_per_block": [redraft] + [0] * (len(passes) - 1),
         "redraft_rounds_per_block": [1] * len(passes), "redraft_slots_per_block": [4] * len(passes),
         "loopguard_slots_per_block": [2] * len(passes), "loopguard_changed_per_block": [1] * len(passes),
         "full_fallback_per_block": list(fallback), "forwards": [[0, 1.0]] * 3}
    if correct is not None:
        r["correct"] = correct
    r.update(extra or {})
    return r


def s_rec(idx, ids, split=SPLIT, passes=(5, 4), draft=3, redraft=2, pred="x", fallback=False, extra=None):
    """An SGLang record (judge_sglang.py --mode decode): gen_ids trimmed at the first EOS, scalar counters."""
    r = {"idx": idx, "split": split, "rid": f"r{idx}", "prompt_tokens": 50, "gen_ids": list(ids), "pred": pred,
         "n_blocks": len(passes), "n_judge": sum(passes), "n_draft_fwd": draft, "n_redraft_rounds": len(passes),
         "n_redraft_slots": 4 * len(passes), "n_redraft_fwd": redraft, "n_loopguard_slots": 2 * len(passes),
         "n_loopguard_changed": len(passes), "full_fallback": fallback, "passes_per_block": list(passes),
         "stopped_eos": True, "engine": "sglang", "cfg": {"loop_guard": 12}}
    r.update(extra or {})
    return r


def put(path, recs, gz=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    op = gzip.open if gz else open
    with op(path, "wt") as fh:
        for r in recs:
            fh.write(json.dumps(r) + "\n")
    return path


def run(tmp_path, sg, hr, *flags, scorer=None, monkeypatch=None):
    """Write both sides, run main(), return (rc, report dict)."""
    put(tmp_path / "sg" / "gpu0" / "records.jsonl", sg)
    put(tmp_path / "h" / "cell.gsm8k500.shard0of1.jsonl", hr)
    if monkeypatch is not None:
        monkeypatch.setattr(JP, "make_scorer", (lambda: scorer) if scorer else (lambda: (_ for _ in ()).throw(ImportError("no deps"))))
    out = tmp_path / "report.json"
    rc = JP.main(["--sglang", str(tmp_path / "sg"), "--harness", str(tmp_path / "h"), "--out", str(out), *flags])
    return rc, json.load(open(out))


@pytest.fixture(autouse=True)
def no_real_scorer(monkeypatch):
    """The real scorer (analyze_s1b -> opencompass, datasets) belongs to /venv/p2: unavailable here unless a test injects one."""
    def unavailable():
        raise ImportError("no scoring deps in the test")
    monkeypatch.setattr(JP, "make_scorer", unavailable)


def seq(n, base=100):
    """n distinct non-EOS tokens then EOS."""
    return list(range(base, base + n)) + [EOS]


def gold_scorer(ds, row):
    """A fake per_item_correct: 1 iff the SGLang pred equals the harness gold (the row carries the harness gold)."""
    return int(row["pred"] == row["gold"])


# ---------------------------------------------------------------------------
# loaders
# ---------------------------------------------------------------------------
def test_loader_dir_glob_file_gz_and_skips(tmp_path):
    a = put(tmp_path / "d" / "gpu0" / "records.jsonl", [s_rec(0, seq(3)), s_rec(1, seq(4))])
    put(tmp_path / "d" / "gpu1" / "records.jsonl.gz", [s_rec(2, seq(5))], gz=True)
    put(tmp_path / "d" / "gpu0" / "cell.shard0.running.jsonl", [s_rec(9, seq(2))])         # a running file: skipped
    put(tmp_path / "d" / "gpu0" / "states.jsonl", [{"pid": "a", "x": 1}])                   # not a record: skipped
    (tmp_path / "d" / "gpu0" / "records.jsonl.done").write_text("ok\n")                     # a marker: not a jsonl
    with open(a, "a") as fh:
        fh.write('{"idx": 7, "gen_ids": [1, 2\n')                                           # truncated last line
    recs, info = JP.read_records(str(tmp_path / "d"))
    assert sorted(k[1] for k in recs) == [0, 1, 2]
    assert info["files"] == 3 and info["records"] == 3 and info["skipped_lines"] == 2        # states.jsonl line + truncated line
    assert sorted(k[1] for k in JP.read_records(str(tmp_path / "d" / "gpu*" / "records.jsonl"))[0]) == [0, 1]
    assert sorted(k[1] for k in JP.read_records(str(a))[0]) == [0, 1]
    assert JP.read_records(str(tmp_path / "nothing"))[0] == {}


def test_loader_duplicate_later_wins_and_is_counted(tmp_path):
    put(tmp_path / "d" / "a.jsonl", [s_rec(0, seq(3))])
    put(tmp_path / "d" / "b.jsonl", [s_rec(0, seq(5))])
    recs, info = JP.read_records(str(tmp_path / "d"))
    assert len(recs[(SPLIT, 0)]["ids"]) == 6 and info["duplicates"] == 1


def test_main_warns_about_duplicate_records(tmp_path, capsys):
    put(tmp_path / "sg" / "a.jsonl", [s_rec(0, seq(3))])
    put(tmp_path / "sg" / "b.jsonl", [s_rec(0, seq(3))])
    put(tmp_path / "h" / "a.jsonl", [h_rec(0, seq(3))])
    assert JP.main(["--sglang", str(tmp_path / "sg"), "--harness", str(tmp_path / "h")]) == 0
    assert "1 duplicate" in capsys.readouterr().err


def test_digest_cuts_harness_filler_and_reads_both_formats():
    h = JP.digest(h_rec(0, seq(3), filler=5))
    s = JP.digest(s_rec(0, seq(3)))
    assert h["ids"] == s["ids"] == seq(3) and h["blocks"] == s["blocks"] == 2 and h["passes"] == s["passes"] == 9
    assert h["draft_fwd"] == s["draft_fwd"] == 5            # harness: draft 3 + redraft 2, NOT prefill 1 / flush 2
    assert h["redraft_slots"] == s["redraft_slots"] == 8 and h["lg_slots"] == s["lg_slots"] == 4
    assert JP.digest(h_rec(0, [5, EOS2, 7], filler=2))["ids"] == [5, EOS2]                  # the second EOS id cuts too
    assert JP.digest(s_rec(0, [5, 6, 7]))["ids"] == [5, 6, 7]                                # no EOS: all of it


def test_digest_passes_fall_back_to_n_judge_and_missing_counts_are_none():
    r = s_rec(0, seq(2)); del r["passes_per_block"]
    d = JP.digest(r)
    assert d["passes"] == 9 and d["blocks"] == 2
    r2 = s_rec(0, seq(2)); del r2["n_draft_fwd"]
    assert JP.digest(r2)["draft_fwd"] is None


# ---------------------------------------------------------------------------
# agreement
# ---------------------------------------------------------------------------
def test_agreement_identical_one_divergence_length_mismatch():
    a = seq(10)
    assert JP.agreement(a, a) == {"exact": True, "prefix": 1.0, "tok": 1.0, "matches": 11, "min_len": 11, "dlen": 0}
    b = list(a); b[4] = 7                                                  # one divergence at position 4 of 11
    g = JP.agreement(a, b)
    assert not g["exact"] and g["prefix"] == pytest.approx(4 / 11) and g["tok"] == pytest.approx(10 / 11)
    # diverge early, agree again later: the prefix stops at the divergence, the token agreement keeps counting
    c = list(a); c[0] = 7
    g = JP.agreement(a, c)
    assert g["prefix"] == 0.0 and g["tok"] == pytest.approx(10 / 11)
    # length mismatch: [1,2,3,EOS] vs [1,2,3,4,5,EOS]: shorter length 4, 3 equal (the EOS meets a 4)
    g = JP.agreement([1, 2, 3, EOS], [1, 2, 3, 4, 5, EOS])
    assert g["prefix"] == 0.75 and g["tok"] == 0.75 and g["dlen"] == 2 and not g["exact"] and g["min_len"] == 4
    # a strict prefix (one side stopped early): full agreement over the shorter length, but not exact and |dlen| = 2
    g = JP.agreement([1, 2, 3], [1, 2, 3, 4, 5])
    assert g["prefix"] == 1.0 and g["tok"] == 1.0 and g["dlen"] == 2 and not g["exact"]


def test_agreement_empty_sides():
    assert JP.agreement([], [])["tok"] == 1.0 and JP.agreement([], [])["exact"]
    g = JP.agreement([], [1, 2])
    assert g["tok"] == 0.0 and g["prefix"] == 0.0 and g["dlen"] == 2


def test_loop_rule_matches_compare_gates():
    loop = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8] * 5
    fresh = list(range(1000, 1060))
    short = [7] * 20
    for ids in (loop, fresh, short, loop[:24], loop[:25], []):
        assert JP.is_looping(ids) == JP.is_looping(list(ids))
    assert JP.is_looping(loop) and not JP.is_looping(fresh) and not JP.is_looping(short[:5])
    cg = pytest.importorskip("drafter_lora.compare_gates")
    assert cg.NGRAM == JP.NGRAM
    for ids in (loop, fresh, short, loop[:24], loop[:25], [], loop + fresh, fresh + loop):
        assert JP.is_looping(ids) == cg.loops({"gen_ids": ids, "n_gen_to_eos": len(ids)})


def test_eos_ids_match_research_protocol():
    P = pytest.importorskip("research_protocol")
    assert tuple(P.EOS_IDS) == JP.EOS_IDS


def test_dataset_of_mirrors_run_da_decode():
    for split, ds in (("gsm8k500", "gsm8k"), ("calibvalid", "gsm8k"), ("mbpp", "mbpp"), ("opencode_pool:0:10", "mbpp"),
                      ("ifeval541", "ifeval"), ("if_pool", "ifeval"), ("math500", "math500"), ("math_pool:0:2000", "math500"),
                      ("aime", "aime"), ("gpqa", "gpqa"), ("humaneval", "humaneval"), ("finqa500", "finqa"),
                      ("chartqa300", "chartqa"), ("chat800", "chat")):
        assert JP.dataset_of(split) == ds


# ---------------------------------------------------------------------------
# join + report
# ---------------------------------------------------------------------------
def test_identical_records_pass_with_every_metric_at_parity(tmp_path):
    ids = {i: seq(10 + i) for i in range(4)}
    sg = [s_rec(i, ids[i]) for i in ids]
    hr = [h_rec(i, ids[i]) for i in ids]
    rc, rep = run(tmp_path, sg, hr)
    o = rep["overall"]
    assert rc == 0 and rep["verdict"]["status"] == "PASS" and rep["verdict"]["failures"] == []
    assert o["n"] == 4 and o["exact_rate"] == 1.0 and o["prefix_agreement"] == 1.0 and o["token_agreement"] == 1.0
    assert o["token_agreement_pooled"] == 1.0 and o["mean_abs_len_diff"] == 0.0
    assert o["passes_per_block"] == {"sglang": 4.5, "harness": 4.5, "rel_gap": 0.0}
    assert o["drafter_fwd_per_block"]["sglang"] == o["drafter_fwd_per_block"]["harness"] == pytest.approx(5 / 2)
    assert o["loopguard_slots_per_block"]["sglang"] == o["loopguard_slots_per_block"]["harness"] == 2.0
    assert o["loops"] == {"sglang": 0, "harness": 0} and o["join"] == {"sglang": 4, "harness": 4, "joined": 4,
                                                                      "sglang_only": 0, "harness_only": 0}
    assert set(rep["per_split"]) == {SPLIT} and rep["per_split"][SPLIT]["n"] == 4
    assert o["fallback_items"] == {"sglang": 0, "harness": 0}


def test_one_divergence_and_length_mismatch_aggregate(tmp_path):
    a, b = seq(10), seq(10, base=300)
    div = list(a); div[4] = 7
    sg = [s_rec(0, a), s_rec(1, div), s_rec(2, [1, 2, 3, EOS]), s_rec(3, b)]
    hr = [h_rec(0, a), h_rec(1, a), h_rec(2, [1, 2, 3, 4, 5, EOS]), h_rec(3, b)]
    rc, rep = run(tmp_path, sg, hr, "--min-token-agreement", "0.5")
    o = rep["overall"]
    assert o["exact_matches"] == 2 and o["exact_rate"] == 0.5
    assert o["prefix_agreement"] == pytest.approx((1 + 4 / 11 + 0.75 + 1) / 4)
    assert o["token_agreement"] == pytest.approx((1 + 10 / 11 + 0.75 + 1) / 4)
    assert o["token_agreement_pooled"] == pytest.approx((11 + 10 + 3 + 11) / (11 + 11 + 4 + 11))
    assert o["mean_abs_len_diff"] == pytest.approx(2 / 4)
    assert o["mean_len_sglang"] == pytest.approx((11 + 11 + 4 + 11) / 4) and o["mean_len_harness"] == pytest.approx((11 + 11 + 6 + 11) / 4)
    assert rc == 0


def test_join_is_on_split_and_idx(tmp_path):
    sg = [s_rec(0, seq(3), split="a"), s_rec(0, seq(3), split="b"), s_rec(1, seq(3), split="a"), s_rec(5, seq(3), split="a")]
    hr = [h_rec(0, seq(3), split="a"), h_rec(1, seq(3), split="b"), h_rec(1, seq(3), split="a"), h_rec(9, seq(3), split="a")]
    rc, rep = run(tmp_path, sg, hr)
    assert rep["overall"]["join"] == {"sglang": 4, "harness": 4, "joined": 2, "sglang_only": 2, "harness_only": 2}
    assert set(rep["per_split"]) == {"a"} and rep["per_split"]["a"]["n"] == 2 and rc == 0


def test_per_split_rows_and_ignore_split(tmp_path):
    sg = [s_rec(0, seq(3), split="gsm8k500"), s_rec(1, seq(3), split="math500")]
    hr = [h_rec(0, seq(3), split="gsm8k500"), h_rec(1, seq(4), split="math500")]
    rc, rep = run(tmp_path, sg, hr, "--min-token-agreement", "0.5")
    assert set(rep["per_split"]) == {"gsm8k500", "math500"}
    assert rep["per_split"]["gsm8k500"]["exact_rate"] == 1.0 and rep["per_split"]["math500"]["exact_rate"] == 0.0
    # the SGLang side names the split differently: no join by default, a join with --ignore-split
    tmp2 = tmp_path / "other"
    rc2, rep2 = run(tmp2, [s_rec(0, seq(3), split="gsm8k_500")], [h_rec(0, seq(3), split="gsm8k500")])
    assert rc2 == 3 and rep2["overall"]["n"] == 0 and rep2["verdict"]["failures"][0] == "joined_items"
    tmp3 = tmp_path / "third"
    rc3, rep3 = run(tmp3, [s_rec(0, seq(3), split="gsm8k_500")], [h_rec(0, seq(3), split="gsm8k500")], "--ignore-split")
    assert rc3 == 0 and rep3["overall"]["n"] == 1 and set(rep3["per_split"]) == {"all"}


def test_passes_per_block_is_sum_over_sum_not_mean_of_ratios(tmp_path):
    # item 0: 1 block, 10 passes; item 1: 9 blocks, 9 passes  ->  19 / 10 = 1.9 (a mean of ratios would say 5.5)
    sg = [s_rec(0, seq(3), passes=(10,)), s_rec(1, seq(3), passes=(1,) * 9)]
    hr = [h_rec(0, seq(3), passes=(10,)), h_rec(1, seq(3), passes=(1,) * 9)]
    rc, rep = run(tmp_path, sg, hr)
    assert rep["overall"]["passes_per_block"]["sglang"] == pytest.approx(1.9) and rc == 0
    # SGLang takes 10% more passes
    sg2 = [s_rec(0, seq(3), passes=(11,)), s_rec(1, seq(3), passes=(1,) * 9)]
    rc, rep = run(tmp_path / "x", sg2, hr)
    pg = rep["overall"]["passes_per_block"]
    assert pg["sglang"] == pytest.approx(2.0) and pg["rel_gap"] == pytest.approx(2.0 / 1.9 - 1) and rc == 3
    assert rep["verdict"]["failures"] == ["passes_per_block_rel_gap"] and "passes_per_block_rel_gap" in rep["verdict"]["message"]


def test_drafter_forwards_exclude_prefill_and_flush(tmp_path):
    sg = [s_rec(0, seq(3), passes=(5, 5), draft=4, redraft=6)]
    hr = [h_rec(0, seq(3), passes=(5, 5), draft=4, redraft=6)]
    rc, rep = run(tmp_path, sg, hr)
    d = rep["overall"]["drafter_fwd_per_block"]
    assert d["sglang"] == d["harness"] == 5.0 and d["rel_gap"] == 0.0           # (4 + 6) / 2 blocks; prefill 1 + flush 2 not counted
    sg = [s_rec(0, seq(3), passes=(5, 5), draft=4, redraft=8)]
    rc, rep = run(tmp_path / "y", sg, hr)
    assert rep["overall"]["drafter_fwd_per_block"]["rel_gap"] == pytest.approx(0.2)     # reported, not a verdict criterion
    assert rc == 0


def test_loops_counted_on_both_sides(tmp_path):
    loop = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8] * 5
    fresh = list(range(1000, 1060))
    sg = [s_rec(0, loop + [EOS]), s_rec(1, fresh + [EOS]), s_rec(2, loop + [EOS])]
    hr = [h_rec(0, loop + [EOS]), h_rec(1, loop + [EOS]), h_rec(2, fresh + [EOS])]
    rc, rep = run(tmp_path, sg, hr, "--min-token-agreement", "0")
    assert rep["overall"]["loops"] == {"sglang": 2, "harness": 2}


def test_fallback_items_count(tmp_path):
    sg = [s_rec(0, seq(3), fallback=True), s_rec(1, seq(3), fallback=False)]
    hr = [h_rec(0, seq(3), fallback=(0, 5)), h_rec(1, seq(3), fallback=(0, 0))]
    rc, rep = run(tmp_path, sg, hr)
    assert rep["overall"]["fallback_items"] == {"sglang": 1, "harness": 1}


# ---------------------------------------------------------------------------
# accuracy
# ---------------------------------------------------------------------------
def test_accuracy_scores_sglang_pred_with_harness_gold(tmp_path, monkeypatch):
    # 4 items. harness correct: 1,1,0,0. SGLang pred == gold on items 0,2 -> sglang correct 1,0,1,0.
    gold = ["a", "b", "c", "d"]
    hc = [1, 1, 0, 0]
    sp = ["a", "WRONG", "c", "WRONG"]
    hr = [h_rec(i, seq(3), gold=gold[i], correct=hc[i], pred="whatever") for i in range(4)]
    sg = [s_rec(i, seq(3), pred=sp[i]) for i in range(4)]
    rc, rep = run(tmp_path, sg, hr, "--max-acc-gap", "10", scorer=gold_scorer, monkeypatch=monkeypatch)
    a = rep["overall"]["accuracy"]
    assert a["n_scored"] == 4 and a["harness"] == 50.0 and a["sglang"] == 50.0 and a["gap_points"] == 0.0
    assert a["sglang_only_correct"] == 1 and a["harness_only_correct"] == 1 and rc == 0
    assert "resolution_warning" in a                                   # 25-point steps vs a 10-point limit
    # SGLang loses one more item: 25% vs 50% -> gap -25 points
    sg[0] = s_rec(0, seq(3), pred="WRONG")
    rc, rep = run(tmp_path / "z", sg, hr, "--max-acc-gap", "10", scorer=gold_scorer, monkeypatch=monkeypatch)
    a = rep["overall"]["accuracy"]
    assert a["sglang"] == 25.0 and a["gap_points"] == -25.0 and rc == 3
    assert rep["verdict"]["failures"] == ["accuracy_gap_points"] and "accuracy_gap_points" in rep["verdict"]["message"]


def test_scorer_row_carries_what_the_ifeval_scorer_reads(tmp_path, monkeypatch):
    seen = []

    def scorer(ds, row):
        seen.append((ds, dict(row)))
        return 1
    hr = [h_rec(0, seq(3), split="ifeval541", gold={"instruction_id_list": ["x"]}, correct=1)]
    sg = [s_rec(0, seq(3), split="ifeval541", pred="text")]
    run(tmp_path, sg, hr, scorer=scorer, monkeypatch=monkeypatch)
    ds, row = seen[0]
    assert ds == "ifeval" and row["gold"] == {"instruction_id_list": ["x"]} and row["pred"] == "text"
    assert row["gen_ids"] == seq(3) and row["n_gen_to_eos"] == 4 and row["idx"] == 0


def test_accuracy_only_over_items_the_harness_scored_and_score_errors_are_dropped(tmp_path, monkeypatch):
    def scorer(ds, row):
        if row["idx"] == 2:
            raise RuntimeError("boom")
        return int(row["pred"] == row["gold"])
    hr = [h_rec(0, seq(3), gold="a", correct=1), h_rec(1, seq(3), gold="a", correct=None),
          h_rec(2, seq(3), gold="a", correct=1), h_rec(3, seq(3), gold="a", correct=0)]
    sg = [s_rec(i, seq(3), pred="a") for i in range(4)]
    rc, rep = run(tmp_path, sg, hr, "--max-acc-gap", "100", scorer=scorer, monkeypatch=monkeypatch)
    a = rep["overall"]["accuracy"]
    assert a["n_harness_scored"] == 3 and a["n_scored"] == 2 and a["score_errors"] == 1
    assert a["harness"] == 50.0 and a["sglang"] == 100.0 and a["gap_points"] == 50.0


def test_missing_correctness_is_null_and_not_a_failure(tmp_path, monkeypatch):
    hr = [h_rec(i, seq(3), correct=None) for i in range(3)]
    sg = [s_rec(i, seq(3)) for i in range(3)]
    rc, rep = run(tmp_path, sg, hr, monkeypatch=monkeypatch, scorer=gold_scorer)
    a = rep["overall"]["accuracy"]
    assert a["sglang"] is None and a["harness"] is None and a["gap_points"] is None and a["n_scored"] == 0
    assert rc == 0 and rep["verdict"]["status"] == "PASS" and rep["harness_has_correctness"] is False
    assert rep["verdict"]["criteria"]["accuracy_gap_points"]["ok"] is None
    rc, rep = run(tmp_path / "req", sg, hr, "--require-accuracy", monkeypatch=monkeypatch, scorer=gold_scorer)
    assert rc == 3 and rep["verdict"]["failures"] == ["accuracy_gap_points"] and "not computable" in rep["verdict"]["message"]


def test_scorer_unavailable_gives_null_accuracy(tmp_path, monkeypatch):
    hr = [h_rec(i, seq(3), correct=1) for i in range(3)]
    sg = [s_rec(i, seq(3)) for i in range(3)]
    rc, rep = run(tmp_path, sg, hr, monkeypatch=monkeypatch)                  # make_scorer raises ImportError
    assert rep["overall"]["accuracy"]["gap_points"] is None and "ImportError" in rep["scorer_error"] and rc == 0
    rc, rep = run(tmp_path / "req", sg, hr, "--require-accuracy", monkeypatch=monkeypatch)
    assert rc == 3


# ---------------------------------------------------------------------------
# verdict thresholds
# ---------------------------------------------------------------------------
def _overall(tok=1.0, pg=0.0, ag=None, n=10):
    return {"n": n, "token_agreement": tok, "passes_per_block": {"rel_gap": pg}, "accuracy": {"gap_points": ag}}


def test_verdict_thresholds_at_below_and_above_the_limit():
    V = lambda **kw: JP.verdict(_overall(**kw), 0.95, 3.0, 0.05, False, 1)
    assert V()["status"] == "PASS"
    assert V(tok=0.95)["status"] == "PASS" and V(tok=0.9499)["failures"] == ["token_agreement"]       # >= the minimum
    assert V(pg=0.05)["status"] == "PASS" and V(pg=-0.05)["status"] == "PASS" and V(pg=0.0501)["failures"] == ["passes_per_block_rel_gap"]
    assert V(pg=-0.0501)["failures"] == ["passes_per_block_rel_gap"]                                     # the gap is two-sided
    assert V(ag=3.0)["status"] == "PASS" and V(ag=-3.0)["status"] == "PASS" and V(ag=3.1)["failures"] == ["accuracy_gap_points"]
    assert V(ag=-3.1)["failures"] == ["accuracy_gap_points"]
    both = V(tok=0.5, pg=0.2, ag=9.0)
    assert both["status"] == "FAIL" and both["failures"] == ["token_agreement", "passes_per_block_rel_gap", "accuracy_gap_points"]
    assert all(k in both["message"] for k in both["failures"])
    assert V(ag=None)["status"] == "PASS"                                                               # accuracy unavailable: not a failure
    assert JP.verdict(_overall(ag=None), 0.95, 3.0, 0.05, True, 1)["failures"] == ["accuracy_gap_points"]
    assert V(n=0)["failures"] == ["joined_items"]
    assert JP.verdict(_overall(n=3), 0.95, 3.0, 0.05, False, 5)["failures"] == ["joined_items"]


def test_verdict_passes_gap_not_computable_is_a_failure():
    v = JP.verdict(_overall(pg=None), 0.95, 3.0, 0.05, False, 1)
    assert v["failures"] == ["passes_per_block_rel_gap"] and "not computable" in v["message"]


def test_main_thresholds_via_flags_and_exit_codes(tmp_path):
    a = seq(19)
    div = list(a); div[10] = 7                                   # 20 tokens, one differs: token agreement 0.95 exactly
    sg = [s_rec(0, div)]
    hr = [h_rec(0, a)]
    rc, rep = run(tmp_path, sg, hr)
    assert rep["overall"]["token_agreement"] == pytest.approx(0.95) and rc == 0               # at the limit: PASS
    rc, rep = run(tmp_path / "b", sg, hr, "--min-token-agreement", "0.96")
    assert rc == 3 and rep["verdict"]["failures"] == ["token_agreement"] and rep["verdict"]["status"] == "FAIL"
    assert rep["thresholds"]["min_token_agreement"] == 0.96
    for k in ("sglang", "harness", "overall", "per_split", "verdict", "thresholds", "utc"):
        assert k in rep


def test_main_no_records_is_exit_2(tmp_path, capsys):
    put(tmp_path / "h" / "a.jsonl", [h_rec(0, seq(3))])
    rc = JP.main(["--sglang", str(tmp_path / "missing"), "--harness", str(tmp_path / "h")])
    assert rc == 2 and "no records" in capsys.readouterr().err


def test_table_names_the_verdict_and_lists_rows(tmp_path, capsys):
    sg = [s_rec(0, seq(3)), s_rec(1, seq(4), split="math500")]
    hr = [h_rec(0, seq(3)), h_rec(1, seq(4), split="math500")]
    run(tmp_path, sg, hr)
    out = capsys.readouterr().out
    assert "gsm8k500" in out and "math500" in out and "ALL" in out and "VERDICT: PASS" in out and "join:" in out
