"""CPU tests for corpus1b/fresh_lists.py (wave C fresh prompt lists): synthetic prompt lists and wave A/B done.txt
files in tmp_path, --no-tokenizer (one test fakes the tokenizer the way test_code_if_extractors.py does).

compose_if's screens need langdetect and the vendored IFEvalG registry's deps (absl, immutabledict, NLTK). Where
those are not installed (the dev Mac) the IF tests replace fresh_lists._if_tools -- the one seam over
compose_if._eligible_base / pick_constraints / admitted_ids / conflict_map and build_if_pool.dump_gold -- with small
fakes; test_if_rows_with_real_compose_if runs the real ones and is skipped where they cannot be imported.

    python3 -m pytest -q code/corpus1b/tests/test_fresh_lists.py
"""
from __future__ import annotations

import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))               # code/corpus1b/tests
CORPUS1B = os.path.abspath(os.path.join(HERE, ".."))             # code/corpus1b
CODE = os.path.abspath(os.path.join(CORPUS1B, ".."))             # code/
for _p in (CORPUS1B, os.path.join(CODE, "compat"), CODE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import research_protocol as P                                    # noqa: E402
import common                                                    # noqa: E402
import fresh_lists                                               # noqa: E402

CRED_TOKEN = "hf_" + "A1b2C3d4E5" * 4                            # matches sanitize_lists.RX
DESC = {"change_case:english_lowercase": "Your entire response should be in English, and in all lowercase letters.",
        "punctuation:no_comma": "In your entire response, refrain from the use of any commas."}


# ---------------------------------------------------------------------------
# synthetic lists and done files
# ---------------------------------------------------------------------------
def _row(domain, tag, source, sid, prompt, key, bucket):
    pid = common.make_pid(domain, tag, source, sid)
    return {"pid": pid, "domain": domain, "source": source, "source_id": str(sid), "prompt": prompt,
            "gen_cap": 1024, "key": key, "meta": {"bucket": bucket}, "wave": common.wave_of(pid)}


def _chat(i, text=None):
    return _row("chat", "ultrachat", "HuggingFaceH4/ultrachat_200k", f"u{i}",
                text or f"Please explain topic number {i} in plain words for a curious beginner.",
                {"type": "chat"}, "ultrachat")


def _setup(tmp_path, n_maths=12, n_code=10, chat_rows=None, decoded_every=4):
    """Writes <tmp>/lists/{maths,code,chat}.jsonl and two done files (wave A / wave B) holding every
    `decoded_every`-th row's pid of each list. -> (lists_dir, done_glob, {domain: rows}, decoded pid set)"""
    lists = {
        "maths": [_row("maths", "gsm8k", "openai/gsm8k", i, f"What is {i} + {i}?", {"type": "math", "answer": str(2 * i)},
                       "gsm") for i in range(n_maths)],
        "code": [_row("code", "mbpptrain", "google-research-datasets/mbpp", i, f"Write a function f{i}.",
                      {"type": "asserts", "tests": f"assert f{i}() == {i}"}, "function") for i in range(n_code)],
        "chat": chat_rows if chat_rows is not None else [_chat(i) for i in range(16)],
    }
    lists_dir = tmp_path / "lists"
    for d, rows in lists.items():
        common.write_jsonl(str(lists_dir / f"{d}.jsonl"), rows)
    decoded = [r["pid"] for rows in lists.values() for i, r in enumerate(rows) if i % decoded_every == 0]
    for k, wave in enumerate("AB"):                              # decoded pids split across two waves' files
        p = tmp_path / "logs" / f"wave{wave}" / f"gpu{k}" / "done.txt"
        p.parent.mkdir(parents=True)
        p.write_text("\n".join(decoded[k::2]) + "\n")
    return str(lists_dir), str(tmp_path / "logs" / "wave*" / "gpu*" / "done.txt"), lists, set(decoded)


def _fake_tools():
    def eligible(text, exclude_norm):
        assert exclude_norm is None
        return "INELIGIBLE" not in text

    def pick(pid, ids_pool, conflicts, base):
        if "NOPICK" in base:
            return None
        if "CREDPICK" in base:
            return ["punctuation:no_comma"], [{}], [f"End with the code {CRED_TOKEN}."]
        n = 1 + int(common.sha256_hex(common.normalize_text(base))[:2], 16) % 2    # 1 or 2, a function of the base
        ids = sorted(DESC)[:n]
        return ids, [{} for _ in ids], [DESC[i] for i in ids]

    def dump_gold(ids, kwargs):
        return json.dumps({"instruction_id_list": list(ids), "kwargs": list(kwargs)}, sort_keys=True)
    return eligible, pick, sorted(DESC), {}, dump_gold


def _run(lists, done, out, targets, extra=()):
    argv = ["--lists", lists, "--done", done, "--out", str(out), "--targets", targets, *extra]
    if "--max-prompt-tokens" not in extra:
        argv.append("--no-tokenizer")
    assert fresh_lists.main(argv) == 0
    rows = {d: list(common.read_jsonl(str(out / f"{d}.jsonl"))) for d in fresh_lists.DOMAINS
            if (out / f"{d}.jsonl").exists()}
    return rows, json.load(open(out / "manifest.json"))


def _ordered(rows, decoded):
    return sorted([r for r in rows if r["pid"] not in decoded], key=lambda r: common.order_key(r["pid"]))


# ---------------------------------------------------------------------------
# maths / code / chat
# ---------------------------------------------------------------------------
def test_list_domains_skip_decoded_and_take_target_in_order(tmp_path, monkeypatch):
    monkeypatch.setattr(fresh_lists, "_if_tools", _fake_tools)
    lists, done, src, decoded = _setup(tmp_path)
    out, man = _run(lists, done, tmp_path / "out", "maths=4,code=3,chat=3,if=2")
    for d, n in (("maths", 4), ("code", 3), ("chat", 3)):
        want = _ordered(src[d], decoded)[:n]
        assert [r["pid"] for r in out[d]] == [r["pid"] for r in want]
        assert out[d] == [dict(r, wave="C") for r in want]          # unchanged except the wave
        assert not {r["pid"] for r in out[d]} & decoded
    assert all(r["wave"] == "C" for rows in out.values() for r in rows)
    assert man["n_decoded_pids"] == len(decoded) and len(man["done_files"]) == 2
    assert man["domains"]["maths"]["n_undecoded_available"] == 9    # 12 rows, 3 decoded (i = 0, 4, 8)
    assert man["domains"]["code"]["n_undecoded_available"] == 7     # 10 rows, 3 decoded


def test_shortfall_when_a_list_runs_out(tmp_path, monkeypatch):
    monkeypatch.setattr(fresh_lists, "_if_tools", _fake_tools)
    lists, done, src, decoded = _setup(tmp_path)
    out, man = _run(lists, done, tmp_path / "out", "maths=50,code=3,chat=3,if=1", ("--domains", "maths,code"))
    assert len(out["maths"]) == 9 and man["domains"]["maths"]["shortfall"] == 41
    assert set(out) == {"maths", "code"} and set(man["domains"]) == {"maths", "code"}


def test_empty_done_is_refused_unless_allowed(tmp_path):
    lists, _, src, _ = _setup(tmp_path)
    nothing = str(tmp_path / "nowhere" / "done.txt")
    assert fresh_lists.main(["--lists", lists, "--done", nothing, "--out", str(tmp_path / "o1"), "--no-tokenizer",
                             "--domains", "maths"]) == 1
    out, man = _run(lists, nothing, tmp_path / "o2", "maths=5", ("--domains", "maths", "--allow-empty-done"))
    assert len(out["maths"]) == 5 and man["n_decoded_pids"] == 0


# ---------------------------------------------------------------------------
# if
# ---------------------------------------------------------------------------
def test_if_rows_shape_and_disjoint_from_chat(tmp_path, monkeypatch):
    monkeypatch.setattr(fresh_lists, "_if_tools", _fake_tools)
    lists, done, src, decoded = _setup(tmp_path)
    out, man = _run(lists, done, tmp_path / "out", "maths=1,code=1,chat=3,if=4")
    chat_order = _ordered(src["chat"], decoded)                     # 12 undecoded chat rows
    assert [r["pid"] for r in out["chat"]] == [r["pid"] for r in chat_order[:3]]
    by_pid = {r["pid"]: r for r in src["chat"]}
    assert [r["meta"]["base_pid"] for r in out["if"]] == [r["pid"] for r in chat_order[3:7]]   # the next 4, in order
    assert not {r["meta"]["base_pid"] for r in out["if"]} & {r["pid"] for r in out["chat"]}
    for r in out["if"]:
        base = by_pid[r["meta"]["base_pid"]]
        gold = json.loads(r["key"]["gold"])
        ids = gold["instruction_id_list"]
        assert r["pid"] == common.make_pid("if", "ifchat", base["source"], base["source_id"])
        assert r["pid"].startswith("if-ifchat-") and common.source_tag_of(r["pid"]) == "ifchat"
        assert (r["domain"], r["source"], r["source_id"], r["gen_cap"], r["wave"]) == \
            ("if", base["source"], base["source_id"], 1024, "C")
        assert r["prompt"] == base["prompt"] + "\n\n" + " ".join(DESC[i] for i in ids)
        assert r["key"]["type"] == "ifeval" and set(gold) == {"instruction_id_list", "kwargs"}
        assert r["meta"] == {"bucket": "composed", "n_constraints": len(ids),
                             "base_prompt_sha": common.sha256_hex(base["prompt"]), "base_pid": base["pid"]}
    m = man["domains"]["if"]
    assert m["n_taken"] == 4 and m["shortfall"] == 0 and m["n_undecoded_available"] == 9
    assert m["n_bases_walked"] == 4 and m["pass_rate"] == 1.0


def test_if_alone_splits_at_the_same_chat_target(tmp_path, monkeypatch):
    monkeypatch.setattr(fresh_lists, "_if_tools", _fake_tools)
    lists, done, _, _ = _setup(tmp_path)
    both, _ = _run(lists, done, tmp_path / "all", "maths=1,code=1,chat=3,if=4")
    alone, man = _run(lists, done, tmp_path / "if", "chat=3,if=4", ("--domains", "if"))
    assert set(alone) == {"if"} and alone["if"] == both["if"]
    assert man["domains"]["if"]["chat_target_skipped"] == 3


def test_if_shortfall_and_counts_add_up(tmp_path, monkeypatch):
    monkeypatch.setattr(fresh_lists, "_if_tools", _fake_tools)
    special = ["INELIGIBLE: a base the screen rejects, long enough to look like a prompt.",
               "NOPICK: a base for which not even one constraint could be assembled.",
               "CREDPICK: a base whose constraint text carries a credential-like string.",
               "Explain the water cycle to me, please.",
               "explain   the WATER cycle to me,  please."]             # a normalize_text duplicate of the one above
    chat = [_chat(i) for i in range(4)] + [_chat(100 + k, t) for k, t in enumerate(special)]
    lists, done, src, decoded = _setup(tmp_path, chat_rows=chat, decoded_every=100)   # only chat row 0 decoded
    out, man = _run(lists, done, tmp_path / "out", "maths=1,code=1,chat=0,if=50")
    m = man["domains"]["if"]
    assert m["n_undecoded_available"] == len(chat) - 1                # chat row 0 is decoded
    assert m["n_bases_walked"] == m["n_undecoded_available"]          # the bases ran out
    assert (m["n_eligible"], m["n_with_constraints"]) == (m["n_bases_walked"] - 1, m["n_bases_walked"] - 2)
    assert (m["n_dropped_credential"], m["n_dropped_dup"], m["n_dropped_token_length"]) == (1, 1, 0)
    assert m["n_taken"] == len(out["if"]) == m["n_bases_walked"] - 4
    assert m["shortfall"] == 50 - m["n_taken"]
    assert m["n_bases_walked"] == ((m["n_bases_walked"] - m["n_eligible"]) + (m["n_eligible"] - m["n_with_constraints"])
                                   + m["n_dropped_credential"] + m["n_dropped_dup"] + m["n_dropped_token_length"]
                                   + m["n_taken"])
    assert m["pass_rate"] == round(m["n_taken"] / m["n_bases_walked"], 4)
    assert not any(fresh_lists.CRED.search(json.dumps(r)) for r in out["if"])
    assert len({common.normalize_text(r["prompt"]) for r in out["if"]}) == len(out["if"])


def test_credential_like_list_rows_are_dropped(tmp_path, monkeypatch):
    monkeypatch.setattr(fresh_lists, "_if_tools", _fake_tools)
    chat = [_chat(i) for i in range(8)] + [_chat(99, f"Why does my token {CRED_TOKEN} not work with the API?")]
    lists, done, src, decoded = _setup(tmp_path, chat_rows=chat, decoded_every=100)
    out, man = _run(lists, done, tmp_path / "out", "maths=1,code=1,chat=20,if=0")
    assert len(out["chat"]) == 7 and CRED_TOKEN not in json.dumps(out["chat"])
    assert man["domains"]["chat"]["n_dropped_credential"] == 1
    assert man["domains"]["chat"]["n_undecoded_available"] == 8 - 1      # chat row 0 is decoded


def test_if_token_length_filter_with_fake_tokenizer(tmp_path, monkeypatch):
    monkeypatch.setattr(fresh_lists, "_if_tools", _fake_tools)

    class FakeBatch:
        def __init__(self, n):
            self.shape = (1, n)

    monkeypatch.setattr(fresh_lists.B, "_get_tokenizer", lambda: object())
    monkeypatch.setattr(P, "encode_prompt", lambda tok, prompt: FakeBatch(len(prompt.split())))
    chat = [_chat(i) for i in range(3)] + [_chat(50, "Tell me a story about " + "a very long winding road " * 10)]
    lists, done, _, _ = _setup(tmp_path, chat_rows=chat, decoded_every=100)
    out, man = _run(lists, done, tmp_path / "out", "maths=1,code=1,chat=0,if=10", ("--max-prompt-tokens", "40"))
    m = man["domains"]["if"]
    assert m["n_dropped_token_length"] == 1 and m["n_taken"] == 2       # chat row 0 decoded; the long base dropped
    assert all(len(r["prompt"].split()) <= 40 for r in out["if"])
    assert man["params"]["tokenizer"] is True


# ---------------------------------------------------------------------------
# the real compose_if screens, where their deps are installed (the GPU box)
# ---------------------------------------------------------------------------
def _real_if_tools_importable() -> bool:
    try:
        fresh_lists._if_tools()
        return True
    except (ImportError, LookupError):
        return False


@pytest.mark.skipif(not _real_if_tools_importable(),
                    reason="compose_if needs langdetect + the vendored IFEvalG deps (absl, immutabledict), not installed")
def test_if_rows_with_real_compose_if(tmp_path):
    import compose_if
    from draftalign import if_scorer as S
    good = ["What is a good recipe for banana bread that does not need a stand mixer or any fancy equipment?",
            "How do I keep my basil plant alive through the winter when my apartment gets very little sunlight?",
            "Can you describe what life was like for a lighthouse keeper on a remote island a century ago?",
            "What should I think about before adopting a retired greyhound if I live in a small flat?"]
    chat = [_chat(i, t) for i, t in enumerate(good)] + [_chat(10, "Hi there")]       # too short: ineligible
    lists, done, src, decoded = _setup(tmp_path, chat_rows=chat, decoded_every=100)  # chat row 0 decoded
    out, man = _run(lists, done, tmp_path / "out", "maths=1,code=1,chat=0,if=10")
    m = man["domains"]["if"]
    assert m["n_bases_walked"] == 4 and m["n_eligible"] == 3
    admitted, conflicts = set(compose_if.admitted_ids()), compose_if.conflict_map()
    by_pid = {r["pid"]: r for r in chat}
    for r in out["if"]:
        base = by_pid[r["meta"]["base_pid"]]
        ids, kwargs = S.parse_gold(r["key"]["gold"])
        assert set(ids) <= admitted and 1 <= len(ids) <= 3 and r["meta"]["n_constraints"] == len(ids)
        assert all(b not in conflicts.get(a, {a}) for a in ids for b in ids if a != b)
        assert r["prompt"].startswith(base["prompt"] + "\n\n") and r["wave"] == "C"
        assert compose_if.pick_constraints(r["pid"], compose_if.admitted_ids(), conflicts, base["prompt"])[0] == ids
