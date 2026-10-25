"""CPU tests for code/draftalign/if_scorer.py (no model, no network).

Positive / negative controls: for each instruction id a hand-built response that complies and one that violates,
scored in strict mode; the all-constraints rule; reasoning stripping; the fail-closed record wrapper; gold
validation; determinism; pinned checker quirks that justify build_if_pool.EXCLUDED_IDS; and the frozen pool's
integrity (sha256, every id implemented, 1-3 constraints, text hashes).
"""
import hashlib
import json
import os
import random
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from draftalign import if_scorer as S                                          # noqa: E402

POOL = os.path.join(HERE, "..", "configs", "if_pool.json")


def gold(*pairs):
    return json.dumps({"instruction_id_list": [p[0] for p in pairs], "kwargs": [p[1] for p in pairs]})


# (instruction id, kwargs, compliant response, violating response)
CONTROLS = [
    ("punctuation:no_comma", {}, "I like apples and pears.", "I like apples, and pears."),
    ("detectable_format:title", {}, "<<Rain on the Roof>>\nSoft rain falls all night.",
     "Rain on the Roof\nSoft rain falls all night."),
    ("keywords:existence", {"keywords": ["ocean", "sunset"]}, "We watched the sunset over the ocean.",
     "We watched the sunrise over the lake."),
    ("keywords:forbidden_words", {"forbidden_words": ["cat", "dog"]}, "I keep a parrot at home.",
     "I keep a dog at home."),
    ("keywords:frequency", {"keyword": "data", "frequency": 3, "relation": "at least"},
     "Data matters. We clean data and label data.", "Data matters. We clean data carefully."),
    ("keywords:letter_frequency", {"letter": "z", "let_frequency": 2, "let_relation": "less than"},
     "A quiet zone.", "The buzzing bees."),
    ("length_constraints:number_words", {"num_words": 6, "relation": "less than"}, "Yes, I fully agree.",
     "This sentence clearly has more than six words in it."),
    ("length_constraints:number_sentences", {"num_sentences": 3, "relation": "at least"},
     "The sky is blue. The grass is green. The sun is warm.", "The sky is blue and the grass is green."),
    ("length_constraints:number_paragraphs", {"num_paragraphs": 2},
     "First paragraph here.\n***\nSecond paragraph here.",
     "First paragraph.\n***\nSecond paragraph.\n***\nThird paragraph."),
    ("length_constraints:nth_paragraph_first_word", {"num_paragraphs": 2, "nth_paragraph": 2, "first_word": "however"},
     "The plan is simple.\n\nHowever, it takes time.", "The plan is simple.\n\nBut it takes time."),
    ("detectable_content:number_placeholders", {"num_placeholders": 2}, "Dear [name], we will ship to [address].",
     "Dear [name], we will ship soon."),
    ("detectable_content:postscript", {"postscript_marker": "P.S."}, "Thanks for your help.\n\nP.S. See you on Monday.",
     "Thanks for your help. See you on Monday."),
    ("detectable_format:number_bullet_lists", {"num_bullets": 2}, "Two reasons:\n* It is cheap.\n* It is fast.",
     "Three reasons:\n* It is cheap.\n* It is fast.\n* It is safe."),
    ("detectable_format:number_highlighted_sections", {"num_highlights": 2}, "This is *important* and *urgent*.",
     "This is *important* only."),
    ("detectable_format:multiple_sections", {"section_spliter": "Section", "num_sections": 2},
     "Section 1\nThe intro.\nSection 2\nThe body.", "Section 1\nThe intro and the body."),
    ("combination:two_responses", {}, "Tea is calming.\n******\nCoffee is energizing.",
     "Tea is calming.\n******\nTea is calming."),
    ("combination:repeat_prompt", {"prompt_to_repeat": "Write a haiku about rain."},
     "Write a haiku about rain.\nSoft rain on the roof.", "Soft rain on the roof."),
    ("startend:end_checker", {"end_phrase": "Any other questions?"}, "The meeting is at noon. Any other questions?",
     "Any other questions? The meeting is at noon."),
    ("startend:quotation", {}, "\"The meeting is at noon.\"", "The meeting is at noon."),
    ("change_case:english_lowercase", {},
     "the meeting has been moved to thursday afternoon because the room is being painted.",
     "The meeting has been moved to Thursday afternoon because the room is being painted."),
    ("change_case:english_capital", {},
     "THE MEETING HAS BEEN MOVED TO THURSDAY AFTERNOON BECAUSE THE ROOM IS BEING PAINTED.",
     "THE MEETING HAS BEEN MOVED to thursday AFTERNOON BECAUSE THE ROOM IS BEING PAINTED."),
    ("change_case:capital_word_frequency", {"capital_frequency": 2, "capital_relation": "less than"},
     "Only ONE word is shouted here.", "THREE WORDS ARE shouted here."),
    ("first_word:first_word_answer", {"first_word": "indeed"}, "Indeed it works well.", "Yes, it works well indeed."),
    ("last_word:last_word_answer", {"last_word": "brief"}, "I will keep this brief.", "Brief answers are best."),
    ("keywords:word_once", {"keyword": "river"}, "We walked along the river at dawn.", "The river met another river."),
    ("count:count_increment_word", {"keyword1": "apple", "keyword2": "pear"}, "An apple, a pear, and one more pear.",
     "An apple and a pear."),
    ("paragraphs:paragraphs", {}, "First paragraph.\n***\nSecond paragraph.", "First paragraph.\n\nSecond paragraph."),
    ("paragraphs:paragraphs2", {}, "First paragraph.\n\nSecond paragraph.",
     "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."),
    ("punctuation:punctuation_exclamation", {}, "Please stay calm.", "Please stay calm!"),
    ("letters:letter_counting2", {"letter": "q", "let_frequency": 1, "let_relation": "at least"}, "A quiet morning.",
     "A silent morning."),
    ("detectable_format:json_format", {}, "```json\n{\"city\": \"Paris\"}\n```", "city: Paris"),
    ("language:response_language", {"language": "en"},
     "The weather today is sunny and warm, perfect for a long walk in the park.",
     "Le temps aujourd'hui est ensoleillé et chaud, parfait pour une longue promenade dans le parc."),
]


@pytest.mark.parametrize("iid,kwargs,good,bad", CONTROLS, ids=[c[0] for c in CONTROLS])
def test_positive_and_negative_controls(iid, kwargs, good, bad):
    g = gold((iid, kwargs))
    assert S.score_ifeval(good, g) == 1, f"{iid}: compliant response scored 0"
    assert S.score_ifeval(bad, g) == 0, f"{iid}: violating response scored 1"


def test_controls_cover_every_admitted_pool_id():
    """Every id the frozen pool can contain has a positive and a negative control above."""
    if not os.path.exists(POOL):
        pytest.skip("if_pool.json not built")
    spec = json.loads(open(POOL, "rb").read())
    assert set(spec["id_policy"]["admitted"]) <= {c[0] for c in CONTROLS}


def test_all_constraints_must_pass():
    g = gold(("punctuation:no_comma", {}), ("detectable_format:title", {}),
             ("startend:end_checker", {"end_phrase": "The END."}))
    assert S.score_ifeval("<<Notes>>\nShort and clear. The END.", g) == 1
    assert S.score_ifeval("<<Notes>>\nShort, and clear. The END.", g) == 0      # a comma
    assert S.score_ifeval("Notes\nShort and clear. The END.", g) == 0           # no title
    assert S.score_ifeval("<<Notes>>\nShort and clear.", g) == 0                # no end phrase
    ids, kw = S.parse_gold(g)
    assert S.check_constraints("<<Notes>>\nShort, and clear. The END.", ids, kw) == [False, True, True]


def test_reasoning_is_stripped_and_unclosed_reasoning_fails():
    g = gold(("punctuation:no_comma", {}), ("detectable_format:title", {}))
    body = "<<Plan>>\nWe start at nine and finish at five."
    assert S.score_ifeval(body, g) == 1
    assert S.score_ifeval("<think>\nFirst, commas, commas.\n</think>\n\n" + body, g) == 1   # commas only in reasoning
    assert S.score_ifeval("<think>a</think>\n\n<think>b, c</think>\n\n" + body, g) == 1    # the last </think> wins
    assert S.score_ifeval("<think>\nstill thinking, and never done", g) == 0               # no answer
    assert S.score_ifeval("<think>\nok\n</think>\n\n", g) == 0                              # reasoning only
    assert S.score_ifeval("<think>ok</think>\n\nWe start, then stop.", g) == 0             # answer violates
    # the separator after </think> is stripped, so paragraph indexing starts at the answer
    g2 = gold(("length_constraints:nth_paragraph_first_word", {"num_paragraphs": 2, "nth_paragraph": 1, "first_word": "first"}))
    assert S.score_ifeval("<think>x</think>\n\nFirst part.\n\nSecond part.", g2) == 1
    # without think tags the text is scored exactly as generated (strict): a leading blank paragraph counts
    assert S.score_ifeval("First part.\n\nSecond part.", g2) == 1
    assert S.score_ifeval("\n\nFirst part.\n\nSecond part.", g2) == 0


def test_empty_responses_fail():
    g = gold(("punctuation:no_comma", {}))
    for pred in ("", "   \n", None):
        assert S.score_ifeval(pred, g) == 0


def test_record_wrapper_fails_closed_on_unfinished_output():
    g = gold(("punctuation:no_comma", {}))
    base = {"pred": "No commas here.", "gold": g}
    ids = S.response_ids("No commas here.")                       # real b32 ids: the scorer decodes gen_ids, not pred
    assert S.score_ifeval_record({**base, "gen_ids": ids + [151645, 151643, 151643], "n_gen_to_eos": len(ids) + 1}) == 1
    assert S.score_ifeval_record({**base, "gen_ids": ids, "n_gen_to_eos": len(ids)}) == 0            # never hit EOS
    # filler after the first EOS is normal block-diffusion output and must NOT reject the answer
    assert S.score_ifeval_record({**base, "gen_ids": ids + [151645, 99, 151643], "n_gen_to_eos": len(ids) + 1}) == 1
    bad = S.response_ids("Commas, here.")
    assert S.score_ifeval_record({**base, "gen_ids": bad + [151645], "n_gen_to_eos": len(bad) + 1}) == 0
    assert S.score_ifeval_record(base) == 1                                                          # pred-only row


def test_gold_validation_raises():
    with pytest.raises(KeyError):
        S.score_ifeval("x", gold(("not:an_id", {})))
    with pytest.raises(ValueError):
        S.score_ifeval("x", json.dumps({"instruction_id_list": ["punctuation:no_comma"], "kwargs": []}))
    with pytest.raises(ValueError):
        S.score_ifeval("x", gold(("keywords:existence", {"keywords": None})))


def test_deterministic_and_leaves_caller_rng_alone():
    import langdetect
    assert langdetect.DetectorFactory.seed == 0
    g = gold(("change_case:english_lowercase", {}), ("keywords:existence", {"keywords": ["river"]}))
    pred = "we walked along the river at dawn and talked about the weather for a long time."
    random.seed(123)
    state = random.getstate()
    assert [S.score_ifeval(pred, g) for _ in range(5)] == [1] * 5
    assert random.getstate() == state
    # a kwarg left out would be drawn from `random`; the fixed build seed makes even that reproducible
    a = S.build_instruction("keywords:existence", {}).get_instruction_args()
    random.seed(999)
    assert S.build_instruction("keywords:existence", {}).get_instruction_args() == a


def test_eos_ids_match_protocol():
    import research_protocol as P
    assert tuple(P.EOS_IDS) == S.EOS_IDS


def test_checker_quirks_behind_the_id_exclusions():
    """Pinned behaviour of the vendored checker that build_if_pool.EXCLUDED_IDS relies on (precision)."""
    # the text asks for exactly 5 occurrences; the hidden relation "less than" accepts none
    assert S.score_ifeval("No such word here.", gold(("keywords:word_count_different_numbers",
                                                      {"keyword": "prior", "frequency": 5, "relation": "less than"}))) == 1
    # "do not include keyword coast": only ' coast ' between two spaces is caught
    assert S.score_ifeval("We drove to the coast.", gold(("keywords:exclude_word_harder", {"keyword": "coast"}))) == 1
    # "include a palindrome": the one-letter word "I" counts
    assert S.score_ifeval("I went home.", gold(("keywords:palindrome", {}))) == 1
    # the prompt lists six options; "I think yes." is one of them but the checker only knows three
    assert S.score_ifeval("I think yes.", gold(("detectable_format:constrained_response", {}))) == 0
    # n_start=0 is falsy, so the span start is re-drawn at random
    a = S.build_instruction("new:copy_span_idx", {"prompt_to_repeat": "abcdefghij" * 5, "n_start": 0, "n_end": 20}, seed=0)
    b = S.build_instruction("new:copy_span_idx", {"prompt_to_repeat": "abcdefghij" * 5, "n_start": 0, "n_end": 20}, seed=1)
    assert a.get_instruction_args()["n_start"] != 0 or b.get_instruction_args()["n_start"] != 0


def test_frozen_pool_integrity():
    if not os.path.exists(POOL):
        pytest.skip("if_pool.json not built")
    raw = open(POOL, "rb").read()
    assert hashlib.sha256(raw).hexdigest() == open(POOL + ".sha256").read().split()[0]
    spec = json.loads(raw)
    items = spec["items"]
    assert len(items) == spec["selection"]["n_selected"] and items
    excluded = set(spec["id_policy"]["excluded"])
    for it in items:
        ids, _ = S.parse_gold(it["gold"])                              # raises on an unimplemented id
        assert 1 <= len(ids) <= 3 and not set(ids) & excluded and len(ids) == it["n_constraints"]
        assert hashlib.sha256(it["prompt"].encode("utf-8")).hexdigest() == it["text_sha256"]
    assert [it["idx"] for it in items] == list(range(len(items)))
    assert len({it["key"] for it in items}) == len(items)
    assert [it["key_sha256"] for it in items] == sorted(it["key_sha256"] for it in items)   # any prefix is uniform
    assert all(hashlib.sha256(it["key"].encode()).hexdigest() == it["key_sha256"] for it in items)
    # a biased order (e.g. dedup keeping the lowest sha of several variants) shows up as a skewed source mix up front
    head = items[:len(items) // 2]
    for src in {it["source"] for it in items}:
        share_all = sum(it["source"] == src for it in items) / len(items)
        share_head = sum(it["source"] == src for it in head) / len(head)
        assert abs(share_head - share_all) < 0.03, (src, share_head, share_all)
