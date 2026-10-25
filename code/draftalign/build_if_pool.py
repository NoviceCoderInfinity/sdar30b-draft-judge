#!/usr/bin/env python
"""Build the frozen verifiable instruction-following pool: code/configs/if_pool.json (+ .sha256).

Every item is scorable by code/draftalign/if_scorer.py::score_ifeval (strict, all constraints) against the
vendored IFEvalG checker (code/vendor/ifeval/). An item's `correct` then means "the response satisfies every
constraint in `gold`" -- deterministic, program-checked, no judge.

Sources, pinned revisions, read from the HF datasets cache:
  allenai/IF_multi_constraints_upto5 @ 2e3a774  95,373 rows; ground truth = instruction ids + kwargs (python literal)
  allenai/RLVR-IFeval                @ 47c03c7  14,973 rows; one constraint each, func_name + args (open-instruct
                                                if_functions.py); mapped to the IFEval id with the same semantics
                                                or dropped (RLVR_MAPPING)
  nvidia/Nemotron-RL-instruction_following @ 3b25389  46,391 rows; WildChat-1M prompts, IFEvalG ids + kwargs

Filters, in order (per-source counts and reasons are written into the config):
  single_turn     exactly one message, role user, non-empty
  gold            ground truth parses and normalises to {"instruction_id_list", "kwargs"} (no nulls)
  implemented     every id is in the vendored INSTRUCTION_DICT
  id_policy       no id in EXCLUDED_IDS: non-deterministic, checker looser than its own instruction text,
                  non-English output, or degenerate / non-prose output (one reason per id below)
  n_constraints   1..MAX_CONSTRAINTS (3): strict all-constraint pass rates fall with the count
  deterministic   build_description under two RNG seeds gives identical args + description, and every checker
                  runs without raising on probe responses
  feasible        count targets a 1024-token generation can meet (infeasible())
  text_consistent the constraint text is in the prompt verbatim (IF_multi: the `constraint` column, and it equals
                  the checker's own description; RLVR: the filled template; Nemotron: the checker's own description
                  minus its final period, as NVIDIA's template joins them); repeat-prompt target == base prompt
  origin          only open-ended chat origins. IF_multi: its `key` (IF_MULTI_ORIGIN_ADMITTED: WildChat, No Robots,
                  OASST1, Aya). RLVR: exact join of the base prompt to allenai/tulu-v2-sft-mixture's first user turn,
                  whose `dataset` column is the origin (RLVR_ORIGIN_ADMITTED: ShareGPT, WizardLM, GPT-4-Alpaca,
                  OASST1, LIMA). The rest have answers the IF checker cannot verify (maths, code, CoT/FLAN/Orca QA,
                  tables, science), are safety/jailbreak probes, persona prompts, or of unknown provenance
  task_screen     base prompt is code, maths or a QA / multiple-choice dump (TASK_SCREEN regexes; RLVR has no origin)
  unlabelled_demand base prompt itself states a verifiable demand (a count, format, case, start/end, word rule,
                  response language, quoting, P.S., a numbered or ordinal paragraph/section, poetic form) that `gold`
                  does not encode (UNLABELLED_DEMAND)
  english         base prompt (prompt minus the constraint text): >= 20 letters, >= 95% ASCII letters, langdetect
                  (seeded, deterministic) top language 'en' with p >= 0.90, and not a translation request
  dedup           one item per normalised prompt and per normalised base prompt (RLVR reuses a base prompt with
                  several constraints); the survivor is the lowest sha256("if_pool/dedup/" + key), a hash independent
                  of the pool order, so dedup does not bias which sources sit early in the order
  length          research_protocol.encode_prompt with the b32 tokenizer <= MAX_PROMPT_TOKENS (600)
  decontam        normalised exact match or containment against google/IFEval (541), allenai/IFBench_test (300)
                  and the six eval sets of run_desvote.build_items (templated prompts and raw question text)
then eligible items are ordered by sha256(key) and the first --n are frozen; idx = position, so any prefix
if_pool:0:N is a uniform sample of the eligible pool.

normalise(s) = NFKC, lower-case, every whitespace character removed.

  HF_HOME=/root/.cache/huggingface /venv/main/bin/python code/draftalign/build_if_pool.py --n 6000
"""
from __future__ import annotations

import argparse
import ast
import bisect
import collections
import datetime
import hashlib
import json
import os
import random
import re
import sys
import unicodedata
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.abspath(os.path.join(HERE, ".."))
os.environ.setdefault("HF_HOME", "/root/.cache/huggingface")        # where the IF sources are cached
sys.path.insert(0, CODE)

import research_protocol as P                                        # noqa: E402  (sets HF_*_OFFLINE=1)
from draftalign import if_scorer as S                                # noqa: E402  (seeds langdetect)

import langdetect                                                    # noqa: E402

SOURCES = {
    "if_multi": {"repo": "allenai/IF_multi_constraints_upto5", "revision": "2e3a77407b7fce69f95b248d64a884e3ae1c2423",
                 "split": "train", "rows": 95373, "license": "odc-by"},
    "rlvr": {"repo": "allenai/RLVR-IFeval", "revision": "47c03c73621c4aab2b824b7818681117d662770e",
             "split": "train", "rows": 14973, "license": "odc-by"},
    # WildChat-1M prompts + open-instruct IFEvalG constraints (dataset card); one JSONL file, read directly because
    # its kwargs column has mixed types that `datasets` cannot cast
    "nemotron_rl_if": {"repo": "nvidia/Nemotron-RL-instruction_following",
                       "revision": "3b253899665cb71334bb54c14eb5d91751beaad7", "file": "instruction_following.jsonl",
                       "rows": 46391, "license": "odc-by"},
}
IF_EVALS = {
    "ifeval": {"repo": "google/IFEval", "revision": "966cd89545d6b6acfd7638bc708b98261ca58e84", "split": "train",
               "rows": 541},
    "ifbench_test": {"repo": "allenai/IFBench_test", "revision": "2e8a48de45ff3bf41242f927254ca81b59ca3ae2",
                     "split": "train", "rows": 300},
}
# RLVR-IFeval prompts = a tulu-v2-sft-mixture prompt + one appended constraint; the mixture's `dataset` column gives
# the origin (exact normalised join on the first user turn matched 14,973 / 14,973 RLVR rows, 2026-09-14)
TULU_V2 = {"repo": "allenai/tulu-v2-sft-mixture", "revision": "6248b175d2ccb5ec7c4aeb22e6d8ee3b21b2c752", "split": "train",
           "rows": 326154, "license": "odc-by"}
RLVR_ORIGIN_ADMITTED = {
    "sharegpt": "real user conversations with ChatGPT",
    "wizardlm": "evolved open-ended instructions",
    "gpt4_alpaca": "general instructions (Alpaca prompts, GPT-4 answers)",
    "oasst1": "human chat (OpenAssistant)",
    "lima": "curated open-ended prompts",
}
RLVR_ORIGIN_EXCLUDED = {
    "cot": "chain-of-thought QA with reference answers", "flan_v2": "FLAN tasks with reference answers",
    "open_orca": "FLAN-style tasks with reference answers", "code_alpaca": "code",
    "science.*": "scientific extraction with gold answers", "hard_coded": "model-identity canned prompts",
    "(multi-label)": "a prompt shared by several subsets is admitted only if every subset is admitted",
}
# raw text of the six eval sets (run_desvote.build_items reads the same datasets; counts are cross-checked)
EVAL_RAW = {
    "gsm8k": (("openai/gsm8k", "main"), "test", "question"),
    "math500": (("HuggingFaceH4/MATH-500", None), "test", "problem"),
    "mbpp": (("google-research-datasets/mbpp", "sanitized"), "test", "prompt"),
    "aime": (("AI-MO/aimo-validation-aime", None), "train", "problem"),
    "gpqa": (("fingertap/GPQA-Diamond", None), "test", "question"),
    "humaneval": (("openai_humaneval", None), "test", "prompt"),
}
MAX_CONSTRAINTS = 3
MAX_PROMPT_TOKENS = 600
CONTAIN_MIN = 50                      # normalised chars; shorter texts are matched exactly only
DEDUP_SALT = "if_pool/dedup/"         # dedup survivor = lowest sha256(DEDUP_SALT + key); pool order = sha256(key)

EXCLUDED_IDS = {
    # precision / determinism: an admitted pass must mean the instruction text was followed
    "keywords:word_count_different_numbers": "text says 'should appear N times'; the checker tests a hidden relation "
                                             "(< N or >= N), so violating responses pass",
    "detectable_format:constrained_response": "prompt lists six options; the checker knows three and accepts them "
                                              "anywhere in the response",
    "keywords:exclude_word_harder": "checker only catches the word between two spaces (misses it beside punctuation, "
                                    "a newline, or at either end)",
    "keywords:palindrome": "any one-character token is a palindrome, so almost every response passes",
    "new:copy_span_idx": "n_start=0 is re-drawn at random ('if not n_start'); the slice excludes the end index the "
                         "text says to include",
    "keywords:keyword_specific_position": "positions are counted over nltk tokens, punctuation included, which the "
                                          "text does not say",
    # output language
    "language:response_language": "asks for a non-English response",
    # output shape: compliant responses are copies or degenerate / non-prose text
    "copy:copy": "response is a verbatim copy of the prompt",
    "copy:copying_simple": "response is a verbatim copy of the prompt",
    "copy:copying_multiple": "response is the prompt copied N times",
    "copy:repeat_phrase": "response repeats a fixed proverb with one-word substitutions",
    "detectable_format:sentence_hyphens": "all sentences joined by hyphens without spaces",
    "detectable_format:square_brackets": "every word in square brackets",
    "detectable_format:bigram_wrapping": "every word bigram wrapped in << >>",
    "keywords:no_adjacent_consecutive": "no two adjacent words may start with consecutive letters",
    "count:count_unique": "no token, punctuation included, may repeat",
    "count:lowercase_counting": "at most 2-3 lowercase words in the whole response",
    "count:counting_composition": "3 paragraphs x exact sentence count x exactly 2-3 words per sentence",
    "letters:letter_counting": "thresholds of 2-3 letters: trivial ('at least') or a near-empty reply ('less than')",
    "first_word:first_word_sent": "every sentence must start with the same word",
    "last_word:last_word_sent": "every sentence must end with the same word",
    "keywords:start_end": "response must start and end with the same token, no closing punctuation",
    "punctuation:punctuation_dot": "no '.' anywhere: no sentence ends, decimals or URLs",
    "detectable_format:json_format": "JSON output, not prose",
}

# RLVR-IFeval func_name -> IFEval id with the same semantics; None = no faithful equivalent in the checker
RLVR_MAPPING = {
    "verify_keywords": "keywords:existence",
    "validate_forbidden_words": "keywords:forbidden_words",
    "validate_highlighted_sections": "detectable_format:number_highlighted_sections",
    "validate_sections": "detectable_format:multiple_sections",
    "verify_postscript": "detectable_content:postscript",
    "validate_placeholders": "detectable_content:number_placeholders",
    "verify_bullet_points": "detectable_format:number_bullet_lists",
    "validate_title": "detectable_format:title",
    "validate_two_responses": "combination:two_responses",
    "validate_uppercase": "change_case:english_capital",
    "validate_lowercase": "change_case:english_lowercase",
    "validate_end": "startend:end_checker",
    "validate_quotation": "startend:quotation",
    "validate_no_commas": "punctuation:no_comma",
    "validate_json_format": "detectable_format:json_format",
    "validate_paragraphs": "length_constraints:nth_paragraph_first_word",
    "validate_word_constraint": "length_constraints:number_words",          # 'around' has no equivalent
    "verify_sentence_constraint": "length_constraints:number_sentences",     # 'around' has no equivalent
    "validate_frequency_capital_words": "change_case:capital_word_frequency",  # 'around' has no equivalent
    "verify_keyword_frequency": None,    # exactly N occurrences; the checker only has < / >=
    "verify_letter_frequency": None,     # exactly N occurrences; the checker only has < / >=
    "verify_paragraph_count": None,      # text uses a '* * *' divider; the checker splits on '***'
    "validate_choice": None,             # free-form option lists; the checker has three fixed options
    "validate_repeat_prompt": None,      # gold original_prompt includes the sentence the text says not to repeat
}
RLVR_TEMPLATE_FIELDS = {"{N}": "N", "{section splitter}": "section_splitter", "{postscript marker}": "postscript_marker",
                        "{word}": "word", "{letter}": "letter", "{end phrase}": "end_phrase", "{first word}": "first_word",
                        "{i}": "i", "{options}": "options"}

IF_MULTI_ORIGIN_ADMITTED = {          # open-ended chat / instruction prompts (IF_multi `key` origin)
    "ai2-adapt-dev/tulu_v3.9_wildchat_100k": "real user chat requests (WildChat)",
    "ai2-adapt-dev/no_robots_converted": "human-written instructions (No Robots)",
    "oasst1": "human chat (OpenAssistant)",
    "ai2-adapt-dev/tulu_v3.9_aya_100k": "Aya prompts; only the English ones survive the language filter",
}
IF_MULTI_ORIGIN_EXCLUDED = {          # documentation of the known origins the allowlist leaves out
    "ai2-adapt-dev/numinamath_tir_math_decontaminated": "maths: the answer is not verified by the IF checker",
    "ai2-adapt-dev/tulu_v3.9_open_math_2_gsm8k_50k": "maths (GSM8K-style): answer unverified",
    "personas_math_easy": "maths word problems: answer unverified",
    "ai2-adapt-dev/evol_codealpaca_heval_decontaminated": "code: correctness unverified; code is its own slice",
    "personahub": "persona prompts: maths/code problems (sampled survivors of every screen still were), or persona-IF "
                  "prompts that carry their own unlabelled constraints",
    "ai2-adapt-dev/tulu_v3.9_wildjailbreak_decontaminated_50k": "adversarial / harmful requests",
    "ai2-adapt-dev/tulu_v3.9_synthetic_finalresp_wildguardmixtrain_decontaminated_50k": "safety / harmful requests",
    "ai2-adapt-dev/tulu_v3.9_table_gpt_5k": "table tasks with a single correct answer",
    "ai2-adapt-dev/flan_v2_converted": "FLAN tasks with reference answers",
    "science.*": "scientific extraction tasks with gold answers",
    "hard_coded": "model-identity canned prompts",
    "(any other)": "unknown provenance (e.g. bare 7-character keys)",
}

_CODE_LANG = (r"python|javascript|typescript|java|c\+\+|c#|\.net|golang|rust|php|ruby|sql|html|css|bash|shell script|regex|"
              r"matlab|swift|kotlin|scala|perl|haskell|vba|excel formula|jquery|react|django|pandas|numpy")
_CODE_WORD = r"code|function|script|program|class|method|query|snippet|algorithm|implementation|compile|library|api"
_LANG_TOK = rf"(?<![\w+#.])(?:{_CODE_LANG})(?![\w+#])"          # \b fails after '#' / '+' (C#, C++)
TASK_SCREEN = [                       # base prompts whose task is code, maths or a QA / multiple-choice dump
    ("code_fence", re.compile(r"```")),
    ("code_syntax", re.compile(r"\bdef\s+\w+\s*\(|#include\s*<|\bpublic\s+(?:static\s+)?(?:void|class|int)\b|"
                               r"\bconsole\.log\b|</?(?:div|span|html|body|script|table)\b|"
                               r"(?m:^\s*(?:using\s+[\w.]+\s*;|import\s+[\w.*]+(?:\s+as\s+\w+)?\s*;?\s*$|from\s+[\w.]+\s+import\s+[\w*, ]+$|"
                               r"package\s+[\w.]+\s*;))")),
    ("code_request", re.compile(rf"(?i){_LANG_TOK}[^\n]{{0,80}}\b(?:{_CODE_WORD})\b|"
                                rf"\b(?:{_CODE_WORD})\b[^\n]{{0,80}}{_LANG_TOK}|"
                                r"\b(?:sample|source|example|following|write|provide|modify|debug|fix|refactor)\s+"
                                r"(?:the\s+|a\s+|some\s+|this\s+|my\s+)?code\b")),
    ("code_generic", re.compile(r"(?i)\b(?:write|create|make|build|give me|provide|generate|develop|implement)\s+(?:me\s+)?"
                                r"(?:a|an|the|some)?\s*(?:[\w-]+\s+){0,2}(?:script|program|function|macro|regex|code|bot|"
                                r"api|plugin)s?\b")),
    ("code_mention", re.compile(r"(?i)(?<![\w+#.])(?:python|javascript|typescript|c\+\+|c#|php|sql|html|css|golang|kotlin|"
                                r"matlab|haskell|jquery|django|numpy|pandas|powershell|latex|json|yaml|regex|node\.js|"
                                r"react\.js|vue\.js|\.net)(?![\w+#])")),
    ("sql", re.compile(r"(?i)\bselect\b[^\n]{0,200}\bfrom\s+\w")),
    ("latex_math", re.compile(r"\\\(|\\\[|\\frac|\\sqrt|\\sum|\\int|\$[^$\n]{1,200}\$")),
    ("arithmetic", re.compile(r"\d\s*[+*/^=×÷]\s*[\d(]")),
    ("math_words", re.compile(r"(?i)\b(?:solve|calculate|compute|equation|integral|derivative)\b")),
    ("qa_dump", re.compile(r"(?im)^\s*(?:q|a|question|answer|options?|choices|answer choices|premise|hypothesis)\s*[:：]")),
    ("qa_words", re.compile(r"(?i)\b(?:true or false|multiple choice|which of the following|"
                            r"stream of (?:thought|consciousness)|what (?:was|is|might be) the question|the answer is|"
                            r"step-by-step reasoning)\b")),
    ("fewshot_rule", re.compile(r"(?m)(?:\+{4,}|\*{4,}|={4,}|#{4,})")),
]
CODE_LINE = re.compile(r"(?m)[;{}]\s*$")
CODE_LINES_MAX = 2                    # 3+ lines ending in ; { or } is source code

# Verifiable demands written into the BASE prompt itself (not in `gold`, so the checker would not test them):
# an item whose response ignored them could still score 1. Drop the item.
_NUM = (r"(?!(?:1[89]|20)\d\d\b)\d+|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|fifteen|twenty|thirty|"
        r"forty|fifty|hundred")                                       # 1800-2099 read as years, not counts
_UNIT = (r"words?|sentences?|paragraphs?|lines?|bullets?|bullet[- ]points?|characters?|chars|letters?|syllables?|"
         r"items?|points?|sections?|stanzas?|verses?|tweets?|hashtags?|emojis?|questions?|steps?|tips?|examples?|"
         r"ideas?|reasons?|names?|titles?|options?|slogans?|taglines?|headlines?|pages?|chapters?|scenes?|rules?|"
         r"facts?|ways?|things?|sentences")
UNLABELLED_DEMAND = [
    ("count", re.compile(rf"(?i)\b(?:{_NUM})(?:[\s-]+[\w'’]+){{0,2}}?[\s-]+(?:{_UNIT})\b")),
    ("single_unit", re.compile(r"(?i)\b(?:one|a single|single)[\s-]+(?:word|sentence|paragraph|line|tweet)\b")),
    ("format", re.compile(r"(?i)\b(?:json|yaml|xml|csv|markdown)\b|\b(?:in|as|using|use|with|into)\s+(?:a\s+|the\s+)?"
                          r"(?:table|tabular|bullet(?:ed)?(?:[\s-]points?|\s+list)?|numbered\s+list|list\s+format)\b|"
                          r"\b(?:create|make|build|generate|provide|present|draw|using|use|in|into|as)\s+(?:a|an|the)?\s*"
                          r"(?:[\w-]+\s+){0,2}table\b|\bdata format\s*:")),
    ("case_punct", re.compile(r"(?i)\b(?:all caps|all[\s-]capital|capital letters|upper[\s-]?case|lower[\s-]?case|"
                              r"no commas?|without (?:any\s+)?(?:commas?|punctuation)|no punctuation|exclamation marks?)\b")),
    ("start_end", re.compile(r"(?i)\b(?:start|begin|end|finish|conclude|close)(?:s|ing)?\s+(?:(?:your|the|each|every)\s+"
                             r"[\w-]+\s+)?with\s+(?:the\s+|a\s+)?(?:word|phrase|sentence|line|question|quote|letter)\b")),
    ("word_rules", re.compile(r"(?i)\b(?:do not|don't|never|avoid|without)\s+(?:use|using|include|including|mention|"
                              r"mentioning|say|saying)\s+(?:the\s+|any\s+)?(?:words?|letters?|phrases?|terms?)\b|"
                              r"\b(?:include|use|mention|contain)s?\s+(?:the\s+)?(?:keywords?|words?|phrases?|terms?)\s*"
                              r"[:\"'“‘]")),
    ("response_language", re.compile(r"(?i)\b(?:respond|reply|answer|write|written)(?:\s+[\w-]+){0,3}\s+in\s+(?:english|"
                                     r"french|spanish|german|chinese|japanese|korean|russian|arabic|hindi|portuguese|"
                                     r"italian|dutch|turkish)\b")),
    ("wrap_quote", re.compile(r"(?i)\b(?:wrap|enclose|surround)\w*\b[^\n]{0,40}\b(?:quotation marks?|quotes|"
                              r"brackets|parenthes[ie]s)\b|\b(?:p\.\s?s\.|postscript)\b|\brepeat (?:the|this|my) "
                              r"(?:request|question|prompt)\b")),
    ("structure_ref", re.compile(r"(?i)\b(?:paragraph|section|part|chapter|stanza|verse|bullet(?: point)?|sentence)\s+"
                                 r"(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten)\b|\b(?:first|second|third|"
                                 r"fourth|fifth|final|last|opening|closing)\s+(?:paragraph|section|sentence|line|stanza|"
                                 r"verse|bullet(?: point)?)s?\b")),
    ("poetic_form", re.compile(r"(?i)\b(?:haiku|limerick|sonnet|acrostic|villanelle|tanka|palindrome|anagram|"
                               r"rhym(?:e|es|ing))\b")),
]

TRANSLATION = re.compile(r"(?i)\btranslat(?:e|ion|ing)\b|\b(?:in|to|into) (?:french|spanish|german|italian|portuguese|"
                         r"chinese|japanese|korean|russian|arabic|hindi|turkish|czech|finnish|romanian|dutch|polish|"
                         r"vietnamese|thai|indonesian|hebrew|greek|swedish|danish|norwegian)\b")


def normalise(s: str) -> str:
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", s)).lower()


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def load_hf(spec: dict, config=None):
    from datasets import load_dataset
    ds = load_dataset(spec["repo"], config, revision=spec["revision"], split=spec["split"])
    assert len(ds) == spec["rows"], (spec["repo"], len(ds), spec["rows"])
    return ds


def load_jsonl(spec: dict):
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(spec["repo"], spec["file"], repo_type="dataset", revision=spec["revision"])
    rows = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
    assert len(rows) == spec["rows"], (spec["repo"], len(rows), spec["rows"])
    return rows


def dump_gold(ids, kwargs) -> str:
    return json.dumps({"instruction_id_list": list(ids), "kwargs": list(kwargs)}, sort_keys=True, ensure_ascii=False)


# ---------------------------------------------------------------------------
# ground truth
# ---------------------------------------------------------------------------
def if_multi_gold(row: dict):
    gt = ast.literal_eval(row["ground_truth"])
    if not (isinstance(gt, list) and len(gt) == 1 and isinstance(gt[0], dict)):
        return None, "shape"
    ids, kws = gt[0].get("instruction_id"), gt[0].get("kwargs")
    if not (isinstance(ids, list) and isinstance(kws, list) and ids and len(ids) == len(kws)):
        return None, "shape"
    pieces = row["constraint"].split("\t")
    if len(pieces) != len(ids):
        return None, "constraint_text_count"
    kws = [{k: v for k, v in (kw or {}).items() if v is not None} for kw in kws]
    return (list(ids), kws, pieces), ""


def nemotron_gold(row: dict):
    g = row["ground_truth"]
    ids, kws = g.get("instruction_id_list"), g.get("kwargs")
    if isinstance(kws, str):
        kws = json.loads(kws)
    if not (isinstance(ids, list) and isinstance(kws, list) and ids and len(ids) == len(kws)):
        return None, "shape"
    kws = [{k: v for k, v in (kw or {}).items() if v is not None} for kw in kws]
    return (list(ids), kws, None), ""          # constraint texts = the checker's own descriptions (text_consistent)


def rlvr_gold(row: dict):
    g = json.loads(row["ground_truth"])
    f = g.get("func_name")
    a = {k: v for k, v in g.items() if k != "func_name" and v is not None}
    if f not in RLVR_MAPPING:
        return None, f"unknown_func:{f}"
    iid = RLVR_MAPPING[f]
    if iid is None:
        return None, f"no_equivalent:{f}"
    q = a.get("quantifier")
    if f in ("validate_word_constraint", "verify_sentence_constraint", "validate_frequency_capital_words"):
        if q == "at least":
            rel, n = "at least", int(a["N"])
        elif q == "at most":
            rel, n = "less than", int(a["N"]) + 1              # count <= N  <=>  count < N + 1
        else:
            return None, f"no_equivalent:{f}:{q}"
        kw = {"validate_word_constraint": {"relation": rel, "num_words": n},
              "verify_sentence_constraint": {"relation": rel, "num_sentences": n},
              "validate_frequency_capital_words": {"capital_relation": rel, "capital_frequency": n}}[f]
    else:
        kw = {"verify_keywords": lambda: {"keywords": list(a["keyword_list"])},
              "validate_forbidden_words": lambda: {"forbidden_words": list(a["forbidden_words"])},
              "validate_highlighted_sections": lambda: {"num_highlights": int(a["N"])},
              "validate_sections": lambda: {"section_spliter": a["section_splitter"], "num_sections": int(a["N"])},
              "verify_postscript": lambda: {"postscript_marker": a["postscript_marker"]},
              "validate_placeholders": lambda: {"num_placeholders": int(a["N"])},
              "verify_bullet_points": lambda: {"num_bullets": int(a["N"])},
              "validate_end": lambda: {"end_phrase": a["end_phrase"]},
              "validate_paragraphs": lambda: {"num_paragraphs": int(a["N"]), "nth_paragraph": int(a["i"]),
                                              "first_word": a["first_word"]},
              }.get(f, lambda: {})()
    # the constraint sentence exactly as the template put it into the prompt
    text = row["constraint"]
    for ph, key in RLVR_TEMPLATE_FIELDS.items():
        if ph in text and g.get(key) is not None:
            text = text.replace(ph, str(g[key]))
    if f == "verify_keywords":
        text = text.replace("{keyword1}, {keyword2}", ", ".join(a["keyword_list"]))
    if f == "validate_forbidden_words":
        text = text.replace("{forbidden words}", ", ".join(a["forbidden_words"]))
    text = text.replace("at least / around / at most", str(q))
    return ([iid], [kw], [text]), ""


# ---------------------------------------------------------------------------
# checks
# ---------------------------------------------------------------------------
PROBES = ("Hello world.",
          "<<Title>>\n\nSection 1\nFirst, a point. Second point!\n***\n* bullet one\n- bullet two\n\n[name]\n\nP.S. done",
          "\"THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG\"")


def _build(iid, kw, seed):
    inst = S.INSTRUCTION_DICT[iid](iid)
    state = random.getstate()
    random.seed(seed)
    try:
        desc = inst.build_description(**kw)
    finally:
        random.setstate(state)
    return inst, desc


_DET_CACHE: dict = {}


def determinism(iid: str, kw: dict):
    """'' if (iid, kw) builds identically under two seeds and its checker survives the probes, else a reason."""
    ck = (iid, json.dumps(kw, sort_keys=True))
    if ck in _DET_CACHE:
        return _DET_CACHE[ck]
    try:
        a, da = _build(iid, kw, 0)
        b, db = _build(iid, kw, 1)
    except Exception:
        _DET_CACHE[ck] = f"build_raised:{iid}"
        return _DET_CACHE[ck]
    if a.get_instruction_args() != b.get_instruction_args() or da != db:
        _DET_CACHE[ck] = f"args_differ:{iid}"
        return _DET_CACHE[ck]
    try:
        for p in PROBES:
            _build(iid, kw, 0)[0].check_following(p)
    except Exception:
        _DET_CACHE[ck] = f"check_raised:{iid}"
        return _DET_CACHE[ck]
    _DET_CACHE[ck] = ""
    return ""


def infeasible(iid: str, kw: dict) -> bool:
    """Count targets a 1024-token response cannot reasonably meet (or that force an empty reply)."""
    rel = kw.get("relation")
    if iid == "length_constraints:number_words":
        return (rel == "at least" and kw["num_words"] > 500) or (rel == "less than" and kw["num_words"] < 10)
    if iid == "length_constraints:number_sentences":
        return (rel == "at least" and kw["num_sentences"] > 25) or (rel == "less than" and kw["num_sentences"] < 2)
    if iid == "length_constraints:number_paragraphs":
        return not 1 <= kw["num_paragraphs"] <= 10
    if iid == "length_constraints:nth_paragraph_first_word":
        return not 1 <= kw["nth_paragraph"] <= kw["num_paragraphs"] <= 10
    if iid == "detectable_format:multiple_sections":
        return not 1 <= kw["num_sections"] <= 10
    if iid == "detectable_format:number_bullet_lists":
        return not 1 <= kw["num_bullets"] <= 10
    if iid == "detectable_format:number_highlighted_sections":
        return kw["num_highlights"] > 10
    if iid == "detectable_content:number_placeholders":
        return kw["num_placeholders"] > 10
    if iid == "keywords:frequency":
        return rel == "at least" and kw["frequency"] > 10
    if iid in ("keywords:letter_frequency", "letters:letter_counting2"):
        return kw["let_relation"] == "at least" and kw["let_frequency"] > 30
    if iid == "change_case:capital_word_frequency":
        return kw["capital_relation"] == "at least" and kw["capital_frequency"] > 40
    return False


def if_multi_origin(key: str) -> str:
    if key.startswith("personahub"):
        return "personahub"
    if key.startswith("science."):
        return "science.*"
    return key.rsplit("_", 1)[0]


def strip_constraints(prompt: str, pieces) -> str:
    base = prompt
    for piece in pieces:
        i = base.rfind(piece)
        if i >= 0:
            base = base[:i] + base[i + len(piece):]
    return base.strip()


def english_reason(base: str) -> str:
    letters = [c for c in base if c.isalpha()]
    if len(letters) < 20:
        return "too_short"
    if sum(c.isascii() for c in letters) / len(letters) < 0.95:
        return "non_ascii"
    try:
        top = langdetect.detect_langs(base)[0]
    except langdetect.LangDetectException:
        return "undetectable"
    if top.lang != "en" or top.prob < 0.90:
        return "not_en"
    if TRANSLATION.search(base):
        return "translation"
    return ""


def pct(values, qs=(0, 10, 50, 90, 95, 99, 100)):
    v = sorted(values)
    if not v:
        return {}
    return {f"p{q}": v[min(len(v) - 1, int(round(q / 100 * (len(v) - 1))))] for q in qs}


def task_screen_reason(base: str) -> str:
    """'' unless `base` (a base prompt, constraint text already stripped) is code, maths or a QA / multiple-choice
    dump (TASK_SCREEN regexes, or more than CODE_LINES_MAX lines that look like source-code statements)."""
    for name, rx in TASK_SCREEN:
        if rx.search(base):
            return f"screen:{name}"
    if len(CODE_LINE.findall(base)) > CODE_LINES_MAX:
        return "screen:code_lines"
    return ""


def unlabelled_demand_reason(base: str) -> str:
    """'' unless `base` itself states a verifiable demand (UNLABELLED_DEMAND regexes) that a `gold` built
    separately from it would not encode."""
    for name, rx in UNLABELLED_DEMAND:
        if rx.search(base):
            return f"demand:{name}"
    return ""


# ---------------------------------------------------------------------------
def eligible_items(workers: int = 32):
    """Run every round-3 filter (this module's docstring, "Filters" list) through decontamination and return the
    full eligible pool -- in dedup-survivor order (sha256(DEDUP_SALT + key) ascending) -- BEFORE the final
    sha256(key) sort and --n cut that main() applies on top. Loads all three SOURCES together: dedup is
    cross-source (a duplicate prompt's survivor can come from if_multi, rlvr or nemotron_rl_if regardless of
    which one a caller ultimately wants), so a caller after a single source's rows must filter `cands`
    afterward -- corpus1b/extract_if.existing_pools does exactly this, once per process (the pipeline below is
    the expensive part: it loads and screens ~156K candidates plus the 326K-row tulu-v2-sft-mixture join and the
    eval-set decontamination texts).

    Returns (cands, stages, controls, eval_texts, length_before):
      cands         the eligible pool (list of candidate dicts: src, source, row, key, key_sha256, orig_key,
                    ids, kwargs, pieces, base, origin, prompt, prompt_tokens, ...)
      stages        collections.OrderedDict of every stage's before/after/dropped-reasons counts (main()'s
                    spec["filters"])
      controls      the decontamination positive-control hit rates (main()'s spec["decontamination"]["positive_controls"])
      eval_texts    {eval set name -> list of texts} decontaminated against (main()'s spec["decontamination"]["eval_sets"] counts)
      length_before [c["prompt_tokens"] for c in cands] BEFORE the length filter (main()'s spec["length_fit"]["before_filter"])
    """
    stages = collections.OrderedDict()

    def record(name, before, kept, reasons=None):
        by_src = lambda xs: dict(collections.Counter(c["src"] for c in xs))
        stages[name] = {"in": len(before), "kept": len(kept), "in_by_source": by_src(before),
                        "kept_by_source": by_src(kept),
                        **({"dropped_reasons": dict(sorted(reasons.items(), key=lambda kv: -kv[1]))} if reasons else {})}
        print(f"[{name:15s}] {len(before):6d} -> {len(kept):6d}  {by_src(kept)}", flush=True)

    # -- load ---------------------------------------------------------------------------------------------------
    cands = []
    for src, spec in SOURCES.items():
        if src == "nemotron_rl_if":
            for i, r in enumerate(load_jsonl(spec)):
                cands.append({"src": src, "source": spec["repo"], "row": i, "key": f"{src}/{i:06d}",
                              "orig_key": str(r["id"]), "messages": [{"role": "user", "content": r["prompt"]}],
                              "ground_truth": {"instruction_id_list": r["instruction_id_list"], "kwargs": r["kwargs"]},
                              "constraint": None})
            continue
        ds = load_hf(spec)
        for i, r in enumerate(ds):
            cands.append({"src": src, "source": spec["repo"], "row": i, "key": f"{src}/{i:06d}",
                          "orig_key": r.get("key"), "messages": r["messages"], "ground_truth": r["ground_truth"],
                          "constraint": r["constraint"]})
    stages["loaded"] = {"rows": len(cands), "by_source": dict(collections.Counter(c["src"] for c in cands))}

    def run(name, cs, fn):
        kept, reasons = [], collections.Counter()
        for c in cs:
            why = fn(c)
            if why:
                reasons[why] += 1
            else:
                kept.append(c)
        record(name, cs, kept, reasons)
        return kept

    def single_turn(c):
        m = c["messages"]
        if len(m) != 1 or m[0].get("role") != "user" or not str(m[0].get("content") or "").strip():
            return "not_single_user_turn"
        c["prompt"] = m[0]["content"]
        return ""

    def gold_stage(c):
        try:
            parsed, why = {"if_multi": if_multi_gold, "rlvr": rlvr_gold, "nemotron_rl_if": nemotron_gold}[c["src"]](c)
        except Exception as e:                                        # noqa: BLE001
            return f"unparseable:{type(e).__name__}"
        if parsed is None:
            return why
        c["ids"], c["kwargs"], c["pieces"] = parsed
        return ""

    def implemented(c):
        missing = [i for i in c["ids"] if i not in S.INSTRUCTION_DICT]
        return f"unimplemented:{missing[0]}" if missing else ""

    def id_policy(c):
        bad = [i for i in c["ids"] if i in EXCLUDED_IDS]
        return f"excluded:{bad[0]}" if bad else ""

    def n_constraints(c):
        return "" if 1 <= len(c["ids"]) <= MAX_CONSTRAINTS else f"n={len(c['ids'])}"

    def deterministic(c):
        for iid, kw in zip(c["ids"], c["kwargs"]):
            why = determinism(iid, kw)
            if why:
                return why
        return ""

    def feasible(c):
        for iid, kw in zip(c["ids"], c["kwargs"]):
            if infeasible(iid, kw):
                return f"infeasible:{iid}"
        return ""

    def text_consistent(c):
        p = c["prompt"]
        if c["src"] == "nemotron_rl_if":
            # NVIDIA's templates insert the checker's own descriptions, joined by " and " and without the final period
            ws = " ".join(p.split())
            pieces = []
            for iid, kw in zip(c["ids"], c["kwargs"]):
                d = " ".join(_build(iid, kw, 0)[1].split()).rstrip(".")
                if d not in ws:
                    return f"description_not_in_prompt:{iid}"
                pieces.append(d)
            c["pieces"], c["base"] = pieces, strip_constraints(ws, pieces)
            return ""
        if any(piece not in p for piece in c["pieces"]):
            return "constraint_text_not_in_prompt"
        c["base"] = strip_constraints(p, c["pieces"])
        if c["src"] == "if_multi":
            for iid, kw, piece in zip(c["ids"], c["kwargs"], c["pieces"]):
                desc = _build(iid, kw, 0)[1]
                if " ".join(desc.split()) != " ".join(piece.split()):
                    return f"description_mismatch:{iid}"
        for iid, kw in zip(c["ids"], c["kwargs"]):
            if iid == "combination:repeat_prompt" and kw["prompt_to_repeat"].strip() != c["base"]:
                return "repeat_prompt_target_is_not_the_base_prompt"
        return ""

    def origin(c):
        if c["src"] == "nemotron_rl_if":
            c["origin"] = "wildchat-1m"                                  # every row, per the dataset card
            return ""
        if c["src"] == "if_multi":
            o = if_multi_origin(c["orig_key"])
            c["origin"] = o
            return "" if o in IF_MULTI_ORIGIN_ADMITTED else f"origin:{o if o in IF_MULTI_ORIGIN_EXCLUDED else '(any other)'}"
        labels = tulu_v2_origin.get(normalise(c["base"]))
        if not labels:
            c["origin"] = "(unmatched)"
            return "origin:rlvr:(unmatched)"
        c["origin"] = "|".join(sorted(labels))
        return "" if all(l in RLVR_ORIGIN_ADMITTED for l in labels) else f"origin:rlvr:{c['origin']}"

    def task_screen(c):
        return task_screen_reason(c["base"])

    def unlabelled_demand(c):
        return unlabelled_demand_reason(c["base"])

    tulu_v2_origin = collections.defaultdict(set)                     # normalised first user turn -> {dataset}
    tv = load_hf(TULU_V2)
    for d, msgs in zip(tv["dataset"], tv["messages"]):
        u = next((mm["content"] for mm in msgs if mm["role"] == "user"), None)
        if u is not None:
            tulu_v2_origin[normalise(u)].add("science.*" if d.startswith("science.") else d)
    del tv

    cands = run("single_turn", cands, single_turn)
    cands = run("gold", cands, gold_stage)
    cands = run("implemented", cands, implemented)
    cands = run("id_policy", cands, id_policy)
    cands = run("n_constraints", cands, n_constraints)
    cands = run("deterministic", cands, deterministic)
    cands = run("feasible", cands, feasible)
    cands = run("text_consistent", cands, text_consistent)
    cands = run("origin", cands, origin)
    cands = run("task_screen", cands, task_screen)
    cands = run("unlabelled_demand", cands, unlabelled_demand)

    with Pool(workers) as pool:                                        # fork: langdetect seed is inherited
        en = pool.map(english_reason, [c["base"] for c in cands], chunksize=64)
    before, kept, reasons = cands, [], collections.Counter()
    for c, why in zip(cands, en):
        (reasons.update([why]) if why else kept.append(c))
    record("english", before, kept, reasons)
    cands = kept

    for c in cands:
        c["key_sha256"] = sha256_text(c["key"])
    # The duplicate that survives is chosen by an INDEPENDENT hash. Keeping the lowest sha256(key) of m variants
    # would put multi-variant sources (RLVR) at the front of the sha256(key) order, so a prefix would not be uniform.
    cands.sort(key=lambda c: sha256_text(DEDUP_SALT + c["key"]))
    seen_p, seen_b = set(), set()

    def dedup(c):
        np_, nb = normalise(c["prompt"]), normalise(c["base"])
        if np_ in seen_p:
            return "duplicate_prompt"
        if nb in seen_b:                               # one constraint variant per base task
            return "duplicate_base_prompt"
        seen_p.add(np_)
        seen_b.add(nb)
        return ""
    cands = run("dedup", cands, dedup)

    # -- length -------------------------------------------------------------------------------------------------
    from transformers import AutoTokenizer
    m = P.MODELS["b32"]
    tok = AutoTokenizer.from_pretrained(m["repo"], revision=m["revision"], cache_dir=m["cache_dir"],
                                        local_files_only=True, trust_remote_code=True)
    for c in cands:
        c["prompt_tokens"] = int(P.encode_prompt(tok, c["prompt"]).shape[1])
    length_before = [c["prompt_tokens"] for c in cands]
    cands = run("length", cands, lambda c: "" if c["prompt_tokens"] <= MAX_PROMPT_TOKENS else "prompt_tokens>600")

    # -- decontamination ----------------------------------------------------------------------------------------
    from run_desvote import build_items
    eval_texts = collections.OrderedDict()                              # set name -> list of texts
    for name, spec in IF_EVALS.items():
        eval_texts[name] = [r["prompt"] for r in load_hf(spec)]
    from datasets import load_dataset
    for name, ((repo, cfg), split, field) in EVAL_RAW.items():
        raw = [r[field] for r in load_dataset(repo, cfg, split=split)]
        templated = [it["prompt"] for it in build_items(name)]
        assert len(raw) == len(templated), (name, len(raw), len(templated))
        eval_texts[name] = raw + templated
    norm_eval = {k: [normalise(t) for t in v] for k, v in eval_texts.items()}
    exact_index = {}
    for k, v in norm_eval.items():
        for t in v:
            exact_index.setdefault(t, k)

    def contam_hits(texts):
        """For normalised pool texts -> {i: (method, eval set)} by exact match, eval-in-pool or pool-in-eval."""
        hits = {}
        for i, t in enumerate(texts):
            for tt in t:
                if tt in exact_index:
                    hits.setdefault(i, ("exact", exact_index[tt]))
        sep = "\x00"
        big = sep.join(t[0] for t in texts)
        starts, pos = [], 0
        for t in texts:
            starts.append(pos)
            pos += len(t[0]) + 1
        for k, v in norm_eval.items():
            for e in v:
                if len(e) < CONTAIN_MIN:
                    continue
                j = big.find(e)
                while j >= 0:
                    hits.setdefault(bisect.bisect_right(starts, j) - 1, ("eval_in_pool", k))
                    j = big.find(e, j + 1)
        ebig = {k: sep.join(v) for k, v in norm_eval.items()}
        for i, t in enumerate(texts):
            b = t[1]
            if len(b) >= CONTAIN_MIN:
                for k, eb in ebig.items():
                    if b in eb:
                        hits.setdefault(i, ("pool_base_in_eval", k))
                        break
        return hits

    texts = [(normalise(c["prompt"]), normalise(c["base"])) for c in cands]
    hits = contam_hits(texts)
    kept, reasons = [], collections.Counter()
    for i, c in enumerate(cands):
        if i in hits:
            reasons[f"{hits[i][0]}:{hits[i][1]}"] += 1
        else:
            kept.append(c)
    record("decontam", cands, kept, reasons)
    cands = kept
    # positive controls: the matcher must find what is planted
    ife = eval_texts["ifeval"]
    gsm = eval_texts["gsm8k"][:50]
    planted = ([(normalise(t), normalise(t)) for t in ife] +
               [(normalise(t + " Your response should contain at least 3 sentences."), normalise(t)) for t in ife] +
               [(normalise("Answer briefly. " + q + " Do not use commas."), normalise("Answer briefly. " + q)) for q in gsm])
    ph = contam_hits(planted)
    controls = {"ifeval_self_exact": f"{sum(1 for i in range(len(ife)) if i in ph)}/{len(ife)}",
                "ifeval_plus_appended_constraint": f"{sum(1 for i in range(len(ife), 2 * len(ife)) if i in ph)}/{len(ife)}",
                "gsm8k_test_question_embedded": f"{sum(1 for i in range(2 * len(ife), len(planted)) if i in ph)}/{len(gsm)}"}
    print("[decontam controls]", controls, flush=True)

    return cands, stages, controls, eval_texts, length_before


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=6000)
    ap.add_argument("--out", default=os.path.join("code", "configs", "if_pool.json"))
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--dry-run", action="store_true", help="print the stage counts, write nothing")
    args = ap.parse_args(argv)
    out_path = P.safe_out(args.out)

    cands, stages, controls, eval_texts, length_before = eligible_items(args.workers)
    m = P.MODELS["b32"]

    # -- selection ----------------------------------------------------------------------------------------------
    cands.sort(key=lambda c: c["key_sha256"])
    selected = cands[:args.n]

    def mix(xs):
        return {"by_source": dict(collections.Counter(c["source"] for c in xs)),
                "by_n_constraints": dict(sorted(collections.Counter(len(c["ids"]) for c in xs).items())),
                "by_instruction_id": dict(collections.Counter(i for c in xs for i in c["ids"]).most_common()),
                "by_origin": dict(collections.Counter(f"{c['src']}:{c['origin']}" for c in xs).most_common()),
                "prompt_tokens": pct([c["prompt_tokens"] for c in xs])}

    items = [{"idx": k, "key": c["key"], "key_sha256": c["key_sha256"], "source": c["source"],
              "source_row": c["row"], "orig_key": c["orig_key"], "origin": c["origin"], "prompt": c["prompt"],
              "gold": dump_gold(c["ids"], c["kwargs"]), "n_constraints": len(c["ids"]),
              "prompt_tokens": c["prompt_tokens"], "text_sha256": sha256_text(c["prompt"])}
             for k, c in enumerate(selected)]
    spec = {
        "name": "if_pool",
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "builder": "code/draftalign/build_if_pool.py",
        "builder_sha256": hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest(),
        "scorer": {"path": "code/draftalign/if_scorer.py", "function": "score_ifeval(pred, gold)",
                   "sha256": hashlib.sha256(open(S.__file__, "rb").read()).hexdigest()},
        "checker": {"vendored": "code/vendor/ifeval/open_instruct/IFEvalG",
                    "upstream": "https://github.com/allenai/open-instruct @ 172e379eada34137885bd1c36543c4f59cb113e4",
                    "VENDORED.sha256": hashlib.sha256(open(os.path.join(CODE, "vendor", "ifeval", "VENDORED.sha256"),
                                                           "rb").read()).hexdigest()},
        "sources": SOURCES,
        "gold_format": "JSON string {\"instruction_id_list\": [...], \"kwargs\": [{...}]} (google/IFEval input format, "
                       "null kwargs removed, keys sorted)",
        "prompt": "the dataset's single user message verbatim; research_protocol.encode_prompt applies the chat template",
        "normalise": "NFKC, lower-case, all whitespace removed (dedup and decontamination)",
        "filters": stages,
        "id_policy": {"admitted": sorted(set(S.INSTRUCTION_DICT) - set(EXCLUDED_IDS)), "excluded": EXCLUDED_IDS,
                      "max_constraints": MAX_CONSTRAINTS,
                      "rlvr_mapping": {k: v for k, v in RLVR_MAPPING.items()},
                      "rlvr_mapping_rules": "at most N -> relation 'less than', N+1 (integer counts); 'around' dropped",
                      "rlvr_origin": {"join": f"{TULU_V2['repo']}@{TULU_V2['revision']}: normalised base prompt == "
                                              "normalised first user turn; origin = the mixture's `dataset`",
                                      "admitted": RLVR_ORIGIN_ADMITTED, "excluded": RLVR_ORIGIN_EXCLUDED},
                      "if_multi_origin_admitted": IF_MULTI_ORIGIN_ADMITTED,
                      "if_multi_origin_excluded": IF_MULTI_ORIGIN_EXCLUDED,
                      "task_screen": {**{n: rx.pattern for n, rx in TASK_SCREEN},
                                      "code_lines": f"more than {CODE_LINES_MAX} lines matching {CODE_LINE.pattern}"},
                      "unlabelled_demand": {n: rx.pattern for n, rx in UNLABELLED_DEMAND}},
        "language_filter": {"method": "on the base prompt (prompt minus its constraint text): >= 20 letters, >= 95% of "
                                      "letters ASCII, langdetect 1.0.9 detect_langs with DetectorFactory.seed = 0 "
                                      "(deterministic) top language 'en' at p >= 0.90, no translation request",
                            "translation_regex": TRANSLATION.pattern},
        "length_fit": {"tokenizer": f"{m['repo']}@{m['revision']} via research_protocol.encode_prompt",
                       "max_prompt_tokens": MAX_PROMPT_TOKENS, "gen_len": 1024,
                       "before_filter": pct(length_before), "selected": pct([c["prompt_tokens"] for c in selected])},
        "decontamination": {"method": "normalised exact match of prompt or base prompt against every eval text; "
                                      f"containment both ways for texts >= {CONTAIN_MIN} normalised chars",
                            "eval_sets": {k: len(v) for k, v in eval_texts.items()},
                            "if_eval_revisions": IF_EVALS,
                            "hits": stages["decontam"].get("dropped_reasons", {}), "positive_controls": controls},
        "selection": {"order": "sha256(key) ascending; idx = position. The eligible pool is a uniformly random "
                               "permutation of its keys, so every prefix if_pool:0:N is a uniform sample",
                      "dedup_survivor": f"lowest sha256({DEDUP_SALT!r} + key), independent of the order hash",
                      "n_requested": args.n, "n_eligible": len(cands), "n_selected": len(selected),
                      "eligible": mix(cands), "selected": mix(selected),
                      "slices": {f"0:{k}": mix(selected[:k]) for k in (500, 1000, 2000, 4000) if k < len(selected)}},
        "items": items,
    }
    print(json.dumps({k: spec["selection"][k] for k in ("n_eligible", "n_selected")}), flush=True)
    print(json.dumps(spec["selection"]["selected"], indent=1)[:3000], flush=True)
    if args.dry_run:
        return spec
    raw = json.dumps(spec, indent=1, ensure_ascii=False).encode("utf-8")
    with open(out_path, "wb") as fh:
        fh.write(raw)
    sha = hashlib.sha256(raw).hexdigest()
    with open(out_path + ".sha256", "w") as fh:
        fh.write(f"{sha}  {os.path.basename(out_path)}\n")
    print(f"wrote {out_path} ({len(items)} items) sha256 {sha}", flush=True)
    return spec


if __name__ == "__main__":
    main()
