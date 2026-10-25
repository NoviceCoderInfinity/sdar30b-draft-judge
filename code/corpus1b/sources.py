"""Registry of every dataset the 1B corpus reads: training sources, held-out test sets, and the
eval sets every training prompt is decontaminated against.

Each TRAINING source is one dict:
    tag        short id used in pids and file names (no '-' or ':')
    domain     one of common.DOMAINS
    repo       Hugging Face dataset repo id (all public; a few are gated -> HF_TOKEN, read-only)
    config     dataset config / subset name, or None
    split      split name(s) to read
    license    as declared on the dataset card
    bucket     mix bucket inside the domain (QUOTAS below)
    quota      max prompts to take from this source (None = all eligible); filled in hash order
    gen_cap    answer token cap at generation
    template   prompt template name (TEMPLATES below)
    key        answer-key type (common.KEY_TYPES)
    extractor  "module:function" in code/corpus1b/ that yields PROMPT ROWs for this source
    filters    source-specific filters the extractor applies (documentation + extractor input)
    notes      anything the operator must know

Quotas follow the plan (docs/CORPUS_1B_PLAN_2026-09-28.md, "Mix within each domain"); the builder
reports the counts it actually reached per bucket, and a short source is topped up from the next
source of the same bucket (FILL_ORDER).
"""
from __future__ import annotations

from typing import Dict, List

GSM8K_TMPL = "{q}\nPlease reason step by step, and put your final answer within \\boxed{{}}."   # == research_protocol.GSM8K_TMPL
MC_TMPL = ("{q}\n\nPlease reason step by step, and put the letter of your final answer ({letters}) "
           "within \\boxed{{}}.")
STDIO_TMPL = ("{q}\n\nWrite a complete Python 3 program that reads the input from standard input and writes the "
              "answer to standard output. You should submit your final solution in the following format: "
              "```python\n\n```")
# function-level code keeps the round-3 shapes: run_desvote.MBPP_TMPL (60%) / run_desvote.HUMANEVAL_TMPL (40%)
TEMPLATES = {
    "gsm8k": GSM8K_TMPL,          # maths and free-form science: boxed final answer
    "mc": MC_TMPL,                # multiple choice: boxed letter
    "stdio": STDIO_TMPL,          # competitive programming
    "mbpp_or_humaneval": None,    # round-3 opencode shapes, chosen per row by hash (build_opencode_pool.py rule)
    "mbpp": None,                 # run_desvote.MBPP_TMPL
    "raw": None,                  # the user message as-is (IF, chat)
}

# ---------------------------------------------------------------------------
# training sources
# ---------------------------------------------------------------------------
SOURCES: List[Dict] = [
    # ---------------- maths (894K prompts) ----------------
    {"tag": "gsm8k", "domain": "maths", "repo": "openai/gsm8k", "config": "main", "split": "train",
     "license": "MIT", "bucket": "gsm", "quota": None, "gen_cap": 1024, "template": "gsm8k", "key": "math",
     "extractor": "extract_maths:gsm8k",
     "filters": ["drop the 200 frozen calib/valid items (code/configs/prompts.json gsm8k_train)"],
     "notes": "gold = the number after '####' in `answer`"},
    {"tag": "mathtrain", "domain": "maths", "repo": "EleutherAI/hendrycks_math", "config": "ALL_7_SUBJECTS",
     "split": "train", "license": "MIT", "bucket": "math_by_level", "quota": None, "gen_cap": 1024,
     "template": "gsm8k", "key": "math", "extractor": "extract_maths:hendrycks_math",
     "filters": ["never the test split (MATH-500 comes from it)"],
     "notes": "configs: algebra, counting_and_probability, geometry, intermediate_algebra, number_theory, "
              "prealgebra, precalculus; gold = last \\boxed{} of `solution`; `level` 'Level 1'..'Level 5' "
              "-> bucket math_easy (1-3) / math_hard (4-5)"},
    {"tag": "omi2gsm", "domain": "maths", "repo": "nvidia/OpenMathInstruct-2", "config": None, "split": "train",
     "license": "CC-BY-4.0", "bucket": "gsm", "quota": 172_000, "gen_cap": 1024, "template": "gsm8k", "key": "math",
     "extractor": "extract_maths:openmathinstruct2",
     "filters": ["problem_source in {'gsm8k', 'augmented_gsm8k'}", "dedupe on `problem` (many rows per problem)"],
     "notes": "14M solution rows, ~600K unique problems; gold = `expected_answer` (majority vote for augmented)"},
    {"tag": "omi2math", "domain": "maths", "repo": "nvidia/OpenMathInstruct-2", "config": None, "split": "train",
     "license": "CC-BY-4.0", "bucket": "math_easy", "quota": 262_000, "gen_cap": 1024, "template": "gsm8k",
     "key": "math", "extractor": "extract_maths:openmathinstruct2",
     "filters": ["problem_source in {'math', 'augmented_math'}", "dedupe on `problem`"],
     "notes": "MATH-level volume; no level field, so it fills the MATH L1-3 bucket"},
    {"tag": "numina", "domain": "maths", "repo": "AI-MO/NuminaMath-1.5", "config": None, "split": "train",
     "license": "Apache-2.0", "bucket": "math_hard", "quota": 206_000, "gen_cap": 1024, "template": "gsm8k",
     "key": "math", "extractor": "extract_maths:numina",
     "filters": ["answer not in {'proof', 'notfound', ''}", "`source` NOT in NUMINA_OLYMPIAD_SOURCES",
                 "question_type != 'proof'", "problem_is_valid / solution_is_valid not 'no' when present"],
     "notes": "MATH L3-5 difficulty (cn_k12, synthetic_math, synthetic_amc, ...)"},
    {"tag": "deepmath", "domain": "maths", "repo": "zwhe99/DeepMath-103K", "config": None, "split": "train",
     "license": "MIT", "bucket": "math_hard", "quota": None, "gen_cap": 1024, "template": "gsm8k", "key": "math",
     "extractor": "extract_maths:deepmath",
     "filters": ["difficulty >= 8 -> bucket olympiad, else math_hard"],
     "notes": "gold = `final_answer`; the most careful decontamination of any candidate"},
    {"tag": "numinaolym", "domain": "maths", "repo": "AI-MO/NuminaMath-1.5", "config": None, "split": "train",
     "license": "Apache-2.0", "bucket": "olympiad", "quota": 70_000, "gen_cap": 1024, "template": "gsm8k",
     "key": "math", "extractor": "extract_maths:numina",
     "filters": ["`source` in NUMINA_OLYMPIAD_SOURCES", "same answer filters as numina"],
     "notes": "olympiad share is capped: the model rarely solves it (AIME 6.7%) and long solutions hit the cap"},
    {"tag": "omr", "domain": "maths", "repo": "nvidia/OpenMathReasoning", "config": None, "split": "cot",
     "license": "CC-BY-4.0", "bucket": "olympiad", "quota": 64_000, "gen_cap": 1024, "template": "gsm8k",
     "key": "math", "extractor": "extract_maths:openmathreasoning",
     "filters": ["problem_type == 'has_answer_extracted'", "dedupe on `problem` (several solutions per problem)"],
     "notes": "AoPS olympiad problems; gold = `expected_answer`"},

    # ---------------- code (710K prompts) ----------------
    {"tag": "oci", "domain": "code", "repo": "nvidia/OpenCodeInstruct", "config": None, "split": "train",
     "license": "CC-BY-4.0", "bucket": "function", "quota": None, "gen_cap": 512,
     "template": "mbpp_or_humaneval", "key": "asserts", "extractor": "extract_code:opencodeinstruct",
     "filters": ["average_test_score == 1.0", "the round-3 contamination checks vs MBPP-257 asserts and "
                 "HumanEval-164 bodies (build_opencode_pool.contaminated), with positive controls",
                 "prompt + reference answer <= 1,200 tokens", "dedupe on the input text"],
     "notes": "fills whatever the competitive sources leave of the 710K; 60/40 MBPP/HumanEval shapes by hash "
              "(build_opencode_pool.HUMANEVAL_PCT); humaneval-shaped rows keep key type 'asserts' (the tests are "
              "the row's unit_tests, as in round 3)"},
    {"tag": "mbpptrain", "domain": "code", "repo": "google-research-datasets/mbpp", "config": "full",
     "split": ["train", "validation", "prompt"], "license": "CC-BY-4.0", "bucket": "function", "quota": None,
     "gen_cap": 512, "template": "mbpp", "key": "asserts", "extractor": "extract_code:mbpp_train",
     "filters": ["never `test` (our eval)"], "notes": "474 rows"},
    {"tag": "taco", "domain": "code", "repo": "likaixin/TACO-verified", "config": None, "split": "train",
     "license": "MIT", "bucket": "competitive", "quota": None, "gen_cap": 1024, "template": "stdio", "key": "stdio",
     "extractor": "extract_code:taco",
     "filters": ["input_output parses to stdin/stdout cases (fn_name problems -> skip)",
                 "<= 20 cases kept per problem (the first 20; all must pass)",
                 "drop problems whose statement says several outputs are accepted unless a checker exists"],
     "notes": "12,898 problems"},
    {"tag": "apps", "domain": "code", "repo": "codeparrot/apps", "config": None, "split": "train",
     "license": "MIT", "bucket": "competitive", "quota": None, "gen_cap": 1024, "template": "stdio", "key": "stdio",
     "extractor": "extract_code:apps",
     "filters": ["difficulty in {'introductory', 'interview'}", "stdin/stdout problems only"],
     "notes": "the repo uses a loading script: read its train.jsonl file directly (hf_hub_download)"},
    {"tag": "cc", "domain": "code", "repo": "deepmind/code_contests", "config": None, "split": "train",
     "license": "CC-BY-4.0", "bucket": "competitive", "quota": None, "gen_cap": 1024, "template": "stdio",
     "key": "stdio", "extractor": "extract_code:code_contests",
     "filters": ["public_tests + private_tests + generated_tests merged, <= 20 kept",
                 "Codeforces problem ids give exact decontamination against the eval sets"],
     "notes": "13,328 problems"},
    {"tag": "cf", "domain": "code", "repo": "open-r1/codeforces", "config": "verifiable", "split": "train",
     "license": "CC-BY-4.0", "bucket": "competitive", "quota": None, "gen_cap": 1024, "template": "stdio",
     "key": "stdio", "extractor": "extract_code:codeforces",
     "filters": ["`official_tests` complete", "problems with a `generated_checker` keep it as key.checker"],
     "notes": "8,338 problems; the only source that ships checkers for multi-answer problems"},

    # ---------------- science (455K prompts) ----------------
    {"tag": "webinst", "domain": "science", "repo": "TIGER-Lab/WebInstruct-verified", "config": None,
     "split": "train", "license": "Apache-2.0", "bucket": "free", "quota": None, "gen_cap": 1024,
     "template": "gsm8k", "key": "free", "extractor": "extract_science:webinstruct",
     "filters": ["category in {'Physics', 'Chemistry', 'Biology'}", "answer_type != 'Multiple Choice' "
                 "(or map to key mc when the options are in the question)"],
     "notes": "free-form; `answer_type` says how to compare (Float, Integer, Expression, ...)"},
    {"tag": "nemosci_rqa", "domain": "science", "repo": "nvidia/Nemotron-Science-v1", "config": None,
     "split": "RQA", "license": "CC-BY-4.0", "bucket": "free", "quota": None, "gen_cap": 1024,
     "template": "raw", "key": "free", "extractor": "extract_science:nemotron_science",
     "filters": ["gold = last \\boxed{} of the assistant turn", "embedding near-duplicate check vs GPQA"],
     "notes": "52,179 rows; the user turn already asks for a boxed answer (keep it raw)"},
    {"tag": "osr2", "domain": "science", "repo": "nvidia/OpenScienceReasoning-2", "config": None,
     "split": "train", "license": "CC-BY-4.0", "bucket": "free_or_mc", "quota": 150_000, "gen_cap": 1024,
     "template": "raw", "key": "free", "extractor": "extract_science:openscience2",
     "filters": ["`expected_answer` non-empty", "a single-letter answer with lettered options in `input` -> key mc",
                 "embedding near-duplicate check vs GPQA"],
     "notes": "307,862 rows (verified via datasets-server /info 2026-09-28; card/earlier notes said 1.6M -- wrong); "
              "tops up whichever half (free / mc) is short. In a 250-row sample spread across the full offset "
              "range, every row was single-letter-MCQ-shaped, so in practice this source mostly feeds 'mc'"},
    {"tag": "nemosci_mcq", "domain": "science", "repo": "nvidia/Nemotron-Science-v1", "config": None,
     "split": "MCQ", "license": "CC-BY-4.0", "bucket": "mc", "quota": None, "gen_cap": 1024, "template": "raw",
     "key": "mc", "extractor": "extract_science:nemotron_science",
     "filters": ["gold = the letter in the assistant turn's last \\boxed{}", "embedding near-duplicate check vs GPQA"],
     "notes": "174,155 rows, 10 options (A-J)"},
    {"tag": "arc", "domain": "science", "repo": "allenai/ai2_arc", "config": ["ARC-Challenge", "ARC-Easy"],
     "split": "train", "license": "CC-BY-SA-4.0", "bucket": "mc", "quota": None, "gen_cap": 1024,
     "template": "mc", "key": "mc", "extractor": "extract_science:arc",
     "filters": ["answerKey in A-E (numeric keys 1-4 are mapped to A-D)"], "notes": "3,370 rows, easy anchor"},

    # ---------------- instruction following (940K prompts) ----------------
    {"tag": "ifnemo", "domain": "if", "repo": "nvidia/Nemotron-RL-instruction_following", "config": None,
     "split": "train", "license": "ODC-BY", "bucket": "pool", "quota": None, "gen_cap": 1024, "template": "raw",
     "key": "ifeval", "extractor": "extract_if:existing_pools",
     "filters": ["every round-3 build_if_pool.py filter (single turn, gold parses, implemented ids, id policy, "
                 "<= 3 constraints, determinism, feasibility, text consistency, English, dedup, decontam)"],
     "notes": "the three existing pools together supply ~15% of IF"},
    {"tag": "ifmulti", "domain": "if", "repo": "allenai/IF_multi_constraints_upto5", "config": None,
     "split": "train", "license": "ODC-BY", "bucket": "pool", "quota": None, "gen_cap": 1024, "template": "raw",
     "key": "ifeval", "extractor": "extract_if:existing_pools", "filters": ["as ifnemo"], "notes": ""},
    {"tag": "ifrlvr", "domain": "if", "repo": "allenai/RLVR-IFeval", "config": None, "split": "train",
     "license": "ODC-BY", "bucket": "pool", "quota": None, "gen_cap": 1024, "template": "raw", "key": "ifeval",
     "extractor": "extract_if:existing_pools", "filters": ["as ifnemo"], "notes": ""},
    {"tag": "ifcomp", "domain": "if", "repo": "allenai/WildChat-4.8M", "config": None, "split": "train",
     "license": "ODC-BY", "bucket": "composed", "quota": None, "gen_cap": 1024, "template": "raw", "key": "ifeval",
     "extractor": "compose_if:composed",
     "filters": ["English first user turn, non-toxic, not redacted, 20-2,000 characters",
                 "not code / maths / a QA dump / a translation request (build_if_pool task screens)",
                 "disjoint from every chat prompt (normalised text)",
                 "1-3 constraints drawn from the admitted IFEvalG ids, worded exactly as build_description "
                 "states them, feasible for a 1,024-token answer, deterministic across build seeds"],
     "notes": "fills IF up to its target (~85% of the domain)"},

    # ---------------- general chat (342K prompts) ----------------
    {"tag": "wildchat", "domain": "chat", "repo": "allenai/WildChat-1M", "config": None, "split": "train",
     "license": "ODC-BY", "bucket": "wildchat", "quota": 188_000, "gen_cap": 1024, "template": "raw", "key": "chat",
     "extractor": "extract_chat:wildchat",
     "filters": ["language == 'English'", "toxic == False", "redacted == False", "first user turn",
                 "20 <= len(prompt) <= 4,000 characters"], "notes": "~55% of chat"},
    {"tag": "ultrachat", "domain": "chat", "repo": "HuggingFaceH4/ultrachat_200k", "config": None,
     "split": "train_sft", "license": "MIT", "bucket": "ultrachat", "quota": 147_000, "gen_cap": 1024,
     "template": "raw", "key": "chat", "extractor": "extract_chat:ultrachat", "filters": ["`prompt` field"],
     "notes": "~43% of chat"},
    {"tag": "oasst2", "domain": "chat", "repo": "OpenAssistant/oasst2", "config": None, "split": "train",
     "license": "Apache-2.0", "bucket": "oasst", "quota": None, "gen_cap": 1024, "template": "raw", "key": "chat",
     "extractor": "extract_chat:oasst2",
     "filters": ["root messages: role == 'prompter' and parent_id is None", "lang == 'en'"], "notes": "~2% of chat"},
]

NUMINA_OLYMPIAD_SOURCES = ("olympiads", "amc_aime", "aops_forum", "inequalities", "number_theory",
                           "olympiads_ref", "cn_contest")

# mix buckets inside each domain: share of the domain's TARGET_PROMPTS (common.TARGET_PROMPTS)
QUOTAS = {
    "maths": {"gsm": 0.20, "math_easy": 0.30, "math_hard": 0.35, "olympiad": 0.15},
    "code": {"function": 0.80, "competitive": 0.20},     # competitive has only ~40K unique problems: all are taken
    "science": {"free": 0.50, "mc": 0.50},
    "if": {"pool": 0.15, "composed": 0.85},              # the pools take all they have; composed fills the rest
    "chat": {"wildchat": 0.55, "ultrachat": 0.43, "oasst": 0.02},
}
# where a short bucket borrows from (same domain)
FILL_ORDER = {
    "maths": ["math_easy", "math_hard", "gsm", "olympiad"],
    "code": ["function"],
    "science": ["free", "mc"],
    "if": ["composed"],
    "chat": ["ultrachat", "wildchat"],
}

# ---------------------------------------------------------------------------
# held-out test domain (never trained; built by eval_sets.py)
# ---------------------------------------------------------------------------
HELD_OUT = [
    {"tag": "finqa500", "repo": "dreamerdeo/finqa", "split": "test", "n": 500, "license": "CC-BY-4.0 (upstream)",
     "notes": "mirror of ibm-research/finqa; the first 500 test items in hash order"},
    {"tag": "chartqa300", "repo": "nmayorga7/chartqa-tables", "join": "ahmed-masry/ChartQA", "join_on": "imgname",
     "split": "test", "n": 300, "license": "MIT (tables) / GPL-3.0 (questions)",
     "notes": "human-written test questions only; hand-check 100 before use, fallback WikiTableQuestions test"},
]

# ---------------------------------------------------------------------------
# eval sets every training prompt is decontaminated against (exact + 13-gram)
# ---------------------------------------------------------------------------
EVAL_SETS = [
    {"tag": "gsm8k_test", "repo": "openai/gsm8k", "config": "main", "split": "test", "field": "question"},
    {"tag": "math_test", "repo": "EleutherAI/hendrycks_math", "config": "ALL_7_SUBJECTS", "split": "test",
     "field": "problem"},
    {"tag": "math500", "repo": "HuggingFaceH4/MATH-500", "config": None, "split": "test", "field": "problem"},
    {"tag": "aime_22_24", "repo": "AI-MO/aimo-validation-aime", "config": None, "split": "train", "field": "problem"},
    {"tag": "aime_25", "repo": "opencompass/AIME2025", "config": ["AIME2025-I", "AIME2025-II"], "split": "test",
     "field": "question", "optional": True},
    {"tag": "gpqa_diamond", "repo": "fingertap/GPQA-Diamond", "config": None, "split": "test", "field": "question"},
    {"tag": "gpqa_all", "repo": "Idavidrein/gpqa", "config": ["gpqa_main", "gpqa_extended", "gpqa_diamond"],
     "split": "train", "field": "Question", "gated": True, "optional": True},
    {"tag": "humaneval", "repo": "openai/openai_humaneval", "config": None, "split": "test", "field": "prompt"},
    {"tag": "mbpp_all", "repo": "google-research-datasets/mbpp", "config": "full",
     "split": ["train", "validation", "test", "prompt"], "field": "text", "only_split_for_decontam": "test"},
    {"tag": "ifeval", "repo": "google/IFEval", "config": None, "split": "train", "field": "prompt"},
    {"tag": "ifbench", "repo": "allenai/IFBench_test", "config": None, "split": "train", "field": "prompt",
     "optional": True},
    {"tag": "finqa_test", "repo": "dreamerdeo/finqa", "config": None, "split": "test", "field": "question"},
    {"tag": "chartqa_test", "repo": "ahmed-masry/ChartQA", "config": None, "split": "test", "field": "query"},
]


def sources_for(domain: str) -> List[Dict]:
    return [s for s in SOURCES if s["domain"] == domain]


def source(tag: str) -> Dict:
    got = [s for s in SOURCES if s["tag"] == tag]
    assert len(got) == 1, tag
    return got[0]


def validate_registry() -> None:
    from common import DOMAINS, KEY_TYPES
    tags = [s["tag"] for s in SOURCES]
    assert len(tags) == len(set(tags)), "duplicate source tags"
    for s in SOURCES:
        assert s["domain"] in DOMAINS, s["tag"]
        assert s["key"] in KEY_TYPES, s["tag"]
        assert "-" not in s["tag"] and ":" not in s["tag"], s["tag"]
        assert s["template"] in TEMPLATES, s["tag"]
        assert s["bucket"] in QUOTAS[s["domain"]] or s["bucket"] in ("math_by_level", "free_or_mc"), s["tag"]
    for d, q in QUOTAS.items():
        assert abs(sum(q.values()) - 1.0) < 1e-9, d
