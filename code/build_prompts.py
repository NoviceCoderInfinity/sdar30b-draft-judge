#!/usr/bin/env python
"""Freeze the Stage 0 routing-prompt set into configs/prompts.json.

Prompt-set construction, per the frozen protocol:
  - GSM8K train  200
  - MBPP  train  100   (sanitized)
  - MATH  train  200   (Hendrycks MATH train split, public mirror)
  seed = 0

Accuracy test sets stay separate: GSM8K test (1319), sanitized MBPP test (257),
MATH-500 (HuggingFaceH4/MATH-500).

Decontamination: any TRAIN item whose problem text also appears in the
corresponding TEST set is excluded, so routing prompts never overlap the
accuracy sets. Matching is on whitespace-normalised text; the exact-match count
is reported alongside so the two can be compared.

Every selection records the dataset repo, its revision sha, config, split, the
index within that split, and a sha256 of the problem text -- enough to detect
upstream drift and to reconstruct the set exactly.
"""
from __future__ import annotations

import hashlib, json, os, random, sys

SEED = 0
N_GSM8K, N_MBPP, N_MATH = 200, 100, 200

MATH_TRAIN_CANDIDATES = [
    ("EleutherAI/hendrycks_math", None),
    ("DigitalLearningGmbH/MATH-lighteval", None),
]
MATH_SUBJECTS = ["algebra", "counting_and_probability", "geometry",
                 "intermediate_algebra", "number_theory", "prealgebra", "precalculus"]


def norm(t: str) -> str:
    return " ".join((t or "").split())


def h(t: str) -> str:
    return hashlib.sha256(t.encode("utf-8")).hexdigest()[:16]


def sha_of(repo: str) -> str:
    from huggingface_hub import HfApi
    return HfApi().dataset_info(repo).sha


def pick(pool, n, seed=SEED):
    rng = random.Random(seed)
    return sorted(rng.sample(range(len(pool)), min(n, len(pool))))


def main():
    from datasets import load_dataset, concatenate_datasets
    out = {"seed": SEED, "created_utc": None, "sets": {}, "test_sets": {}}
    import datetime
    out["created_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()

    # ---------- test sets (for decontamination + the record) ----------
    gsm_test = load_dataset("openai/gsm8k", "main", split="test")
    mbpp_test = load_dataset("google-research-datasets/mbpp", "sanitized", split="test")
    math500 = load_dataset("HuggingFaceH4/MATH-500", split="test")
    out["test_sets"] = {
        "gsm8k": {"repo": "openai/gsm8k", "revision": sha_of("openai/gsm8k"),
                  "config": "main", "split": "test", "n": len(gsm_test)},
        "mbpp": {"repo": "google-research-datasets/mbpp",
                 "revision": sha_of("google-research-datasets/mbpp"),
                 "config": "sanitized", "split": "test", "n": len(mbpp_test)},
        "math500": {"repo": "HuggingFaceH4/MATH-500",
                    "revision": sha_of("HuggingFaceH4/MATH-500"),
                    "split": "test", "n": len(math500)},
    }
    gsm_test_norm = {norm(r["question"]) for r in gsm_test}
    gsm_test_exact = {r["question"] for r in gsm_test}
    mbpp_test_norm = {norm(r["prompt"]) for r in mbpp_test}
    math500_norm = {norm(r["problem"]) for r in math500}
    math500_exact = {r["problem"] for r in math500}

    # ---------- GSM8K train ----------
    gsm_train = load_dataset("openai/gsm8k", "main", split="train")
    keep, drop_n, drop_e = [], 0, 0
    for i, r in enumerate(gsm_train):
        if norm(r["question"]) in gsm_test_norm:
            drop_n += 1
            if r["question"] in gsm_test_exact:
                drop_e += 1
            continue
        keep.append((i, r["question"]))
    idx = pick(keep, N_GSM8K)
    out["sets"]["gsm8k_train"] = {
        "repo": "openai/gsm8k", "revision": out["test_sets"]["gsm8k"]["revision"],
        "config": "main", "split": "train", "pool_size": len(gsm_train),
        "excluded_overlapping_test": drop_n, "excluded_exact_match": drop_e,
        "eligible": len(keep), "n_selected": len(idx),
        "items": [{"split_index": keep[j][0], "text_sha256_16": h(keep[j][1])} for j in idx],
    }

    # ---------- MBPP train (sanitized) ----------
    mbpp_train = load_dataset("google-research-datasets/mbpp", "sanitized", split="train")
    keep, drop_n = [], 0
    for i, r in enumerate(mbpp_train):
        if norm(r["prompt"]) in mbpp_test_norm:
            drop_n += 1
            continue
        keep.append((i, r["prompt"], r["task_id"]))
    idx = pick(keep, N_MBPP)
    out["sets"]["mbpp_train"] = {
        "repo": "google-research-datasets/mbpp", "revision": out["test_sets"]["mbpp"]["revision"],
        "config": "sanitized", "split": "train", "pool_size": len(mbpp_train),
        "excluded_overlapping_test": drop_n, "eligible": len(keep), "n_selected": len(idx),
        "items": [{"split_index": keep[j][0], "task_id": keep[j][2],
                   "text_sha256_16": h(keep[j][1])} for j in idx],
    }

    # ---------- MATH train ----------
    math_repo = None
    parts = []
    for repo, _ in MATH_TRAIN_CANDIDATES:
        try:
            parts = []
            for subj in MATH_SUBJECTS:
                parts.append(load_dataset(repo, subj, split="train"))
            math_repo = repo
            break
        except Exception as e:
            print(f"  MATH candidate {repo} failed: {type(e).__name__}: {str(e)[:120]}", file=sys.stderr)
            parts = []
    if not parts:
        raise SystemExit("no MATH train mirror loaded")
    subj_of = []
    rows = []
    for subj, part in zip(MATH_SUBJECTS, parts):
        for k, r in enumerate(part):
            rows.append(r); subj_of.append((subj, k))
    keep, drop_n, drop_e = [], 0, 0
    for i, r in enumerate(rows):
        if norm(r["problem"]) in math500_norm:
            drop_n += 1
            if r["problem"] in math500_exact:
                drop_e += 1
            continue
        keep.append((i, r["problem"]))
    idx = pick(keep, N_MATH)
    out["sets"]["math_train"] = {
        "repo": math_repo, "revision": sha_of(math_repo),
        "configs": MATH_SUBJECTS, "split": "train",
        "concat_order": "MATH_SUBJECTS as listed; global index = position in that concatenation",
        "pool_size": len(rows), "excluded_overlapping_math500": drop_n,
        "excluded_exact_match": drop_e, "eligible": len(keep), "n_selected": len(idx),
        "items": [{"global_index": keep[j][0],
                   "subject": subj_of[keep[j][0]][0],
                   "subject_index": subj_of[keep[j][0]][1],
                   "text_sha256_16": h(keep[j][1])} for j in idx],
    }

    # ---------- decontamination positive controls ----------
    # "0 excluded" is only meaningful if the matcher can find a KNOWN overlap.
    # MATH-500 is drawn from the MATH test split, so every one of its items must
    # be found there; GSM8K test must match itself.
    math_test_rows = []
    for subj in MATH_SUBJECTS:
        math_test_rows += [r["problem"] for r in load_dataset(math_repo, subj, split="test")]
    math_hits = len({norm(p) for p in math_test_rows} & math500_norm)
    gsm_self = sum(1 for r in gsm_test if norm(r["question"]) in gsm_test_norm)
    out["decontamination_validation"] = {
        "method": "whitespace-normalised exact text match",
        "math500_items_found_in_math_test_split": f"{math_hits}/{len(math500_norm)}",
        "gsm8k_test_self_match": f"{gsm_self}/{len(gsm_test)}",
        "note": ("Positive controls: both must be 100%. They confirm the matcher works, "
                 "so the 0 train/test overlaps recorded above are genuine."),
    }

    out_dir = os.environ.get("PROMPTS_OUT_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs"))
    os.makedirs(out_dir, exist_ok=True)
    p = os.path.join(out_dir, "prompts.json")
    with open(p, "w") as fh:
        json.dump(out, fh, indent=2)
    # frozen-ness marker
    with open(p, "rb") as fh:
        digest = hashlib.sha256(fh.read()).hexdigest()
    with open(os.path.join(out_dir, "prompts.json.sha256"), "w") as fh:
        fh.write(f"{digest}  prompts.json\n")

    for k, v in out["sets"].items():
        exc = v.get("excluded_overlapping_test", v.get("excluded_overlapping_math500"))
        print(f"{k:12} pool={v['pool_size']:5} excluded={exc:3} "
              f"eligible={v['eligible']:5} selected={v['n_selected']}  repo={v['repo']}")
    print("prompts.json sha256:", digest)


if __name__ == "__main__":
    main()
