"""Decontamination index for the 1B corpus (docs/CORPUS_1B_BUILD.md): every training prompt is checked
against every eval set before it is allowed into a prompt list.

`DecontamIndex` is built once per build_prompt_lists.py run from `load_eval_texts()` -- every
sources.EVAL_SETS entry, every sources.HELD_OUT entry's question text, and
HuggingFaceH4/ultrachat_200k split=test_sft `prompt` (the chat eval's own decontam set, per the build
brief; not part of sources.EVAL_SETS because it is not itself an eval CELL, just text the chat domain
must not leak). Loading is offline (research_protocol sets HF_*_OFFLINE=1 on import); an EVAL_SETS entry
marked optional=True (or gated=True) that is not in the local cache is skipped with a printed warning, not
a hard failure -- decontam coverage is then reported as reduced, never silently assumed complete.

sources.EVAL_SETS' `only_split_for_decontam` is honoured: mbpp_all lists split=[train,validation,test,
prompt] (every split some *other* consumer needs -- e.g. extract_code.py's mbpptrain source legitimately
trains on train/validation/prompt), but only its `test` split is an eval set a training prompt must not
duplicate. Missing that field would wrongly flag every legitimate MBPP training row as contamination.

Three checks, in increasing cost:
  (a) exact      normalize_text(prompt) == normalize_text(eval_text)                      O(1) dict lookup
  (b) containment normalize_text(eval_text) is a substring of normalize_text(prompt),       see flag_batch
                  for eval texts >= 40 normalised chars (a handful of words is too short to
                  be a meaningful leak signal; also keeps the containment pass's candidate
                  list, and so its cost, down)
  (c) 13-gram     prompt and an eval text share a run of 13 consecutive lower-cased          O(len(prompt))
                  \\w+ word tokens (hashed to a 64-bit int so the set never holds Python str
                  objects -- see MEMORY below)

(a) and (c) are checked per prompt in `flag()` / `flag_batch()`, O(1) and O(len(prompt)) respectively --
this is what scales to millions of prompts. (b) is the expensive one: a naive "for each prompt: for each
eval text: substring search" is O(n_prompts * n_eval_texts), intractable once n_eval_texts is in the tens
of thousands (this build's eval texts, dominated by ultrachat_200k test_sft, number ~35-40K). flag_batch
instead adapts draftalign/build_if_pool.py's trick: concatenate the WHOLE prompt batch into one string
(NUL-separated, with a prefix-sum of offsets) and, for each containment-eligible eval text, do repeated
`str.find` over that one big string, mapping hit offsets back to a prompt index with `bisect`. That makes
the containment pass O(n_eval_texts * batch_chars) -- one scan per eval text instead of one scan per
(prompt, eval text) pair. Call flag_batch with domain-sized chunks (build_prompt_lists.py uses the whole
domain's candidate list, up to ~1-2M rows); pass check_containment=False for a fast exact+13-gram-only pass
if that cost turns out to be prohibitive on the training box (this was designed, not benchmarked at full
corpus scale -- there is no million-row corpus to time it against yet).

MEMORY: the 13-gram set holds one Python dict entry (int64 hash -> str eval tag) per unique gram across
every eval text. ~38K eval texts at a rough 60-120 word-tokens each give on the order of 1-3M (13-gram)
windows before accounting for overlap; a CPython dict of that size with int keys and short str values is
on the order of a few hundred MB (int objects ~28 bytes, dict slots ~50-100 bytes each at typical load
factor) -- easily fits in a training box's RAM, but would not scale another 100x without moving to a
packed numpy array of sorted hashes (not done here: the task's own eval-set count does not need it).

POSITIVE CONTROL: `DecontamIndex.positive_control()` re-checks every eval text against the index built
from it and raises AssertionError -- the build must refuse -- if any eval text fails to flag itself. This
is the sanity check that the three matchers actually work, not just that they compile.
"""
from __future__ import annotations

import bisect
import hashlib
import os
import re
import sys
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))       # code/corpus1b
CODE = os.path.dirname(HERE)                             # code/
for _p in (os.path.join(CODE, "compat"), CODE, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import common                                              # noqa: E402
import sources                                               # noqa: E402

GRAM_N = 13
CONTAIN_MIN_CHARS = 40
# HELD_OUT entries don't carry an EVAL_SETS-style {config, split, field}; this is the question-text source
# for each (see module docstring: it reads the full underlying test split, a superset of the eventual
# curated held-out selection eval_sets.py builds -- conservative and safe for decontamination purposes).
HELD_OUT_TEXT_SOURCE = {
    "finqa500": {"repo": "dreamerdeo/finqa", "config": None, "split": "test", "field": "question"},
    "chartqa300": {"repo": "ahmed-masry/ChartQA", "config": None, "split": "test", "field": "query"},
}
CHAT_EVAL_EXTRA = {"tag": "ultrachat_test_sft", "repo": "HuggingFaceH4/ultrachat_200k", "config": None,
                   "split": "test_sft", "field": "prompt"}


# ---------------------------------------------------------------------------
# text helpers
# ---------------------------------------------------------------------------
def word_tokens(s: str) -> List[str]:
    return re.findall(r"\w+", s.lower())


def hashed_grams(tokens: Sequence[str], n: int = GRAM_N) -> Iterable[int]:
    """64-bit int per n-gram of `tokens` (sha256-based, so it is stable across processes/PYTHONHASHSEED)."""
    for i in range(len(tokens) - n + 1):
        gram = " ".join(tokens[i:i + n])
        yield int(hashlib.sha256(gram.encode("utf-8")).hexdigest()[:16], 16)


# ---------------------------------------------------------------------------
# eval text loading (mockable: tests monkeypatch `_load_field_texts`)
# ---------------------------------------------------------------------------
def _load_field_texts(repo: str, config, split: str, field: str) -> List[str]:
    from datasets import load_dataset
    ds = load_dataset(repo, config, split=split)
    return [r[field] for r in ds if r.get(field)]


def _split_names(spec: dict) -> List[str]:
    if "only_split_for_decontam" in spec:
        return [spec["only_split_for_decontam"]]
    s = spec["split"]
    return list(s) if isinstance(s, (list, tuple)) else [s]


def _configs(spec_config) -> List[Optional[str]]:
    if spec_config == "ALL_7_SUBJECTS":                  # a sentinel extract_maths.py expands, not an HF config name
        from extract_maths import HENDRYCKS_MATH_SUBJECTS
        return list(HENDRYCKS_MATH_SUBJECTS)
    return list(spec_config) if isinstance(spec_config, (list, tuple)) else [spec_config]


def _load_tagged(tag: str, repo: str, config, split_names: List[str], field: str,
                  warnings: List[str]) -> List[str]:
    texts: List[str] = []
    if repo == "dreamerdeo/finqa":                       # a loading script only: read the upstream JSON (finqa_local)
        import finqa_local
        for sp in split_names:
            try:
                texts.extend(str(r[field]) for r in finqa_local.rows(sp) if r.get(field))
            except Exception as e:                          # noqa: BLE001
                warnings.append(f"{tag} ({repo} split={sp}): {type(e).__name__}: {e}")
        return texts
    for cfg in _configs(config):
        for sp in split_names:
            try:
                texts.extend(_load_field_texts(repo, cfg, sp, field))
            except Exception as e:                          # noqa: BLE001 -- optional/gated/uncached sets must not abort
                warnings.append(f"{tag} ({repo} config={cfg} split={sp}): {type(e).__name__}: {e}")
    return texts


def load_eval_texts(include_optional: bool = True) -> Dict[str, List[str]]:
    """tag -> raw (un-normalised) text list, for every EVAL_SETS + HELD_OUT entry plus the chat decontam
    extra (see module docstring). A tag that loads 0 texts is left out of the result; required (non-
    optional) tags that load 0 texts are still warned about loudly on stderr."""
    out: Dict[str, List[str]] = {}
    warnings: List[str] = []
    for spec in sources.EVAL_SETS:
        if spec.get("optional") and not include_optional:
            continue
        texts = _load_tagged(spec["tag"], spec["repo"], spec["config"], _split_names(spec), spec["field"], warnings)
        if texts:
            out[spec["tag"]] = texts
        elif not spec.get("optional"):
            warnings.append(f"{spec['tag']}: 0 texts loaded (a REQUIRED eval set -- decontam coverage is incomplete)")

    for tag, spec in HELD_OUT_TEXT_SOURCE.items():
        texts = _load_tagged(tag, spec["repo"], spec["config"], [spec["split"]], spec["field"], warnings)
        if texts:
            out[tag] = texts

    extra = CHAT_EVAL_EXTRA
    texts = _load_tagged(extra["tag"], extra["repo"], extra["config"], [extra["split"]], extra["field"], warnings)
    if texts:
        out[extra["tag"]] = texts

    for w in warnings:
        print(f"[decontam] WARNING: {w}", file=sys.stderr)
    return out


# ---------------------------------------------------------------------------
# the index
# ---------------------------------------------------------------------------
class DecontamIndex:
    def __init__(self, eval_texts: Dict[str, List[str]]):
        self.eval_texts = eval_texts
        self._exact: Dict[str, str] = {}
        self._contain: List[Tuple[str, str]] = []            # (normalised eval text, tag), len >= CONTAIN_MIN_CHARS
        self._grams: Dict[int, str] = {}
        for tag, texts in eval_texts.items():
            for t in texts:
                norm = common.normalize_text(t)
                if not norm:
                    continue
                self._exact.setdefault(norm, tag)
                if len(norm) >= CONTAIN_MIN_CHARS:
                    self._contain.append((norm, tag))
                for g in hashed_grams(word_tokens(t)):
                    self._grams.setdefault(g, tag)

    @classmethod
    def build(cls, include_optional: bool = True) -> "DecontamIndex":
        return cls(load_eval_texts(include_optional=include_optional))

    def n_eval_texts(self) -> int:
        return sum(len(v) for v in self.eval_texts.values())

    # -- cheap, per-item: exact + 13-gram -----------------------------------------------------------------
    def flag(self, prompt: str) -> Optional[str]:
        """"exact:<tag>" / "13gram:<tag>" / None. O(len(prompt)); does not check containment (see flag_batch)."""
        norm = common.normalize_text(prompt)
        tag = self._exact.get(norm)
        if tag is not None:
            return f"exact:{tag}"
        for g in hashed_grams(word_tokens(prompt)):
            tag = self._grams.get(g)
            if tag is not None:
                return f"13gram:{tag}"
        return None

    # -- the full 3-tier check, batched for the containment pass ------------------------------------------
    def flag_batch(self, prompts: Sequence[str], check_containment: bool = True) -> List[Optional[str]]:
        """One verdict per prompt (None = clean). See the module docstring for why containment is batched."""
        out: List[Optional[str]] = [self.flag(p) for p in prompts]
        if not check_containment or not self._contain:
            return out
        pending = [i for i, v in enumerate(out) if v is None]
        if not pending:
            return out
        norm_pending = [common.normalize_text(prompts[i]) for i in pending]
        hits = self._containment_hits(norm_pending)
        for local_i, tag in hits.items():
            out[pending[local_i]] = f"containment:{tag}"
        return out

    def _containment_hits(self, norm_prompts: List[str]) -> Dict[int, str]:
        """local index (into `norm_prompts`) -> eval tag, for every eval text found as a substring of that
        prompt. One `str.find` pass per containment-eligible eval text over the whole concatenated batch."""
        sep = "\x00"
        starts: List[int] = []
        pos = 0
        for t in norm_prompts:
            starts.append(pos)
            pos += len(t) + 1
        big = sep.join(norm_prompts)
        n_workers = min(int(os.environ.get("C1B_DECONTAM_WORKERS", "48")), os.cpu_count() or 1, len(self._contain))
        if n_workers <= 1 or len(big) < int(os.environ.get("C1B_DECONTAM_PAR_MIN_CHARS", "50000000")):
            hits: Dict[int, str] = {}
            for etext, tag in self._contain:
                j = big.find(etext)
                while j >= 0:
                    i = bisect.bisect_right(starts, j) - 1
                    hits.setdefault(i, tag)
                    j = big.find(etext, j + 1)
            return hits
        # one str.find scan per eval text is embarrassingly parallel: fork workers share `big` copy-on-write; each
        # returns prompt -> (first eval index, tag) for its slice and the merge keeps the lowest index, so the
        # verdicts equal the sequential loop's (setdefault = first eval text in self._contain order)
        import multiprocessing as mp
        global _CT_BIG, _CT_STARTS, _CT_CONTAIN
        _CT_BIG, _CT_STARTS, _CT_CONTAIN = big, starts, self._contain
        n = len(self._contain)
        step = max(1, n // (n_workers * 8))
        ranges = [(a, min(n, a + step)) for a in range(0, n, step)]
        best: Dict[int, Tuple[int, str]] = {}
        try:
            with mp.get_context("fork").Pool(n_workers) as pool:
                for part in pool.imap_unordered(_containment_slice, ranges):
                    for i, (k, tag) in part.items():
                        if i not in best or k < best[i][0]:
                            best[i] = (k, tag)
        finally:
            _CT_BIG = _CT_STARTS = _CT_CONTAIN = None
        return {i: tag for i, (k, tag) in best.items()}

    # -- sanity ---------------------------------------------------------------------------------------------
    def positive_control(self) -> Dict[str, str]:
        """Every eval text must flag against its own index -- else the build refuses (AssertionError)."""
        report: Dict[str, str] = {}
        failed: List[str] = []
        for tag, texts in self.eval_texts.items():
            if not texts:
                continue
            verdicts = self.flag_batch(list(texts), check_containment=True)
            n_hit = sum(1 for v in verdicts if v is not None)
            report[tag] = f"{n_hit}/{len(texts)}"
            if n_hit != len(texts):
                failed.append(tag)
        assert not failed, (f"decontam positive control failed for {failed}: an eval set did not fully flag "
                             f"against its own index -- refusing to build on top of a decontam index that "
                             f"cannot even catch its own eval texts. report={report}")
        return report


_CT_BIG: Optional[str] = None
_CT_STARTS: Optional[List[int]] = None
_CT_CONTAIN = None


def _containment_slice(rng: Tuple[int, int]) -> Dict[int, Tuple[int, str]]:
    """Worker: prompt index -> (lowest eval index, tag) over eval texts rng[0]..rng[1]-1 (fork-inherited globals)."""
    big, starts, contain = _CT_BIG, _CT_STARTS, _CT_CONTAIN
    out: Dict[int, Tuple[int, str]] = {}
    for k in range(rng[0], rng[1]):
        etext, tag = contain[k]
        j = big.find(etext)
        while j >= 0:
            i = bisect.bisect_right(starts, j) - 1
            if i not in out:
                out[i] = (k, tag)
            j = big.find(etext, j + 1)
    return out


# ---------------------------------------------------------------------------
# optional embedding near-duplicate check (nemosci_mcq / nemosci_rqa / osr2 vs GPQA)
# ---------------------------------------------------------------------------
def gpqa_embedding_flags(prompts: Sequence[str], gpqa_questions: Sequence[str],
                         model_name: str = "BAAI/bge-small-en-v1.5", threshold: float = 0.92,
                         batch_size: int = 256) -> List[bool]:
    """True per prompt if its cosine similarity to some GPQA question is >= `threshold`. Requires the
    optional `sentence-transformers` package; returns all-False with one clear stderr warning when it is
    not installed (never raises -- the science domain's build must still complete without it)."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("[decontam] WARNING: sentence-transformers is not installed -- skipping the GPQA embedding "
              "near-duplicate check for nemosci_mcq/nemosci_rqa/osr2 (pip install sentence-transformers to "
              "enable it; exact/containment/13-gram decontamination against GPQA still applies)",
              file=sys.stderr)
        return [False] * len(prompts)
    import numpy as np
    model = SentenceTransformer(model_name)
    gpqa_emb = model.encode(list(gpqa_questions), batch_size=batch_size, normalize_embeddings=True,
                            show_progress_bar=False)
    flags: List[bool] = []
    for i in range(0, len(prompts), batch_size):
        chunk = list(prompts[i:i + batch_size])
        emb = model.encode(chunk, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False)
        sims = emb @ np.asarray(gpqa_emb).T                  # cosine similarity: both sides L2-normalised
        flags.extend((sims.max(axis=1) >= threshold).tolist())
    return flags
