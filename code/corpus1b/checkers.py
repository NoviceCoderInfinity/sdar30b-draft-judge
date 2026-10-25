"""Per-item checkers for the 1B-token judge corpus: the qualification gate every decoded training
candidate must pass before it is kept (docs/CORPUS_1B_BUILD.md). CPU-only; the only subprocesses it
starts are the sandboxed programs it is scoring.

  check(key, gen_ids, tok_decode, *, sandbox_dir, gen_cap) -> verdict dict (single item)
  check_many(items, workers=8)                             -> [verdict dict, ...] (same order as items)

verdict dict: {"ok": bool, "reason": str, "loop": bool, "eos": bool, "n_gen": int}

Common gate (every key type, common.KEY_TYPES)
-----------------------------------------------
`gen_ids[:gen_cap]` must contain an EOS id (common.EOS_IDS) -- else {"ok": False, "reason": "no_eos",
"eos": False, "loop": common.has_loop(gen_ids[:gen_cap]), "n_gen": len(gen_ids[:gen_cap])}. There is no
"before EOS" slice to compute loop over in this case, so the statistic falls back to the full capped
window -- it is still a real loop signal (a repeating tail run within the cap trips it), just measured
over one token more than usual.

When an EOS is found at position `i` (0-based, within the capped window): the answer text is
`tok_decode(gen_ids[:i])` (ids strictly BEFORE the EOS); `loop = common.has_loop(gen_ids[:i])`;
`n_gen = i + 1`, i.e. generated tokens up to and INCLUDING the first EOS -- the same convention
common.py's GENERATION RECORD docstring uses for its own `n_gen` field. `loop` is always reported (a
statistic for every domain); only for key type "chat" does it also gate `ok`.

Key-type scoring
-----------------
math      math_postprocess_v2(text) vs key["answer"], MATHEvaluator(v2).is_equiv (exceptions -> False).
          NOTE (verified empirically against the vendored evaluator): v2's is_equiv normalises LaTeX
          spelling (dfrac/tfrac -> frac, \\left/\\right removal, a/b -> \\frac{a}{b}, comma stripping,
          whitespace) but does NOT evaluate expressions numerically -- "\\frac{1}{2}" and "0.5" are
          NOT considered equal under v2 (only under v1). Tests below use verified-true/false pairs, not
          assumed ones.
free      last \\boxed{} (research_protocol.last_boxed; whole answer if unboxed). Numeric parse strips
          \\left/\\right, \\text{...} units, commas, "%", and "a \\times 10^{n}" / "a \\cdot 10^{n}" ->
          "aen"; both sides numeric -> 1% relative tolerance (absolute 1e-9 when |gold| < 1e-9); else
          MATHEvaluator(v2).is_equiv; else case/space-normalised string equality. answer_type starting
          with "bool" (case-insensitive) short-circuits to a yes/y/true vs no/n/false comparison.
mc        last \\boxed{} with LaTeX commands stripped (\\text{E}, \\mathrm{E} -> E), else an "answer: X"
          / lone-trailing-letter fallback (generalises analyze_s1b._gpqa_letter from A-D to A..
          LETTERS[n_options-1], n_options <= 10); exact letter match against key["answer"].
asserts / humaneval
          code = oc_datasets.mbpp.MBPPEvaluator()._process_answer(text) (the vendored extractor).
          asserts:   (key["setup"] + "\\n" if present) + code + "\\n" + key["tests"]
          humaneval: code + "\\n\\n" + key["test"] + "\\n\\ncheck(" + key["entry_point"] + ")\\n"
          (byte-identical to analyze_s1b._humaneval_score's program shape). Executed by the sandboxed
          runner below -- NEVER by oc_datasets.mbpp.execution(), which has no chdir and runs the
          program in the caller's cwd (AGENTS.md 9(p)); ok iff exit code 0.
stdio     the LAST ```python fenced block in `text` (fallback: the whole text if it looks like code --
          contains one of "def ", "import ", "print(", "input(", "sys.stdin"); no code -> reason
          "no_code". Run once per key["cases"] item with case["input"] on stdin (default per-case
          timeout 4s, 60s total budget across all cases; the budget is checked before every case, so a
          slow early case can still leave later cases reason "timeout"). Default comparison: stdout vs
          case["output"], split on whitespace, token-by-token; a token pair that both parse as float
          matches within 1e-6 absolute-or-relative, everything else must match exactly.

          key["checker"], when present, is {"lang": "python"|"cpp", "source": str} -- there is no
          upstream spec for this yet (no extractor writes `checker` today), so this module DEFINES the
          protocol an extractor must produce: the checker is invoked as
          `<program> <input_path> <actual_output_path> <expected_output_path>` (a python checker via
          `python -I checker.py in out ans`, a cpp checker as a compiled binary) with stdin closed; it
          must exit 0 and print "1" (accept) or "0" (reject) as the first stdout token. A cpp checker is
          compiled once with `g++ -O2` into `<sandbox_dir>/_checker_cache/<sha256(source)[:16]>`
          (install is an atomic os.replace, safe under concurrent workers building the same checker); no
          g++ on PATH -> reason "no_checker_runtime". Any other/missing `lang` -> "no_checker_runtime"
          too.
ifeval    draftalign.if_scorer.score_ifeval(text, key["gold"]) (strict, every constraint). A malformed
          gold or an unimplemented instruction id RAISES out of parse_gold by design (if_scorer's own
          docstring: "that is a pipeline bug, not a wrong answer") -- this module does not swallow it.
chat      ok iff eos, not loop, and the text reads as English: langdetect top language "en" with
          p >= 0.9 when the text has >= 20 alphabetic characters, else >= 85% of its alphabetic
          characters are ASCII (a reply with zero alphabetic characters at all is accepted vacuously --
          there is nothing to judge as non-English). reasons "loop", "not_english".

The sandboxed runner
---------------------
Every code execution (asserts / humaneval / stdio / a checker) writes its program into a FRESH
subdirectory of `sandbox_dir` (tempfile.mkdtemp, so concurrent callers never collide), runs
`python -I <script> [argv...]` via subprocess with cwd = that subdirectory, absolute paths only, a wall-
clock `timeout=` (the real, platform-independent bound -- verified to fire correctly on both platforms
tested), and (POSIX only, via preexec_fn, each in its own try/except so an unsupported one is silently
skipped) RLIMIT_AS 2 GiB, RLIMIT_CPU ~= timeout, RLIMIT_NPROC 32 in the child, as defence in depth. RLIMIT_AS
AND RLIMIT_CPU are both skipped outright on macOS (sys.platform == "darwin"): Darwin's dynamic linker
reserves address space far beyond a process's real working set, so a 2 GiB RLIMIT_AS there routinely stops
the interpreter from starting at all; and on the sandboxed macOS shell this module was developed and tested
in, installing an 11 s RLIMIT_CPU on a forked child (confirmed installed by reading it back inside the
child) still got the child SIGXCPU-killed after only ~3-4 REAL seconds of a single-threaded busy loop --
i.e. the CPU-time accounting behind RLIMIT_CPU, not the setrlimit call, misbehaves in that environment,
plausibly because the outer sandbox does not reset per-process CPU-time accounting on fork. RLIMIT_NPROC
was verified NOT to share that problem and stays on for both platforms. The subdirectory is always removed
afterwards, even on a timeout or a launch failure.

check_many
-----------
The EOS/loop/decode gate and the cheap scorers (math, free, mc, ifeval, chat) run directly in the
calling process -- the IPC round trip through a pool would cost more than the check itself. Only
asserts / humaneval / stdio verdicts -- which already fork a real subprocess for the sandbox -- are
farmed out to a ProcessPoolExecutor, so a big shard uses every core without paying pool overhead on the
items that do not need it. Results are reassembled into the input order before returning.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Callable, Dict, List, Optional, Sequence, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))            # code/corpus1b
CODE = os.path.abspath(os.path.join(HERE, ".."))             # code/
for _p in (CODE, os.path.join(CODE, "vendor")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import research_protocol as RP                               # noqa: E402  (sets HF_*_OFFLINE=1; CPU-only, no model load)
from draftalign import if_scorer                              # noqa: E402  (seeds langdetect on import)
from corpus1b import common                                   # noqa: E402
from oc_datasets.math import math_postprocess_v2, MATHEvaluator    # noqa: E402
from oc_datasets.mbpp import MBPPEvaluator                    # noqa: E402

import langdetect                                              # noqa: E402

CODE_KEY_TYPES = ("asserts", "humaneval", "stdio")
LETTERS = "ABCDEFGHIJ"
STDIO_PER_CASE_TIMEOUT = 4.0
STDIO_TOTAL_TIMEOUT = 60.0
ASSERT_TIMEOUT = 10.0
CHILD_MEM_BYTES = 2 * 1024 ** 3


# ---------------------------------------------------------------------------
# the sandboxed runner
# ---------------------------------------------------------------------------
def _child_limits(timeout: float, mem_bytes: int, nproc: int) -> Callable[[], None]:
    """-> a preexec_fn that applies best-effort resource limits in the forked child. Each limit is set in
    its own try/except. RLIMIT_AS and RLIMIT_CPU are both skipped on macOS (see module docstring); the
    wall-clock `subprocess.run(timeout=...)` in `_run_sandboxed` is the real, platform-independent bound
    in every case -- these are defence in depth, not the primary guarantee. RLIMIT_CPU verified BAD on
    macOS here, not just "unverified": on this sandboxed darwin shell, a forked child with RLIMIT_CPU set
    to a value well above the intended wall-clock timeout was still SIGXCPU-killed a few seconds in --
    reproduced with `resource.getrlimit` read back inside the child confirming the limit was installed
    as requested, so the accounting itself (not the setrlimit call) is what misbehaves here, plausibly
    because this shell's outer sandbox does not reset per-process CPU-time accounting on fork. RLIMIT_NPROC
    was verified NOT to share that problem (a plain busy loop under RLIMIT_NPROC alone ran to its full
    wall-clock timeout), so it stays enabled on every platform."""

    def _apply():
        import resource
        if sys.platform != "darwin":
            try:
                resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
            except Exception:
                pass
            try:
                cpu = max(1, int(timeout) + 1)
                resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
            except Exception:
                pass
        try:
            resource.setrlimit(resource.RLIMIT_NPROC, (nproc, nproc))
        except Exception:
            pass

    return _apply


def _run_sandboxed(program: str, sandbox_dir: str, *, argv: Sequence[str] = (), stdin: Optional[str] = None,
                    timeout: float = ASSERT_TIMEOUT, mem_bytes: int = CHILD_MEM_BYTES,
                    nproc: int = 32) -> Tuple[int, str, str]:
    """Write `program` into a fresh subdirectory of `sandbox_dir` and run it as `python -I prog.py
    [argv...]`, cwd = that subdirectory. stdin=None closes stdin; a str pipes it in. Always cleans up the
    subdirectory. Never raises: a timeout is reported as (-1, "", "TIMEOUT"), a launch failure as
    (-1, "", "LAUNCH_ERROR: ...")."""
    os.makedirs(sandbox_dir, exist_ok=True)
    work = tempfile.mkdtemp(prefix="chk_", dir=sandbox_dir)
    try:
        script = os.path.join(work, "prog.py")
        with open(script, "w", encoding="utf-8") as fh:
            fh.write(program)
        kw = {}
        if os.name == "posix":
            kw["preexec_fn"] = _child_limits(timeout, mem_bytes, nproc)
        cmd = [sys.executable, "-I", script, *argv]
        try:
            if stdin is None:
                p = subprocess.run(cmd, cwd=work, stdin=subprocess.DEVNULL, capture_output=True,
                                   text=True, timeout=timeout, **kw)
            else:
                p = subprocess.run(cmd, cwd=work, input=stdin, capture_output=True, text=True,
                                   timeout=timeout, **kw)
            return p.returncode, p.stdout, p.stderr
        except subprocess.TimeoutExpired:
            return -1, "", "TIMEOUT"
        except Exception as e:                                # pragma: no cover -- defensive
            return -1, "", f"LAUNCH_ERROR: {e!r}"
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _run_code_verdict(program: str, sandbox_dir: str, timeout: float = ASSERT_TIMEOUT) -> Tuple[bool, str]:
    """asserts/humaneval verdict: exit 0 -> pass. Reason vocabulary mirrors oc_datasets.mbpp's own
    execution() ("timeout" / "wrong_answer" (AssertionError) / "failed" (anything else)), even though this
    runs the program in an isolated subprocess instead of MBPPEvaluator's in-process exec()."""
    rc, _out, err = _run_sandboxed(program, sandbox_dir, stdin=None, timeout=timeout)
    if rc == 0:
        return True, ""
    if err == "TIMEOUT":
        return False, "timeout"
    if "AssertionError" in err:
        return False, "wrong_answer"
    return False, "failed"


# ---------------------------------------------------------------------------
# cheap (in-process) scorers: math, free, mc, ifeval
# ---------------------------------------------------------------------------
def _score_math(key: dict, text: str) -> Tuple[bool, str]:
    pred = math_postprocess_v2(text)
    try:
        ok = bool(MATHEvaluator(version="v2").is_equiv(pred, key["answer"]))
    except Exception:
        ok = False
    return ok, ("" if ok else "wrong_answer")


_UNIT_TEXT_RE = re.compile(r"\\text\{[^{}]*\}")
_TIMES_TEN_RE = re.compile(r"\\(?:times|cdot)\s*10\s*\^\s*\{?(-?\d+)\}?")
_BOOL_MAP = {"true": True, "yes": True, "y": True, "t": True,
             "false": False, "no": False, "n": False, "f": False}


def _to_plain_number(s) -> Optional[float]:
    """Best-effort numeric parse: strips \\left/\\right, \\text{...} units, commas, "%", and rewrites
    "a \\times 10^{n}" / "a \\cdot 10^{n}" to "aen" so Python's float() can read it (also handles a bare
    "6.02e23" unchanged). None if the result does not parse as a float."""
    if s is None:
        return None
    t = str(s).strip()
    if not t:
        return None
    t = t.replace("\\left", "").replace("\\right", "")
    t = _UNIT_TEXT_RE.sub("", t)
    t = _TIMES_TEN_RE.sub(r"e\1", t)
    t = t.replace(",", "").replace("$", "")
    t = t.replace("\\%", "").replace("%", "")
    t = re.sub(r"\s+", "", t)                       # numbers never carry internal whitespace
    if not t:
        return None
    try:
        return float(t)
    except ValueError:
        return None


def _numeric_close(a: float, b: float, rel: float = 0.01, near_zero_abs: float = 1e-9) -> bool:
    if abs(b) < near_zero_abs:
        return abs(a - b) < near_zero_abs
    return abs(a - b) <= rel * abs(b)


def _to_bool(s) -> Optional[bool]:
    return _BOOL_MAP.get(str(s).strip().strip(".").lower())


def _norm_str(s) -> str:
    return re.sub(r"\s+", " ", str(s).strip().lower()).strip(" .")


def _score_free(key: dict, text: str) -> Tuple[bool, str]:
    box = RP.last_boxed(text)
    pred = box if box is not None else text.strip()
    gold = key["answer"]
    if str(key.get("answer_type", "")).lower().startswith("bool"):
        pb, gb = _to_bool(pred), _to_bool(gold)
        ok = pb is not None and gb is not None and pb == gb
        return ok, ("" if ok else "wrong_answer")
    pn, gn = _to_plain_number(pred), _to_plain_number(gold)
    if pn is not None and gn is not None:
        ok = _numeric_close(pn, gn)
        return ok, ("" if ok else "wrong_answer")
    try:
        ok = bool(MATHEvaluator(version="v2").is_equiv(pred, gold))
    except Exception:
        ok = False
    if not ok:
        ok = _norm_str(pred) == _norm_str(gold)
    return ok, ("" if ok else "wrong_answer")


def _extract_mc_letter(pred: str, n_options: int) -> str:
    """Generalises analyze_s1b._gpqa_letter (A-D only) to A..LETTERS[n_options-1]."""
    allowed = LETTERS[:max(1, min(int(n_options), len(LETTERS)))]
    s = str(pred)
    box = RP.last_boxed(s)
    if box:
        cleaned = re.sub(r"\\[a-zA-Z]+", " ", box)             # drop \text, \mathrm, ...
        cleaned = re.sub(r"[^A-Za-z]", " ", cleaned)
        for tok in cleaned.split():
            if len(tok) == 1 and tok.upper() in allowed:
                return tok.upper()
    m = re.findall(r"(?i)answer\s*(?:is)?\s*[:\-]?\s*\(?([A-Ja-j])\)?[^A-Za-z]", s + " ")
    for cand in reversed(m):
        if cand.upper() in allowed:
            return cand.upper()
    m = re.findall(r"(?m)^\s*\(?([A-Ja-j])\)?[\.\)]?\s*$", s)
    for cand in reversed(m):
        if cand.upper() in allowed:
            return cand.upper()
    return ""


def _score_mc(key: dict, text: str) -> Tuple[bool, str]:
    letter = _extract_mc_letter(text, key["n_options"])
    if not letter:
        return False, "no_letter"
    ok = letter == str(key["answer"]).strip().upper()
    return ok, ("" if ok else "wrong_answer")


def _score_ifeval(key: dict, text: str) -> Tuple[bool, str]:
    # No broad try/except: a malformed gold or an unimplemented instruction id is meant to raise
    # (if_scorer.parse_gold's contract) -- a pipeline bug must not be silently scored as "wrong answer".
    ok = bool(if_scorer.score_ifeval(text, key["gold"]))
    return ok, ("" if ok else "wrong_answer")


def _score_cheap(key: dict, text: str) -> Tuple[bool, str]:
    kt = key["type"]
    if kt == "math":
        return _score_math(key, text)
    if kt == "mc":
        return _score_mc(key, text)
    if kt == "free":
        return _score_free(key, text)
    if kt == "ifeval":
        return _score_ifeval(key, text)
    raise ValueError(f"unsupported key type for _score_cheap: {kt!r}")


def _is_english(text: str) -> bool:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return True                                            # nothing alphabetic to judge as non-English
    if len(letters) >= 20:
        try:
            langs = langdetect.detect_langs(text)
        except Exception:
            return False
        if not langs:
            return False
        top = langs[0]                                          # detect_langs sorts by probability, descending
        return bool(top.lang == "en" and top.prob >= 0.9)
    ascii_frac = sum(1 for c in letters if ord(c) < 128) / len(letters)
    return ascii_frac >= 0.85


def _score_chat(text: str, loop: bool) -> Tuple[bool, str]:
    if loop:
        return False, "loop"
    if not _is_english(text):
        return False, "not_english"
    return True, ""


# ---------------------------------------------------------------------------
# code scorers: asserts, humaneval, stdio (module-level -> picklable for check_many's pool)
# ---------------------------------------------------------------------------
def _score_asserts(key: dict, text: str, sandbox_dir: str) -> Tuple[bool, str]:
    code = MBPPEvaluator()._process_answer(text)
    setup = key.get("setup") or ""
    program = (setup + "\n" if setup else "") + code + "\n" + key["tests"]
    return _run_code_verdict(program, sandbox_dir)


def _score_humaneval(key: dict, text: str, sandbox_dir: str) -> Tuple[bool, str]:
    code = MBPPEvaluator()._process_answer(text)
    program = code + "\n\n" + key["test"] + "\n\ncheck(" + key["entry_point"] + ")\n"
    return _run_code_verdict(program, sandbox_dir)


def _looks_like_code(text: str) -> bool:
    return bool(re.search(r"\b(def |import |print\(|input\(|sys\.stdin)", text[:4000]))


def _extract_stdio_code(text: str) -> Optional[str]:
    blocks = re.findall(r"```python\s*\n?(.*?)```", text, re.DOTALL)
    if not blocks:
        blocks = re.findall(r"```\s*\n?(.*?)```", text, re.DOTALL)
    if blocks:
        return blocks[-1]
    return text if _looks_like_code(text) else None


def _tokens_match(a: str, b: str) -> bool:
    ta, tb = a.split(), b.split()
    if len(ta) != len(tb):
        return False
    for x, y in zip(ta, tb):
        if x == y:
            continue
        try:
            fx, fy = float(x), float(y)
        except ValueError:
            return False
        if abs(fx - fy) > max(1e-6, 1e-6 * max(abs(fx), abs(fy))):
            return False
    return True


def _compile_cpp_checker(source: str, sandbox_dir: str) -> Tuple[Optional[str], str]:
    """-> (binary path, "") or (None, reason). Cached by sha256(source) under sandbox_dir/_checker_cache/,
    installed with an atomic os.replace so concurrent check_many workers racing on the same checker never
    observe a partially-written binary."""
    if shutil.which("g++") is None:
        return None, "no_g++"
    h = common.sha256_hex(source)[:16]
    cache_dir = os.path.join(sandbox_dir, "_checker_cache")
    os.makedirs(cache_dir, exist_ok=True)
    binpath = os.path.join(cache_dir, h)
    if os.path.exists(binpath) and os.access(binpath, os.X_OK):
        return binpath, ""
    tmp = tempfile.mkdtemp(prefix="ccbuild_", dir=cache_dir)
    try:
        src = os.path.join(tmp, "checker.cpp")
        with open(src, "w", encoding="utf-8") as fh:
            fh.write(source)
        outp = os.path.join(tmp, "checker_bin")
        try:
            p = subprocess.run(["g++", "-O2", "-o", outp, src], capture_output=True, text=True, timeout=60)
        except subprocess.TimeoutExpired:
            return None, "compile_timeout"
        if p.returncode != 0 or not os.path.exists(outp):
            return None, "compile_failed"
        os.chmod(outp, 0o755)
        os.replace(outp, binpath)
        return binpath, ""
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _write_case_files(work: str, case_input: str, actual_output: str, expected_output: str) -> List[str]:
    paths = []
    for name, content in (("in.txt", case_input), ("out.txt", actual_output), ("ans.txt", expected_output)):
        p = os.path.join(work, name)
        with open(p, "w", encoding="utf-8") as fh:
            fh.write(content)
        paths.append(p)
    return paths


def _run_checker_argv(argv: Sequence[str], in_p: str, out_p: str, ans_p: str, cwd: str, timeout: float) -> bool:
    kw = {}
    if os.name == "posix":
        kw["preexec_fn"] = _child_limits(timeout, CHILD_MEM_BYTES, 32)
    try:
        p = subprocess.run([*argv, in_p, out_p, ans_p], cwd=cwd, stdin=subprocess.DEVNULL,
                           capture_output=True, text=True, timeout=timeout, **kw)
    except Exception:
        return False
    return p.returncode == 0 and p.stdout.strip().startswith("1")


def _run_python_checker(source: str, case_input: str, actual_output: str, expected_output: str,
                        sandbox_dir: str, timeout: float) -> bool:
    os.makedirs(sandbox_dir, exist_ok=True)
    work = tempfile.mkdtemp(prefix="pychk_", dir=sandbox_dir)
    try:
        in_p, out_p, ans_p = _write_case_files(work, case_input, actual_output, expected_output)
        checker_path = os.path.join(work, "checker.py")
        with open(checker_path, "w", encoding="utf-8") as fh:
            fh.write(source)
        return _run_checker_argv([sys.executable, "-I", checker_path], in_p, out_p, ans_p, work, timeout)
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _run_cpp_checker(binpath: str, case_input: str, actual_output: str, expected_output: str,
                     sandbox_dir: str, timeout: float) -> bool:
    os.makedirs(sandbox_dir, exist_ok=True)
    work = tempfile.mkdtemp(prefix="cppchk_", dir=sandbox_dir)
    try:
        in_p, out_p, ans_p = _write_case_files(work, case_input, actual_output, expected_output)
        return _run_checker_argv([binpath], in_p, out_p, ans_p, work, timeout)
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _score_stdio(key: dict, text: str, sandbox_dir: str, per_case_timeout: float = STDIO_PER_CASE_TIMEOUT,
                 total_timeout: float = STDIO_TOTAL_TIMEOUT) -> Tuple[bool, str]:
    code = _extract_stdio_code(text)
    if code is None:
        return False, "no_code"
    cases = key.get("cases") or []
    if not cases:
        return False, "no_cases"

    checker = key.get("checker")
    lang = None
    cpp_bin = None
    if checker is not None:
        lang = checker.get("lang") if isinstance(checker, dict) else None
        if lang == "cpp":
            cpp_bin, reason = _compile_cpp_checker(checker["source"], sandbox_dir)
            if cpp_bin is None:
                return False, "no_checker_runtime"
        elif lang != "python":
            return False, "no_checker_runtime"

    t0 = time.time()
    for case in cases:
        remaining = total_timeout - (time.time() - t0)
        if remaining <= 0:
            return False, "timeout"
        case_timeout = min(per_case_timeout, remaining)
        rc, out, err = _run_sandboxed(code, sandbox_dir, stdin=case["input"], timeout=case_timeout)
        if rc != 0:
            return False, ("timeout" if err == "TIMEOUT" else "failed")
        if checker is None:
            ok = _tokens_match(out, case["output"])
        elif lang == "python":
            ok = _run_python_checker(checker["source"], case["input"], out, case["output"], sandbox_dir, case_timeout)
        else:
            ok = _run_cpp_checker(cpp_bin, case["input"], out, case["output"], sandbox_dir, case_timeout)
        if not ok:
            return False, "wrong_answer"
    return True, ""


def _score_code(key: dict, text: str, sandbox_dir: str) -> Tuple[bool, str]:
    kt = key["type"]
    if kt == "asserts":
        return _score_asserts(key, text, sandbox_dir)
    if kt == "humaneval":
        return _score_humaneval(key, text, sandbox_dir)
    if kt == "stdio":
        return _score_stdio(key, text, sandbox_dir)
    raise ValueError(f"unsupported key type for _score_code: {kt!r}")


# ---------------------------------------------------------------------------
# the EOS/loop/decode gate, shared by check() and check_many()
# ---------------------------------------------------------------------------
def _gate(gen_ids: Sequence[int], tok_decode: Callable[[Sequence[int]], str], gen_cap: int) -> dict:
    capped = list(gen_ids[:gen_cap])
    eos_idx = common.first_eos(capped)
    if eos_idx < 0:
        return {"ok": False, "loop": common.has_loop(capped), "eos": False, "n_gen": len(capped), "text": None}
    pre = capped[:eos_idx]
    return {"ok": True, "loop": common.has_loop(pre), "eos": True, "n_gen": eos_idx + 1,
            "text": tok_decode(pre)}


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------
def check(key: dict, gen_ids: Sequence[int], tok_decode: Callable[[Sequence[int]], str], *,
         sandbox_dir: str, gen_cap: int) -> dict:
    g = _gate(gen_ids, tok_decode, gen_cap)
    if not g["ok"]:
        return {"ok": False, "reason": "no_eos", "loop": g["loop"], "eos": False, "n_gen": g["n_gen"]}
    kt = key["type"]
    if kt in CODE_KEY_TYPES:
        ok, reason = _score_code(key, g["text"], sandbox_dir)
    elif kt == "chat":
        ok, reason = _score_chat(g["text"], g["loop"])
    else:
        ok, reason = _score_cheap(key, g["text"])
    return {"ok": ok, "reason": reason, "loop": g["loop"], "eos": True, "n_gen": g["n_gen"]}


def check_many(items: Sequence[dict], workers: int = 8) -> List[dict]:
    """items: [{"key": dict, "gen_ids": [...], "tok_decode": callable, "sandbox_dir": str, "gen_cap": int}, ...].
    Returns verdict dicts in the SAME order as items (see module docstring for the parallelisation split)."""
    gates = [_gate(it["gen_ids"], it["tok_decode"], it["gen_cap"]) for it in items]
    out: List[Optional[dict]] = [None] * len(items)
    pending = []                                                # (index, key, text, sandbox_dir)
    for i, (it, g) in enumerate(zip(items, gates)):
        if not g["ok"]:
            out[i] = {"ok": False, "reason": "no_eos", "loop": g["loop"], "eos": False, "n_gen": g["n_gen"]}
            continue
        kt = it["key"]["type"]
        if kt in CODE_KEY_TYPES:
            pending.append((i, it["key"], g["text"], it["sandbox_dir"]))
        elif kt == "chat":
            ok, reason = _score_chat(g["text"], g["loop"])
            out[i] = {"ok": ok, "reason": reason, "loop": g["loop"], "eos": True, "n_gen": g["n_gen"]}
        else:
            ok, reason = _score_cheap(it["key"], g["text"])
            out[i] = {"ok": ok, "reason": reason, "loop": g["loop"], "eos": True, "n_gen": g["n_gen"]}
    if pending:
        with ProcessPoolExecutor(max_workers=max(1, workers)) as pool:
            futs = {pool.submit(_score_code, key, text, sbx): i for (i, key, text, sbx) in pending}
            for fut in futs:
                i = futs[fut]
                ok, reason = fut.result()
                g = gates[i]
                out[i] = {"ok": ok, "reason": reason, "loop": g["loop"], "eos": True, "n_gen": g["n_gen"]}
    assert all(o is not None for o in out), "check_many left an item unscored"
    return out
