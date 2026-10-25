# Vendored instruction-following checker (IFEvalG)

Source: https://github.com/allenai/open-instruct @ `172e379eada34137885bd1c36543c4f59cb113e4` (HEAD of `main`, 2026-09-11)
Path:   `open_instruct/IFEvalG/{instructions,instructions_registry,instructions_util}.py`
Last upstream commit touching that directory: `ebca9f7d4921042d7d8a65369e513d6566f0f3ba` (2026-04-08, "Fix `RepeatPhraseChecker` constraints (#1044)")
Fetched: 2026-09-14
Licence: Apache-2.0 (`LICENSE`, copied from the repository root; the three files carry the Google Research Apache-2.0 header)

`open_instruct/IFEvalG/*.py` are **unmodified copies** (byte-identical; sha256 below and in `VENDORED.sha256`).
`open_instruct/__init__.py` and `open_instruct/IFEvalG/__init__.py` are shims, not upstream: they let the copies
import themselves as `open_instruct.IFEvalG` (their own import lines) without installing open-instruct.
`code/draftalign/if_scorer.py` puts `code/vendor/ifeval` on `sys.path`.

| file | sha256 | upstream git blob |
|---|---|---|
| `open_instruct/IFEvalG/instructions.py` | `0bad560c99419fdafa52d7c400f72ce5c4cb8290f3471c3f1b31c035b68e9947` | `33138f0c3508901c375c569025a14a68106a4c78` |
| `open_instruct/IFEvalG/instructions_registry.py` | `5436911d616a7c38afcbbc18312551d8a9ed3f06bca44d74920071c1690c7fc3` | `71db32f97391aeaf0b2a7c515f2ef4a3e8f4a428` |
| `open_instruct/IFEvalG/instructions_util.py` | `c78ec28b5e9ae97d6cc7076a265d0c80a6312146214e6ab5aa79855e24ab0334` | `b37f5321eddc9fd03137e17b94519343a0408d73` |
| `LICENSE` | `b998a796c55d006a974a48c73aa32674e6901e8d49aa98d74722e0c1063b40b6` | — |

## Why this checker

`allenai/IF_multi_constraints_upto5` ("constraints sampled from IFEval (25) and IFBench-Train (29)") is scored in
open-instruct by `ground_truth_utils.IFEvalVerifier`, which uses exactly this registry; the IFBench README says
"The new training constraints and verification functions are here:
https://github.com/allenai/open-instruct/tree/main/open_instruct/IFEvalG". The registry holds 54 ids. The 25
classic classes match Google's `instruction_following_eval/instructions.py` (google-research@`08a8d67`) in
logic: a whitespace-insensitive per-class diff shows only reformatting, `pyrefly` comments and
`for`-loops folded into `any()`/`all()`. `allenai/IFBench`'s pip package re-ships the classic verifiers but not
the 29 training ones.

`allenai/RLVR-IFeval` was built for open-instruct's separate `if_functions.py` (not vendored). Its 24 function
names are mapped to IFeval ids with the same semantics by `build_if_pool.RLVR_MAPPING`; five have no faithful
equivalent here and are dropped: `verify_keyword_frequency` and `verify_letter_frequency` (exact counts),
`verify_paragraph_count` (`* * *` divider), `validate_choice` (free option lists), and `validate_repeat_prompt`
(its gold includes the sentence the prompt says not to repeat).

## Coverage (checked 2026-09-14 against the pinned dataset revisions)

* `IF_multi_constraints_upto5` @ `2e3a774`: 54 distinct ids; 54 implemented; 0 rows with an unimplemented id.
* `RLVR-IFeval` @ `47c03c7`: 24 function names; 19 mapped to implemented ids; 5 without an equivalent (above).
* Implemented-but-inadmissible ids (24 of 54) are listed with reasons in
  `code/draftalign/build_if_pool.py::EXCLUDED_IDS`. Behaviours that justify the precision-driven exclusions are
  pinned by `code/tests/test_if_scorer.py::test_checker_quirks_behind_the_id_exclusions`.

## Runtime dependencies

Installed 2026-09-14 into **both** `/venv/main` and `/venv/r3` (all pure Python, versions pinned to
open-instruct's `uv.lock` except absl-py, which matches the version already in `/venv/main`):

| package | version | used by |
|---|---|---|
| `nltk` | 3.9.2 | sentence / word tokenisation (`count_sentences`, `word_tokenize`) |
| `langdetect` | 1.0.9 | language + English-case checkers |
| `immutabledict` | 1.2.0 | `LANGUAGE_CODES` |
| `absl-py` | 2.5.0 | `absl.logging` (installed into `/venv/r3`; already present in `/venv/main`) |
| `joblib` 1.6.0, `cloudpickle` 3.1.2 | — | pulled in by nltk |

NLTK data: `punkt` and `punkt_tab` downloaded to `/root/nltk_data/tokenizers/` (NLTK's default search path for
root). nltk 3.9.2 serves `nltk.data.load("nltk:tokenizers/punkt/english.pickle")` from `punkt_tab`; that
resource alone is sufficient (verified with a data path holding only `punkt_tab`). A decode running as another
user needs `NLTK_DATA=/root/nltk_data` or its own copy.

## Determinism

The vendored code is not deterministic by itself: `build_description` draws from the global `random` module for
any kwarg left out, and `langdetect` is randomised. `if_scorer` seeds `langdetect.DetectorFactory.seed = 0` at
import, builds every checker under `random.seed(0)` (restoring the caller's state), and the pool builder drops
rows whose checker args differ between two build seeds.

Checksums in `VENDORED.sha256` (paths relative to `code/`).
