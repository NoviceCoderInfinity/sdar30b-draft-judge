# IF pool wiring patch — 2026-09-14

The exact, additive edits that wire the verifiable instruction-following pool into the round-3 pipeline:
`code/configs/if_pool.json` (built by `code/draftalign/build_if_pool.py`), scored by
`code/draftalign/if_scorer.py` (vendored IFEvalG checker in `code/vendor/ifeval/`).

Nothing below is applied yet. The four target files are shared: the edits are written against the working tree
as of 2026-09-14 14:42, which already contains the `opencode_pool` wiring, and every anchor is quoted verbatim.
No existing line is deleted or reordered.

## 0. Preconditions (already done, CPU only)

* Checker deps in **both** `/venv/main` and `/venv/r3`: `nltk==3.9.2 langdetect==1.0.9 immutabledict==1.2.0`
  (+ `absl-py==2.5.0`, `joblib`, `cloudpickle`). NLTK `punkt_tab` (and `punkt`) in `/root/nltk_data`, which is
  NLTK's default path for root. A decode running as another user needs `NLTK_DATA=/root/nltk_data`.
* `code/configs/if_pool.json` and `if_pool.json.sha256` exist.
* `/venv/r3/bin/python -m pytest code/tests/test_if_scorer.py -q` passes.

## 1. `code/research_protocol.py`

**1a. `GEN_LEN`** — add the key, otherwise startup raises. Replace

```python
"mbpp_pool": 512, "opencode_pool": 512}
```

with

```python
"mbpp_pool": 512, "opencode_pool": 512, "if_pool": 1024}
```

**1b. Loader** — insert immediately above `POOLS = {` (after `opencode_pool_items`):

```python
IF_POOL_PATH = os.path.join(HERE, "configs", "if_pool.json")


def if_pool_items():
    """The frozen verifiable instruction-following pool: code/configs/if_pool.json (built by
    draftalign/build_if_pool.py; sha256-checked against its .sha256). prompt = the dataset's raw single user message
    (encode_prompt applies the chat template, nothing else is added); gold = JSON {"instruction_id_list", "kwargs"},
    scored by draftalign/if_scorer.py (strict, every constraint must pass). idx = rank in the frozen order.
    split is "if_pool", never train/train_pool, so build_train_set's frozen-GSM8K guard cannot fire."""
    raw = open(IF_POOL_PATH, "rb").read()
    want = open(IF_POOL_PATH + ".sha256").read().split()[0]
    assert _sha(raw) == want, f"{IF_POOL_PATH}: sha256 {_sha(raw)} != {want}"
    out = []
    for it in json.loads(raw)["items"]:
        assert _sha(it["prompt"].encode("utf-8")) == it["text_sha256"], f"if_pool text sha mismatch at idx {it['idx']}"
        out.append({"idx": int(it["idx"]), "rank": int(it["idx"]), "split": "if_pool", "key": it["key"],
                    "prompt": it["prompt"], "gold": it["gold"]})
    assert [o["idx"] for o in out] == list(range(len(out))), "if_pool idx must be 0..n-1 in file order"
    return out
```

(`os`, `json`, `HERE` and `_sha` are already defined in the module.)

**1c. `POOLS`** — replace

```python
         "opencode_pool": opencode_pool_items}
```

with

```python
         "opencode_pool": opencode_pool_items, "if_pool": if_pool_items}
```

`items_for_split("if_pool:0:3000")` then works through the existing slice syntax. `test_items` needs no change
(the pool is not a test set).

## 2. The two dataset-family maps (AGENTS.md §9(b): the fall-through is `gsm8k`, silently)

**2a. `code/draftalign/run_da_decode.py`** — insert one line after the `opencode` line:

```python
          "mbpp" if base.startswith("opencode") else          # opencode_pool: gold = unit-test asserts, MBPP sandbox
          "ifeval" if base.startswith("if") else              # if_pool: IFEval-style constraints, draftalign/if_scorer
```

**2b. `code/run_research_baselines.py`** — the same line, with that file's variable name:

```python
          "mbpp" if _b.startswith("opencode") else            # opencode_pool: gold = unit-test asserts, MBPP sandbox
          "ifeval" if _b.startswith("if") else                # if_pool: IFEval-style constraints, draftalign/if_scorer
```

No other split name starts with `if`, so the position in the chain does not matter.

Pre-existing quirk, not part of this patch: `run_research_baselines.py` reads `GEN_LEN[args.split]`, not
`GEN_LEN[_b]`, so a sliced name such as `if_pool:0:3000` raises `KeyError` there. `run_da_decode.py` uses
`GEN_LEN[base]` and is fine. The datagen path below uses only `run_da_decode.py`.

## 3. `code/analyze_s1b.py::per_item_correct`

Insert as the first branch, directly after the docstring `"""-> dict idx -> 0/1"""`:

```python
    if ds == "ifeval":
        from draftalign.if_scorer import score_ifeval_record     # strict IFEvalG, every constraint must pass
        return {x["idx"]: score_ifeval_record(x) for x in rows}
```

This must come before the final MATH fall-through, which would otherwise score IF responses with
`math_postprocess_v2`.

Why `score_ifeval_record` and not bare `score_ifeval(pred, gold)`: `make_record` stores
`pred = tok.decode(<all non-mask generated ids>, skip_special_tokens=True)`. The wrapper returns 0 in two cases,
then defers to `score_ifeval(rec["pred"], rec["gold"])`:

* the response never reached EOS inside the 1024-token budget, where "at least N words/sentences" constraints
  can pass on an unfinished answer;
* any non-EOS token follows the first EOS, where that text would be scored as part of the answer.

It uses `gen_ids` and `n_gen_to_eos`, which both runners' records carry. A row without those fields
(pred + gold only) is scored directly.

`code/` is already on `sys.path` wherever `per_item_correct` is called (both runners insert it; running
`analyze_s1b.py` as a script puts `code/` first), so `draftalign.if_scorer` imports. The scorer puts
`code/vendor/ifeval` on the path itself.

## 4. What the family string does downstream (checked, no change needed)

* `run_da_decode.cell_name(ds, …)` makes the family the cell prefix. The blind datagen cell is
  `ifeval_B32_t95_dv-s8_blind_q50` and records are
  `ifeval_B32_t95_dv-s8_blind_q50.if_pool:0:N.shard{k}of{n}.jsonl`.
  `cfg["dataset"] = "ifeval"` enters `config_hash`, so these can never mix with another family's records.
* The default `--ref-cell` becomes `ifeval_B32_t95_stock`. No such stock cell exists, so `load_ref_records`
  returns `{}` and no `ref_match` is written. That is expected for datagen and matches the other pool decodes.
* The W&B tags get `ifeval`.
* `build_train_set.py`:
  * `split == "if_pool"` is neither `train` nor `train_pool`, so the 100-frozen-GSM8K-index guard cannot drop
    items. `TEST_SPLITS = ("test",)` does not match.
  * `tokenizer_prompt_ids("if_pool:0:N")` resolves through `items_for_split` → `POOLS`, i.e. through 1b and 1c.
* `eval_table.py` / `compare_table.py`: not needed (training pool, not a reported eval set).

## 5. Commands

Stage 1: decode with the training-free blind rule (DATA.md §4). Replace `N` with the slice size.
`idx` is sha256(key) order, so any prefix `if_pool:0:N` is a uniform sample of the pool; decode a prefix,
measure the strict-pass yield, then extend or trim.

```bash
CUDA_VISIBLE_DEVICES={GPU} /venv/r3/bin/python code/draftalign/run_da_decode.py \
  --model b32 --drafter d32 --items if_pool:0:N --draft s8 --judge blind --tau-acc-q .50 \
  --shard k --nshards 8 --gpu {GPU} --score \
  --out logs/draftalign/datagen/if_pool_0_N_dv-s8_blind_q50_r3
```

Stage 2: keep only strictly correct items and chop them into 32-token blocks:

```bash
D=logs/draftalign/datagen/if_pool_0_N_dv-s8_blind_q50_r3
/venv/r3/bin/python code/draftalign/build_train_set.py --records $D --traces $D/traces --layout da \
  --split-name if_pool:0:N --out logs/draftalign/trainset_r3/if_pool_0_N \
  --only-correct --max-seq 1200 --exclude-frozen
```

## 6. Post-apply checks (CPU, about 1 minute)

```bash
cd /root/workspace/diffusion-moe-general-expert
/venv/r3/bin/python - <<'EOF'
import sys; sys.path.insert(0, "code")
import research_protocol as P
items = P.items_for_split("if_pool:0:5")
assert P.GEN_LEN["if_pool"] == 1024 and items[0]["split"] == "if_pool" and len(items) == 5
base = P.split_base("if_pool:0:5")
ds = ("mbpp" if base.startswith("mbpp") else "mbpp" if base.startswith("opencode") else
      "ifeval" if base.startswith("if") else "math500" if base.startswith("math") else "gsm8k")
assert ds == "ifeval"
from analyze_s1b import per_item_correct
it = items[0]
rec = {"idx": it["idx"], "pred": "", "gold": it["gold"], "gen_ids": [151645], "n_gen_to_eos": 1}
assert per_item_correct("ifeval", [rec]) == {it["idx"]: 0}          # empty answer -> 0, and no exception
print("ok", len(P.items_for_split("if_pool")), "items")
EOF
grep -n '"ifeval" if' code/draftalign/run_da_decode.py code/run_research_baselines.py   # both maps edited
/venv/r3/bin/python -m pytest code/tests/test_if_scorer.py -q
```
