# RESULTS

Every number below was recomputed from the raw per-item record files on
**2026-09-09 05:41 UTC (11:11 IST)**, using `eval_table.py`'s formulas verbatim.
Nothing here is copied from an earlier summary.

Cells marked **partial** were still being written by running jobs at snapshot
time; their `n` is stated and will only grow.

---

## 0. Metric definitions

Used identically in every table. All are **weighted sums over items**, not means
of per-item means.

```
passes/blk = sum(n_forward - n_prefill - n_flush) / sum(n_blocks)
exp/token  = sum(unique_expert_layer_loads) / sum(n_gen_to_eos)
ms/token   = 1000 * sum(seconds) / sum(n_gen_to_eos)
+ drafter  = exp/token + sum(drafter_forwards) * 364.64 / sum(n_gen_to_eos)
acc        = 100 * sum(correct) / n     (vendored OpenCompass evaluator, per item)
Δacc       = paired bootstrap over the items two cells share,
             2,000 resamples, seed 0, reported as point [2.5%, 97.5%]
```

* `364.64 = 1,720,573,440 / 4,718,592` = SDAR-1.7B parameters excluding its token
  embedding, divided by one 30B expert (3 × 2048 × 768). Defined at
  `code/draftalign/compare_table.py:23`. `drafter_forwards` =
  `sum(drafter_n_by_kind.values())` = prefill + draft + redraft + flush.
* `unique_expert_layer_loads` = distinct (layer, expert) pairs actually
  dispatched, read from `sdar_harness.moe_instr.SINK`. Under a pool it counts
  the restricted set, i.e. what really ran.
* `passes/blk` counts forwards over a 32-token block. Where arms differ in what
  a pass costs, compare `exp/token` and `ms/token` instead.
* `eval_table.py` also emits `fwd/blk (im)`, a mean of per-item means. It is a
  *different* number (calibvalid stock: 9.31 block-weighted vs 9.02 item-mean).
  Nothing in this file uses it.

**Producers.** All decode records: `code/draftalign/run_da_decode.py`, except
the block-32 threshold-sweep cells, which are
`code/run_research_baselines.py`. Tables: `code/draftalign/eval_table.py`
(canonical, → `results/reports/EVAL_TABLE.md`) and
`code/draftalign/compare_table.py` (adds `exp/layer-fwd` and `+ drafter`).
Per-item scoring: `code/analyze_s1b.py::per_item_correct` via
`code/vendor/oc_datasets/`.

**Common configuration** unless stated: model `b32`
(`JetLM/SDAR-30B-A3B-Chat-b32` @ `c351bbc37d240aa6871f167e8f92d694281b0c22`),
block 32, stock threshold 0.95, drafter `d32` at setting `s8`, judge adapter
`logs/draftalign/train/t2b_union_flag/step003057` (weights hash
`734af65681ab6bb8`, cell tag `_ad734af656`) at `--tau-commit 0.95`, blind rule at
`--tau-acc-q 0.50`, pools as `--pool M --pool-rule des --pool-flush`.

---

## 1. Stock block-32 baselines

The reference every other table is paired against.

| dataset | n | passes/blk | exp/token | ms/token | acc % |
|---|---:|---:|---:|---:|---:|
| GSM8K-500 | 500 | 10.06 | 1164.2 | 197.6 | 90.2 |
| MATH-500 | 500 | 8.52 | 853.5 | 150.5 | 72.4 |
| MBPP (sanitized test) | 257 | 11.93 | 1714.0 | 295.6 | 65.4 |
| HumanEval | 164 | 8.26 | 982.3 | 172.5 | 78.0 |
| GPQA-Diamond | 198 | 16.07 | 1381.1 | 245.8 | 37.4 |
| AIME | 90 | 10.77 | 997.7 | 178.3 | 6.7 |
| calibvalid (100 frozen GSM8K *train* prompts) | 100 | 9.31 | 1090.6 | 187.1 | 95.0 |

Records: `logs/draftalign/da_decode/<ds>_B32_t95_stock/` for every row except
GSM8K-500, whose stock reference lives at
`logs/research_v1/baselines_b32/gsm8k_B32_t95_stock.gsm8k500.shard{0,1,2}of3.jsonl`
(`run_research_baselines.py`). This is why `EVAL_TABLE.md`'s gsm8k500 section has
no stock row and its `d_acc` column there is *vs the free rule*, not vs stock.

> **calibvalid ms/token, 184.3 vs 187.1.** Two independent stock decodes of the
> same 100 items exist — one under `logs/research_v1/baselines_b32/` (184.3), one
> under `logs/draftalign/da_decode/` (187.1). They are **bit-identical** in
> `gen_ids` and `passes_per_block` for all 100 items; only wall-clock differs.
> Every comparison in this file uses 187.1, the `da_decode` twin, so both sides
> of each pair come from the same run family.

## 2. Blind accept — training-free, zero extra MoE forwards

`--judge blind --tau-acc-q .50 --draft s8`. Cell
`<ds>_B32_t95_dv-s8_blind_q50`.

| dataset | n | passes/blk (Δ%) | exp/token | **+ drafter** | ms/token | acc % | Δacc vs stock [95% CI] |
|---|---:|---:|---:|---:|---:|---:|---|
| GSM8K-500 | 500 | 8.49 (−15.6%) | 1016.8 | 1138.3 | 184.5 | 90.4 | +0.2 [−1.2, +1.8] |
| MATH-500 | 500 | 7.40 (−13.1%) | 764.0 | 872.8 | 141.8 | 70.8 | −1.6 [−4.8, +1.4] |
| MBPP | 257 | 10.20 (−14.5%) | 1523.9 | 1692.2 | 279.1 | 60.3 | **−5.1 [−8.6, −1.6]** |
| HumanEval | 164 | 6.33 (−23.4%) | 782.5 | 915.6 | 146.6 | 75.6 | −2.4 [−6.1, +1.2] |
| GPQA-Diamond | 198 | 15.48 (−3.7%) | 1348.4 | 1456.9 | 243.3 | 35.4 | −2.0 [−8.1, +3.5] |
| AIME | 90 | 9.75 (−9.5%) | 927.3 | 1033.5 | 170.6 | 5.6 | −1.1 [−4.4, +2.2] |
| calibvalid | 100 | 7.93 (−14.8%) | 962.7 | **1085.9** | 171.6 | 95.0 | +0.0 [+0.0, +0.0] |

### 2.1 The blind-accept catch

On calibvalid the rule looks like a **1.13x** saving in expert work
(1090.6 → 962.7). Charge the drafter's own forwards at 364.64
expert-equivalents each and the total is **1085.9**:

```
1090.6 / 962.7  = 1.133x   (what exp/token alone says)
1090.6 / 1085.9 = 1.004x   (the honest total)
```

**0.4%.** On three of the seven splits the honest total is *worse* than stock —
MATH-500 872.8 vs 853.5, GPQA 1456.9 vs 1381.1, AIME 1033.5 vs 997.7 — and the
best of the other four is GSM8K-500's 1138.3 vs 1164.2, a 2% win. The rule costs
nothing and saves nothing. Its one real property is that it does not hurt accuracy on GSM8K —
which is what made it usable as a **training-data generator** (see `DATA.md`).

## 3. The trained judge

`--adapter .../t2b_union_flag/step003057 --judge full --draft s8 --tau-commit .95`.
Cell `<ds>_B32_t95_dv-s8_full_c95_ad734af656`.

| dataset | n | passes/blk (×) | exp/token | + drafter | ms/token | acc % | Δacc vs stock [95% CI] |
|---|---:|---:|---:|---:|---:|---:|---|
| GSM8K-500 | 500 | 6.64 (1.52x) | 849.3 | 971.5 | 158.8 | 88.8 | −1.4 [−3.6, +0.8] |
| MATH-500 | 500 | 6.29 (1.35x) | 680.1 | 788.9 | 126.9 | 68.0 | **−4.4 [−7.6, −1.2]** |
| MBPP | 257 | 10.43 (1.14x) | 1588.2 | 1754.6 | 300.6 | 51.4 | **−14.0 [−20.2, −7.8]** |
| HumanEval | 164 | 5.03 (**1.64x**) | 610.4 | 733.9 | 117.1 | 64.6 | **−13.4 [−20.7, −6.1]** |
| GPQA-Diamond | 198 | 14.20 (1.13x) | 1281.6 | 1390.2 | 238.1 | 32.8 | −4.5 [−12.1, +2.5] |
| AIME | 90 | 8.07 (1.33x) | 800.1 | 906.3 | 149.8 | 5.6 | −1.1 [−3.3, +0.0] |
| calibvalid | 100 | 6.00 (1.55x) | 780.8 | 903.8 | 148.2 | 94.0 | −1.0 [−5.0, +2.0] |

On MBPP the judge is **slower in wall-clock than stock** (300.6 vs 295.6 ms/token)
*and* more expensive once the drafter is charged (1754.6 vs 1714.0) *and* 14
points less accurate. It is a strict loss on that dataset.

## 4. DES pooling

`--pool M --pool-rule des --pool-flush`, applied to every block forward
(judge / denoise), never the prefill, and to the flush because of `--pool-flush`.

### 4.1 Under the stock decode — no drafter, no adapter, no training

| dataset | n | passes/blk | exp/token | ms/token | acc % | Δacc vs stock [95% CI] |
|---|---:|---:|---:|---:|---:|---|
| calibvalid, DES M=32 | 100 | 9.94 | **654.9** | 127.3 | 93.0 | −2.0 [−7.0, +3.0] |
| calibvalid, DES M=48 | 100 | 9.27 | 871.8 | 160.6 | 94.0 | −1.0 [−4.0, +2.0] |
| GSM8K-500, DES M=32 | 500 | 10.46 | 676.2 | 132.0 | 89.0 | −1.2 [−3.6, +1.2] |
| GSM8K-500, DES M=48 | 500 | 9.96 | 920.5 | 171.2 | 91.2 | +1.0 [−0.8, +3.0] |
| MBPP, DES M=32 | 257 | 12.30 | 1129.6 | 217.3 | 65.0 | −0.4 [−5.8, +5.1] |
| MBPP, DES M=48 | 257 | 12.09 | 1482.3 | 273.3 | 63.8 | −1.6 [−5.8, +3.1] |
| MATH-500, DES M=48 **(partial)** | 158 | 8.58 | 703.9 | 132.7 | 72.8 | +0.6 [−5.1, +6.3] |

There is no `+ drafter` column because there is no drafter. **This is the single
strongest result in the repo**: on calibvalid, 654.9 expert-executions/token at
93.0 accuracy — cheaper than *any* drafter-assisted arm's honest total — from a
change to the router's dispatch alone.

### 4.2 Under the trained judge

| dataset | n | passes/blk | exp/token | + drafter | ms/token | acc % | Δacc vs stock [95% CI] |
|---|---:|---:|---:|---:|---:|---:|---|
| calibvalid, judge + DES32 | 100 | 6.67 | 463.9 | 587.8 | 100.9 | 88.0 | **−7.0 [−13.0, −1.0]** |
| calibvalid, judge + DES48 | 100 | 6.09 | 612.7 | 735.5 | 124.0 | 94.0 | −1.0 [−4.0, +2.0] |
| GSM8K-500, judge + DES32 | 500 | 7.24 | 491.9 | 614.3 | 105.6 | 88.0 | −2.2 [−4.8, +0.2] |
| GSM8K-500, judge + DES48 | 500 | 6.76 | 667.7 | 790.0 | 133.7 | 87.8 | −2.4 [−5.0, +0.2] |
| MATH-500, judge + DES32 **(partial)** | 227 | 6.90 | 406.5 | 515.0 | 89.8 | 65.2 | −6.2 [−10.6, −1.3] |
| MATH-500, judge + DES48 | 500 | 6.39 | 549.5 | 658.1 | 111.1 | 69.2 | −3.2 [−6.6, +0.4] |
| MBPP, judge + DES32 | 257 | 10.74 | 1006.5 | 1173.7 | 210.1 | 55.6 | −9.7 [−16.0, −3.5] |
| MBPP, judge + DES48 | 257 | 10.77 | 1366.9 | 1534.0 | 272.2 | 53.7 | −11.7 [−18.3, −5.4] |
| HumanEval, judge + DES48 | 164 | 5.15 | 533.6 | 657.5 | 107.6 | 62.2 | −15.9 [−22.6, −9.1] |
| GPQA, judge + DES48 | 198 | 14.60 | 1120.0 | 1228.8 | 218.7 | 27.8 | −9.6 [−17.2, −2.0] |
| AIME, judge + DES48 | 90 | 8.35 | 672.9 | 779.1 | 131.3 | 8.9 | +2.2 [−2.2, +6.7] |

**AIME judge + DES48** is the only arm anywhere that gains accuracy: 6.7 → 8.9
(+2.2) with 1.29x fewer passes and 1.28x less total compute. It is **not
statistically significant** — the CI spans zero, and the paired lost/gain is 1/3,
i.e. a net of **two items out of ninety**. Do not report it as a win.

### 4.3 Why DES M=32 fails under the judge — the measurement

100 paired calibvalid items, judge `full_c95` with and without the M=32 pool.
Computed from the records' `committed_judge_per_block`,
`full_commit_eq_draft_per_block` and `remask_per_block`
(602 vs 597 blocks; "override" = a commit whose argmax differs from the draft):

| per block | judge | judge + DES32 | judge + DES48 |
|---|---:|---:|---:|
| commits | 29.31 | **29.29** | 29.32 |
| — of which confirm the draft (`argmax == D`) | 17.12 | 17.03 | 16.86 |
| — of which **override** the draft | 12.19 | **12.26** | 12.46 |
| **re-masks** | 45.34 | **50.80 (+12.0%)** | 46.16 |
| tentative kept | 12.88 | 14.40 | 12.86 |
| **accuracy** | **94.0** | **88.0** | **94.0** |

The judge's *decisions* are unchanged: it commits the same number of slots and
overrides the draft the same number of times. What changes is the **churn** — it
re-masks 12% more slots, i.e. it keeps declaring itself unsure, on the same
content. The pruned forward gives it a subtly different distribution and it has
no way to know: **the judge cannot tell a pruned prediction from a clean one.**
M=48 leaves enough of the router's mass intact that the churn (and the accuracy
loss) does not appear.

This is the mechanism to fix if anyone resumes this line — the judge needs the
pool state as an input, or the pool needs to be held out of the judge forward.

## 5. Distribution effect: accuracy tracks the training mix, speed does not

Training-mix share is by **block** (the unit of a training example), from the
four `build_summary.json` files under `logs/draftalign/trainset_union_r2/`.
Accuracy delta is the trained judge (`full_c95`) vs stock, paired.

| dataset | share of training blocks | Δaccuracy | speed-up (passes/blk) |
|---|---:|---:|---:|
| GSM8K | 41,093 / 65,521 = **62.7 %** | −1.4 | 1.52x |
| MATH | 24,267 / 65,521 = **37.0 %** | −4.4 | 1.35x |
| MBPP | 161 / 65,521 = **0.2 %** | −14.0 | 1.14x |
| HumanEval | **0 %** (never in training) | −13.4 | **1.64x — the fastest** |
| GPQA-Diamond | 0 % | −4.5 | 1.13x |
| AIME | 0 % | −1.1 | 1.33x |

Accuracy damage is monotone in training share across the three trained families
(62.7 % → −1.4, 37.0 % → −4.4, 0.2 % → −14.0). **Speed is not**: HumanEval was
never in the training mix and is the *fastest* arm of all six, while losing 13.4
points. The adapter learned to commit aggressively as a general behaviour, and
learned *where that is safe* only on the distributions it saw.

By **item** rather than block the mix is 79.7 % / 19.5 % / 0.8 % — say which one
you mean.

## 6. The bottom line, in one comparison

calibvalid, 100 items, **honest total compute** (experts/token with the drafter
charged), best arm of each family:

| approach | total exp/token | acc % | needs a drafter? | needs training? |
|---|---:|---:|:--:|:--:|
| stock block 32 | 1090.6 | 95.0 | no | no |
| blind accept | 1085.9 | 95.0 | yes | no |
| trained judge | 903.8 | 94.0 | yes | **yes** |
| judge + DES M=48 | 735.5 | 94.0 | yes | **yes** |
| **stock + DES M=32** | **654.9** | 93.0 | **no** | **no** |

The cheapest arm in the table is the one that required no drafter and no
training: expert pooling applied to the stock model.

## 7. Provenance of every table

| table | records | producer |
|---|---|---|
| §1 rows except GSM8K-500 | `logs/draftalign/da_decode/<ds>_B32_t95_stock/` | `run_da_decode.py --stock` |
| §1 GSM8K-500 | `logs/research_v1/baselines_b32/gsm8k_B32_t95_stock.<split>.shard*of3.jsonl` | `run_research_baselines.py` |
| §2 | `logs/draftalign/da_decode/<ds>_B32_t95_dv-s8_blind_q50/` | `run_da_decode.py --judge blind --tau-acc-q .50` |
| §3 | `.../<ds>_B32_t95_dv-s8_full_c95_ad734af656/` | `run_da_decode.py --adapter ... --judge full` |
| §4.1 | `.../<ds>_B32_t95_stock_p{32,48}desf/` | `run_da_decode.py --stock --pool M --pool-rule des --pool-flush` |
| §4.2, §4.3 | `.../<ds>_B32_t95_dv-s8_full_c95_p{32,48}desf_ad734af656/` | as §3 plus the pool flags |
| §5 shares | `logs/draftalign/trainset_union_r2/{gsm8k_a,gsm8k_b,math,mbpp}/build_summary.json` | `build_train_set.py` via `build_union_index.sh` |
| adapter identity | `logs/draftalign/train/t2b_union_flag/step003057/adapter_config.json` + `.safetensors` | `train.py` |

None of `logs/` is in this repository. `EXCLUDED.md` names the mirrors.

## 8. Reading notes

* **Every number in this file uses `model_key: "b32"`** — the block-32 SFT
  checkpoint, pinned at revision `c351bbc37d240aa6871f167e8f92d694281b0c22`.
  Each per-item record carries the field, so it is checkable.
* **`--pool-rule des`** is the faithful per-forward expert re-vote. It is not a
  static per-block coreset; the two are different algorithms and their numbers
  are not interchangeable.
* **`results/reports/EVAL_TABLE.md`** is regenerated by the hourly sync and was
  slightly stale at snapshot time for the partial math500 cells. Its gsm8k500
  `d_acc` column is *vs the free rule*, not vs stock. Its `splits` dict covers
  only calibvalid / gsm8k500 / math500 / mbpp, so AIME, GPQA and HumanEval appear
  in **no** generated report — §1–§4 above are the only written record of them.
