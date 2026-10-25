# Phase 5: the judge loop in SGLang (data generation only)

# **RULE: SGLANG IS FOR DATA GENERATION ONLY. EVERY BENCHMARK NUMBER (TOKENS/S, PASSES PER BLOCK, SECONDS PER ANSWER, ACCURACY) AND EVERY TRAINING RUN GOES THROUGH THE HF/TORCH HARNESS (`run_da_decode.py`, `train_c1b.py`). NO SPEED MEASURED IN SGLANG IS EVER REPORTED.**

This port exists for one reason: generating the judge's own training states (every pass of every block, with the
judge and the drafter in the loop) costs 54 box-hours for 52M tokens on the batch-1 harness, because the judge's
flag-gated LoRA lived only there. In SGLang the same loop runs over 64–256 blocks per forward. The decisions must be
the harness's; only the kernels differ. This document says what was built, why each piece is exact, what is only
"the same up to bf16 kernels", how parity is measured on the box before any state is trusted, and what is still
not written.

Branch `phase5-loop-guard`, repo NoviceCoderInfinity/diffusion-moe-general-expert. Status at the time of writing:
**every file below is CPU-tested against the harness's own functions on fake models; nothing has run on a GPU or
inside a real SGLang engine.** The first box session is the parity session (section 8). All of it is on the branch
(part 1 commit 1da9b5f, part 2 the next commit).

---

## 1. What the harness does, and what had to move

The decode under test is `run_da_decode.py --judge full --draft s8 --full-redraft-every {1|2} --redraft-ban-rejected
--loop-guard 12` with the 128M judge adapter and the Phase 4 merged drafter. Per block (32 slots):

1. **First draft.** The 1.7B drafter fills every generated slot, 8 static steps of 4 (`drafter.draft_block`, setting
   `s8`), fp32 softmax with the [MASK] column suppressed, top-k by confidence per step; the loop guard may forbid
   tokens that complete a 12-gram already in the answer.
2. **Judge passes.** The 30B with the adapter sees the block full of draft tokens. The *flag* marks the open slots
   that hold a draft token (`open & (cur != [MASK])`); at those positions the LoRA delta and the draft-flag
   embedding are on, everywhere else the model is the stock 30B. Read at every open slot: fp32 softmax, [MASK]
   suppressed, `p` = top-1, `argmax`, `q` = p(draft). Commit `argmax` where `p > 0.95` (strict); if nothing passes,
   commit the single best slot; of the rest, slots where `argmax == draft` stay tentative (keep the draft token),
   the others become [MASK] (holes).
3. **Re-draft.** After every pass (rd1) or every second pass (rd2), if holes exist: the drafter refills ONLY the
   holes, seeing the committed tokens and [MASK] at the tentative slots (it never sees a tentative token), in
   `min(8, ceil(holes/4))` steps, with the rejected draft banned at each hole and the loop guard on.
4. Repeat 2–3 until every slot is committed (cap 32 passes; after the cap the uncommitted slots are re-masked and the
   STOCK rule finishes the block: bf16 confidence, strict > 0.95, one transfer per step).
5. **Closing forward.** One forward with the flag OFF over the final tokens stores the block's KV (so the next block's
   context is the stock model's view of the committed text, not the adapter's).
6. Stop after a block that contains EOS.

`da_decode.judge_full_block` + `da_generate` are the reference; `code/draftalign/adapter.py` is the adapter.

What SGLang lacked:

| Need | SGLang 0.5.10.post1 | What was built |
|---|---|---|
| per-token gated LoRA on q,k,v,o + draft-flag embedding | LoRA is per request, uniform over tokens; no embedding hook | `c1b_judge_lora.py`: instance-level patch of `qkv_proj` / `o_proj`, a flattened flag context, `input_embeds` with the draft vector |
| the judge loop (passes, commit rule, re-draft cadence, cap, closing forward) | `LowConfidence` only (the stock rule) | `judge_core.py` (pure torch, batched) + `c1b_judge_loop.py` (the SGLang algorithm) |
| a drafter that fills a subset of masked slots with bans and the guard | `StaticDraftTraced` fills every mask, no bans | `draft_core.py` + `c1b_judge_draft.py` (`DraftTracedV2`), driven by a per-request side file |
| two models in one loop | one engine, one model | the drafter is a second SGLang server on the same GPU; the judge algorithm calls it once per draft step for the whole batch |
| teacher forcing between blocks (the recorder) | — | `closing_ids`: the closing forward stores vanilla's block; the pipeline's own tokens are recorded |

---

## 2. Files

| File | What | Verified by |
|---|---|---|
| `code/corpus1b/sglang_plugin/judge_core.py` | the batched judge block loop with two callbacks (`forward_fn`, `draft_fn`); copies of `blind_read`, `full_commit_rule`, the stock transfer rule and `get_num_transfer_tokens` | `tests/test_judge_core.py` (15): each copy equals the harness function on random inputs (fp32 and bf16); whole decodes equal `da_generate` token for token and counter for counter for rd0, rd1+ban+guard, rd2+ban, the pass cap (stock fallback), a 2-item lock-step batch, EOS stopping one item early, teacher forcing |
| `code/corpus1b/sglang_plugin/c1b_judge_lora.py` | `attach_judge_adapter`, `JudgeCtx` (flattened flag), `judge_forward_prep` (draft-flag embeds) | `tests/test_judge_lora_sgl.py` (16): outputs `torch.equal` to `adapter.LoRALinear` / `DraftEmbed` under the same flag, fp32 and bf16; all-False flag bit-identical to the base and the delta never computed; real checkpoint key names |
| `code/corpus1b/sglang_plugin/draft_core.py` | `DraftFill` stepper = `drafter.draft_block` without the SeqCache | `tests/test_draft_core.py` (7): equals `draft_block` on first drafts, re-drafts (hidden tentative slots, bans), the loop-guard cycle case; same forward inputs step by step |
| `code/corpus1b/sglang_plugin/c1b_judge_draft.py` | `DraftTracedV2`: the drafter engine's algorithm; side-file protocol; trace rows | `tests/test_judge_draft_plugin.py` (6): run() on fake SGLang modules equals `draft_block` per block in a mixed batch; no side file = `StaticDraftTraced` |
| `code/corpus1b/sglang_plugin/c1b_judge_loop.py` | `JudgeLoopTraced`: the judge engine's algorithm; per-request state from a side file; flags and embeds around every forward; the drafter client (one POST per draft step for the batch, trace polling); judge trace rows; rollback on a re-run block | `tests/test_judge_loop_plugin.py` (11): the glue's per-block summaries and states equal `judge_core.run_blocks` driven directly, decode and record mode, rd1+ban+guard and the pass cap; a run with the flag ignored decides differently; nine deliberate breakages of the glue are each caught |
| `code/judge_round/judge_sglang.py` | the orchestrator: one process per GPU; launches the drafter server and the judge engine; `--mode decode` (parity records) and `--mode record` (training states, `states.NNNNN.npz` with flat arrays and offset tables); resume via `done.txt`; `run.json` refuses a resume with another config | `tests/test_judge_sglang_cli.py` (21): parser, yaml, side files for decode and record items, EOS trimming, records assembly, the npz layout loads with `allow_pickle=False` |
| `code/judge_round/judge_parity.py` | SGLang records vs harness records on the same items: token agreement, accuracy gap, passes per block | `tests/test_judge_parity.py` |
| `code/judge_round/smoke_judge_sgl.sh` | the box session: install, decode 8 items both ways, parity, record 4 items | `bash -n` only |
| `code/corpus1b/sglang_plugin/install.py` | copies the plugin modules into `sglang/srt/dllm/algorithm/` (pinned to 0.5.10.post1) | `--check` lists `JudgeLoopTraced`, `DraftTracedV2` |

Run all CPU tests (104 tests, about 15 s): `python3 -m pytest -q code/corpus1b/tests/test_judge_core.py code/corpus1b/tests/test_judge_lora_sgl.py code/corpus1b/tests/test_draft_core.py code/corpus1b/tests/test_judge_draft_plugin.py code/corpus1b/tests/test_judge_loop_plugin.py code/tests/test_judge_sglang_cli.py code/tests/test_judge_parity.py`.

`install.py` copies only the three algorithm modules (`c1b_judge_draft.py`, `c1b_judge_loop.py`, plus Phase 1's two) into SGLang's algorithm package; `judge_core.py`, `c1b_judge_lora.py` and `draft_core.py` are imported from the repo through `C1B_CODE_DIR` (`install.py` defaults it to `<repo>/code`; the orchestrator sets it for both engines). One copy per process matters: `c1b_judge_lora.JudgeCtx` is a module singleton.

---

## 3. Exactness ledger

"Exact" below means: given the same logits, the same decision; the CPU tests prove it on fake models where the
logits ARE the same. On the GPU the logits differ between SGLang and HF (fused qkv matmul, fused RMSNorm, FlashInfer
or Triton attention, SGLang's MoE kernels, bf16 accumulation order). That difference is the only one, and it is
measured, not assumed (section 8).

| Piece | Status | Why |
|---|---|---|
| LoRA delta `(alpha/r) B(A x)` in bf16, gated per token, concatenated q,k,v | exact | same ops, same dtypes, same order as `LoRALinear.forward`; `torch.equal` in the test |
| all-False flag | exact, bit-identical base | the delta is skipped, no embeds built: the model runs the plain input_ids path |
| draft-flag embedding | exact | `(e.float() + flag * w).to(e.dtype)`, as `DraftEmbed.apply_to` |
| read-out at open slots (`p`, `argmax`, `q`) | exact | copy of `blind_read`/`slot_metrics`, pinned by test |
| commit rule, progress guarantee, tentative/holes | exact | copy of `full_commit_rule`, pinned by test |
| flag of a pass = `open & (cur != [MASK])` | exact | same expression as `judge_full_block` |
| re-draft input (tentative hidden), hole set, steps, ban | exact | same expressions; drafter stepper equals `draft_block` |
| loop guard | exact | `draftalign.loop_guard.LoopGuard` itself, called with the same arguments |
| stock fallback after the cap | exact decision; confidence in the logits' dtype | copy of `stock_pass`'s rule; SGLang's `full_logits` dtype decides the softmax dtype (bf16 in the harness) — pinned by a test on the rule, the dtype is checked on the box |
| closing forward with the flag off | exact in intent | the harness's `kvflush` forward is also flag-off; in SGLang it is the last forward of `run()`, whose KV persists |
| batch lock-step | exact per block | a block's pass counter advances only on passes it took part in; the 2-item test proves single-item equality |
| prefix (what earlier blocks the judge sees) | exact | decode mode: the pipeline's own closed blocks; record mode: vanilla's (teacher forcing), stored by the closing forward |
| EOS | exact after trimming | the harness stops after the block containing EOS and trims the record at the first EOS; SGLang finishes the request on EOS among the block's tokens; records are trimmed the same way |
| base-model numerics | NOT exact | different kernels; Phase 1 measured SGLang-vs-harness vanilla decodes as "accuracy within ±3 points, passes/block within ±5%, token agreement reported, bit identity not expected" (`corpus1b/parity_check.py`). The same bar applies here, with the 95% drafts bar for token agreement |

Two SGLang behaviours the design depends on and that the first session must confirm: `forward_batch.rids` carries the
request ids into the algorithm (the Phase 1 traces relied on it), and `forward_batch.input_embeds`, when set, reaches
the model in a dLLM extend forward (`model_runner.forward_extend` passes `input_embeds.bfloat16()`).

---

## 4. The two engines and their protocol

One GPU runs both: the judge engine (`sgl.Engine`, 30B bf16 ≈ 61 GB + KV) in the orchestrator process and the
drafter server (`sglang.launch_server`, 1.7B ≈ 4 GB + KV) as a subprocess. Both with `--disable-cuda-graph` (the
patched projections branch in Python on the flag; a captured graph would freeze one branch) and
`--disable-radix-cache` (a cached block written with the flag on, or a re-draft block with scattered [MASK]s, must
never be matched as a prefix).

**Judge request** (one per item): `input_ids` = prompt token ids; `max_new_tokens` = the generation cap (decode) or
the teacher answer's generated length (record, with `ignore_eos`). Before sending, the orchestrator writes
`$C1B_JUDGE_SIDE_DIR/<rid>.json`:

```
{"prompt_ids": [...], "mode": "decode" | "record", "max_new_tokens": N,
 "teacher": [[32 ids of decode block 0, prompt tail included], [32 ids of block 1], ...]}   (record mode only)
```

The algorithm reads it when the rid's first block arrives, deletes it, and keeps per rid: the final sequence so far
(prompt + closed blocks = what the judge's KV holds), the block index, the prompt length.

**Drafter request** (one per block per draft step, batched over the judge's batch): `input_ids` = that sequence up
to the block start + the 32-token block as the drafter must see it ([MASK] at the fill slots and at the hidden
tentative slots); `max_new_tokens` = 1 (ends the request after its one block), `ignore_eos`; side file
`$C1B_DRAFTER_SIDE_DIR/<rid>#<block>#<serial>.json`:

```
{"fill": [slot indices], "steps": 8 | min(8, ceil(holes/4)), "ban": [[slot, token], ...],
 "guard": {"n": 12, "prompt_length": L, "prefix_ids": [the sequence before the block]}}
```

The drafter algorithm writes one trace row per block to `$C1B_DRAFTER_TRACE_DIR/trace.<pid>.jsonl` (`ids[32]`,
`conf[32]` = the committing forward's fp32 probability, -1 where not filled, `n_fwd`, `guard_slots`,
`guard_changed`). The judge algorithm polls the trace until every rid of the call has a row (timeout 120 s), then
continues the pass. The HTTP round trip is one POST per draft step per batch.

**Judge trace**: one row per block to `$C1B_JUDGE_TRACE_DIR/trace.<pid>.jsonl` with the block's counters
(`judge_core.block_summary`) and, in record mode, the per-pass states (section 6).

Environment of the judge engine (set by the orchestrator before `sgl.Engine` is built; the scheduler subprocess
inherits it): `C1B_CODE_DIR`, `C1B_JUDGE_ADAPTER`, `C1B_JUDGE_SIDE_DIR`, `C1B_JUDGE_TRACE_DIR`, `C1B_DRAFTER_URL`,
`C1B_DRAFTER_SIDE_DIR`, `C1B_DRAFTER_TRACE_DIR`. Of the drafter server: `C1B_CODE_DIR`, `C1B_SIDE_DIR`,
`C1B_TRACE_DIR`.

Every forward the judge algorithm issues is inside `JudgeCtx.flag_scope(flag, any_)`: judge passes carry the pass's
flag; prefill chunks, stock passes and the closing forward carry an all-False flag. The adapter is attached on the
algorithm's first call; SGLang's own warm-up request goes through the same path.

---

## 5. Modes

**`--mode decode`** (parity and nothing else): items of a benchmark split, exactly as `run_da_decode.py --items`
selects them (`--shard/--nshards` pick the same items on both sides). Output `records.jsonl`: `gen_ids` trimmed at
the first EOS, `pred`, passes per block, drafter forwards, re-draft and guard counters. Used only by
`judge_parity.py`. **Never report a speed from it.**

**`--mode record`** (the product): trainset items (`--index`, `--pids`). The prompt and vanilla's answer come from
the item npz; the judge decodes every block with the pipeline, the closing forward stores vanilla's tokens, so block
b+1 is decoded on vanilla's prefix (teacher forcing between blocks, exactly as Phase 1's states and the 50-hour
plan's recorder). Every pass of every block is recorded (section 6).

---

## 6. Recorded states (record mode)

Per block, per pass `p`:

| Field | Shape | Meaning |
|---|---|---|
| `input` | [32] | the block the judge saw at this pass (draft tokens at tentative/open slots, [MASK] at holes, committed tokens) |
| `flag` | [32] bool | the adapter's gate at this pass = open & (input != [MASK]) |
| `open` | [32] bool | not yet committed |
| `draft` | [32] | the current draft D (re-drafts replace it at the holes) |
| `commit`, `tentative`, `remask` | [32] bool | the pass's decision at the open slots (`forced` when the progress guarantee fired) |
| `argmax`, `p`, `q` | [32] | the judge's read-out at the open slots |
| `redraft` | or null | `holes` [32] bool, the re-drafted `tokens`, their `p_commit`, the drafter's `input`, `n_fwd` |

Plus per block: the first draft and its confidences, the final tokens of the pipeline AND vanilla's (`tokens_out`),
all counters. A training state for the judge at pass p is (`input`, `flag`) with targets = vanilla's tokens at the
open slots; a training state for the drafter at a re-draft is (`redraft.input`, holes) with targets = vanilla's
tokens at the holes. **The sampler that reads these records into `train_c1b.py` / the drafter trainer is NOT written
yet** (section 10).

---

## 7. Cost

Per block: `n_judge` judge forwards (Phase 4 measured 7.7 on average) plus the drafter's forwards (8 for the first
draft, up to 8 per re-draft) plus the closing forward, all batched over the engine's running blocks. The batch is
lock-step: a batch's judge forwards = the slowest block's pass count, so stragglers cost; `--max-running` trades
that against GPU utilisation (start at 64, tune on the box; Phase 1's vanilla generation lived with the same
property). The drafter round trips add one POST per draft step per batch.

Expectation, from Phase 1's vanilla rate (1.3–7 Mtok kept per GPU-hour) and the extra forwards: roughly 4 box-hours
for 52M tokens of states, against 54 on the batch-1 harness. **A projection, not a measurement; the first session
gives the number.**

---

## 8. The box session: parity before anything else

`bash code/judge_round/smoke_judge_sgl.sh install | decode | parity | record | report` (stage markers under
`logs/phase5/judge_sgl/`). What each stage proves:

1. `install`: the plugin modules are discoverable by the pinned SGLang (`install.py --check` lists `JudgeLoopTraced`
   and `DraftTracedV2`).
2. `decode`: 8 GSM8K items in SGLang (judge + drafter, rd1 + ban + guard, 128M adapter, Phase 4 merged drafter) on
   GPU 0; the same 8 items on the harness on GPU 1. The first thing to check in the SGLang log: the adapter attached
   (385 tensors, draft vector norm > 0), `rids` present (no `?0` request ids in the traces), drafter trace rows
   complete.
3. `parity`: `judge_parity.py`. **Acceptance, written before the run:** token agreement ≥ 95% (Phase 1's drafts
   bar), accuracy gap within ±3 points on the full parity set (8 items are a smoke; the real parity set is one
   quarter of a benchmark, ≥ 100 items), passes per block within ±5%. Below that: stop, diff the first diverging
   pass (the judge trace has every pass; the harness has `--trace`), find the kernel or the bug, no training on
   these states.
4. `record`: 4 trainset items end to end; the states file exists and its per-pass counts match the decode counters.

Then a 100-item parity set, then the first real recording run.

---

## 9. Known differences and failure modes

- **Kernels.** Expect some commit decisions to flip near 0.95 between SGLang and HF. Parity is a statistical bar
  (section 8), not bit identity. If a flip rate is high, compare `p` histograms between the two traces first.
- **Stock fallback dtype.** The harness's stock pass ranks confidences in bf16; SGLang's `full_logits` dtype decides
  here. Read `full_logits.dtype` from the first run's log. The fallback is reached only after 32 judge passes with
  open slots, which cannot happen at block 32 with the progress guarantee, so this is a corner.
- **Chunked prefill.** Long prompts are prefilled in chunks; every chunk is a mask-free batch handled by the
  fast path with an all-False flag. Nothing to do.
- **Request ids.** If `forward_batch.rids` is missing, both algorithms silently degrade (the drafter to plain s8, the
  judge raises on the missing side file). The judge raising is the intended guard.
- **Drafter prefix cost.** With the radix cache off, every drafter request re-prefills its prefix (1.7B, up to 2k
  tokens, batched): a few ms per block. Accepted for exactness.
- **Memory.** `--mem-fraction` and `--drafter-mem-fraction` are fractions of the WHOLE GPU. SGLang's own
  `mem_fraction_static` is a fraction of the memory free when that engine starts (`model_runner_kv_cache_mixin.profile_max_num_token`:
  the static pool = pre-load free memory × the fraction), and the drafter starts first, so the orchestrator passes the judge
  `0.62 / (1 − 0.18 − 0.02)`. At the defaults the judge's pool is ~59 GiB = weights 56.8 GiB + ~2.5 GiB of KV (96 KiB per token),
  about 26k tokens, so SGLang admits ~20 live requests, not `--max-running 64`. Arithmetic, not measured. After the first run
  reads the free memory, raise `--mem-fraction` to about 0.68–0.70 for throughput. If the judge engine OOMs on logits
  (`full_logits` is [B·32, 151936] fp32 ≈ 20 MB per block), lower `--max-running`.
- **Straggler blocks.** One block at 30 passes holds the batch 30 forwards. Watch `passes_per_block` spread; a cap
  below 32 would change the decode rule and is NOT allowed for parity or records.

---

## 10. Not written yet

1. **The state sampler for training.** `train_c1b.py` samples states from the corpus npz items; a sampler over the
   recorded per-pass states (section 6), and the drafter trainer's counterpart, do not exist. Without them the
   records are data on disk. About one day each.
2. **The 100-item parity set and the box verification** of the two SGLang behaviours named in section 3.
3. **Speed tuning** (`--max-running`, attention backend) after parity passes. Never reported as a benchmark.
