# ARCHITECTURE

How the decode loop works, what the LoRA adapter touches, and exactly what is
frozen. Source of truth: the module docstrings of
`code/draftalign/da_decode.py` and `code/draftalign/adapter.py` — read those
before changing anything here.

---

## 1. The model

`JetLM/SDAR-30B-A3B-Chat-b32` @ `c351bbc37d240aa6871f167e8f92d694281b0c22`.

| | |
|---|---|
| architecture | `SDARMoeForCausalLM` — block-diffusion (masked-denoising) decoder |
| layers | 48, every one a MoE layer (`decoder_sparse_step: 1`, `mlp_only_layers: []`) |
| hidden size | 2048 |
| attention | 32 query heads × head_dim 128 (q: 2048→4096), 4 KV heads (k, v: 2048→512), o: 4096→2048 |
| experts | 128 per layer, **top-8** routed, `norm_topk_prob: true`, no shared expert |
| expert size | `3 × 2048 × 768` = **4,718,592** params (gate, up, down at `moe_intermediate_size` 768) |
| vocab | 151,936, `tie_word_embeddings: false` |
| total params | ≈ 30.5 B (48 × ~623 M per layer + 622 M for the two embedding matrices) |
| block size | 32 (`MASK_ID = 151669`) |
| dtype / attention | bf16, `attn_implementation="sdpa"` |

`code/compat/flash_attn/` is a **pure-PyTorch shim**, not real flash-attention.
The vendor's `modeling_sdar_moe.py` has an unguarded top-level
`from flash_attn.ops.triton.layer_norm import rms_norm_fn`, so the module cannot
be imported without it, regardless of `attn_implementation`. The shim supplies
only `rms_norm_fn`. Every entry point puts `code/compat` on `sys.path` **before**
`code`. On an image that *also* has a real `flash_attn` installed, that ordering is what
breaks the import: transformers 4.52.4 then imports `flash_attn.layers.rotary`, which resolves
into the shim (it has no `layers`) — see `WORKTREE.md`, *Known environment hazard*.

## 2. The stock decode loop (what we are trying to beat)

`sdar_harness/decode.py::block_diffusion_generate`. Per 32-token block:

1. **prefill** — one forward over the prompt (and every finished block), KV cached.
2. **denoise** — repeat: one forward over the 32-token block; bf16 softmax;
   commit every slot whose top-1 probability is **strictly >** `threshold`
   (0.95); at least one commit per pass (the highest-confidence open slot) so it
   always terminates. Stop when no `[MASK]` remains.
3. **kv flush** — one more forward with `store_kv=True` to write the finished
   block into the cache.

Cost accounting: the prefill and the flush are *not* charged to `passes/blk` —
`passes/blk = (n_forward − n_prefill − n_flush) / n_blocks`. On GSM8K-500 this is
**10.06**, i.e. the 30B model reads each block ten times.

The MoE forward is replaced throughout by
`sdar_harness/moe_instr.py::instrumented_moe_forward`, a **bit-exact**
hit-experts-only dispatch that also records the `selected_experts` tensor into a
module-level `SINK`. That is where `unique_expert_layer_loads` comes from: the
count of distinct (layer, expert) pairs that actually executed.

## 3. The drafter

`JetLM/SDAR-1.7B-Chat-b32` @ `2c83c05d84be90a01c72443b2123ebf0376f6fd4` — dense,
1.7 B, **same tokenizer and same `MASK_ID`** (verified 2026-09-06), so its tokens
drop straight into the 30B's block.

`draftalign/drafter.py::draft_block` fills the block's open slots. Settings:
`s4` (4 denoising steps), `s8` (8 steps — used for every published cell), `dyn90`
(dynamic, threshold 0.90). `draftalign/seqcache.py` keeps the drafter's KV cache
persistent across blocks, and each finished 30B block is flushed into it, so the
drafter always conditions on the **committed** prefix, never on its own guesses.

The drafter is charged honestly: one drafter forward = **364.64
expert-equivalents** = (1.7 B params minus its token-embedding matrix) / (one 30B
expert = 4,718,592 params) = 1,720,573,440 / 4,718,592.

## 4. The three decode rules

### 4.1 `--judge blind` — training-free, zero extra MoE forwards

The drafter fills the block → `D`. Then the block's **first ordinary stock pass**
runs exactly as stock (bf16 softmax, strict `>` threshold, ≥1 forced commit).
From that *same* forward, in fp32 with the `[MASK]` logit suppressed, read for
every slot that was masked at that forward:

* `a_i` = argmax, `p_i` = its probability,
* `q_i = p_i(D_i)` — the model's probability of the **draft's** token.

After the stock commits, accept `D_i` at every **still-open** slot with
`q_i > tau_acc_q` (published cells: **0.50**). A stock commit is never
overridden. `n_judge = 0`; the pass is counted as a stock pass. The block then
finishes with ordinary stock passes.

This is the cheapest possible use of a drafter: the information was already in a
forward we were paying for anyway.

### 4.2 `--judge full` — the trained regime

Per block, **one MoE forward over a block with no masks at all**: every open
valid slot holds the drafter's token. Read at *every* open slot (fp32 softmax,
`[MASK]` suppressed) → `a_i`, `p_i`, `q_i`.

Per slot (`da_decode.py::full_commit_rule`):

| condition | outcome |
|---|---|
| `p_i > tau_commit` (strict) | **commit** to `a_i` |
| else and `a_i == D_i` | keep `D_i` as **tentative** context |
| else | **re-mask** |

`--full-remask-only` is the ablation that never keeps a tentative token.

The same judge pass then repeats on the new state — committed slots hold their
tokens, tentative slots hold `D_i`, re-masked slots hold `[MASK]`, and the read
covers tentative + masked — until every valid slot is committed. At least one
commit per pass (highest `p_i` among open slots), so it terminates. After
`--full-max-passes` (default 32) judge passes the remainder falls back to the
stock rule. Then the kv flush, then the drafter flush.

These forwards are counted as `judge` forwards and they are what
`passes_per_block` measures for this arm.

> A third rule, `--judge halves` (two complementary-half forwards per block, each
> slot read from the forward in which it was masked), was the original design and
> is still in the code. It is superseded — every current headline uses `blind` or
> `full`.

### 4.3 DES pooling — `--pool M --pool-rule des --pool-flush`

Orthogonal to the decode rule: a **per-forward top-M expert restriction** applied
to every *block* forward (`denoise`, `judge`, `rejudge` — `POOLED_KINDS`),
**never** the prefill, and the flush only with `--pool-flush`.

Per layer, per forward:

1. take the fp32 router softmax of all 32 block rows;
2. the **valid** slots vote; the pool is the top-M experts by vote;
3. rules: `full` = sum of the full softmax (dMoE); `top8` = sum of the
   **renormalised** top-k weights; **`des`** = sum of the **unnormalised** top-k
   weights — this is DES-Vote's Algorithm 3, `sdar_harness/desvote.py:96-99`;
4. zero the probabilities outside the pool, take top-8, renormalise, run the
   hit-only expert loop. For `M ≥ 128` this is bit-identical to the unrestricted
   forward.
5. the forward writes the **restricted** `selected_experts` into `moe_instr`'s
   `SINK`, so the expert counter counts what actually ran.

The pool is **re-elected on every forward**. This is what distinguishes it from
a static per-block coreset (`sdar_harness/coreset.py`, kept but unused), which
freezes one expert set from a single readout forward. The two are different
algorithms and their numbers are not interchangeable.

### 4.4 Numerics discipline

The stock rule keeps `decode.py`'s **bf16** softmax and strict `>` threshold
untouched. fp32 is used **only inside the judge** (`moe_probe.slot_metrics`),
where the `[MASK]` logit is suppressed before the softmax. This is why a
`--stock` run of `run_da_decode.py` is required — and checked — to reproduce
`block_diffusion_generate` token for token.

## 5. The trained judge: what the LoRA touches

`code/draftalign/adapter.py`. **No `peft`** — the wrappers are hand-written so
the arithmetic can be pinned exactly.

### 5.1 Trained (26,740,736 parameters, ≈ 0.088 % of ~30.5 B)

| component | shape | count |
|---|---|---|
| `LoRALinear` on `self_attn.q_proj` | A [32, 2048], B [4096, 32] | 196,608 / layer |
| `LoRALinear` on `self_attn.k_proj` | A [32, 2048], B [512, 32] | 81,920 / layer |
| `LoRALinear` on `self_attn.v_proj` | A [32, 2048], B [512, 32] | 81,920 / layer |
| `LoRALinear` on `self_attn.o_proj` | A [32, 4096], B [2048, 32] | 196,608 / layer |
| — subtotal | 4 modules × 48 layers = **192 modules** | 557,056 × 48 = **26,738,688** |
| `da_draft_embed` (`DraftEmbed.weight`) | [2048], fp32, zero-init | **2,048** |
| **total** | | **26,740,736** |

`r = 32`, `alpha = 64.0`, `dropout = 0.0`. `y = Wx + (alpha/r)·B(Ax)`; `A` is
kaiming-uniform, **`B` is zeros**, so the adapter is the *identity* at
initialisation — bit-identical outputs, pinned by a test. A/B are kept in fp32
as AdamW master weights and cast to bf16 inside the forward, so the matmuls run
exactly like the base ones.

Confirmed in the trainer's own log line for the checkpoint in use:
`wrapped {'lora': 192, 'gates': 0, 'draft_embed': 1}, trainable {'lora': 26738688, 'gate': 0, 'draft_embed': 2048, 'total': 26740736}`.
Adapter file: `logs/draftalign/train/t2b_union_flag/step003057/adapter.safetensors`,
107,005,856 bytes, sha256[:16] = **`734af65681ab6bb8`** (the `_ad734af656` cell tag).
`n_tensors: 385` = 192 modules × 2 + 1.

### 5.2 Frozen

Everything else, explicitly:

* **the router** (`mlp.gate`, Linear 2048→128) in all 48 layers —
  `train_gates: false` in this run's config;
* **all 128 experts** in all 48 layers (~29.0 B parameters);
* **`embed_tokens`** and **`lm_head`** (151,936 × 2048 each);
* every RMSNorm and every rotary parameter;
* the MLP-only layers — there are none.

The adapter has an optional `GateFP32` wrapper for training the router in fp32
(the master copy is fp32, the forward casts to bf16 before the matmul so the
router logits are bit-identical to stock at initialisation). **It was not used
for this checkpoint** (`train_gates: false`, `gates: 0`).

### 5.3 The draft flag — why the adapter is not merged

The single trainable vector `da_draft_embed` [2048] is **added to the input
embedding of every position whose input token is a tentative draft token**, and
the *same* per-position 0/1 flag `g_i` gates every LoRA delta:

```
y_i = W x_i + g_i · (alpha/r) · B A x_i
```

So the model can *see* which slots are tentative, and the adapter's delta is
**confined to them**. At an all-False flag the delta is skipped entirely and no
vector is added: the model is **bit-identical to the frozen base**.

Which forwards arm the flag (`run_da_decode.load_adapter_into` +
`da_decode`'s `MoEMeter(draft_flag=True)`):

| forward | flag |
|---|---|
| a `--judge full` judge pass | armed at the block's **open non-`[MASK]` (tentative draft) slots** |
| prefill | all-False |
| kv flush | all-False |
| stock denoise pass | all-False |
| the halves / blind judges | all-False |

`merge_lora()` **refuses** on a flag-gated adapter (metadata `draft_embed: true`):
a merged weight applies at every position, which is exactly what the gate exists
to prevent. The flag lives in a module-level context (`adapter.CTX`, armed with
`flag_scope`) because the delta is applied deep inside the model's own attention
forward, which this code does not own. `CTX.flag is None` is the pre-flag
behaviour: an ungated delta everywhere.

## 6. Training

`code/draftalign/train.py`, `train_forward.py`, `dataset.py`, `losses.py`.

**Loss** (`losses.py::masked_ce_and_certainty`):

```
loss = mean_i CE(logits_i, T_i)  +  beta · mean_{i : argmax_i == T_i} H(softmax(logits_i / temp))
```

with `beta = 2.0`, `temp = 0.5`. The entropy term is dParallel-style *certainty
forcing* (arXiv 2509.26488), applied **only where the model is already right**,
so it sharpens a correct distribution instead of pushing a wrong one. The
`[MASK]` column (id 151669) is **deleted** from the logits before the entropy
softmax — `torch.cat([t[:, :mask_id], t[:, mask_id+1:]], dim=1)`, not set to
`-inf`, because `-inf` gives a `0·log 0` NaN gradient. Otherwise the model could
lower its entropy by piling probability on `[MASK]`, which is never a legal
output.

Logits are fp32 and computed on the ~32 active block rows only
(`active_logits` applies `lm_head` to those rows), so the `[S, 151936]` tensor is
never materialised.

**Training states** (`dataset.py::StateSampler`). One example is an (item, block)
pair; its state is drawn fresh on each visit:

* with probability `p_pass1` (0.4) nothing is committed — the pass-1 state the
  judge actually meets at inference;
* otherwise a recorded denoising pass `p` is drawn uniformly and the committed
  set is everything the trajectory had committed before pass `p`;
* every open (valid, uncommitted) slot then holds the **drafter's token** with
  probability `1 − m` and `[MASK]` with probability `m`, where `m ~ U(0,1)`;
  `p_draftfull` (0.3 in this run) puts extra mass on `m = 0`;
* with probability `p_allmask` (0.0 in this run; 0.2 in round 1) the whole open
  set is `[MASK]` instead — the stock regime, so the adapter does not forget
  mask-init decoding.

The prefix is **clean**: prompt + the *final* tokens of earlier blocks, which is
exactly what the inference KV cache holds. Loss rows = the open valid slots.
Everything is deterministic given (seed, epoch, example), so every rank and every
rerun sees the same state.

**Distributed**: one **full bf16 model per rank** (61 GB of weights, no
sharding), batch 1 per micro-step, gradient accumulation, and a manual
all-reduce of the adapter gradients at the end of each optimiser step. DDP is
deliberately *not* used — the training forward bypasses `model.forward`, so DDP's
autograd hooks would never see the graph.

**The run in use**: `t2b_union_flag`, resumed from `t2a_gsm8k_flag/step001057`,
7 ranks × grad-accum 5, LoRA lr 2e-5, warmup 50, `min_lr_frac 0.5`, `max_seq
1200`, 150 held-out items, **3,057 steps**. See `DATA.md` for the data.

## 7. Instrumentation and record keeping

* `da_decode.MoEMeter` wraps the model: times every forward (device-synchronised),
  labels it by kind (`prefill`/`denoise`/`kvflush`/`judge`/`rejudge`), attributes
  `SINK`'s expert counts to it, and arms the draft flag when asked.
* `da_decode.DABlock` (a `BlockTrace`) accumulates per-block counters:
  `commit / tentative / remask / forced` per judge pass, drafter agreement,
  anchors, pool usage.
* `da_decode.make_record` writes one JSON object per item; `save_trace` writes a
  per-block NPZ next to it.
* `research_protocol.LaunchLog` appends a start and an end record (with
  `config_hash`, the full config, GPU identity and exit code) to
  `results/audits/research_v1/launches.jsonl` for **every** run.
* `research_protocol.config_hash` is the 16-hex sha256 of the canonical JSON of
  the config, and it goes into every record; `prompts.json` and
  `research_protocol_v1.json` are sha256-checked on load.
