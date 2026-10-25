# Phase 1 corpus: continuing data generation (wave A done → wave B)

Written 2026-09-29 (IST) from the files on the wave-A box (`/workspace/dmge`, branch `phase1-datacorpus`, HEAD 32b9b57).
Every number below comes from a file on disk; the path is given next to it. Anything **inferred** (a projection, or a
cause the logs no longer show) is labelled that way. All times are IST unless marked UTC.

Short version:

* **Wave A is completely generated.** All 287,222 wave-A prompts were decoded (the done set equals the wave-A set
  exactly), 165,989 were kept, 165,438 became training items, which hold **74,759,690 corpus tokens** (prompt plus
  kept answer).
* **Wave B has not started.** Nothing exists under `logs/c1b/gen/waveB`, `logs/c1b/draft/waveB`,
  `logs/c1b/trainset/waveB` or `logs/c1b/waveB_state`.
* **What wave B has left:** 1,952,150 prompts. At wave A's measured rates they should give about 1.13M kept items and
  **about 509M corpus tokens** (inferred). That is about 28 h of SGLang generation, about 4 h of drafting and about
  1.5 to 2 h of conversion on 8x RTX PRO 6000 (inferred).
* **Toward the 1B target:** A + B comes to about **584M tokens, 58% of 1B**. The plan's 1B needs more prompts than the
  lists hold (see §3.4).
* **Do not run `run_wave_b.sh` unchanged.** It has four issues that matter in practice (§4.4). The biggest two:
  * it drafts with the settings that ran out of GPU memory in wave A;
  * its per-domain time box would cut maths off unless the generation window is about 60 h or longer.

---

## 1. What exists

### 1.1 Prompt lists: `data/c1b/prompt_lists/<domain>.jsonl`

Each row has `pid, domain, source, source_id, prompt, gen_cap, key, meta{bucket,…}, wave`. The counts below were
computed by scanning every row of the five files. They are the authority: `manifest.json` is off by 1 to 5 rows for
code and chat, because `sanitize_lists.py` later dropped credential-like rows (manifest `dropped_credential_like`).

| domain | wave A | wave B | total | B/A | gen_cap |
|---|---:|---:|---:|---:|---|
| maths | 98,772 | 670,037 | 768,809 | 6.784 | 1024 |
| science | 34,342 | 232,900 | 267,242 | 6.782 | 1024 |
| code | 90,986 | 619,012 | 709,998 | 6.803 | 512 (A 89,234 / B 607,148); 1024 (A 1,752 / B 11,864) |
| if | 19,794 | 133,511 | 153,305 | 6.745 | 1024 |
| chat | 43,328 | 296,690 | 340,018 | 6.848 | 1024 |
| **all** | **287,222** | **1,952,150** | **2,239,372** | 6.797 | |

The same totals appear in `logs/c1b/train/main/seen_report.md`: 2,239,372 prompts on disk, 287,222 in wave A.

Per source (source tag = the middle field of the pid, `<domain>-<tag>-<h16>`):

| domain | source tag | A | B |
|---|---|---:|---:|
| maths | deepmath | 11,660 | 79,559 |
| maths | gsm8k | 296 | 2,091 |
| maths | mathtrain | 189 | 1,386 |
| maths | numina | 26,485 | 179,515 |
| maths | numinaolym | 9,022 | 60,978 |
| maths | omi2gsm | 9,169 | 62,459 |
| maths | omi2math | 33,620 | 228,380 |
| maths | omr | 8,331 | 55,669 |
| science | arc | 450 | 2,918 |
| science | nemosci_mcq | 1,095 | 7,675 |
| science | nemosci_rqa | 4,276 | 29,156 |
| science | osr2 | 19,267 | 130,733 |
| science | webinst | 9,254 | 62,418 |
| code | apps | 14 | 111 |
| code | cc | 1,055 | 6,921 |
| code | cf | 76 | 647 |
| code | mbpptrain | 30 | 268 |
| code | oci | 89,204 | 606,880 |
| code | taco | 607 | 4,185 |
| if | ifcomp | 19,278 | 130,143 |
| if | ifmulti | 73 | 484 |
| if | ifnemo | 420 | 2,733 |
| if | ifrlvr | 23 | 151 |
| chat | oasst2 | 683 | 4,340 |
| chat | ultrachat | 18,669 | 128,331 |
| chat | wildchat | 23,976 | 164,019 |

The lists fall short of the plan's per-domain targets (`common.TARGET_PROMPTS`: maths 894K, code 710K, science 455K,
IF 940K, chat 342K), mostly in IF and science. `manifest.json` → `domains.<d>.buckets.shortfall` gives the shortfall by
bucket:

* **IF:** composed short by 649,579 and pool short by 137,116.
* **Science:** mc short by 173,370 and free short by 14,388.
* **Maths:** gsm short by 88,420, math_hard by 31,424 and math_easy by 5,347.
* **Chat:** short by 1,977 in all.
* **Code:** meets its target.

Late decontamination is described in `code/corpus1b/postdecontam.py`. MATH-test (`math_test`) was screened after the
lists were built. The flags live in `data/c1b/prompt_lists/postdecontam_<domain>.json`, and the `list_sha256` in each
file matches the current list, checked with `sha256sum` on all five. `to_trainset.py` drops any flagged pid, but
`gen_sglang.py` does **not** skip them: flagged prompts are still generated.

| domain | flagged, wave A | flagged, wave B |
|---|---:|---:|
| maths | 1,185 | 8,214 |
| science | 0 | 6 |
| code | 0 | 0 |
| if | 0 | 0 |
| chat | 4 | 22 |

### 1.2 Wave A results by domain

Sources:

* **Generated / kept / reject reasons:** every row of `logs/c1b/gen/waveA/gpu*/records.*.jsonl.gz`.
* **Items and corpus tokens:** `logs/c1b/trainset/waveA/index.jsonl`.
* **Verify:** `logs/c1b/waveA_state/verify.jsonl`.

| domain | generated | kept | rejected (by reason) | drafted (= kept) | excluded late-decontam | items | corpus tokens |
|---|---:|---:|---|---:|---:|---:|---:|
| maths | 98,772 | 44,416 (45.0%) | wrong_answer 41,397; no_eos 12,959 | 44,416 | 548 | 43,868 | 21,666,293 |
| science | 34,342 | 6,930 (20.2%) | wrong_answer 23,204; no_letter 2,990; no_eos 1,218 | 6,930 | 0 | 6,930 | 2,356,179 |
| code | 90,986 | 66,557 (73.2%) | failed 12,208; wrong_answer 11,584; no_eos 476; timeout 130; no_checker_runtime 31 | 66,557 | 0 | 66,557 | 33,963,179 |
| if | 19,794 | 8,852 (44.7%) | wrong_answer 10,236; no_eos 706 | 8,852 | 0 | 8,852 | 2,791,936 |
| chat | 43,328 | 39,234 (90.6%) | no_eos 1,997; not_english 1,273; loop 824 | 39,234 | 3 | 39,231 | 13,982,103 |
| **total** | **287,222** | **165,989** | **121,233** | **165,989** | **551** | **165,438** | **74,759,690** |

How the totals reconcile:

* **Generated, 287,222:** equals the sum of `wc -l logs/c1b/gen/waveA/gpu*/done.txt`. By GPU:

  | gpu0 | gpu1 | gpu2 | gpu3 | gpu4 | gpu5 | gpu6 | gpu7 |
  |---:|---:|---:|---:|---:|---:|---:|---:|
  | 35,950 | 35,957 | 35,914 | 35,984 | 35,670 | 35,657 | 35,821 | 36,269 |

* **Kept and rejected:** 165,989 kept + 121,233 rejected = 287,222. The kept count is also the sum of
  `wc -l logs/c1b/draft/waveA/gpu*/done.txt` (every kept item was drafted) and the sum of `kept_records` in
  `verify.jsonl`.
* **Items:** 165,989 − 551 late-decontam exclusions = 165,438 items, with no_draft 0, bad 0 and no_usable_block 0
  (`logs/c1b/trainset/waveA/build_summary.json`, last run = chat).
* **Tokens:** 74,759,690 = `logs/c1b/waveA_state/main_max_tokens` = the `corpus_tokens` sum over `index.jsonl`.
  `corpus_tokens = prompt_len + n_gen` (`to_trainset.py`).
* **All tokens processed:** generated answer tokens were 89,610,452 and prompt tokens were 63,413,159, 153.02M in all.
  This matches "153.02M tokens incl. prompts" in `logs/c1b/train/main/seen_report.md`.
* **Drafted blocks (kept blocks):** 1,316,563 in all.

  | maths | science | code | if | chat |
  |---:|---:|---:|---:|---:|
  | 608,301 | 35,036 | 248,389 | 68,032 | 356,805 |

Of these 74.76M tokens, training saw only **28.50M (38%)**, on 65,867 items touched (`logs/c1b/train/main/seen_report.md`,
`main/final/trainer_state.json` tokens = 28,498,676). So wave A itself still has about 46M tokens that were never
trained on (§6).

### 1.3 Wave A per source (keep rate, conversion, tokens/item)

This table is the basis for the projections in §3. Keep rate = kept / generated. Conversion = items / kept (below 100%
only where the late decontam removed items).

| domain | source | A gen | keep | conv | tok/item | A items | A tokens |
|---|---|---:|---:|---:|---:|---:|---:|
| maths | deepmath | 11,660 | 44.7% | 95.9% | 681 | 5,000 | 3,406,156 |
| maths | gsm8k | 296 | 97.0% | 100% | 240 | 287 | 68,801 |
| maths | mathtrain | 189 | 77.2% | 94.5% | 450 | 138 | 62,036 |
| maths | numina | 26,485 | 51.2% | 99.5% | 426 | 13,490 | 5,741,403 |
| maths | numinaolym | 9,022 | 19.6% | 99.4% | 619 | 1,756 | 1,087,387 |
| maths | omi2gsm | 9,169 | 83.8% | 100% | 269 | 7,682 | 2,068,057 |
| maths | omi2math | 33,620 | 44.1% | 98.4% | 589 | 14,592 | 8,595,923 |
| maths | omr | 8,331 | 11.2% | 98.9% | 690 | 923 | 636,530 |
| science | arc | 450 | 93.1% | 100% | 312 | 419 | 130,847 |
| science | nemosci_mcq | 1,095 | 67.7% | 100% | 426 | 741 | 315,963 |
| science | nemosci_rqa | 4,276 | 2.7% | 100% | 720 | 117 | 84,198 |
| science | osr2 | 19,267 | 25.8% | 100% | 306 | 4,969 | 1,521,282 |
| science | webinst | 9,254 | 7.4% | 100% | 444 | 684 | 303,889 |
| code | apps | 14 | 57.1% | 100% | 445 | 8 | 3,559 |
| code | cc | 1,055 | 14.9% | 100% | 451 | 157 | 70,798 |
| code | cf | 76 | 7.9% | 100% | 309 | 6 | 1,852 |
| code | mbpptrain | 30 | 46.7% | 100% | 213 | 14 | 2,982 |
| code | oci | 89,204 | 74.2% | 100% | 510 | 66,209 | 33,799,105 |
| code | taco | 607 | 26.9% | 100% | 521 | 163 | 84,883 |
| if | ifcomp | 19,278 | 45.2% | 100% | 315 | 8,706 | 2,738,556 |
| if | ifmulti | 73 | 37.0% | 100% | 356 | 27 | 9,622 |
| if | ifnemo | 420 | 25.7% | 100% | 378 | 108 | 40,799 |
| if | ifrlvr | 23 | 47.8% | 100% | 269 | 11 | 2,959 |
| chat | oasst2 | 683 | 92.7% | 99.8% | 235 | 632 | 148,823 |
| chat | ultrachat | 18,669 | 93.0% | 100% | 436 | 17,370 | 7,570,890 |
| chat | wildchat | 23,976 | 88.6% | 100% | 295 | 21,229 | 6,262,390 |

Tokens per item by domain (index): maths 494, science 340, code 510, if 315, chat 356. Code items are long because of
their prompts: code prompt tokens were 39.4M of code's 48.6M generated-plus-prompt tokens.

---

## 2. Exactly where wave A ends

### 2.1 How a prompt is assigned to wave A or B

The assignment is a hash, not a position in the list. `code/corpus1b/common.py`:

```python
WAVE_A_FRACTION = 0.128              # 128M of 1B tokens; a hash of the pid decides, so wave A is uniform
def wave_of(pid, frac_a=WAVE_A_FRACTION):
    u = int(sha256_hex("c1b-wave|" + pid)[:12], 16) / float(16 ** 12)
    return "A" if u < frac_a else "B"
```

* `build_prompt_lists.py` sets `r["wave"] = common.wave_of(r["pid"])` after quota filling (step 6 of its docstring) and
  writes the result into each row's `wave` field. Every runner reads that **field**; nothing recomputes the hash at
  run time.
* The pid is `f"{domain}-{source_tag}-{sha256(source|source_id)[:16]}"` (`common.make_pid`), so membership is
  deterministic and stable across rebuilds.
* The measured wave-A share is 287,222 / 2,239,372 = 12.83%.
* Quotas (`sources.QUOTAS`, `sources.FILL_ORDER`, per-source `quota`) decide which rows enter a list, in ascending
  `order_key(pid)` order. They are applied **before** the wave split, so both waves have the same mix; B/A is 6.75 to
  6.85 in every domain and 6.3 to 8.5 in every source.
* **If you rebuild the lists, wave membership of the surviving pids does not change, but the set of rows can** (for
  example if a source revision changes). Do not rebuild the lists for wave B: the current files are the ones wave A was
  cut from (sha256 values in `postdecontam_<domain>.json` → `list_sha256`).

### 2.2 How work is sharded across GPUs and ordered

**Generation** (`gen_sglang.py`):

* `shard_of(pid, n) = int(sha256("c1b-shard|" + pid)[:12], 16) % n`.
* Each process takes the rows of its wave and `--domains` whose shard equals `--gpu`, minus the pids already in its
  `done.txt`. It sorts them by `order_key(pid)` (salt `c1b-order`) and decodes them in chunks:
  * `--chunk` 4096 by default;
  * with `--stop-at`: first chunk `--first-chunk` 1024, then sized from the measured rate, never below
    `--min-chunk` 256.
* The shard of a pid depends only on the pid and `--num-gpus`. **Keep `--num-gpus 8`** in wave B, or a pid could move
  to a different GPU folder. That would be harmless for correctness, since each GPU only reads its own `done.txt`, but
  it breaks per-GPU resume after a change of GPU count.

Wave-B prompts per GPU shard (computed with the same hash):

| domain | gpu0 | gpu1 | gpu2 | gpu3 | gpu4 | gpu5 | gpu6 | gpu7 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| maths | 84,287 | 83,937 | 83,726 | 83,564 | 83,684 | 83,793 | 83,647 | 83,399 |
| science | 29,423 | 29,228 | 29,299 | 28,883 | 29,205 | 28,942 | 29,079 | 28,841 |
| code | 77,698 | 77,551 | 77,140 | 77,211 | 77,467 | 77,566 | 77,280 | 77,099 |
| if | 16,866 | 16,528 | 16,625 | 16,668 | 16,803 | 16,768 | 16,630 | 16,623 |
| chat | 37,145 | 37,116 | 37,127 | 37,116 | 37,011 | 36,984 | 37,223 | 36,968 |
| **total** | 245,419 | 244,360 | 243,917 | 243,442 | 244,170 | 244,053 | 243,859 | 242,930 |

**Drafting** (`draft_sglang.py`): a **different** hash, `shard_of("draft|" + pid, ngpu)`. The GPU that drafts an item
is independent of the GPU that generated it. It streams every kept record under `logs/c1b/gen/wave<W>/gpu*/` that is
not in its own `logs/c1b/draft/wave<W>/gpu<k>/done.txt`. As a result, the draft step after domain *d* drafts whatever
is kept and undrafted, across all domains generated so far.

### 2.3 Done markers: what "done" means, in order of authority

| marker | written by | meaning |
|---|---|---|
| `logs/c1b/gen/wave<W>/gpu<k>/records.<n>.jsonl.gz` | `gen_sglang.finish()` | one chunk; written to `*.tmp.gz`, then `os.replace` (atomic) |
| `logs/c1b/gen/wave<W>/gpu<k>/done.txt` | same, **after** the records file is in place | pids finished (kept or rejected). **This is the resume set.** |
| `logs/c1b/gen/wave<W>/gpu<k>/stats.jsonl` | same | per chunk: `chunk, n_prompts, left, elapsed_h, decode_s, by_domain{prompts,kept,gen_tokens,blocks,passes,drop:*}` (cumulative for that process run) |
| `logs/c1b/draft/wave<W>/gpu<k>/drafts.<n>.npz` + `done.txt` | `draft_sglang.py` | same pattern: npz via `.tmp.npz` + `os.replace`, then pids appended |
| `logs/c1b/trainset/wave<W>/index.jsonl` (+ `items/<h2>/<pid>.npz`) | `to_trainset.py` | resume set of conversion: pids already in the index are skipped |
| `logs/c1b/wave<W>_state/.done_{gen,draft,trainset,verify,clean}_<d>`, `.done_push_data`, `.done_main`, … | orchestrators | **step-level** flags only. `.done_gen_<d>` means "the gen step returned 0", **not** "every prompt was decoded" (see the `--stop-at` note below) |
| `logs/c1b/wave<W>_state/T_<d>`, `gen_start`, `R_hours` | orchestrators | the frozen time box of each domain (unix seconds) |

If a process dies, the chunk in flight is lost and re-decoded next time: it was never added to `done.txt`. A half-written
records file is only ever a `.tmp.gz`. `gen_sglang.py` wipes its own `gpu<k>/trace/` at start-up, because request ids
equal pids, so a stale trace would mix into the new answers. New record chunks are numbered from
`1 + max(existing n)` (`next_record_index`).

**`--stop-at` does not finish a domain.** After `--stop-at`, `gen_sglang.py` submits no new chunk, finishes the one in
flight, and exits 0. The orchestrator then writes `.done_gen_<d>` even if prompts are left. In wave A this never
happened: every domain finished before its `T_d`, no "`--stop-at reached`" line appears in any `logs/c1b/gen_waveA_*.log`,
and done = list. In wave B it can happen (§4.4), so always check completeness with the command in §2.5.

### 2.4 Wave A boundaries, concretely

* **Prompts:** all of them. The union of `logs/c1b/gen/waveA/gpu*/done.txt` equals the set of `wave=="A"` rows in the
  five lists: 287,222 each, no difference in either direction. The set of pids in the records files is identical too.
  **No wave-A prompt remains.** Everything left is wave B.
* **Record chunks per GPU:** every GPU wrote `records.00000`–`records.00014` (15 files). Each chunk holds exactly one
  domain, identical on all 8 GPUs:

  | domain | chunks (every gpu) | prompts per gpu (gpu0..7) |
  |---|---|---|
  | maths | 00000–00003 | 12,417 / 12,418 / 12,379 / 12,307 / 12,139 / 12,327 / 12,315 / 12,470 |
  | science | 00004–00005 | 4,285 / 4,194 / 4,258 / 4,365 / 4,298 / 4,317 / 4,211 / 4,414 |
  | code | 00006–00009 | 11,352 / 11,401 / 11,446 / 11,366 / 11,350 / 11,264 / 11,418 / 11,389 |
  | if | 00010–00011 | 2,490 / 2,442 / 2,500 / 2,471 / 2,437 / 2,466 / 2,476 / 2,512 |
  | chat | 00012–00014 | 5,406 / 5,502 / 5,331 / 5,475 / 5,446 / 5,283 / 5,401 / 5,484 |

* **Drafts:** `logs/c1b/draft/waveA/gpu<k>/drafts.00000`–`drafts.00012.npz` (13 per GPU, 2,000 items per chunk,
  domains mixed).

  | | gpu0 | gpu1 | gpu2 | gpu3 | gpu4 | gpu5 | gpu6 | gpu7 |
  |---|---:|---:|---:|---:|---:|---:|---:|---:|
  | done.txt lines | 20,588 | 20,724 | 20,719 | 20,714 | 20,769 | 20,766 | 20,949 | 20,760 |

  Total 165,989 = kept.
* **Training set:** `logs/c1b/trainset/waveA/index.jsonl` holds 165,438 lines, with `idx` 0..165437 and the domains
  appended in the order maths → science → code → if → chat. Items live in `items/<h2>/<pid>.npz` (256 folders), and
  `items_tar/<h2>.tar` + `<h2>.count` hold the Hub packing (`hf_sync.pack_items`).
* **Orchestrator state:** `logs/c1b/waveA_state/` has every `.done_{gen,draft,trainset,verify,clean}_<d>` for all five
  domains, plus `.done_push_data`, `.done_main` and `.done_evalA`.
* **Wave B state:** none. `logs/c1b/waveB_state`, `logs/c1b/gen/waveB`, `logs/c1b/draft/waveB` and
  `logs/c1b/trainset/waveB` do not exist.

### 2.5 Listing what is done or remaining (read-only one-liners)

```bash
cd /workspace/dmge
# done pids of a wave (all domains); per domain: grep '^maths-' etc.
cat logs/c1b/gen/waveA/gpu*/done.txt | sort -u > /tmp/doneA.txt; wc -l /tmp/doneA.txt          # 287222
grep -c '^maths-' /tmp/doneA.txt                                                                  # 98772
# kept / reasons of one domain from the records
python3 - <<'EOF'
import glob, gzip, json, collections
c = collections.Counter()
for f in glob.glob("logs/c1b/gen/waveA/gpu*/records.*.jsonl.gz"):
    for l in gzip.open(f, "rt"):
        r = json.loads(l)
        if r["domain"] == "maths": c["kept" if r["kept"] else "drop:" + r["reason"]] += 1
print(c)
EOF
# REMAINING prompts of a wave/domain (use after every wave-B gen step; 0 = complete)
W=B; d=maths; python3 - "$W" "$d" <<'EOF'
import glob, json, sys
W, d = sys.argv[1], sys.argv[2]
done = set()
for f in glob.glob(f"logs/c1b/gen/wave{W}/gpu*/done.txt"): done |= set(open(f).read().split())
left = sum(1 for l in open(f"data/c1b/prompt_lists/{d}.jsonl") if (r := json.loads(l))["wave"] == W and r["pid"] not in done)
print(f"wave {W} {d}: {left} prompts not generated")
EOF
```

(The code list is 3.2 GB. A full JSON scan of all five lists takes a few minutes of one core.)

---

## 3. How much remains (wave B)

### 3.1 Prompts

1,952,150 prompts remain: maths 670,037, science 232,900, code 619,012, if 133,511, chat 296,690. Per source see §1.1,
per GPU see §2.2.

### 3.2 Expected output (inferred: wave A's per-source rates applied to wave B's per-source counts)

Method, per source *s*:

* kept(B) = B_s × kept_s / gen_s;
* items(B) = B_s × items_s / gen_s (this carries the late-decontam exclusions);
* tokens(B) = B_s × tokens_s / gen_s.

Worked example, maths/omi2math: 228,380 × (14,822 / 33,620) = 100,686 kept, and
228,380 × (8,595,923 / 33,620) = 58.39M tokens.

| domain | B prompts | exp. kept | exp. items | exp. corpus tokens | naive check (A tokens × B/A) | kept blocks to draft |
|---|---:|---:|---:|---:|---:|---:|
| maths | 670,037 | 301,810 | 298,080 | 147.2M | 21.67M × 6.784 = 147.0M | 4,132,259 |
| science | 232,900 | 47,038 | 47,038 | 16.0M | 2.36M × 6.782 = 16.0M | 237,470 |
| code | 619,012 | 452,832 | 452,832 | 231.1M | 33.96M × 6.803 = 231.1M | 1,689,908 |
| if | 133,511 | 59,727 | 59,727 | 18.8M | 2.79M × 6.745 = 18.8M | 458,979 |
| chat | 296,690 | 268,664 | 268,644 | 95.8M | 13.98M × 6.848 = 95.7M | 2,444,954 |
| **total** | **1,952,150** | **1,130,072** | **1,126,321** | **508.9M** | 74.76M × 6.797 = 508.1M | **8,963,570** |

Largest sources by expected tokens:

| maths omi2math | maths numina | maths deepmath | science osr2 | code oci | if ifcomp | chat ultrachat | chat wildchat |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 58.39M | 38.92M | 23.24M | 10.32M | 229.94M | 18.49M | 52.04M | 42.84M |

Code/oci alone is 45% of wave B's tokens. Wave B's maths list carries 8,214 decontam-flagged pids. They will be
generated (about 1.2% of maths prompts) but excluded at conversion, which the projection already accounts for.

Tokens the engine processes for wave B, prompt plus generated, are about 1,040M (same method, using
`gen_tok + prompt_tok`). Local disk needed at about 6.8× wave A is roughly 0.8 GB of records, 5.2 GB of drafts and
about 14 GB of items plus tars (inferred from wave A: `gen/waveA` 119 MB, `draft/waveA` 762 MB, `trainset/waveA`
2.2 GB). Free disk now is 722 GB.

### 3.3 Time on 8x RTX PRO 6000 Blackwell (inferred from wave A wall clock)

Wall-clock times come from `logs/c1b/waveA.log`, UTC, 2026-09-28. Settings were `--max-running 128 --mem-fraction 0.85`,
FlashInfer with CUDA graphs (the CUDA 12.9 path), and 6 check workers per GPU. Per-GPU times ("done in X h") are in
`logs/c1b/gen_waveA_<d>.gpu*.log`, and GPU decode time is the sum of `decode_s` in `stats.jsonl`.

| domain | wave A gen wall | per-GPU "done in" | Σ decode_s (GPU-h) | prompts/h (8 GPUs) | generated tok/s/GPU | (gen+prompt) tok/s/GPU | wave B gen (A wall × B/A) |
|---|---|---|---:|---:|---:|---:|---:|
| maths | 15:24:05→17:36:01 = 2.199 h | 2.11–2.19 h | 17.21 | 44,919 | 865 | 1,017 | **14.92 h** |
| science | 18:06:46→18:32:50 = 0.434 h | 0.41–0.43 h | 3.32 | 79,048 | 652 | 1,264 | **2.95 h** |
| code | 18:34:30→19:09:31 = 0.584 h (a) | n/a (log overwritten) | 4.44 | 155,902 | 549 | 2,892 | **3.97 h** |
| if | 19:22:25→19:38:32 = 0.269 h | 0.25–0.26 h | 2.03 | 73,690 | 612 | 903 | **1.81 h** |
| chat | 19:40:38→20:21:10 = 0.676 h | 0.64–0.67 h | 5.20 | 64,137 | 653 | 884 | **4.63 h** |
| **total** | **4.16 h** | | 32.2 | 69,000 | 748 | | **28.3 h** |

(a) Code's first run ended with one per-GPU process `Killed` at 19:09:31 UTC. All code chunks had already been written:
the relaunch printed "0 prompts to decode (28054 already done)". So 0.584 h is the decode wall. The cause of the kill
is not in the logs, because the relaunch overwrote them (inferred: a hang or kill at engine shutdown). A further 4 min 40 s
was lost before the relaunch.

Maths dominates because 13% of its answers hit the 1,024 cap (no_eos) and its answers are long: 555 generated tokens
per prompt on average.

**Drafting** (SDAR-1.7B, `--max-running 256 --mem-fraction 0.70`). Blocks/s per GPU are from the last line of each
`logs/c1b/draft_waveA_<d>.gpu*.log`:

| | maths | science | code | if | chat |
|---|---:|---:|---:|---:|---:|
| blocks/s per GPU | ≈84 | ≈80 | ≈69.5 | ≈91 | ≈94 |

Wave B time = kept blocks / (8 × rate):

| | maths | science | code | if | chat | total |
|---|---:|---:|---:|---:|---:|---:|
| wave B draft time | 1.71 h | 0.10 h | 0.84 h | 0.18 h | 0.90 h | **≈3.7 h** |

Add about 1 to 2 min of engine load per step. Cross-check against wave A draft wall clock: code 19:14:22→19:22:25 =
8.0 min, and 8.0 × 6.8 = 55 min, consistent with the 0.84 h above.

**Conversion + verify + clean** (`to_trainset.py`, one CPU process). Wave A wall clock:

| | maths | science | code | if | chat | total |
|---|---:|---:|---:|---:|---:|---:|
| wave A wall | 5.6 min | 0.5 min | 3.2 min | 0.7 min | 3.9 min | 13.9 min for 165,989 records |

That is about 12K records/min, so wave B's 1.13M kept records take **about 1.6 h** (inferred). It may run somewhat
longer, because each run re-reads the whole `index.jsonl` and every drafts npz header (`load_drafts`), and both grow.

**Total wave B data pipeline:**

* **Run sequentially**, as `run_wave_b.sh` does (it does not overlap post-processing with the next generation):
  28.3 + 3.7 + 1.6 + loads ≈ **34 h**.
* **Overlapped** as in wave A (conversion of *d* while *d+1* generates): about 32 h.

For reference, `run_wave_b.sh`'s header says "~74 h" for wave B **including** training. The training rate measured on
this box is 5.2 to 5.7M corpus tokens/h (`logs/c1b/HANDOFF.md` §3). So training on 509M tokens alone would take about
90 to 98 h (inferred), and the 50.4 h `R` that `run_wave_b.sh` derives is too small (§4.4).

### 3.4 Progress toward the 1B-token plan

* `docs/CORPUS_1B_PLAN_2026-09-28.md` is **not in the repo or on this box** (git history has no such file). It is
  referenced by `common.py`, `sources.py` and `train_c1b.py`, and `CORPUS_1B_BUILD.md` says it also lives in Notion as
  "Phase 1: 1B-token training corpus for the judge". The plan numbers below come from the code that encodes it:
  * `common.WAVE_A_FRACTION = 0.128` ("128M of 1B tokens");
  * `run_wave_b.sh`: "generate the other 872M tokens";
  * `train_c1b.py --total-tokens 1e9`, milestones `1.84e6,8e6,32e6,128e6,512e6,1e9`;
  * `common.TARGET_PROMPTS`, 3,341,000 prompts across both waves.
* Wave A delivered 74.76M of the planned 128M (58.4%). Wave B is the same mix with 6.8× the prompts, so it is expected
  to deliver 508.9M of the planned 872M (58.4%).
* **Wave A + B ≈ 583.7M corpus tokens = 58% of 1B, a shortfall of about 416M.** Three things cause it:
  1. the lists hold 2.24M prompts, not 3.34M (IF and science far short, §1.1);
  2. keep rates are low in science (20%), IF (45%) and maths (45%);
  3. tokens per kept item are lower than the plan assumed (inferred from the ratio; the plan text is not available to
     check).
* The `512e6` milestone is reachable; 1e9 is not, without new prompts. Getting there needs new sources or relaxed
  quotas in `sources.py` and a list rebuild. New rows get waves by the same hash, so do not reuse `wave` for new
  batches; an extra wave label, "C", would need code changes in the `choices=("A","B")` of gen, draft and to_trainset.

---

## 4. Step by step: continuing on this box or a new one

### 4.0 Environment facts (this box)

* **Hardware:** 8x RTX PRO 6000 Blackwell (sm_120, 96 GB).
* **Venvs:**
  * `/venv/train`: torch 2.11.0 cu128. Used for the trainer, `to_trainset`, `hf_sync` and the verify scripts.
  * `/venv/sgl`: `sglang 0.5.10.post1`, torch 2.9.1. Used for `gen_sglang.py` and `draft_sglang.py`.
* **SGLang plugin:** `code/corpus1b/sglang_plugin/install.py` copies `c1b_traced.py` (LowConfidenceTraced) and
  `c1b_static_draft.py` (StaticDraftTraced) into `site-packages/sglang/srt/dllm/algorithm/`. It also applies the single
  approved SGLang edit, `patch_fused_rope`: in `sglang/jit_kernel/rope.py` it casts `out_loc` to `positions.dtype`,
  with a `.orig` backup. It refuses any SGLang version other than 0.5.10.post1. Check it with
  `/venv/sgl/bin/python code/corpus1b/sglang_plugin/install.py --check`.
* **CUDA:** `/usr/local/cuda` is 12.8. Blackwell sm_120 needs a CUDA ≥ 12.9 toolkit for FlashInfer's JIT
  (`logs/c1b/setup.log`: "SM 12.x requires CUDA >= 12.9"), so a 12.9 toolkit sits beside it at `/usr/local/cuda-12.9`.
  * The installed apt packages are `cuda-nvcc-12-9 12.9.86-1`, `cuda-cudart-dev-12-9 12.9.79-1` and
    `cuda-toolkit-12-9-config-common`. `setup_box.sh` does **not** install them; on a new box run
    `apt-get install -y cuda-nvcc-12-9 cuda-cudart-dev-12-9`. That is toolkit only; never `cuda` or a driver package.
  * When `/usr/local/cuda-12.9/bin/nvcc` exists, the wave scripts run SGLang with
    `env CUDA_HOME=/usr/local/cuda-12.9 PATH=/usr/local/cuda-12.9/bin:$PATH` and an empty `SGL_FLAGS` (FlashInfer + CUDA
    graphs).
  * Without it they fall back to `SGL_FLAGS="--attention-backend triton --disable-cuda-graph"`, at about half the decode
    speed (`run_wave_a.sh` comment). The 3.3 time estimates assume the 12.9 path.
  * The stage-1 smoke tests passed on both paths (`logs/c1b/stage1/report.jsonl`): `smoke_gen_gsm8k500` acc 91.0 (ref
    90.2), and `smoke_gen_gsm8k500_flashinfer_cudagraph` acc 90.4, ppb 10.015 (ref 10.06).
* **FlashInfer workspace for CUDA-graph capture:** `SGLANG_FLASHINFER_WORKSPACE_SIZE=2147483648` for gen (128 running)
  and `=4294967296` for drafts (`run_wave_a.sh` `GEN_ENV` / `DRAFT_ENV`).
* **Paths:**
  * Models: `hf_cache/hub` (`research_protocol.MODELS[...]["cache_dir"]`, pinned revisions `b32` c351bbc…, `d32`
    2c83c05…; 61 GB).
  * Datasets: `HF_HOME=$ROOT/hf_home` (202 GB). Wave B does **not** need the source datasets, because the prompt lists
    are final.
* **Secrets:** `.env` holds `HF_TOKEN`, `GITHUB_TOKEN`, `BACKUP_TO_HF`, `HF_PUSH_REPO`, `HF_CKPT_REPO`. Load it with
  `set -a; . ./.env; set +a` and never print it.

### 4.1 A new box: setup

```bash
# clone as in docs/INSTANCE_PROMPT.md steps 1-3 (token only via the credential helper), then:
cd /workspace/dmge && cp ~/.env .env && git check-ignore .env
apt-get install -y cuda-nvcc-12-9 cuda-cudart-dev-12-9          # toolkit beside 12.8 (see 4.0); never a driver
SKIP_CHECKS=1 bash code/corpus1b/setup_box.sh 2>&1 | tee -a logs/c1b/setup.log
```

`setup_box.sh` steps, each leaving `logs/c1b/stage1/.done_<step>`:

* `env`: builds the two venvs and runs `sglang_plugin/install.py`.
* `models`: `snapshot_download` of both pinned models into `hf_cache/hub`.
* `data`: downloads the source datasets, about 200 GB (see the note below).
* `prompts`: rebuilds the lists (**do not let it do this**, see below).
* `evalsets`.
* The GPU checks. `SKIP_CHECKS=1` skips them, and `run_wave_b.sh` does not require them.

Wave B needs the wave-A **lists**, not a rebuild. Before running `setup_box.sh`, restore
`data/c1b/prompt_lists/*.jsonl` (§4.2) and `touch logs/c1b/stage1/.done_prompts logs/c1b/stage1/.done_prompts_{chat,maths,science,code,if}`
so the prompts step is skipped. Also `touch logs/c1b/stage1/.done_data` unless you need the source datasets:
`postdecontam.py` needs the MATH test set only if the `postdecontam_*.json` files are missing (§5).
`run_wave_b.sh` itself only waits for `logs/c1b/stage1/.done_prompts_<d>` for each domain.

### 4.2 A new box: restoring wave-A artefacts from Hugging Face

`hf_sync.py data` uploads with `folder_path=ROOT`, so each file sits in `Arushhh/c1b-judge-corpus-waveA` (a dataset)
under its **repo-relative path**, as listed in `hf_sync.DATA_ALLOW`:

* `logs/c1b/gen/wave*/gpu*/{records.*.jsonl.gz,done.txt,stats.jsonl}`
* `logs/c1b/draft/wave*/gpu*/{drafts.*.npz,done.txt}`
* `logs/c1b/trainset/wave*/{index.jsonl,build_summary.json,items_tar/*.tar}`
* `data/c1b/prompt_lists/{*.jsonl,manifest.json,spotcheck/**}`
* `data/c1b/eval/**`
* `logs/c1b/stage1/*`, `logs/c1b/waveA_state/*.jsonl`, trainer logs/probes/configs, eval summaries, `logs/c1b/*.log`

**Not uploaded** (not matched by `DATA_ALLOW`):

* `data/c1b/prompt_lists/postdecontam_*.json`
* `logs/c1b/trainset/wave*/items/` (only the tars go up)
* `logs/c1b/waveA_state/.done_*`, `T_*`, `R_hours`
* traces, `.env`, GPQA records

Checkpoints are in the model repo `Arushhh/c1b-judge-lora-sdar30b-b32` under `<run>/<ckpt>/`, for example `main/final/`
(28.50M tokens) and `main/tok_000008000000/`; the list is in `docs/PHASE1_HF_ARTIFACTS.md`.

A sketch follows. It is **not run on this box**; `huggingface_hub` `snapshot_download` with `allow_patterns` is a
standard API.

```bash
cd /workspace/dmge; set -a; . ./.env; set +a
HF_HUB_OFFLINE=0 /venv/train/bin/python - <<'EOF'
from huggingface_hub import snapshot_download
# minimum for wave B generation: the lists (+ manifest)
snapshot_download("Arushhh/c1b-judge-corpus-waveA", repo_type="dataset", local_dir=".",
                  allow_patterns=["data/c1b/prompt_lists/*", "data/c1b/eval/**"])
# optional: wave A itself (records, drafts, index, item tars) -- needed to train on wave A again
snapshot_download("Arushhh/c1b-judge-corpus-waveA", repo_type="dataset", local_dir=".",
                  allow_patterns=["logs/c1b/gen/waveA/**", "logs/c1b/draft/waveA/**", "logs/c1b/trainset/waveA/**"])
# the checkpoint wave B continues from
snapshot_download("Arushhh/c1b-judge-lora-sdar30b-b32", local_dir="logs/c1b/train", allow_patterns=["main/final/*", "main/config.json"])
EOF
# item tars -> items/ (arcname is items/<h2>/<pid>.npz)
for t in logs/c1b/trainset/waveA/items_tar/*.tar; do tar -xf "$t" -C logs/c1b/trainset/waveA; done
ls logs/c1b/trainset/waveA/items | wc -l          # 256
# verify the lists are the wave-A ones (sha256 must equal list_sha256 in postdecontam_<d>.json, see 1.1):
sha256sum data/c1b/prompt_lists/{maths,science,code,if,chat}.jsonl
#   maths   dc4a63ffbad0251e5214be466697315804e1d72a05b1172b430afc545e1fec95
#   science b1eee9e86cbef0167672c85b2676b77652edd0e556004fb15d3a453c955ed715
#   code    211842669f44988f9127e204f5184fe1e66d12fa0341a4d61a1411e28de68270
#   if      18a293f4b6598dbca1205dde08297c8e8a40fad4058fc368a873e295bb9d6167
#   chat    c39cbc6c8fa5e1a519d59df317f73b47cf2610f58513b578fd5d3b4e20295190
```

Copy `data/c1b/prompt_lists/postdecontam_*.json` (5 files, 0.5 MB) off this box **before it dies**, for example to the
dataset repo or into git. Without them, `to_trainset.py` recomputes the flags, which needs the `math_test` eval set in
`HF_HOME` (`postdecontam._index` asserts that it loaded).

### 4.3 Running wave B manually, per domain (recommended)

Manual runs give the same resume semantics as the script without its wave-B defects (§4.4). Run from the repo root
inside tmux. Every command is resumable: rerun it after any failure and it skips the done pids.

```bash
cd /workspace/dmge; set -a; . ./.env; set +a
export HF_HOME=$PWD/hf_home TOKENIZERS_PARALLELISM=false
C129="CUDA_HOME=/usr/local/cuda-12.9 PATH=/usr/local/cuda-12.9/bin:$PATH"
GEN_ENV="env $C129 SGLANG_FLASHINFER_WORKSPACE_SIZE=2147483648"
DRAFT_ENV="env $C129 SGLANG_FLASHINFER_WORKSPACE_SIZE=4294967296"
PYS=/venv/sgl/bin/python; PYT=/venv/train/bin/python
mkdir -p logs/c1b/gen/waveB logs/c1b/draft/waveB

per_gpu(){  # name, command with {K}; relaunch a failed GPU once after 30 s (the wave-A EADDRINUSE fix); logs APPEND
  local name=$1; shift; local pids=()
  for k in 0 1 2 3 4 5 6 7; do
    ( CUDA_VISIBLE_DEVICES=$k bash -c "${*//\{K\}/$k}" >> logs/c1b/$name.gpu$k.log 2>&1 \
      || { sleep 30; CUDA_VISIBLE_DEVICES=$k bash -c "${*//\{K\}/$k}" >> logs/c1b/$name.gpu$k.log 2>&1; } ) & pids+=($!)
  done
  local rc=0; for p in "${pids[@]}"; do wait $p || rc=1; done; return $rc
}

for d in maths science code if chat; do
  # 1) generation (defaults = wave A: --max-running 128 --mem-fraction 0.85 --chunk 4096 --check-workers 6).
  #    Optional time box: add --stop-at <unix time>; the domain is then NOT complete (check with the 2.5 snippet).
  per_gpu gen_waveB_$d "$GEN_ENV $PYS code/corpus1b/gen_sglang.py --wave B --domains $d --gpu {K} --num-gpus 8"
  # 2) drafts (the wave-A flags; defaults 512 / 0.80 OOM)
  per_gpu draft_waveB_$d "$DRAFT_ENV $PYS code/corpus1b/draft_sglang.py --wave B --gpu {K} --num-gpus 8 --max-running 256 --mem-fraction 0.70"
  # 3) training items for this domain (appends to logs/c1b/trainset/waveB/index.jsonl; one at a time)
  $PYT code/corpus1b/to_trainset.py --wave B --domains $d > logs/c1b/trainset_waveB_$d.log 2>&1
done
```

After each domain, verify it with the same rule as the scripts: index entries ≥ 95% of the domain's kept records, and
`no_draft == 0` in this run's `build_summary.json`. Also check completeness with the §2.5 snippet (`W=B d=<domain>` → 0).

```bash
d=maths; $PYT - "$d" <<'EOF'
import glob, json, sys
sys.path.insert(0, "code/corpus1b"); import common as C
d = sys.argv[1]
kept = sum(1 for f in glob.glob("logs/c1b/gen/waveB/gpu*/records.*.jsonl.gz") for r in C.read_jsonl(f) if r.get("kept") and r.get("domain") == d)
idx = sum(1 for e in C.read_jsonl("logs/c1b/trainset/waveB/index.jsonl") if e.get("domain") == d)
bs = json.load(open("logs/c1b/trainset/waveB/build_summary.json"))["counts"]
print(json.dumps({"domain": d, "kept_records": kept, "index_entries": idx, "no_draft": bs["no_draft"],
                  "excluded_late_decontam": bs.get("excluded_late_decontam", 0), "ok": kept > 0 and idx >= 0.95 * kept and bs["no_draft"] == 0}))
EOF
```

Expected `excluded_late_decontam` for wave B maths: a few thousand (8,214 flagged prompts × about 45% keep, inferred).
Wave A maths had 548 excluded out of 44,416 kept (98.8% converted), so the 95% rule still holds.

**Overlap (optional, as in wave A):** run step 3 for domain *d* in the background while step 1 runs for *d+1*. Never
run two `to_trainset.py` at once, because they append to one index. Run the draft step only between generation steps:
it and gen both need all 8 GPUs.

**Monitoring:** watch these, all in UTC:

* `tail -1 logs/c1b/gen/waveB/gpu*/stats.jsonl` (the `left` field and `elapsed_h`);
* the `[gen] chunk N written … | maths:kept/prompts ppb=… N/h` lines in `logs/c1b/gen_waveB_<d>.gpu*.log`;
* the `[draft] gpu k: … blocks/s` lines.

Compare them with the per-domain rates in §3.3. Keep free disk above 50 GB (`CORPUS_1B_BUILD.md`). The generation
traces (`logs/c1b/gen/waveB/gpu*/trace`) can be deleted only while no gen runs.

### 4.4 Using `run_wave_b.sh` instead: what to know or fix first

The one-shot orchestrator is

```bash
tmux new -d -s wb 'FORCE=1 DEADLINE_FILE=/workspace/deadline_b.env R_HOURS=<h> bash code/corpus1b/run_wave_b.sh 2>&1 | tee -a logs/c1b/waveB.log'
```

Knobs: `SGL_FLAGS`, `DEADLINE_FILE` (a file setting `START END DEADLINE` in unix seconds; default
`/workspace/deadline.env`), `R_HOURS`, `CONF`, `FORCE`. It resumes through `logs/c1b/waveB_state/.done_*`. Issues found
in the code:

1. **Drafts run with the settings that ran out of memory.** `run_wave_b.sh` calls `draft_sglang.py … $SGL_FLAGS` with
   no `DRAFT_FLAGS`, so the defaults `--max-running 512 --mem-fraction 0.80` apply. In wave A exactly those settings
   OOMed on maths: `torch.OutOfMemoryError`, "tried to allocate 9.13 GiB, GPU 0 had 1.95 GiB free"
   (`logs/c1b/monitor.log` 17:38 UTC). The comment in `run_wave_a.sh` explains it: "512 running x 32 slots x
   152K-vocab fp32 logits (~10 GB per forward) OOMs at mem-fraction 0.80 on 96 GB".
   * Fix: port `DRAFT_FLAGS=${DRAFT_FLAGS:---max-running 256 --mem-fraction 0.70}` and add `$DRAFT_FLAGS` to the draft
     call.
   * Do not put these flags in `SGL_FLAGS`: that would also lower generation's `--max-running`/`--mem-fraction`.
2. **The time box would cut maths off.** `T_d = GEN_START + (GEN_END − GEN_START) × cumulative share`, with shares
   maths 0.25, science 0.15, code 0.25, if 0.15, chat 0.20. Wave B maths needs about 14.9 h of the about 28.3 h total
   (53%). To finish maths by its box, the window must satisfy 0.25 × W ≥ 14.9 h, so **W ≥ about 60 h**. With a shorter
   window, maths stops at 0.25 × W, `.done_gen_maths` is still written, and the maths remainder is never generated.
   * Check with §2.5, then either rerun those prompts manually (§4.3 step 1) or change `SHARE` to the measured split
     (maths 0.53, science 0.10, code 0.14, if 0.06, chat 0.17) before the first run. `T_<d>` is frozen once written.
3. **`R_HOURS` default.** Without `R_HOURS` it computes `1 + (8.25 − 1) × 872/128 = 50.4 h` from
   `logs/c1b/waveA_state/R_hours` (8.25), and puts `GEN_END` at `DEADLINE − 50.4 h`. That 8.25 came from a 1.5M-token
   benchmark (`R_hours.note`: "~305 corpus tok/s"). The steady-state training rate later measured was 5.2 to 5.7M
   tok/h (`logs/c1b/HANDOFF.md`), which would make 509M tokens take about 90 to 98 h (inferred). Set `R_HOURS`
   explicitly: `R_HOURS=1` if training is launched separately.
4. **The gate check passes only by accident.** It is `grep -q "CONTINUE" logs/c1b/gate_A.md`. The verdict in
   `gate_A.md` line 17 is "**Verdict: STOP**", but line 29 contains "would read **CONTINUE** on these probes", so the
   grep matches. Pass `FORCE=1` explicitly so the intent is recorded, and do not rely on the grep.
5. Smaller issues:
   * its `per_gpu` has **no relaunch-once** (the wave-A fix feaf04d for SGLang's random internal port being taken,
     EADDRINUSE);
   * its logs are opened with `>` (a rerun overwrites the failure evidence, which is how the wave-A code failure lost
     its cause);
   * `to_trainset.py --wave B` runs without `--domains` (harmless: it converts every kept, drafted record);
   * no conversion is overlapped with the next generation, so GPUs idle during to_trainset/verify/clean;
   * after the data loop it pushes to HF (`backup_hf.sh data`) and then **starts training automatically** (§6).
   To keep the data pipeline separate, stop the script after `push_data` (touch `logs/c1b/waveB_state/.done_train`
   first only if you mean to skip training).

### 4.5 Backups to Hugging Face

`code/corpus1b/backup_hf.sh`, which acts only when `.env` has `BACKUP_TO_HF=1`, `HF_PUSH_REPO` and `HF_CKPT_REPO`:

```bash
bash code/corpus1b/backup_hf.sh data                                    # corpus once (DATA_TIMEOUT default 1500 s)
bash code/corpus1b/backup_hf.sh ckpt                                    # checkpoints + verify + prune + links doc
nohup bash code/corpus1b/backup_hf.sh loop > logs/c1b/backup_hf.log 2>&1 &      # every BACKUP_EVERY (1800 s)
nohup bash code/corpus1b/backup_hf.sh ckptloop > logs/c1b/backup_ckptloop.log 2>&1 &   # checkpoints every CKPT_EVERY (300 s)
```

* `hf_sync.py data` picks up `wave*` paths, so **wave B goes to the same `HF_PUSH_REPO`**, currently
  `Arushhh/c1b-judge-corpus-waveA`. That is fine, since paths carry `waveB`, but the repo name is misleading. Set
  `HF_PUSH_REPO` to a new repo if you want a separate wave-B dataset. This is a decision for the user.
* It packs `items/<h2>/` into `items_tar/<h2>.tar` (rebuilt when the file count changes). Files that are still growing
  (`index.jsonl`, logs, `done.txt`, `stats.jsonl`) are uploaded from snapshot copies in `/dev/shm/c1b_hf_stage`.
* It scans every text file for tokens first; a hit aborts the upload (§5).
* It takes an exclusive lock (`logs/c1b/hf_sync/hub.lock`), so both loops can run at once.
* Per `code/corpus1b/hf_sync.py` and the memory note on the HF policy: public repos (account Arushhh); upload the data
  before training; keep the 5 newest local checkpoints (`KEEP = 5`), deleting only after a verified upload.

---

## 5. Known pitfalls and how they were handled

| symptom | where seen | cause | fix / handling |
|---|---|---|---|
| Draft OOM: `torch.OutOfMemoryError`, 9.13 GiB alloc | maths drafts, 2026-09-28 17:36 UTC (`monitor.log`) | 512 running × 32 slots × 152K-vocab fp32 logits ≈ 10 GB per forward at `--mem-fraction 0.80` | `DRAFT_FLAGS --max-running 256 --mem-fraction 0.70` in `run_wave_a.sh` (commit 1d80e6f "drafts at 256 running"); rerun resumed from `done.txt`. **Not ported to `run_wave_b.sh`** (§4.4). |
| A per-GPU SGLang process fails at start | `run_wave_a.sh` comment | SGLang's random internal port still taken (EADDRINUSE) after the previous engine | `per_gpu` relaunches a failed GPU once after 30 s (commit feaf04d). Not in `run_wave_b.sh`. |
| A gen process `Killed` after finishing all its chunks | code gen, 19:09:31 UTC (`waveA.log` line 74) | unknown (log overwritten by the relaunch; inferred: shutdown hang or kill) | rerun printed "0 prompts to decode"; no data lost. Append logs (`>>`) to keep evidence. |
| All 8 draft processes `Killed` | `waveA.log` lines 40–47 (the OOM episode) | the orchestrator's failure path | rerun resumes |
| SGLang fails on sm_120 | stage 1 (`setup.log`: "SM 12.x requires CUDA >= 12.9") | FlashInfer's JIT needs CUDA ≥ 12.9 on Blackwell; dLLM + CUDA graphs forces FlashInfer | CUDA 12.9 nvcc/cudart beside 12.8 plus `CUDA_HOME`/`PATH` for SGLang only, and larger `SGLANG_FLASHINFER_WORKSPACE_SIZE`; fallback `--attention-backend triton --disable-cuda-graph` (about half speed) |
| triton backend: fused RoPE rejects int64/int32 index mix | stage 1 | dLLM + triton pass int64 positions with int32 cache locations | the approved one-line cast in `sglang/jit_kernel/rope.py` (`install.py::patch_fused_rope`, `.orig` kept) |
| **Secret scan hit**, data upload blocked | 2026-09-28 16:04 UTC, `data/c1b/prompt_lists/chat.jsonl` | third-party credential-like strings (tokens users pasted into WildChat etc.) in prompts | `sanitize_lists.py` (same regex in `build_prompt_lists.py` from commit 62b3ad4) removed 5 chat + 2 code rows; `hf_sync.scan_for_secrets` aborts any upload on a hit (env token values + `hf_…`, `ghp_…`, `github_pat_…` patterns). Never weaken the scan. |
| Late decontamination | `postdecontam.py` docstring | `decontam._configs` did not expand `ALL_7_SUBJECTS` until 15:16 UTC, so the lists were built without the MATH-test set | lists screened afterwards; `postdecontam_<d>.json` caches flags keyed by list sha256; `to_trainset.py` drops flagged pids (wave A: 551 items). Keep those JSONs with the lists (§4.2). |
| HF upload stuck forever: "LFS pointer pointed to a file that does not exist" | 19:24–19:37 UTC (`monitor.log`) | `index.jsonl` grew after `upload_large_folder` hashed it; huggingface_hub 0.36 re-queues the same stale commit forever and blocks the backup loop | commit 80dc3f0: growing files go up from snapshot copies in one commit (`APPENDED` in `hf_sync.py`); the stuck process was SIGINT/SIGTERMed. |
| HF `429 Too Many Requests` on commit | same round (`monitor.log`, 550+ lines) | too many commits from the retry storm | capped, slower data rounds (80dc3f0, `DATA_TIMEOUT` 1500 s); one Hub writer at a time with a file lock (f2f8ceb); loops retry next round. |
| Code prompt list build failed once | `waveA.log` 15:09 UTC ("prompt list code FAILED … rebuilding") | see `logs/c1b/stage1/prompt_lists_code.log` | rebuilt; `run_wave_a.sh` never lets a failed list block other domains. Not relevant to wave B unless you rebuild lists (do not). |
| `.done_gen_<d>` written although prompts remain | design (`--stop-at`) | time box reached | always run the §2.5 remaining check; rerun gen without `--stop-at` |
| Stale trace mixes into answers | `gen_sglang.py` | request ids = pids | gen wipes `gpu<k>/trace` at start; never run two gens on one GPU folder |
| `pkill -f` / `grep` matches your own shell | `logs/c1b/HANDOFF.md` §6 | pattern in the command line | bracketed patterns (`[g]en_sglang.py`), as the scripts' `adopt()` does |
| `/dev/shm` is noexec; code must never run from the repo root | HANDOFF §6, AGENTS.md 9(p) | | checkers execute in `--sandbox /dev/shm/c1b_sandbox` subprocesses (the default) |

---

## 6. How the wave-B index plugs into training (brief)

`code/corpus1b/train_c1b.py` docstring:

```bash
torchrun --standalone --nproc_per_node=8 code/corpus1b/train_c1b.py \
    --index logs/c1b/trainset/waveB/index.jsonl --out logs/c1b/train/main_b \
    --continue-from logs/c1b/train/main/tok_000128000000 --max-tokens 1e9
```

* `--continue-from` loads the adapter, the optimizer and the token count of an **earlier** run's checkpoint and continues
  the same WSD schedule (`--total-tokens 1e9`, decay over the last `--decay-frac 0.15`). `--resume` continues the *same*
  `--out` from its newest checkpoint.
* **`tok_000128000000` does not exist.** Main stopped at 28.50M tokens. `run_wave_b.sh` falls back to
  `logs/c1b/train/main/final` (step 7418, 28,498,676 tokens, LR 8e-5 from `main/config.json`, index = waveA).
  `logs/c1b/train/lr_choice.txt` does not exist, so the LR stays at 8e-5.
* The wave-A main run later used `--domain-weights` (maths 25 / code 25 / science 15 / IF 15 / chat 20 from 14.47M)
  with `--repeat-domains science,if --max-passes 4` (HANDOFF §3). Pass the same flags to keep the mix.
* Continuing on the wave-B index alone **skips the about 46M wave-A tokens never trained on** (seen 28.50M of 74.36M
  train-split tokens, `seen_report.md`; per-item seen tokens in `logs/c1b/train/main/seen_items.json`). `train_c1b.py`
  takes one `--index`, so training on both needs a merged index file. That would be a new step: concatenate the two
  `index.jsonl` files with `file` paths made relative to one root, which is **not implemented**. Leave it to the
  training section.
* Rough training time at the measured 5.2 to 5.7M tokens/h: about 90 to 98 h for about 509M wave-B tokens (inferred).
