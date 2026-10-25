# Prompt for Claude Code on the GPU box, Phase 2 MVP (paste everything inside the block as the first message)

```text
=====================================================================================================================
PHASE 2 MVP ON THIS BOX: F2 (CORPUS) + S1 (SIZE) DRAFTER SWAP, ALL SEVEN BENCHMARKS, ZERO-SHOT, FULLY LOGGED
=====================================================================================================================

You are running Phase 2 of our research project on this 8x RTX PRO 6000 Blackwell (96 GB each) Vast.ai box.
Work autonomously. Ask me only when (a) a go/no-go check still fails after two fix attempts, or (b) a rule below
says "stop and tell me". Never print, log or commit a token or any value from ~/.env.
Nothing in this plan may be skipped. If anything cannot run, TELL ME before moving on; never drop a benchmark, an
item, a drafter or a log silently.

---------------------------------------------------------------------------------------------------------------------
PART A. WHAT THIS PROJECT IS (read this first, it explains every number you will see)
---------------------------------------------------------------------------------------------------------------------

A1. The system under test
  - Target model: JetLM/SDAR-30B-A3B-Chat-b32 @ c351bbc, a 30B mixture-of-experts diffusion LM: 48 layers, 128
    experts per layer, 8 used per token (~3B active). It writes text in BLOCKS OF 32 TOKENS. It is FROZEN: never
    trained, never modified.
  - Drafter: a small model that guesses all 32 tokens of a block first. Today: JetLM/SDAR-1.7B-Chat-b32 (called S0).
  - Judge: our LoRA adapter (~27M parameters) on the 30B. It reads the drafter's guesses and, per slot, commits
    (> 95% confident), keeps tentatively, or re-masks, pass after pass, until the block is done.
    The judge used today: r3_B step 710 (weights hash starts ad8a6485).
  - DES48: an expert pool limiting each forward to 48 experts per layer; on in every drafter run.
  - Cost is counted in EXPERTS LOADED PER TOKEN; a drafter is charged its own cost per forward.

A2. The two Phase 2 questions, and this run (the MVP)
  1. Family / corpus: does the TRAINING CORPUS of the drafter change speed and accuracy, domain by domain?
  2. Size: how small can the drafter get and still pay for itself?
  Every drafter is a RELEASED checkpoint used exactly as downloaded: never converted, never trained. The ONLY thing
  that may ever be trained in Phase 2 is the LoRA judge adapter, and NOT in this run.
  | id | model                                           | base / size       | its authors' training corpus              | cost/fwd |
  | S0 | JetLM/SDAR-1.7B-Chat-b32 (baseline)             | Qwen3 1.7B        | in-house web / STEM / code (50B) + 4B SFT | 364.64   |
  | F2 | Salesforce/CoDA-v0-Instruct            (family) | Qwen3 1.7B (= S0) | code-heavy pretraining + code-only SFT    | 364.64   |
  | S1 | dllm-hub/Qwen3-0.6B-diffusion-bd3lm-v0.1 (size) | Qwen3 0.6B        | general SFT: Tulu-3, SmolTalk, OpenCoder  | 126.32   |
  | F1 | Efficient-Large-Model/Fast_dLLM_v2_1.5B         | Qwen2.5 1.5B      | broad synthetic (Nemotron post-training)  | 327.16   |
  F1 is ONLY the fallback: it runs in F2's place if (and only if) F2's re-draft is stopped by the fit rule.
  - S0 -> F2 changes the corpus with the same base, size and tokenizer. CoDA is a full-sequence model with no KV cache
    (every draft step recomputes the context: slower wall clock, same charged cost; predictions shifted), so read F2
    as a PATTERN across domains relative to S0 (code vs maths vs science vs instruction following), not as levels.
  - S0 -> S1 answers "is a small drafter worth it". Best case on HumanEval: 1.39x charged speedup vs S0's 1.24x. It
    saves ~89 expert-equivalents per token of drafter cost, so it wins only if the MoE's experts per token rise by
    less than ~13.6% and accuracy stays inside S0's interval. S1 changes size AND corpus: its result is "this 0.6B
    drafter", not size alone. S1 uses the Qwen3 vocabulary and a block-causal cache like S0 (no shift).
  - F1 wrapper facts (only if it runs): Qwen2.5 vocabulary (mask 151665 remapped; the Qwen3-only tokens <think>,
    </think>, <tool_response>, </tool_response> become a placeholder on input and can never be drafted), shifted.
  - Deliberately NOT in this run: S6 SDAR-4B (dense 4B, more than the MoE's 3B active; best case ~1.01x), S2, and the
    candidates that need a tokenizer bridge or extra wiring. Do not add them.

A3. The benchmarks: ALL SEVEN, for both drafters
  | set             | items | domain vs the judge's training mix | S0 reference (judge + DES48)   | stock reference          |
  | HumanEval       | 164   | in-domain (code)                   | round 3: 75.0                  | round 3                  |
  | MBPP (sanitized)| 257   | in-domain (code)                   | round 3: 61.1                  | round 3                  |
  | GSM8K-500       | 500   | in-domain (maths)                  | round 3: 90.8                  | round 3                  |
  | IFEval-541      | 541   | in-domain (instruction following)  | NONE -> this run decodes S0    | Phase 1 run: 52.7 (setup)|
  | MATH-500        | 500   | out-of-mix, near-domain            | round 3: 69.2                  | round 3                  |
  | AIME            | 90    | out-of-mix, near-domain            | round 3: 7.8                   | round 3                  |
  | GPQA-Diamond    | 198   | out of domain                      | round 3: 28.8                  | round 3                  |
  S0 reference numbers: drafter agreement 55.5%; charged speedup 1.24x on HumanEval (1.50x if the drafter were free);
  probe keep rate 90.5%, fix rate 33.0%, commit precision 99.5%, ECE 0.008.
  IFEval is scored strictly (every constraint must pass; draftalign/if_scorer.py). The stock 30B is NOT decoded in
  this run: its IFEval records come from Phase 1's complete run on Hugging Face (setup step 3c checks them).

A4. The six steps of the Phase 2 plan (this run does 2-5)
  1. Fixed pass/fail and stop rules: code/phase2/rules.json, version phase2-rules-v2 (7 sets; IFEval in-domain,
     AIME near-domain; thresholds unchanged). Its sha256 is in every output. Never edit it.
  2. Plug in the drafter: wrapper, mask remap, own cost, and proof that the 30B still decodes token for token like
     round 3 (setup check_stock: 16/16).
  3. Drafter agreement ("redraft"): the drafter alone re-drafts the round-3 training set (4,444 answers,
     23,558 blocks) with the 30B's real text as context; agreement vs S0 on the same blocks. Below 0.5 x S0 the arm
     STOPS ("the drafter does not fit this MoE") -- a result, not a failure. For F2 this triggers the F1 fallback.
  4. Calibration probe: the unchanged S0 judge on 150 held-out answers filled with the new drafts: keep rate, fix rate,
     commit precision (floor 0.99), ECE; plus which experts the judge routes to.
  5. Zero-shot decode: new drafter + unchanged S0 judge + DES48 on the seven benchmarks, 8 shards each: paired delta
     accuracy vs S0 (95% bootstrap CI), charged speedup, and the full expert / probability logs (PART E3).
  6. Retrain the judge only if step 5 lost in-domain accuracy. NOT in this run (P2_NO_RETRAIN=1): flag it only.

A5. How to read a verdict (logs/phase2/gate/<ARM>_zeroshot/gate.md)
  - alignment_hurt: an in-domain set's delta CI entirely below 0, or probe commit precision < 0.99
  - retrain_needed: alignment hurt on an in-domain set (flag only)
  - cross_domain_gain_sets: MATH-500 / AIME / GPQA with the delta CI entirely above 0 (only GPQA is truly out of domain)
  - size_pays_off_sets: charged speedup above S0's with accuracy held (the S1 question)
  - drafter_fits: re-draft agreement >= 0.5 x S0's
  Break-even (HumanEval): S1 beats S0's 1.24x if experts per token rise by < ~13.6%; F2 only at 0% (same cost);
  F1 at < ~2%.

---------------------------------------------------------------------------------------------------------------------
PART B. SOURCES TO READ BEFORE STARTING (15 minutes, no GPU)
---------------------------------------------------------------------------------------------------------------------
  B1. If a Notion connector is available in this session, fetch the plan page
      "Phase 2: Drafter swap ablations (conversion corpus, size, compute)" (under "Conscious Engines"):
      https://app.notion.com/p/3e93b0ea304c81e7bf5de5cea513e00b
      Read the 6-step plan, "What can fool us", Tables 1 and 2 (family, size), Table 3 (claimed scores), and the
      training-strategy table. The Notion page lists five benchmarks and more drafters; THIS run is the MVP above
      (F2 + S1, seven benchmarks), as written in docs/PHASE2_DRAFTER_SWAP.md. Where they differ, the repo doc wins.
  B2. In the repo (after D2): AGENTS.md sections 1 and 9 (the footguns), docs/PHASE2_DRAFTER_SWAP.md,
      code/phase2/rules.json, and the docstrings of code/phase2/run_all_p2.sh, setup_p2.sh, run_phase2.sh,
      decode_hooks.py, explog.py and expert_report.py.
  B3. Reference data on Hugging Face (setup downloads it; know where it is):
      https://huggingface.co/datasets/Anupam-Rawat-IITB/dlm-exp1-drafter-voting-research-artifacts
        Nemotron-split-moe-train/r3_B/checkpoints/step000710          the S0 judge
        Nemotron-split-moe-train/r3_B/evals/step000710/*/judge_des48  S0 records for six of the seven sets
        Nemotron-split-moe-train/reference_evals/                     stock records, six sets (speedup reference)
        Nemotron-data-subset-training-diffusion-moe/processed/trainset_r3  the round-3 training set (step 3 input)
      https://huggingface.co/datasets/Arushhh/c1b-judge-corpus-waveA  (Phase 1)
        data/c1b/eval/ifeval541.jsonl (+ .sha256)                     the frozen IFEval-541
        logs/c1b/eval/checkpointB/ifeval541_released/                 the stock 30B on IFEval-541 (541/541)
  B4. Send me a short summary of what you understood (questions, arms, fallback, sets, logs, stop conditions, plan
      and ETA) before starting D4.

---------------------------------------------------------------------------------------------------------------------
PART C. HARD RULES
---------------------------------------------------------------------------------------------------------------------
  - Other team members' jobs may run from /workspace/dmge (if it exists on this box). Never stop, pause, kill, renice
    or modify any process you did not start, and never write anything inside /workspace/dmge. You may READ its model
    cache (/workspace/dmge/hf_cache/hub) and data/c1b/eval.
  - Never change code/phase2/rules.json, the decode rule, the models or revisions, the S0 adapter, the drafter list,
    the benchmark list, the item counts, the thresholds or the logging level. Never relax a check to make it pass.
  - No training of any kind in this run.
  - Every GPU job runs in its own /dev/shm sandbox cwd (the scripts do this); never run a code benchmark or scorer from
    the repository root (AGENTS.md 9(p)).
  - Bracketed patterns with pgrep/grep on process lists (run_da_decod[e], redraf[t].py, p2_job[s].py): AGENTS.md 9(d).
  - Do not commit data or logs. Code fixes only, on branch phase2-drafter-swap, with clear commit messages and no AI
    attribution lines.

---------------------------------------------------------------------------------------------------------------------
PART D. PROCEDURE
---------------------------------------------------------------------------------------------------------------------
D1. Secrets and access
      set -a; . ~/.env; set +a            # HF_TOKEN, GITHUB_TOKEN, optionally WANDB_API_KEY
      curl -s -o /dev/null -w "%{http_code}\n" -H "Authorization: token $GITHUB_TOKEN" \
        https://api.github.com/repos/NoviceCoderInfinity/diffusion-moe-general-expert        # must print 200
      python3 -c "from huggingface_hub import HfApi; print(HfApi().whoami()['name'])"    # tell me the account
      The backup goes to <that account>/phase2-drafter-swap (private) unless I give you P2_HF_REPO.
      Check free disk: df -h /workspace  (logs ~35 GB; + ~75 GB of models and drafters on a fresh box; tell me if
      less than 200 GB is free). Check nvidia-smi shows 8 GPUs and tell me what is running on them.
D2. Clone the Phase 2 branch NEXT TO (not into) any existing checkout; no token in the URL or git config:
      git -c credential.helper='!f(){ echo username=x-access-token; echo "password=$GITHUB_TOKEN"; }; f' \
        clone -b phase2-drafter-swap https://github.com/NoviceCoderInfinity/diffusion-moe-general-expert.git \
        /workspace/dmge_p2
      cd /workspace/dmge_p2 && cp ~/.env .env && git check-ignore .env && mkdir -p logs/phase2
      git log --oneline -3      # tell me the commit
      (If /workspace/dmge_p2 already exists from an earlier attempt: git -C /workspace/dmge_p2 pull, keep logs/.)
D3. Read PART B and send me the summary (B4).
D4. Start the whole plan in one tmux session. It runs the CPU/network setup now, WAITS BY ITSELF until all 8 GPUs are
    free, runs the go/no-go checks, then the plan in this order (nothing skipped; every step resumes if restarted):
      0. go/no-go checks                                                         ~0.35 h
      1. F2 re-draft first (its "does it fit" answer early)                      ~0.35 h
      2. S0 judge + DES48 on IFEval-541 (the reference round 3 lacks)            ~0.70 h
      3. F2: probe, 7 sets, report, gate                                         ~3.65 h
         (if step 1 stopped F2 by the fit rule: F1 in full instead, ~3.43 h)
      4. S1: re-draft, probe, 7 sets, report, gate                               ~3.47 h
      5. STATUS.md and final backup                                              ~0.20 h
      total ~8.7 h after the GPUs free up (+/- 20%); add ~1 h of setup first on a fresh box
    Which box are you on?
      - A box where Phase 1 ran (/workspace/dmge/hf_cache/hub exists): link its two stock models read-only instead of
        downloading 60+ GB again:
          tmux new -d -s p2 'cd /workspace/dmge_p2 && export P2_MODELS_FROM=/workspace/dmge/hf_cache/hub && \
            bash code/phase2/run_all_p2.sh 2>&1 | tee -a logs/phase2/run_all.log'
      - A fresh box (no /workspace/dmge): setup builds /venv/p2 from scratch and downloads the models (~65 GB), the
        drafters, the references and the benchmark datasets itself (add ~45-60 min before the GPU work starts):
          tmux new -d -s p2 'cd /workspace/dmge_p2 && bash code/phase2/run_all_p2.sh 2>&1 | tee -a logs/phase2/run_all.log'
    The benchmark datasets always go to the checkout's own cache (/workspace/dmge_p2/hf_home); never point HF_HOME
    into /workspace/dmge.
    Only if I give you a hard end time: add P2_DEADLINE_IST="YYYY-MM-DD HH:MM" (30 min before the box stops) to the
    export line. The launcher then warns when the rest of the plan will not fit, and at that time stops cleanly
    (records kept) and backs up. Tell me at once if a warning appears, with what would not finish.

---------------------------------------------------------------------------------------------------------------------
PART E. WHAT TO CHECK, AND WHEN
---------------------------------------------------------------------------------------------------------------------
E1. Setup, CPU part (logs/phase2/setup/, logs/phase2/run_all.log)
    - markers .done_env, .done_models, .done_drafters, .done_refs, .done_datasets, .done_evalsets, .done_stock_ifeval
      in logs/phase2/setup/
    - "torch ... transformers 4.54.1"; b32 and d32 linked from /workspace/dmge/hf_cache/hub (or downloaded, fresh box)
    - "drafter F2", "drafter S1", "drafter F1" each downloaded at its pinned revision
    - "benchmarks offline: humaneval=164 mbpp=257 gsm8k500=500 math500=500 gpqa=198 aime=90"
    - "reference files for <set>: S0 N, stock N" for humaneval, mbpp, gsm8k500, math500, gpqa, aime (S0 > 0 each)
    - "IFEval-541: 541 items, sha256 verified, scorer imports"
    - "stock IFEval-541: 541/541 items, revision ok, golds match; accuracy 52.7, exp/token 1576.5"
    - no "round-3 index sha256" failure, no "item files ... missing"
E2. Go/no-go GPU checks, after the GPUs free up. ALL must pass, else nothing else runs:
    | check         | where                                  | pass condition                                          |
    | tests         | logs/phase2/setup/tests.log            | all Phase 2 CPU tests pass                               |
    | adapter       | run_all.log                            | S0 judge weights hash starts ad8a6485                    |
    | stock         | logs/phase2/setup/check_stock.log      | "gen_ids match 16/16"                                    |
    | redraft smoke | logs/phase2/setup/check_redraft_*.log  | F2, S1 and F1 draft 16 items well above chance; S0       |
    |               |                                        | reproduces round 3's stored drafts on >= 0.97 of slots   |
E3. The logs this run produces (check they are being written; report anything missing):
    per decoded item, in logs/phase2/decode/<ARM>/<set>/:
      - records (*.shard*of8.jsonl): accuracy, experts per token, charged cost, passes per block, agreement, judge
        commits / tentatives / remasks, and the expert summary exp_* (per forward kind: experts per layer in ONE
        forward, union per item, load entropy, router top-8 mass; per-layer profile; top-10 layer/expert pairs) and
        judge_* (mean top-1 probability and share > 0.95 at each pass)
      - experts/<set>/<idx>.npz: per forward, per layer, which experts ran (bitmask), and every row's top-8 expert ids
        and router probabilities (full level)
      - judge/<set>/<idx>.npz: every judge read: pass number and, per slot, top-1 probability, argmax, probability of
        the draft token, draft token, open flag
      - traces/<set>/<idx>.npz: the drafter's confidence per slot and the stock passes' confidences (round-3 format)
      - *.running.jsonl: running metrics over the whole cell after every item (below)
    per stage: logs/phase2/redraft/<ARM>/summary.json (step 3), logs/phase2/probe/<ARM>__r3_B/probe.json and
      experts.npz (step 4), logs/phase2/report/<ARM>/experts.md (experts by stage, judge probabilities by pass,
      drafter reliability), logs/phase2/gate/<ARM>_zeroshot/gate.md (verdict)
E4. Every 15 minutes:
    - tail -5 logs/phase2/run_all.log ; tail -5 logs/phase2/<ARM>/pipeline.log
    - step 3: redraft summary.json -> per_domain agreement, prev_agreement, ratio_to_prev, raw_mask_rate
    - step 4: probe.json -> all/conf_precision, keep_rate, fix_rate, ece, experts/block_uniq_per_layer
    - step 5: last line of each logs/phase2/decode/<ARM>/<set>/*.running.jsonl -> items, accuracy, delta_acc_vs_s0 with
      delta_acc_ci95_lo/hi, speedup_charged_vs_stock, exp_per_token, exp_per_fwd_per_layer_judge,
      router_top8_mass_judge, judge_p_mean_pass1, draft_agreement, passes_per_block, sec_per_item;
      shard exits in logs/phase2/jobs/decode_<ARM>/run.log
    - stop events: find logs/phase2 -name stop_events.jsonl -o -name STOPPED.json
    - backup: tail -3 logs/phase2/hf_backup.log (a sync every 30 min; no "SECRET FOUND"; no repeated failures)
    - nvidia-smi (all 8 busy during decodes), df -h /workspace
E5. Every hour, send me this report:
      time (IST) | plan step (done / running / queued) and its ETA | which family arm is running (F2, or F1 as the
      fallback) | step-3 agreement vs S0 per domain | step-4 precision / keep / fix | step-5 per set: running
      accuracy, delta vs S0 [CI], charged speedup, experts per token vs S0's, experts per layer per judge forward,
      judge mean top-1 prob at pass 1 | stop events | backup last sync | anything unusual

---------------------------------------------------------------------------------------------------------------------
PART F. WHAT TO DO WHEN SOMETHING GOES WRONG
---------------------------------------------------------------------------------------------------------------------
    | symptom                                        | action                                                         |
    | a setup CPU step fails (pip, download, sha,    | read logs/phase2/setup/*.log, fix the cause (network retry,    |
    | IFEval copy, stock IFEval check)               | missing package), rerun D4: every step resumes. If the stock   |
    |                                                | IFEval records fail their check, STOP and tell me (never       |
    |                                                | decode stock instead without my approval)                      |
    | tests fail                                     | fix code under code/phase2/ only; /venv/p2/bin/python -m pytest |
    |                                                | -q code/phase2/tests; commit; push (same credential helper);   |
    |                                                | rerun D4                                                        |
    | check_stock is not 16/16                       | one fix attempt at most; then STOP and tell me (the new venv   |
    |                                                | changed the 30B's outputs). Never relax the check.             |
    | F2, S1 or F1 fails to load                     | fix its wrapper in code/phase2/drafters.py if the cause is     |
    |                                                | clear, test, commit, push, rerun; else STOP and tell me        |
    |                                                | (dropping or swapping a drafter needs my approval)             |
    | F2's re-draft stops (drafter_misfit)           | a RESULT ("does not fit"): report its STOPPED.json; the        |
    |                                                | launcher runs F1 in its place automatically. If it looks like  |
    |                                                | a bug (agreement ~0, a crash), tell me first                   |
    | S1's re-draft stops                            | a RESULT ("a 0.6B drafter does not fit"): report it; the plan  |
    |                                                | ends; no replacement without my approval                       |
    | a decode set stops (futility, no_eos, scorer)  | a RESULT, not a failure: report it; do not rerun it            |
    | a job crashes (exit not 0 or 3)                | read logs/phase2/jobs/<stage>_<ARM>/<job>.log; if it is a Phase |
    |                                                | 2 code bug: fix + test + commit + push, then rerun that step   |
    |                                                | (bash code/phase2/run_phase2.sh <ARM>; finished shards skip)   |
    | IFEval scoring errors (score_error in records) | tell me at once (checker dependencies: langdetect, nltk)       |
    | backup fails 2 syncs in a row                  | read logs/phase2/hf_backup.log; tell me; never print the token |
    | disk < 50 GB free                              | tell me before deleting anything; never delete in /workspace/dmge|
    | the GPUs are still busy 1 h after you started  | keep waiting (never touch other jobs); tell me the ETA         |

---------------------------------------------------------------------------------------------------------------------
PART G. AT THE END
---------------------------------------------------------------------------------------------------------------------
  1. Show me logs/phase2/STATUS.md, both gates (logs/phase2/gate/F2_zeroshot/gate.md -- or F1_zeroshot if F1 ran --
     and S1_zeroshot/gate.md) and the reports (logs/phase2/report/{F2 or F1,S1,S0}/experts.md).
  2. One table per drafter: agreement vs S0 per domain; probe precision / keep / fix / ECE; and per set: accuracy,
     S0 accuracy, delta with 95% CI, charged speedup (drafter vs S0), experts per token vs S0's, passes per block,
     experts per layer per judge forward, judge mean top-1 probability at pass 1, drafter confidence vs accuracy,
     stopped or complete.
  3. The corpus reading (F2 or F1): where its agreement and accuracy differ from S0, set by set, and whether the
     pattern follows its corpus (code vs maths vs science vs instruction following).
  4. The size reading (S1): its experts per token vs S0's (the ~13.6% break-even), its charged speedup, and whether
     accuracy held, set by set: is the 0.6B drafter worth it?
  5. Per drafter, the outcome: alignment transfers / alignment hurt (retrain needed, later rental) / cross-domain gain /
     size pays off / does not fit.
  6. Confirm the final backup succeeded; give the Hugging Face URL.
  7. List any code you changed (commit hashes) and anything that did not run, with why.
```
