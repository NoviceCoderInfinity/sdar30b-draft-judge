# Phase 5 ad-hoc scripts (kept for the record)

One-off scripts run by hand from `/workspace/HANDOFF/` during the Phase 5 wrap-up on instance 57349095 (TEXAS).
Copied here verbatim before the server was destroyed. They are NOT part of the normal pipeline; paths are hard-coded
to that box. Listed newest-last.

- **vtest.sh** (23 Oct ~15:00 IST) — SGLang vanilla-generation smoke test: one GPU (7), `gen_sglang.py` on gsm8k500 into
  `logs/phase5/judge_sgl/vanilla_sgl_gsm8k500`, timing the run. Used to sanity-check the SGLang venv/plugin setup early
  in the session. (SGLang = data-gen only; never a reported benchmark number.)
- **diag.sh** (23 Oct ~23:00 IST) — the 4-adapter rd1 bisect that found Phase 5 = FAIL. Runs the plain HF harness
  (`run_da_decode.py`, exact rd1 flags) on ifeval541's first 16 items for a128=start_128m / b136=rec_main/tok_136M /
  c147=rec_main/final(147.61M) / d155=rec_main_ep2/final(155.66M), 2 GPUs each. Output under `logs/phase5/eval_diag/{a128,b136,c147,d155}`.
  Result: only 128M terminates (acc 56%, 89% stop@EOS); every retrained checkpoint loops (acc 7–9%, 86–88% truncated).
- **chain128.sh** (24 Oct 04:39 IST) — the final same-box GPU chain for the Phase-4 (128M) judge: rd1 eval on
  gsm8k500/math500/humaneval/mbpp (done), then the vanilla multiplier check (N=28, 7 sets), then gpqa/aime (+ifeval541 if
  time), with a 07:10 IST hard-stop watchdog that kills decodes by PID. Stopped by PID at 05:46 IST for server removal
  (eval complete; vanilla partial — ifeval541 + math500 only). Numbers folded into `results/phase5/REPORT.md`.
