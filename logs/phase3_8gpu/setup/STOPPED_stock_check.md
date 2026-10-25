# Setup step 4 stopped by the project lead (2026-10-07 07:23 UTC)

- Ran and passed: setup_p2.sh (environment, references, datasets), 128M judge download + weights hash 35805974cd9db999
  (both copies), Phase 1 records byte-identical from both sources, CPU tests (56 passed, 2 deselected as in the 4-GPU run),
  judge check: base pipeline on 16 HumanEval items, 16/16 token-identical to the Phase 1 records.
- Stopped at 13/16 items on the project lead's instruction ("Do no smoke test and get started with the benchmark for rd1"):
  the stock check (--stock on 16 GSM8K-500 items); the 13 finished items were all token-identical to the stock records
  (logs/phase3_8gpu/setup/check_stock.log, "ref_gen_ids True").
- checks.json was therefore not written; run_p3_8gpu.sh was started with P3G_SKIP_SETUP=1 P3G_SKIP_FIRST_CANARY=1.
