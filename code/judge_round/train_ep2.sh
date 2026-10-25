#!/usr/bin/env bash
# Phase 5 epoch-2 resume (Mac-session decision 03:04 IST, 24 Oct): training finished one epoch early at 147.61M
# (logs/phase5/judge/rec_main/final). Continue that adapter+optimizer+token-count for a SECOND epoch (a new uniform
# draw of the recorded passes) on the SAME recorded states/index, writing to a NEW out dir so the 147.61M final stays
# untouched. Stops at the FIRST of: --stop-at 04:05 IST (unix 1792794900 = 2026-10-23 22:35:00 UTC) or --max-tokens
# 160e6; either way train_c1b.py saves `final` + probe at the stop. Milestone at 152e6. Recipe/numerics unchanged
# (continue-from). --epochs 4 only so an epoch boundary cannot stop it before the time/token cap.
# STEP-0 GATE OFF (--step0-states 0): the step-0 check verifies that a FRESH continue-from adapter reproduces the
# RECORDING (128M) judge's argmax on its own states, catching a sampler/kernel mismatch before training. Here we
# continue the ALREADY-TRAINED 147.61M adapter, which by design has diverged from the 128M recording judge (measured
# agreement 0.7256 at 03:06 IST — that divergence IS the epoch-1 training signal). The sampler/kernel/states are
# identical to epoch 1, whose step-0 already passed (0.898 >= 0.85), so there is nothing new to gate; the trainer
# itself skips step-0 on --resume of an already-trained run (train_c1b_rec.py load_and_check).
set -u
ROOT=/workspace/diffusion-moe-general-expert; cd "$ROOT"
export HF_HOME=$ROOT/hf_home HF_HUB_OFFLINE=1 C1B_CODE_DIR=$ROOT/code
. logs/phase5/sgl_setup/sgl_env.sh 2>/dev/null
say(){ echo "[ep2 $(TZ=Asia/Kolkata date '+%H:%M:%S IST')] $*"; }
say "resume from rec_main/final (147.61M) -> rec_main_ep2, stop-at 1792794900 (04:05 IST) / max 160e6"
/venv/p2/bin/torchrun --standalone --nproc_per_node=8 code/judge_round/train_c1b_rec.py \
  --states logs/phase5/judge_states/rec_main --index logs/c1b/trainset/fresh/train_index.jsonl \
  --out logs/phase5/judge/rec_main_ep2 --continue-from logs/phase5/judge/rec_main/final \
  --epochs 4 --max-tokens 160e6 --save-at-tokens 152e6,160e6 --stop-at 1792794900 \
  --step0-states 0 2>&1 | tee -a logs/phase5/judge/rec_main_ep2_train.log
say "train_c1b_rec exited rc=${PIPESTATUS[0]}"
