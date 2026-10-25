# Box prompt: loop guard on IFEval (phase 5, experiment 1) — fresh 8x RTX PRO 6000 box

You are Claude Code on a freshly rented Vast box with 8x RTX PRO 6000 (96 GB each), working for Arush (GitHub
NoviceCoderInfinity / commits as Arush777; Hugging Face Arushhh). One experiment, then stop: decode IFEval-541 in rd1 and
rd2 with the Phase 4 drafter, the ban and the new n-gram loop guard, and compare with Phase 4's records. Read
docs/PHASE5_LOOP_GUARD.md first: it holds the rule and the pass / kill decision, written before this run.

## Non-negotiables. If one cannot be met, STOP and ask Arush (through the operator)
1. Nothing is trained. No instance other than this one. No benchmark other than IFEval-541. Nothing outside this prompt.
2. Secrets live only in /workspace/.env (mode 600). Never print, log, commit or upload them. Before every push or
   upload scan the files for the two token values and for `hf_[A-Za-z0-9]{30,}`. Commit as
   "Arush777 <23je0154@iitism.ac.in>", no Co-Authored-By trailer.
3. Never state a speed or a time you have not measured on this box; label projections as projections.
4. The judge adapter is flag-gated: never merge it (AGENTS.md footgun e). Decodes run from per-shard sandbox directories
   under /dev/shm (footgun p). In monitoring shells use bracketed patterns such as `run_da_decode[.]py` (footgun d).
5. Box time: at most 24 hours in total; the stages are run in order and each later stage only after the earlier one
   has been reported. If setup is not done after 1.5 hours, stop and report.
6. Every record of the new cells must carry `loop_guard: 12`; the cell name must end in `_rd1_ban_lg12_dr3d174340_ad35805974`
   (rd2: `_rd2_ban_lg12_...`). If not, stop: the flag did not reach the decoder.

## 1. Keys
Create /workspace/.env (mode 600) with exactly:

GITHUB_TOKEN=<GITHUB_TOKEN>
HF_TOKEN=<HF_TOKEN>

Load it in every shell with: set -a; . /workspace/.env; set +a

## 2. Setup (stop at the first failure)
1. GPUs: `nvidia-smi` shows 8x RTX PRO 6000 and a driver for CUDA 12.8 or newer (Blackwell is sm_120).
2. Repo, token in a header, never in the URL:

   git -c credential.helper= -c "http.extraHeader=Authorization: Basic $(printf 'x-access-token:%s' "$GITHUB_TOKEN" | base64 -w0)" clone -b phase5-loop-guard https://github.com/NoviceCoderInfinity/diffusion-moe-general-expert /workspace/dmge
   cd /workspace/dmge

   Use the same -c options for every push. Read AGENTS.md section 9 before changing anything. Do not change the decoder.
3. `ROOT=/workspace/dmge bash code/loop_guard/setup_lg.sh`. It runs Phase 3/4's setup (builds /venv/p2 with torch 2.11.0
   cu128 and transformers 4.54.1, downloads the 30B, the 1.7B, the 128M judge adapter to logs/phase2/ref/c1b_128M/tok_000128000000,
   the references and the benchmark sets, and checks that vanilla and base decoding are 16/16 token-identical to
   Phase 1), then downloads the Phase 4 merged drafter and the IFEval records of Phase 4 rd1, rd2 and the same-box vanilla
   decode from the public dataset Arushhh/phase4-drafter-lora, then runs the CPU tests. Phase 4 measured about 27 minutes
   for the setup part on a 1 Gbps link (30B 61 GB in 10m41s). Anything short of 16/16, a missing input or a failing
   test: stop and report. Record the measured setup time.

## 3. Smoke (about 5 minutes, GPU 0)
One shard of 4 items with the guard:

   mkdir -p /dev/shm/p5_smoke && cd /dev/shm/p5_smoke && CUDA_VISIBLE_DEVICES=0 /venv/p2/bin/python /workspace/dmge/code/draftalign/run_da_decode.py --model b32 --items ifeval541 --drafter /workspace/dmge/logs/phase4/train/main/merged --draft s8 --judge full --adapter /workspace/dmge/logs/phase2/ref/c1b_128M/tok_000128000000 --judge-diag --full-redraft-every 1 --redraft-ban-rejected --loop-guard 12 --shard 0 --nshards 136 --gpu 0 --score --no-trace --out /workspace/dmge/logs/phase5/smoke

Check the cell name and the record fields (non-negotiable 6), and that `loopguard_slots_per_block` is not all zeros
across the 4 items. Report the seconds per item.

## 4. Decode (all 8 GPUs; Phase 4 measured 49 min for rd1 and 52 min for rd2 on IFEval on its box)
   cd /workspace/dmge && ROOT=/workspace/dmge bash code/loop_guard/run_lg.sh decode
rd1 then rd2, 8 shards each; records resume on a retry. Tell the operator when rd1 finishes, with the measured time.

## 5. Report
   ROOT=/workspace/dmge bash code/loop_guard/run_lg.sh report
It writes results/phase5_loop_guard/lg12_{rd1,rd2}.json (every measured quantity, paired with Phase 4 and with vanilla on
the same questions) and verdict_{rd1,rd2}.json (PASS / KILL / BETWEEN by the pre-registered rule). Write
results/phase5_loop_guard/REPORT.md with: the two verdict lines; one table per arm, Phase 4 -> guard: IFEval accuracy,
delta vs Phase 4 with its 95% range, delta vs vanilla with its range, looping answers, length-cap hits, judge passes per
block, drafter forwards per block, tokens per answer, seconds per answer (labelled cross-machine unless this box is
2x AMD EPYC 9555 / 755 GB like Phase 4's), the guard's slots and changed tokens per block; the machine (nvidia-smi, lscpu
model name, RAM); the measured times of every stage. Nothing else.

## 6. Stage 2: the three probes (about 1 h 30 min, no training)
   ROOT=/workspace/dmge bash code/loop_guard/run_probes.sh bins     # judge confidence bins, ~5 min
   ROOT=/workspace/dmge bash code/loop_guard/run_probes.sh oracle   # rd1 with vanilla's tokens as drafts, the Gate-3 quarter
   ROOT=/workspace/dmge bash code/loop_guard/run_probes.sh tau      # rd1 + ban + guard, agreed drafts commit at 0.90, the quarter
   ROOT=/workspace/dmge bash code/loop_guard/run_probes.sh report
Report to the operator: from results/phase5_loop_guard/probe_bins.json the "all" block of "trained": commit share,
commit precision, and the eased_rule entries (extra commit share and precision at 0.80 / 0.85 / 0.90); from
probe_oracle.json and probe_tau90.json the printed lines (judge passes per block vs Phase 4 rd1 and vs vanilla,
accuracy delta with its range, loops). Three numbers decide nothing here; they are read by Arush.

## 7. Stage 3: the judge round (about 9 h to the gate, 14 h with the full eval; docs/PHASE5_JUDGE_ROUND.md has the rules)
   cd /workspace/dmge && export ROOT=/workspace/dmge
   bash code/judge_round/setup_sgl.sh              # SGLang env + plugin (Phase 1 recipe); ends with "sgl smoke ok"; projected 30 min
   bash code/judge_round/run_jr.sh subset          # ~123k items, 52M corpus tokens, never the Phase 4 held-out items
   bash code/judge_round/run_jr.sh drafts          # D0 in SGLang, one engine per GPU; report blocks/s after 5 min (expect 68-94 per GPU)
   bash code/judge_round/run_jr.sh redrafts        # D1 on the holes, HF harness, 8 shards; report items/s after 10 min
   bash code/judge_round/run_jr.sh swap            # items with D0 and D1 -> logs/c1b/trainset/p4drafts
   bash code/judge_round/run_jr.sh train           # 128M -> 178M on K1/K2/K3 states; report tokens/h after 20 min; stop and ask if under 7M/h
   bash code/judge_round/run_jr.sh gate            # quarter, rd1 + ban + guard, new judge; exit 0 = PASS, 3 = kill rule
   bash code/judge_round/run_jr.sh full            # ONLY on a gate pass: 7 sets, rd1 and rd2
   bash code/judge_round/run_jr.sh report
SGLang is used only in the `drafts` step. Never report a speed from it. If `setup_sgl.sh` fails after two attempts,
stop and ask: the fallback (HF replay for D0, `eval_agree.py --trained-only --dump-drafts`, about 4 h 50 min) is a
decision for Arush, not for you. On a gate kill, skip `full`, back up, and report. Put the gate verdict and, if run,
both report JSONs' pooled and per-set numbers (accuracy vs vanilla and vs Phase 4 with ranges, judge passes per block,
drafter forwards per block, loops, seconds per answer) in results/phase5_loop_guard/REPORT.md under "Judge round".

## 8. Back up, then stop
After every stage: push results/phase5_loop_guard/ to branch phase5-loop-guard (code untouched) and upload logs/phase5/
(decode records, probe outputs, the judge checkpoint tok_000160000000 and its train log; not the swapped corpus copy)
to a NEW PUBLIC Hugging Face dataset Arushhh/phase5-loop-guard (huggingface_hub.upload_folder, token from the
environment, secret scan first). At the end report: every verdict, the tables, the measured time of every stage, the
box hours used, the paths. Then stop; the operator destroys the instance.
