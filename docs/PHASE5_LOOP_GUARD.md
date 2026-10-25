# Phase 5, experiment 1: the loop guard (IFEval only, rd1 and rd2)

**Why.** Phase 4 (`results/phase4_drafter/`) trained the drafter and left the IFEval gap where it was: −6.5 / −6.7 points
vs vanilla, with 104 / 107 looping answers (vanilla 44). The attribution table (`ANALYSIS.md` §7) puts the loss on
re-drafting (128M ckpt → rd1 −4.4 IFEval points, all through loops), not on the drafter. The pre-registered rule of
8 Oct was: if drafter SFT does not bring rd2's IFEval loops to ≤ 85 with +2 points, the fix is in the decode loop.
Loops stayed at 107. This is that decode-loop fix, taken from Phase 3's own list of untried feedback rules
(Phase 3 page, Table 15, "block loop continuations").

**The rule** (`--loop-guard 12`; `code/draftalign/loop_guard.py`). A drafter fill, first draft or re-draft, may not
complete a 12-token sequence that already appears in the committed output. At every drafter step, each open slot whose
11 preceding tokens are known gets the logits of every such completing token set to −inf, before tokens are chosen and
confidences ranked (the phase-4 ban's mechanism). The softmax renormalises after the ban, so a banned slot's confidence becomes the runner-up's share; when that is lower than the other open slots' the s8 schedule fills it in a later step, and its neighbours are drafted before they see the replacement token (the CPU test pins this). No extra forward. The judge and its rule are untouched: only the judge
commits, still at p > 0.95. Off, the decode is byte-identical to Phase 4 (`code/tests/test_da_decode_loopguard.py`).

**Arms.** rd1 + ban + guard and rd2 + ban + guard, Phase 4 merged drafter (`dr3d174340`), 128M judge, IFEval-541,
8 shards. Paired against Phase 4's own IFEval records (same box, `logs/phase4/decode/{rd1,rd2}/ifeval541`) and
vanilla on the same box (`logs/phase4/decode/vanilla_here/ifeval541`).

**Decision, written before the run** (`code/loop_guard/verdict_lg.py`, per arm):

| | rd1 | rd2 |
|---|---|---|
| Phase 4 IFEval loops | 104 | 107 |
| Phase 4 IFEval accuracy vs vanilla | −6.5 | −6.7 |
| **Pass** | loops ≤ 85 **and** paired Δacc vs Phase 4 ≥ +2.0 | same |
| Target | loops ≤ 60 (base's level), IFEval CI vs vanilla covering 0 | same |
| **Kill** | loops > 95 **or** Δacc < +1.0 | same |
| Also reported | judge passes/block, drafter forwards/block, seconds per answer vs Phase 4, guard slots / changed tokens per block | |

Kill means the loops are not something the drafter's fills control: the judge writes them itself (forced commits,
confident continuations of a repeated prefix), and the next lever is judge-side, not drafter-side.

**Cost on a fresh box.** Setup from the public backups about 35 min (Phase 4 measured 27 min for the base setup plus
the Phase 4 inputs), tests and smoke about 10 min, decode about 1 h 45 min (IFEval took 49 min for rd1 and 52 min for rd2
on 8× RTX PRO 6000 in Phase 4), report and upload about 10 min: about 2 h 45 min, rent 4 h. At $11.3/h (Vast offer
55456374, Texas, verified, the same CPU and RAM as Phase 4's box) about $45 for 4 hours including disk.

**Timing caveat.** Phase 4 was timed on its own box. Judge passes, drafter forwards, accuracy and loops are
machine-independent and carry the decision; seconds per answer are comparable only if the new box matches Phase 4's
(2× AMD EPYC 9555, 755 GB, Server Edition), otherwise they are labelled cross-machine.

**Not in this experiment.** No training, no corpus generation, no other benchmark, no change to the commit rule. The
oracle-draft probe (the speed ceiling under the 0.95 rule) is the next experiment, decided after this one.


---

# Phase 5 as one rental: the three stages, their times and their rules

Every time below has a measured basis; the two that are projections are marked.

| Stage | What runs | Time | Basis |
|---|---|---|---|
| 0. Setup | `code/loop_guard/setup_lg.sh`: venv, 30B + 1.7B, judge, refs, benchmark sets, 16/16 checks, Phase 4 inputs, 128M checkpoint with optimizer, CPU tests | 40 min | Phase 4 measured 27 min for the base setup (30B 61 GB in 10m41s on a 1 Gbps link); the Texas offer has 570 Mbps, so the download part is projected at about 15 min; plus about 2 GB of Phase 4 inputs and the tests |
| 1. Loop guard | smoke 4 items; IFEval-541 rd1 and rd2, Phase 4 drafter + ban + guard (`run_lg.sh decode`, `report`) | 1 h 50 min | Phase 4 Gate 4 on IFEval: rd1 49 min, rd2 52 min on the same box type; smoke about 5 min |
| 2. Probes | A `run_probes.sh bins` (judge confidence bins, Gate-2 states); B `oracle` (rd1 with vanilla's tokens as drafts, Gate-3 quarter); C `tau` (rd1 + ban + guard, agreed drafts commit at 0.90, quarter) | 1 h 30 min | Gate 2's probe took 48.5 s + model load; Gate 3's quarter decode took 41.5 min; the oracle arm needs fewer passes so at most that |
| 3. Judge round | `run_jr.sh subset` (34M corpus tokens, about 80k items) → `drafts` (trained drafter's first drafts) → `swap` → `train` (128M → 160M tokens, same recipe) → `gate` (quarter, rd1 + ban + guard, new judge) → `full` (7 sets, rd1 and rd2) → `report` | 12 h 30 min: drafts 3.2 h, swap 10 min, train 3.4 h, gate 45 min, full 4 h 50 min | drafts: Phase 4's HF replay ran 5.8-9.2 items/s on 8 GPUs (3,000 held-out items in 326-521 s), 80k items at 7 items/s = 3.2 h (projection from that rate); train: Phase 1 measured 9.5M tokens/h on 8× RTX PRO 6000; gate: Gate 3 took 41.5 min; full: Gate 4 took 4 h 51 min |
| Total | | about 16 h 30 min of work | rent 20 h; stops early if stage 1 kills (then about 4 h) or the stage 3 gate kills (then about 12 h) |

**Stage 2 reads.** A answers whether an agreed draft at confidence 0.80-0.95 is right often enough (97% or more) to be
committed at a lower bar; B gives the pass count a perfect drafter still needs under the 0.95 rule, i.e. the ceiling of
any drafter work; C is the one decode-rule change that already exists as flags, judged like Gate 3 (at least 5% fewer
judge passes per block at accuracy within noise and no more loops). None of the three changes what stage 3 trains; A
and B decide what comes after stage 3.

**Stage 3, written before the run.** The judge continues from the 128M checkpoint (adapter + optimizer + token count)
for 32M corpus tokens on the same recipe (weighted CE + certainty + calibration; round 3 / Phase 1 defaults), with one
change in the data: the uncommitted slots of every state carry the Phase 4 drafter's first drafts instead of the
untrained drafter's. Labels (stock's tokens and commit order) are untouched.
* Gate (quarter, rd1 + ban + guard, new judge, paired with Phase 4's rd1_gate3 on the same questions, `compare_gates.py --gate 3`):
  pass = at least 5% fewer judge passes per block with the accuracy CI covering 0 and IFEval loops not higher;
  kill = under 2% fewer passes, or accuracy CI below 0. Kill means the drafter mismatch was not what held the judge
  back; the lever is then the commit rule (probe A/C) or re-draft-shaped states, which need an infill drafter path.
* Full (7 sets, both modes, vs Phase 4 Gate 4 and vs vanilla on the same questions): the headline is judge passes per
  block, accuracy, loops and seconds per answer, all on this box.
* Expectation, stated honestly: the sure gain is the judge's trust calibration on the new drafter (Gate 2 saw its
  wrong-draft commit precision fall 92.7 to 87.8 points); the speed gain is likely small (Phase 1's judge scaling gave
  4% fewer passes per 4.5× data). That is why the quarter gate comes before the 5-hour full evaluation.
