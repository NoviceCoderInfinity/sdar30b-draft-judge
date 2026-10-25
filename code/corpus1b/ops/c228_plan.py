#!/usr/bin/env python
"""Time plan for the 128M -> 228M continuation (stdlib only). Run from the repo root by c228_run.sh after setup:

    python3 code/corpus1b/ops/c228_plan.py            # reads /workspace/c228_deadline.env (END=<unix>, MODE=full|box)

MODE=full  convert 100M NEW corpus tokens (maths 35.7M, chat 28.6M, code 35.7M of which ~24.3M is the wave-B code
           already generated on 30 Sep), train to 228M tokens, full eval (7 benchmarks). If that cannot finish
           before END the plan says SHORTFALL=1 and c228_run.sh stops and asks.
MODE=box   whatever fits before END: training gets the time left after the conversion it needs (maths 25% and
           chat 20% of the trained tokens are generated fresh, code comes from the already generated wave-B code,
           science/IF are further passes), +10% conversion headroom; quick eval (GSM8K-500, HumanEval, MBPP) unless
           >= 18 h remain (then the full eval); no eval if < 6 h remain.
Rates are the Phase 1 box's MEASURED ones (logs/c1b/gen/wave*/gpu*/stats.jsonl, logs/c1b/draft_*.log,
logs/c1b/train/main_128m/train.rank0.jsonl). Writes logs/c1b/c228_state/plan.env (sourced by c228_run.sh) + plan.md.
"""
import json
import os
import sys
import time

RATE = {"maths": 1.28, "chat": 2.73, "code": 6.32}       # M kept corpus tokens per GPU-hour (stock 30B, SGLang)
BLOCKS_PER_M = {"maths": 25400, "chat": 27300, "code": 8800}   # drafter blocks per M corpus tokens
DRAFT_BPS = 68.0                                            # drafter blocks / s / GPU (measured 68-94)
TRAIN_MPH = 9.5                                             # M corpus tokens / h on 8 GPUs (main_128m, 512 states)
GPUS = 8
PENDING_CODE_M = 24.3                                       # wave-B code records generated 30 Sep, not yet converted
H_PENDING, H_SWITCH, H_INDEX, H_PUSH = 0.20, 0.75, 0.10, 0.40
H_EVAL = {"full": 2.5, "quick": 0.5, "none": 0.0}
FULL_NEW_M = {"maths": 35.7, "chat": 28.6, "code": 35.7 - PENDING_CODE_M}
FROM_TOKENS = 128.003e6


def gen_h(d, m):
    return m / (GPUS * RATE[d])


def draft_h(blocks):
    return blocks / (GPUS * DRAFT_BPS * 3600.0)


def ist(t):
    return time.strftime("%d %b %H:%M IST", time.gmtime(t + 5.5 * 3600))


def main():
    env = {}
    path = os.environ.get("C228_DEADLINE", "/workspace/c228_deadline.env")
    for line in open(path):
        if "=" in line and not line.strip().startswith("#"):
            k, v = line.strip().split("=", 1)
            env[k] = v.strip().strip('"')
    end, mode = float(env["END"]), env.get("MODE", "full").lower()
    now = time.time()
    left = (end - now) / 3600.0
    fixed = H_PENDING + H_SWITCH + H_INDEX + H_PUSH
    plan = {"mode": mode, "end": end, "planned_at": now, "hours_left": round(left, 2)}

    full_conv = sum(gen_h(d, m) for d, m in FULL_NEW_M.items())
    full_draft = draft_h(sum(FULL_NEW_M[d] * BLOCKS_PER_M[d] for d in FULL_NEW_M))
    full_train = 100.0 / TRAIN_MPH
    full_need = fixed + full_conv + full_draft + full_train + H_EVAL["full"]

    if mode == "full":
        new = dict(FULL_NEW_M)
        eval_mode = "full"
        train_h = full_train
        plan["shortfall"] = int(full_need > left)
    else:
        eval_mode = "full" if left >= 18 else ("quick" if left >= 6 else "none")
        avail = left - fixed - H_EVAL[eval_mode]
        per_train_h = (0.25 * TRAIN_MPH * 1.1) / (GPUS * RATE["maths"]) + (0.20 * TRAIN_MPH * 1.1) / (GPUS * RATE["chat"]) \
            + draft_h(0.25 * TRAIN_MPH * 1.1 * BLOCKS_PER_M["maths"] + 0.20 * TRAIN_MPH * 1.1 * BLOCKS_PER_M["chat"])
        train_h = max(0.0, avail / (1.0 + per_train_h))
        tok = TRAIN_MPH * train_h
        new = {"maths": 0.25 * tok * 1.1, "chat": 0.20 * tok * 1.1, "code": max(0.0, 0.25 * tok * 1.1 - PENDING_CODE_M)}
        plan["shortfall"] = int(train_h <= 0.25)
    gen_hours = {d: round(gen_h(d, m), 3) for d, m in new.items()}
    drafts = draft_h(sum(new[d] * BLOCKS_PER_M[d] for d in new))
    train_end = end - (H_PUSH + H_EVAL[eval_mode]) * 3600.0
    eval_end = end - H_PUSH * 3600.0
    plan.update({"eval_mode": eval_mode, "new_target_M": {d: round(v, 2) for d, v in new.items()},
                 "gen_hours": gen_hours, "draft_hours": round(drafts, 2), "train_hours": round(train_h, 2),
                 "train_end": train_end, "eval_end": eval_end,
                 "expected": {"converted_new_M": round(PENDING_CODE_M + sum(new.values()), 1),
                              "trained_M": round(min(100.0, TRAIN_MPH * train_h), 1),
                              "end_tokens_M": round(FROM_TOKENS / 1e6 + min(100.0, TRAIN_MPH * train_h), 1)},
                 "full_plan_hours_after_setup": round(full_need, 2)})
    os.makedirs("logs/c1b/c228_state", exist_ok=True)
    with open("logs/c1b/c228_state/plan.env", "w") as fh:
        fh.write(f"MODE={mode}\nEND={int(end)}\nSHORTFALL={plan['shortfall']}\nEVAL_MODE={eval_mode}\n"
                 f"TRAIN_END={int(train_end)}\nEVAL_END={int(eval_end)}\nMAX_TOKENS=228e6\nTRAIN_H={train_h:.2f}\n"
                 + "".join(f"GEN_H_{d.upper()}={gen_hours[d]}\nTARGET_M_{d.upper()}={new[d]:.2f}\n" for d in new))
    json.dump(plan, open("logs/c1b/c228_state/plan.json", "w"), indent=2)
    md = [f"# 128M -> 228M continuation plan ({mode.upper()}), made {ist(now)}", "",
          f"Rental ends {ist(end)} ({left:.1f} h from now). Full plan needs {full_need:.1f} h after setup.", "",
          "| stage | hours | target |", "|---|---:|---|",
          f"| convert wave-B code already generated | {H_PENDING:.1f} | ~{PENDING_CODE_M:.1f}M tokens |"]
    for d in ("maths", "chat", "code"):
        md.append(f"| generate {d} | {gen_hours[d]:.2f} | {new[d]:.1f}M new tokens |")
    md += [f"| drafts (1.7B) | {drafts:.2f} | |",
           f"| model loads, index build | {H_SWITCH + H_INDEX:.2f} | |",
           f"| train 128M -> {plan['expected']['end_tokens_M']:.0f}M | {train_h:.2f} | stops {ist(train_end)} |",
           f"| eval ({eval_mode}) | {H_EVAL[eval_mode]:.1f} | ends {ist(eval_end)} |",
           f"| final push + verify | {H_PUSH:.1f} | |", "",
           f"Expected: ~{plan['expected']['converted_new_M']}M new tokens converted, ~{plan['expected']['trained_M']}M trained."]
    if plan["shortfall"]:
        md += ["", "**SHORTFALL: the target does not fit before the rental ends. Stop and ask the user.**"]
    open("logs/c1b/c228_state/plan.md", "w").write("\n".join(md) + "\n")
    print("\n".join(md), flush=True)
    sys.exit(3 if (plan["shortfall"] and mode == "full") else 0)


if __name__ == "__main__":
    main()
