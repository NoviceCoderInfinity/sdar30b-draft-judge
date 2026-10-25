#!/usr/bin/env bash
# one-shot status of wave A (run from anywhere); bracketed patterns only (AGENTS.md 9(d))
cd /workspace/dmge
. /workspace/deadline.env
now=$(date +%s)
echo "=== $(date -u +%H:%M) UTC / $(TZ=Asia/Kolkata date +%H:%M) IST | to DEADLINE $(( (DEADLINE-now)/60 )) min | free disk $(df -BG --output=avail . | tail -1 | tr -d ' ')"
echo "--- orchestrator: $(tail -1 logs/c1b/waveA.log | cut -c1-170)"
for f in logs/c1b/waveA_state/T_*; do [ -f "$f" ] && echo "    T_${f##*T_} = $(TZ=Asia/Kolkata date -d @$(cat $f) +%H:%M) IST"; done
[ -f logs/c1b/waveA_state/R_hours ] && echo "    R = $(cat logs/c1b/waveA_state/R_hours) h (measured)"
echo "--- prompt lists: $(ls data/c1b/prompt_lists/*.jsonl 2>/dev/null | xargs -n1 basename 2>/dev/null | tr '\n' ' ') | failed: $(ls -a logs/c1b/stage1 | grep failed_prompts | tr '\n' ' ')"
python3 - <<'EOF'
import glob, json, os, collections
tot = collections.Counter(); dom = collections.defaultdict(collections.Counter); el = []
for g in sorted(glob.glob("logs/c1b/gen/waveA/gpu*")):
    st = os.path.join(g, "stats.jsonl")
    if not os.path.exists(st):
        continue
    rows = [json.loads(l) for l in open(st) if l.strip()]
    # stats rows are cumulative per gen run; sum the last row of every run (a run restarts at chunk index ci0)
    last = {}
    for r in rows:
        last[(tuple(sorted(r["by_domain"])), r.get("elapsed_h", 0) < 0)] = r
    for r in rows[-1:]:
        el.append(r["elapsed_h"])
    runs, prev = [], None
    for r in rows:
        if prev is not None and r["elapsed_h"] < prev["elapsed_h"]:
            runs.append(prev)
        prev = r
    if prev is not None:
        runs.append(prev)
    for r in runs:
        for d, c in r["by_domain"].items():
            dom[d].update(c)
for d, c in sorted(dom.items()):
    p = c["prompts"]; k = c["kept"]
    drops = {kk[5:]: v for kk, v in c.items() if kk.startswith("drop:")}
    top = sorted(drops.items(), key=lambda x: -x[1])[:3]
    print(f"    gen {d}: {p} prompts, kept {k} ({100*k/max(1,p):.1f}%), ppb {c['passes']/max(1,c['blocks']):.2f}, "
          f"gen tok {c['gen_tokens']/1e6:.1f}M, top drops {top}")
EOF
for k in 0 1 2 3 4 5 6 7; do l=$(ls logs/c1b/gen_waveA_*.gpu$k.log logs/c1b/draft_waveA_*.gpu$k.log 2>/dev/null | xargs ls -t 2>/dev/null | head -1); [ -n "$l" ] && echo "    gpu$k $(basename $l .log | cut -d. -f1): $(grep -hE '\[gen\] chunk|\[draft\]|Traceback|Error' $l | tail -1 | cut -c1-150)"; done
echo "--- GPUs: $(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits | awk -F', ' '{printf "%s:%dG/%s%% ", $1, $2/1024, $3}')"
echo "--- procs: gen=$(ps -eo args | grep -c '[g]en_sglang.py --wave') draft=$(ps -eo args | grep -c '[d]raft_sglang.py --wave') train=$(ps -eo args | grep -c '[t]rain_c1b.py --index') eval=$(ps -eo args | grep -c '[r]un_da_decode.py --model') builds=$(ps -eo args | grep -c '[b]uild_prompt_lists.py --domains') backup_loop=$(ps -eo args | grep -c '[b]ackup_hf.sh loop')"
echo "--- backup: $(grep -E 'round done|data ok|failed|HIT|verified' logs/c1b/backup_hf.log | tail -2 | tr '\n' ' ' | cut -c1-200)"
echo "--- monitor: $(tail -1 logs/c1b/monitor.log 2>/dev/null | cut -c1-200)"
