#!/bin/bash
# Detached takeover of the re-queue watcher that the tmux-0 monitoring session held as a foreground shell.
# When orchestrator v5 pauses the queue for the T2b wave it pkills the anchored-draft shard that is mid-flight;
# this re-queues every anchored shard that has not printed SHARD DONE and is not already in the paused queue,
# at the head, so the wave's "queue resumed" picks them up first. Shards resume from their own records.
set -u; cd "$(dirname "$0")/../.."
LOG=logs/draftalign/orchestrate_round1.log
Q=logs/draftalign/queue.paused
PAT='T2b finished; pausing the queue'
LINE_TMPL='CUDA_VISIBLE_DEVICES={GPU} nohup /venv/main/bin/python code/draftalign/run_da_decode.py --model b32 --drafter d32 --items calibvalid --draft s8 --judge blind --tau-acc-q .50 --draft-after-pass 1 --shard SH --nshards 4 --gpu {GPU} --score > logs/draftalign/train/eval_anchored_blind_q50_da1.shardSHof4.log 2>&1'
log(){ echo "$(date -u +%FT%TZ) $*" >> "$LOG"; }
log "re-queue watcher taken over as a detached process (pid $$) by the new monitoring session; old tmux-0 shell retired"
# 1. wait for the pause line
until grep -q "$PAT" "$LOG"; do sleep 15; done
# 2. wait for the queue runner to be gone and the paused queue to exist
until [ "$(pgrep -fc 'gpu_q[u]eue.sh')" -eq 0 ] && [ -f "$Q" ]; do sleep 5; done
sleep 5
# 3. any anchored shard with no SHARD DONE and not already queued gets prepended
ADD=""
for sh in 0 1 2 3; do
  L=logs/draftalign/train/eval_anchored_blind_q50_da1.shard${sh}of4.log
  grep -q 'SHARD DONE' "$L" 2>/dev/null && continue
  grep -q -- "--shard $sh --nshards 4 .*anchored\|anchored.*shard${sh}of4" "$Q" 2>/dev/null && continue
  grep -q "eval_anchored_blind_q50_da1.shard${sh}of4.log" "$Q" 2>/dev/null && continue
  ADD="$ADD${LINE_TMPL//SH/$sh}"$'\n'
done
if [ -n "$ADD" ]; then
  { printf '%s' "$ADD"; cat "$Q"; } > "$Q.new" && mv "$Q.new" "$Q"
  log "post-pause: re-queued interrupted anchored shards at the head -> $(printf '%s' "$ADD" | grep -o -- '--shard [0-9]' | awk '{print $2}' | tr '\n' ' ')"
else
  log "post-pause: no anchored shard needed re-queuing (all done or already queued)"
fi
