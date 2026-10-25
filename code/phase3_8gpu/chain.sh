#!/usr/bin/env bash
# Unattended chain (the project lead, 07 Oct 2026 ~21:40 IST): let rd1 / rd2 finish (launcher in tmux p3g), then run the
# base pipeline (SDAR-1.7B drafter + 128M judge, the same flags minus the re-draft) on all 7 benchmarks (tmux p3g_base),
# then write ALL_DONE (the backup / page loops do one last round and stop). A launcher that exits without its DONE
# marker is restarted up to twice (finished runs are skipped, unfinished ones resume item by item).
cd "$(dirname "$0")/../.."
L=logs/phase3_8gpu
log(){ echo "[chain $(date -u +%H:%M:%S) UTC] $*" | tee -a $L/run.log; }
wait_session(){ while tmux has-session -t "$1" 2>/dev/null; do sleep 30; done; }
start(){ tmux new -d -s "$1" "cd /workspace/dmge && P3G_SKIP_SETUP=1 P3G_ARMS='$2' bash code/phase3_8gpu/run_p3_8gpu.sh 2>&1 | tee -a $L/run.log"; }
rd_done(){ grep -q "\] DONE: results/phase3_8gpu/REPORT.md" $L/run.log || [ -f $L/DONE_rd1_rd2 ]; }

log "chain armed: rd1/rd2 (tmux p3g) -> base on all 7 benchmarks (tmux p3g_base) -> ALL_DONE"
wait_session p3g
for try in 1 2; do
  rd_done && break
  log "rd1/rd2 launcher ended without DONE: restart $try"; start p3g "rd1 rd2"; sleep 5; wait_session p3g
done
if rd_done; then log "rd1/rd2 finished; starting the base pipeline"; else log "rd1/rd2 NOT finished after 2 restarts; starting the base pipeline anyway"; fi
start p3g_base base; sleep 5; wait_session p3g_base
for try in 1 2; do
  [ -f $L/DONE_base ] && break
  log "base launcher ended without DONE: restart $try"; start p3g_base base; sleep 5; wait_session p3g_base
done
/venv/p2/bin/python code/phase3_8gpu/report_8gpu.py >> $L/analysis.log 2>&1
/venv/p2/bin/python code/phase3_8gpu/build_page.py >> $L/analysis.log 2>&1
/venv/p2/bin/python code/phase3_8gpu/backup_8gpu.py --tag final >> $L/backup.log 2>&1
touch $L/ALL_DONE
log "ALL_DONE: rd1, rd2 and base on all 7 benchmarks; final report, page and backup written"
