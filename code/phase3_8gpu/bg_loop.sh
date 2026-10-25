#!/usr/bin/env bash
# Background jobs of the 8-GPU Phase 3 re-run, run by supervisor (restarted if they ever crash); niced and idle-IO
# so they yield the CPU to the decode processes:
#   bg_loop.sh backup   every 15 min at :07 :22 :37 :52 UTC -> backup_8gpu.py (GitHub + HF, secret scan, verified)
#   bg_loop.sh page     every 20 min at :01 :21 :41 UTC     -> report_8gpu.py + build_page.py (Claude republishes the page)
# Every job's start / end goes to logs/phase3_8gpu/bg_jobs.tsv, so the analysis can test whether decoded items that
# overlapped a job ran slower than the others. Both loops end once logs/phase3_8gpu/ALL_DONE exists.
set -uo pipefail
cd "$(dirname "$0")/../.."
L=logs/phase3_8gpu; PY=/venv/p2/bin/python
case ${1:-} in
  backup) PERIOD=900; OFFSET=420 ;;
  page)   PERIOD=1200; OFFSET=60 ;;
  *) echo "usage: bg_loop.sh backup|page"; exit 2 ;;
esac
echo "[bg $1 $(date -u +%H:%M:%S) UTC] loop started (every $((PERIOD / 60)) min)"
while true; do
  now=$(date +%s); next=$(( ( (now - OFFSET) / PERIOD + 1 ) * PERIOD + OFFSET )); sleep $(( next - now ))
  t0=$(date +%s.%N)
  if [ "$1" = backup ]; then
    nice -n 19 ionice -c3 $PY code/phase3_8gpu/backup_8gpu.py --tag scheduled >> $L/backup.log 2>&1; rc=$?
  else
    rc=0
    for s in report_8gpu.py build_page.py; do
      [ -f code/phase3_8gpu/$s ] && { nice -n 19 ionice -c3 $PY code/phase3_8gpu/$s >> $L/analysis.log 2>&1 || rc=1; }
    done
    printf '%s\t%s\tpage\t%s\tscheduled\n' "$t0" "$(date +%s.%N)" "$([ $rc = 0 ] && echo ok || echo failed)" >> $L/bg_jobs.tsv
  fi
  echo "[bg $1 $(date -u +%H:%M:%S) UTC] rc=$rc"
  [ -f $L/ALL_DONE ] && { echo "[bg $1] ALL_DONE: loop ends"; exit 0; }   # chain.sh sets it after the last run (rd1, rd2, base)
done
