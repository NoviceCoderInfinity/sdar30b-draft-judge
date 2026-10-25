#!/usr/bin/env bash
# Rebuild the results page now (niced; logged in bg_jobs.tsv like the scheduled builds) and print the run's health.
cd "$(dirname "$0")/../.."
L=logs/phase3_8gpu; t0=$(date +%s.%N); rc=0
for s in report_8gpu.py build_page.py; do nice -n 19 ionice -c3 /venv/p2/bin/python code/phase3_8gpu/$s >> $L/analysis.log 2>&1 || rc=1; done
printf '%s\t%s\tpage\t%s\tpublish\n' "$t0" "$(date +%s.%N)" "$([ $rc = 0 ] && echo ok || echo failed)" >> $L/bg_jobs.tsv
echo "page build rc=$rc"; tail -3 $L/run.log
for f in $L/STOP $L/canary/MACHINE_FLAG; do [ -e $f ] && echo "PRESENT: $f"; done
grep -E "FAILED|STOP|CANARY|Traceback" $L/run.log | tail -3
echo "decode processes: $(pgrep -f 'run_da_decod[e].py' | wc -l); supervisor: $(supervisorctl status | grep p3g | awk '{print $1":"$2}' | paste -sd' ')"
tail -2 $L/backup.log 2>/dev/null
