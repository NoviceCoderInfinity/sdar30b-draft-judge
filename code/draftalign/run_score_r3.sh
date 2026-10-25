#!/usr/bin/env bash
# Score finished code-pool decode shards as the unprivileged user `r3score` (see score_r3_records.py).
#   bash code/draftalign/run_score_r3.sh opencode_pool_0_6000          # one pass over newly finished shards
#   nohup bash code/draftalign/run_score_r3.sh opencode_pool_0_6000 loop > logs/draftalign/score_r3.log 2>&1 &
# Isolation: non-root uid (cannot write /root, /home, or delete root-owned files in /tmp,/dev/shm), a throwaway
# cwd, read-only code copy, file-size and process-count limits, empty environment (no tokens). Root copies only
# the resulting *.jsonl / *.done into <cell>_scored/, which build_r3_index.sh reads for code pools.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
NAME="$1"; MODE="${2:-once}"
if [[ "$NAME" == /* ]]; then REC="$NAME"; FINAL="${3:?out dir}"; NAME=abs$(echo "$REC" | md5sum | cut -c1-10)   # absolute records dir (probes); short: TMPDIR=$SB holds AF_UNIX sockets (108-char limit)
else REC=/dev/shm/dlm/logs/draftalign/datagen/${NAME}_dv-s8_blind_q50_r3; FINAL=${REC}_scored; fi
CODE=/dev/shm/dlm/scorer_code
SB=/dev/shm/dlm/sandbox/score_${NAME}
id r3score >/dev/null 2>&1 || useradd --system --no-create-home --shell /usr/sbin/nologin r3score
mkdir -p "$FINAL"
count() { { find "$1" -maxdepth 1 -name "$2" 2>/dev/null || true; } | wc -l; }   # 0 (not an abort) when the dir is missing or empty
SRC_REC="$REC"
while :; do
  if [[ "$SRC_REC" == /root/* ]]; then   # r3score cannot traverse /root: score a world-readable copy (.done first => each copied .jsonl is complete)
    REC=/dev/shm/dlm/score_rec/$NAME; mkdir -p "$REC"
    cp -f "$SRC_REC"/*.jsonl.done "$REC"/ 2>/dev/null || true; for f in "$REC"/*.jsonl.done; do [ -e "$f" ] && cp -f "$SRC_REC/$(basename "${f%.done}")" "$REC"/; done
    chmod -R a+rX /dev/shm/dlm/score_rec
  fi
  rm -rf "$CODE" && mkdir -p "$CODE" && cp -r "$ROOT/code/." "$CODE/" && chown -R root:root "$CODE" && chmod -R a+rX,go-w "$CODE"
  rm -rf "$SB" && mkdir -p "$SB/out" && chown -R r3score:r3score "$SB"
  (cd "$SB" && ulimit -f 4194304 -u 8192 && setpriv --reuid=r3score --regid=r3score --clear-groups \
     env -i HOME="$SB" PATH=/usr/bin:/bin TMPDIR="$SB" MPLCONFIGDIR="$SB/.mpl" PYTHONDONTWRITEBYTECODE=1 \
     /venv/r3/bin/python "$CODE/draftalign/score_r3_records.py" --records "$REC" --out "$SB/out" \
       --skip-dir "$FINAL" --code "$CODE") || echo "[run_score_r3] scorer exited non-zero"
  for f in "$SB"/out/*.jsonl; do
    [ -e "$f" ] || continue
    n=$(basename "$f"); [ -e "$SB/out/$n.done" ] || continue
    cp "$f" "$FINAL/$n" && cp "$SB/out/$n.done" "$FINAL/$n.done"
  done
  echo "[run_score_r3] $(date -u +%FT%TZ) scored shards: $(count "$FINAL" '*.jsonl.done')"
  [ "$MODE" = loop ] || break
  nsc=$(count "$FINAL" '*.jsonl.done'); nall=$(count "$REC" 'shard*of*.log')
  [ "$nall" -gt 0 ] && [ "$nsc" -ge "$nall" ] && { echo "[run_score_r3] all $nall shards scored"; break; }
  sleep 300
done
