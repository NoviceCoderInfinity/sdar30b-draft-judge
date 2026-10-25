# Phase 3: stock SDAR-1.7B drafter + judge, re-drafting the rejected slots after every judge pass.
DESC="Phase 3 rd1: re-draft after every judge pass"
NEEDS="judge d32"
FLAGS="--drafter d32 --draft s8 --judge full --adapter $ROOT/weights/judge --judge-diag --full-redraft-every 1"
