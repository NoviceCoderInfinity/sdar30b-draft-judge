# Phase 3: stock SDAR-1.7B drafter + judge, re-drafting the rejected slots after every second judge pass.
DESC="Phase 3 rd2: re-draft after every second judge pass"
NEEDS="judge d32"
FLAGS="--drafter d32 --draft s8 --judge full --adapter $ROOT/weights/judge --judge-diag --full-redraft-every 2"
