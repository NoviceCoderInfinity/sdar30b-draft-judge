# Phase 4: the trained drafter (SDAR-1.7B + LoRA, merged) + judge, re-draft after every second pass, rejected tokens banned.
DESC="Phase 4 rd2: trained drafter + judge, re-draft every second pass"
NEEDS="judge drafter"
FLAGS="--drafter $ROOT/weights/drafter --draft s8 --judge full --adapter $ROOT/weights/judge --judge-diag --full-redraft-every 2 --redraft-ban-rejected"
