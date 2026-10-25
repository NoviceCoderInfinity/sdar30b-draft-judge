# Phase 4: the trained drafter (SDAR-1.7B + LoRA, merged) + judge, re-draft after every pass, rejected tokens banned on re-draft.
DESC="Phase 4 rd1: trained drafter + judge, re-draft every pass"
NEEDS="judge drafter"
FLAGS="--drafter $ROOT/weights/drafter --draft s8 --judge full --adapter $ROOT/weights/judge --judge-diag --full-redraft-every 1 --redraft-ban-rejected"
