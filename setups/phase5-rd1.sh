# Phase 5 (FAILED, kept to reproduce the negative result): the judge retrained on SGLang-recorded states (155.66M),
# with the Phase 4 drafter and the Phase 5 eval flags. Its outputs loop; see RESULTS.md section 4.
DESC="Phase 5 rd1: retrained judge (failed)"
NEEDS="judge_phase5 drafter"
FLAGS="--drafter $ROOT/weights/drafter --draft s8 --judge full --adapter $ROOT/weights/judge_phase5 --full-redraft-every 1 --redraft-ban-rejected --loop-guard 12"
