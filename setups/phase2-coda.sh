# Phase 2: the drafter swapped for the released CoDA-1.7B (Salesforce/CoDA-v0-Instruct @ f2d40ce), judge frozen.
DESC="Phase 2: CoDA-1.7B drafter + judge"
NEEDS="judge F2"
FLAGS="--drafter F2 --draft s8 --judge full --adapter $ROOT/weights/judge"
