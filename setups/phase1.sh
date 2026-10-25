# Phase 1: stock SDAR-1.7B drafter + the trained judge (128M checkpoint), one first draft per block, no re-drafting.
DESC="Phase 1: SDAR-1.7B drafter + judge"
NEEDS="judge d32"
FLAGS="--drafter d32 --draft s8 --judge full --adapter $ROOT/weights/judge"
