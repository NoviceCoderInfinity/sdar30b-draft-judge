# Phase 2: the drafter swapped for the released Qwen3-0.6B diffusion drafter (dllm-hub/Qwen3-0.6B-diffusion-bd3lm-v0.1), judge frozen.
DESC="Phase 2: Qwen3-0.6B drafter + judge"
NEEDS="judge S1"
FLAGS="--drafter S1 --draft s8 --judge full --adapter $ROOT/weights/judge"
