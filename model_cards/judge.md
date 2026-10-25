---
license: apache-2.0
base_model: JetLM/SDAR-30B-A3B-Chat-b32
library_name: custom
tags: [lora, block-diffusion, speculative-decoding, judge, sdar, moe]
datasets: [Anupam-Rawat-IITB/sdar30b-draft-judge-data]
---

# SDAR-30B judge LoRA

A small LoRA "judge" on the frozen **JetLM/SDAR-30B-A3B-Chat-b32** (block-diffusion MoE, block 32) that makes it
decode faster. A drafter fills each 32-token block with guesses; the 30B with this judge reads the block once per pass
and commits the guesses it is sure of. Blocks finish in fewer 30B passes (12.8 → 9.7 on average, 7.7 with the
trained drafter and re-drafting), at about the same accuracy.

Part of [sdar30b-draft-judge](https://github.com/NoviceCoderInfinity/sdar30b-draft-judge): code, setups and full results.
Companion drafter: [Anupam-Rawat-IITB/SDAR-1.7B-drafter-LoRA](https://huggingface.co/Anupam-Rawat-IITB/SDAR-1.7B-drafter-LoRA).
Data and eval records: [Anupam-Rawat-IITB/sdar30b-draft-judge-data](https://huggingface.co/datasets/Anupam-Rawat-IITB/sdar30b-draft-judge-data).

## Use

```bash
curl -sL https://raw.githubusercontent.com/NoviceCoderInfinity/sdar30b-draft-judge/main/get.sh | bash -s phase4-rd1
cd sdar30b-draft-judge && . env.sh && ./run.sh phase4-rd1 gsm8k500
```

Weights only: `hf download Anupam-Rawat-IITB/SDAR-30B-judge-LoRA --include "adapter*"`.

**Never merge this adapter.** It is flag-gated: it acts only at positions that hold a drafter guess, and the model is
bit-identical to the base everywhere else (prefill, KV flush, non-drafted slots). A merged weight would apply
everywhere. Load it with `draftalign.adapter` / `run_da_decode.py --adapter <dir>` from the GitHub repo; generic PEFT
loading is not supported.

## Files

| Path | What |
|---|---|
| `adapter.safetensors`, `adapter_config.json` | **the judge**: the 128M-token checkpoint (weights sha256 `35805974…`), used by every result |
| `trainer_state.pt`, `trainer_state.json` | its AdamW state and token count (resume with `--continue-from`) |
| `checkpoints/main/…` | first run, 0 → 28.5M tokens: every rolling and milestone checkpoint |
| `checkpoints/main_128m/…` | continuation, 28.5M → 128M tokens (`tok_000128000000` = the root files) |
| `checkpoints/README_source.md` | the original checkpoint table (tokens, step, upload time) |

## Model

- LoRA rank 32, alpha 64 on q, k, v, o of all 48 attention layers, plus a learned draft-flag embedding:
  26.7M trainable parameters (0.09% of the 30B). Base model frozen.
- Training: `code/corpus1b/train_c1b.py`, 8 GPUs, lr 8e-5, 128M tokens. Data: the 30B's own checked answers to
  maths, code, chat, instruction-following and science prompts (corpus waves A and B, domain weights maths 25 /
  code 25 / chat 20 / science 15 / IF 15), each block replayed with the stock SDAR-1.7B's drafts. Loss: cross-entropy
  towards the 30B's token at every open slot, override-weighted, with certainty and calibration terms.
- Decode rule: commit a slot when the judge's probability exceeds 0.95; the rest go back to [MASK].

## Results (2,250 questions, 7 benchmarks, plain HF harness)

| Setup | Accuracy | 30B passes/block | Speed vs vanilla |
|---|---|---|---|
| vanilla 30B | 65.5 | 12.79 | 1.00× |
| + SDAR-1.7B drafter + this judge (Phase 1) | 65.0 | 9.72 | 1.12× compute |
| + re-drafting, rd1 / rd2 (Phase 3) | 64.4 / 64.9 | 7.64 / 7.99 | 1.20× / 1.26× wall-clock |
| + trained drafter, rd1 / rd2 (Phase 4) | 64.5 / 64.2 | 7.68 / 7.83 | **1.25× / 1.26× wall-clock** |

Per benchmark: [RESULTS.md](https://github.com/NoviceCoderInfinity/sdar30b-draft-judge/blob/main/RESULTS.md).

## Limitations

- IFEval is the weak spot: 51.0% with this judge vs 52.7% vanilla, and 46–48% once re-drafting is on.
- Trained and evaluated only with SDAR-30B-A3B-Chat-b32 @ `c351bbc`, block 32, threshold 0.95.
- A later attempt to retrain this judge on SGLang-recorded states (Phase 5) failed: the outputs loop. This checkpoint
  remains the judge to use. See RESULTS.md section 4.

## Credits

Anupam Rawat and Arush ([Arushhh](https://huggingface.co/Arushhh)); consolidated from
`Arushhh/c1b-judge-lora-sdar30b-b32`. Apache-2.0, as the base model.
