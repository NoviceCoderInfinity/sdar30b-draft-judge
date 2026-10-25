---
license: apache-2.0
base_model: JetLM/SDAR-1.7B-Chat-b32
library_name: transformers
tags: [lora, block-diffusion, speculative-decoding, drafter, sdar]
datasets: [Anupam-Rawat-IITB/sdar30b-draft-judge-data]
---

# SDAR-1.7B drafter LoRA (merged)

**JetLM/SDAR-1.7B-Chat-b32** fine-tuned with a LoRA to draft for the 30B block-diffusion MoE
**JetLM/SDAR-30B-A3B-Chat-b32**: it learns to write what the 30B writes, so the 30B's judge
([Anupam-Rawat-IITB/SDAR-30B-judge-LoRA](https://huggingface.co/Anupam-Rawat-IITB/SDAR-30B-judge-LoRA)) rejects fewer
guesses and fewer re-drafts are needed. The repo root holds the LoRA merged into a full bf16 checkpoint.

Part of [sdar30b-draft-judge](https://github.com/NoviceCoderInfinity/sdar30b-draft-judge): code, setups and full results.

## Use

```bash
curl -sL https://raw.githubusercontent.com/NoviceCoderInfinity/sdar30b-draft-judge/main/get.sh | bash -s phase4-rd1
cd sdar30b-draft-judge && . env.sh && ./run.sh phase4-rd1 all
```

Weights only: `hf download Anupam-Rawat-IITB/SDAR-1.7B-drafter-LoRA --exclude "training/*"`. The root loads with
`AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True)` like the base SDAR-1.7B, and with
`run_da_decode.py --drafter <dir>`.

## Files

| Path | What |
|---|---|
| `model.safetensors`, config, tokenizer, `modeling_sdar.py` | **the drafter**: base + LoRA merged, bf16 (4.1 GB) |
| `drafter_lora_merge.json` | record of the merge |
| `training/final_adapter/` | the LoRA alone (34.9M parameters) |
| `training/ckpt/` | trainer checkpoint with optimizer state (resume) |
| `training/config.json`, `training/train.rank0.jsonl`, `training/*.json` | run config, per-step log, speed, 25% check, held-out ids |

## Model

- LoRA rank 32, alpha 64 on q, k, v, o, gate, up, down of all 28 layers: 34,865,152 trainable parameters.
- Training: `code/drafter_lora/train_drafter_lora.py`, one pass over corpus waves A + B (331,818 items, the 30B's own
  checked answers), 40% first-draft and 60% re-draft states, lr 8e-5. Loss: cross-entropy towards the 30B's token at
  every open slot, plus 0.1 × KL to the original drafter where the original was already right (keeps what it knew).
- Decode-time companion change: `--redraft-ban-rejected` (a token the judge just rejected is not re-proposed).

## Results (2,250 questions, 7 benchmarks; with the judge, plain HF harness, one machine)

| | rd1 | rd2 |
|---|---|---|
| Speed vs vanilla 30B, wall-clock per answer | **1.25×** [1.23, 1.27] | **1.26×** [1.24, 1.28] |
| vs the stock SDAR-1.7B drafter (Phase 3) | 1.04× | 1.00× |
| Drafter forwards per block vs stock drafter | −7.6% | −5.3% |
| Accuracy: vanilla / stock drafter / this drafter | 65.5 / 64.4 / 64.5 | 65.5 / 64.9 / 64.2 |

rd1 = re-draft after every judge pass, rd2 = after every second pass. Per benchmark:
[RESULTS.md](https://github.com/NoviceCoderInfinity/sdar30b-draft-judge/blob/main/RESULTS.md).

## Limitations

- A drafter only: on its own it is a 1.7B model tuned towards the 30B's style, not a better standalone chat model.
- IFEval: 46.2% / 46.0% (rd1 / rd2) vs 52.7% vanilla; repetition loops rise there.

## Credits

Anupam Rawat and Arush ([Arushhh](https://huggingface.co/Arushhh)); consolidated from `Arushhh/phase4-drafter-lora`.
Apache-2.0, as the base model.
