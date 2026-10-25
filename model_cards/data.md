---
license: apache-2.0
pretty_name: sdar30b-draft-judge data
tags: [block-diffusion, speculative-decoding, sdar, distillation]
configs:
- config_name: samples
  data_files: samples/corpus_sample.jsonl
---

# sdar30b-draft-judge data

Everything behind [sdar30b-draft-judge](https://github.com/NoviceCoderInfinity/sdar30b-draft-judge) except the two
models: the generated training corpus, readable samples, training inputs and checkpoints of every phase, every
per-item eval record, and the run logs. Models:
[judge](https://huggingface.co/Anupam-Rawat-IITB/SDAR-30B-judge-LoRA),
[drafter](https://huggingface.co/Anupam-Rawat-IITB/SDAR-1.7B-drafter-LoRA).

## Layout

Each phase folder keeps the paths of that phase's original backup, so `scripts/fetch_data.sh <phase> --into-repo`
puts every file where that phase's scripts expect it.

| Folder | Size | Contents |
|---|---|---|
| `samples/` | small | `corpus_sample.jsonl`: 90 training items decoded to text (prompt, the 30B's answer, passes per block, drafter agreement), 5 domains × waves A/B |
| `phase1/` | 7.3 GiB | the corpus: prompt lists (`data/c1b/prompt_lists*`), generation records (`logs/c1b/gen/`), packed training items waves A + B (`logs/c1b/trainset/`), first drafts, frozen eval sets (`data/c1b/eval/`), Phase 1 eval records and logs |
| `phase2/` | 19.5 GiB | released-drafter decodes (`phase2/decode/`, incl. traces), tables and analysis (`phase2/analysis/`), reference records (`assets/`) |
| `phase3/` | 0.3 GiB | rd1 / rd2 decode records (`logs/phase3_8gpu/decode/`), results, setup logs |
| `phase4/` | 8.3 GiB | drafter training run (`logs/phase4/train/main/`, also in the drafter repo), gate outputs, rd1 / rd2 and same-machine vanilla records (`logs/phase4/decode/`), results |
| `phase5/` | 12.9 GiB | wave C corpus (`logs/c1b/gen/waveC`, `logs/c1b/trainset/waveC/items.tar`), recorded judge states (`logs/phase5/judge_sglang/`, `judge_states/`), retrained judge checkpoints (`logs/phase5/judge/`), eval and diagnosis records, environment freeze, `MANIFEST.sha256`, failure analysis |

`<phase>/README_source.md` is the original backup's README.

## Get it

```bash
bash scripts/fetch_data.sh phase4                 # from the GitHub repo: one phase into data_hf/phase4
bash scripts/fetch_data.sh phase5 --into-repo     # and into the paths the code expects
hf download Anupam-Rawat-IITB/sdar30b-draft-judge-data --repo-type dataset --include "samples/*"
```

## Corpus

The training corpus is the vanilla 30B's own answers: prompts from public datasets (maths, code, chat,
instruction following, science; sources in `phase1/data/c1b/sources_manifest.json`), decoded by
SDAR-30B-A3B-Chat-b32, kept only when the answer reached EOS and, for maths, code and science, checked correct;
decontaminated against the seven
benchmarks, then packed per 32-token block with the vanilla passes and the stock 1.7B's drafts. Waves A + B: 331,818
items (judge and drafter training). Wave C: 150,866 fresh items (Phase 5).

## Credits

Anupam Rawat and Arush ([Arushhh](https://huggingface.co/Arushhh)); consolidated from `Arushhh/c1b-judge-corpus-waveA`,
`Arushhh/phase2-drafter-swap`, `Arushhh/phase3-redraft`, `Arushhh/phase4-drafter-lora`, `Arushhh/phase5-loop-guard`.
Prompts keep their source datasets' licenses; generated answers and code Apache-2.0.
