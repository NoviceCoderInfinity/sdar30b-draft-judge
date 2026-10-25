# Drafter candidates (from the Notion page "Phase 2: Drafter swap ablations", Conscious Engines, 2026-09-29)
In the Notion tables, cells that DIFFER from S0 are highlighted orange; recommended picks have a green ID.

## Table 1: Family / corpus variation (same size, about 1.5-1.7B)
| ID | Weights | Base | Size | How its authors built it | Their corpus | Tokenizer | Block | Role | Status in this run |
|---|---|---|---|---|---|---|---|---|---|
| S0 | JetLM/SDAR-1.7B-Chat-b32 | Qwen3-1.7B | 1.7B | Pretraining (50B) + SFT (4B) | In-house web / STEM / code | Qwen3 | 32 | Baseline | reference (Phase 1 records) |
| F1 (green) | Efficient-Large-Model/Fast_dLLM_v2_1.5B | Qwen2.5-1.5B [diff] | 1.5B | SFT only (3.15B; card says ~1B) [diff] | Synthetic LLaMA-Nemotron post-training data [diff] | Qwen2.5 (mask remap) [diff] | 32 | Different family and corpus | not run (fallback only) |
| F2 (green) | Salesforce/CoDA-v0-Instruct | Qwen3-1.7B | 1.7B | Pretraining (179B) + mid-training (21B) + SFT | Public: DCLM, Stack v2, OpenWebMath, arXiv, Wikipedia; OpenCoder SFT [diff] | Qwen3 | Full-sequence [diff] | Partial control: same Qwen3-1.7B base, code-focused corpus | RUN |
| F3 | JetLM/MDM-1.7B | InternLM2 arch, from scratch [diff] | 1.7B | Pretraining only (1.1T) [diff] | Unclear [diff] | 128k vocab, needs bridge [diff] | Full-sequence [diff] | Different family; needs a tokenizer bridge | not wired |

## Table 2: Size variation (any family)
| ID | Weights | Base | Size | How its authors built it | Their corpus | Tokenizer | Block | Role | Status in this run |
|---|---|---|---|---|---|---|---|---|---|
| S0 | JetLM/SDAR-1.7B-Chat-b32 | Qwen3-1.7B | 1.7B | Pretraining (50B) + SFT (4B) | In-house web / STEM / code | Qwen3 | 32 | Baseline | reference |
| S1 (green) | dllm-hub/Qwen3-0.6B-diffusion-bd3lm-v0.1 | Qwen3-0.6B [diff] | 0.6B [diff] | SFT only [diff] | Tulu-3, SmolTalk, OpenCoder SFT [diff] | Qwen3 | 32 | First size test | RUN |
| S2 (green) | dllm-hub/Qwen2.5-Coder-0.5B-Instruct-diffusion-bd3lm-v0.1 | Qwen2.5-Coder-0.5B [diff] | 0.49B [diff] | SFT only [diff] | OpenCoder SFT, code only [diff] | Qwen2.5 (remap) [diff] | 32 | Small code-only drafter | not run |
| S3 | fredzzp/open-dcoder-0.5B | Qwen2.5-Coder-0.5B [diff] | 0.49B [diff] | Continued pretraining, masked diffusion, no SFT [diff] | FineCode, code only [diff] | Qwen2.5 (remap) [diff] | Full-sequence [diff] | Lower bound on quality | not wired |
| S4 | nieshen/SMDM | LLaMA-style, from scratch [diff] | 1.1B [diff] | Pretraining only [diff] | SlimPajama (540B) [diff] | Llama-2, needs bridge [diff] | Full-sequence [diff] | Mid-size; needs a bridge | not wired |
| S5 | OpenGVLab/SDLM-3B-D4 | Qwen2.5-3B [diff] | 3.09B [diff] | SFT only (2.3B) [diff] | Mixed SFT (maths, code, chat, science) [diff] | Qwen2.5 (remap) [diff] | 4 [diff] | Stronger drafter at the limit | not wired (block 4) |
| S6 (green) | JetLM/SDAR-4B-Chat-b32 | Qwen3-4B [diff] | 4B [diff] | Same as S0 | Same as S0 | Qwen3 | 32 | Only size changes (upward) | not run (over the 3B-active budget) |
| S7 | nvidia/Nemotron-Labs-Diffusion-3B | Ministral-3B [diff] | 3.83B [diff] | AR + diffusion pretraining (1.3T) + SFT [diff] | NVIDIA mix [diff] | Mistral, needs bridge [diff] | 32 | Strong other-family drafter | not wired |

## Claimed scores by the authors (their settings, not ours), % — Table 3 of the Notion page
| ID | GSM8K | MATH | HumanEval | MBPP | GPQA-D | IFEval |
|---|---|---|---|---|---|---|
| Target SDAR-30B-A3B (block 4) | 91.4 | 77.8 | 87.2 | 71.6 | 36.7 | 60.6 |
| S0 SDAR-1.7B (block 4) | 80.1 | 63.2 | 61.6 | 61.1 | 29.8 | 43.4 |
| F1 Fast-dLLM v2 1.5B | 62.0 | 38.1 (full MATH) | 43.9 | 50.0 | 27.7 | 47.0 |
| F2 CoDA-v0-Instruct | – | – | 54.3 | 47.2 | – | – |
| S1 Qwen3-0.6B bd3lm | 46.6 | 13.9 (full MATH) | 47.6 | 32.0 | – | – |

## Why these two were chosen (repo doc docs/PHASE2_DRAFTER_SWAP.md, decided 2026-10-05)
- F2 (CoDA-1.7B) = the CORPUS question. Same Qwen3-1.7B base, same size, same tokenizer and same cost per forward (364.64) as S0; only the training corpus differs (code-heavy public corpus, 179B + 21B tokens). Caveat: it is a full-sequence model (no KV cache, recomputes the context each draft step), so read its results as a pattern across domains, not absolute levels. F1 (Fast-dLLM v2) was the fallback if F2 did not fit; F2 fit (agreement ratio ~0.76 of S0's), so F1 was not run.
- S1 (Qwen3-0.6B bd3lm) = the SIZE question. The only drafter below 1.7B that covers all seven benchmarks with the same Qwen3 tokenizer and a block-32 cache like S0; it costs 126.32 per forward vs 364.64 (2.9x cheaper). Pre-run best case on HumanEval: 1.39x charged speedup vs S0's 1.24x; break-even if the MoE's experts per token rise by less than ~13.6%. It changes size AND corpus, so its result means "this 0.6B drafter", not size alone.
- Not run: S6 SDAR-4B (more than the MoE's 3B active, best case ~1.01x), S2 (code-only sets), and every candidate needing a tokenizer bridge or block-4 grid.
