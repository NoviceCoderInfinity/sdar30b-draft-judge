# Phase 4 (drafter LoRA): where everything is

**One command** gets all of it. Clone this branch and run the fetch script:

```bash
git clone -b phase4-drafter-lora https://github.com/NoviceCoderInfinity/diffusion-moe-general-expert.git
cd diffusion-moe-general-expert
bash code/drafter_lora/fetch_phase4_all.sh                # everything (~8.3 GiB, mostly weights)
# bash code/drafter_lora/fetch_phase4_all.sh --no-weights # everything except checkpoints / adapter / merged drafter (~0.44 GiB)
# bash code/drafter_lora/fetch_phase4_all.sh --dry-run    # list what would be fetched
```

The script needs `git` and `pip install huggingface_hub`. No token is needed, because both backups are public. It puts
the logs branch in `phase4_backup/logs/` and the Hugging Face dataset in `phase4_backup/hf/`.

| What | Where | Link |
|---|---|---|
| Code, results, report, status | GitHub branch `phase4-drafter-lora` (this branch) | https://github.com/NoviceCoderInfinity/diffusion-moe-general-expert/tree/phase4-drafter-lora |
| All logs (no weights) | GitHub branch `phase4-logs` | https://github.com/NoviceCoderInfinity/diffusion-moe-general-expert/tree/phase4-logs |
| Everything incl. weights | Hugging Face dataset `Arushhh/phase4-drafter-lora` (public) | https://huggingface.co/datasets/Arushhh/phase4-drafter-lora/tree/main |
| Training metrics | W&B run `phase4_main`, plus one run per decode shard | https://wandb.ai/arushsharma536-spe-iit/Conscious_Engines/runs/phase4_main |

**This branch holds:**
* `results/phase4_drafter/`:
  * `REPORT.md` (full report; speed on this machine in sections 9c and 9d);
  * `ANALYSIS.md` (first-principles analysis);
  * `FINAL_TABLE.md` and `baseline_tables.md`;
  * `vanilla_same_box.json/.md` (Phase 4 vs the vanilla 30B, both measured here);
  * `phase3_same_box.json/.md` (Phase 3 on this machine);
  * `judge_lora_cost.json/.md`;
  * the gate JSONs and `measured_*.json`;
  * `machine.txt` and `pip_freeze.txt`.
* `code/drafter_lora/`: every script of the run.
* `STATUS.md`: the run log and operator decisions.

**The Hugging Face dataset holds every file under `logs/phase4/` on the run machine**, plus `results/phase4_drafter/` and
`STATUS.md`. Direct file URL:
`https://huggingface.co/datasets/Arushhh/phase4-drafter-lora/resolve/main/<path>`

| Path | Contents |
|---|---|
| `logs/phase4/train/main/final_adapter/` | the trained LoRA adapter (34,865,152 params) |
| `logs/phase4/train/main/merged/` | the merged bf16 SDAR-1.7B drafter (tag `dr3d174340`) |
| `logs/phase4/train/main/ckpt/step_000129` … `step_001285/` | adapter checkpoints every 10%, with optimizer and trainer state |
| `logs/phase4/train/main/` | `config.json`, `train.rank0.jsonl`, `holdout_pids.json`, `check_25pct.json`, `DONE.json` |
| `logs/phase4/decode/{rd1_gate3,rd1,rd2,vanilla_here}/` | every per-question decode record (Gate 3, Gate 4 rd1/rd2, vanilla 30B re-decoded here) |
| `logs/phase4/gate1/`, `logs/phase4/*.out`, `*.log`, `events.tsv` | gate outputs and every stage log |
| `logs/phase4/vanilla_measure/` | the 20-minute measurement snapshots during the vanilla run |
| `logs/phase4/session_scratchpad/` | session work files: diagnosis scripts and outputs (`diag/A`, `B`, `C`), the analysis scripts (`analysis_tools/`), the results-page builder, helper and loop scripts |
| `logs/phase4/ref/` | the Phase 2 and Phase 3 reference records used for comparison |

**Not in any public backup:**
* `/workspace/.env` (secrets).
* The base models. They are public on Hugging Face: SDAR-30B-A3B-Chat-b32 at rev c351bbc3, SDAR-1.7B-Chat-b32 at rev 2c83c05d.
* The frozen 128M judge adapter, which is in the Phase 2 backups.
* The private results page: https://claude.ai/artifact/863c7J9dPKnaVRuJpVs9VP
