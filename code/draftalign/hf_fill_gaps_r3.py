#!/usr/bin/env python
"""Make the PUBLIC Anupam-Rawat-IITB HF dataset self-contained for every round-3 comparison (user, 2026-09-15 16:20 IST).

  Nemotron-split-moe-train/round2_t2b_union_flag/           round-2 (GSM8K-heavy) published judge: step003057 + config + train log
  Nemotron-split-moe-train/reference_evals/da_decode/<cell>/ stock / stock+DES32/48 / blind / round-2 judge(+DES) decode records
  Nemotron-split-moe-train/reference_evals/research_v1/      GSM8K-500 stock records (the stock reference for GSM8K)
  Nemotron-split-moe-train/r3_B/analysis/                    training calibration evals + plots, live results summary, report html
  README.md (repo root)                                      index: architecture -> weights / eval records / logs

Idempotent (skips a path already on the Hub with the same size). Runs as the logged-in account; .env's HF_TOKEN is popped.
    /venv/r3/bin/python code/draftalign/hf_fill_gaps_r3.py [--only analysis,readme]
"""
import argparse, glob, hashlib, io, os, sys, time

os.environ.pop("HF_TOKEN", None)
os.environ["HF_HUB_OFFLINE"] = "0"
from huggingface_hub import CommitOperationAdd, HfApi  # noqa: E402

REPO = "Anupam-Rawat-IITB/dlm-exp1-drafter-voting-research-artifacts"
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MIR = "/dev/shm/dlm/artifacts_mirror/logs"
SPLIT = "Nemotron-split-moe-train"
HF = f"https://huggingface.co/datasets/{REPO}/tree/main"
T2B_SHA = "734af65681ab6bb8"
DS = ("humaneval", "mbpp", "gsm8k", "math500", "gpqa", "aime")
ARMS = ("stock", "stock_p32desf", "stock_p48desf", "dv-s8_blind_q50", "dv-s8_full_c95_ad734af656",
        "dv-s8_full_c95_p32desf_ad734af656", "dv-s8_full_c95_p48desf_ad734af656")


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def commit(api, remote, pairs, msg):
    ops = [CommitOperationAdd(path_in_repo=dst, path_or_fileobj=src) for src, dst in pairs
           if not (isinstance(src, str) and remote.get(dst) == os.path.getsize(src))]
    if not ops:
        print(f"[fill] {msg}: nothing new ({len(pairs)} files already on the Hub)", flush=True)
        return
    for i in range(0, len(ops), 400):
        api.create_commit(REPO, repo_type="dataset", operations=ops[i:i + 400], commit_message=f"{msg} ({i // 400 + 1})")
    print(f"[fill] {msg}: committed {len(ops)} files", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="t2b,refs,analysis,readme")
    only = set(ap.parse_args().only.split(","))
    api = HfApi()
    assert api.whoami()["name"] == "Anupam-Rawat-IITB", "not logged in as Anupam-Rawat-IITB"
    remote = {f.path: getattr(f, "size", None) for f in api.list_repo_tree(REPO, repo_type="dataset", recursive=True)
              if hasattr(f, "size")}

    if "t2b" in only:
        src = f"{MIR}/draftalign/train/t2b_union_flag"
        assert sha256(f"{src}/step003057/adapter.safetensors").startswith(T2B_SHA), "round-2 judge sha mismatch"
        pairs = [(f"{src}/step003057/{f}", f"{SPLIT}/round2_t2b_union_flag/step003057/{f}")
                 for f in ("adapter.safetensors", "adapter_config.json", "trainer_state.pt")]
        pairs += [(f"{src}/{f}", f"{SPLIT}/round2_t2b_union_flag/train_logs/{f}") for f in ("config.json", "train.rank0.jsonl")
                  if os.path.exists(f"{src}/{f}")]
        readme = (f"# round2_t2b_union_flag — the round-2 published judge (GSM8K-heavy union curriculum)\n\n"
                  f"LoRA r32/α64 on q,k,v,o + draft flag on `JetLM/SDAR-30B-A3B-Chat-b32`; `step003057/adapter.safetensors` sha256 `{T2B_SHA}…`\n"
                  "(decode-record tag `_ad734af656`). Copied from `Arushhh/dlm-exp1-drafter-voting-research-artifacts:"
                  "logs/draftalign/train/t2b_union_flag/` so every compared model lives in this public repo.\n"
                  f"Its eval records: [`reference_evals/da_decode`]({HF}/{SPLIT}/reference_evals/da_decode) (cells `*_dv-s8_full_c95*_ad734af656`).\n")
        pairs.append((io.BytesIO(readme.encode()), f"{SPLIT}/round2_t2b_union_flag/README.md"))
        commit(api, remote, pairs, "round-2 judge t2b_union_flag step003057 (copied from the team repo)")
        info = api.get_paths_info(REPO, [f"{SPLIT}/round2_t2b_union_flag/step003057/adapter.safetensors"], repo_type="dataset")[0]
        assert info.lfs and info.lfs.sha256.startswith(T2B_SHA), "round-2 judge sha mismatch on the Hub"
        print(f"[fill] round-2 judge verified on the Hub: sha {info.lfs.sha256[:16]}", flush=True)

    if "refs" in only:
        pairs = []
        for ds in DS:
            for arm in ARMS:
                cell = f"{ds}_B32_t95_{arm}"
                for f in sorted(glob.glob(f"{MIR}/draftalign/da_decode/{cell}/{cell}.*.shard*")):
                    if ".oldcfg" in f or not (f.endswith(".jsonl") or f.endswith(".jsonl.done")):
                        continue
                    pairs.append((f, f"{SPLIT}/reference_evals/da_decode/{cell}/{os.path.basename(f)}"))
        for f in sorted(glob.glob(f"{MIR}/research_v1/baselines_b32/gsm8k_B32_t95_stock.gsm8k500.shard*")):
            pairs.append((f, f"{SPLIT}/reference_evals/research_v1/baselines_b32/{os.path.basename(f)}"))
        readme = ("# reference_evals — decode records every round-3 comparison is paired against\n\n"
                  "Cells `<set>_B32_t95_<arm>`: `stock`, `stock_p32desf` / `stock_p48desf` (stock + DES32/48), `dv-s8_blind_q50` "
                  "(blind accept, no training), `dv-s8_full_c95[_p32desf|_p48desf]_ad734af656` (round-2 judge [+ DES]).\n"
                  "One JSON record per item (`correct`, `n_forward`, `unique_expert_layer_loads`, `n_gen_to_eos`, `seconds`, …); "
                  "metrics as in `code/draftalign/eval_summary_r3B.py`. New round-3 stock + DES records (HumanEval, AIME, GPQA) are in "
                  f"[`r3_A/evals/stock`]({HF}/{SPLIT}/r3_A/evals/stock).\n")
        pairs.append((io.BytesIO(readme.encode()), f"{SPLIT}/reference_evals/README.md"))
        commit(api, remote, pairs, "reference eval records (stock, stock+DES, blind, round-2 judge)")

    if "analysis" in only:
        pairs = [(f, f"{SPLIT}/r3_B/analysis/calib/{os.path.basename(f)}")
                 for f in sorted(glob.glob(f"{ROOT}/logs/draftalign/train/r3_B/calib/*")) if os.path.isfile(f)]
        pairs += [(f, f"{SPLIT}/r3_B/analysis/results/{os.path.relpath(f, ROOT + '/docs/results/r3_B')}")
                  for f in sorted(glob.glob(f"{ROOT}/docs/results/r3_B/**/*", recursive=True)) if os.path.isfile(f)]
        for f in ("docs/report/r3_report.html", "docs/MODELS_AND_RESULTS_2026-09-15.md"):
            if os.path.exists(f"{ROOT}/{f}"):
                pairs.append((f"{ROOT}/{f}", f"{SPLIT}/r3_B/analysis/{os.path.basename(f)}"))
        remote_now = dict(remote)
        for _, dst in pairs:            # results/summary files change every pass: always re-upload them
            if "/results/" in dst or dst.endswith(".html") or dst.endswith(".md"):
                remote_now.pop(dst, None)
        commit(api, remote_now, pairs, "r3_B analysis: calibration evals, results summary, report")

    if "readme" in only:
        rows = [
            ("stock SDAR-30B-A3B-Chat-b32 (+ 1.7B drafter where used)", "`JetLM/SDAR-30B-A3B-Chat-b32` @ `c351bbc3…` (Hugging Face)",
             f"[reference_evals/da_decode]({HF}/{SPLIT}/reference_evals/da_decode) (`*_stock`), [r3_A/evals/stock]({HF}/{SPLIT}/r3_A/evals/stock)"),
            ("stock + DES32 / DES48 (decode-time expert pooling, no weights)", "—",
             f"[reference_evals/da_decode]({HF}/{SPLIT}/reference_evals/da_decode) (`*_stock_p32desf`, `*_stock_p48desf`), [r3_A/evals/stock]({HF}/{SPLIT}/r3_A/evals/stock)"),
            ("round-2 judge, GSM8K-heavy (t2b_union_flag step 3057)", f"[round2_t2b_union_flag/step003057]({HF}/{SPLIT}/round2_t2b_union_flag/step003057) · sha `{T2B_SHA}…`",
             f"[reference_evals/da_decode]({HF}/{SPLIT}/reference_evals/da_decode) (`*_ad734af656`)"),
            ("r3_A — judge LoRA, CE + certainty (β 2), no calibration; stopped at step 2000 (2.82 epochs)",
             f"[r3_A/checkpoints]({HF}/{SPLIT}/r3_A/checkpoints): epoch 1 `step000700` (sha `8f3b73a4…`), epoch 2 `step001400` (`2e99a87b…`), last `step002000` (`d4e694e5…`)",
             f"[r3_A/evals]({HF}/{SPLIT}/r3_A/evals) · logs [r3_A/train_logs]({HF}/{SPLIT}/r3_A/train_logs)"),
            ("r3_B — calibrated recipe (β 0.5 + calibration BCE + override weights 2/3), from r3_A step 700, 710 steps → epoch 2",
             f"[r3_B/checkpoints]({HF}/{SPLIT}/r3_B/checkpoints): final `step000710` (sha `ad8a6485…`), every 100 steps",
             f"[r3_B/evals]({HF}/{SPLIT}/r3_B/evals) · calibration + results [r3_B/analysis]({HF}/{SPLIT}/r3_B/analysis) · logs [r3_B/train_logs]({HF}/{SPLIT}/r3_B/train_logs)"),
        ]
        body = ("# dlm-exp1 drafter-voting research artifacts (public)\n\n"
                "Draft-aligned judge LoRA for `SDAR-30B-A3B-Chat-b32` with a 1.7B drafter and DES expert pooling: goal = decode speedup "
                "with minimal accuracy loss vs the stock model.\n\n"
                "| model / setting | weights | eval records |\n|---|---|---|\n"
                + "\n".join(f"| {a} | {b} | {c} |" for a, b, c in rows) +
                f"\n\nTraining data (processed + raw, strategy docs, code as run): [Nemotron-data-subset-training-diffusion-moe]({HF}/Nemotron-data-subset-training-diffusion-moe).\n"
                "Code: GitHub `NoviceCoderInfinity/diffusion-moe-general-expert`, branch `round3-2026-09-14` "
                "(results table: `docs/results/r3_B/README.md`; model index: `docs/MODELS_AND_RESULTS_2026-09-15.md`).\n"
                "W&B: https://wandb.ai/arushsharma536-spe-iit/sdar-draftalign (runs `r3_A`, `r3_B`).\n\n"
                f"_Updated {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}_\n")
        api.create_commit(REPO, repo_type="dataset", commit_message="repo index README",
                          operations=[CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=io.BytesIO(body.encode()))])
        print("[fill] root README index committed", flush=True)


if __name__ == "__main__":
    main()
