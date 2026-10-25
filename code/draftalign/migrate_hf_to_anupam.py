#!/usr/bin/env python
"""Copy the round-3 backups from Arushhh's HF dataset to Anupam Rawat's public HF dataset (user, 2026-09-15).
Folders: Nemotron-data-subset-training-diffusion-moe/ (data; its checkpoints/ subfolder is NOT copied -- checkpoints
live only in Nemotron-split-moe-train/ now) and Nemotron-split-moe-train/ (checkpoints, train logs, evals).
Token: Anupam's token from /root/.cache/huggingface/token. Verifies every file (size + LFS sha256) after upload."""
import os, sys
from concurrent.futures import ThreadPoolExecutor
from huggingface_hub import HfApi, hf_hub_download
SRC = "Arushhh/dlm-exp1-drafter-voting-research-artifacts"
DST = "Anupam-Rawat-IITB/dlm-exp1-drafter-voting-research-artifacts"
FOLDERS = ["Nemotron-data-subset-training-diffusion-moe", "Nemotron-split-moe-train"]
LOCAL = "/dev/shm/dlm/hf_migrate"
tok = open("/root/.cache/huggingface/token").read().strip()
src, dst = HfApi(), HfApi(token=tok)
assert dst.whoami()["name"] == "Anupam-Rawat-IITB"
dst.create_repo(DST, repo_type="dataset", private=False, exist_ok=True)
assert not dst.repo_info(DST, repo_type="dataset").private, "target repo is private"
files = [f for d in FOLDERS for f in src.list_repo_tree(SRC, repo_type="dataset", path_in_repo=d, recursive=True)
         if hasattr(f, "size") and not f.path.startswith(f"{FOLDERS[0]}/checkpoints/")]
print(f"[migrate] {len(files)} files, {sum(f.size for f in files)/1e9:.2f} GB", flush=True)
with ThreadPoolExecutor(16) as ex:
    list(ex.map(lambda f: hf_hub_download(SRC, f.path, repo_type="dataset", local_dir=LOCAL), files))
print("[migrate] downloaded", flush=True)
dst.upload_large_folder(repo_id=DST, repo_type="dataset", folder_path=LOCAL, num_workers=16)
have = {f.path: f for d in FOLDERS for f in dst.list_repo_tree(DST, repo_type="dataset", path_in_repo=d, recursive=True) if hasattr(f, "size")}
bad = [f.path for f in files if f.path not in have or have[f.path].size != f.size
       or (f.lfs and (not have[f.path].lfs or have[f.path].lfs.sha256 != f.lfs.sha256))]
print(f"[migrate] verified {len(files) - len(bad)}/{len(files)} files; mismatches: {bad[:10]}", flush=True)
print(f"[migrate] https://huggingface.co/datasets/{DST}", flush=True)
sys.exit(1 if bad else 0)
