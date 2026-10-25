#!/usr/bin/env python
"""Pre-shutdown check for the 128M -> 228M continuation. Exit 0 and print ALL PASS only if everything that matters is
off the box: the newest main_228m checkpoint on the HF model repo (sha256 match), the 228M training index, the wave-B
index and the eval summary on the HF dataset repo (size match), and the git branch pushed. Prints no secrets."""
import glob
import hashlib
import json
import os
import subprocess
import sys

from huggingface_hub import HfApi

env = dict(l.strip().split("=", 1) for l in open(".env") if "=" in l and not l.lstrip().startswith("#"))
api = HfApi(token=env.get("HF_TOKEN"))
ok = True


def res(name, good, info=""):
    global ok
    ok &= bool(good)
    print(f"{'PASS' if good else 'FAIL'} {name} {info}", flush=True)


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


try:                                                                  # 1. newest checkpoint on the model repo
    c = [(json.load(open(f))["tokens"], os.path.dirname(f)) for f in glob.glob("logs/c1b/train/main_228m/*/trainer_state.json")
         if not f.split("/")[-2].startswith("eval_")]
    tokens, d = max(c)
    name = os.path.basename(d)
    files = {s.rfilename: s for s in api.model_info(env["HF_CKPT_REPO"], files_metadata=True).siblings}
    hub = files.get(f"main_228m/{name}/adapter.safetensors")
    res("newest checkpoint on HF", hub is not None and hub.lfs and hub.lfs.sha256 == sha(f"{d}/adapter.safetensors"),
        f"main_228m/{name} at {tokens / 1e6:.2f}M tokens")
    res("its trainer_state.pt on HF", f"main_228m/{name}/trainer_state.pt" in files)
except Exception as e:                                                # noqa: BLE001
    res("newest checkpoint on HF", False, repr(e)[:200])
try:                                                                  # 2. index + eval summary on the dataset repo
    ds = {s.rfilename: s.size for s in api.dataset_info(env["HF_PUSH_REPO"], files_metadata=True).siblings}
    want = ["logs/c1b/trainset/index_228m.jsonl", "logs/c1b/trainset/waveB/index.jsonl"]
    want += [p for p in ("logs/c1b/eval/checkpoint228M_nopool/summary.md", "logs/c1b/eval/checkpoint228M_nopool/compare.md")
             if os.path.exists(p)]
    for p in want:
        res(f"{p} on HF", p in ds and ds[p] == os.path.getsize(p), f"{ds.get(p)} vs local {os.path.getsize(p)}")
except Exception as e:                                                # noqa: BLE001
    res("dataset repo", False, repr(e)[:200])
try:                                                                  # 3. git pushed
    br = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True).strip()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    helper = '!f(){ echo username=x-access-token; echo "password=$GITHUB_TOKEN"; }; f'
    remote = subprocess.check_output(["git", "-c", f"credential.helper={helper}", "ls-remote", "origin", f"refs/heads/{br}"],
                                     text=True, env={**os.environ, **env}).split()[0]
    res("git pushed", head == remote, f"{br} head={head[:8]} remote={remote[:8]}")
except Exception as e:                                                # noqa: BLE001
    res("git pushed", False, repr(e)[:200])
print("ALL PASS" if ok else "NOT SAFE", flush=True)
sys.exit(0 if ok else 1)
