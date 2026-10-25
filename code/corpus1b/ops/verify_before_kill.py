"""Pre-destroy checks (2026-09-29, box ends 18:45 IST). Exit 0 only if everything that matters is off the box:
final checkpoint on the HF model repo (sha256 match), checkpointB eval summary + training probe on the HF dataset
repo (size match), git branch pushed. Reads HF with the token from .env; prints no secrets."""
import hashlib, json, os, subprocess, sys
from huggingface_hub import HfApi
ROOT = "/workspace/dmge"; os.chdir(ROOT)
env = dict(l.strip().split("=", 1) for l in open(".env") if "=" in l and not l.startswith("#"))
tok = env.get("HF_TOKEN") or env.get("HUGGING_FACE_HUB_TOKEN") or None
api = HfApi(token=tok); ok = True
def res(name, good, info=""):
    global ok; ok &= bool(good); print(f"{'PASS' if good else 'FAIL'} {name} {info}", flush=True)
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()
# 1. final checkpoint
try:
    st = json.load(open("logs/c1b/train/main/final/trainer_state.json"))
    files = {s.rfilename: s for s in api.model_info(env["HF_CKPT_REPO"], files_metadata=True).siblings}
    hub = files.get("main/final/adapter.safetensors")
    loc = sha("logs/c1b/train/main/final/adapter.safetensors")
    res("final checkpoint on HF", hub and hub.lfs and hub.lfs.sha256 == loc and st["tokens"] > 24e6,
        f"tokens={st['tokens']/1e6:.2f}M sha={loc[:12]}")
    res("final trainer_state.pt on HF", "main/final/trainer_state.pt" in files)
except Exception as e: res("final checkpoint on HF", False, repr(e)[:200])
# 2. eval summary + probe on the dataset repo
try:
    ds = {s.rfilename: s.size for s in api.dataset_info(env["HF_PUSH_REPO"], files_metadata=True).siblings}
    for p in ["logs/c1b/eval/checkpointB/summary.md", "logs/c1b/train/main/probe.jsonl"]:
        m = [k for k in ds if k.endswith(p)]
        res(f"{p} on HF", os.path.exists(p) and m and ds[m[0]] == os.path.getsize(p), m[0] if m else "missing")
except Exception as e: res("dataset repo", False, repr(e)[:200])
# 3. git pushed
try:
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    helper = '!f(){ echo username=x-access-token; echo "password=$GITHUB_TOKEN"; }; f'
    remote = subprocess.check_output(["git", "-c", f"credential.helper={helper}", "ls-remote", "origin",
                                      "refs/heads/phase1-datacorpus"], text=True, env={**os.environ, **env}).split()[0]
    dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], text=True).strip()
    res("git pushed", head == remote and not dirty, f"head={head[:8]} remote={remote[:8]} dirty={bool(dirty)}")
except Exception as e: res("git pushed", False, repr(e)[:200])
print("ALL PASS" if ok else "NOT SAFE", flush=True); sys.exit(0 if ok else 1)
