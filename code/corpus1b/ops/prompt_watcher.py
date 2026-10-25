"""Build each domain's prompt list as soon as its sources + every decontamination eval set are downloaded."""
import glob, json, os, re, subprocess, sys, time
ROOT = "/workspace/dmge"; os.chdir(ROOT)
sys.path.insert(0, "code/corpus1b")
import sources as S
S1 = "logs/c1b/stage1"
NEED = {d: {s["tag"] for s in S.SOURCES if s["domain"] == d} for d in ("maths", "science", "code", "if", "chat")}
EVAL = {e["tag"] for e in S.EVAL_SETS} | {h["tag"] for h in S.HELD_OUT}
OPTIONAL = {e["tag"] for e in S.EVAL_SETS if e.get("optional")}
ORDER = ["maths", "science", "chat", "if", "code"]          # launch whichever is ready; if waits for chat
WORKERS = int(os.environ.get("GW", "60"))

def tags_ok():
    ok = set()
    for log in glob.glob("logs/c1b/dl/dl_*.log") + ["logs/c1b/setup.log"]:
        cur = None
        for line in open(log, errors="ignore").read().replace("\r", "\n").splitlines():
            m = re.match(r"^\[download\] (\S+) \((source|eval|held_out)\)", line)
            if m: cur = m.group(1)
            elif cur and re.match(r"^\[download\]\s+ok\b", line): ok.add(cur); cur = None
    return ok

def merge_manifests():
    out = "data/c1b/sources_manifest.json"
    merged = json.load(open(out)) if os.path.exists(out) else {}
    for p in sorted(glob.glob("data/c1b/dl_parts/manifest_*.json")):
        try: part = json.load(open(p))
        except ValueError: continue
        for k, v in part.items():
            if isinstance(v, dict) and isinstance(merged.get(k), dict): merged[k].update(v)
            elif k not in merged: merged[k] = v
    json.dump(merged, open(out + ".tmp", "w"), indent=1, sort_keys=True); os.replace(out + ".tmp", out)

running = {}
while True:
    ok = tags_ok()
    for d in ORDER:
        if d in running or os.path.exists(f"{S1}/.done_prompts_{d}"):
            continue
        missing = (NEED[d] | (EVAL - OPTIONAL)) - ok
        if d == "if" and not os.path.exists(f"{S1}/.done_prompts_chat"):
            missing |= {"<chat list>"}
        if missing:
            continue
        merge_manifests()
        log = open(f"{S1}/prompt_lists_{d}.log", "a")
        print(f"[watcher {time.strftime('%H:%M:%S')}] build {d}", flush=True)
        running[d] = subprocess.Popen(["/venv/train/bin/python", "code/corpus1b/build_prompt_lists.py", "--domains", d,
                                       "--workers", str(WORKERS)], stdout=log, stderr=subprocess.STDOUT)
    for d, p in list(running.items()):
        rc = p.poll()
        if rc is None: continue
        del running[d]
        if rc == 0:
            open(f"{S1}/.done_prompts_{d}", "w").close(); print(f"[watcher {time.strftime('%H:%M:%S')}] {d} DONE", flush=True)
        else:
            print(f"[watcher {time.strftime('%H:%M:%S')}] {d} FAILED rc={rc} (see {S1}/prompt_lists_{d}.log)", flush=True)
            open(f"{S1}/.failed_prompts_{d}", "w").close()
    if all(os.path.exists(f"{S1}/.done_prompts_{d}") for d in NEED) and not running:
        rc = subprocess.call("/venv/train/bin/python code/corpus1b/spotcheck.py > logs/c1b/stage1/spotcheck.log 2>&1", shell=True)
        if rc == 0: open(f"{S1}/.done_prompts", "w").close()
        print(f"[watcher] all domains built; spotcheck rc={rc}", flush=True); break
    if any(os.path.exists(f"{S1}/.failed_prompts_{d}") for d in NEED) and not running and \
       all(os.path.exists(f"{S1}/.done_prompts_{d}") or os.path.exists(f"{S1}/.failed_prompts_{d}") for d in NEED):
        print("[watcher] stopping: some domains failed", flush=True); break
    time.sleep(20)
