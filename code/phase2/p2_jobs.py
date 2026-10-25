#!/usr/bin/env python
"""Run a list of GPU jobs, one per free GPU, and wait for all of them (Phase 2 stages).

    python code/phase2/p2_jobs.py --jobs logs/phase2/jobs/F1_decode.jsonl --log logs/phase2/jobs/F1_decode.log \
        [--gpus 0,1,2,3,4,5,6,7]

Each line of --jobs is {"name", "group", "cmd"} where cmd contains {GPU}. Every job runs with cwd = its own
throwaway sandbox under /dev/shm/p2_sandbox/<name> (generated code is executed by the code scorers: AGENTS.md 9(p));
all repository paths in cmd must be absolute or resolve through safe_out. A GPU counts as free when it holds
< 1000 MiB and no job of this runner is on it.

Stop handling, checked every poll:
  * exit 3 (a stop rule fired): every other job of the SAME group is terminated (SIGINT, then SIGTERM) and its
    queued jobs are skipped -- the cell stops as a whole; other groups go on.
  * a file STOP in the log's directory or logs/phase2/STOP: no new job starts; running jobs finish their current
    item and stop at their next check (every Phase 2 process reads the same STOP file).
  * P2_DEADLINE (epoch seconds) in the environment: after it, no new job starts.
The summary (<log>.summary.json) lists every job with its exit code: 0 done, 3 stopped by rule, other = crashed,
"skipped" = not started because its group stopped. Exit code: 0 if all 0, 3 if any stopped by rule, else 1.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
GLOBAL_STOP = os.path.join(ROOT, "logs", "phase2", "STOP")


def free_gpus(allowed):
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return []
    free = []
    for line in out.strip().splitlines():
        i, mem = [x.strip() for x in line.split(",")]
        if int(i) in allowed and int(mem) < 1000:
            free.append(int(i))
    return free


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--log", required=True)
    ap.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    ap.add_argument("--poll", type=float, default=15.0)
    ap.add_argument("--sandbox", default="/dev/shm/p2_sandbox")
    ap.add_argument("--assume-free", action="store_true", help="skip the nvidia-smi memory check (tests)")
    args = ap.parse_args(argv)
    allowed = [int(x) for x in args.gpus.split(",")]
    jobs = [json.loads(l) for l in open(args.jobs) if l.strip()]
    os.makedirs(os.path.dirname(os.path.abspath(args.log)), exist_ok=True)
    local_stop = os.path.join(os.path.dirname(os.path.abspath(args.log)), "STOP")
    logf = open(args.log, "a")

    def say(msg):
        line = f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {msg}"
        print(line, flush=True)
        logf.write(line + "\n"); logf.flush()

    queue = list(jobs)
    running = {}                    # gpu -> (job, Popen, logfile handle)
    result = {}
    stopped_groups = set()
    say(f"{len(queue)} jobs on GPUs {allowed}")
    while queue or running:
        # reap
        for g, (job, pr, fh) in list(running.items()):
            rc = pr.poll()
            if rc is None:
                continue
            fh.close()
            del running[g]
            result[job["name"]] = rc
            say(f"GPU {g} {job['name']} exit {rc}")
            if rc == 3 and job.get("group"):
                stopped_groups.add(job["group"])
                say(f"group {job['group']} STOPPED BY RULE: stopping its other jobs")
                for g2, (j2, p2, _) in running.items():
                    if j2.get("group") == job["group"]:
                        try:
                            os.killpg(p2.pid, signal.SIGINT)          # the job's whole session (bash -c + python)
                        except ProcessLookupError:
                            pass
        # skip queued jobs of stopped groups
        for job in [j for j in queue if j.get("group") in stopped_groups]:
            queue.remove(job)
            result[job["name"]] = "skipped"
        dl = float(os.environ.get("P2_DEADLINE", "0") or 0)          # epoch seconds: start nothing after it
        late = bool(dl) and time.time() >= dl
        halt = os.path.exists(local_stop) or os.path.exists(GLOBAL_STOP) or late
        if halt and queue:
            say(f"{'DEADLINE passed' if late else 'STOP file present'}: {len(queue)} queued jobs not started")
            for job in queue:
                result[job["name"]] = "skipped"
            queue = []
        # launch
        for g in (allowed if args.assume_free else free_gpus(allowed)):
            if not queue:
                break
            if g in running:
                continue
            job = queue.pop(0)
            box = os.path.join(args.sandbox, job["name"])
            os.makedirs(box, exist_ok=True)
            cmd = job["cmd"].replace("{GPU}", str(g))
            jl = os.path.join(os.path.dirname(os.path.abspath(args.log)), f"{job['name']}.log")
            fh = open(jl, "a")
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g))
            pr = subprocess.Popen(["bash", "-c", cmd], cwd=box, stdout=fh, stderr=subprocess.STDOUT, env=env,
                                  start_new_session=True)
            running[g] = (job, pr, fh)
            say(f"GPU {g} start {job['name']} pid {pr.pid} -> {jl}")
        time.sleep(args.poll)
    json.dump({"jobs": result, "stopped_groups": sorted(stopped_groups)}, open(args.log + ".summary.json", "w"),
              indent=2)
    codes = list(result.values())
    if all(c == 0 for c in codes):
        return 0
    if any(c == 3 for c in codes) or stopped_groups:
        return 3
    return 1


if __name__ == "__main__":
    sys.exit(main())
