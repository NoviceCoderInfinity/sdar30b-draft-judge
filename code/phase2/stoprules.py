"""Stop rules for every Phase 2 stage, checked after every unit of work.

The rules live in `code/phase2/rules.json` (fixed before any run; its sha256 goes into every output). A stage
keeps a dict of running metrics and calls `StopMonitor.check(metrics, counts)` after each drafted item, decoded
item, optimizer step or probe. A rule that fires is appended to `<out>/stop_events.jsonl`; a `stop` rule also
writes `<out>/STOPPED.json` and makes `check` return a Decision with `stop=True` -- the caller then saves what it
has and exits with STOP_EXIT (3), which the orchestrator reads as "stopped by rule", not "crashed".

A file named STOP in the run directory or in logs/phase2/ is a manual kill switch: every stage stops at its next
check.

    mon = StopMonitor("decode", out_dir, refs={"s0_agreement": 0.555})
    d = mon.check({"truncated_rate": 0.1, "draft_agreement": 0.52}, counts={"items": 40})
    if d.stop: ...; sys.exit(STOP_EXIT)
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
RULES_PATH = os.path.join(HERE, "rules.json")
GLOBAL_STOP = os.path.join(ROOT, "logs", "phase2", "STOP")
STOP_EXIT = 3


def load_rules(path: str = RULES_PATH) -> dict:
    raw = open(path, "rb").read()
    rules = json.loads(raw)
    rules["_sha256"] = hashlib.sha256(raw).hexdigest()
    return rules


def rules_sha(path: str = RULES_PATH) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def _nonfinite(x) -> bool:
    try:
        return not math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def _cmp(op: str, x: float, thr: float) -> bool:
    if op == "<":
        return x < thr
    if op == ">":
        return x > thr
    if op == "<=":
        return x <= thr
    if op == ">=":
        return x >= thr
    raise ValueError(f"unknown op {op}")


@dataclass
class Decision:
    stop: bool = False
    fired: List[dict] = field(default_factory=list)

    @property
    def reason(self) -> str:
        s = [f["id"] for f in self.fired if f["action"] == "stop"]
        return ",".join(s)


class StopMonitor:
    """One stage's rules. `refs` holds reference values (S0's agreement, S0's seconds per item, ...)."""

    def __init__(self, stage: str, out_dir: str, refs: Optional[Dict[str, float]] = None,
                 rules_path: str = RULES_PATH, rank: int = 0, context: Optional[dict] = None):
        self.stage = stage
        self.out = out_dir
        self.rank = rank
        rules = load_rules(rules_path)
        self.sha = rules["_sha256"]
        self.rules = [r for r in rules["stop"].get(stage, [])]
        self.refs = dict(refs or {})
        self.context = dict(context or {})
        self._streak: Dict[str, int] = {}
        self._warned_ref: set = set()
        self._last_fire: Dict[str, float] = {}
        self.stopped: Optional[dict] = None
        os.makedirs(out_dir, exist_ok=True)

    # -- plumbing ---------------------------------------------------------------------------------------------
    def _event(self, row: dict) -> None:
        if self.rank != 0:
            return
        row = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "stage": self.stage,
               "rules_sha256": self.sha, **row}
        with open(os.path.join(self.out, "stop_events.jsonl"), "a") as fh:
            fh.write(json.dumps(row) + "\n")

    def kill_switch(self) -> Optional[str]:
        for p in (os.path.join(self.out, "STOP"), GLOBAL_STOP):
            if os.path.exists(p):
                return p
        return None

    def threshold(self, rule: dict) -> Optional[float]:
        v = float(rule["value"])
        ref = rule.get("ref")
        if not ref:
            return v
        rv = self.refs.get(ref)
        if rv is None or _nonfinite(rv):
            if rule["id"] not in self._warned_ref:
                self._warned_ref.add(rule["id"])
                self._event({"rule": rule["id"], "action": "skipped", "why": f"reference '{ref}' unknown"})
            return None
        return v * float(rv)

    # -- the check ----------------------------------------------------------------------------------------------
    def check(self, metrics: Dict[str, float], counts: Optional[Dict[str, float]] = None) -> Decision:
        counts = dict(counts or {})
        d = Decision()
        ks = self.kill_switch()
        if ks:
            f = {"id": "kill_switch", "action": "stop", "why": f"STOP file {ks}", "metric": None, "value": None}
            d.fired.append(f); d.stop = True
        for r in self.rules:
            m = r["metric"]
            if m not in metrics or metrics[m] is None:
                continue
            x = metrics[m]
            n = counts.get(r.get("count", "n"), counts.get("n", 0))
            if n < float(r.get("min_n", 0)):
                self._streak[r["id"]] = 0
                continue
            if r["op"] == "nonfinite":
                hit = _nonfinite(x)
                thr = None
            else:
                if _nonfinite(x):
                    continue
                thr = self.threshold(r)
                if thr is None:
                    continue
                hit = _cmp(r["op"], float(x), thr)
            self._streak[r["id"]] = self._streak.get(r["id"], 0) + 1 if hit else 0
            if hit and self._streak[r["id"]] >= int(r.get("patience", 1)):
                f = {"id": r["id"], "action": r["action"], "why": r["why"], "metric": m, "value": float(x),
                     "threshold": thr, "count": n}
                d.fired.append(f)
                if r["action"] == "stop":
                    d.stop = True
        for f in d.fired:
            # warnings are logged at most once a minute per rule, stops always
            if f["action"] == "warn" and time.time() - self._last_fire.get(f["id"], 0.0) < 60.0:
                continue
            self._last_fire[f["id"]] = time.time()
            self._event({**f, "counts": counts})
        if d.stop and self.stopped is None:
            self.stopped = {"stage": self.stage, "reason": d.reason, "fired": d.fired, "counts": counts,
                            "metrics": {k: (v if not _nonfinite(v) else str(v)) for k, v in metrics.items()
                                        if isinstance(v, (int, float))},
                            "refs": self.refs, "context": self.context, "rules_sha256": self.sha,
                            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
            if self.rank == 0:
                with open(os.path.join(self.out, "STOPPED.json"), "w") as fh:
                    json.dump(self.stopped, fh, indent=2)
            print(f"[STOP {self.stage}] {d.reason}: " + "; ".join(
                f"{f['id']} ({f.get('metric')}={f.get('value')} vs {f.get('threshold')})" for f in d.fired
                if f["action"] == "stop"), flush=True)
        return d


# ---------------------------------------------------------------------------------------------------------------
# small running statistics shared by the stages
# ---------------------------------------------------------------------------------------------------------------
class RunningMedian:
    """Median of the last `n` values (cheap, exact)."""

    def __init__(self, n: int = 200):
        self.n, self.v = n, []

    def add(self, x: float) -> None:
        if not _nonfinite(x):
            self.v.append(float(x))
            if len(self.v) > self.n:
                self.v.pop(0)

    def median(self) -> Optional[float]:
        if not self.v:
            return None
        s = sorted(self.v)
        k = len(s) // 2
        return s[k] if len(s) % 2 else 0.5 * (s[k - 1] + s[k])


def paired_delta(a: List[int], b: List[int], resamples: int = 2000, seed: int = 0, ci: float = 0.95):
    """Paired bootstrap of mean(a) - mean(b) over shared items (the repo's rule: 2,000 resamples, seed 0)."""
    import numpy as np
    a_, b_ = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    assert a_.shape == b_.shape
    n = len(a_)
    if n == 0:
        return {"n": 0, "delta": float("nan"), "lo": float("nan"), "hi": float("nan")}
    d = a_ - b_
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(resamples, n))
    boots = d[idx].mean(1)
    lo, hi = np.quantile(boots, [(1 - ci) / 2, 1 - (1 - ci) / 2])
    return {"n": int(n), "delta": float(d.mean()), "lo": float(lo), "hi": float(hi)}
