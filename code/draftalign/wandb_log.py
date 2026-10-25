"""Optional Weights & Biases mirror of the trainers' JSONL rows (rank 0 only).

Reads WANDB_API_KEY / WANDB_ENTITY / ENABLE_WANDB / DA_WANDB_PROJECT from the repo's `.env`
(values are never printed).  NEVER raises: no package, no key, no network or ENABLE_WANDB=0 all
degrade to a silent no-op, so a training run cannot die because of logging.  Online first; if the
server cannot be reached the run is written offline under `<out>/wandb` (push later with
`wandb sync <out>/wandb/offline-run-*`).

    from draftalign import wandb_log
    wb = wandb_log.init("t2b_union_flag", cfg, out, group="round2b", job_type="judge", rank=rank)
    wb.log({"train/loss": 1.23}, step=1058)     # non-numeric values are dropped
    wb.finish()
"""
from __future__ import annotations

import os
import re
import sys
from typing import Any, Dict, Iterable, Optional

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ENV_FILE = os.path.join(REPO, ".env")
DEFAULT_PROJECT = "sdar-draftalign"
_KEYS = ("WANDB_API_KEY", "WANDB_ENTITY", "ENABLE_WANDB", "DA_WANDB_PROJECT")


def read_env(path: str = ENV_FILE) -> Dict[str, str]:
    """The KEY=VALUE pairs of `.env` we care about (quotes stripped, `export` allowed).
    A value already present in os.environ wins over the file."""
    got: Dict[str, str] = {}
    try:
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.strip().removeprefix("export ").strip()
                if k in _KEYS:
                    got[k] = v.strip().strip('"').strip("'")
    except OSError:
        pass
    for k in _KEYS:
        if os.environ.get(k):
            got[k] = os.environ[k]
    return got


def _num(row: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in row.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}


class Noop:
    active = False
    url: Optional[str] = None

    def log(self, row: Dict[str, Any], step: Optional[int] = None) -> None:
        pass

    def finish(self) -> None:
        pass


class Run:
    active = True

    def __init__(self, run, mode: str):
        self.run, self.mode = run, mode
        self.url = getattr(run, "url", None) if mode == "online" else None

    def log(self, row: Dict[str, Any], step: Optional[int] = None) -> None:
        try:
            self.run.log(_num(row), step=step)
        except Exception as e:                                   # noqa: BLE001
            print(f"[wandb] log failed: {e}", file=sys.stderr, flush=True)

    def finish(self) -> None:
        try:
            self.run.finish()
        except Exception as e:                                   # noqa: BLE001
            print(f"[wandb] finish failed: {e}", file=sys.stderr, flush=True)


def init(name: str, config: Dict[str, Any], out_dir: str, *, group: Optional[str] = None,
         job_type: Optional[str] = None, tags: Iterable[str] = (), rank: int = 0,
         notes: Optional[str] = None):
    """A `Run` on rank 0 when W&B is usable, otherwise a `Noop`.  Never raises."""
    if rank != 0:
        return Noop()
    env = read_env()
    if env.get("ENABLE_WANDB", "1").lower() not in ("1", "true", "yes", "on"):
        print("[wandb] disabled by ENABLE_WANDB", flush=True)
        return Noop()
    if not env.get("WANDB_API_KEY"):
        print("[wandb] no WANDB_API_KEY in .env or the environment; logging to JSONL only", flush=True)
        return Noop()
    try:
        import wandb                                              # noqa: WPS433
    except Exception as e:                                        # noqa: BLE001
        print(f"[wandb] package unavailable ({e}); logging to JSONL only", flush=True)
        return Noop()
    os.environ["WANDB_API_KEY"] = env["WANDB_API_KEY"]
    os.environ.setdefault("WANDB_SILENT", "true")
    project = env.get("DA_WANDB_PROJECT") or DEFAULT_PROJECT
    entity = env.get("WANDB_ENTITY") or None
    run_id = re.sub(r"[^A-Za-z0-9_.-]", "-", name)[:64]          # stable id: a relaunch resumes the same run
    kw: Dict[str, Any] = dict(project=project, entity=entity, name=name, id=run_id, resume="allow",
                              group=group, job_type=job_type, tags=list(tags) or None, notes=notes,
                              config=config, dir=out_dir)
    os.makedirs(out_dir, exist_ok=True)
    for mode in ("online", "offline"):
        try:
            run = wandb.init(mode=mode, settings=wandb.Settings(init_timeout=120), **kw)
            where = run.url if mode == "online" else os.path.join(out_dir, "wandb")
            print(f"[wandb] {mode} run {entity or ''}/{project}/{name} -> {where}", flush=True)
            return Run(run, mode)
        except Exception as e:                                    # noqa: BLE001
            print(f"[wandb] {mode} init failed: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
    return Noop()


def prefixed(row: Dict[str, Any], prefix: str) -> Dict[str, Any]:
    """`{"loss": 1}` -> `{"train/loss": 1}`; `step` is kept unprefixed so panels can use it as x."""
    return {("step" if k == "step" else f"{prefix}/{k}"): v for k, v in row.items()}
