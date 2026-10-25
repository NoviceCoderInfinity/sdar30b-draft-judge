"""Phase 2 per-item hooks for `draftalign/run_da_decode.py`: expert-activation logging, running metrics and the
decode stop rules, checked after EVERY decoded item. Off unless the runner gets --expert-log or --stop-rules, so
every earlier cell decodes, names and hashes exactly as before.

Per item it adds to the record (none of these enter `cfg` / `config_hash`):
    exp_*                     the expert-activation summary (explog.ExpertLogMoEMeter.summary): per forward kind the
                              experts per layer in one forward, the union over the item, load entropy, router top-8
                              mass; the per-layer profile; the 10 most loaded (layer, expert) pairs
    judge_*                   the judge's mean top-1 probability and commit fraction (> 0.95) at each pass of a block
    drafter_key, drafter_eq_per_forward, drafter_eq_tokens, charged_exp_per_token
and writes, per item:
    <out>/experts/<split>/<idx>.npz   per forward, per layer: the experts that ran (bitmask), per-kind token counts,
                                      router top-8 mass; with --expert-log full also every row's top-8 expert ids and
                                      router probabilities
    <out>/judge/<split>/<idx>.npz     every judge read: forward index, pass within the block, and per slot the top-1
                                      probability, argmax, probability of the draft token, draft token, open flag
The decode trace (<out>/traces/<split>/<idx>.npz, as in round 3) keeps the drafter's confidence per slot and the
stock passes' confidences.

Running metrics over the WHOLE cell -- every shard's records so far, read incrementally from the shard files --
(<out>/<dest>.running.jsonl, one row per item of this shard; W&B run/*): accuracy so far, experts per token,
drafter forwards per token, charged experts per token and speedup vs the stock reference (when given), passes per
block, draft agreement, judge commits / tentatives / remasks per block, fallback rate, truncation rate, seconds per
item, and -- against the S0 reference cell -- the paired delta accuracy with its 95% and 99% bootstrap intervals.
"""
from __future__ import annotations

import glob
import json
import os
from typing import Dict, List, Optional

import numpy as np

from phase2.stoprules import STOP_EXIT, StopMonitor, paired_delta, rules_sha


class CellReader:
    """All records of one cell and split across every shard file, read incrementally (only new lines are parsed),
    so the running metrics and the stop rules see the whole cell -- all shards -- not just this process's items."""

    def __init__(self, cell_dir: str, split: str, cell: Optional[str] = None):
        self.dir, self.split = cell_dir, split
        self.pattern = os.path.join(cell_dir, f"{cell or '*'}.{split}.shard*of*.jsonl")
        self.pos: Dict[str, int] = {}
        self.recs: Dict[int, dict] = {}

    def refresh(self) -> Dict[int, dict]:
        for f in sorted(glob.glob(self.pattern)):
            if f.endswith(".running.jsonl"):            # shard*of*.jsonl also matches <shard>.jsonl.running.jsonl
                continue
            with open(f) as fh:
                fh.seek(self.pos.get(f, 0))
                while True:
                    line = fh.readline()
                    if not line or not line.endswith("\n"):
                        break                                   # a partly written last line is read next time
                    self.pos[f] = fh.tell()
                    try:
                        r = json.loads(line)
                    except ValueError:
                        continue
                    self.recs[int(r["idx"])] = _slim(r)
        return self.recs


def _slim(r: dict) -> dict:
    """Only the fields the running metrics use (records carry large per-forward lists)."""
    keep = ("idx", "correct", "score_error", "n_gen_to_eos", "n_blocks", "unique_expert_layer_loads", "n_forward",
            "n_prefill", "n_flush", "drafter_n_by_kind", "draft_agreement_per_block", "committed_judge_per_block",
            "tentative_per_block", "remask_per_block", "full_fallback_per_block", "truncated", "seconds", "ref_match",
            "exp_block_load_entropy", "exp_by_kind", "exp_uniq_per_layer_forward_block", "exp_union_per_layer_item",
            "judge_p_mean_by_pass", "judge_commit_frac_by_pass", "judge_n_passes")
    return {k: r.get(k) for k in keep}


class JudgeCapture:
    """Records every judge read of a decode -- the judge's probabilities, pass by pass -- without changing it.

    `da_decode.judge_full_block` (and the blind rule) turn each judge forward's logits into per-slot numbers through
    the module-level `da_decode.blind_read`; this wraps that function, keeps a CPU copy of what it returns, and
    returns the result unchanged. Per read it keeps: the MoE forward index (so kind / block come from the record's
    `forwards` list), the pass number within its block, and per slot (32): top-1 probability p, argmax, the
    probability of the draft token q, the draft token and whether the slot was read (open)."""

    def __init__(self):
        self.moe = None
        self.rows: List[dict] = []
        self._orig = None

    def install(self, moe):
        from draftalign import da_decode as DA
        self.moe = moe
        if self._orig is not None:
            return
        self._orig = DA.blind_read
        cap = self

        def blind_read(logits, D, read, mask_id):
            r = cap._orig(logits, D, read, mask_id)
            cap.rows.append({"fwd": len(cap.moe.records) - 1 if cap.moe is not None else -1,
                             "p": r["p"].detach().float().cpu().numpy().astype(np.float16),
                             "argmax": r["argmax"].detach().cpu().numpy().astype(np.int32),
                             "q": r["q"].detach().float().cpu().numpy().astype(np.float16),
                             "draft": D.detach().cpu().numpy().astype(np.int32),
                             "open": read.detach().cpu().numpy().astype(bool)})
            return r

        DA.blind_read = blind_read

    def uninstall(self):
        if self._orig is not None:
            from draftalign import da_decode as DA
            DA.blind_read = self._orig
            self._orig = None

    def take(self, forwards: list, tau: float = 0.95) -> dict:
        """Arrays for the npz + per-pass summaries for the record; clears the buffer."""
        rows, self.rows = self.rows, []
        if not rows:
            return {}
        from draftalign.da_decode import MOE_KIND_CODE
        flush = MOE_KIND_CODE["kvflush"]
        # pass number within the block = judge reads since the last kv flush before this forward
        codes = [f[0] for f in forwards]
        flush_at = [j for j, c in enumerate(codes) if c == flush]
        pass_no, last_flush, n = [], None, 0
        for r in rows:
            lf = max([j for j in flush_at if j < r["fwd"]], default=-1)
            if lf != last_flush:
                last_flush, n = lf, 0
            pass_no.append(n)
            n += 1
        p = np.stack([r["p"] for r in rows]).astype(np.float32)
        op = np.stack([r["open"] for r in rows])
        arrays = {"judge_fwd": np.array([r["fwd"] for r in rows], np.int32), "judge_pass": np.array(pass_no, np.int16),
                  "judge_p": np.stack([r["p"] for r in rows]), "judge_argmax": np.stack([r["argmax"] for r in rows]),
                  "judge_q": np.stack([r["q"] for r in rows]), "judge_draft": np.stack([r["draft"] for r in rows]),
                  "judge_open": op}
        by_pass_p, by_pass_c = [], []
        for k in range(max(pass_no) + 1):
            sel = np.array(pass_no) == k
            v = p[sel][op[sel]]
            by_pass_p.append(round(float(np.nanmean(v)), 4) if v.size else None)
            by_pass_c.append(round(float(np.mean(v > tau)), 4) if v.size else None)
        summ = {"judge_n_reads": len(rows), "judge_n_passes": int(max(pass_no) + 1),
                "judge_p_mean_by_pass": by_pass_p[:8], "judge_commit_frac_by_pass": by_pass_c[:8],
                "judge_p_mean": round(float(np.nanmean(p[op])), 4) if op.any() else None}
        return {"arrays": arrays, "summary": summ}


def read_cell_records(cell_dir: str, split: str, cell: Optional[str] = None) -> Dict[int, dict]:
    return CellReader(cell_dir, split, cell).refresh()


def _exp_per_token(recs: List[dict]) -> Optional[float]:
    g = sum(int(r.get("n_gen_to_eos", 0)) for r in recs)
    return (sum(int(r.get("unique_expert_layer_loads", 0)) for r in recs) / g) if g else None


class DecodeHooks:
    def __init__(self, args, out: str, dest: str, split: str, is_stock: bool, drafter_key: Optional[str],
                 drafter_eq: float, s0_dir: Optional[str], stock_dir: Optional[str], expert_level: str,
                 rules_path: Optional[str], integrity: bool, cell: Optional[str] = None,
                 stock_cell: Optional[str] = None):
        self.out, self.dest, self.split = out, dest, split
        self.is_stock, self.integrity = is_stock, integrity
        self.drafter_key, self.drafter_eq = drafter_key, float(drafter_eq or 0.0)
        self.expert_level = expert_level
        self.cell_reader = CellReader(out, split, cell)              # this cell, every shard
        self.s0_reader = CellReader(s0_dir, split) if s0_dir else None
        self.s0 = self.s0_reader.refresh() if self.s0_reader else {}
        self.stock = read_cell_records(stock_dir, split, stock_cell) if stock_dir else {}
        refs = {}
        if self.s0:
            ag = [a for r in self.s0.values() for a in r.get("draft_agreement_per_block", []) if a is not None]
            refs["s0_agreement"] = float(np.mean(ag)) if ag else None
            refs["s0_sec_per_item"] = float(np.mean([r["seconds"] for r in self.s0.values()]))
        self.stock_exp_tok = _exp_per_token(list(self.stock.values())) if self.stock else None
        self.mon = None
        if rules_path is not None:
            kw = {} if rules_path in ("", "default") else {"rules_path": rules_path}
            self.mon = StopMonitor("decode", out, refs=refs, context={"dest": os.path.basename(dest),
                                                                      "drafter": drafter_key}, **kw)
            self.rules_sha = self.mon.sha
        else:
            self.rules_sha = rules_sha()
        self.recs: List[dict] = []
        self.runf = open(dest + ".running.jsonl", "a")
        print(f"[phase2] hooks on: expert-log {expert_level}, stop rules {'on' if self.mon else 'off'}, "
              f"S0 ref {len(self.s0)} items, stock ref {len(self.stock)} items, refs {refs}", flush=True)

    # -- the meter ----------------------------------------------------------------------------------------------
    def meter_kwargs(self) -> dict:
        return {"level": self.expert_level} if self.expert_level != "off" else {}

    def meter_class(self, default_cls):
        if self.expert_level == "off":
            return default_cls
        from phase2.explog import make_meter_class
        return make_meter_class()

    def attach(self, moe) -> None:
        """Called once the meter exists: start recording the judge's per-pass probabilities."""
        if self.expert_level != "off":
            self.judge = JudgeCapture()
            self.judge.install(moe)

    # -- per item ----------------------------------------------------------------------------------------------
    def after_item(self, rec: dict, moe) -> bool:
        """Enrich `rec` in place (call BEFORE it is written); returns True when a stop rule fired."""
        if self.expert_level != "off" and hasattr(moe, "summary"):
            rec.update(moe.summary())
            moe.save(os.path.join(self.out, "experts", self.split, f"{int(rec['idx']):05d}.npz"))
        jc = getattr(self, "judge", None)
        if jc is not None:
            got = jc.take(rec.get("forwards") or [])
            if got:
                rec.update(got["summary"])
                p = os.path.join(self.out, "judge", self.split, f"{int(rec['idx']):05d}.npz")
                os.makedirs(os.path.dirname(p), exist_ok=True)
                np.savez_compressed(p, **got["arrays"])
        n_dr = sum((rec.get("drafter_n_by_kind") or {}).values())
        rec["drafter_key"] = self.drafter_key
        rec["drafter_eq_per_forward"] = self.drafter_eq
        rec["drafter_eq_tokens"] = round(n_dr * self.drafter_eq, 2)
        g = max(1, int(rec["n_gen_to_eos"]))
        rec["charged_exp_per_token"] = round((rec["unique_expert_layer_loads"] + n_dr * self.drafter_eq) / g, 3)
        rec["phase2_rules_sha256"] = self.rules_sha
        self.recs.append(rec)
        self.cell_reader.refresh()
        self.cell_reader.recs[int(rec["idx"])] = _slim(rec)            # this item is written right after the check
        if self.s0_reader is not None:
            self.s0 = self.s0_reader.refresh()
        m = self.metrics()
        self.runf.write(json.dumps({"idx": rec["idx"], "n_shard": len(self.recs), **{k: v for k, v in m.items()
                                                                                  if not isinstance(v, (list, dict))}}) + "\n")
        self.runf.flush()
        self.last = m
        if self.mon is None:
            return False
        d = self.mon.check(m, counts={"n": m["items"], "items": m["items"], "paired": m.get("paired_n", 0),
                                      "blocks": m.get("blocks", 0)})
        return d.stop

    def metrics(self) -> Dict[str, float]:
        R = list(self.cell_reader.recs.values())                   # the whole cell (every shard) so far
        n = len(R)
        gen = sum(int(r["n_gen_to_eos"]) for r in R)
        blocks = sum(int(r["n_blocks"]) for r in R)
        uniq = sum(int(r["unique_expert_layer_loads"]) for r in R)
        n_dr = sum(sum((r.get("drafter_n_by_kind") or {}).values()) for r in R)
        passes = sum(int(r["n_forward"]) - int(r["n_prefill"]) - int(r["n_flush"]) for r in R)
        scored = [r for r in R if r.get("correct") is not None]
        ag = [a for r in R for a in r.get("draft_agreement_per_block", []) if a is not None]
        m = {"items": n, "blocks": blocks,
             "accuracy": (sum(int(r["correct"]) for r in scored) / len(scored)) if scored else None,
             "exp_per_token": uniq / gen if gen else None,
             "drafter_fwd_per_token": n_dr / gen if gen else None,
             "charged_exp_per_token": (uniq + n_dr * self.drafter_eq) / gen if gen else None,
             "passes_per_block": passes / blocks if blocks else None,
             "draft_agreement": float(np.mean(ag)) if ag else None,
             "judge_commits_per_block": sum(sum(r.get("committed_judge_per_block", [])) for r in R) / blocks if blocks else None,
             "tentative_per_block": sum(sum(r.get("tentative_per_block", [])) for r in R) / blocks if blocks else None,
             "remask_per_block": sum(sum(r.get("remask_per_block", [])) for r in R) / blocks if blocks else None,
             "full_fallback_rate": (sum(sum(1 for x in r.get("full_fallback_per_block", []) if x) for r in R) / blocks)
             if blocks else None,
             "truncated_rate": sum(1 for r in R if r.get("truncated")) / n if n else None,
             "score_error_rate": sum(1 for r in R if r.get("score_error")) / n if n else None,
             "sec_per_item": float(np.mean([r["seconds"] for r in R])) if R else None,
             "ms_per_token": 1000.0 * sum(r["seconds"] for r in R) / gen if gen else None,
             "exp_block_load_entropy": float(np.nanmean([r["exp_block_load_entropy"] for r in R
                                                         if r.get("exp_block_load_entropy") is not None]))
             if any(r.get("exp_block_load_entropy") is not None for r in R) else None}
        # experts that ran, averaged over the cell's items: per forward kind (prefill / denoise / kvflush / judge),
        # experts per layer IN ONE FORWARD and the union per layer over the item, and the router's top-8 mass
        for kind in ("prefill", "denoise", "kvflush", "judge"):
            v = [r["exp_by_kind"][kind] for r in R if isinstance(r.get("exp_by_kind"), dict) and kind in r["exp_by_kind"]]
            if v:
                for key, name in (("per_fwd_uniq_per_layer", "exp_per_fwd_per_layer"), ("uniq_per_layer", "exp_union_per_layer"),
                                  ("top8_mass", "router_top8_mass"), ("entropy", "exp_load_entropy")):
                    xs = [x[key] for x in v if x.get(key) is not None]
                    if xs:
                        m[f"{name}_{kind}"] = float(np.mean(xs))
        for key in ("exp_uniq_per_layer_forward_block", "exp_union_per_layer_item"):
            xs = [r[key] for r in R if r.get(key) is not None]
            if xs:
                m[key] = float(np.mean(xs))
        jp = [r["judge_p_mean_by_pass"] for r in R if r.get("judge_p_mean_by_pass")]
        if jp:
            m["judge_passes_per_block_max"] = float(np.mean([r.get("judge_n_passes") or 0 for r in R]))
            for k in range(3):                                          # the judge's mean top-1 prob at passes 1-3
                xs = [x[k] for x in jp if len(x) > k and x[k] is not None]
                if xs:
                    m[f"judge_p_mean_pass{k + 1}"] = float(np.mean(xs))
            cf = [r["judge_commit_frac_by_pass"] for r in R if r.get("judge_commit_frac_by_pass")]
            xs = [x[0] for x in cf if x and x[0] is not None]
            if xs:
                m["judge_commit_frac_pass1"] = float(np.mean(xs))
        if self.stock_exp_tok and m["charged_exp_per_token"]:
            m["speedup_exp_vs_stock"] = self.stock_exp_tok / m["exp_per_token"]
            m["speedup_charged_vs_stock"] = self.stock_exp_tok / m["charged_exp_per_token"]
        refm = [r["ref_match"]["gen_ids"] for r in R if isinstance(r.get("ref_match"), dict)]
        if self.is_stock:
            m["stock_ref_mismatch"] = sum(1 for x in refm if not x)
        elif self.integrity and refm:
            m["ref_mismatch_rate"] = sum(1 for x in refm if not x) / len(refm)
        if self.s0:
            pairs = [(int(r["correct"]), int(self.s0[int(r["idx"])]["correct"])) for r in scored
                     if int(r["idx"]) in self.s0 and self.s0[int(r["idx"])].get("correct") is not None]
            m["paired_n"] = len(pairs)
            if len(pairs) >= 2:
                a, b = zip(*pairs)
                d95 = paired_delta(list(a), list(b), ci=0.95)
                d99 = paired_delta(list(a), list(b), ci=0.99)
                m.update({"delta_acc_vs_s0": d95["delta"], "delta_acc_ci95_lo": d95["lo"], "delta_acc_ci95_hi": d95["hi"],
                          "delta_acc_ci99_lo": d99["lo"], "delta_acc_ci99_hi": d99["hi"]})
        return m

    def wandb_row(self) -> dict:
        return {f"run/{k}": v for k, v in getattr(self, "last", {}).items() if isinstance(v, (int, float))}

    def stopped(self, reason_file: str) -> int:
        """Mark the shard stopped (never .done) and return the exit code."""
        with open(reason_file, "w") as fh:
            json.dump({"stopped": True, "n": len(self.recs), "metrics": self.last,
                       "stop": getattr(self.mon, "stopped", None)}, fh, indent=2, default=str)
        self.runf.close()
        return STOP_EXIT

    def close(self):
        if getattr(self, "judge", None) is not None:
            self.judge.uninstall()
        if not self.runf.closed:
            self.runf.close()
