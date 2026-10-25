#!/usr/bin/env python
"""run_da_decode.py + per-commit confidence probes, WITHOUT editing shared code (user request 2026-09-30).
Wraps da_decode.full_commit_rule (every full-judge pass) to record, for every committed slot: the judge's top-1
probability p, whether it confirms (argmax == draft) or overrides the draft, and whether the pass was a forced commit
(nothing cleared tau). Decisions are unchanged (the wrapper only observes). Per item, a summary is added to the
record as `commit_probe`. Then calls the unmodified run_da_decode.main() with the same arguments."""
import os, sys
ROOT = "/workspace/dmge"
for p in (os.path.join(ROOT, "code"), os.path.join(ROOT, "code", "draftalign")):
    if p not in sys.path: sys.path.insert(0, p)
import numpy as np
import research_protocol as P                                            # noqa: F401 (env)
from draftalign import da_decode as DD
import run_da_decode as R
_BUF = []
_orig_rule = DD.full_commit_rule
def _rule(cur, argmax, p, D, open_, tau_commit, mask_id, remask_only=False, tau_override=None, cov=None, tau_cov=0.0):
    out = _orig_rule(cur, argmax, p, D, open_, tau_commit, mask_id, remask_only, tau_override=tau_override, cov=cov, tau_cov=tau_cov)
    new, commit, tentative, remask, forced = out
    c = commit.cpu()
    if bool(c.any()):
        pc = p.detach().float().cpu()[c].numpy(); ag = (argmax == D).cpu()[c].numpy()
        _BUF.append((pc, ag, bool(forced)))
    return out
DD.full_commit_rule = _rule
BINS = [0.0, 0.5, 0.8, 0.9, 0.95, 0.97, 0.99, 0.999, 1.0000001]
def _summary():
    if not _BUF: return {"n_commit": 0}
    p = np.concatenate([b[0] for b in _BUF]); ag = np.concatenate([b[1] for b in _BUF])
    fo = np.concatenate([np.full(len(b[0]), b[2]) for b in _BUF])
    def m(x): return float(np.mean(x)) if len(x) else None
    return {"n_commit": int(len(p)), "n_passes_with_commit": len(_BUF), "n_forced": int(fo.sum()),
            "n_confirm": int(ag.sum()), "n_override": int((~ag).sum()),
            "mean_p": m(p), "mean_p_confirm": m(p[ag]), "mean_p_override": m(p[~ag]), "mean_p_forced": m(p[fo]),
            "mean_p_unforced": m(p[~fo]), "p_quantiles": [float(q) for q in np.quantile(p, [0.01, 0.05, 0.25, 0.5])],
            "hist_bins": BINS[:-1], "hist": np.histogram(p, bins=BINS)[0].tolist(),
            "hist_override": np.histogram(p[~ag], bins=BINS)[0].tolist()}
_orig_item = R.decode_item
def _item(*a, **k):
    _BUF.clear()
    rec, res = _orig_item(*a, **k)
    rec["commit_probe"] = _summary()
    return rec, res
R.decode_item = _item
if __name__ == "__main__":
    R.main(sys.argv[1:])
