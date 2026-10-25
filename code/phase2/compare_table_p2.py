"""Run from the repository root.
Per-benchmark comparison: stock / SDAR-1.7B + 128M judge (Phase 1) / new drafter + 128M judge. Prints a markdown
table and appends the numbers to metrics.json (for the artifact). Usage: table.py ARM SET [SET ...]"""
import json, glob, sys, os, numpy as np
arm, sets = sys.argv[1], sys.argv[2:]
NOPOOL = arm.endswith('_nopool'); base = arm.replace('_nopool', ''); POOLTXT = ' (no pool)' if NOPOOL else ' (DES48)'
NAMES = {'F2': 'CoDA-1.7B', 'S1': 'Qwen3-0.6B'}; COST = {'S0': 364.64, 'F2': 364.64, 'S1': 126.32}; COST['F2_nopool'] = COST['F2']; COST['S1_nopool'] = COST['S1']
SIZE = {'humaneval': 164, 'mbpp': 257, 'gsm8k500': 500, 'ifeval541': 541, 'math500': 500, 'gpqa': 198, 'aime': 90}
OUT = os.environ.get("P2_METRICS", "logs/phase2/analysis/tables/metrics.json"); os.makedirs(os.path.dirname(OUT), exist_ok=True)
def load(pat):
    R = {}
    for f in glob.glob(pat):
        if f.endswith('.running.jsonl'): continue
        for l in open(f):
            if l.strip():
                try: r = json.loads(l)
                except ValueError: continue
                R[int(r['idx'])] = r
    return R
def stats(R, eq):
    ids = sorted(R); g = sum(R[i]['n_gen_to_eos'] for i in ids); u = sum(R[i]['unique_expert_layer_loads'] for i in ids)
    nd = sum(sum((R[i].get('drafter_n_by_kind') or {}).values()) for i in ids)
    dsec = sum(R[i].get('drafter_seconds') or 0 for i in ids)
    sc = [int(R[i]['correct']) for i in ids if R[i].get('correct') is not None]
    ag = [a for i in ids for a in (R[i].get('draft_agreement_per_block') or []) if a is not None]
    return {'n': len(ids), 'acc': 100 * np.mean(sc) if sc else None, 'exp_tok': u / g,
            'drafter_fwd_tok': nd / g, 'drafter_cost_tok': nd * eq / g, 'charged_tok': (u + nd * eq) / g,
            'drafter_ms_tok': 1000 * dsec / g, 'ms_tok': 1000 * sum(R[i]['seconds'] for i in ids) / g,
            'passes_blk': sum(R[i]['n_forward'] - R[i]['n_prefill'] - R[i]['n_flush'] for i in ids) / sum(R[i]['n_blocks'] for i in ids),
            'kept': 100 * np.mean(ag) if ag else None}
allm = json.load(open(OUT)) if os.path.exists(OUT) else {}
for s in sets:
    st = stats(load(f'logs/phase2/ref/stock_flat_by_set/{s}/*.jsonl'), 0)
    s0R = load(f"logs/phase2/ref/{'S0_128M_nopool_by_set' if NOPOOL else 'S0_128M_by_set'}/{s}/*.jsonl"); s0 = stats(s0R, COST['S0'])
    aR = load(f'logs/phase2/decode/{arm}/{s}/*.shard*of8.jsonl')
    if len(aR) < SIZE[s] and not os.environ.get('PARTIAL'): print(f'{s}: {arm} has {len(aR)}/{SIZE[s]} items, not complete'); continue
    a = stats(aR, COST[arm])
    common = sorted(set(s0R) & set(aR)); d = np.array([int(aR[i]['correct']) - int(s0R[i]['correct']) for i in common])
    rng = np.random.default_rng(0); bs = [rng.choice(d, len(d)).mean() for _ in range(2000)]
    delta = (100 * d.mean(), 100 * np.percentile(bs, 2.5), 100 * np.percentile(bs, 97.5))
    for x in (s0, a):
        x['sp_judge'] = st['exp_tok'] / x['exp_tok']; x['sp_full'] = st['exp_tok'] / x['charged_tok']
    a['sp_drafter_compute'] = s0['drafter_cost_tok'] / a['drafter_cost_tok']; s0['sp_drafter_compute'] = 1.0
    a['sp_drafter_time'] = s0['drafter_ms_tok'] / a['drafter_ms_tok']; s0['sp_drafter_time'] = 1.0
    allm.setdefault(s, {})['stock'] = st; allm[s]['S0_nopool' if NOPOOL else 'S0'] = s0; allm[s][arm] = a; allm[s][arm + '_delta'] = delta
    f = lambda v, p=1: '—' if v is None else f'{v:.{p}f}'
    print(f'\n### {s} ({SIZE[s]} items)\n')
    print('| Setup | Accuracy | Experts / token (30B) | Speedup: 30B + judge (drafter free) | Drafter cost / token (expert units) | **Speedup: full pipeline (drafter counted)** | Drafter-only speedup vs SDAR-1.7B (compute) | Drafter time / token (ms) | Drafter-only speedup vs SDAR-1.7B (time) | Passes / block | Drafter tokens kept |')
    print('|---|---|---|---|---|---|---|---|---|---|---|')
    print(f"| Stock 30B (no drafter, no judge) | {f(st['acc'])}% | {f(st['exp_tok'])} | 1.00× | 0 | **1.00×** | — | — | — | {f(st['passes_blk'],2)} | — |")
    for name, x in (('SDAR-1.7B + 128M judge' + POOLTXT + ' (Phase 1)', s0), (f"**{NAMES[base]} + 128M judge{POOLTXT} (new)**", a)):
        print(f"| {name} | {f(x['acc'])}% | {f(x['exp_tok'])} | {f(x['sp_judge'],2)}× | {f(x['drafter_cost_tok'])} | **{f(x['sp_full'],2)}×** | {f(x['sp_drafter_compute'],2)}× | {f(x['drafter_ms_tok'],2)} | {f(x['sp_drafter_time'],2)}× | {f(x['passes_blk'],2)} | {f(x['kept'])}% |")
    print(f"\nPaired Δ accuracy ({NAMES[base]} − SDAR-1.7B), {len(common)} items: {delta[0]:+.1f} pts, 95% CI [{delta[1]:+.1f}, {delta[2]:+.1f}]")
json.dump(allm, open(OUT, 'w'), indent=1, default=float)
