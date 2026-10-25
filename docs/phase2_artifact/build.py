"""Re-embed the data blocks into drafter-swap.html in place, so the page can be republished to the same artifact URL.

Usage: python3 build.py [path/to/metrics.json]      (default: ../metrics.json, as written by table.py)

Four data blocks are refreshed (each is a <script type="application/json" id="..."> in the page):
  metrics-data   metrics.json, only: stock, S0_nopool, F2_nopool(+_delta), S1_nopool(+_delta). A benchmark is embedded
                 only once a no-pool run has written it. Everything else (S0, F2, F2_delta, S1, ... = DES48) is dropped.
                 A new-drafter run with fewer items than the benchmark is left out (the page says "full sets").
  analysis-data  a compact extract of /workspace/dmge_p2/logs/phase2/analysis/nopool/analysis.json (read-only):
                 dip tests, content agreement, pass attribution, lever estimates. Missing keys are skipped, and the
                 page hides a figure whose data is absent. Override the path with env ANALYSIS_JSON.
  flow-data      flow_example.json, one real decode written by extract_flow.py (static; rerun that script to change it).
  cand-data      notion_candidates.md (the team's Notion tables), parsed into tables.
"""
import datetime, json, math, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, '..', 'metrics.json')
ANALYSIS = os.environ.get('ANALYSIS_JSON', '/workspace/dmge_p2/logs/phase2/analysis/nopool/analysis.json')
CANDS = os.path.join(HERE, 'notion_candidates.md')
FLOW = os.path.join(HERE, 'flow_example.json')
PAGE = os.path.join(HERE, 'drafter-swap.html')
KEEP = ('stock', 'S0_nopool', 'F2_nopool', 'F2_nopool_delta', 'S1_nopool', 'S1_nopool_delta')
NEW = {'F2_nopool': 'CoDA-1.7B', 'S1_nopool': 'Qwen3-0.6B'}


def clean(o):
    if isinstance(o, float):
        return round(o, 6) if math.isfinite(o) else None
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    return o


def select(d):
    out = {}
    for s, arms in d.items():
        kept = {k: arms[k] for k in KEEP if k in arms}
        full = (kept.get('stock') or {}).get('n')
        for k in NEW:  # the page says "full sets": never show a partial run as finished
            n = (kept.get(k) or {}).get('n')
            if k in kept and full and n != full:
                print(f'  WARNING {s}: {NEW[k]} has n={n} of {full}; left out as incomplete')
                kept.pop(k); kept.pop(k + '_delta', None)
        if any(k != 'stock' for k in kept):
            out[s] = kept
    return out


def utc(ts):
    return datetime.datetime.fromtimestamp(ts, datetime.timezone.utc).strftime('%Y-%m-%d %H:%M UTC')


def dumps(o):
    return json.dumps(clean(o), separators=(',', ':'), allow_nan=False, ensure_ascii=False).replace('</', '<\\/')


def embed(html, ident, blob, stamp=None):
    pat = r'(<script type="application/json" id="%s"(?: data-snapshot="[^"]*")?>).*?(</script>)' % ident
    head = '<script type="application/json" id="%s"%s>' % (ident, ' data-snapshot="%s"' % stamp if stamp else '')
    html, n = re.subn(pat, lambda m: head + blob + m.group(2), html, count=1, flags=re.S)
    assert n == 1, ident + ' block not found in ' + PAGE
    return html


# ---------------------------------------------------------------- analysis extract
LEVER_KEYS = ('central', 'lo', 'hi', 'best_set', 'best_set_cut', 'ranked')


def extract_analysis(d):
    out = {'generated': d.get('generated_utc'), 'arms': {}}
    cd = d.get('content_deep_dive') or {}
    for arm in NEW:
        try:
            P = d['pooled'][arm]
            ans = d['answers'][arm]
        except KeyError:
            continue
        a = {'label': (d['sets'][P['sets'][0]]['arms'][arm].get('label') if P.get('sets') else None) or NEW[arm],
             'sets': list(P.get('sets') or [])}
        v = P.get('vs_A') or {}
        a['pooled'] = {k: v.get(k) for k in ('n', 'n_sets', 'delta', 'lo', 'hi', 'mcnemar_p', 'x_only', 'y_only',
                                              'macro_delta', 'macro_lo', 'macro_hi', 'acc_x', 'acc_y')}
        a['pooled']['verdict'] = ans.get('q1_pooled')
        a['per_set'] = {}
        for s in a['sets']:
            try:
                sv = d['sets'][s]['arms'][arm]['vs_A']
                ref = d['sets'][s]['reference']['A']
            except KeyError:
                continue
            a['per_set'][s] = {k: sv.get(k) for k in ('n', 'acc_x', 'acc_y', 'delta', 'lo', 'hi', 'mcnemar_p', 'x_only', 'y_only')}
            a['per_set'][s]['verdict'] = (ans.get('q1_per_set') or {}).get(s)
            a['per_set'][s]['A_passes'] = ref.get('passes_per_block')
        a['ind'] = ans.get('indicators') or {}
        a['content'], a['classes'], a['gap'] = {}, {}, {}
        for s, sv in ((cd.get('arms') or {}).get(arm, {}).get('sets') or {}).items():
            try:
                n, A = sv['agree_new'], sv['agree_A']
                a['content'][s] = {'raw_new': n['all'], 'content_new': n['content'], 'raw_A': A['all'], 'content_A': A['content'],
                                   'content_A_range': A.get('content_range'), 'header_new': n.get('header_share'),
                                   'tail_new': n.get('tail_share'), 'tail_A': A.get('tail_share')}
                at = sv['attribution']
                a['classes'][s] = {'header': at['header'], 'padding': at['padding (tail)'], 'kept': at['content, kept'],
                                   'plausible': at['content, plausible alternative'], 'miss': at['content, genuine miss'],
                                   'total': at['total']}
                ab = sv['abc']
                a['gap'][s] = {'first': ab['a_first_block'], 'padding': ab['a_padding'], 'plausible': ab['b_plausible_alternative'],
                               'miss': ab['c_genuine_miss'], 'mix': ab['mix'], 'total': ab['total']}
            except KeyError:
                continue
        try:
            pl = cd['pooled_levers'][arm]
            a['levers'] = {k: {x: lv.get(x) for x in LEVER_KEYS} for k, lv in pl['levers'].items()}
            a['speedup_now'] = pl.get('speedup_now')
        except KeyError:
            pass
        out['arms'][arm] = a
    return out


# ---------------------------------------------------------------- candidates (Notion export)
def md_table(lines):
    rows = [[c.strip() for c in l.strip().strip('|').split('|')] for l in lines if l.strip().startswith('|')]
    return rows[0], [r for r in rows[2:]]


def cell(t, idcol=False):
    d = '[diff]' in t
    t = t.replace('[diff]', '').strip()
    pick = False
    if idcol and '(green)' in t:
        pick = True; t = t.replace('(green)', '').strip()
    return {'t': t, 'd': int(d), 'pick': int(pick)}


def parse_cands(path):
    txt = open(path).read()
    first = txt.splitlines()[0]
    m = re.search(r'\(from (the )?(.*)\)\s*$', first)
    out = {'source': m.group(2) if m else 'Notion', 'tables': {}}
    for sec in re.split(r'\n(?=## )', txt)[1:]:
        head, *body = sec.splitlines()
        if not any(l.strip().startswith('|') for l in body):
            continue
        cols, rows = md_table(body)
        if head.startswith('## Table 1') or head.startswith('## Table 2'):
            key = 't1' if 'Table 1' in head else 't2'
            out['tables'][key] = {'title': head.split(':', 1)[1].strip(), 'cols': cols,
                                  'rows': [[cell(c, i == 0) for i, c in enumerate(r)] for r in rows]}
        elif head.startswith('## Claimed'):
            def num(c):
                m = re.match(r'\s*([0-9.]+)\s*(.*)', c)
                return {'v': float(m.group(1)), 'n': m.group(2).strip('() ')} if m else {'v': None, 'n': ''}
            out['tables']['claimed'] = {'title': 'Claimed scores by the authors (their settings, not ours), %', 'cols': cols,
                                        'rows': [[r[0]] + [num(c) for c in r[1:]] for r in rows]}
    return out


# ---------------------------------------------------------------- main
data = select(clean(json.load(open(SRC))))
html = open(PAGE).read()
html = embed(html, 'metrics-data', dumps(data), utc(os.path.getmtime(SRC)))

notes = []
try:
    an = extract_analysis(json.load(open(ANALYSIS)))
    html = embed(html, 'analysis-data', dumps(an), (an['generated'] or '').replace('T', ' ')[:16] + ' UTC')
    notes.append('analysis: %s; ' % an['generated'] + ', '.join('%s %d sets' % (NEW[k], len(v['sets'])) for k, v in an['arms'].items()))
except (OSError, ValueError, KeyError) as e:
    html = embed(html, 'analysis-data', '{}')
    notes.append('analysis: NOT embedded (%s: %s)' % (type(e).__name__, e))
try:
    html = embed(html, 'flow-data', dumps(json.load(open(FLOW))))
    notes.append('flow example embedded')
except OSError as e:
    html = embed(html, 'flow-data', '{}'); notes.append('flow example missing (%s)' % e)
try:
    cands = parse_cands(CANDS)
    html = embed(html, 'cand-data', dumps(cands))
    notes.append('candidates: %s' % ', '.join('%s %d rows' % (k, len(t['rows'])) for k, t in cands['tables'].items()))
except OSError as e:
    html = embed(html, 'cand-data', '{}'); notes.append('candidates missing (%s)' % e)
open(PAGE, 'w').write(html)

print(f'snapshot {utc(os.path.getmtime(SRC))}; embedded {len(data)} benchmark(s): {", ".join(data) or "none"}')
for k, name in NEW.items():
    fin = [s for s, arms in data.items() if k in arms and k + '_delta' in arms and 'S0_nopool' in arms]
    print(f'  {name:11s} finished: {", ".join(fin) or "none"}')
for n in notes:
    print('  ' + n)
print(f'page: {PAGE}')
