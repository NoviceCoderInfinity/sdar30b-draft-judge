"""One-off extractor: turns ONE real decode (Qwen3-0.6B drafter, GSM8K item 0, no pool) into flow_example.json, which
build.py embeds for the animated "how a block is written" figure. Read-only on the logs; stdlib only (no numpy).

Usage: python3 extract_flow.py [arm] [set] [idx]      (default: S1_nopool gsm8k500 0)

Per block it replays the logged judge reads with the decode rule (da_decode.full_commit_rule):
  commit = open slots with judge p > 0.95 (if none: the single best slot, "forced"); the rest keep the draft token
  if the judge's argmax equals it ("keep"), otherwise are re-masked. The replay is checked against the record's
  per-pass counts, and the final tokens against the saved block tokens.
"""
import ast, glob, json, os, struct, sys, zipfile

ARM = sys.argv[1] if len(sys.argv) > 1 else 'S1_nopool'
SET = sys.argv[2] if len(sys.argv) > 2 else 'gsm8k500'
IDX = int(sys.argv[3]) if len(sys.argv) > 3 else 0
ROOT = '/workspace/dmge_p2/logs/phase2/decode'
TOK = glob.glob('/workspace/dmge_p2/hf_cache/hub/models--JetLM--SDAR-30B-A3B-Chat-b32/snapshots/*/')[0]
HERE = os.path.dirname(os.path.abspath(__file__))
TAU = 0.95
DRAFT_COST = {'S1_nopool': 126.32, 'F2_nopool': 364.64}[ARM]
QUESTION = {('gsm8k500', 0): "Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins "
            "for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. "
            "How much in dollars does she make every day at the farmers' market?"}


def npz(path):
    z = zipfile.ZipFile(path); out = {}
    fmt = {'<i4': 'i', '<f4': 'f', '<f2': 'e', '|b1': '?', '|i1': 'b', '<i2': 'h'}
    for n in z.namelist():
        with z.open(n) as f:
            f.read(6); major = f.read(1)[0]; f.read(1)
            hl = struct.unpack('<H', f.read(2))[0] if major == 1 else struct.unpack('<I', f.read(4))[0]
            h = ast.literal_eval(f.read(hl).decode('latin1')); raw = f.read()
        c = fmt[h['descr']]; k = struct.calcsize('<' + c)
        vals = list(struct.unpack('<%d%s' % (len(raw) // k, c), raw))
        shape = h['shape']
        def rs(v, s):
            if len(s) <= 1: return v
            step = len(v) // s[0]
            return [rs(v[i * step:(i + 1) * step], s[1:]) for i in range(s[0])]
        out[n[:-4]] = vals[0] if shape == () else rs(vals, shape)
    return out


def byte_decoder():
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256)); cs = bs[:]; n = 0
    for b in range(256):
        if b not in bs: bs.append(b); cs.append(256 + n); n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


BD = byte_decoder()
VOC = json.load(open(TOK + 'vocab.json')); INV = {v: k for k, v in VOC.items()}
for t in json.load(open(TOK + 'tokenizer.json')).get('added_tokens', []): INV[t['id']] = t['content']
SPECIAL = {151645: '<end>', 151643: '<pad>'}


def piece(i):
    if i in SPECIAL: return SPECIAL[i]
    s = INV.get(i, '?')
    try: return bytes(BD[c] for c in s).decode('utf8', 'replace')
    except KeyError: return s


def show(i):
    s = piece(i).replace('\n\n', '↵↵').replace('\n', '↵')
    return s.strip(' ') or '·'


rec = None
for f in sorted(glob.glob(f'{ROOT}/{ARM}/{SET}/*.shard?of8.jsonl')):
    for l in open(f):
        if l.strip():
            r = json.loads(l)
            if int(r['idx']) == IDX: rec = r
assert rec, 'record not found'
T = npz(f'{ROOT}/{ARM}/{SET}/traces/{SET}/{IDX:05d}.npz')
J = npz(f'{ROOT}/{ARM}/{SET}/judge/{SET}/{IDX:05d}.npz')
fw = rec['forwards']                      # [kind, ms, unique expert-layer loads, layers, npos, pool]
nb = rec['n_blocks']; ppb = rec['judge_forwards_per_block']
assert sum(ppb) == len(J['judge_fwd']) and len(T['tokens']) == nb

blocks, row, problems = [], 0, []
EOS = 151645
flat = []
for b in range(nb):
    D, fin, valid = T['draft_tokens'][b], T['tokens'][b], [bool(v) for v in T['valid'][b]]
    cur = [D[i] if valid[i] else None for i in range(32)]
    committed = [False] * 32; forced_slot = [False] * 32
    passes = []
    for k in range(ppb[b]):
        r = row + k
        op, p, am = J['judge_open'][r], J['judge_p'][r], J['judge_argmax'][r]
        pz = [(-1.0 if x != x else x) for x in p]
        commit = [bool(op[i]) and pz[i] > TAU for i in range(32)]
        forced = False
        if any(op) and not any(commit):
            best = max((i for i in range(32) if op[i]), key=lambda i: pz[i]); commit[best] = True; forced = True
        rest = [bool(op[i]) and not commit[i] for i in range(32)]
        keep = [rest[i] and am[i] == D[i] for i in range(32)]
        rem = [rest[i] and not keep[i] for i in range(32)]
        exp_c = rec['full_passes_per_block'][b][k]
        if (sum(commit), sum(keep), sum(rem), forced) != (exp_c['commit'], exp_c['tentative'], exp_c['remask'], exp_c['forced']):
            problems.append(('counts', b, k, (sum(commit), sum(keep), sum(rem), forced), exp_c))
        st = ''
        for i in range(32):
            if not valid[i]: st += '.'
            elif commit[i]: st += 'f' if forced else ('a' if am[i] == D[i] else 'o')
            elif committed[i]: st += 'A' if cur[i] == D[i] and not forced_slot[i] else ('O' if cur[i] != D[i] else 'A')
            elif keep[i]: st += 'k'
            else: st += 'm'
        for i in range(32):
            if commit[i]: committed[i] = True; cur[i] = am[i]; forced_slot[i] = forced
        passes.append({'st': st, 'n_commit': sum(commit), 'n_confirm': sum(1 for i in range(32) if commit[i] and am[i] == D[i]),
                       'n_keep': sum(keep), 'n_remask': sum(rem), 'forced': forced,
                       'cost': fw[J['judge_fwd'][r]][2], 'pmax': round(max([pz[i] for i in range(32) if op[i]] or [0]), 3)})
    row += ppb[b]
    if any(valid[i] and cur[i] != fin[i] for i in range(32)): problems.append(('final', b))
    last = J['judge_fwd'][row - 1]
    flush = fw[last + 1][2] if last + 1 < len(fw) and fw[last + 1][0] == 2 else 0
    toks = [show(fin[i]) if valid[i] else '' for i in range(32)]
    text = ''.join(piece(fin[i]) for i in range(32) if valid[i] and fin[i] not in SPECIAL)
    blocks.append({'valid': ''.join('1' if v else '0' for v in valid), 'draft': [show(D[i]) if valid[i] else '' for i in range(32)],
                   'final': toks, 'text': text, 'passes': passes, 'flush_cost': flush,
                   'agreement': round(rec['draft_agreement_per_block'][b], 3)})
    flat += [fin[i] for i in range(32) if valid[i]]

n_eos = flat.index(EOS) + 1 if EOS in flat else len(flat)
assert n_eos == rec['n_gen_to_eos'], (n_eos, rec['n_gen_to_eos'])
cum = 0
for b in blocks:
    cum += b['valid'].count('1'); b['tokens_total'] = min(cum, n_eos)
tot_loads = sum(f[2] for f in fw)
assert tot_loads == rec['unique_expert_layer_loads'], (tot_loads, rec['unique_expert_layer_loads'])
nd = sum(rec['drafter_n_by_kind'].values())
out = {'arm': ARM, 'set': SET, 'idx': IDX, 'question': QUESTION.get((SET, IDX), ''), 'answer': rec['pred'].strip(),
       'correct': bool(rec['correct']), 'tau': TAU, 'tokens': n_eos, 'prefill_cost': fw[0][2],
       'draft_cost': DRAFT_COST, 'drafter_forwards_per_draft': rec['drafter_forwards_per_block'][0],
       'drafter_prefill_forwards': rec['drafter_n_by_kind']['prefill'], 'drafter_flush_forwards': 1,
       'charged_per_token': round((tot_loads + nd * DRAFT_COST) / n_eos, 1), 'loads_total': tot_loads, 'blocks': blocks}
assert abs(out['charged_per_token'] - rec['charged_exp_per_token']) < 0.2, (out['charged_per_token'], rec['charged_exp_per_token'])
json.dump(out, open(os.path.join(HERE, 'flow_example.json'), 'w'), ensure_ascii=False, separators=(',', ':'))
print('ok', 'blocks', nb, 'passes', sum(ppb), 'tokens', n_eos, 'charged/token', out['charged_per_token'], 'problems', problems[:6], len(problems))
print(os.path.getsize(os.path.join(HERE, 'flow_example.json')), 'bytes')
