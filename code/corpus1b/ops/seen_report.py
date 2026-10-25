"""What the final checkpoint has seen, by domain / difficulty / source, vs everything on disk (CPU only).
Inputs: prompt lists (all waves), generation records (wave A), training index, seen_items.json (replay_seen.py)."""
import ast, collections as C, glob, gzip, json
def meta(r):
    m = r.get("meta") or {}
    return ast.literal_eval(m) if isinstance(m, str) else m
SHORT = {"AI-MO/NuminaMath-1.5": "NuminaMath-1.5", "EleutherAI/hendrycks_math": "MATH (Hendrycks) train",
         "nvidia/OpenMathInstruct-2": "OpenMathInstruct-2", "nvidia/OpenMathReasoning": "OpenMathReasoning",
         "openai/gsm8k": "GSM8K train", "zwhe99/DeepMath-103K": "DeepMath-103K", "nvidia/OpenCodeInstruct": "OpenCodeInstruct",
         "google-research-datasets/mbpp": "MBPP train/val", "deepmind/code_contests": "CodeContests",
         "likaixin/TACO-verified": "TACO-verified", "open-r1/codeforces": "open-r1/codeforces", "codeparrot/apps": "APPS"}
def group(d, s, m):
    n = SHORT.get(s, s.split("/")[-1])
    if d == "maths":
        b = {"gsm": "1 grade-school (GSM)", "math_easy": "2 MATH-easy", "math_hard": "3 MATH-hard", "olympiad": "4 olympiad"}.get(m.get("bucket"), m.get("bucket"))
        sub = n + (f" {m['level']}" if "level" in m else "") + (f" [{m['source']}]" if s.endswith("NuminaMath-1.5") else "")
        return b, sub
    if d == "code":
        if m.get("bucket") == "function":
            return "1 easier: function-level (Python)", n + (f" [{m.get('generation_algorithm')}, {m.get('oci_domain')}]" if "OpenCode" in s else f" [{m.get('split')}]")
        lvl = m.get("source_platform") or m.get("difficulty") or (f"rating {m['rating']}" if "rating" in m else "")
        if "code_contests" in s: lvl = m.get("source_platform", "")
        return "2 competitive (CF-style, stdin/stdout)", f"{n} [{lvl}]"
    if d == "science":
        return f"{m.get('bucket')}", n + (f" [{m['category']}]" if "category" in m else "") + (f" [{m['config']}]" if "config" in m else "")
    if d == "if":
        return f"{m.get('n_constraints')} constraint(s)", n
    return "chat", n
G = {}; plist = C.Counter(); plistA = C.Counter()
for d in ["maths", "code", "science", "if", "chat"]:
    for l in open(f"data/c1b/prompt_lists/{d}.jsonl"):
        r = json.loads(l); k = (d,) + group(d, r["source"], meta(r)); G[r["pid"]] = k
        plist[k] += 1; plistA[k] += r.get("wave") == "A"
gen = C.Counter(); gen_tok = C.Counter(); rej = C.Counter(); rej_tok = C.Counter()
for f in glob.glob("logs/c1b/gen/waveA/gpu*/records.*.jsonl.gz"):
    for l in gzip.open(f, "rt"):
        r = json.loads(l); k = G.get(r["pid"]);
        if k is None: continue
        t = int(r["n_gen"]) + int(r["prompt_len"]); gen[k] += 1; gen_tok[k] += t
        if not (r["kept"] in (True, "True")): rej[k] += 1; rej_tok[k] += t
idx_n = C.Counter(); idx_tok = C.Counter()
for l in open("logs/c1b/trainset/waveA/index.jsonl"):
    r = json.loads(l); k = G[r["pid"]]; idx_n[k] += 1; idx_tok[k] += r["corpus_tokens"]
S = json.load(open("logs/c1b/train/main/seen_items.json"))
tr_tok = C.Counter(); seen_tok = C.Counter(); touched = C.Counter(); tr_n = C.Counter()
for pid, v in S["items"].items():
    k = G[pid]; tr_n[k] += 1; tr_tok[k] += v["item_tokens"]; seen_tok[k] += v["seen_tokens"]; touched[k] += v["seen_tokens"] > 0
keys = sorted(set(plist) | set(idx_n))
rows = []
for k in keys:
    rows.append(dict(domain=k[0], group=k[1], sub=k[2], prompts=plist[k], promptsA=plistA[k], gen=gen[k], gen_tok=gen_tok[k],
                     rej=rej[k], rej_tok=rej_tok[k], items=idx_n[k], item_tok=idx_tok[k], train_items=tr_n[k], train_tok=tr_tok[k],
                     seen_tok=seen_tok[k], touched=touched[k]))
json.dump({"trainer_tokens": S["trainer_tokens"], "rows": rows}, open("logs/c1b/train/main/seen_report.json", "w"))
print("rows", len(rows), "seen total", round(sum(r["seen_tok"] for r in rows) / 1e6, 3), "M")
