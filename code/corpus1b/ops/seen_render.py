import json, collections as C
R = json.load(open("logs/c1b/train/main/seen_report.json")); rows = R["rows"]
M = lambda x: f"{x/1e6:.2f}M" if x >= 1e4 else f"{x/1e3:.1f}K" if x >= 100 else f"{x:.0f}"
pct = lambda a, b: f"{100*a/b:.0f}%" if b else "—"
out = []
tot = C.Counter()
for d in ["maths", "code", "science", "if", "chat"]:
    rs = [r for r in rows if r["domain"] == d]
    s = {k: sum(r[k] for r in rs) for k in rs[0] if isinstance(rs[0][k], (int, float))}
    for k, v in s.items(): tot[k] += v
    out.append(f"\n### {d.upper()}  — seen {M(s['seen_tok'])} of {M(s['train_tok'])} converted train tokens ({pct(s['seen_tok'], s['train_tok'])}); "
               f"items touched {s['touched']:,}/{s['train_items']:,} ({pct(s['touched'], s['train_items'])})\n")
    out.append("| group | source [subset] | prompts on disk | generated (wave A) | rejected | converted items (tokens) | seen tokens | % of converted seen | items touched |")
    out.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for g in sorted({r["group"] for r in rs}):
        gs = [r for r in rs if r["group"] == g]
        gsum = {k: sum(r[k] for r in gs) for k in s}
        out.append(f"| **{g}** | **all** | **{gsum['prompts']:,}** | **{gsum['gen']:,}** | **{gsum['rej']:,}** | **{gsum['items']:,} ({M(gsum['item_tok'])})** | "
                   f"**{M(gsum['seen_tok'])}** | **{pct(gsum['seen_tok'], gsum['train_tok'])}** | **{pct(gsum['touched'], gsum['train_items'])}** |")
        for r in sorted(gs, key=lambda r: -r["prompts"]):
            if r["prompts"] < 50 and r["items"] == 0: continue
            out.append(f"| | {r['sub']} | {r['prompts']:,} | {r['gen']:,} | {r['rej']:,} | {r['items']:,} ({M(r['item_tok'])}) | {M(r['seen_tok'])} | "
                       f"{pct(r['seen_tok'], r['train_tok'])} | {pct(r['touched'], r['train_items'])} |")
out.insert(0, f"**Totals:** prompts on disk {tot['prompts']:,} (wave A {tot['promptsA']:,}); generated {tot['gen']:,} answers ({M(tot['gen_tok'])} tokens incl. prompts); "
              f"rejected {tot['rej']:,} ({M(tot['rej_tok'])}); converted to blocks {tot['items']:,} items ({M(tot['item_tok'])}); "
              f"train split {tot['train_items']:,} items ({M(tot['train_tok'])}); **seen {M(tot['seen_tok'])} = {pct(tot['seen_tok'], tot['train_tok'])} of converted train tokens**, "
              f"items touched {tot['touched']:,} ({pct(tot['touched'], tot['train_items'])}). Trainer count {M(R['trainer_tokens'])}.")
open("logs/c1b/train/main/seen_report.md", "w").write("\n".join(out) + "\n"); print("\n".join(out))
