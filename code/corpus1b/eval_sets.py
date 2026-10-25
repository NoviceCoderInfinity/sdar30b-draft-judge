#!/usr/bin/env python
"""Freeze the four held-out eval sets into data/c1b/eval/<name>.jsonl (+ .sha256), one JSON object per
line, fields {idx, prompt, gold, split}. "split" is this protocol's ROLE tag ("test" for every row here,
matching research_protocol.test_items()'s convention), not necessarily the upstream HF split name.

Run on the GPU box only (network + HF datasets cache required; this Mac never downloads a full dataset --
field names below were verified with tiny HF datasets-server requests, 2026-09-28):

  python code/corpus1b/eval_sets.py --set ifeval541
  python code/corpus1b/eval_sets.py --set finqa500
  python code/corpus1b/eval_sets.py --set chartqa300
  python code/corpus1b/eval_sets.py --set chat800
  python code/corpus1b/eval_sets.py --set all        # default

research_protocol.py's loaders (ifeval541_items() etc.) read the frozen output back and sha256-check it,
the same way if_pool_items() does for code/configs/if_pool.json.

Each builder is split into a pure row-transform (`_x_rows`, no network, unit-tested on synthetic rows
shaped like the verified schema below) and a thin `build_x()` wrapper that loads the real HF split and
calls it -- see code/corpus1b/tests/test_checkers_eval.py.

Verified source schemas (HF datasets-server `/splits` and `/rows`, 2026-09-28)
--------------------------------------------------------------------------------
google/IFEval                        default/train (541 rows; ONLY split -- there is no "test")
  key: int, prompt: str, instruction_id_list: list[str],
  kwargs: list[dict] (EVERY kwarg key present per entry, unused ones are null)
dreamerdeo/finqa                     default/test (1,147 rows)
  id: str, pre_text: list[str], post_text: list[str], question: str, answer: str, gold_evidence:
  list[str], table: list[list[str]] (first row is the header). Exactly 7 columns -- there is no separate
  "exe_ans"/executed-answer field in this mirror, so `answer` (e.g. "94", "14%", "-4.4%") IS the gold;
  verified against 100 sampled test rows, ~29% of which are percentages.
nmayorga7/chartqa-tables              default/test (1,509 rows, one per chart image)
  imgname: str, table_csv: str (real CSV, header + data rows), table_formatted: str (a "|"/"&" pseudo-
  table -- NOT markdown, not used here)
ahmed-masry/ChartQA                   default/test (2,500 rows: exactly 1,250 "human" + 1,250
  "augmented", verified by full-split tally)
  imgname: str, query: str, label: str, type: str ("human" | "augmented"), image: binary
HuggingFaceH4/ultrachat_200k          default/test_sft (23,110 rows)
  prompt: str, prompt_id: str (sha256 of the raw first-turn text -- a stable ordering key), messages: [...]
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))            # code/corpus1b
CODE = os.path.abspath(os.path.join(HERE, ".."))              # code/
if CODE not in sys.path:
    sys.path.insert(0, CODE)

from corpus1b import common                                    # noqa: E402

BOXED_TAIL = "Please reason step by step, and put your final answer within \\boxed{}."


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------
def _freeze(name: str, rows: list, out_dir: str = None) -> dict:
    """Writes <out_dir>/<name>.jsonl and its .sha256 sidecar (sha256sum format: '<hex>  <basename>\\n',
    the same format code/configs/if_pool.json.sha256 already uses, so research_protocol's loaders can
    keep reading it with `.read().split()[0]`). out_dir defaults to common.EVAL_DIR
    (data/c1b/eval/); tests pass a tmp_path instead so nothing lands in the real worktree."""
    out_dir = out_dir or common.EVAL_DIR
    path = os.path.join(out_dir, f"{name}.jsonl")
    n = common.write_jsonl(path, rows)
    h = common.sha256_file(path)
    with open(path + ".sha256", "w", encoding="utf-8") as fh:
        fh.write(f"{h}  {os.path.basename(path)}\n")
    return {"name": name, "n": n, "path": path, "sha256": h}


def markdown_table(rows: list) -> str:
    """rows: list[list[str]], first row = header. Plain GitHub-flavoured Markdown; "|" and newlines
    inside a cell are escaped/collapsed so the table does not break."""
    clean = [[("" if c is None else str(c)).replace("|", "\\|").replace("\n", " ").strip() for c in r]
             for r in rows]
    clean = [r for r in clean if r]
    if not clean:
        return ""
    head, body = clean[0], clean[1:]
    out = ["| " + " | ".join(head) + " |", "| " + " | ".join(["---"] * len(head)) + " |"]
    out += ["| " + " | ".join(r) + " |" for r in body]
    return "\n".join(out)


def csv_to_rows(csv_text: str) -> list:
    return [row for row in csv.reader(io.StringIO(csv_text or ""))]


# ---------------------------------------------------------------------------
# ifeval541 -- google/IFEval, all 541 rows (its only split is "train")
# ---------------------------------------------------------------------------
def _ifeval_rows(ds_rows) -> list:
    """gold kwargs stripped of None values exactly as draftalign/build_if_pool.py does for IFEval-format
    golds (nemotron_gold: `{k: v for k, v in (kw or {}).items() if v is not None}`), then dumped the same
    way as its dump_gold (sort_keys=True, ensure_ascii=False)."""
    rows = []
    for idx, r in enumerate(ds_rows):
        kwargs = [{k: v for k, v in (kw or {}).items() if v is not None} for kw in r["kwargs"]]
        gold = json.dumps({"instruction_id_list": list(r["instruction_id_list"]), "kwargs": kwargs},
                          sort_keys=True, ensure_ascii=False)
        rows.append({"idx": idx, "prompt": r["prompt"], "gold": gold, "split": "test"})
    return rows


def build_ifeval541(out_dir: str = None) -> dict:
    from datasets import load_dataset
    ds = load_dataset("google/IFEval", split="train")
    assert len(ds) == 541, f"google/IFEval train split has {len(ds)} rows, expected 541"
    return _freeze("ifeval541", _ifeval_rows(ds), out_dir)


# ---------------------------------------------------------------------------
# finqa500 -- dreamerdeo/finqa test, first 500 by common.order_key(id)
#
# Scoring rule (documented here; implemented in analyze_s1b.per_item_correct's "finqa" branch): gold is
# `answer` verbatim (e.g. "94" or "14%"). Numeric match: strip a trailing "%", parse both sides as
# floats, accept if they agree within 1% relative tolerance in EITHER of two forms -- percent-as-written
# (e.g. gold "42%" -> 42) or percent-as-fraction (gold "42%" -> 0.42) -- since a correct model answer to a
# percentage question is written either way about equally often. A non-percentage gold only matches the
# percent-as-written form (no /100 ambiguity). Falls back to case-insensitive exact string match when
# either side does not parse as a number.
# ---------------------------------------------------------------------------
def _finqa_rows(ds_rows) -> list:
    ordered = sorted(list(ds_rows), key=lambda r: common.order_key(str(r["id"])))[:500]
    rows = []
    for idx, r in enumerate(ordered):
        sections = []
        if r.get("pre_text"):
            sections.append("\n".join(r["pre_text"]))
        table_md = markdown_table(r.get("table") or [])
        if table_md:
            sections.append(table_md)
        if r.get("post_text"):
            sections.append("\n".join(r["post_text"]))
        sections.append(r["question"].strip() + "\n" + BOXED_TAIL)
        rows.append({"idx": idx, "prompt": "\n\n".join(sections), "gold": str(r["answer"]).strip(),
                    "split": "test"})
    return rows


def build_finqa500(out_dir: str = None) -> dict:
    from datasets import load_dataset
    import finqa_local                                   # dreamerdeo/finqa is a loading script datasets>=4 refuses
    return _freeze("finqa500", _finqa_rows(finqa_local.rows("test")), out_dir)


# ---------------------------------------------------------------------------
# chartqa300 -- nmayorga7/chartqa-tables JOIN ahmed-masry/ChartQA(test, type == "human") on imgname,
# first 300 by common.order_key("<imgname>|<query>") (ChartQA QA rows carry no native row id; imgname
# alone is not unique since a chart can carry several questions, so the pair is the natural key).
#
# Scoring rule: ChartQA's own "relaxed accuracy" -- numeric answers match within 5% relative tolerance,
# non-numeric answers match case-insensitively (exact, after stripping surrounding whitespace).
# ---------------------------------------------------------------------------
def _chartqa_rows(table_rows, qa_rows) -> list:
    table_csv_by_img = {r["imgname"]: r["table_csv"] for r in table_rows}
    human = [r for r in qa_rows if r.get("type") == "human" and r["imgname"] in table_csv_by_img]
    ordered = sorted(human, key=lambda r: common.order_key(f"{r['imgname']}|{r['query']}"))[:300]
    rows = []
    for idx, r in enumerate(ordered):
        table_md = markdown_table(csv_to_rows(table_csv_by_img[r["imgname"]]))
        prompt = (table_md + "\n\n" + r["query"].strip() + "\n" + BOXED_TAIL) if table_md \
            else (r["query"].strip() + "\n" + BOXED_TAIL)
        rows.append({"idx": idx, "prompt": prompt, "gold": str(r["label"]).strip(), "split": "test"})
    return rows


def build_chartqa300(out_dir: str = None) -> dict:
    from datasets import load_dataset
    tables = load_dataset("nmayorga7/chartqa-tables", split="test")
    qa = load_dataset("ahmed-masry/ChartQA", split="test")
    return _freeze("chartqa300", _chartqa_rows(tables, qa), out_dir)


# ---------------------------------------------------------------------------
# chat800 -- HuggingFaceH4/ultrachat_200k test_sft, first 800 by common.order_key(prompt_id); gold empty
# (no correctness -- these prompts are held out from training via decontam, see analyze_s1b "chat" branch)
# ---------------------------------------------------------------------------
def _chat_rows(ds_rows) -> list:
    ordered = sorted(list(ds_rows), key=lambda r: common.order_key(str(r["prompt_id"])))[:800]
    return [{"idx": idx, "prompt": r["prompt"], "gold": "", "split": "test"}
            for idx, r in enumerate(ordered)]


def build_chat800(out_dir: str = None) -> dict:
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="test_sft")
    return _freeze("chat800", _chat_rows(ds), out_dir)


BUILDERS = {"ifeval541": build_ifeval541, "finqa500": build_finqa500,
           "chartqa300": build_chartqa300, "chat800": build_chat800}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", choices=sorted(BUILDERS) + ["all"], default="all")
    ap.add_argument("--out", default=None, help="default data/c1b/eval/ (common.EVAL_DIR)")
    args = ap.parse_args(argv)
    names = sorted(BUILDERS) if args.set == "all" else [args.set]
    for name in names:
        summary = BUILDERS[name](args.out)
        print(f"[{name}] n={summary['n']} sha256={summary['sha256']} -> {summary['path']}")


if __name__ == "__main__":
    main()
