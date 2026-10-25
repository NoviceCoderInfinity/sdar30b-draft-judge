"""FinQA without its loading script: `datasets` >= 4 refuses dreamerdeo/finqa (the repo is only a `finqa.py` script
that downloads github.com/czyssrs/FinQA), so decontam.py and eval_sets.py read the upstream JSON directly and map
it exactly as that script's _generate_examples does (id, pre_text, post_text, question, answer, table,
gold_evidence). The file lives under data/c1b/raw/finqa/ (fetched once from raw.githubusercontent.com).
"""
from __future__ import annotations

import json
import os
import urllib.request
from typing import Dict, List

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.abspath(os.path.join(HERE, "..", "..", "data", "c1b", "raw", "finqa"))
URL = "https://raw.githubusercontent.com/czyssrs/FinQA/main/dataset/{split}.json"
SPLITS = {"train": "train", "validation": "dev", "dev": "dev", "test": "test"}


def path_for(split: str) -> str:
    return os.path.join(RAW, f"{SPLITS[split]}.json")


def rows(split: str = "test") -> List[Dict]:
    p = path_for(split)
    if not os.path.exists(p):
        os.makedirs(RAW, exist_ok=True)
        tmp = p + ".part"
        urllib.request.urlretrieve(URL.format(split=SPLITS[split]), tmp)
        os.replace(tmp, p)
    with open(p, encoding="utf-8") as fh:
        data = json.load(fh)
    return [{"id": ex["id"], "post_text": ex["post_text"], "pre_text": ex["pre_text"],
             "question": ex["qa"]["question"], "answer": ex["qa"]["answer"], "table": ex["table"],
             "gold_evidence": list(ex["qa"]["gold_inds"].values())} for ex in data]
