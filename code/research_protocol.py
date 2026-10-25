#!/usr/bin/env python
"""Frozen conventions for the decode + training experiments in this repository.

Everything here is CPU-only and import-safe: no model is loaded at import.

  load_protocol()              -> (dict, sha256) of code/configs/research_protocol_v1.json
  calibration_items()          -> first 50 frozen GSM8K TRAIN records, with gold
  validation_items()           -> next 50 frozen GSM8K TRAIN records, with gold
  train_items(lo, hi)          -> frozen train records [lo, hi)
  test_items("gsm8k500"|"mbpp")-> the existing frozen test items (run_desvote.build_items)
  items_for_split(name)        -> calib | valid | calibvalid | gsm8k500 | mbpp
  encode_prompt(tok, text)     -> chat-templated input ids exactly as the existing runners
  config_hash(cfg)             -> 16-hex sha256 of the canonical JSON of a config
  safe_out(path)               -> absolute path, refused unless under this repository
  LaunchLog(...)               -> appends launch/exit records to results/audits/research_v1/launches.jsonl

Calibration/validation come from code/configs/prompts.json::gsm8k_train (the same
frozen records the Stage 0 traces and the CPU audit used; lp_b32_blank covers the
first 50). Text sha256 of each question is checked against the frozen record.
Gold answers are the dataset's own `answer` field (scored with the vendored
opencompass gsm8k postprocess, as every existing accuracy number on this branch).
"""
from __future__ import annotations

import atexit, hashlib, json, os, socket, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
os.environ.setdefault("HF_HOME", "/workspace/.hf_home")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
# remote model code is copied by transformers into HF_MODULES_CACHE; keep that OUT of the shared cache
os.environ.setdefault("HF_MODULES_CACHE", os.path.join(ROOT, "hf_cache", "modules"))

PROTOCOL_PATH = os.path.join(HERE, "configs", "research_protocol_v1.json")
PROMPTS_PATH = os.path.join(HERE, "configs", "prompts.json")
FORBIDDEN = ("/workspace/stage0", "/workspace/dlm-general-expert-exp1", "/workspace/.hf_home")

MOE = "JetLM/SDAR-30B-A3B-Chat-b32"
MODEL_REVISION = "c351bbc37d240aa6871f167e8f92d694281b0c22"
# checkpoints: b32 = the block-32 SFT MoE under test (project-local cache); it is the only
# 30B checkpoint this repository uses. d32 = the dense 1.7B block-32 drafter.
MODELS = {
    "b32": {"repo": "JetLM/SDAR-30B-A3B-Chat-b32", "revision": "c351bbc37d240aa6871f167e8f92d694281b0c22",
            "cache_dir": os.path.join(ROOT, "hf_cache", "hub")},
    # d32 = the dense 1.7B block-32 drafter (same tokenizer / MASK id, verified 2026-09-06), project-local cache
    "d32": {"repo": "JetLM/SDAR-1.7B-Chat-b32", "revision": "2c83c05d84be90a01c72443b2123ebf0376f6fd4",
            "cache_dir": os.path.join(ROOT, "hf_cache", "hub")},
}


def load_sdar(model_key: str, device: str, dtype=None):
    """Tokenizer + model for MODELS[model_key], cache only, revision pinned and asserted."""
    import torch
    try:                                             # Phase 2 venv (transformers >= 4.54): LossKwargs shim, else no-op
        from phase2.tf_compat import apply as _tf_compat
        _tf_compat()
    except ImportError:
        pass
    from transformers import AutoModelForCausalLM, AutoTokenizer
    m = MODELS[model_key]
    kw = {"trust_remote_code": True, "local_files_only": True, "revision": m["revision"]}
    if m["cache_dir"]:
        kw["cache_dir"] = m["cache_dir"]
    tok = AutoTokenizer.from_pretrained(m["repo"], **kw)
    assert tok(tok.mask_token)["input_ids"][0] == MASK_ID
    model = AutoModelForCausalLM.from_pretrained(m["repo"], torch_dtype=dtype or torch.bfloat16,
                                                 attn_implementation="sdpa", device_map=device, **kw).eval()
    got = getattr(model.config, "_commit_hash", None)
    assert got in (None, m["revision"]), f"loaded revision {got} != {m['revision']}"
    return tok, model
MASK_ID = 151669
EOS_IDS = [151645, 151643]
GSM8K_TMPL = "{q}\nPlease reason step by step, and put your final answer within \\boxed{{}}."
GEN_LEN = {"gsm8k": 1024, "gsm8k500": 1024, "calib": 1024, "valid": 1024, "calibvalid": 1024, "mbpp": 512,
           "math500": 1024, "aime": 1024, "gpqa": 1024, "humaneval": 512, "gsm8k_pool": 1024, "math_pool": 1024, "mbpp_pool": 512, "opencode_pool": 512, "if_pool": 1024,
           "ifeval541": 1024, "finqa500": 1024, "chartqa300": 1024, "chat800": 1024}
N_CALIB, N_VALID = 50, 50


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def load_protocol():
    raw = open(PROTOCOL_PATH, "rb").read()
    return json.loads(raw), _sha(raw)


def config_hash(cfg: dict) -> str:
    return _sha(json.dumps(cfg, sort_keys=True, separators=(",", ":")).encode())[:16]


def safe_out(path: str) -> str:
    p = path if os.path.isabs(path) else os.path.join(ROOT, path)
    p = os.path.abspath(p)
    assert p.startswith(ROOT + os.sep), f"refusing to write outside {ROOT}: {p}"
    assert not any(p.startswith(b) for b in FORBIDDEN), f"never write into a colleague's tree: {p}"
    return p


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def _gsm8k_split(split: str):
    from datasets import load_dataset
    return load_dataset("openai/gsm8k", "main", split=split)


def train_items(lo: int, hi: int):
    """Frozen GSM8K train records [lo, hi) from code/configs/prompts.json, with gold."""
    raw = open(PROMPTS_PATH, "rb").read()
    spec = json.loads(raw)["sets"]["gsm8k_train"]
    assert spec["split"] == "train" and spec["repo"] == "openai/gsm8k"
    ds = _gsm8k_split("train")
    out = []
    for rank, it in enumerate(spec["items"][lo:hi], start=lo):
        r = ds[int(it["split_index"])]
        q = r["question"]
        h_raw = _sha(q.encode())[:16]
        h_norm = _sha(" ".join(q.split()).encode())[:16]
        assert it["text_sha256_16"] in (h_raw, h_norm), f"frozen text sha mismatch at train rank {rank}"
        out.append({"idx": int(it["split_index"]), "rank": rank, "split": "train",
                    "prompt": GSM8K_TMPL.format(q=q), "gold": r["answer"],
                    "text_sha256_16": it["text_sha256_16"]})
    return out


def calibration_items():
    return train_items(0, N_CALIB)


def validation_items():
    return train_items(N_CALIB, N_CALIB + N_VALID)


def test_items(dataset: str):
    from run_desvote import build_items          # the existing frozen test items
    assert dataset in ("gsm8k500", "mbpp", "math500", "aime", "gpqa", "humaneval")
    items = build_items(dataset)
    for it in items:
        it["split"] = "test"
    return items


# ---------------------------------------------------------------------------
# training-prompt pools (2026-09-06): the full decontaminated train pools of code/configs/prompts.json
# minus every frozen train rank (calibration/validation and the rest of the frozen set). Never test items.
# ---------------------------------------------------------------------------
def _frozen_set(name: str) -> dict:
    return json.loads(open(PROMPTS_PATH, "rb").read())["sets"][name]


def last_boxed(s: str):
    """Content of the last \\boxed{...} in s (brace-balanced), or None."""
    i = s.rfind("\\boxed")
    if i < 0:
        return None
    j = s.find("{", i)
    if j < 0:
        return None
    depth, k = 0, j
    while k < len(s):
        if s[k] == "{":
            depth += 1
        elif s[k] == "}":
            depth -= 1
            if depth == 0:
                return s[j + 1:k]
        k += 1
    return None


def gsm8k_pool_items():
    """Every GSM8K train record whose split index is NOT among the 200 frozen train items; idx = split index."""
    spec = _frozen_set("gsm8k_train")
    frozen = {int(it["split_index"]) for it in spec["items"]}
    ds = _gsm8k_split("train")
    assert len(ds) == spec["pool_size"], (len(ds), spec["pool_size"])
    return [{"idx": i, "rank": i, "split": "train_pool", "prompt": GSM8K_TMPL.format(q=r["question"]),
             "gold": r["answer"]} for i, r in enumerate(ds) if i not in frozen]


def math_pool_items():
    """All 7,500 MATH train problems (EleutherAI/hendrycks_math, subjects concatenated in the frozen order);
    idx = global index; gold = the solution's last boxed answer (scored like math500). Frozen text hashes are checked."""
    from datasets import load_dataset
    spec = _frozen_set("math_train")
    out = []
    for subj in spec["configs"]:
        ds = load_dataset(spec["repo"], subj, split=spec["split"])
        for r in ds:
            g = len(out)
            out.append({"idx": g, "rank": g, "split": "train_pool", "subject": subj, "level": r.get("level"),
                        "prompt": GSM8K_TMPL.format(q=r["problem"]), "gold": last_boxed(r["solution"]) or "",
                        "problem": r["problem"]})
    assert len(out) == spec["pool_size"], (len(out), spec["pool_size"])
    for it in spec["items"]:
        q = out[int(it["global_index"])]["problem"]
        assert it["text_sha256_16"] in (_sha(q.encode())[:16], _sha(" ".join(q.split()).encode())[:16]), \
            f"frozen MATH text sha mismatch at global index {it['global_index']}"
        assert out[int(it["global_index"])]["subject"] == it["subject"]
    for o in out:
        o.pop("problem")
    return out


def mbpp_pool_items():
    """The 120 MBPP-sanitized TRAIN records (prompt format of the frozen MBPP test cell); idx = split index."""
    from datasets import load_dataset
    from run_desvote import MBPP_TMPL
    spec = _frozen_set("mbpp_train")
    ds = load_dataset(spec["repo"], spec["config"], split=spec["split"])
    assert len(ds) == spec["pool_size"], (len(ds), spec["pool_size"])
    out = []
    for i, r in enumerate(ds):
        tl = "\n".join(r["test_list"])
        out.append({"idx": i, "rank": i, "split": "train_pool", "task_id": r["task_id"],
                    "prompt": MBPP_TMPL.format(text=r["prompt"], test_list=tl), "gold": tl})
    return out


def opencode_pool_items():
    """The frozen round-3 code pool: nvidia/OpenCodeInstruct rows in code/configs/opencode_pool.json (built by
    draftalign/build_opencode_pool.py; sha256-checked against its .sha256). Prompts are stored templated (MBPP or
    HumanEval shape); gold = the row's unit-test asserts, scored like mbpp; idx = rank in the frozen hash order.
    split is "opencode_pool", never train/train_pool, so build_train_set's frozen-GSM8K guard cannot fire."""
    path = os.path.join(HERE, "configs", "opencode_pool.json")
    raw = open(path, "rb").read()
    frozen = open(path + ".sha256").read().split()[0]
    assert _sha(raw) == frozen, f"{path}: sha256 {_sha(raw)} != frozen {frozen}"
    spec = json.loads(raw)
    out = [{"idx": it["idx"], "rank": it["idx"], "split": "opencode_pool", "task_id": it["id"],
            "template": it["template"], "prompt": it["prompt"], "gold": it["gold"]} for it in spec["items"]]
    assert [o["idx"] for o in out] == list(range(spec["selection"]["n_select"])), "opencode_pool idx is not 0..N-1"
    return out


IF_POOL_PATH = os.path.join(HERE, "configs", "if_pool.json")


def if_pool_items():
    """The frozen verifiable instruction-following pool: code/configs/if_pool.json (built by
    draftalign/build_if_pool.py; sha256-checked against its .sha256). prompt = the dataset's raw single user message
    (encode_prompt applies the chat template, nothing else is added); gold = JSON {"instruction_id_list", "kwargs"},
    scored by draftalign/if_scorer.py (strict, every constraint must pass). idx = rank in the frozen order.
    split is "if_pool", never train/train_pool, so build_train_set's frozen-GSM8K guard cannot fire."""
    raw = open(IF_POOL_PATH, "rb").read()
    want = open(IF_POOL_PATH + ".sha256").read().split()[0]
    assert _sha(raw) == want, f"{IF_POOL_PATH}: sha256 {_sha(raw)} != {want}"
    out = []
    for it in json.loads(raw)["items"]:
        assert _sha(it["prompt"].encode("utf-8")) == it["text_sha256"], f"if_pool text sha mismatch at idx {it['idx']}"
        out.append({"idx": int(it["idx"]), "rank": int(it["idx"]), "split": "if_pool", "key": it["key"],
                    "prompt": it["prompt"], "gold": it["gold"]})
    assert [o["idx"] for o in out] == list(range(len(out))), "if_pool idx must be 0..n-1 in file order"
    return out


POOLS = {"gsm8k_pool": gsm8k_pool_items, "math_pool": math_pool_items, "mbpp_pool": mbpp_pool_items,
         "opencode_pool": opencode_pool_items, "if_pool": if_pool_items}


# ---------------------------------------------------------------------------
# held-out corpus-1b eval sets (2026-09-28): frozen by corpus1b/eval_sets.py into
# data/c1b/eval/<name>.jsonl (+ .sha256), read back here the same way if_pool_items() reads
# code/configs/if_pool.json -- sha256-checked against the sidecar, idx must be 0..n-1 in file order.
# Never test items in the run_desvote sense (test_items()); registered separately from POOLS because
# these are held-out EVAL sets, not training-prompt pools.
# ---------------------------------------------------------------------------
C1B_EVAL_DIR = os.path.join(ROOT, "data", "c1b", "eval")


def _read_c1b_eval(name: str):
    path = os.path.join(C1B_EVAL_DIR, f"{name}.jsonl")
    raw = open(path, "rb").read()
    want = open(path + ".sha256").read().split()[0]
    assert _sha(raw) == want, f"{path}: sha256 {_sha(raw)} != frozen {want}"
    out = []
    for line in raw.decode("utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        it = json.loads(line)
        out.append({"idx": int(it["idx"]), "rank": int(it["idx"]), "split": it.get("split", "test"),
                    "prompt": it["prompt"], "gold": it["gold"]})
    assert [o["idx"] for o in out] == list(range(len(out))), f"{name}: idx must be 0..n-1 in file order"
    return out


def ifeval541_items():
    """google/IFEval (541), frozen by corpus1b/eval_sets.py::build_ifeval541. Scored via
    draftalign/if_scorer.score_ifeval, dispatched as family "ifeval" (base.startswith("if"))."""
    return _read_c1b_eval("ifeval541")


def finqa500_items():
    """dreamerdeo/finqa test, first 500 by common.order_key(id); frozen by
    corpus1b/eval_sets.py::build_finqa500. Scored via analyze_s1b.per_item_correct's "finqa" branch."""
    return _read_c1b_eval("finqa500")


def chartqa300_items():
    """nmayorga7/chartqa-tables x ahmed-masry/ChartQA(test, human), first 300 by common.order_key;
    frozen by corpus1b/eval_sets.py::build_chartqa300. Scored via analyze_s1b.per_item_correct's
    "chartqa" branch (ChartQA relaxed accuracy)."""
    return _read_c1b_eval("chartqa300")


def chat800_items():
    """HuggingFaceH4/ultrachat_200k test_sft, first 800 by common.order_key(prompt_id); frozen by
    corpus1b/eval_sets.py::build_chat800. gold is always "" -- analyze_s1b.per_item_correct's "chat"
    branch reports no correctness (None) rather than scoring these."""
    return _read_c1b_eval("chat800")


C1B_EVAL_SETS = {"ifeval541": ifeval541_items, "finqa500": finqa500_items,
                 "chartqa300": chartqa300_items, "chat800": chat800_items}


def split_base(name: str) -> str:
    """'math_pool:0:2000' -> 'math_pool' (slice syntax name:lo:hi, python semantics)."""
    return name.split(":")[0]


def items_for_split(name: str):
    if ":" in name:
        base, lo, hi, step = (name.split(":") + ["", "", ""])[:4]
        items = items_for_split(base)
        return items[int(lo) if lo else 0:int(hi) if hi else None:int(step) if step else None]
    if name in POOLS:
        return POOLS[name]()
    if name in C1B_EVAL_SETS:
        return C1B_EVAL_SETS[name]()
    if name == "calib":
        return calibration_items()
    if name == "valid":
        return validation_items()
    if name == "calibvalid":
        return train_items(0, N_CALIB + N_VALID)
    return test_items(name)


def encode_prompt(tok, prompt: str):
    s = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                add_generation_prompt=True, tokenize=False)
    return tok([s], return_tensors="pt", add_special_tokens=False)["input_ids"]


def prompts_config_sha() -> str:
    return _sha(open(PROMPTS_PATH, "rb").read())


# ---------------------------------------------------------------------------
# launch bookkeeping
# ---------------------------------------------------------------------------
def gpu_identity():
    """Physical GPU behind cuda:0 of this process (CUDA_VISIBLE_DEVICES) + name/uuid."""
    vis = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    info = {"cuda_visible_devices": vis, "physical_index": vis.split(",")[0] if vis else None}
    try:
        import torch
        p = torch.cuda.get_device_properties(0)
        info["name"] = p.name
        info["uuid"] = str(getattr(p, "uuid", None))
        info["total_mem_gib"] = round(p.total_memory / 2**30, 1)
    except Exception as e:                 # pragma: no cover
        info["error"] = repr(e)
    return info


class LaunchLog:
    """Append {start} at construction and {end, exit_code} at interpreter exit."""

    def __init__(self, tag: str, cfg: dict, path: str = None):
        self.path = safe_out(path or os.path.join("results", "audits", "research_v1", "launches.jsonl"))
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self.rec = {"tag": tag, "pid": os.getpid(), "host": socket.gethostname(),
                    "command": " ".join(sys.argv), "cwd": os.getcwd(),
                    "config_hash": config_hash(cfg), "config": cfg,
                    "gpu": gpu_identity(),
                    "start_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "t0": time.time()}
        self._write({**self.rec, "event": "start"})
        self.exit_code = None
        atexit.register(self._end)

    def _write(self, rec):
        with open(self.path, "a") as fh:
            fh.write(json.dumps({k: v for k, v in rec.items() if k != "t0"}) + "\n")

    def _end(self):
        code = self.exit_code
        if code is None:
            code = 0 if sys.exc_info()[0] is None else 1
        self._write({**self.rec, "event": "end", "exit_code": code,
                     "end_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                     "elapsed_s": round(time.time() - self.rec["t0"], 1)})


if __name__ == "__main__":
    p, sha = load_protocol()
    print("protocol", PROTOCOL_PATH, "sha256", sha)
    c, v = calibration_items(), validation_items()
    print("calibration", len(c), [it["idx"] for it in c[:5]], "...", "validation", len(v), [it["idx"] for it in v[:5]])
    print("prompts.json sha256", prompts_config_sha())
