#!/usr/bin/env bash
# Phase 5, judge round: build Phase 1's SGLang environment (/venv/sgl) on a fresh 8x RTX PRO 6000 (Blackwell, sm_120)
# Vast box that has ALREADY run code/loop_guard/setup_lg.sh (that gave /venv/p2, the pinned 30B and 1.7B in hf_cache/hub,
# the merged Phase 4 drafter). Afterwards code/corpus1b/draft_sglang.py (and any sibling script with the same engine call)
# runs with the dLLM algorithm StaticDraftTraced and config code/corpus1b/sglang_plugin/draft_s8.yaml.
#
#   bash code/judge_round/setup_sgl.sh        # from the repo root; needs root (apt) + network; logs itself to logs/phase5/setup_sgl.log
#
# RULE: SGLang is for data generation only; never report a speed measured in it; never train in it.
#
# Every step is a port of a Phase 1 file. Files marked [here] are also on this branch; the others exist only on
# origin/phase1-continue-228m (c228_setup.sh, docs/DATA_GENERATION_CONTINUATION.md) or on both phase1 branches.
#   step 0 preflight  c228_setup.sh section 0 (8 GPUs, free disk) -- code/corpus1b/ops/c228_setup.sh
#   step 1 cuda       c228_setup.sh section 1: CUDA 12.9 toolkit (nvcc + cudart only, never a driver) beside the image's
#                     12.8, because FlashInfer's JIT on sm_120 needs >= 12.9 ("SM 12.x requires CUDA >= 12.9");
#                     docs/DATA_GENERATION_CONTINUATION.md 4.0/4.1 (phase1 branches only)
#   step 2 venv       code/corpus1b/setup_box.sh section 1 [here], the "venv /venv/sgl" block: python3 -m venv, pip install
#                     "sglang[all]==0.5.10.post1", pip install -r code/corpus1b/requirements_sgl_extra.txt [here], the nltk
#                     punkt data. (The /venv/train half of that section is NOT built: Phase 5 uses /venv/p2.)
#   step 3 plugin     code/corpus1b/sglang_plugin/install.py [here], run by setup_box.sh right after the venv: COPIES
#                     c1b_traced.py (LowConfidenceTraced) and c1b_static_draft.py (StaticDraftTraced) into
#                     <site-packages>/sglang/srt/dllm/algorithm/, where SGLang discovers any module defining `Algorithm`
#                     (no pip install -e, no symlink, no env var), and applies the ONE approved SGLang edit
#                     (jit_kernel/rope.py: cast out_loc to positions.dtype, .orig backup). Then `install.py --check`.
#                     The algorithm is selected at engine launch by dllm_algorithm="StaticDraftTraced" +
#                     dllm_algorithm_config=.../draft_s8.yaml (see code/corpus1b/draft_sglang.py).
#   step 4 smoke      the engine call of code/corpus1b/draft_sglang.py [here] on the 1.7B pinned snapshot
#                     (research_protocol.MODELS["d32"] via gen_sglang.snapshot_dir), one block, with Phase 1's draft
#                     settings (run_wave_a.sh DRAFT_ENV/DRAFT_FLAGS: workspace 4 GiB, --max-running 256, --mem-fraction 0.70).
#                     Phase 1's own smoke (c228_setup.sh section 4) decodes GSM8K-500 with the 30B; this one does not
#                     need the 30B. It also warms FlashInfer's JIT cache (~/.cache/flashinfer) for the 1.7B on sm_120.
#
# Versions, as recorded in the repo:
#   sglang       0.5.10.post1  PINNED  setup_box.sh `pip install "sglang[all]==0.5.10.post1"`; install.py PINNED (refuses others)
#   torch        2.9.1         NOT pinned by any pip command: it is what pip resolved as a dependency of sglang in Phase 1
#                              (docs/DATA_GENERATION_CONTINUATION.md 4.0: "/venv/sgl: sglang 0.5.10.post1, torch 2.9.1").
#                              This script does not pin it either; it WARNS if the resolved torch differs.
#   flashinfer   NOT RECORDED anywhere in the repo or its docs. It is whatever sglang==0.5.10.post1 pulls in.
#                TODO: after the first good run, commit logs/phase5/sgl_setup/sgl_freeze.txt (written by step 2) so the
#                flashinfer / sgl-kernel versions of a working box are on record. Do not guess a pin before that.
#   CUDA toolkit 12.9          apt cuda-nvcc-12-9 (12.9.86-1 on the Phase 1 box), cuda-cudart-dev-12-9 (12.9.79-1); installed
#                              unpinned by c228_setup.sh. This script does the same.
# Environment variables Phase 1 used for SGLang (run_wave_a.sh / c228_run.sh GEN_ENV, DRAFT_ENV):
#   CUDA_HOME=/usr/local/cuda-12.9 PATH=/usr/local/cuda-12.9/bin:$PATH     (written to logs/phase5/sgl_setup/sgl_env.sh)
#   SGLANG_FLASHINFER_WORKSPACE_SIZE=4294967296 for drafts (2147483648 for the 30B generation); CUDA-graph capture needs it
#   C1B_TRACE_DIR=<dir>   set by draft_sglang.py / gen_sglang.py themselves before the engine starts (the plugin's trace sink)
#   Phase 1's DRAFT_FLAGS: --max-running 256 --mem-fraction 0.70 (512 running x 32 slots x 152K-vocab fp32 logits OOMs at 0.80)
#
# Re-runnable: each finished step leaves logs/phase5/sgl_setup/.done_<step> holding a stamp; a step is skipped only if the
# stamp matches AND a probe of the real state agrees (the Phase 3 box once skipped its whole setup because .done_* markers
# had been committed to git; a marker alone never skips anything here). FORCE=1 ignores all markers.
# Knobs: ROOT, SGL_VENV (default /venv/sgl), NGPU (default 8), SMOKE_GPU (0), SMOKE_MAX_RUNNING (256),
#        SMOKE_MEM_FRACTION (0.70), SGLANG_FLASHINFER_WORKSPACE_SIZE (4294967296), SMOKE_TIMEOUT (1500 s), FORCE.
set -Eeuo pipefail
ROOT=${ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}
cd "$ROOT"
VENV=${SGL_VENV:-/venv/sgl}; PYS=$VENV/bin/python
NGPU=${NGPU:-8}
L=logs/phase5; S=$L/sgl_setup; mkdir -p "$S"
export TOKENIZERS_PARALLELISM=false PIP_DISABLE_PIP_VERSION_CHECK=1
PLUG=code/corpus1b/sglang_plugin
TIMES=(); CUR=preflight; T_ALL=$(date +%s)
say(){ echo "[sgl-setup $(date +%H:%M:%S)] $*" | tee -a "$L/setup_sgl.log"; }
fail(){ say "FAILED: $*"; exit 1; }
trap 'rc=$?; say "FAILED (exit $rc) in step ${CUR}, line $LINENO; see $L/setup_sgl.log and $S/"' ERR

# ---- step runner: skip only if the marker stamp matches AND the probe of the real state passes ------------------------
done_ok(){ # name stamp [probe_fn]
  [ "${FORCE:-0}" = 1 ] && return 1
  [ -f "$S/.done_$1" ] && [ "$(cat "$S/.done_$1")" = "$2" ] || return 1
  [ -z "${3:-}" ] || "$3"
}
step(){ # name stamp run_fn [probe_fn]
  local n=$1 st=$2 fn=$3 probe=${4:-} t0; t0=$(date +%s); CUR=$n
  if done_ok "$n" "$st" "$probe"; then
    say "[$n] already done (marker and probe agree): skipped"; TIMES+=("$n|skipped|$(( $(date +%s) - t0 ))"); return 0
  fi
  say "[$n] start"
  "$fn"
  echo "$st" > "$S/.done_$n"
  say "[$n] done in $(( $(date +%s) - t0 )) s"; TIMES+=("$n|ran|$(( $(date +%s) - t0 ))")
}
timed(){ # name fn : a step that always runs (its own detection is the idempotence check)
  local n=$1 fn=$2 t0; t0=$(date +%s); CUR=$n; say "[$n] start"; "$fn"
  say "[$n] done in $(( $(date +%s) - t0 )) s"; TIMES+=("$n|ran|$(( $(date +%s) - t0 ))")
}

# ---- step 0: the box (c228_setup.sh section 0) -----------------------------------------------------------------------
s_preflight(){
  [ "$(uname -m)" = x86_64 ] || fail "x86_64 only (the NVIDIA apt repo URL below is x86_64)"
  say "GPUs:"; nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader | tee "$S/gpus.txt"
  [ "$(nvidia-smi -L | wc -l)" -ge "$NGPU" ] || fail "need $NGPU GPUs"
  local used; used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "${SMOKE_GPU:-0}" | tr -dc 0-9)
  [ "${used:-0}" -lt 2000 ] || say "WARNING: GPU ${SMOKE_GPU:-0} already uses ${used} MiB; the smoke (mem-fraction ${SMOKE_MEM_FRACTION:-0.70}) may OOM"
  say "CPU: $(lscpu | grep -m1 'Model name' | sed 's/.*: *//') x $(nproc) threads; RAM $(free -g | awk '/Mem:/{print $2}') GB"
  say "free disk: /venv $(df -BG --output=avail "$(dirname "$VENV")" 2>/dev/null | tail -1 | tr -dc 0-9)G, repo $(df -BG --output=avail . | tail -1 | tr -dc 0-9)G (sglang[all] + torch need about 15 GB)"
  say "python3: $(python3 --version 2>&1)"
  for f in "$PLUG/install.py" "$PLUG/c1b_traced.py" "$PLUG/c1b_static_draft.py" "$PLUG/draft_s8.yaml" \
           code/corpus1b/requirements_sgl_extra.txt code/corpus1b/gen_sglang.py code/corpus1b/draft_sglang.py; do
    [ -f "$f" ] || fail "missing $f (is this the phase5 branch with code/corpus1b?)"
  done
  # the 1.7B pinned snapshot must be on disk (setup_lg.sh -> setup_p3_8gpu.sh -> setup_p2.sh put it there)
  local py0=python3; [ -x /venv/p2/bin/python ] && py0=/venv/p2/bin/python
  D32=$("$py0" -c 'import sys; sys.path.insert(0, "code/corpus1b"); import gen_sglang as G; print(G.snapshot_dir("d32"))') \
    || fail "the pinned SDAR-1.7B-Chat-b32 snapshot is not in hf_cache/hub: run code/loop_guard/setup_lg.sh first"
  say "1.7B drafter snapshot: $D32"
}

# ---- step 1: CUDA >= 12.9 toolkit beside the image's 12.8 (c228_setup.sh section 1) ----------------------------------
# Needed only if no 12.x toolkit >= 12.9 exists. Detected, not assumed: Phase 1 tested only /usr/local/cuda-12.9.
CUDA_HOME=""
cuda_release(){ "$1" --version 2>/dev/null | sed -n 's/.*release \([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' | tail -1; }
cuda_ok(){ local v; v=$(cuda_release "$1") || true; [ -n "$v" ] || return 1
           [ "${v%%.*}" = 12 ] && [ "$(printf '%s\n%s\n' 12.9 "$v" | sort -V | head -1)" = 12.9 ]; }
find_cuda(){ local n
  for n in /usr/local/cuda-12.9/bin/nvcc "$(command -v nvcc || true)" /usr/local/cuda/bin/nvcc; do
    if [ -n "$n" ] && [ -x "$n" ] && cuda_ok "$n"; then CUDA_HOME=$(dirname "$(dirname "$n")"); return 0; fi
  done
  return 1; }
install_cuda129(){
  [ "$(id -u)" = 0 ] || fail "CUDA 12.9 toolkit missing and this is not root (needs apt-get)"
  say "installing cuda-nvcc-12-9 + cuda-cudart-dev-12-9 (toolkit only, never a driver)"
  export DEBIAN_FRONTEND=noninteractive
  # Phase 1 grepped for "Candidate"; that also matches "Candidate: (none)", so here a real candidate is required
  if ! apt-cache policy cuda-nvcc-12-9 2>/dev/null | grep -q 'Candidate: [^(]'; then
    . /etc/os-release; local ub="ubuntu${VERSION_ID//./}"
    curl -fsSL -o /tmp/cuda-keyring.deb "https://developer.download.nvidia.com/compute/cuda/repos/${ub}/x86_64/cuda-keyring_1.1-1_all.deb" || return 1
    dpkg -i /tmp/cuda-keyring.deb || return 1
    apt-get update -qq || return 1
  fi
  apt-get install -y -qq cuda-nvcc-12-9 cuda-cudart-dev-12-9 || return 1
}
s_cuda(){
  if find_cuda; then say "CUDA toolkit already ok: $CUDA_HOME ($("$CUDA_HOME/bin/nvcc" --version | tail -1))"
  else
    say "no CUDA 12.x toolkit >= 12.9 found (nvcc: $(command -v nvcc || echo none); /usr/local/cuda: $(cuda_release /usr/local/cuda/bin/nvcc || true))"
    install_cuda129 || fail "CUDA 12.9 install failed. Phase 1's documented fallback is --attention-backend triton --disable-cuda-graph (about half the decode speed, docs/DATA_GENERATION_CONTINUATION.md 4.0); it is not wired into this script"
    find_cuda || fail "cuda-nvcc-12-9 installed but no usable nvcc found"
    say "installed: $("$CUDA_HOME/bin/nvcc" --version | tail -1)"
  fi
  echo "$CUDA_HOME" > "$S/.done_cuda"
  # what every SGLang launcher must run under (Phase 1: GEN_ENV / DRAFT_ENV in run_wave_a.sh and c228_run.sh)
  cat > "$S/sgl_env.sh" <<EOF
# written by code/judge_round/setup_sgl.sh: source this (or copy the two exports) before launching any SGLang process
export CUDA_HOME=$CUDA_HOME
export PATH=$CUDA_HOME/bin:\$PATH
# plus, for drafts, SGLANG_FLASHINFER_WORKSPACE_SIZE=4294967296 and --max-running 256 --mem-fraction 0.70 (Phase 1 DRAFT_ENV/DRAFT_FLAGS)
EOF
}

# ---- step 2: /venv/sgl (setup_box.sh section 1, the sgl block) ---------------------------------------------------------
s_venv(){
  say "venv $VENV (sglang 0.5.10.post1)"
  python3 -m venv "$VENV"
  "$PYS" -m pip install -q -U pip wheel
  "$PYS" -m pip install -q "sglang[all]==0.5.10.post1" 2>&1 | tee -a "$S/pip_sglang.log"
  "$PYS" -m pip install -q -r code/corpus1b/requirements_sgl_extra.txt 2>&1 | tee -a "$S/pip_sglang.log"
  "$PYS" -m nltk.downloader -q -d /root/nltk_data punkt punkt_tab || true       # as Phase 1 (the IF checker; harmless here)
  "$PYS" -m pip freeze > "$S/sgl_freeze.txt"
  "$PYS" - <<'PYEOF' | tee "$S/sgl_versions.txt"
import importlib.metadata as md, sys
import torch, sglang
def v(n):
    try:
        return md.version(n)
    except md.PackageNotFoundError:
        return "-"
print("python", sys.version.split()[0])
print("sglang", getattr(sglang, "__version__", "?"), "| sgl-kernel", v("sgl-kernel"), "| flashinfer-python", v("flashinfer-python"),
      "| flashinfer-cubin", v("flashinfer-cubin"), "| transformers", v("transformers"))
print("torch", torch.__version__, "| torch cuda", torch.version.cuda)
a = torch.cuda.get_arch_list()
print("torch arch list", a)
assert "sm_120" in a, f"this torch has no sm_120 kernels: {a}"
if torch.__version__.split("+")[0] != "2.9.1":
    print("WARNING: Phase 1's /venv/sgl had torch 2.9.1 (docs/DATA_GENERATION_CONTINUATION.md 4.0); this one has", torch.__version__)
PYEOF
  say "full freeze: $S/sgl_freeze.txt (flashinfer was never recorded in Phase 1: commit this file after the first good run)"
}
probe_venv(){ [ -x "$PYS" ] && "$PYS" -c 'import importlib.metadata as m, sys; sys.exit(0 if m.version("sglang") == "0.5.10.post1" else 1)' 2>/dev/null; }

# ---- step 3: register the plugin (sglang_plugin/install.py, as setup_box.sh did right after the venv) --------------
sgl_pkg(){ "$PYS" -c 'import importlib.util as u, os; print(os.path.dirname(u.find_spec("sglang").origin))' 2>/dev/null; }
s_plugin(){
  "$PYS" "$PLUG/install.py"
  "$PYS" "$PLUG/install.py" --check
}
probe_plugin(){ local pkg; pkg=$(sgl_pkg) || return 1
  cmp -s "$PLUG/c1b_traced.py" "$pkg/srt/dllm/algorithm/c1b_traced.py" \
    && cmp -s "$PLUG/c1b_static_draft.py" "$pkg/srt/dllm/algorithm/c1b_static_draft.py" \
    && grep -q "c1b patch" "$pkg/jit_kernel/rope.py"; }

# ---- step 4: smoke without the 30B: the engine call of draft_sglang.py, one block --------------------------------------
cat > "$S/smoke_sgl.py" <<'PYEOF'
"""Setup smoke: SGLang + StaticDraftTraced on the pinned SDAR-1.7B-Chat-b32, one teacher-forced block for a dummy prompt.
The Engine arguments are those of code/corpus1b/draft_sglang.py; the request is block 0 of draft_sglang.block_requests.
This checks that the environment works. It measures no speed and its output is not data."""
import os
import shutil
import sys
import time

ROOT = os.environ["ROOT"]


def main() -> None:
    sys.path.insert(0, os.path.join(ROOT, "code", "corpus1b"))
    import gen_sglang as G                                       # first: it puts code/ and code/compat on sys.path
    import common as C
    import research_protocol as P
    BLOCK = C.BLOCK
    trace_dir = os.environ["SMOKE_TRACE_DIR"]
    shutil.rmtree(trace_dir, ignore_errors=True)
    os.makedirs(trace_dir)
    os.environ["C1B_TRACE_DIR"] = trace_dir                      # inherited by SGLang's scheduler process (the plugin's sink)
    d32 = G.snapshot_dir("d32")
    tok = G.load_tokenizer("d32")
    prompt_ids = P.encode_prompt(tok, "What is 17 times 3? Answer in one sentence.")[0].tolist()
    L = len(prompt_ids)
    lo0 = (L // BLOCK) * BLOCK                                   # as gen_sglang: the block holding the prompt tail
    tail = L - lo0
    max_new = BLOCK - tail                                       # exactly one decode block
    print(f"[smoke] prompt {L} tokens, block at lo0={lo0}, prompt tail {tail}, max_new_tokens {max_new}", flush=True)

    import sglang as sgl
    t0 = time.time()
    engine = sgl.Engine(
        model_path=d32, tokenizer_path=d32, trust_remote_code=True,
        skip_tokenizer_init=True, dtype="bfloat16", random_seed=0, log_level="warning",
        dllm_algorithm="StaticDraftTraced",
        dllm_algorithm_config=os.path.join(ROOT, "code", "corpus1b", "sglang_plugin", "draft_s8.yaml"),
        max_running_requests=int(os.environ["SMOKE_MAX_RUNNING"]), mem_fraction_static=float(os.environ["SMOKE_MEM_FRACTION"]),
        context_length=4096, disable_cuda_graph=False)
    print(f"[smoke] engine up after {time.time() - t0:.0f} s (includes FlashInfer JIT and CUDA-graph capture on the first run)", flush=True)
    try:
        reader = G.TraceReader(trace_dir)
        engine.generate(input_ids=[prompt_ids],
                        sampling_params=[{"temperature": 0.0, "max_new_tokens": max_new, "ignore_eos": True}],
                        rid=["smoke#0"])
        got = None
        for _ in range(100):                                     # the scheduler appends the trace line; give it a moment
            reader.poll()
            got = reader.take("smoke#0").get(lo0)
            if got is not None:
                break
            time.sleep(0.1)
        assert got is not None, f"no trace row for block lo={lo0} in {trace_dir} (is StaticDraftTraced the algorithm that ran?)"
        ids, conf, start = got["ids"], got["conf"], int(got["start"])
        assert len(ids) == BLOCK and len(conf) == BLOCK, (len(ids), len(conf))
        assert start == tail, (start, tail)
        assert ids[:start] == prompt_ids[lo0:L], "prompt tail was altered"
        drafted = ids[start:]
        assert drafted and all(t != C.MASK_ID and t >= 0 for t in drafted), "a drafted slot is still [MASK]"
        assert all(0.0 <= c <= 1.0 for c in conf[start:]), conf[start:]
        assert all(c == -1.0 for c in conf[:start]), conf[:start]
        print("[smoke] drafted:", repr(tok.decode(drafted)), "| confidences min/max", round(min(conf[start:]), 3), round(max(conf[start:]), 3), flush=True)
    finally:
        engine.shutdown()
    print("sgl smoke ok", flush=True)


if __name__ == "__main__":
    main()
PYEOF
s_smoke(){
  local gpu=${SMOKE_GPU:-0}
  rm -f "$S/smoke.log"
  CUDA_VISIBLE_DEVICES=$gpu ROOT="$ROOT" SMOKE_TRACE_DIR="$ROOT/$S/smoke_trace" \
    SMOKE_MAX_RUNNING=${SMOKE_MAX_RUNNING:-256} SMOKE_MEM_FRACTION=${SMOKE_MEM_FRACTION:-0.70} \
    CUDA_HOME="$CUDA_HOME" PATH="$CUDA_HOME/bin:$PATH" \
    SGLANG_FLASHINFER_WORKSPACE_SIZE=${SGLANG_FLASHINFER_WORKSPACE_SIZE:-4294967296} \
    timeout "${SMOKE_TIMEOUT:-1500}" "$PYS" "$S/smoke_sgl.py" 2>&1 | tee "$S/smoke.log"
  grep -q '^sgl smoke ok$' "$S/smoke.log" || fail "the smoke did not print 'sgl smoke ok' (see $S/smoke.log). If it is a FlashInfer/sm_120 error, check CUDA_HOME=$CUDA_HOME"
}

# ---- run ---------------------------------------------------------------------------------------------------------------
timed preflight s_preflight
timed cuda s_cuda                                      # sets CUDA_HOME; installs only when no toolkit >= 12.9 exists
EXTRA_SHA=$(sha256sum code/corpus1b/requirements_sgl_extra.txt | cut -c1-12)
step venv "sglang==0.5.10.post1;extra=$EXTRA_SHA" s_venv probe_venv
PLUG_SHA=$(cat "$PLUG/install.py" "$PLUG/c1b_traced.py" "$PLUG/c1b_static_draft.py" "$PLUG/draft_s8.yaml" | sha256sum | cut -c1-12)
step plugin "plugin=$PLUG_SHA" s_plugin probe_plugin
GPU0=$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader -i "${SMOKE_GPU:-0}" | tr -d ' ')
if [ "${SKIP_SMOKE:-0}" = 1 ]; then say "[smoke] SKIPPED (SKIP_SMOKE=1, operator 23 Oct: install only, no smoke runs)"; else
step smoke "plugin=$PLUG_SHA;cuda=$CUDA_HOME;host=$(hostname);gpu=$GPU0" s_smoke
fi

# ---- wall time of each step ---------------------------------------------------------------------------------------------
say "step wall times:"
{ for t in "${TIMES[@]}"; do IFS='|' read -r n what dt <<<"$t"; printf '  %-10s %-8s %6s s\n' "$n" "$what" "$dt"; done
  printf '  %-10s %-8s %6s s\n' total "" "$(( $(date +%s) - T_ALL ))"; } | tee -a "$L/setup_sgl.log"
say "SGL SETUP DONE. $PYS has the StaticDraftTraced plugin; sgl smoke ok."
say "Launch every SGLang process under: CUDA_HOME=$CUDA_HOME PATH=$CUDA_HOME/bin:\$PATH (file: $S/sgl_env.sh)"
say "and, for drafts, SGLANG_FLASHINFER_WORKSPACE_SIZE=4294967296 --max-running 256 --mem-fraction 0.70 as in Phase 1 (draft_sglang.py defaults 512/0.80 OOM)."
say "Reminder: SGLang is for data generation only; never report a speed measured in it; never train in it."
