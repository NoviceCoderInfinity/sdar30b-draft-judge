#!/bin/bash
# Phase 3 8-GPU re-run, STEP 0 machine verification. Writes every check's raw output.
# Usage: machine_check.sh <outfile>
OUT="$1"
mkdir -p "$(dirname "$OUT")"
exec > "$OUT" 2>&1
sec() { echo; echo "=================================================================="; echo "== $*"; echo "== $(date -u +%Y-%m-%dT%H:%M:%SZ)"; echo "=================================================================="; }

echo "Phase 3 re-draft, 8-GPU machine check. Started $(date -u +%Y-%m-%dT%H:%M:%SZ) (UTC)"
echo "Note: model downloads (network/disk only, no GPU) were in progress during this check."

sec "(a) nvidia-smi -L"
nvidia-smi -L
echo "GPU count: $(nvidia-smi -L | wc -l)"
echo "Distinct models: $(nvidia-smi --query-gpu=name --format=csv,noheader | sort | uniq -c)"

sec "(b1) vastai show instance (selected non-secret fields only)"
timeout 60 vastai show instance "$CONTAINER_ID" --api-key "$CONTAINER_API_KEY" --raw 2>&1 | python3 -c "
import json,sys
d=json.load(sys.stdin)
if isinstance(d,list): d=d[0]
for k in ['id','machine_id','host_id','gpu_name','num_gpus','gpu_frac','gpu_totalram','gpu_ram','cpu_name','cpu_cores','cpu_cores_effective','cpu_ram','mem_limit','disk_space','driver_version','cuda_max_good','compute_cap','mobo_name','pci_gen','gpu_lanes','pcie_bw','gpu_mem_bw','bw_nvlink','geolocation','verification','gpu_util','cpu_util','cur_state']:
    print(f'{k}: {d.get(k)}')
"

sec "(b2) nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
echo "--- per-GPU memory used / utilization / persistence ---"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,persistence_mode,pci.bus_id --format=csv

sec "(b3) nvidia-smi -q | grep -iE 'MIG Mode|Virtualization' (with following lines)"
nvidia-smi -q | grep -iE -A2 "MIG Mode|Virtualization"

sec "(b4) CPU: nproc, lscpu, cgroup cpu.max, cpu.stat, cpuset, affinity"
echo "nproc: $(nproc)"
lscpu
echo "--- /sys/fs/cgroup/cpu.max ---"; cat /sys/fs/cgroup/cpu.max
echo "--- /sys/fs/cgroup/cpu.stat ---"; cat /sys/fs/cgroup/cpu.stat
echo "--- /sys/fs/cgroup/cpuset.cpus.effective ---"; cat /sys/fs/cgroup/cpuset.cpus.effective 2>&1
echo "--- /sys/fs/cgroup/memory.max ---"; cat /sys/fs/cgroup/memory.max 2>&1
echo "--- affinity of this shell ---"; taskset -p $$

sec "(b5) uptime / load average (before the 60 s GPU watch)"
uptime; cat /proc/loadavg

sec "(b6) nvidia-smi dmon -s u -c 60 (60 x 1 s samples while idle)"
nvidia-smi dmon -s u -c 60
echo "--- uptime after the 60 s watch ---"; uptime

sec "(c) nvidia-smi -q -d PERFORMANCE,CLOCK"
nvidia-smi -q -d PERFORMANCE,CLOCK

sec "(c2) power limits and persistence: nvidia-smi -q -d POWER; persistence query"
nvidia-smi -q -d POWER
nvidia-smi --query-gpu=index,persistence_mode,power.limit,power.default_limit,power.max_limit,power.draw,clocks.sm,clocks.mem,clocks.max.sm,clocks.max.mem,temperature.gpu,pstate --format=csv

sec "(c3) enable persistence mode (nvidia-smi -pm 1)"
nvidia-smi -pm 1 2>&1
nvidia-smi --query-gpu=index,persistence_mode --format=csv

sec "(d) driver / CUDA / torch / transformers versions"
nvidia-smi | head -4
nvcc --version 2>&1 | tail -2
/venv/main/bin/python -c "
import sys, platform
print('python', sys.version.split()[0], platform.platform())
try:
    import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'cudnn', torch.backends.cudnn.version(), 'is_available', torch.cuda.is_available(), 'device_count', torch.cuda.device_count())
    print('arch list', torch.cuda.get_arch_list())
except Exception as e: print('torch import error:', e)
try:
    import transformers; print('transformers', transformers.__version__)
except Exception as e: print('transformers import error:', e)
for m in ['accelerate','peft','numpy','safetensors','tokenizers','huggingface_hub','flash_attn']:
    try:
        mod=__import__(m); print(m, getattr(mod,'__version__','?'))
    except Exception as e: print(m, 'not installed')
"

sec "(d2) GPU topology: nvidia-smi topo -m"
nvidia-smi topo -m

echo; echo "Finished $(date -u +%Y-%m-%dT%H:%M:%SZ) (UTC)"
