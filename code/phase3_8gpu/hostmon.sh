#!/usr/bin/env bash
# Host snapshot between runs (never during one): GPU clock-event counters, cgroup CPU throttling, load, GPU processes.
cd "$(dirname "$0")/../.."
L=logs/phase3_8gpu; mkdir -p $L/hostmon
f=$L/hostmon/$(date -u +%Y%m%dT%H%M%SZ)_$1.txt
{ echo "utc $(date -u +%Y-%m-%dT%H:%M:%SZ)"; echo "loadavg $(cat /proc/loadavg)"
  grep -E "^(nr_periods|nr_throttled|throttled_usec)" /sys/fs/cgroup/cpu.stat
  nvidia-smi --query-gpu=index,clocks_event_reasons.active,clocks_event_reasons_counters.sw_power_cap,clocks_event_reasons_counters.sw_thermal_slowdown,clocks_event_reasons_counters.hw_thermal_slowdown,clocks_event_reasons_counters.hw_power_brake_slowdown,temperature.gpu,power.draw,clocks.sm,clocks.mem,pstate,memory.used --format=csv
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv; } > $f 2>&1
