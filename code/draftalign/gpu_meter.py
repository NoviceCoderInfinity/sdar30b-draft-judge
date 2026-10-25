"""Exact GPU-active time of one decoded item, from the GPU's own hardware counters (NVML GPM, Hopper / Blackwell).

Phase 3 (8-GPU re-run). run_da_decode takes one counter sample immediately BEFORE decode_item and one immediately AFTER
it returns -- both outside decode_item's timed span (synchronize + perf_counter), so the item's `seconds` is untouched.
Between the two samples NVML_GPM_METRIC_GRAPHICS_UTIL is the percentage of time any compute work was active on this GPU
(hardware cycle counters, not sampling); one process runs per GPU, so that is this decode's GPU work (30B and drafter).

Fields added to every record:
  gpu_active_s     GRAPHICS_UTIL x the interval between the two samples: seconds the GPU was executing work
  gpu_idle_s       seconds - gpu_active_s: time the GPU sat idle inside the decode (CPU work, kernel launches, syncs)
  gpu_util_pct, sm_util_pct, sm_occupancy_pct, gpm_interval_s   the raw counter values and interval
  cpu_thread_s, cpu_process_s   CPU time of the decoding thread / the whole process over the same interval (includes
                   CUDA's spin-waiting at synchronisations, so it is not "useful CPU work")
Never raises: without NVML / GPM (or with P3G_GPM=0) it adds {"gpm": "unavailable: <why>"} and nothing else.
"""
from __future__ import annotations

import os
import time

try:
    import pynvml as N
except ImportError:          # pragma: no cover
    N = None


class GpuMeter:
    def __init__(self):
        self.ok = False
        self.why = ""
        if os.environ.get("P3G_GPM", "1") == "0":
            self.why = "disabled (P3G_GPM=0)"; return
        if N is None:
            self.why = "pynvml not installed"; return
        try:
            N.nvmlInit()
            self.h = self._handle()
            if not N.nvmlGpmQueryDeviceSupport(self.h).isSupportedDevice:
                self.why = "GPM not supported on this GPU"; return
            self.s = [N.nvmlGpmSampleAlloc(), N.nvmlGpmSampleAlloc()]
            self.ok = True
        except Exception as e:                       # noqa: BLE001
            self.why = f"{type(e).__name__}: {e}"

    @staticmethod
    def _handle():
        """The NVML handle of the one GPU this process sees: matched by PCI bus id, else CUDA_VISIBLE_DEVICES."""
        try:
            import torch
            p = torch.cuda.get_device_properties(0)
            bus = f"{p.pci_domain_id:08X}:{p.pci_bus_id:02X}:{p.pci_device_id:02X}.0"
            return N.nvmlDeviceGetHandleByPciBusId_v2(bus)
        except Exception:                            # noqa: BLE001
            return N.nvmlDeviceGetHandleByIndex(int(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]))

    def start(self):
        """Call right before decode_item (outside its timed span)."""
        if not self.ok:
            return None
        N.nvmlGpmSampleGet(self.h, self.s[0])
        return (time.perf_counter(), time.thread_time(), time.process_time())

    def stop(self, st, seconds: float) -> dict:
        """Call right after decode_item returns; `seconds` is the item's timed decode."""
        if not self.ok or st is None:
            return {"gpm": f"unavailable: {self.why}"}
        t1, c1, p1 = time.perf_counter(), time.thread_time(), time.process_time()
        try:
            N.nvmlGpmSampleGet(self.h, self.s[1])
            m = N.c_nvmlGpmMetricsGet_t()
            m.version = N.NVML_GPM_METRICS_GET_VERSION
            m.numMetrics = 3
            m.sample1, m.sample2 = self.s[0], self.s[1]
            m.metrics[0].metricId = N.NVML_GPM_METRIC_GRAPHICS_UTIL
            m.metrics[1].metricId = N.NVML_GPM_METRIC_SM_UTIL
            m.metrics[2].metricId = N.NVML_GPM_METRIC_SM_OCCUPANCY
            N.nvmlGpmMetricsGet(m)
        except Exception as e:                       # noqa: BLE001
            return {"gpm": f"error: {type(e).__name__}: {e}"}
        interval = t1 - st[0]
        util = float(m.metrics[0].value)
        active = util / 100.0 * interval
        return {"gpm": "ok", "gpm_interval_s": round(interval, 6), "gpu_util_pct": round(util, 4),
                "sm_util_pct": round(float(m.metrics[1].value), 4), "sm_occupancy_pct": round(float(m.metrics[2].value), 4),
                "gpu_active_s": round(active, 6), "gpu_idle_s": round(seconds - active, 6),
                "cpu_thread_s": round(c1 - st[1], 6), "cpu_process_s": round(p1 - st[2], 6)}
