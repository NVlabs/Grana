# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Optional per-update host timers and CUDA event spans, without phase barriers."""
from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
import time

import torch


class UpdateProfiler:
    """Measure a bounded number of real training updates.

    Host durations include dispatch and any existing synchronization. CUDA
    spans include dispatch gaps between the events, rather than summed kernel
    time. Host and CUDA measurements overlap and must not be added together.
    The caller must synchronize the update before reading metrics.
    """

    def __init__(self, *, enabled: bool, device):
        self.enabled = enabled
        self.device = torch.device(device)
        self._host = defaultdict(float)
        self._events = defaultdict(list)

    @contextmanager
    def section(self, name: str, *, cuda: bool = False):
        if not self.enabled:
            yield
            return
        events = None
        if cuda and self.device.type == "cuda":
            with torch.cuda.device(self.device):
                events = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
                events[0].record()
        started = time.perf_counter()
        try:
            yield
        finally:
            self._host[name] += time.perf_counter() - started
            if events is not None:
                with torch.cuda.device(self.device):
                    events[1].record()
                self._events[name].append(events)

    def metrics(self) -> dict[str, float]:
        values = {f"time/profile/{name}_host_seconds": seconds for name, seconds in self._host.items()}
        for name, pairs in self._events.items():
            values[f"time/profile/{name}_cuda_span_seconds"] = sum(start.elapsed_time(end) for start, end in pairs) / 1000
        return values
