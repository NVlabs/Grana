# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded CPU read-ahead without changing the independent A/B/C draw stream."""
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import time


def forward_role_requests(roles: dict) -> tuple[tuple[str, ...], ...]:
    """A cycle of requests; both C lanes materialize as the same loss role C."""
    if roles["D"]:
        independent = roles["C"] - roles["D"]
        if independent < 0 or roles["A"] % 2 or roles["B"] % 2:
            raise ValueError("Paired schedule requires even A/B and C>=D")
        ab = tuple(role for role in ("A", "B") for _ in range(roles[role] // 2))
        return (ab + ("C_paired",) * roles["D"] + ("C_independent",) * ((independent + 1) // 2),
                ab + ("D",) * roles["D"] + ("C_independent",) * (independent // 2))
    return (tuple(role for role in ("A", "B", "C") for _ in range(roles[role])),)


def independent_role_order(roles: dict, accumulation: int) -> tuple[str, ...]:
    """Match actual sample calls, preserving lanes and excluding derived D."""
    if roles["D"] and accumulation % 2:
        raise ValueError("D requires complete two-forward cycles")
    phases = forward_role_requests(roles)
    return tuple(request for phase in phases for request in phase if request != "D") * (accumulation // len(phases))


class SamplePrefetcher:
    """One CPU producer and at most one future update ahead of the consumer.

    Only this producer calls sampler.sample when enabled. The sampler owns
    independent Python RNGs, so no model/dropout RNG is consumed in the thread.
    The main thread retains C-to-D construction and all consumed-row accounting.
    No CUDA work runs here. Queued RGB memory is bounded by one local update,
    and the finite update budget prevents extra draws at normal completion.
    """

    def __init__(self, sampler, roles, accumulation, updates, *, enabled=False):
        self.sampler = sampler
        self.enabled = enabled
        self.order = independent_role_order(roles, accumulation)
        if type(enabled) is not bool or type(updates) is not int or updates < 1 or not self.order:
            raise ValueError("Prefetch requires a boolean mode, positive updates and independent rows")
        self.updates = updates
        self.submitted = 0
        self.consumed = 0
        self.prepare_seconds = 0.0
        self.current = deque()
        self.pending = None
        self.executor = None
        self.closed = False

    def __enter__(self):
        if self.closed or self.executor is not None:
            raise RuntimeError("A sample prefetcher cannot be reopened")
        if self.enabled:
            self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="grana-sample-prefetch")
            self._submit()
        return self

    def _prepare(self):
        started = time.perf_counter()
        rows = [self.sampler.sample(role) for role in self.order]
        return rows, time.perf_counter() - started

    def _submit(self):
        if self.submitted < self.updates:
            self.pending = self.executor.submit(self._prepare)
            self.submitted += 1

    def sample(self, role):
        if self.closed or self.consumed >= len(self.order) * self.updates:
            raise RuntimeError("Sample request exceeds the open training budget")
        expected = self.order[self.consumed % len(self.order)]
        if role != expected:
            raise RuntimeError(f"Training sample order changed: expected {expected}, got {role}")
        if not self.enabled:
            row = self.sampler.sample(role)
        else:
            if self.executor is None:
                raise RuntimeError("Use the sample prefetcher as a context manager")
            if not self.current:
                rows, self.prepare_seconds = self.pending.result()
                self.pending = None
                self.current.extend(rows)
                self._submit()
            row = self.current.popleft()
        self.consumed += 1
        return row

    def close(self):
        self.closed = True
        try:
            if self.executor is not None:
                self.executor.shutdown(wait=True, cancel_futures=True)
        finally:
            self.current.clear()
            self.pending = None
            self.executor = None

    def __exit__(self, *exc):
        self.close()
