# Engineering defaults

- Keep source, tests and documentation portable. Use user-selected paths and
  explicit configuration; do not commit datasets, checkpoints, credentials,
  machine inventories or local execution receipts.
- Read the contributor guide before changes. Preserve upstream notices and
  sign off commits. Keep runtime dependency pins and model compatibility
  constraints unless the requested change includes validating a replacement.
- Test the changed path with local fixtures before expensive work. Configuration
  loading and dry-runs must not require a GPU, model download or private data.
  Distinguish schema validation, data validation and executed training evidence.

# Training defaults

- Enable gradient checkpointing by default for training, smoke tests, profiling,
  memory/throughput benchmarks and timing measurements. Keep it enabled when
  comparing batch sizes or gradient accumulation; compare it disabled only when
  explicitly requested.
- Preserve role budgets, loss normalization, frame sampling and metric definitions
  when changing resource use. Record the resolved configuration, source/checkpoint
  identity, actual batch sizes and device allocation with each executed run.
- Select devices, worker counts, concurrency and memory budgets for the current
  environment. Explicit user limits take precedence. Do not assume ownership of
  every visible GPU or a particular machine or accelerator count.

# Inference and evaluation

- For latency benchmarks, default to physical batch size 1 and identify the
  measured request path. Throughput benchmarks follow their declared workload.
- For offline evaluation, use the selected command's documented defaults and
  explicit device/batch limits. Review the allocation before launching bulk
  work. This policy does not change inference runtime defaults.
- Isolate an offline worker to its assigned GPU UUID and use local `cuda:0`;
  preserve the parent logical index and UUID in receipts.
- Treat inference batch size as a per-device physical window count. Use automatic
  sizing only when explicitly selected, honor memory bounds and record actual
  batch sizes. Never silently switch to CPU or another inference implementation.
- Floating-point differences alone do not establish that a batched prediction is
  invalid. Preserve frame selection, each request's C-conditioned D focus,
  checkpoint identity and metric definitions when comparing results.
- Render saved predictions without rerunning inference. Keep reproducibility
  records in user-selected artifact storage outside the public source tree.
