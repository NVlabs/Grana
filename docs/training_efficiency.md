# Measuring training efficiency

Measure synchronized optimizer-update time and evaluation/media cost
separately. Deployment latency does not establish training throughput,
backward correctness or evaluation cost. Recorded results and their limits
are summarized in [benchmarks](benchmarks.md).

## Gradient checkpointing default and tradeoff

`model.gradient_checkpointing` defaults to `true` for training, smoke tests,
profiling and timing. Keep it enabled when comparing physical batch size and
gradient accumulation. Checkpointing-disabled comparisons require an explicit
user request; they are not an automatic optimization candidate.

The non-reentrant implementation saves activation memory by recomputing
intermediates during backward. The measured
[memory/speed tradeoff](benchmarks.md#gradient-checkpointing) depends on model,
batch, image grid and sequence length. Short checkpointing-off runs do not
establish equivalent convergence. Saved configurations retain their resolved
setting; an explicit YAML value overrides the default for a new run.

## Optional CPU sample prefetch

Set `training.prefetch_samples=true` to prepare the next update's independent
A/B/C examples in one CPU thread while the current update runs. The default
is `false`. The bounded queue holds at most one future local update; allow
host RAM for those RGB images in addition to the current batch.

The producer performs no CUDA work and does not draw from the model's RNG.
Sampling order, pixels and targets retain their synchronous order. D is
constructed on the training thread from its matching C prediction and
observation objects. Coverage counts only consumed examples. Producer errors
propagate to the worker, and the producer is joined on completion or failure.

With profiling enabled, `sample_host_seconds` measures the consumer's wait
and dequeue time; `sample_prepare_host_seconds` records producer work.
Preparation overlaps model work, so it must not be added to update time.
Report filesystem-cache conditions: repeated short runs can accelerate RGB
reads even without prefetch. See the bounded
[prefetch comparison](benchmarks.md#cpu-sample-prefetch).

## Per-update measurements

Set `training.profile_steps=N` to profile the first N real optimizer updates.
The default is zero. This option neither limits the training budget nor
discards warmup samples; `training.max_steps` still controls the update count.
Choose and report a separate warmup policy when summarizing timings.

```bash
bash scripts/train.sh --config /path/to/recipe.yaml \
  --set training.output_dir=/path/to/new-profile-run \
  --set model.gradient_checkpointing=true \
  --set training.profile_steps=10
```

The profiler records CUDA events on the worker's current stream and reads
them after the update's existing synchronization. It adds no per-phase
barriers or model passes. `training_metrics.jsonl` and enabled W&B logging
receive these measurements:

| Metric | Scope |
|---|---|
| `time/profile/sample_host_seconds` | Sampling, RGB materialization and observation hashing, or consumer wait when prefetch is enabled |
| `time/profile/collate_and_transfer_host_seconds` | Serialization, processor work and host-to-device transfer |
| `time/profile/forward_*`, `backward_*`, `objective_and_route_*`, `optimizer_*`, `metrics_reduce_*` | Host durations and CUDA event spans |
| `time/profile/derive_d_host_seconds` | C observation check, focus prompt and D target construction |
| `time/checkpoint_and_logging_seconds` | Save, checkpoint-log submission and distributed checkpoint barrier, outside update time |

`training_compute_seconds` and `time/training_seconds` sum complete
synchronized update wall time: sampling, preprocessing, transfers, forward,
backward, optimizer and DDP. They are not pure GPU kernel time. Host and CUDA
spans overlap and must not be added. CUDA spans include dispatch gaps; use a
profiler trace to isolate kernels or collectives. The lightweight profiler
does not separate processor work from H2D or communication from backward.

## Comparing runs

Keep checkpoint initialization, role counts, frame selection, image/token
workload and effective batch explicit. `training.num_workers` counts Ray GPU
workers, each with a model replica; it does not count CPU loader workers.
Report physical rows per forward separately from accumulation and global
rows per update. Preserve C-to-D dependencies when changing batching.

Measure evaluation and persistence at the same boundaries in both runs.
Metric evaluation, render cohorts, dense prediction, encoding and logging can
delay the next update. Less frequent evaluation changes diagnostic coverage.
Count actual evaluated windows and model calls, and distinguish computation
from media and upload time.

Record model/runtime versions, precision, warmups, repeats, memory measurement
type, CPU/filesystem contention and cache state. Compare examples or tokens
per second alongside validation quality. A short run can reveal a bottleneck
or qualify an execution path; it cannot establish long-run throughput or
convergence.
