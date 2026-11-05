# Benchmark methodology and results

These are summaries of completed measurements, not fresh benchmarks of every
source revision. Each section states its workload and timing boundary.
Model-quality results are in [reference results](results.md); instructions for
new measurements are in [inference](inference.md) and
[training efficiency](training_efficiency.md). Raw execution receipts,
recordings and model payloads are not distributed with the source release.

## HTTP latency

Both measurements below used one GPU, physical batch size 1, concurrency 1,
binary RGB over loopback HTTP, four 320×240 cameras and eight frames per
camera. CTSP(16,4) specifies the context/tail strides. C-to-D requests run C
first, then condition D on C's stage at the selected slot. GPU image
preprocessing, merged LoRA, batched vision SDPA and compiled vision/decoder
were enabled. The recorded runtimes used Torch 2.10.0+cu130, Transformers
5.6.0.dev0, Triton 3.6.0, FLA 0.5.2 and causal-conv1d 1.7.0; H100 also used
TileLang 0.1.15.

Each case used a fixed input, five warmups and 100 measured sequential
requests to a persistent server. Warmups, loading and initial compilation
are excluded. Percentiles use linear interpolation. Server execution is
already included in client latency and must not be added again.

### H100: prepared HTTP request to full response

The checkpoint was HotWheels D100 Qwen3.5-4B step 1000 on an H100 80 GB.
Timing starts before `urlopen` with a prepared request and ends after
`response.read`, including connection setup, upload, server execution and
download. Binary packing, Request construction and response JSON parsing
are excluded.

| Roles | Selected frame | Mean, ms | P50, ms | P95, ms |
|---|---|---:|---:|---:|
| A | POI, slot 5 | 94.04 | 93.63 | 98.00 |
| A | LAST, slot 7 | 94.27 | 93.92 | 98.11 |
| C | POI, slot 5 | 93.77 | 93.58 | 96.06 |
| C | LAST, slot 7 | 93.38 | 93.04 | 95.21 |
| C,D | POI, slot 5 | 155.51 | 155.25 | 158.21 |
| C,D | LAST, slot 7 | 155.95 | 155.79 | 158.46 |

All 630 requests, including warmups, returned HTTP 200. These timings use
an existing package environment and compiler caches. A separate run of the
same checkpoint/window measured complete-client C,D/LAST mean 154.6 ms,
P50 154.5 ms and P95 158.7 ms, including packing and JSON parsing. These are
separate sample sets; subtracting their summaries does not estimate overhead.

### GB300: first socket send to full response

The checkpoint was Qwen3.5-4B Pretrain199 step 400, evaluated on a HotWheels
window at POI slot 5. Timing starts at the first `socket.sendall` entry after
connection setup and ends after reading all response bytes, before JSON
parsing. This includes request headers/body transmission and server work.

| Interval | Mean, ms | P50, ms | P95, ms |
|---|---:|---:|---:|
| First socket send to full response | 123.320 | 121.924 | 139.346 |
| Complete client call, including packing and parsing | 124.718 | 123.333 | 140.742 |

All 105 requests returned HTTP 200. H100 and GB300 used different checkpoints
and timing boundaries; these numbers do not establish a controlled hardware
speedup. Neither measurement covers remote-network latency, startup or all
possible prompt/window shapes.

## Offline batch capacity and throughput

On eight H100 80 GB GPUs, the HotWheels D100 4B step-1000 checkpoint completed
three full batches per GPU at both physical batch 32 and 64, using four
cameras, eight slots and C followed by C-conditioned D. Batch size counts
independent windows per GPU, not frames or camera images.

| Batch per GPU | Aggregate capacity, 8 GPUs | Completed windows | Peak allocated per GPU, GiB | Sampled device-used memory, GiB |
|---|---:|---:|---:|---:|
| 32 | 256 | 768 | 26.9–28.0 | 32.0–33.7 |
| 64 | 512 | 1,536 | 45.2–45.5 | 54.5–57.9 |

Both runs completed without OOM, retry, missing records or duplicates.
Allocated memory comes from PyTorch's peak allocator measurement. Device-used
memory includes cache and other allocations and was sampled about once per
second, so it can miss brief peaks. Different sample counts and compilation
costs prevent comparing total campaign times as a throughput experiment.
This capacity result qualifies the stated checkpoint/workload, not every GPU
or model. Explicit resource limits still apply.

A separate controlled throughput comparison used one H100, the same checkpoint
and 64 distinct CTSP(16,4) POI windows covering all 24 evaluation trajectories.
It alternated serial batch-1 calls and one batch-64 call over five measured
rounds, after two full warmup rounds per variant.

| Variant | Mean total for 64 windows, s | Amortized ms per selected POI | Windows/s | Peak allocated, GiB |
|---|---:|---:|---:|---:|
| 64 serial batch-1 calls | 9.718 | 151.846 | 6.586 | 10.740 |
| One batch-64 call | 9.212 | 143.938 | 6.947 | 45.476 |

Timing surrounds complete synchronized predictor calls, including GPU
preprocessing, both roles and CPU output processing. Input loading, hashing,
validation and HTTP are excluded. Batch 64 achieved 1.055× throughput and
5.2% less amortized time, but results for the entire batch took 9.212 seconds.
The denominator is 64 selected POIs, not 512 returned slots or 2,048 images.
These are throughput measurements, not HTTP request latencies.

## Gradient checkpointing

Bounded GB300 runs used Tractor train100, four views × eight time slots,
BF16, SDPA, FLA auto, one worker and aggregate A2/B2/C1/D1 per update. Each
update has two forward/backward phases of physical width 3. Each condition
started from the same model-size pretrain checkpoint and ran ten updates;
P50 uses updates 4–10. Evaluation and remote logging were disabled.

| Model | Checkpointing on, s/update | Off, s/update | On/off time ratio | Peak allocated GiB, on → off |
|---|---:|---:|---:|---:|
| 0.8B | 1.676 | 1.250 | 1.34× | 4.17 → 27.36 |
| 4B | 2.717 | 2.250 | 1.21× | 14.70 → 81.95 |

Update time includes sampling, preprocessing, transfers, both forwards and
backwards, and optimizer work. Initialization, saving, evaluation and final
restore checks are outside the timer. Peak allocated memory covers all ten
updates and excludes CUDA context/workspaces outside the PyTorch allocator.
Other GPUs had active work and shared the host and filesystem.

The 0.8B table uses a repeated warm baseline. Its first baseline had median
RGB materialization 0.60 s rather than about 0.04 s after caching and is
excluded from the ratio. First-update losses matched between checkpointing
variants, but gradients and later trajectories differed. Finite gradients,
save/reload and within-run prediction parity passed; convergence equivalence
was not established. Checkpointing remains enabled by default.

With checkpointing on, setting `TORCH_CUDNN_SDPA_DEPRIORITIZED=1` changed P50
from 1.676 to 1.649 s for 0.8B and 2.717 to 2.681 s for 4B. A single short
comparison without kernel tracing does not establish a useful training gain.
These runs do not qualify 2B/9B/27B steady-state training performance.

## CPU sample prefetch

Four eight-update runs compared synchronous and prefetched sampling on two
GB300 GPUs, using raw Qwen3.5-4B initialization, physical batch 32 per worker,
accumulation 2, global batch 128 (A80/B32/C16/D0), 200 tasks and 5,006 training
trajectories. Checkpointing, GPU preprocessing and batched frozen-vision SDPA
were enabled. Values below average updates 3–8.

| Condition, in execution order | Data seed | Seconds/update |
|---|---:|---:|
| Synchronous, first pass | 97531 | 28.000 |
| Prefetch, same draws | 97531 | 21.809 |
| Prefetch, second seed | 97532 | 21.498 |
| Synchronous, warm repeat of preceding draws | 97532 | 23.778 |

The reverse-order warm comparison measured 9.6% less time per update or
10.6% more updates per second. No OS caches were flushed, so the larger
first-pass difference cannot be attributed entirely to prefetch. Each
same-seed pair had identical update losses, gradient norms, learning rates,
coverage and final checkpoint tensor hashes, with about 63.8 GiB peak
allocated GPU memory. The second-seed pair also had equal predictions for
100 evaluation windows across 50 tasks at updates 4 and 8, using both ranks.
This did not repeat the full 1,374-trajectory evaluation or qualify its speed.
Short runs on a shared host do not establish long-run speed or convergence.

## GPU image preprocessing

A GB300 preprocessing-only measurement used in-memory RGB images through
collation and completed transfers, no model forward, maximum 81,920 pixels,
five warmups and 30 measurements.

| Images | Native CPU P50, ms | CPU resize / GPU remainder P50, ms | Full GPU P50, ms |
|---|---:|---:|---:|
| 32 | 31.06 | 23.12 | 17.72 |
| 64 | 50.67 | 31.52 | 21.68 |

Training/inference inputs matched bitwise on the tested GPU-preprocessed
fixture and reversed batch. CPU/GPU resize differed by at most one uint8
level for the tested 240×320 → 256×320 fixture; this is not a global bound.

A separate full CPU-versus-GPU preprocessing comparison used two GB300
workers, raw Qwen3.5-4B initialization, checkpointing, physical batch 32,
accumulation 2 and global A0/B32/C48/D48. Each process ran six updates.
A warm-repeat GPU/CPU/CPU/GPU comparison, fixed before the last two runs,
measured update medians of **21.030 s CPU / 20.568 s GPU** (2.2% less time),
and evaluation medians of **12.957 s / 12.138 s** (6.3% less time). Updates
3–6 contributed eight samples per arm; evaluations at updates 4 and 6
contributed four per arm, each covering 64 B/C/D windows with both ranks.
Media and remote logging were disabled.

This changes resize, normalization and patchification together. Separate
CPU- and GPU-preprocessed training runs produced different predictions:
282/1,536 C slots and 40/192 D focus rows changed, with maximum B/D progress difference
0.0418178. Model-specific D focus can also change prompt length, so token
workload was not guaranteed identical. Initial-run drift and shared-host
activity limit the timing claim. These checks do not establish dataset-wide
accuracy or long-run throughput equivalence.

## Evaluation cost

A distinct six-run GB300 timing campaign used 400-update Pretrain199 and
200-update posttrain workloads, checkpointing, physical forward width 3 and
six role rows per worker update. Pretrain used two workers and posttrain
used one. The evaluation cohort used PNG/GIF rendering and predates dense
MP4 evaluation; it does not estimate dense-recipe cost.

| Task | Model | Updates, s | Evaluation and persistence, s | Engine elapsed, s | Evaluation share |
|---|---|---:|---:|---:|---:|
| Pretrain199 | 0.8B | 897.9 | 341.7 | 1253.5 | 27.3% |
| Pretrain199 | 4B | 1109.4 | 429.9 | 1558.8 | 27.6% |
| Spatula | 0.8B | 388.0 | 460.7 | 856.8 | 53.8% |
| Spatula | 4B | 526.8 | 583.4 | 1120.7 | 52.1% |
| Tractor | 0.8B | 453.5 | 755.4 | 1217.2 | 62.1% |
| Tractor | 4B | 553.7 | 945.3 | 1509.9 | 62.6% |

Engine elapsed excludes initial model/data setup, Ray startup and logger
shutdown. Remaining engine time includes checkpointing, reporting and final
verification. Runs overlapped; their elapsed times must not be summed as
campaign wall time. Evaluation and media can dominate this workload, so
report them alongside update throughput.

## Numerical validation limits

Fixed-input repeatability is narrower than accuracy or training correctness.
A GB300 FLA 0.5.2 output-guard comparison covered 0.8B/2B/4B/9B/27B,
two fresh processes per condition, five warmups and 30 measured C-to-D
requests per process. Guarded forwards were repeatable for all sizes;
unguarded 27B logits and progress probabilities were not. Guarded 27B C+D
forward P50 was 287.508 ms versus 288.872 ms unguarded, averaging per-run
P50s. That timer covers tensor-ready forward CUDA events, including dispatch
gaps, but excludes preprocessing, HTTP and response conversion.

The 2B/9B/27B checkpoints in that study were support-smoke examples. Stable
forwards do not qualify backward/gradient behavior or a replacement training
backend. Batched BF16 outputs and CPU/GPU preprocessing can differ even when
execution is valid; report task-level metrics separately from repeatability.
