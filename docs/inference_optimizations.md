# Inference runtime

## One supported inference runtime

`GranaPredictor`, `grana infer`, `python -m grana.inference.server` and the
server launcher use the same required CUDA configuration. The public predictor
constructor accepts only checkpoint, model path, CUDA device, pixel budget and
view policy. The required acceleration components are not selectable runtime modes.
HTTP requests use only `grana-binary-v1`; JSON is retained for local request
files, binary headers, responses and metadata. The predictor, HTTP client,
CLIs and review browser default to C,D outputs, matching the measured two-forward
workload. Explicit A or B outputs add language forwards; they are requested
prediction tasks, not alternative acceleration modes.

```python
from grana.inference import GranaPredictor

predictor = GranaPredictor(checkpoint="/absolute/path/checkpoint", device="cuda:0")
```

Use the [pinned full CUDA environment](installation.md#full-cuda-environment).
The predictor requires the components below and checks the actual selected
implementations. Missing dependencies or unsupported inputs raise an error
instead of selecting CPU inference, Torch delta rules or uncompiled execution.
The checkpoint's training backend does not select a different inference mode.

| Required component | Behavior |
|---|---|
| CUDA resize, rescale, normalization and patchification | Original uint8 frames move once through pinned memory; the model's image processor runs on CUDA |
| SDPA and batched equal-length vision attention | Image attention is batched across independent equal-length image sequences |
| FLA delta rules and causal-conv1d | Both CUDA kernel packages must be present and active |
| FLA output guard | The process-wide autotuner excludes the unsafe eight-warp output-kernel configuration |
| LoRA merge | Merge into BF16 base weights in memory once at load, with FP32 accumulation and TF32 off |
| Host multimodal position IDs | Exact host calculation, checked against HF initially; unsupported inputs or failed verification raise errors |
| Compiled decoder and vision tower | `torch.compile` with static shapes and no CUDA graphs; compilation errors or the recompile limit fail explicitly |
| Image reuse | Preprocessing, transfers and image features are shared only between roles of the current request |
| Text-input cache | A predictor-lifetime LRU reuses exact processor text outputs for matching text, grid and options |
| CPU thread budget | Eight Torch CPU threads, set by the predictor and reported in runtime metadata |
| HTTP execution | A single persistent inference worker preserves warmed thread-local CUDA handles and cuDNN plans |

The model patches apply to the predictor's in-memory instance. The FLA output
guard and Torch CPU thread budget affect the process. Checkpoint files and
training configurations are unchanged; shared training/evaluation code retains
its separate CPU and reference implementations.

GPU resize is also used for older CPU-trained checkpoints.
`gpu_image_resize`, `checkpoint_gpu_image_resize` and `resize_matches_checkpoint`
report whether the runtime resize matches training. Use
`data.gpu_image_resize: true` in training to align the image pipeline. CUDA
interpolation can round differently from host uint8 resize and flip near-tied
stage predictions; LoRA merging and compilation also change BF16 rounding.
Qualification of accuracy remains separate from latency measurement.

### Cold starts and comparable latency

Compilation is lazy. The first prediction and new prompt lengths or image grids
may compile new graphs; warm representative requests before measuring latency.
The expected graph breaks around FLA and causal-conv1d are part of this compiled
path. The strict runtime does not suppress compiler errors or silently switch
the whole decoder to eager after reaching its recompile limit. It rejects
`TORCH_COMPILE_DISABLE=1`, `TORCHDYNAMO_DISABLE=1` and a disabled Dynamo
configuration rather than reporting compilation while running eager.

Measure the shared evaluation CTSP(16,4) window with the
[HTTP benchmark](inference.md#measure-http-latency). Comparable latency requires comparable GPU capacity, pinned libraries,
checkpoint, input geometry/prompts, readout slot, warmup, sequential load and
network. Compare only measurements with matching timing boundaries and workloads;
see [benchmark methodology](benchmarks.md).
HTTP preserves the worker across client connections. For local Python inference, load once and call the
predictor repeatedly from the same thread; creating a fresh calling thread
can discard thread-local CUDA/cuDNN caches.

C and D retain independent prompts, position IDs and language forwards; only
image work is reused within one request. No prefix, KV or recurrent state is
shared between them.

### Runtime evidence

Server health and every prediction's `checkpoint.inference_runtime` report the
required configuration, including `policy: "cuda_optimized_v1"`,
`fallback_policy: "error"`, `cpu_threads: 8`, `batched_vision_sdpa`,
`reuse_images`, `merge_lora`, the `lora_merge` report, `cpu_position_ids`,
`compile_decoder`, `compile_vision`, compile settings, `causal_conv1d`,
`fla_output_guard`, `text_input_cache`, `gpu_image_preprocessing`, resize policy
and actual attention/delta-rule implementations. `runtime_counters()` reports
fast-path execution counters; `text_cache_stats()` reports entries, hits and
misses.

## Cache and numerical behavior

Images are reused only within a request. C and D share preprocessing, transfers,
and vision features while retaining independent language forwards and prompts.
There is no cross-request image, prefix, KV, or recurrent-state cache.

The text-input LRU caches processor text outputs for exact text, image-grid and
option matches. Changing prompts, camera names, grids or options creates a new
entry. Runtime counters and cache statistics describe actual reuse; they are
not substitutes for end-to-end timing.

The inference LoRA merge uses FP32 accumulation before casting to BF16. It
changes only the in-memory model. CUDA interpolation, merged weights and
compiled execution can differ numerically from training or reference paths.
Evaluate accuracy with the same runtime and input policy used for deployment.

## Measuring performance

Use the [HTTP latency command](inference.md#measure-http-latency) for sequential
batch-1 requests and the [batch throughput command](inference.md#measure-batch-throughput-per-selected-poi)
for local offline work. Report cold-start costs separately from warm samples.
Preserve checkpoint identity, roles, selected frame, image geometry, software,
hardware and timer boundaries with every result.

The [benchmark guide](benchmarks.md) explains timing boundaries and capacity
limits. [Reference results](results.md#execution-validation) distinguish
installation, functional execution and training validation.
