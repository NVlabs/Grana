# Inference

Run commands from the source root after [installation](installation.md).
Use the CLI for one request, the Python API for repeated local calls, or the HTTP
server for client requests. All use the same checkpoint and prediction contract.

## Inference from a trained checkpoint

Use a completed pretrain or posttrain checkpoint for the intended model size.
Replace the example checkpoint paths and update numbers below with an actually
saved checkpoint; inference does not require a particular training step.
Inference loads the base model, all adapters, both complete heads, and marker embeddings;
it does not open training caches, start Ray, or initialize W&B. Supply a local
base-model snapshot with `--model-name-or-path` if the checkpoint's original
base-model location is unavailable.

Inference has one supported execution path. It requires CUDA, GPU image
resize/rescale/normalization/patchification, SDPA attention, FLA delta rules,
causal-conv1d, the FLA output guard, LoRA merging, exact host-side position IDs,
compiled decoder and vision modules, and image/text reuse. It fixes Torch's CPU
thread budget at eight. The CLI and Python constructor expose checkpoint,
model path, CUDA device, pixel budget and view policy; they do not expose
backend or per-feature mode switches. C,D are the default outputs for the
Python predictor/client, CLIs and review browser. Explicitly requesting A or B
adds language-model work and changes the latency workload. Missing dependencies,
unsupported model
components or inputs, and compilation failures raise errors. Install the
[full CUDA environment](installation.md#full-cuda-environment) before loading a
predictor; inference does not fall back to CPU or a native Torch delta rule.

Train with `data.gpu_image_resize: true` to use the same image processor during
training and evaluation. Inference always uses GPU resize, including for older
CPU-trained checkpoints. Checkpoints record their original preprocessing
policy, and runtime metadata reports whether inference resizing matches it.
CPU and GPU interpolation can round differently and flip near-tied stage
predictions. LoRA merging and compilation also change BF16 rounding; this
runtime contract does not establish prediction accuracy.

The HTTP client and server support only lossless `grana-binary-v1` requests.
JSON remains the format for request files, binary headers, responses and audit
metadata. There is no JSON/base64 request route or automatic transport fallback.
The [runtime guide](inference_optimizations.md) describes required components,
cache behavior and runtime metadata. Warm up representative requests before
measuring latency; the first prediction and new input shapes may compile graphs.
See [benchmark methodology](benchmarks.md#http-latency) for comparable timings.

By default, create `request.json` with the checkpoint's camera names **in the saved order**.
This example uses `top`, `left`, `right`, `bottom`. These paths refer to
existing RGB uint8 arrays shaped `[T,H,W,3]`; paths may be absolute or relative
to the JSON file. Select exactly eight observations explicitly:

```json
{
  "views": {
    "top": "/data/episode/top.npy",
    "left": "/data/episode/left.npy",
    "right": "/data/episode/right.npy",
    "bottom": "/data/episode/bottom.npy"
  },
  "frame_indices": [72, 88, 104, 112, 116, 120, 124, 128],
  "prediction_frame": "poi",
  "task": "synthetic-token-and-lid",
  "compressed_task_prompt": "Place the blue token in the tray, then close the lid.",
  "stage_prompts": ["Place the blue token inside the tray", "Close the tray lid with the token inside"],
  "focus_stage": 1
}
```

Use the task descriptions and ordered stage definitions appropriate to your
model. The example text illustrates the schema; it does not replace your
training prompts. Eight-frame arrays can omit `frame_indices`. Each view may
instead contain a list of eight RGB image-file paths, with no `frame_indices`.
Repeated frames and explicit `frame_indices` remain in the supplied order.
The model's processor performs resizing. For full `.npy` video arrays, omit
`frame_indices` and pass `--frame-index N` to compose a window using the shared
evaluation CTSP(16,4) geometry described below. Explicit `frame_indices` and
`--frame-index` cannot be combined.

Both local and HTTP client CLIs accept `--prediction-frame poi|last`:

| Selection | Input slot, zero-based | Meaning |
|---|---:|---|
| `poi` | 5 | Third observation from the end; the default CTSP point of interest |
| `last` | 7 | Final supplied observation; use for a current-frame readout |

The CLI default is **`poi`**, unless the input JSON explicitly selects
`prediction_frame` or `poi_slot`. An explicit CLI selection overrides the file.
`--poi-slot 0..7` remains available for another occurrence; do not combine it
with `--prediction-frame`. If JSON contains both fields, they must agree.
Repeated physical frames still occupy separate input slots. Direct Python
`InferenceRequest(...)` calls use the same default `poi_slot=5`;
set `poi_slot=7` for a last-frame readout.

```bash
source .venv-qwen35-unsloth/bin/activate
grana infer \
  --checkpoint /runs/my-task/checkpoint-000200 \
  --input request.json --roles C D \
  --output prediction.json --plot prediction.png
```

Choose only the outputs you need; the default is `--roles C D`:

| CLI usage | Selection | Language-model forwards |
|---|---|---:|
| A only | `--roles A` | 1 |
| C only | `--roles C` | 1 |
| C followed by D | `--roles C D` | 2 |

Use `--prediction-frame last` in any row for a slot-7 readout.
The same options work with `python -m grana.inference.client`. `--plot PATH.png`
renders all eight returned values, with the role, selected slot and known
source-frame index identified. It does not make another model call.

| Role | Required text | Prediction |
|---|---|---|
| A | `compressed_task_prompt` | Whole-task progress in [0,1] |
| B | `task`, `stage_prompts`, explicit zero-based `focus_stage` | Progress relative to the requested focus |
| C | `task`, `stage_prompts` | Zero-based stage IDs, valid-class logits and probabilities |
| D | `task`, `stage_prompts` | C's predicted stage, followed by progress relative to that focus |

No role needs ground-truth labels. B's focus is supplied by the caller; D always
uses C's hard prediction at the **selected input slot** on the same RGB
observations. For `last`, D must be run with C's slot-7 focus. Reading
`D.values[7]` from an old slot-5 request does not perform this change: all eight
D values in that response were conditioned on C's slot-5 focus.

Progress is the expectation of the ten-bin distribution. JSON retains all
eight predictions, logits/probabilities and the strict checkpoint load report.
`poi_slot` identifies the selected slot for compatibility;
`prediction_frame` is `poi`, `last` or `custom`. `selected_prediction` gives
the selected `slot`, per-role `progress`, per-role `focus_stage`, and `stage_id`
when C ran; `frame_index` is present when source indices are known.
D includes its prerequisite C output. B/D progress is
focus-relative, not a normalized whole-task score. Results are unsmoothed.
Output files must be new, so earlier inference results remain intact.

An accepted role does not establish its accuracy: evaluate each requested role
on held-out data, especially if that role had no supervision during training.

### Eight-frame composition and online alignment

Both training phases default to 60% `regular_ctsp` / 40% `irregular_stratified`
selection and, independently, 50% `forward` / 20% `backward` / 30% `tail_rewind`
query order. Training chooses uniformly among `(context_stride, tail_stride)`
pairs `(10,2)`, `(16,4)`, `(24,6)` and `(32,8)`. Evaluation uses `(16,4)` with
`regular_ctsp` selection and `forward` order independently of the training mixture.
The data manifests select caches and roles; these stride/profile settings come
from the phase configuration. See [the selector](../grana/data/frame_sampling.py)
and [posttraining defaults](../configs/posttraining/default.yaml).

For an interior POI `p`, context stride `c` and tail stride `t`, the eight slots
are `[p-3c, p-2c, p-c, p-2t, p-t, p, p+t, p+2t]`. If `p < 3c`, the first three
become `[0, p//3, 2*p//3]`; local indices are clipped to the available frame
range. Duplicates stay supervised. `forward` preserves this selector order,
which can be nonchronological near the start: at `p=9`, `(c,t)=(16,4)` gives
`[0,3,6,1,5,9,13,17]`. Do not sort it.

All eight occurrences receive training loss, with weights
`[0.1,0.3,0.5,1.0,1.5,2.0,2.0,2.0]`; slot 7 is supervised with the same weight
as slot 5. The progress temporal head attends across all eight observations.
Consequently, a slot-5 progress readout from a normal CTSP window can use its
two later observations. At a 10 FPS cache rate, `(16,4)` needs up to eight
frames, or 0.8 seconds, of lookahead relative to the POI.

For a live readout, all supplied observations must be available at or before
the current frame, and the selected last slot must represent that frame.
`--frame-index 128 --prediction-frame last` composes
`[72,88,104,112,116,120,124,128]`: it anchors CTSP at 120 and reads slot 7 at
128. `--frame-index 120 --prediction-frame poi` composes the same interior
window but reads slot 5 at 120. The former uses no observations later than
128; the latter requires frames through 128. Early last-frame windows anchor
at `max(0, N-8)` and clip to the frames available through `N`.

For example, with camera `.npy` paths and task text in `video_request.json`
and no explicit `frame_indices`:

```bash
grana infer --checkpoint /runs/my-task/checkpoint-000200 \
  --input video_request.json --frame-index 128 \
  --roles C D --prediction-frame last --output latest.json --plot latest.png
```

Selecting `last` on an arbitrary supplied window does not inspect its capture
times or make it chronological. It selects the final supplied occurrence.
The server continues to return all eight outputs for analysis; earlier-slot
progress outputs from an online request can still use later observations
within that request. Training/evaluation and archived comparison plots retain
their designated POI semantics.

The HTTP benchmark and replay scripts use this shared CTSP(16,4) composer.
The helper uses source-array frame indices and does not resample the frame rate.

For the same target source frame 2696, the two readouts use:

| Readout | CTSP anchor | Eight source-frame indices | Selected slot |
|---|---:|---|---:|
| `poi` | 2696 | `[2648,2664,2680,2688,2692,2696,2700,2704]` | 5 |
| `last` | 2688 | `[2640,2656,2672,2680,2684,2688,2692,2696]` | 7 |

Both follow `(16,4)` geometry. They align the requested physical target to
different input slots, so their observations and D focus can differ.

### Measure HTTP latency

Start the server below, then point the benchmark at a request containing full
camera `.npy` video paths and task text, without explicit `frame_indices`:

```bash
python scripts/inference/benchmark_http.py \
  --server http://127.0.0.1:8787 --input fullvideo_request.json \
  --frame-index 2696 --roles C D \
  --warmup 5 --repeats 100 --output result.json
```

The primary `client_send_to_receive_ms` interval starts at the first request
`socket.sendall` entry, after packing, request construction and connection
setup, and stops when `response.read` has received the full body, before JSON
parsing. It covers
the full request transmission, HTTP transfer, server inference and response
serialization. `client_send_ms` separately marks the final `sendall` return;
`client_send_complete_to_receive_ms` measures from there through response receipt.
`http_round_trip_ms` additionally includes connection setup, while
`client_total_ms` includes packing, connection setup and parsing. Frame loading,
window construction, server startup and warmup are outside the steady-state summary.
The client exposes these measurements through an optional `timing={}` argument;
the prediction envelope is unchanged.
Change `--roles` to `A`
or `C`, and `--prediction-frame` to `last`, to measure each of the six requested
role/readout combinations. Keep the checkpoint and target frame fixed for
the comparison.

Trajectory replay videos score 32 sampled target frames with changing windows.
Their per-request times can include new-shape compilation and other cold
work. Video playback at a chosen samples-per-second rate is a presentation
setting; it is not inference throughput. Use the repeated fixed-target
benchmark for steady-state latency.

### Six role and readout replays

With the same server and full-video request, record and render all six cases:

```bash
python scripts/inference/replay_http.py \
  --server http://127.0.0.1:8787 --input video_request.json \
  --metadata display_metadata.json --output new_predictions_dir --samples 32 \
  --name-prefix my_task \
  --model-label 'My task | Qwen3.5-4B | step 200'

python scripts/inference/render_http_replay.py \
  --manifest new_predictions_dir/manifest.json --output new_videos_dir
```

Both output directories must be new. `display_metadata.json` contains
`trajectory_id`, `source_fps`, `num_frames`, and the arrays `whole_progress`,
`stage_ids`, `within_stage_progress`, with one label per source frame. These
labels are used only for display and comparisons; they are never sent to the
server. The request file contains camera arrays, task text, ordered stage
descriptions and `compressed_task_prompt`, with no explicit `frame_indices`.

The six HTTP replay combinations are A/POI, A/LAST, C/POI, C/LAST, C+D/POI
and C+D/LAST. They use the same sampled target source frames and the shared
CTSP(16,4) composer. POI and LAST align each target as shown above; they do
not reuse a prediction made with another readout's window or D focus.

Each video identifies its roles, selected input slot and current source frame.
The server's full eight-slot outputs remain in the saved prediction records.
A shows whole-task progress, C shows the stage, and C+D shows the stage plus
progress relative to its predicted focus. The 32 sampled positions show changing model predictions over a trajectory;
their playback rate and request timings are separate from the fixed-target
latency benchmark. The renderer defaults to 2 playback FPS, producing six
16-second H.264 MP4s plus posters, an HTML index and a verification receipt.
It reads saved HTTP predictions and does not run the model again.

### Compare POI and LAST across an evaluation cohort

Set `GRANA_EVAL_MANIFEST` to your held-out cache manifest, `GRANA_DATA_ROOT`
to its data root, and `CHECKPOINT`/`WEIGHTS_SHA256` to the checkpoint path and
its weights digest. Relative cache paths resolve under `--data-root` (or the
manifest directory when that option is omitted); environment variables must be
set before preparation. Use that manifest to prepare paired C,D requests for multiple
sampling seeds. Each seed draws one target from each of up to 128 disjoint
temporal bins per trajectory; shorter trajectories use every frame. The two
modes predict the same physical targets using their own CTSP(16,4) windows.
Eight separate fixed targets per trajectory check repeatability across runs.
These anchors are excluded from the quality metrics.

```bash
python scripts/inference/evaluate_http_frames.py prepare \
  --config "$GRANA_EVAL_MANIFEST" \
  --data-root "$GRANA_DATA_ROOT" \
  --checkpoint "$CHECKPOINT" --weights-sha256 "$WEIGHTS_SHA256" \
  --output /tmp/grana-frame-comparison --seeds 11 29 47 \
  --samples 128 --anchors 8

# Bulk evaluation defaults to all visible GPUs, with batch 64 per GPU.
python scripts/inference/evaluate_frames.py \
  --plan /tmp/grana-frame-comparison/plan.json \
  --output /tmp/grana-frame-comparison-local
python scripts/inference/summarize_http_frames.py \
  --experiment /tmp/grana-frame-comparison-local \
  --output /tmp/grana-frame-comparison-summary
```

The offline runner balances complete trajectories across all visible CUDA
devices and defaults to **64 windows per GPU**, verified on eight H100 80 GB
cards. This fixed default skips sizing probes. Short tails use the remaining
window count; a full batch that cannot fit reports OOM.

Optional `--batch-size auto` enables capacity-based sizing. It starts at 32
windows per GPU, respects the remaining workload and explicit user limits,
and grows until constrained by memory or those bounds. The default auto memory
budget retains 10% of currently free memory. Auto chooses the largest measured
fitting batch; prediction differences and short throughput fluctuations do not
limit selection. CUDA OOM or a measured memory-budget excess reduces an auto
batch and retries without losing targets. Other model errors fail the run.

Explicit controls override these defaults:

```bash
python scripts/inference/evaluate_frames.py \
  --plan /tmp/grana-frame-comparison/plan.json \
  --output /tmp/grana-limited-evaluation \
  --devices 0,1 --batch-size 32
```

`--batch-size` is **per GPU**, not the aggregate across the selected cards.
With full default batches, 64 per GPU gives aggregate capacity 128 on two
cards, 384 on six cards, or 512 on eight cards. These offline workers run independently;
the capacity sum does not require synchronized batches.

`--devices` uses parent-process logical indices within `CUDA_VISIBLE_DEVICES`.
The launcher resolves each selected GPU to its UUID and exposes only that GPU
to its worker. Every worker therefore uses local `cuda:0`; the plan and receipts
retain both the parent index and GPU UUID, and workers verify the mapping.
`--memory-fraction` and optional `--max-batch-size` limit automatic sizing.
An explicit physical batch skips tuning and is never silently reduced on OOM.
Sizing warmups can compile new shapes; those warmups are excluded from candidate
throughput measurements. Local batch timings are labeled separately from
HTTP request latency.
The merged output is published only after every planned prediction succeeds.

For a transport-specific evaluation, explicitly use the HTTP collector.
Independent server processes also allow fixed anchors to check process
repeatability; multiple rounds on one persistent local worker only measure
within-process repeatability:

```bash
python scripts/inference/evaluate_http_frames.py run \
  --plan /tmp/grana-frame-comparison/plan.json --seed 11 \
  --server http://127.0.0.1:8787
# Repeat for seeds 29 and 47, then summarize the completed HTTP runs:
python scripts/inference/summarize_http_frames.py \
  --experiment /tmp/grana-frame-comparison \
  --output /tmp/grana-http-frame-comparison-summary
```

The primary prediction is the zero-based C stage plus D's progress relative
to that predicted stage. The target is GT stage plus GT within-stage progress;
normalized errors divide by the trajectory's number of stages. Each
trajectory's squared errors are combined across the sampling seeds before
taking its RMSE. **Macro RMSE is the arithmetic mean of those trajectory
RMSEs**, giving short and long trajectories equal weight. The report also
retains individual trajectories, each sampling run, cache groups, and pooled
metrics. Fixed anchors measure execution repeatability, not extra quality
samples. A macro advantage does not establish an advantage on every trajectory.

The summary also reports **macro MSE** (equal weight per trajectory) and
**micro MSE/RMSE** (pool all sampled quality observations). The square root
of macro MSE is not the mean of trajectory RMSEs. Repeated targets count once
per sampling round in the primary micro metric; a separately labeled
deduplicated result counts each physical target once. Sampling a similar
number of targets per trajectory does not produce full-video-duration weights.

See [POI and LAST evaluation](results.md#poi-and-last-evaluation) for the scope
of the reference comparison. A readout choice changes both observations and D's
conditioning; assess it on your own deployment data.

### Measure batch throughput per selected POI

Use the same real windows for one physical batch and serial singleton calls:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/inference/benchmark_batch.py \
  --plan /tmp/grana-frame-comparison/plan.json \
  --seed 11 --batch-size 64 --roles C D --prediction-frame poi \
  --warmup 2 --repeats 5 \
  --output /tmp/grana-bs64-throughput.json
```

This benchmark requires one visible GPU. It selects distinct quality targets
across the prepared trajectories, loads them once, warms both variants, and
alternates execution order across measured rounds. `--prediction-frame last`
selects slot 7 and conditions D on that slot's predicted C stage; `poi` selects
slot 5. Both use the shared evaluation composition. The report records exact
windows, source and checkpoint hashes, warmups, individual rounds and memory.

**Amortized milliseconds per selected POI = whole-workload milliseconds / 64**
for this command. Each window contributes one selected POI despite returning
eight temporal slots, regardless of camera count. Windows per second is the
reciprocal throughput.
The complete batch latency remains the wait for all 64 results. These local
public-predictor timings include GPU resize and CPU postprocessing, with CUDA
synchronization; they exclude loading and HTTP transport. For request-to-response
HTTP latency, use `benchmark_http.py` above.

See [batch capacity and throughput](benchmarks.md#offline-batch-capacity-and-throughput)
for reporting requirements and the distinction between throughput and latency.

### Repeated local prediction

For repeated local calls, load the model once and call it from the same thread
so CUDA handles and cuDNN execution plans remain warm:

```python
from grana.inference import GranaPredictor, InferenceRequest

predictor = GranaPredictor("/runs/my-task/checkpoint-000200", device="cuda:0")
request = InferenceRequest(
    images={"top": top_frames, "left": left_frames,
            "right": right_frames, "bottom": bottom_frames},
    task=task_description,
    stage_prompts=ordered_stage_descriptions,
    prediction_frame="poi",  # optional: POI slot 5 is the default
)
result = predictor.predict(request)  # C,D by default
print(result["selected_prediction"])
```

Each `*_frames` is eight RGB PIL images or a uint8 array `[8,H,W,3]`.
Keep original camera pixels; resizing is performed by the checkpoint's processor.
The compiled vision path requires equal resized image token counts across the
request; incompatible image geometry is rejected.
`--max-pixels` is an explicit override and can change predictions.
Use the HTTP service for clients arriving on different threads; it serializes
model calls on one persistent worker. This is an explicit-window scorer; the
caller supplies the frame history and chooses its readout. The HTTP transport
below preserves this same eight-observation contract.

To use one checkpoint with different camera sets, select `--view-policy request`
on `grana infer`, or construct `GranaPredictor(..., view_policy="request")`.
Each call then uses the nonempty ordered camera mapping in its request: for
example, one exterior camera, exterior plus wrist, or four cameras. Every
camera still supplies exactly eight observations. Prompts and image groups
use the actual camera names and count; C and D share the exact same images.
The response records `view_names`, `checkpoint_view_names`, and `view_policy`.
The default `checkpoint` policy continues to reject camera mismatches. Accepting
a new camera set establishes input compatibility; reward accuracy on that set
needs separate evaluation.

Image sizing is explicit. Training defaults to `data.max_pixels: null`, which
uses the processor's default resize settings and preserves the existing
recipes' actual image geometry. Set an integer, such as `76800`, to impose a
per-image pixel budget. The collator supplies both resize bounds and checks
the resulting image grids before model execution. If patch rounding exceeds
the requested budget, it raises an error instead of silently exceeding it.
Original RGB arrays remain unchanged.

Inference inherits the checkpoint's pixel-budget contract unless an integer
`--max-pixels` override is supplied; resizing itself always runs on CUDA. New checkpoints store a versioned
`image_preprocessing` descriptor. Old checkpoints without that field retain
their historical pixel-budget interpretation: their old large numeric setting
used the processor defaults, while old small budgets used their original
geometry. The interpolation kernel still runs on CUDA.
An explicit override always uses the new enforced-budget contract. Responses
and `/healthz` report the effective policy and resize bounds. Weights-only
posttraining uses the new run's image configuration.

Memory depends on the actual image grids and camera count. A smaller pixel
budget can change predictions and fine-detail visibility; input compatibility
alone does not qualify reward accuracy.

## Client/server inference and HTML review

The server loads one checkpoint once and serves predictions over HTTP. The
client sends lossless RGB bytes, ordered camera names, eight
observation slots, task text and requested roles. It sends no server-side file
paths or ground-truth labels. The server uses the same `GranaPredictor` as the
local API/CLI, including the optional `poi_slot` and C's selected-slot focus for D.
GPU requests run serially on one persistent inference worker, preserving warmed
thread-local CUDA handles and cuDNN execution plans across HTTP connections.
Shutdown waits for active inference and releases the worker. Health and static
video requests can proceed concurrently. This small service is for a
private, loopback deployment through SSH or VS Code port forwarding.

Start the server with `--view-policy request` to accept different camera sets
on successive client requests without reloading weights. `/healthz` reports
the server's policy and checkpoint camera defaults. Clients keep the same
request schema; they cannot change the server's policy. The 48 MiB per-request
RGB budget still applies, so supported camera count also depends on input resolution.

Start an inference server on a free local port:

```bash
CUDA_VISIBLE_DEVICES=0 python -m grana.inference.server \
  --checkpoint /runs/my-task/checkpoint-000200 \
  --host 127.0.0.1 --port 8787 --audit-log /reports/grana-requests.jsonl
```

`GET /healthz` becomes available only after checkpoint loading succeeds. It
reports checkpoint identity, complete head loading, camera order, the default POI slot and the request-selected slot policy.
`POST /api/predict` accepts only `grana-binary-v1` and returns a JSON response
with the request ID, elapsed time and inference result. Review pages send the
same binary format. The UTF-8 JSON header uses schema `grana.inference.v1` and
ordered view entries containing `shape: [8,H,W,3]`, byte offset and byte count;
raw uint8 RGB follows the header. Transport performs no sampling, resizing or
image compression. Unknown/GT/path fields and malformed images fail before
inference. Role B still needs an explicit `focus_stage`. The audit log contains
request IDs, status, wire format, shapes, elapsed time and checkpoint identity,
without recording RGB or prompt text.

The Python client accepts the same file request JSON as `grana infer`:

```bash
python -m grana.inference.client --server http://127.0.0.1:8787 \
  --input request.json \
  --roles C D \
  --output /reports/http_prediction.json --plot /reports/http_prediction.png
```

Keep the foreground server terminal alive and use Ctrl-C to stop it. Serving
does not start training or W&B. Use SSH or VS Code port forwarding to reach
loopback from another machine. Server readiness and a working client tunnel
are separate checks.

`--media-manifest PATH` optionally serves a prepared review page and its exact
media allowlist. It provides no directory browser. Health and static media can
be served while the persistent worker serializes prediction requests. Saved
predictions keep their recorded roles, frame windows and D focus; they cannot
be relabelled as a different readout.

### Binary wire format

The sole HTTP request format is `grana-binary-v1`: `POST /api/predict` with
`Content-Type: application/vnd.grana.binary-v1`. Other request content types,
including `application/json`, receive HTTP 415. The body contains a 12-byte
preamble (magic `GRNB`, little-endian uint32 version `1`, little-endian uint32
header length 2..1 MiB), a UTF-8 JSON header, then raw uint8 RGB.

Each view is `{"shape":[8,H,W,3],"offset":int,"nbytes":int}`. Offsets are
relative to the raw section; `nbytes` equals the product of `shape`. Views cover
the raw section exactly, contiguously, in view order, without gaps, overlaps or
trailing bytes. The RGB budget is 48 MiB (50,331,648 bytes) per request; the total
body limit is the preamble plus a 1 MiB header and that RGB budget. Larger bodies
receive HTTP 413, and an invalid or oversized RGB section receives HTTP 400.
Per-image dimension limits still apply. The server exposes views of the received
pixel buffer without copying it and records `wire_format` in the audit log.

`/healthz` advertises `wire_formats: ["grana-binary-v1"]`. The Python client,
`encode_request`, CLI and review browser always use this format. There is no
`wire_format` argument, `--wire-format` option, capability negotiation or retry
as JSON. An older server without binary support must be upgraded; a header over
1 MiB is an error. JSON responses and local `request.json` files are unchanged.

## Compare checkpoints and render videos

Prepare a comparison spec that names the source runs supplying held-out cohorts
and the checkpoints to score. Paths resolve relative to the spec file. For
example, save this as `/reports/compare.json` after replacing its paths:

```json
{
  "cohorts": [{"id": "heldout", "run_dir": "/runs/my-task-001"}],
  "arms": [
    {"id": "before", "label": "Before adaptation", "checkpoint": "/runs/pretrain/checkpoint-005000", "cohorts": ["heldout"]},
    {"id": "after", "label": "After adaptation", "checkpoint": "/runs/my-task-001/checkpoint-000200", "cohorts": ["heldout"]}
  ]
}
```

Freeze shared inputs, score each named arm, then render the saved predictions:

```bash
python scripts/compare_inference.py --prepare \
  --spec /reports/compare.json --manifest /reports/comparison/manifest.json
CUDA_VISIBLE_DEVICES=0 python scripts/compare_inference.py --run \
  --manifest /reports/comparison/manifest.json --arm before
CUDA_VISIBLE_DEVICES=0 python scripts/compare_inference.py --run \
  --manifest /reports/comparison/manifest.json --arm after
python scripts/render_inference_comparison.py \
  --manifest /reports/comparison/manifest.json
```

The manifest freezes checkpoint identities, held-out trajectories and exact RGB
windows. The comparison renderer supports one to four named arms per cohort
and requires exactly four ordered camera views; camera aliases are configurable.
Labels are used for evaluation and plots, never as predictor inputs.
Rendering reads saved predictions and needs FFmpeg with libx264 (`--ffmpeg` can
select the binary). Keep the HTML index and its media together. Sampled-frame
videos are qualitative diagnostics; their playback rate is not inference
throughput or original-speed video. Compare D's focus-relative error alongside
C stage accuracy and the accumulated progress metric.

To add a live review page, choose an arm and an episode ID from that manifest:

```bash
python scripts/prepare_review.py \
  --comparison-manifest /reports/comparison/manifest.json \
  --arm after --episode YOUR_EPISODE_ID --output /reports/review
```

Use the checkpoint printed by the preparation command when starting the HTTP
server, and pass `--media-manifest /reports/review/media_manifest.json`.
Preparation verifies saved comparison and RGB identities and creates an exact
media allowlist. The review browser defaults to C,D with POI slot 5; live
requests apply the selected frame and C-conditioned D focus.
