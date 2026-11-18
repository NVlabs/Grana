# Inference

Run commands from the source root after [installation](installation.md).
Use the CLI for one request, the Python API for repeated local calls, or the HTTP
server for client requests. All use the same checkpoint and prediction contract.

## Inference from a trained checkpoint

Use a completed pretrain or posttrain checkpoint for the intended model size.
Replace example paths with your saved checkpoint and data.
Inference loads the base model, all adapters, both complete heads, and marker embeddings;
it does not open training caches, start Ray, or initialize W&B. Supply a local
base-model snapshot with `--model-name-or-path` if the checkpoint's original
base-model location is unavailable.

### Runtime requirements

Install the [full CUDA environment](installation.md#full-cuda-environment).
Inference uses CUDA image preprocessing, SDPA, FLA delta rules and
causal-conv1d, with compiled decoder and vision modules. These form one
required runtime; missing dependencies, unsupported inputs and compilation
failures raise errors. There is no CPU or alternate-backend fallback.
The predictor uses eight Torch CPU threads.

The paper's 4B BF16 inference benchmark uses **less than 10 GB** of GPU memory
at batch size 1 with eight timesteps and four camera views (32 images).

Compilation is lazy: the first request and new prompt lengths or image grids
may compile graphs. Warm representative requests before timing. Load a local
predictor once and call it from the same thread to preserve CUDA/cuDNN caches;
the HTTP server maintains one inference worker across connections.

LoRA weights merge into the in-memory model without changing checkpoint files.
C and D reuse image processing and vision features within a request. C runs
first; D uses C's selected stage in a separate language forward. There is no
cross-request image, KV or recurrent state cache.

Inference always resizes on CUDA. Train with `data.gpu_image_resize: true` to
align preprocessing; runtime metadata reports whether resizing matches the
checkpoint. Interpolation, BF16 LoRA merging, compilation and batching can
change numerical results. Evaluate accuracy with the runtime and input policy
used for deployment. `/healthz` and `checkpoint.inference_runtime` expose the
effective configuration.

### Request and outputs

Create `request.json` with the checkpoint's camera names **in the saved order**
and your task and stage descriptions. Paths may be absolute or relative to the
JSON file; each points to an RGB uint8 array shaped `[T,H,W,3]`. Select exactly
eight observations:

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
  "stage_prompts": ["Place the blue token inside the tray", "Close the tray lid with the token inside"]
}
```

This example supplies the task fields needed for C,D. B additionally requires
a zero-based `focus_stage`; A requires `compressed_task_prompt`. D derives its
focus internally from C.

Eight-frame arrays can omit `frame_indices`. Each view may instead contain
a list of eight RGB image-file paths, with no `frame_indices`.
TIFF files, including BigTIFF and files renamed to another extension, are rejected
before Pillow opens them. Other Pillow-supported RGB image formats remain accepted.
Repeated frames and explicit `frame_indices` remain in the supplied order.
For full `.npy` video arrays, omit `frame_indices` and pass `--frame-index N`
to compose a window using the shared
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

The predictor, client, CLIs and review browser default to C,D. Use `--roles A`
or `--roles C` for a single role.
`--plot PATH.png` renders all eight returned values and identifies the selected
slot and known source-frame index without another model call.

| Role | Required task fields from the caller | Source of the focus stage | Prediction |
|---|---|---|---|
| A | `compressed_task_prompt` | — | Whole-task progress in [0,1] |
| B | `task`, `stage_prompts`, zero-based `focus_stage` | Caller's `focus_stage` | Progress relative to that focus |
| C | `task`, `stage_prompts` | — | First unfinished stage: zero-based IDs, valid-class logits and probabilities |
| D | `task`, `stage_prompts` | C's hard prediction at the selected input slot | Progress relative to that focus |

B and D both receive task, ordered stages and a focus stage in the same model
prompt format and use the same progress head. The table lists caller-supplied
fields; the API inserts D's focus after running C. A supplied `focus_stage`
controls B only and does not override D's focus.

C selects the first unfinished stage in task order. Failed or incomplete
prerequisites remain unfinished even if the robot attempts a later stage.
If all stages are complete, C selects the final stage.

Inference requires no ground-truth labels. D always uses C's hard prediction
at the **selected input slot** on the same RGB observations. That one focus
conditions all eight D outputs. For `last`, D must be run with C's slot-7 focus.
Reading `D.values[7]` from an old slot-5 request does not perform this change:
all eight D values in that response were conditioned on C's slot-5 focus.

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

Automatic inference windows use CTSP(16,4): context stride 16, tail stride 4,
regular selection and forward order. Training mixtures are configured
separately; see [training configuration](TRAINING_CONFIGS.md).

For an interior POI `p`, context stride `c` and tail stride `t`, the eight slots
are `[p-3c, p-2c, p-c, p-2t, p-t, p, p+t, p+2t]`. If `p < 3c`, the first three
become `[0, p//3, 2*p//3]`; local indices are clipped to the available frame
range. Repeated indices keep their slots. `forward` preserves this selector order,
which can be nonchronological near the start: at `p=9`, `(c,t)=(16,4)` gives
`[0,3,6,1,5,9,13,17]`. Do not sort it.

The progress head attends across all eight observations. A normal slot-5 POI
readout can therefore use two later observations: up to eight source frames
of lookahead with CTSP(16,4). The composer uses source-array indices and does
not resample the frame rate.

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
within that request.

### Repeated local prediction

Load one predictor for repeated calls from the same thread:

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
TIFF-backed PIL images are rejected before their pixels are loaded by the predictor,
HTTP client, or prediction renderer. Callers remain responsible for any decoding
they perform before supplying PIL images or arrays.
Keep original camera pixels; the model's processor resizes them. The compiled
vision path requires equal resized image token counts across the request and
rejects incompatible geometry. The caller supplies frame history and readout;
use the HTTP service for clients arriving on different threads.

To use one checkpoint with different camera sets, select `--view-policy request`
on `grana infer`, or construct `GranaPredictor(..., view_policy="request")`.
Each call then uses its nonempty ordered camera mapping. Every camera still
supplies exactly eight observations; prompts use the actual names and count.
The response records `view_names`, `checkpoint_view_names`, and `view_policy`.
The default `checkpoint` policy continues to reject camera mismatches. Accepting
a new camera set establishes input compatibility; reward accuracy on that set
needs separate evaluation.

Inference inherits the checkpoint's image-pixel budget unless an integer
`--max-pixels` override is supplied. Explicit budgets are enforced after patch
rounding; a grid that exceeds the budget raises an error. Checkpoints without
an `image_preprocessing` descriptor retain their original pixel-budget
interpretation. Resizing always runs on CUDA, and responses and `/healthz`
report the effective policy. Original RGB arrays remain unchanged.

Evaluate new camera sets or pixel budgets on your own data; both affect memory
use and predictions.

## Client/server inference and HTML review

The server loads one checkpoint and applies the same request contract as the
local predictor. Clients send lossless RGB, ordered cameras, eight slots, task
text and requested roles. Model calls run serially on one persistent worker;
health and static media requests can proceed concurrently. Shutdown waits for
active inference. Deploy privately on loopback and use SSH or VS Code port
forwarding for remote clients.

Start the server with `--view-policy request` to accept different camera sets
without reloading weights. Clients cannot override this server policy, and
the per-request RGB budget still applies.

Start an inference server on a free local port:

```bash
CUDA_VISIBLE_DEVICES=0 python -m grana.inference.server \
  --checkpoint /runs/my-task/checkpoint-000200 \
  --host 127.0.0.1 --port 8787 --audit-log /reports/grana-requests.jsonl
```

`GET /healthz` becomes available after checkpoint loading succeeds and reports
checkpoint identity, head loading, camera order and readout policy.
`POST /api/predict` returns a JSON response with the request ID, elapsed time
and prediction. Unknown fields, ground truth, server-side paths and malformed
images fail before inference. The audit log records request IDs, status, wire
format, shapes, elapsed time and checkpoint identity, without RGB or prompt text.

The Python client accepts the same file request JSON as `grana infer`:

```bash
python -m grana.inference.client --server http://127.0.0.1:8787 \
  --input request.json \
  --roles C D \
  --output /reports/http_prediction.json --plot /reports/http_prediction.png
```

Keep the server terminal alive and use Ctrl-C to stop it. Serving does not
start training or W&B. Check both server readiness and the client tunnel.

`--media-manifest PATH` optionally serves a prepared review page and its exact
media allowlist. It provides no directory browser. Saved predictions keep
their recorded roles, frame windows and D focus; they cannot
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

The header uses schema `grana.inference.v1` with ordered view entries.
Transport preserves raw pixels and supplied frame order without sampling,
resizing or compression. `/healthz` advertises `grana-binary-v1`; the Python
client, CLI and review browser use it without a format switch or JSON fallback.
Local request files and responses remain JSON.

## Offline evaluation

Use [evaluate_http_frames.py](../scripts/inference/evaluate_http_frames.py)
`prepare --help` to create a plan from a held-out cache manifest and checkpoint.
Relative cache paths resolve under `--data-root`, or the manifest directory
when omitted. The plan freezes paired POI/LAST windows for the same physical
targets and separate repeatability anchors. Run it locally, then summarize:

```bash
python scripts/inference/evaluate_frames.py \
  --plan /reports/evaluation/plan.json --output /reports/evaluation-local
python scripts/inference/summarize_http_frames.py \
  --experiment /reports/evaluation-local --output /reports/evaluation-summary
```

The offline runner defaults to **logical GPU 0 and one window per batch**.
Use `--devices 0,1 --batch-size 8` for explicitly sized parallel work, or
`--devices auto` to select every visible GPU. Batch size is a per-GPU physical
window count. A fixed batch fails on OOM and is never silently reduced.

`--batch-size auto` enables memory-based sizing and OOM retries. Bound it with
`--memory-fraction` and `--max-batch-size`; by default it retains 10% of currently
free memory. Sizing warmups can compile new shapes and are excluded from timing.
Select devices within the parent's `CUDA_VISIBLE_DEVICES`; workers are isolated
to the selected GPU UUID and use local `cuda:0`. Receipts retain both identities.
The merged output is published only after every planned prediction succeeds.

### Quality metrics

The primary C→D prediction is C's zero-based stage plus D's progress relative
to that stage. Its target is GT stage plus GT within-stage progress; normalized
errors divide by the trajectory's number of stages. B/D focus-relative error
is a separate diagnostic. POI/LAST choices change both the observations and
D's conditioning; evaluate the readout used in deployment.

| Metric | Aggregation |
|---|---|
| Macro RMSE | Compute each trajectory's RMSE across sampling seeds, then average trajectories equally |
| Macro MSE | Average each trajectory's MSE with equal trajectory weight; its square root is not macro RMSE |
| Micro MSE/RMSE | Pool sampled quality observations; repeated targets count once per sampling round |
| Deduplicated metrics | Count each physical target once |

Fixed anchors measure repeatability and are excluded from quality metrics.
Repeated rounds on one persistent worker measure within-process repeatability;
independent server processes are needed to compare across processes. Similar
sample counts per trajectory do not give full-video-duration weights. See
[reference results](results.md#poi-and-last-evaluation) for their scope.

## Advanced tools

These tools retain explicit checkpoint identities, frame selections and saved
predictions. Use `python SCRIPT --help` (or a subcommand's `--help`) for options.
Keep outputs in new directories outside the source tree.

| Task | Tools and input |
|---|---|
| Evaluate via HTTP | [evaluate_http_frames.py](../scripts/inference/evaluate_http_frames.py) `run` collects a prepared plan for one seed through a persistent server; [summarize_http_frames.py](../scripts/inference/summarize_http_frames.py) summarizes completed runs |
| Measure HTTP latency | [benchmark_http.py](../scripts/inference/benchmark_http.py): server URL, full-video request JSON without `frame_indices`, and target frame; sequential physical batch 1 |
| Measure local batch throughput | [benchmark_batch.py](../scripts/inference/benchmark_batch.py): prepared plan or full-video request; one visible GPU and an explicit workload/batch size |
| Record and render readouts | [replay_http.py](../scripts/inference/replay_http.py) records A, C and C+D at POI/LAST; [render_http_replay.py](../scripts/inference/render_http_replay.py) renders its saved manifest |
| Compare checkpoints | [compare_inference.py](../scripts/compare_inference.py) prepares a shared manifest from a spec, then scores each named arm; [render_inference_comparison.py](../scripts/render_inference_comparison.py) renders saved results |
| Prepare a review browser | [prepare_review.py](../scripts/prepare_review.py): comparison manifest, arm and episode; pass its `media_manifest.json` to the server's `--media-manifest` |

A comparison spec names `cohorts` (`id`, `run_dir`) and `arms` (`id`, `label`,
`checkpoint`, `cohorts`); paths resolve relative to the spec. Its renderer accepts
one to four arms per cohort and requires four ordered camera views.
Replay display metadata supplies `trajectory_id`, `source_fps`, `num_frames`
and per-frame `whole_progress`, `stage_ids`, `within_stage_progress`.
Labels are only for evaluation or display and are never predictor inputs.
Renderers read saved predictions without running inference and require FFmpeg
with libx264. Keep each HTML index with its media; playback rate is independent
of inference throughput.

### Timing boundaries

Warm representative inputs before timing: first requests and new shapes may
compile. Keep checkpoint, roles, readout, geometry and prompts fixed for a
comparison. HTTP warmup, frame loading, window composition and server startup
are outside steady-state samples.

| HTTP timing field | Boundary |
|---|---|
| `client_send_to_receive_ms` | First socket send through full response receipt; excludes connection setup, packing and JSON parsing |
| `client_send_ms` | First socket send through the final send return |
| `client_send_complete_to_receive_ms` | Final send return through full response receipt |
| `http_round_trip_ms` | HTTP open through full response receipt, including connection setup |
| `client_total_ms` | Complete client call, including packing and response parsing |

Server execution is already included in client latency. The Python client
exposes these intervals through optional `timing={}` without changing predictions.
Local batch timings include CUDA resize and CPU postprocessing with CUDA
synchronization, and exclude loading and HTTP transport. Amortized time per
selected prediction is whole-workload time divided by the number of windows;
each window contributes one selected prediction even though eight slots return.
Complete batch latency remains the wait for all results.
