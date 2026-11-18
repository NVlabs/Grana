# Local annotation playground

This source-checkout prototype puts the three annotation routes for one recipe on one local home page. It needs an existing valid cache recipe and its local LeRobot v2/v3 source. Existing labels and caches are not copied.

Use the repository's CPU annotation environment and run from the checkout root:

```sh
.venv-annotation/bin/python -m tools.annotation_playground \
  --config /absolute/path/to/cache-recipe.yaml \
  --workspace /absolute/path/to/private-playground
```

Open **http://127.0.0.1:8840/**. The home page uses `--port` (default 8840) and the four label pages the next four ports (human, API drafts, seeded drafts, seed page); Ctrl+C stops all five. Those ports must be free. Reusing the same command resumes the workspace without resetting labels, jobs, steps or the budget; a resumed workspace may use another `--port`. `--init-only` prepares the workspace without starting any service.

A single episode is enough to start the playground and use human labeling or pure API annotation independently. Only seed transfer needs at least two episodes, with separate seed and held-out validation episodes; the seed view explains this when its selection contains just one.

| Option, fixed when the workspace is created | Default |
|---|---|
| `--human-episodes`, `--api-episodes`, `--seeded-episodes` (e.g. `0,3,5-9`) | the recipe's episodes |
| `--max-requests`: HTTP requests all routes and models may send together | 3 |
| `--settings` or `--criteria`: the API route's task policy | generic draft criteria |
| `--models`: the models the page may generate with ([below](#models)) | explicit CLI/environment configuration, or no models |

## The home page

The home page first asks for a mode: **Human labeling**, **API drafts** or **Seed examples**, each with one line, its status and Enter. Each mode has its own view, so reload and Back preserve your place; *← Home* returns to the chooser. The header (the task, the recipe, its episode count, the requests left, the workspace, English/中文) stays on every view, and home keeps a small *Side by side…* link that opens one episode in the three editors at once. Each label page links back to its mode view. Messages from the local service, refusals included, appear in the page's language.

1. **Human labeling** opens the label page of `grana label` for the human route.
2. **API drafts** lists the route's episodes as not generated, generated, draft or submitted for the model chosen above the list; a row also names the other models asked for that episode, and a draft the model it came from. **Generate stages** runs bounded first-round requests with that model; the unchecked **Also annotate AQ after stages** option adds short second-round requests within each frozen stage. **Add AQ** runs that optional round later from the saved stages; **Import** writes its result as the episode's draft with `grana.annotation.candidate_import` (`failure_quality` regressed); **Open** shows it on the label page. **Import all** runs `grana draft-labels` on that model's results as a dry run first, lists what would be written and why the rest is skipped, and writes only after **Confirm import**. Results made elsewhere can go through the same preview from a folder.
3. **Seed examples** follows the [annotation guide](../../docs/ANNOTATION.md#bounded-campaigns-and-the-local-playground):
   1. *Choose* 1 to 5 seeds and at least one held-out episode among those submitted on the seed page; each shows whether it was reviewed with the review tick, as an 11-key file, or started from an API draft, and is Unused until you choose. A sixth seed is turned down where it is chosen. The episodes without submitted labels are the targets.
   2. *Build example* exports the seeds as one template with `grana.annotation.templates.export_template` into `seeded/templates/`, the seeds sharing its 96 sampled images. A refused seed (an 11-key file, say) is named, with a link to review and submit it again on the seed page.
   3. *Validate* generates the held-out episodes with the model chosen there, every request carrying all seeds, and runs `seed_examples.py compare` with the template's policy: subtask agreement and, only when AQ was requested, AQ recall/precision per class, human events found and boundary error, a row per held-out episode and model. A row opens that episode in detail: the human and the model's subtasks with their outcomes, each human cut beside the nearest model cut and their difference, action quality, and the human progress preview. The API rounds do not infer outcomes, so no model-derived training curve is manufactured.
   4. *Apply* generates the targets you choose with a model only after all selected held-out episodes have a successful, unchanged comparison under the current example, then imports them all through the same preview.

   Changing a seed's labels marks its example out of date and lists the episodes to rebuild. Rebuild the example and validate again before applying it; earlier results and comparisons stay with their original example. Saving identical labels again does not require rebuilding. Batch import checks that the example and comparison are still current when you confirm, including after a preview was opened. A draft is never replaced: once an episode has one, other models' results for it are skipped.

## Models

No provider or model is built in. Without complete provider configuration the workspace
opens for human labeling and offline preparation, with generation unavailable.
For one model, supply `--base-url` and `--model`, or set
`GRANA_ANNOTATION_BASE_URL` and `GRANA_ANNOTATION_MODEL` before creating the
workspace. CLI values take precedence. The key is read from
`GRANA_ANNOTATION_API_KEY`, or the variable named by `--api-key-env`.

For multiple models, pass `--models /absolute/path/to/models.json` when creating
the workspace. Each entry specifies its provider independently; it does not
inherit environment defaults. The first entry is selected in the page. The
following is a format example: replace the `.invalid` endpoint and model
placeholder with values from your provider before use. Placeholders are rejected.
Lines starting with `//` are comments.

```json
[
  {"id": "primary", "label": "My vision model", "base_url": "https://provider.invalid/v1",
   "model": "YOUR-MODEL-NAME", "api_key_env": "GRANA_ANNOTATION_API_KEY",
   "reasoning_effort": "xhigh", "protocol": "responses"}
]
```

An empty list `[]` explicitly creates a workspace for human labeling only.

| Field | Value |
|---|---|
| `id` | short and unique: 1–32 lowercase letters, digits, `.`, `_` or `-`; jobs and comparisons are filed under it |
| `label` | the name the page shows |
| `base_url` | an HTTPS endpoint without credentials, query or fragment; requests go to `<base_url>/responses` |
| `model` | the provider's model name, sent with every request |
| `api_key_env` | the name of the environment variable that holds the key (letters, digits and `_`), never the key |
| `reasoning_effort` | `high` or `xhigh` |
| `protocol` | `responses`, the only protocol supported today; any other is refused |

Each entry needs exactly these fields. A Responses-compatible endpoint takes the request with a JSON-schema output format and answers with the same `model` and reasoning effort; any other answer fails that request. Export every key variable before launching: the page shows each model's label and whether its key variable is set, never a value, and Generate is off for a model whose variable is unset. The list is checked when the workspace is created and kept there, so a resumed workspace uses the same models; a changed explicit provider configuration is refused. Resume without provider
options to reuse the saved list, even if the current provider environment has
changed. To add a provider to a workspace created without one, create a fresh
workspace; preserve the original human labels and import them through the
normal label workflow.

## Requests and the workspace

Never put a credential value on the command line, in a recipe or in a model list. Human labeling and offline preparation work without a provider or key; saved
results can be verified and imported without credentials.

Requests start only from a Generate stages or Add AQ click. Every route and model spends the same persisted HTTP-attempt budget and keeps receipts in `api-budget/`. One episode can need several requests: stages first, then optional stage-local AQ. Increase `--max-requests` when creating a workspace if the three-attempt preview budget is too small; failures count and saved limits never silently grow. A request interrupted by a stop keeps its episode blocked for receipt inspection and is never retried; a queued one that never started is cancelled when the service restarts. Stage clips default to at most 120 seconds sampled every 2 seconds; AQ clips to at most 10 seconds sampled every 0.2 seconds. Both use at most 96 target mosaics of 320×240 per view. Optional seed references have their own shared 96-image cap. Changing stages in a label page makes it differ from the route's settings; generation then stops before the request and keeps those edits.

Keep the workspace private and outside code checkouts and raw datasets: it holds labels, sampled images, seed templates, receipts and model candidates. The home page serves none of its files, and accepts changes only from itself. The launcher starts no background services and schedules nothing.

Skipped AQ is recorded as `default_normal_as_not_requested`: valid unmarked time defaults to Normal, explicitly distinct from model-annotated Normal and excluded from AQ accuracy metrics. AQ failure retains the completed stage result. Imported drafts still require the normal explicit human review before full training-label submission. Comparison completeness is a gate, not an accuracy threshold.
