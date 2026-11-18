# Contributing to Grana

Contributions to Grana are accepted under the [Apache License 2.0](LICENSE).
Preserve existing upstream copyright and license notices. When adding or
adapting third-party code, document its source, revision, license, and any
modifications in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md), and include
its required license text in [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md).

## Source headers

Source files in this repository use SPDX headers for the
[Apache License 2.0](LICENSE). For NVIDIA-authored Python (`.py`) and shell
(`.sh`) files, use:

```python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
```

Use the same SPDX fields in a C-style comment for JavaScript (`.js`), CommonJS
(`.cjs`), and CSS (`.css`):

```c
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
```

Apply NVIDIA copyright headers only where NVIDIA authorship is established.
Use the file's established copyright year or range; the examples above use
2026. Preserve existing copyright years, upstream notices, source comments,
and full license texts. Keep executable shebangs and Python encoding
declarations first. Documentation and configuration formats, including
Markdown, HTML, YAML, JSON, and TOML, and binary assets do not require these
comment headers. Preserve any license declarations already present.

Keep the root license, attribution, security and contribution documents listed
in `release-files.txt` with source releases. Wheel distributions include the
license and notice documents declared in `pyproject.toml` and README metadata.
The formatting provenance of these source-header and contribution-document
conventions is recorded in [NOTICE](NOTICE); third-party source attribution
belongs in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Signing Your Work

Every contributor must sign off each commit under the
[Developer Certificate of Origin 1.1](https://developercertificate.org/).
A sign-off certifies that you created the contribution or have the right to
submit it under the applicable open-source license. This project uses DCO
sign-off and does not require a separate contributor license agreement.

Use your real name and email address:

```bash
git commit -s -m "Describe the change"
```

This adds a trailer such as:

```text
Signed-off-by: Your Name <your.email@example.com>
```

Commits without the required sign-off will not be accepted.

## Package organization

```text
grana/
├── annotation/       # Human labels, evidence, API proposals, review and export
├── data/             # Cache builder, reader, role sampling and target policies
├── modeling/         # Qwen/LoRA model, prompts, inputs and checkpoints
├── training/         # Ray worker, losses, held-out evaluation and logging
├── inference/        # Predictor, request CLI and HTTP client/server
├── cli.py            # The grana command
├── config.py         # Recipe loading and validation
└── runtime.py        # Checkout environment and Ray worker checks
```

Use relative imports within a domain and explicit domain paths across domains.
Keep package initializers lightweight so annotation and planning utilities do
not import model runtimes. Use `grana._paths.PROJECT_ROOT` to locate checkout
resources.

New code uses domain paths such as `grana.data.builder`,
`grana.annotation.workflow`, and `grana.training.engine`.
`grana/_compat.py` preserves historical flat imports, including `grana.cache`
and `grana.annotate`, and module entry points such as `python -m grana.serve`
and `python -m grana.client`. The public `grana` CLI and
`from grana.inference import GranaPredictor, InferenceRequest` API remain stable.

## Source releases

Update `release-files.txt` when adding or removing a public source file. The
packager validates this exact manifest and records SHA-256 digests:

```bash
python scripts/package_release.py --output dist/grana-source.tar.gz
```

Run `python scripts/check_release.py` to check the checkout without packaging.
After extracting an archive, check it with
`python scripts/check_release.py --root /path/to/extracted/grana`.
Keep datasets, model weights, environments, credentials, and local execution
artifacts outside source releases. See [NOTICE](NOTICE) for attribution.

## Development workflow

1. Fork the repository and create a feature branch from `main`.
2. Follow the [installation guide](docs/installation.md) for the CPU annotation
   environment or full CUDA runtime. It distinguishes installation checks from
   training capacity and model qualification. Use
   [standalone project recipes](docs/project-configuration.md) for your own
   data; installation does not require the shipped task examples.
3. Keep changes focused on the cached-data training, inference, and annotation
   workflows. Add or update tests when behavior changes, and describe the
   expected behavior and validation in your pull request.
4. Run the relevant tests while developing. Before submitting a change, run
   the complete suite in the configured environment:

   ```bash
   uv python install 3.12.3
   bash scripts/setup_env.sh
   source .venv-qwen35-unsloth/bin/activate
   python -m pip install -r requirements/annotation.txt
   python -m pytest tests
   ```

   Annotation API tests use local fixtures or mocks; do not include credentials
   or private robot recordings in a test or pull request. GPU training and
   external API runs are separate from the unit-test suite. Report their
   configuration and results when they are relevant to your change.
5. For changes to the public file set or packaging, follow
   [Source releases](#source-releases) and check the extracted archive.
6. Open a pull request against `main` describing the problem, the resulting
   behavior, and the checks you ran. Preserve the distinction between an API
   annotation suggestion and a reviewed training annotation.

### Optional local checks

Annotation tests use synthetic recordings. Video fixtures require `ffmpeg` with
libx264 and `ffprobe` on `PATH`. JavaScript checks need Node.js 18+; the optional
browser check needs Node.js 22+ and Chrome/Chromium (`CHROME_BIN` can select it).
Run these from the source root in an environment with the test dependencies:

```bash
python -m pytest tests/test_labeling.py tests/test_label_cli.py tests/test_label_ui.py
GRANA_BROWSER_SMOKE=1 python -m pytest -s tests/test_label_ui_browser.py
```

Set `GRANA_BROWSER_OUTPUT` to an external directory to retain browser artifacts.
A skipped test does not establish that the browser path ran. An optional CUDA
compilation check uses a synthetic model without downloading weights; select an
available GPU:

```bash
CUDA_VISIBLE_DEVICES=0 python -m pytest \
  tests/test_inference_acceleration.py::test_real_compile_on_cuda_matches_eager_closely
```

## Portable installation documentation

Keep installation and environment checks independent of task presets, private
recordings and model payloads. Publish instructions for both source checkouts
and release archives. Use user-selected paths, environment variables and
standalone project recipes; avoid personal repository URLs, hostnames and
machine-specific directories. Documentation, configuration keys and task
definitions use English; localized interface copy belongs in the UI catalog.

The installer, environment checker and dependency requirement files are runtime
assets, not task examples. Keep them in source releases. When retiring example
configs or campaign fixtures, update their dedicated documentation, tests and
exact entries in `release-files.txt` in the same change. The
generic installation and project-configuration guides must remain usable
without those examples. Keep local verification receipts outside the release.

Follow [SECURITY.md](SECURITY.md) for private vulnerability reports and handling
of credentials, recordings, model artifacts, and local services. Contributors
are expected to communicate respectfully and professionally.

## Full text of the DCO

```text
Developer Certificate of Origin
Version 1.1

Copyright (C) 2004, 2006 The Linux Foundation and its contributors.

Everyone is permitted to copy and distribute verbatim copies of this
license document, but changing it is not allowed.


Developer's Certificate of Origin 1.1

By making a contribution to this project, I certify that:

(a) The contribution was created in whole or in part by me and I
    have the right to submit it under the open source license
    indicated in the file; or

(b) The contribution is based upon previous work that, to the best
    of my knowledge, is covered under an appropriate open source
    license and I have the right under that license to submit that
    work with modifications, whether created in whole or in part
    by me, under the same open source license (unless I am
    permitted to submit under a different license), as indicated
    in the file; or

(c) The contribution was provided directly to me by some other
    person who certified (a), (b) or (c) and I have not modified
    it.

(d) I understand and agree that this project and the contribution
    are public and that a record of the contribution (including all
    personal information I submit with it, including my sign-off) is
    maintained indefinitely and may be redistributed consistent with
    this project or the open source license(s) involved.
```
