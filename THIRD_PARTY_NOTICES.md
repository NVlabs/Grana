# Third-party notices

Grana is Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE).

Each third-party component retains its original license. See
[THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md) for license references.

These notices cover the Grana training, inference, cache, and annotation source
release. See [NOTICE](NOTICE) for project attribution and the reference for its
licensing-document conventions.

## Python packages and external artifacts

Grana installs Python dependencies separately. Their source trees and local
Python environments are not bundled in the source release. The dependency
inventory, including optional annotation packages, native inference kernels,
the separate CPU labeling profile, and setup/test tools, is in
[THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md).

The source release includes dependency declarations and wheel download URLs,
not the referenced dependency wheels. In particular, the separately installed
causal-conv1d kernel retains its BSD-3-Clause license and Ninja retains its
Apache-2.0 license. The local annotation UI uses the JavaScript and stylesheets
shipped under `grana/annotation/label_ui/`; its pages do not load a third-party
JavaScript bundle or web font from a CDN.

These notices identify source-release dependencies. They are not a complete
license bundle for a redistributed Python environment or container. If
third-party source or binaries are added to a distribution, include their
applicable license texts and notices with that distribution; preserve upstream
attributions and record copied or adapted code as described in
[CONTRIBUTING.md](CONTRIBUTING.md).

Qwen model weights, datasets, training caches, checkpoints, generated videos,
FFmpeg, and CUDA libraries are obtained or produced separately and retain
any applicable terms. No rights to those artifacts are granted by Grana's
source-code license.
