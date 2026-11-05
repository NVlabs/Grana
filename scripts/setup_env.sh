#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Build the independent native Qwen3.5 training environment. No system changes.
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${PROJECT_ROOT}/.venv-qwen35-unsloth"
PYTHON_VERSION=3.12.3
export UV_LINK_MODE=copy UV_NO_CONFIG=1
# A host-wide pip constraint (some containers set one for their system Python)
# must not override this checkout's pinned constraints.
unset PIP_CONSTRAINT
export USE_TF=0 TRANSFORMERS_NO_TF=1 USE_FLAX=0 TRANSFORMERS_NO_FLAX=1
# FLA_TILELANG/FLA_DISABLE_BACKEND_DISPATCH come from the venv FLA profile (scripts/fla_profile.py).

if [[ $# == 1 && "${1:-}" == "--help" ]]; then
    cat <<'HELP'
Usage: bash scripts/setup_env.sh [--plan | --check]
Builds .venv-qwen35-unsloth using native Transformers, PEFT, FLA and Ray.
Requires an existing uv-managed CPython 3.12.3 (uv python install 3.12.3), git,
NVIDIA CUDA 13 compatible driver, and Linux. SOURCE_PYTHON can select that interpreter.
GRANA_FLA_TILELANG=auto|1|0 (default auto) selects the FLA runtime profile:
auto adds TileLang and a CUDA 13.0 compiler (requirements/hopper-tilelang.txt)
when a local GPU is Hopper (sm_90), where FLA 0.5.2 refuses Triton 3.6's gated
backward; 1/0 force or remove it. See scripts/fla_profile.py.
The Python runtime and packages are copied into this checkout; no links or
system packages are created. Existing copied environments are updated in place.
Set PIP_CACHE_DIR and UV_PYTHON_INSTALL_DIR to reuse your own package/Python caches.
HF_HOME should point at your external model cache when training.
It also installs the hash-pinned causal-conv1d inference kernel
(requirements/inference-fast.txt) and verifies it on the GPU.
CPU architecture automatically selects the x86_64 or aarch64 native wheels.
--plan prints CPU/GPU detection and selected requirements without installing.
--check only verifies the existing environment. See docs/installation.md.
HELP
    exit 0
fi
if [[ $# == 1 && "${1:-}" == "--plan" ]]; then
    PLAN_PYTHON="${SOURCE_PYTHON:-}"
    if [[ -z "${PLAN_PYTHON}" ]]; then
        if [[ -x "${VENV}/bin/python" ]]; then
            PLAN_PYTHON="${VENV}/bin/python"
        else
            PLAN_PYTHON="$(command -v python3)" || { echo "--plan needs Python 3; set SOURCE_PYTHON to its path" >&2; exit 1; }
        fi
    fi
    exec "${PLAN_PYTHON}" "${PROJECT_ROOT}/scripts/fla_profile.py" plan --mode "${GRANA_FLA_TILELANG:-auto}"
fi
if [[ $# == 1 && "${1:-}" == "--check" ]]; then
    [[ -x "${VENV}/bin/python" ]] || { echo "No environment at ${VENV}; run: bash scripts/setup_env.sh" >&2; exit 1; }
    exec "${VENV}/bin/python" "${PROJECT_ROOT}/scripts/check_env.py"
fi
[[ $# == 0 ]] || { echo "Unknown arguments; use --help" >&2; exit 2; }
[[ "$(uname -s)" == Linux ]] || { echo "This runtime requires Linux" >&2; exit 1; }
[[ ! -L "${VENV}" ]] || { echo "Refusing a linked environment: ${VENV}" >&2; exit 1; }
if [[ ! -e "${VENV}" ]]; then
    command -v uv >/dev/null || { echo "uv is required: https://docs.astral.sh/uv/getting-started/installation/" >&2; exit 1; }
    SOURCE_PYTHON="${SOURCE_PYTHON:-$(uv python find --system --managed-python --no-python-downloads "${PYTHON_VERSION}")}" \
        || { echo "Install the managed interpreter first: uv python install ${PYTHON_VERSION}" >&2; exit 1; }
    "${SOURCE_PYTHON}" "${PROJECT_ROOT}/scripts/create_copied_venv.py" \
        --source-python "${SOURCE_PYTHON}" --venv "${VENV}" --python-version "${PYTHON_VERSION}"
fi
[[ -f "${VENV}/grana-python-copy.json" ]] || { echo "Refusing a non-owned environment: ${VENV}" >&2; exit 1; }
PYTHON="${VENV}/bin/python"
FLA_MODE="${GRANA_FLA_TILELANG:-auto}"
RUNTIME_PLAN="$("${PYTHON}" "${PROJECT_ROOT}/scripts/fla_profile.py" plan --mode "${FLA_MODE}")"
printf '%s\n' "${RUNTIME_PLAN}"
FLA_PROFILE="$("${PYTHON}" -c 'import json, sys; print(json.loads(sys.argv[1])["profile"])' "${RUNTIME_PLAN}")"
pip_install() { "${PYTHON}" -m pip --disable-pip-version-check install --no-build-isolation --constraint "${PROJECT_ROOT}/requirements/constraints-gb300.txt" "$@"; }
pip_install 'pip==26.2.1' 'setuptools==84.0.0' 'wheel==0.48.0' 'hatchling==1.32.4'
pip_install --extra-index-url https://download.pytorch.org/whl/cu130 \
    'torch==2.10.0+cu130' 'torchvision==0.25.0+cu130' 'triton==3.6.0'
pip_install --extra-index-url https://download.pytorch.org/whl/cu130 \
    --requirement "${PROJECT_ROOT}/requirements/runtime-gb300.txt"
pip_install 'pytest==9.1.1'
HOPPER_REQUIREMENTS="${PROJECT_ROOT}/requirements/hopper-tilelang.txt"
if [[ "${FLA_PROFILE}" == hopper-tilelang ]]; then
    pip_install --requirement "${HOPPER_REQUIREMENTS}"
else
    # Updating a former hopper-tilelang environment in place removes its packages.
    mapfile -t HOPPER_PACKAGES < <(sed -e 's/#.*//' -e 's/==.*//' -e '/^[[:space:]]*$/d' "${HOPPER_REQUIREMENTS}")
    "${PYTHON}" -m pip --disable-pip-version-check uninstall --yes "${HOPPER_PACKAGES[@]}" 2>&1 \
        | grep -v 'WARNING: Skipping' || true
fi
"${PYTHON}" "${PROJECT_ROOT}/scripts/fla_profile.py" write --venv "${VENV}" \
    --profile "${FLA_PROFILE}" --mode "${FLA_MODE}"
if [[ -f "${PROJECT_ROOT}/pyproject.toml" ]]; then
    pip_install --no-deps --editable "${PROJECT_ROOT}"
fi
pip_install --no-deps --requirement "${PROJECT_ROOT}/requirements/inference-fast.txt"
"${PYTHON}" "${PROJECT_ROOT}/scripts/check_env.py"
echo "Activate: source ${VENV}/bin/activate"
