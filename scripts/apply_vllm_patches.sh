#!/usr/bin/env bash
# Apply the two BlitzInfer patches to the installed vLLM.
#
# Both patches are no-ops outside the gateway (gated on env vars), so it's
# safe to apply them to a vLLM installation that's also used by other
# tools — `vllm serve` etc. will behave identically when BLITZ_GPU_GO_FILE
# / BLITZ_SHARED_POOL are unset.
#
# Idempotent: if a patch is already applied, `patch` reports "Reversed
# (or previously applied)" and skips it.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PATCH_DIR="${REPO_DIR}/patches"

# Locate the installed vLLM. Prefer the active venv, fall back to the
# user/system import path.
VLLM_DIR="$(python3 -c 'import os, vllm; print(os.path.dirname(vllm.__file__))' 2>/dev/null || true)"
if [[ -z "${VLLM_DIR}" || ! -d "${VLLM_DIR}" ]]; then
    echo "error: vLLM is not importable. Activate your venv and 'pip install vllm==0.20.1' first." >&2
    exit 1
fi

# vLLM lives at <site-packages>/vllm/...; the patch headers reference
# vllm/v1/worker/... and vllm/model_executor/... — so apply from the
# site-packages root.
SITE_ROOT="$(dirname "${VLLM_DIR}")"

echo "vLLM at: ${VLLM_DIR}"
echo "applying from: ${SITE_ROOT}"
echo

cd "${SITE_ROOT}"
for p in "${PATCH_DIR}"/0*.patch; do
    name="$(basename "${p}")"
    if patch --dry-run --silent -p1 -R < "${p}" >/dev/null 2>&1; then
        echo "  ${name}: already applied, skipping"
        continue
    fi
    if patch -p1 -N --silent < "${p}"; then
        echo "  ${name}: applied"
    else
        echo "  ${name}: FAILED — inspect manually" >&2
        exit 1
    fi
done

echo
echo "done. To revert: cd '${SITE_ROOT}' && for p in '${PATCH_DIR}'/0*.patch; do patch -R -p1 < \"\$p\"; done"
