#!/usr/bin/env bash
# Run chatterbox_tester with the NORMAL pipeline venv + isolated Higgs Audio worker.
# Prerequisites:
#   1. higgs-audio/scripts/higgs_venv_install.sh (once)
#   2. python3 scripts/download_pinned_models.py (once, in higgs-audio repo)
#   3. export CHATTERBOX_TESTER_HIGGS_PYTHON=.../.venv-higgs/bin/python OR default below.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENV_PY="${PIPELINE}/venv/bin/python"

if [[ ! -x "${VENV_PY}" ]]; then
    echo "Pipeline venv missing: ${VENV_PY}"
    exit 1
fi

if [[ -z "${CHATTERBOX_TESTER_HIGGS_PYTHON:-}" ]]; then
    DEFAULT_HIGGS="$(cd "${PIPELINE}/.." && pwd)/higgs-audio/.venv-higgs/bin/python"
    if [[ -x "${DEFAULT_HIGGS}" ]]; then
        export CHATTERBOX_TESTER_HIGGS_PYTHON="${DEFAULT_HIGGS}"
        echo "(Using default CHATTERBOX_TESTER_HIGGS_PYTHON=${DEFAULT_HIGGS})"
    else
        echo "Set CHATTERBOX_TESTER_HIGGS_PYTHON to higgs-audio/.venv-higgs/bin/python"
        exit 1
    fi
fi

if [[ -z "${CHATTERBOX_TESTER_HIGGS_ROOT:-}" ]]; then
    DEFAULT_ROOT="$(cd "${PIPELINE}/.." && pwd)/higgs-audio"
    if [[ -d "${DEFAULT_ROOT}/boson_multimodal" ]]; then
        export CHATTERBOX_TESTER_HIGGS_ROOT="${DEFAULT_ROOT}"
        echo "(Using default CHATTERBOX_TESTER_HIGGS_ROOT=${DEFAULT_ROOT})"
    fi
fi

export PYTHONPATH="${PIPELINE}/src:${PYTHONPATH:-}"
exec "${VENV_PY}" "${PIPELINE}/chatterbox_tester.py" "$@"
