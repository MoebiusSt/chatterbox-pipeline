#!/usr/bin/env bash
# Run chatterbox_tester with the NORMAL pipeline venv + isolated DramaBox worker.
# Prerequisites:
#   1. scripts/dramabox_venv_install.sh (once)
#   2. export CHATTERBOX_TESTER_DRAMABOX_PYTHON=.../.venv-drama/bin/python OR default below.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENV_PY="${PIPELINE}/venv/bin/python"

if [[ ! -x "${VENV_PY}" ]]; then
    echo "Pipeline venv missing: ${VENV_PY}"
    exit 1
fi

if [[ -z "${CHATTERBOX_TESTER_DRAMABOX_PYTHON:-}" ]]; then
    DEFAULT_DR="$(cd "${PIPELINE}/.." && pwd)/DramaBox/.venv-drama/bin/python"
    if [[ -x "${DEFAULT_DR}" ]]; then
        export CHATTERBOX_TESTER_DRAMABOX_PYTHON="${DEFAULT_DR}"
        echo "(Using default CHATTERBOX_TESTER_DRAMABOX_PYTHON=${DEFAULT_DR})"
    else
        echo "Set CHATTERBOX_TESTER_DRAMABOX_PYTHON to DramaBox/.venv-drama/bin/python"
        exit 1
    fi
fi

export PYTHONPATH="${PIPELINE}/src:${PYTHONPATH:-}"
exec "${VENV_PY}" "${PIPELINE}/chatterbox_tester.py" "$@"
