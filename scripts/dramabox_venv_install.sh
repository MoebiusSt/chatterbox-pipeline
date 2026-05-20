#!/usr/bin/env bash
# Create a DramaBox-only venv (recommended). Safe for cbpipe: run once, never merges deps.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_DRAMA="$(cd "${SCRIPT_DIR}/../.." && pwd)/DramaBox"
DRAMA_ROOT="${1:-${DRAMABOX_ROOT:-${DEFAULT_DRAMA}}}"
DRAMA_ROOT="$(cd "${DRAMA_ROOT}" && pwd)"

if [[ ! -f "${DRAMA_ROOT}/requirements.txt" ]]; then
    echo "Not a DramaBox checkout (missing requirements.txt): ${DRAMA_ROOT}"
    echo "Usage: DRAMABOX_ROOT=/path/to/DramaBox $0"
    exit 1
fi

VENV_DIR="${DRAMA_ROOT}/.venv-drama"
python3 -m venv "${VENV_DIR}"
# shellcheck source=/dev/null
source "${VENV_DIR}/bin/activate"

pip install --upgrade pip wheel
pip install -r "${DRAMA_ROOT}/requirements.txt"

echo ""
echo "DramaBox venv: ${VENV_DIR}"
echo "Point the tester worker at:"
echo "  export CHATTERBOX_TESTER_DRAMABOX_PYTHON=\"${VENV_DIR}/bin/python\""
echo "Then: ${SCRIPT_DIR}/run_chatterbox_tester_dramabox.sh"
