#!/usr/bin/env bash
# Re-install chatterbox-pipeline dependencies into the repo venv after optional
# experiments that changed Torch/Gradio/Transformers versions.
#
# chatterbox-tts is not declared in requirements.txt because its wheel pins transformers==4.46.3,
# which conflicts with qwen-tts (transformers==4.57.3). requirements.txt installs a unified stack +
# chatterbox transitive deps; this script reinstalls chatterbox without pulling those pins (--no-deps).

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENV_PYTHON="${ROOT}/venv/bin/python"
if [[ ! -x "${VENV_PYTHON}" ]]; then
    echo "No executable at ${VENV_PYTHON} — create with: cd ${ROOT} && python3 -m venv venv"
    exit 1
fi

echo ">>> Interpreter: ${VENV_PYTHON}"

"${VENV_PYTHON}" -m pip install --upgrade pip
"${VENV_PYTHON}" -m pip install -r "${ROOT}/requirements.txt"

# chatterbox must not appear in requirements.txt (transformers vs qwen-tts resolver conflict).
CHATTERBOX_PIN="${CHATTERBOX_PIN:-chatterbox-tts==0.1.3}"
echo ">>> Installing ${CHATTERBOX_PIN} (--no-deps, skips wheel pins incl. transformers/pkuseg) …"
"${VENV_PYTHON}" -m pip install --force-reinstall --no-deps "${CHATTERBOX_PIN}"

echo ""
echo "Restore step finished."
echo ""
echo 'If Dramabox bumped Torch beyond chatterbox specs, align CUDA wheels explicitly,'
echo 'e.g. Torch 2.6 + torchvision 0.21 + torchaudio 2.6 from https://pytorch.org/get-started/locally/'
echo ''
echo 'WhisperX / pyannote (validation Alignment): torchvision must ABI-match pip torch.'
echo 'If you see ``torchvision::nms does not exist`` or cudnn dropouts/pyannote aborts:'
echo '  pip install torchvision==0.23.0 --extra-index-url https://download.pytorch.org/whl/cu128'
echo '(adjust cu121/cu124/cu128 to match python -c "import torch; print(torch.__version__)").'
