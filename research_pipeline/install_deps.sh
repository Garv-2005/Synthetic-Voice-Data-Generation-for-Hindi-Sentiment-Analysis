#!/usr/bin/env bash
set -euo pipefail

VENV_NAME=${1:-venv}

echo "Creating virtual environment: ${VENV_NAME}"
python3 -m venv "${VENV_NAME}"

VENV_PY="${VENV_NAME}/bin/python"
if [ ! -x "${VENV_PY}" ]; then
  echo "ERROR: virtualenv python not found at ${VENV_PY}" >&2
  exit 1
fi

echo "Upgrading pip in virtual environment"
"${VENV_PY}" -m pip install --upgrade pip

echo "Installing dependencies from requirements.txt"
"${VENV_PY}" -m pip install -r requirements.txt

echo "Installation complete. To activate the virtual environment run:"
echo "  source ${VENV_NAME}/bin/activate"
echo "If you need GPU support, install NVIDIA drivers, CUDA and cuDNN manually as described in TENSORFLOW_GPU_CONFIG.md"
