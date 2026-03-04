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
echo ""
echo "GPU Setup Notes:"
echo "  - For RTX 4500 Ada: Use Python 3.11 + TensorFlow 2.16.1 for best GPU performance"
echo "  - For development: Current setup works on CPU (slower but functional)"
echo "  - See GPU_SETUP_ISSUES_AND_WORKAROUNDS.md for detailed instructions"
