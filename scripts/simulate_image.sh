#!/usr/bin/env bash
# Simulates the Docker image without Docker, in two steps:
#   1. builds the weights into a fresh folder, like the Dockerfile's `RUN python download_weights.py` (with internet);
#   2. starts the worker inside a namespace with no network at all, as on RunPod, and sends a health check.
# HOME, the Hugging Face cache and PKUSEG_HOME are fresh as well, so no cache on this machine can hide a
# download that the image would make at startup.
#
# Usage (from the repo root):  scripts/simulate_image.sh [FOLDER]    (default: /var/tmp/chatterbox-image, ~3 GB)
# PYTHON selects the interpreter (default: .venv/bin/python).
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
# -s keeps the venv's symlinked interpreter; resolving it would bypass the venv.
PYTHON=$(realpath -s "${PYTHON:-$ROOT/.venv/bin/python}")
FOLDER=$(realpath -m "${1:-/var/tmp/chatterbox-image}")
cd "$ROOT"
mkdir -p "$FOLDER/home"

# The same variables as the Dockerfile, with /models replaced by the folder.
export HOME="$FOLDER/home" HF_HOME="$FOLDER/models/hf" CHATTERBOX_MODELS="$FOLDER/models/chatterbox" \
       PKUSEG_HOME="$FOLDER/models/pkuseg"

echo "== build (with internet)"
env -u LD_LIBRARY_PATH "$PYTHON" download_weights.py

echo "== start (no network)"
env -u LD_LIBRARY_PATH HF_HUB_OFFLINE=1 unshare -rn "$PYTHON" -c \
    "import handler; print(handler.init_service().handle({'health_check': True}))"
