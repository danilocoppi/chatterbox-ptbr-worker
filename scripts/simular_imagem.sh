#!/usr/bin/env bash
# Simula a imagem sem Docker, em duas etapas:
#   1. monta os pesos numa pasta nova, como o `RUN python baixar_pesos.py` do Dockerfile (com internet);
#   2. liga o worker num namespace sem rede nenhuma, como no RunPod, e pede um health check.
# HOME, cache do Hugging Face e PKUSEG_HOME também são novos, para nenhum cache desta máquina mascarar
# um download que a imagem faria ao ligar.
#
# Uso (na raiz):  scripts/simular_imagem.sh [PASTA]    (padrão: /var/tmp/chatterbox-imagem, ~3 GB)
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON=${PYTHON:-$HOME/Projects/61-Claudinei/chatterbox-tts/.venv/bin/python}
PASTA=$(realpath -m "${1:-/var/tmp/chatterbox-imagem}")
mkdir -p "$PASTA/home"

# As mesmas variáveis do Dockerfile, com /models trocado pela pasta.
export HOME="$PASTA/home" HF_HOME="$PASTA/models/hf" CHATTERBOX_MODELOS="$PASTA/models/chatterbox" \
       PKUSEG_HOME="$PASTA/models/pkuseg"

echo "== montagem (com internet)"
env -u LD_LIBRARY_PATH "$PYTHON" baixar_pesos.py

echo "== partida (sem rede)"
env -u LD_LIBRARY_PATH HF_HUB_OFFLINE=1 unshare -rn "$PYTHON" -c \
    "import handler; print(handler.iniciar().atender({'health_check': True}))"
