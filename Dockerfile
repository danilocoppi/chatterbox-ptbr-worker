FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models/hf \
    CHATTERBOX_MODELOS=/models/chatterbox

RUN apt-get update \
    && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
# Versões exatas do lock (gerado a partir do requirements.txt), as mesmas validadas localmente.
COPY requirements.lock .
RUN pip install --no-deps -r requirements.lock

# Pesos e voz padrão dentro da imagem: o worker liga sem internet.
COPY ptbr.py baixar_pesos.py ./
RUN python baixar_pesos.py

COPY sintese.py handler.py ./
ENV HF_HUB_OFFLINE=1
CMD ["python", "-u", "handler.py"]
