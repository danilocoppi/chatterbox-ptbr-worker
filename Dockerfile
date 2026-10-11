FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models/hf \
    CHATTERBOX_MODELS=/models/chatterbox \
    PKUSEG_HOME=/models/pkuseg

RUN apt-get update \
    && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
# Exact versions from the lock (generated from requirements.txt), the same set validated locally.
COPY requirements.lock .
RUN pip install --no-deps -r requirements.lock

# Weights, tokenizer data and default voice baked into the image: the worker starts without internet.
COPY ptbr_model.py download_weights.py ./
RUN python download_weights.py

COPY synthesis.py handler.py ./
ENV HF_HUB_OFFLINE=1
CMD ["python", "-u", "handler.py"]
