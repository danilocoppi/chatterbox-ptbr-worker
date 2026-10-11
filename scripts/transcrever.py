#!/usr/bin/env python3
"""Transcreve os WAVs de uma pasta com o Parakeet local e calcula o CER contra o manifesto.

Uso:  uv run --python 3.11 --with sherpa-onnx --with soundfile python scripts/transcrever.py PASTA [LIMITE]
Sai com código 1 se o CER médio não ficar abaixo do LIMITE (padrão 0.05).
"""
import json
import re
import sys
import unicodedata
from pathlib import Path

import sherpa_onnx
import soundfile as sf

PARAKEET = Path.home() / ".claudinei/speech/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8"


def normalizar(texto: str) -> str:
    texto = unicodedata.normalize("NFC", texto.lower())
    texto = re.sub(r"[^\w\s]", " ", texto)
    return re.sub(r"\s+", " ", texto).strip()


def cer(referencia: str, hipotese: str) -> float:
    ref, hip = normalizar(referencia), normalizar(hipotese)
    anterior = list(range(len(hip) + 1))
    for i, a in enumerate(ref, 1):
        atual = [i]
        for j, b in enumerate(hip, 1):
            atual.append(min(anterior[j] + 1, atual[j - 1] + 1, anterior[j - 1] + (a != b)))
        anterior = atual
    return anterior[-1] / max(len(ref), 1)


def main():
    pasta = Path(sys.argv[1])
    limite = float(sys.argv[2]) if len(sys.argv) > 2 else 0.05
    reconhecedor = sherpa_onnx.OfflineRecognizer.from_transducer(
        encoder=str(PARAKEET / "encoder.int8.onnx"), decoder=str(PARAKEET / "decoder.int8.onnx"),
        joiner=str(PARAKEET / "joiner.int8.onnx"), tokens=str(PARAKEET / "tokens.txt"),
        num_threads=8, model_type="nemo_transducer")
    itens = json.loads((pasta / "manifesto.json").read_text(encoding="utf-8"))
    erros = []
    for item in itens:
        audio, sr = sf.read(pasta / item["arquivo"], dtype="float32")
        fluxo = reconhecedor.create_stream()
        fluxo.accept_waveform(sr, audio)
        reconhecedor.decode_stream(fluxo)
        transcrito = fluxo.result.text.strip()
        erros.append(cer(item["texto"], transcrito))
        print(f"{item['arquivo']:24s} CER {erros[-1]:6.1%} | {transcrito}")
    media = sum(erros) / len(erros)
    print(f"\nCER médio: {media:.1%} em {len(erros)} áudios (limite {limite:.0%})")
    sys.exit(0 if media < limite else 1)


if __name__ == "__main__":
    main()
