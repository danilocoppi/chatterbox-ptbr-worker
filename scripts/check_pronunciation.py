#!/usr/bin/env python3
"""Transcribes the WAV files in a folder with a local Parakeet model and computes the CER against the manifest.

Usage:  uv run --python 3.11 --with sherpa-onnx --with soundfile python scripts/check_pronunciation.py FOLDER [LIMIT]
Exits with code 1 unless the average CER is below LIMIT (default 0.05).
The model folder comes from PARAKEET_DIR (sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8).
"""
import json
import os
import re
import sys
import unicodedata
from pathlib import Path

import sherpa_onnx
import soundfile as sf

PARAKEET = Path(os.environ.get(
    "PARAKEET_DIR", Path.home() / ".claudinei/speech/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8"))


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text.lower())
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def cer(reference: str, hypothesis: str) -> float:
    ref, hyp = normalize(reference), normalize(hypothesis)
    previous = list(range(len(hyp) + 1))
    for i, a in enumerate(ref, 1):
        current = [i]
        for j, b in enumerate(hyp, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (a != b)))
        previous = current
    return previous[-1] / max(len(ref), 1)


def main():
    folder = Path(sys.argv[1])
    limit = float(sys.argv[2]) if len(sys.argv) > 2 else 0.05
    recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
        encoder=str(PARAKEET / "encoder.int8.onnx"), decoder=str(PARAKEET / "decoder.int8.onnx"),
        joiner=str(PARAKEET / "joiner.int8.onnx"), tokens=str(PARAKEET / "tokens.txt"),
        num_threads=8, model_type="nemo_transducer")
    items = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    errors = []
    for item in items:
        audio, sr = sf.read(folder / item["file"], dtype="float32")
        stream = recognizer.create_stream()
        stream.accept_waveform(sr, audio)
        recognizer.decode_stream(stream)
        transcript = stream.result.text.strip()
        errors.append(cer(item["text"], transcript))
        print(f"{item['file']:24s} CER {errors[-1]:6.1%} | {transcript}")
    average = sum(errors) / len(errors)
    print(f"\nAverage CER: {average:.1%} over {len(errors)} files (limit {limit:.0%})")
    sys.exit(0 if average < limit else 1)


if __name__ == "__main__":
    main()
