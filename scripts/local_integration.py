#!/usr/bin/env python3
"""Local integration test: runs the worker with the real model on the GPU and saves the audio files and a manifest.

Usage (from the repo root):  .venv/bin/python scripts/local_integration.py
"""
import base64
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import handler  # noqa: E402
import ptbr_model  # noqa: E402

OUTPUT = ROOT / "output_local"


def main():
    service = handler.init_service()
    voice = base64.b64encode(ptbr_model.download_default_voice().read_bytes()).decode()
    jobs = [
        ("01_neutral", {"text": "O lembrete ficou para amanhã de manhã. Não esqueça a reunião com o pessoal do projeto."}),
        ("02_happy_tag", {"text": "[alegre] Consegui! A entrevista foi ótima e eles querem me ver de novo na sexta."}),
        ("03_voice_base64", {"text": "Fala baixinho, que o bebê acabou de dormir.", "emotion": "whisper",
                             "reference_audio_base64": voice}),
    ]
    OUTPUT.mkdir(exist_ok=True)
    manifest = []
    for name, job_input in jobs:
        result = service.handle(job_input)
        if "error" in result:
            raise SystemExit(f"{name}: error: {result['error']}")
        path = OUTPUT / f"{name}.wav"
        path.write_bytes(base64.b64decode(result["audio_base64"]))
        manifest.append({"file": path.name, "text": result["text"]})
        print(f"{name}: {result['duration_seconds']:.2f} s of audio | "
              f"{json.dumps(result['metrics'], ensure_ascii=False)}", flush=True)
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
