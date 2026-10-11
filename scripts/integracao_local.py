#!/usr/bin/env python3
"""Integração local: roda o worker com o modelo real na GPU e salva os áudios e o manifesto.

Uso (na raiz):  $PY scripts/integracao_local.py
"""
import base64
import json
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

import handler  # noqa: E402
import ptbr  # noqa: E402

SAIDA = RAIZ / "saida_local"


def main():
    servico = handler.iniciar()
    voz = base64.b64encode(ptbr.baixar_voz_padrao().read_bytes()).decode()
    tarefas = [
        ("01_neutro", {"text": "O lembrete ficou para amanhã de manhã. Não esqueça a reunião com o pessoal do projeto."}),
        ("02_alegre", {"text": "[alegre] Consegui! A entrevista foi ótima e eles querem me ver de novo na sexta."}),
        ("03_voz_base64", {"text": "Fala baixinho, que o bebê acabou de dormir.", "emotion": "sussurro",
                           "reference_audio_base64": voz}),
    ]
    SAIDA.mkdir(exist_ok=True)
    manifesto = []
    for nome, entrada in tarefas:
        resposta = servico.atender(entrada)
        if "error" in resposta:
            raise SystemExit(f"{nome}: erro: {resposta['error']}")
        arquivo = SAIDA / f"{nome}.wav"
        arquivo.write_bytes(base64.b64decode(resposta["audio_base64"]))
        manifesto.append({"arquivo": arquivo.name, "texto": resposta["text"]})
        print(f"{nome}: {resposta['duration_seconds']:.2f} s de áudio | "
              f"{json.dumps(resposta['metrics'], ensure_ascii=False)}", flush=True)
    (SAIDA / "manifesto.json").write_text(json.dumps(manifesto, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
