#!/usr/bin/env python3
"""Testa o endpoint serverless no RunPod com as mesmas tarefas da integração local.

Uso:  python3 scripts/testar_endpoint.py ENDPOINT_ID
A chave vem de ~/.config/runpod/api_key. Grava os áudios e o manifesto em saida_runpod/ENDPOINT_ID/.
"""
import base64
import json
import sys
import time
import urllib.request
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
CHAVE = (Path.home() / ".config/runpod/api_key").read_text().strip()
VOZ_URL = ("https://storage.googleapis.com/chatterbox-demo-samples/"
           "mtl-v3-single-language-prompts/pt-br/pt_br_f2.wav")
TAREFAS = [
    ("01_neutro", {"text": "O lembrete ficou para amanhã de manhã. Não esqueça a reunião com o pessoal do projeto."}),
    ("02_alegre", {"text": "[alegre] Consegui! A entrevista foi ótima e eles querem me ver de novo na sexta."}),
    ("03_voz_url", {"text": "Fala baixinho, que o bebê acabou de dormir.", "emotion": "sussurro",
                    "reference_audio_url": VOZ_URL}),
]
REPETICOES_QUENTES = 3
FINAIS = ("COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT")


def chamar(metodo: str, url: str, corpo=None) -> dict:
    dados = json.dumps(corpo).encode() if corpo is not None else None
    requisicao = urllib.request.Request(url, data=dados, method=metodo, headers={
        "Authorization": f"Bearer {CHAVE}", "Content-Type": "application/json"})
    with urllib.request.urlopen(requisicao, timeout=60) as resposta:
        return json.loads(resposta.read() or b"{}")


def executar(base: str, entrada: dict, espera_fila_s: float):
    inicio = time.perf_counter()
    job = chamar("POST", f"{base}/run", {"input": entrada})
    while True:
        time.sleep(1)
        status = chamar("GET", f"{base}/status/{job['id']}")
        if status.get("status") in FINAIS:
            return status, time.perf_counter() - inicio
        if status.get("status") == "IN_QUEUE" and time.perf_counter() - inicio > espera_fila_s:
            chamar("POST", f"{base}/cancel/{job['id']}")
            raise SystemExit(f"Ninguém pegou a tarefa em {espera_fila_s:.0f} s; ela foi cancelada.")


def main():
    endpoint = sys.argv[1]
    base = f"https://api.runpod.ai/v2/{endpoint}"
    saida = RAIZ / "saida_runpod" / endpoint
    saida.mkdir(parents=True, exist_ok=True)
    rodadas = list(TAREFAS) + [(f"9{i}_quente", TAREFAS[0][1]) for i in range(REPETICOES_QUENTES)]
    manifesto = []
    for i, (nome, entrada) in enumerate(rodadas):
        # A primeira tarefa pode esperar a máquina baixar a imagem de ~10 GB.
        status, total = executar(base, entrada, 900 if i == 0 else 180)
        saida_job = status.get("output") or {}
        if status.get("status") != "COMPLETED" or "error" in saida_job:
            print(f"{nome}: {status.get('status')} | {status.get('error') or saida_job.get('error')}", flush=True)
            continue
        arquivo = saida / f"{nome}.wav"
        arquivo.write_bytes(base64.b64decode(saida_job["audio_base64"]))
        manifesto.append({"arquivo": arquivo.name, "texto": saida_job["text"]})
        m = saida_job["metrics"]
        print(f"{nome}: total {total:.1f} s (fila e arranque {status.get('delayTime', 0) / 1000:.1f} s, "
              f"execução {status.get('executionTime', 0) / 1000:.1f} s) | geração {m['generation_seconds']:.2f} s, "
              f"1ª frase {m['first_sentence_seconds']:.2f} s, RTF {m['rtf']}, {m['gpu']}, frio={m['cold_start']}",
              flush=True)
    (saida / "manifesto.json").write_text(json.dumps(manifesto, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
