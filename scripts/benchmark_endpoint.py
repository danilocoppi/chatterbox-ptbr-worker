#!/usr/bin/env python3
"""Benchmarks the RunPod serverless endpoint with the same jobs as the local integration test.

Usage:  python3 scripts/benchmark_endpoint.py ENDPOINT_ID
The API key is read from ~/.config/runpod/api_key. Audio files and the manifest go to output_runpod/ENDPOINT_ID/.
"""
import base64
import json
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
API_KEY = (Path.home() / ".config/runpod/api_key").read_text().strip()
VOICE_URL = ("https://storage.googleapis.com/chatterbox-demo-samples/"
             "mtl-v3-single-language-prompts/pt-br/pt_br_f2.wav")
JOBS = [
    ("01_neutral", {"text": "O lembrete ficou para amanhã de manhã. Não esqueça a reunião com o pessoal do projeto."}),
    ("02_happy_tag", {"text": "[alegre] Consegui! A entrevista foi ótima e eles querem me ver de novo na sexta."}),
    ("03_voice_url", {"text": "Fala baixinho, que o bebê acabou de dormir.", "emotion": "whisper",
                      "reference_audio_url": VOICE_URL}),
]
WARM_REPEATS = 3
FINAL_STATES = ("COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT")


def call(method: str, url: str, body=None) -> dict:
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(url, data=data, method=method, headers={
        "Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read() or b"{}")


def run_job(base: str, job_input: dict, max_queue_s: float):
    start = time.perf_counter()
    job = call("POST", f"{base}/run", {"input": job_input})
    while True:
        time.sleep(1)
        status = call("GET", f"{base}/status/{job['id']}")
        if status.get("status") in FINAL_STATES:
            return status, time.perf_counter() - start
        if status.get("status") == "IN_QUEUE" and time.perf_counter() - start > max_queue_s:
            call("POST", f"{base}/cancel/{job['id']}")
            raise SystemExit(f"No worker picked up the job within {max_queue_s:.0f} s; it was cancelled.")


def main():
    endpoint = sys.argv[1]
    base = f"https://api.runpod.ai/v2/{endpoint}"
    output = ROOT / "output_runpod" / endpoint
    output.mkdir(parents=True, exist_ok=True)
    rounds = list(JOBS) + [(f"9{i}_warm", JOBS[0][1]) for i in range(WARM_REPEATS)]
    manifest = []
    for i, (name, job_input) in enumerate(rounds):
        # The first job may wait while the machine pulls the ~10 GB image.
        status, total = run_job(base, job_input, 900 if i == 0 else 180)
        result = status.get("output") or {}
        if status.get("status") != "COMPLETED" or "error" in result:
            print(f"{name}: {status.get('status')} | {status.get('error') or result.get('error')}", flush=True)
            continue
        path = output / f"{name}.wav"
        path.write_bytes(base64.b64decode(result["audio_base64"]))
        manifest.append({"file": path.name, "text": result["text"]})
        m = result["metrics"]
        print(f"{name}: total {total:.1f} s (queue + start {status.get('delayTime', 0) / 1000:.1f} s, "
              f"execution {status.get('executionTime', 0) / 1000:.1f} s) | generation {m['generation_seconds']:.2f} s, "
              f"first chunk {m['first_chunk_seconds']:.2f} s, RTF {m['rtf']}, {m['gpu']}, cold={m['cold_start']}",
              flush=True)
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
