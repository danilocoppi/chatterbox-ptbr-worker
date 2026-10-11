# chatterbox-ptbr-worker

RunPod Serverless (queue) worker that turns **Brazilian Portuguese text into speech with emotion control**. It runs Resemble AI's Chatterbox Multilingual V3 with the official pt-BR language pack. The model weights, the tokenizer data and a default voice are baked into the image, so a worker starts without downloading anything.

- [Features](#features)
- [Quick start](#quick-start)
- [API reference](#api-reference)
- [Performance and cost](#performance-and-cost)
- [Deploying on RunPod](#deploying-on-runpod)
- [How it works](#how-it-works)
- [Local development](#local-development)
- [Limitations](#limitations)
- [Troubleshooting](#troubleshooting)
- [Credits and licenses](#credits-and-licenses)

## Features

- **Brazilian Portuguese only,** with the official pt-BR pack [`ResembleAI/Chatterbox-Multilingual-pt-br`](https://huggingface.co/ResembleAI/Chatterbox-Multilingual-pt-br). Measured pronunciation error is 0–1.6% character error rate, checked with the Parakeet speech recognizer.
- **Seven emotion presets.** Names work in English or Portuguese. Fine control comes from `exaggeration`, `cfg_weight`, `temperature` and `seed`.
- **Inline emotion tag** at the start of the text, such as `[alegre] Consegui!`.
- **Voice cloning** from a reference clip, sent by URL or as base64. WAV, MP3, FLAC and OGG are accepted.
- **Long text** of up to 1,000 characters per job, split into sentences automatically.
- **Per-job metrics:** generation time, time to the first sentence, real-time factor, GPU and cold start.
- **Offline start with pinned dependencies.** The image installs exact versions from a lock file and never downloads at startup.
- **Safe failures.** Invalid input gets a clear error message, and an unexpected error never kills the worker.
- **Watermark.** Every output carries Resemble's imperceptible [Perth](https://github.com/resemble-ai/Perth) watermark.

## Quick start

```bash
export RUNPOD_API_KEY="..."   # RunPod console → Settings → API Keys
export ENDPOINT_ID="..."      # RunPod console → Serverless → your endpoint
```

**Synchronous call** that saves a WAV file (needs `jq`):

```bash
curl -s -X POST "https://api.runpod.ai/v2/$ENDPOINT_ID/runsync" \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"input": {"text": "[alegre] Consegui! A entrevista foi ótima."}}' \
  | jq -r '.output.audio_base64' | base64 -d > speech.wav
```

`/runsync` waits up to 90 s by default, and you can raise this up to 300 s with `?wait=<ms>`. The first job after the endpoint scales from zero can take longer, because the worker pulls a ~10 GB image and loads the model. For that first call, and for long texts, use `/run` and poll `/status`:

```bash
JOB_ID=$(curl -s -X POST "https://api.runpod.ai/v2/$ENDPOINT_ID/run" \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"input": {"text": "Bom dia! Seu pedido saiu para entrega.", "emotion": "calm"}}' | jq -r .id)

curl -s "https://api.runpod.ai/v2/$ENDPOINT_ID/status/$JOB_ID" \
  -H "Authorization: Bearer $RUNPOD_API_KEY" | jq '{status, error, metrics: .output.metrics}'
```

Results of `/run` stay available on `/status` for 30 minutes after the job ends. Results of `/runsync` stay available for 1 minute.

**Python** (`requests`):

```python
import base64
import os
import time

import requests

URL = f"https://api.runpod.ai/v2/{os.environ['ENDPOINT_ID']}"
HEADERS = {"Authorization": f"Bearer {os.environ['RUNPOD_API_KEY']}"}


def speak(job_input: dict, path: str) -> dict:
    job = requests.post(f"{URL}/run", headers=HEADERS, json={"input": job_input}, timeout=30).json()
    while True:
        status = requests.get(f"{URL}/status/{job['id']}", headers=HEADERS, timeout=30).json()
        if status["status"] in ("COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"):
            break
        time.sleep(1)
    if status["status"] != "COMPLETED":
        raise RuntimeError(f"{status['status']}: {status.get('error')}")
    with open(path, "wb") as f:
        f.write(base64.b64decode(status["output"]["audio_base64"]))
    return status["output"]["metrics"]


print(speak({"text": "Que bom te ver por aqui!", "emotion": "happy"}, "speech.wav"))
```

## API reference

### Request

Send the fields inside `input`: `{"input": {...}}`.

| Field | Type | Default | Accepted values | Description |
|---|---|---|---|---|
| `text` | string | required | 1–1,000 characters after removing the tag | Brazilian Portuguese text to speak |
| `emotion` | string | `neutral` | see [Emotions](#emotions) | Preset that sets `exaggeration` and `cfg_weight` |
| `exaggeration` | number | from the emotion | 0.25–2.0 | Emotional intensity; overrides the preset |
| `cfg_weight` | number | from the emotion | 0.0–1.0 | Guidance weight; lower values give slower, more deliberate pacing. Overrides the preset |
| `temperature` | number | 0.8 | 0.05–2.0 | Sampling randomness |
| `seed` | integer | random | 0 to 2⁶⁴−1 | Reproducible output on the same GPU type; applied before every sentence |
| `reference_audio_url` | string | none | public http(s) URL | Voice to clone; see [Reference voice](#reference-voice) |
| `reference_audio_base64` | string | none | base64, ≤ 10 MB decoded | Voice to clone, inline; ignored if a URL is also sent |
| `health_check` | boolean | `false` | | Returns the worker status without generating audio |
| `speed` | number | | only `1.0` | Not supported; any other value is an error |

Fields not listed here are ignored.

### Emotions

| Emotion | Portuguese name | `exaggeration` | `cfg_weight` | Typical use |
|---|---|---|---|---|
| `neutral` | `neutro` | 0.5 | 0.5 | Default narration |
| `happy` | `alegre` | 0.8 | 0.3 | Good news, enthusiasm |
| `excited` | `animado` | 0.9 | 0.3 | High energy |
| `sad` | `triste` | 0.35 | 0.5 | Bad news, sympathy |
| `empathetic` | `empático` or `empatico` | 0.4 | 0.5 | Support, care |
| `calm` | `calmo` | 0.35 | 0.5 | Meditation, instructions |
| `whisper` | `sussurro` | 0.3 | 0.5 | Soft, low-intensity speech |

Names ignore letter case and accents, and the response always returns the English name.

Chatterbox has no discrete emotion classes. A preset sets the intensity and the pace, while the emotion you actually hear comes mostly from the reference voice and from the text itself. For a clearly different emotion, such as a real whisper, use a reference clip recorded with that emotion.

### Emotion tag

A tag at the very start of the text sets the emotion: `[triste] Que pena que não deu certo.`

- Tags contain letters and underscores only (`[A-Za-zÀ-ÿ_]+`). So `[1] Primeiro item` is spoken as written.
- The tag is always removed from the spoken text.
- When `emotion` is also sent, the field wins, and the tag is still removed.
- An unknown tag such as `[raiva]` is an error.
- Tags in the middle of the text are not interpreted; they are spoken as text.

### Reference voice

Without a reference, the worker uses the default voice `pt_br_f2`, the female pt-BR voice from Resemble's official demo.

- **Precedence:** if both `reference_audio_url` and `reference_audio_base64` are sent, the URL wins.
- **Formats:** WAV, MP3, FLAC and OGG.
- **URL rules:**
  - it must be `http(s)`, download within 15 s and be at most 10 MB;
  - it must resolve to a public address. Internal addresses, such as `localhost`, private networks or the cloud metadata service, are refused, also after redirects;
  - it must point to the audio file itself, not to a sharing page.
- **Base64 rules:**
  - at most 10 MB once decoded;
  - a `data:audio/...;base64,` prefix and line breaks are accepted.

  RunPod caps the whole request at 10 MB on `/run` and 20 MB on `/runsync`. Base64 adds a third to the size, so through `/run` you can send about 7 MB of audio. Use a URL for bigger files.
- **How much is used:** the model takes the first 10 s for the voice timbre and the first 6 s as the speech prompt. A clean 10 s clip is the sweet spot.

Tips for a good reference clip:
- 5–15 s of continuous speech from a single speaker;
- mono, little background noise, no music;
- a natural pace;
- the same emotion you want to hear.

Only clone voices with the speaker's consent.

### Response

The job's `output`:

```json
{
  "audio_base64": "UklGRi...",
  "sample_rate": 24000,
  "duration_seconds": 6.67,
  "text": "O lembrete ficou para amanhã de manhã. Não esqueça a reunião com o pessoal do projeto.",
  "emotion": "neutral",
  "metrics": {
    "generation_seconds": 3.85,
    "first_sentence_seconds": 1.756,
    "rtf": 0.577,
    "sentences": 2,
    "gpu": "NVIDIA GeForce RTX 3060 Ti",
    "cold_start": true,
    "model_load_seconds": 7.27
  }
}
```

The audio is a 16-bit PCM mono WAV at 24 kHz. `text` is the spoken text, with the tag removed, and `emotion` is the canonical English name.

| Metric | Meaning |
|---|---|
| `generation_seconds` | Synthesis time for all sentences; excludes preparing the reference voice and encoding the WAV |
| `first_sentence_seconds` | Synthesis time of the first sentence; your time to first audio if you play sentence by sentence |
| `rtf` | Real-time factor, `generation_seconds / duration_seconds`; below 1 means faster than real time |
| `sentences` | Number of pieces the text was split into |
| `gpu` | GPU that served the job |
| `cold_start` | `true` on the first job a worker serves after starting |
| `model_load_seconds` | Model load time when the worker started |

RunPod wraps this in its job envelope: `{"id", "status": "COMPLETED", "delayTime", "executionTime", "output": {...}}`. Both times are in milliseconds. `delayTime` covers the time in the queue plus any worker start.

### Health check

```json
{"input": {"health_check": true}}
```

The response is `{"status": "healthy", "model_loaded": true, "gpu": "NVIDIA GeForce RTX 3090"}`. This differs from RunPod's own `GET /v2/{endpoint_id}/health`, which reports queue and worker counts.

### Errors

On invalid input or a failed generation, the job ends as `FAILED` and the message is in `error`:

```json
{"id": "...", "status": "FAILED", "error": "Unknown emotion: 'raiva'. Accepted: neutral, happy, excited, sad, empathetic, calm, whisper (or in Portuguese: neutro, alegre, animado, triste, empatico, calmo, sussurro)."}
```

| Cause | Message |
|---|---|
| `input` is not an object | `The input must be a JSON object.` |
| Missing or blank `text` | `Provide the text to speak in 'text'.` |
| Only a tag, no text | `There is no text to speak after the emotion tag.` |
| More than 1,000 characters | `The text has N characters; the maximum is 1000.` |
| Unknown emotion or tag | `Unknown emotion: '...'. Accepted: ...` |
| Wrong type | `'exaggeration' must be a number.`, `'emotion' must be a string.`, `'seed' must be an integer greater than or equal to zero.` |
| Out of range | `'cfg_weight' must be between 0.0 and 1.0.` (one message per field) |
| `speed` other than 1.0 | `'speed' is not supported; send 1.0 or omit the field.` |
| Bad URL | `'reference_audio_url' must be an http(s) URL.` or `... points to an internal address, which is not allowed.` |
| Download problem | `Could not download the reference voice: ...`, `Downloading the reference voice took more than 15 s.`, `The reference voice exceeds 10 MB.` |
| Bad base64 | `'reference_audio_base64' is not valid base64.`, `... must be a base64 string.`, `The reference voice is empty.` |
| The file is not audio | `Could not read the reference voice: ...` |
| Anything else | `Unexpected generation failure: ...` (the traceback goes to the worker logs, and the worker keeps serving) |

### More examples

Emotion field with a Portuguese name, and a fixed seed:

```json
{"input": {"text": "Sinto muito pela sua perda. Estou aqui se precisar.", "emotion": "empático", "seed": 42}}
```

Overriding the preset:

```json
{"input": {"text": "Atenção: o embarque foi antecipado!", "emotion": "excited", "exaggeration": 1.2, "cfg_weight": 0.2, "temperature": 0.7}}
```

Voice cloning by URL:

```json
{"input": {"text": "Oi, essa é a minha voz clonada.", "reference_audio_url": "https://example.com/minha-voz.wav"}}
```

Voice cloning with base64. This builds the payload from a local file. Use `--rawfile`, because the base64 of a real clip is longer than the shell allows for a single argument:

```bash
base64 -w0 minha-voz.wav > voice.b64
jq -n --rawfile voice voice.b64 \
  '{input: {text: "Oi, essa é a minha voz clonada.", reference_audio_base64: $voice}}' > request.json
curl -s -X POST "https://api.runpod.ai/v2/$ENDPOINT_ID/run" \
  -H "Authorization: Bearer $RUNPOD_API_KEY" -H "Content-Type: application/json" -d @request.json
```

## Performance and cost

These were measured on a local RTX 3060 Ti (8 GB) with the same code and package versions as the image:

| Measure | Value |
|---|---|
| Model load (weights on local disk) | 6–7 s |
| GPU memory | ~3.0 GiB of weights, ~3.5 GiB peak |
| Real-time factor | 0.5–0.58, so speech is generated about 2× faster than it plays |
| A ~4 s sentence | ~2.2 s |
| A short phrase (2 s of audio) | ~1.4 s |
| First sentence of a job | 1.0–1.8 s |
| Pronunciation (Parakeet CER) | 0–1.6% |

Cloud figures come from `scripts/benchmark_endpoint.py` and will be added after the first deployment.

**Cost estimate.** These use RunPod flex prices on 2026-10-11: the 24 GB tier (L4, A5000, RTX 3090) costs $0.00019/s, and the 48 GB tier (A40, A6000) costs $0.00034/s. They also assume the RTF measured on the 3060 Ti. Check [runpod.io/pricing](https://www.runpod.io/pricing) for current prices.

| Speech length | Generation time (RTF 0.5) | 24 GB tier | 48 GB tier |
|---|---|---|---|
| 5 s | ~2.5 s | ~$0.0005 | ~$0.0009 |
| 30 s | ~15 s | ~$0.003 | ~$0.005 |
| 60 s | ~30 s | ~$0.006 | ~$0.010 |

Billing is per second while a worker runs, not only while it executes jobs. A request that scales the endpoint from zero also pays for the worker start: the image start plus the model load. After the last job, the worker also bills the idle timeout, which is 5 s by default.

## Deploying on RunPod

1. **Connect GitHub.** In **Settings → Connections**, connect GitHub and grant access to this repository. A RunPod account can link only one GitHub account.
2. **Create the endpoint.** In **Serverless → New Endpoint → GitHub Repo**, pick this repository, the `main` branch and the `Dockerfile` at the root.
3. **GPU.** The model needs ~3.5 GB, so any GPU with 8 GB or more works. The 24 GB tier (L4, A5000, RTX 3090) is the cheapest option; use the 48 GB tier (A40, A6000) only for comparisons.
4. **Workers and timeouts:**
   - for testing, set active (minimum) workers to 0 and max workers to 1;
   - the default idle timeout of 5 s and execution timeout of 600 s are fine, since 1,000 characters take well under a minute.
5. **Container and CUDA.** Use a 20 GB container disk, and allow CUDA 12.4 or newer. The image uses PyTorch 2.6 with CUDA 12.4.
6. **Environment variables.** None are required. The image already sets `HF_HOME`, `CHATTERBOX_MODELS`, `PKUSEG_HOME` and `HF_HUB_OFFLINE=1`.
7. **Build.** Wait for the build to finish. RunPod allows 30 minutes for the `docker build` step and 160 minutes for the whole build. The image is about 10 GB, well under the 80 GB limit.
8. **Updates.** RunPod builds a new version when you **create a GitHub release**. Pushing commits alone does not redeploy:

   ```bash
   gh release create v1.0.1 --generate-notes
   ```
9. **Smoke test.** Save an API key with access to the endpoint in `~/.config/runpod/api_key`, then run:

   ```bash
   python3 scripts/benchmark_endpoint.py "$ENDPOINT_ID"   # 3 jobs + 3 warm repeats → output_runpod/<id>/
   uv run --python 3.11 --with sherpa-onnx --with soundfile python scripts/check_pronunciation.py "output_runpod/$ENDPOINT_ID"
   ```

## How it works

**At worker start (once):**
1. `ptbr_model.load()` loads the model from `/models` with no network access. The pt-BR decoder (`s3gen_v3.pt`) is loaded with `strict=False`, and the loader checks that only its two known fixed buffers are missing.
2. The default voice's conditionals are prepared and kept.
3. A short warm-up generation runs, so the first job doesn't pay for CUDA initialization.
4. If any step fails, the process exits and RunPod replaces the worker.

**For each job:**
1. A `health_check` returns immediately.
2. The input is validated, and the emotion and parameters are resolved (`synthesis.parse_input`).
3. The voice is resolved:
   - the default voice is restored first;
   - a custom voice is downloaded or decoded to a temporary file, turned into conditionals, and the file is deleted.
4. The text is split after `.`, `!`, `?` or `…`. Pieces longer than 300 characters are split at `,` or `;`, then between words.
5. Each sentence is generated with the same parameters, and the seed is applied before each one.
6. The sentences are joined with 150 ms of silence, encoded as WAV and returned with the metrics.

Each worker serves one job at a time, which is the RunPod default.

### Project structure

```
.
├── handler.py             RunPod entry point; Service: validation → voice → generation → metrics
├── synthesis.py           GPU-free logic: input, emotions, sentence splitting, WAV, reference voice
├── ptbr_model.py          loads Chatterbox V3 + the pt-BR pack: checkpoint, tokenizer, default voice
├── download_weights.py    image build step: weights, tokenizer data and default voice
├── Dockerfile             python:3.11-slim, pinned packages, baked-in weights
├── requirements.txt       direct dependencies (edit this one)
├── requirements.lock      exact versions installed in the image (generated)
├── tests/                 unit tests, no GPU needed
├── scripts/
│   ├── local_integration.py    real model on the local GPU → output_local/
│   ├── check_pronunciation.py  transcribes with Parakeet and checks the CER
│   ├── simulate_image.sh       builds into a fresh folder and starts with no network
│   └── benchmark_endpoint.py   runs the same jobs on the RunPod endpoint
└── docs/superpowers/      design spec and implementation plan (in Portuguese)
```

## Local development

You need:
- Linux;
- Python 3.11 and [uv](https://docs.astral.sh/uv/);
- an NVIDIA GPU with 8 GB or more, for the model. The unit tests need no GPU.

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python --no-deps -r requirements.lock
uv pip install --python .venv/bin/python pytest

.venv/bin/python -m pytest tests -q              # unit tests, no GPU
.venv/bin/python download_weights.py             # weights and default voice into ./models
.venv/bin/python scripts/local_integration.py    # real model on the GPU → output_local/
scripts/simulate_image.sh                        # the image without Docker: fresh folder, no network (~3 GB)
```

**Through the RunPod SDK, locally.** You can run a single job, or start a local API on `http://localhost:8000`. On that API, use `/runsync`, because the local `/run` only returns a placeholder id.

```bash
.venv/bin/python handler.py --test_input '{"input": {"text": "[alegre] Teste local do worker."}}'
.venv/bin/python handler.py --rp_serve_api
```

**Pronunciation check.** This needs the sherpa-onnx Parakeet model:

```bash
mkdir -p ~/models
curl -L https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8.tar.bz2 | tar -xj -C ~/models
export PARAKEET_DIR=~/models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8
uv run --python 3.11 --with sherpa-onnx --with soundfile python scripts/check_pronunciation.py output_local
```

The script exits with code 1 unless the average CER is below 5%. You can pass another limit as a second argument.

**Updating dependencies.** Edit `requirements.txt`, then regenerate the lock, which is what the image installs:

```bash
uv pip compile requirements.txt --python-version 3.11 -o requirements.lock
```

`runpod` is pinned to 1.10.0 because of a dependency conflict:
- Chatterbox at commit `5de7a54` requires `gradio==6.8.0`, which needs `tomlkit<0.14`;
- `runpod` 1.10.1 and later need `tomlkit>=0.15`.

## Limitations

- **Language:** Brazilian Portuguese only.
- **No streaming:** the response arrives once the whole text is synthesized. For lower latency, send short sentences as separate jobs. On a 3060 Ti, the first sentence takes 1–2 s.
- **No speed control.**
- **Text size:** at most 1,000 characters per job.
- **Sentence splitting** relies on punctuation, so an abbreviation such as "Sr." starts a new piece and adds a short pause.
- **`seed`** is reproducible only on the same GPU type and software versions.
- **Emotions:** presets control intensity and pace. A clearly different emotion needs a reference clip recorded that way.
- **Watermark:** every output carries the Perth watermark.
- **Default voice:** `pt_br_f2` is Resemble's demo voice and is meant for tests. For real use, record your own voice or use one with documented consent.

## Troubleshooting

| Symptom | Likely cause | What to do |
|---|---|---|
| The job stays `IN_QUEUE` | The build isn't finished, no GPU of the chosen type is free, or a cold start is pulling the ~10 GB image | Check the build and worker status in the console; allow more GPU types; wait for the first pull |
| `/runsync` returns `IN_QUEUE` or `IN_PROGRESS` | The job took longer than the sync wait (90 s by default) | Poll `GET /status/{id}`, or use `/run` |
| `Could not read the reference voice: ...` | The URL returned a web page (a Drive or Dropbox share page, say) or a format that isn't audio | Use a direct download link to a WAV, MP3, FLAC or OGG file |
| `... points to an internal address ...` | The URL resolves to a private or local address | Host the clip on a public URL, or send it as base64 |
| Code changes don't show up on RunPod | Commits don't trigger builds | Create a GitHub release |
| The build fails or times out | The `docker build` step has a 30-minute limit | Retry the build; check the build logs for download errors |
| Locally: `GLIBCXX_... not found` | `LD_LIBRARY_PATH` points to an older libstdc++ | Run Python with `env -u LD_LIBRARY_PATH ...` |

## Credits and licenses

- **Model:**
  - [Chatterbox](https://github.com/resemble-ai/chatterbox) by Resemble AI, MIT license, pinned at commit `5de7a54`;
  - the pt-BR language pack [`ResembleAI/Chatterbox-Multilingual-pt-br`](https://huggingface.co/ResembleAI/Chatterbox-Multilingual-pt-br), MIT license.
- **Watermark:** [Perth](https://github.com/resemble-ai/Perth) by Resemble AI, MIT license.
- **API compatibility:** the field names follow [profzeller/chatterbox-wellness-worker](https://github.com/profzeller/chatterbox-wellness-worker), so most requests work unchanged. The exceptions are `speed` values other than 1.0 and the `angry` and `surprised` emotions, which this worker doesn't have. All code here was written from scratch.
- **This repository:** no license has been granted yet.
