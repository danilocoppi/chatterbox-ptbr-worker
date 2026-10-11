# chatterbox-ptbr-worker

RunPod Serverless (queue) worker that turns **Brazilian Portuguese text into speech with emotion control**. It runs Resemble AI's Chatterbox Multilingual V3 with the official pt-BR language pack. The model weights, the tokenizer data and a default voice are baked into the image, so a worker starts without downloading anything.

- [Capabilities](#capabilities)
- [Deployment on RunPod](#deployment-on-runpod)
- [API Usage](#api-usage)
- [Response Format](#response-format)
- [Cost Estimates](#cost-estimates)
- [Reference Audio Requirements](#reference-audio-requirements)
- [Examples by use case](#examples-by-use-case)
- [How it works](#how-it-works)
- [Local Development](#local-development)
- [Limitations](#limitations)
- [Troubleshooting](#troubleshooting)
- [Credits and License](#credits-and-license)

## Capabilities

- **Text-to-Speech in Brazilian Portuguese.** Uses the official pt-BR pack [`ResembleAI/Chatterbox-Multilingual-pt-br`](https://huggingface.co/ResembleAI/Chatterbox-Multilingual-pt-br). Measured pronunciation error is 0–1.6% character error rate, checked with the Parakeet speech recognizer.
- **Voice Cloning.** Clones any voice from a reference clip sent by URL or as base64. WAV, MP3, FLAC and OGG are accepted.
- **Emotion Control:**
  - seven presets, named in English or Portuguese;
  - an inline tag such as `[alegre] Consegui!`;
  - fine control with `exaggeration`, `cfg_weight`, `temperature` and `seed`.
- **Long Text.** Accepts up to 1,000 characters per job and splits them into sentences automatically.
- **Metrics in Every Response:** generation time, time to the first sentence, real-time factor, GPU and cold start.
- **Offline Start and Pinned Dependencies.** The image installs exact versions from a lock file and never downloads at startup.
- **Safe Failures.** Invalid input gets a clear error message, and an unexpected error never kills the worker.
- **Watermark.** Every output carries Resemble's imperceptible [Perth](https://github.com/resemble-ai/Perth) watermark.

## Deployment on RunPod

### 1. Connect this repo to RunPod

In **Settings → Connections**, connect GitHub and grant access to this repository. A RunPod account can link only one GitHub account.

### 2. Create the Serverless Endpoint

1. Go to **Serverless → New Endpoint → GitHub Repo**.
2. Pick this repository, the `main` branch and the `Dockerfile` at the root.
3. **GPU:** the model needs ~3.5 GB, so any GPU with 8 GB or more works. The 24 GB tier (L4, A5000, RTX 3090) is the cheapest option; use the 48 GB tier (A40, A6000) only for comparisons.
4. **Workers:** for testing, set active (minimum) workers to 0 and max workers to 1.
5. **Timeouts:** keep the default idle timeout of 5 s and execution timeout of 600 s. A job of 1,000 characters takes well under a minute.
6. **Container disk:** 20 GB.
7. **Allowed CUDA versions:** 12.4 or newer. The image uses PyTorch 2.6 with CUDA 12.4.
8. **No network volume needed.** The model (~3 GB) is included in the ~10 GB image.

The build takes a while. RunPod allows 30 minutes for the `docker build` step and 160 minutes for the whole build. Images can be up to 80 GB.

### 3. Environment Variables

None are required, because everything is bundled. The image already sets:

| Variable | Value | Purpose |
|---|---|---|
| `HF_HOME` | `/models/hf` | Hugging Face cache holding the weights |
| `CHATTERBOX_MODELS` | `/models/chatterbox` | Assembled checkpoint and default voice |
| `PKUSEG_HOME` | `/models/pkuseg` | Data the tokenizer would otherwise download at startup |
| `HF_HUB_OFFLINE` | `1` | Guarantees the worker never downloads at startup |

### 4. Updating

RunPod builds a new version when you **create a GitHub release**. Pushing commits alone does not redeploy:

```bash
gh release create v1.0.1 --generate-notes
```

### 5. Smoke test

Save an API key with access to the endpoint in `~/.config/runpod/api_key`, then run:

```bash
python3 scripts/benchmark_endpoint.py "$ENDPOINT_ID"   # 3 jobs + 3 warm repeats → output_runpod/<id>/
uv run --python 3.11 --with sherpa-onnx --with soundfile python scripts/check_pronunciation.py "output_runpod/$ENDPOINT_ID"
```

## API Usage

Every request is a JSON object with the fields inside `input`. Send it to the endpoint at `https://api.runpod.ai/v2/<ENDPOINT_ID>`, with the header `Authorization: Bearer <RUNPOD_API_KEY>`.

| Operation | Method and path | Notes |
|---|---|---|
| Run and wait | `POST /runsync` | Waits up to 90 s by default, up to 300 s with `?wait=<ms>`. Request up to 20 MB; the result is kept for 1 minute |
| Run in the background | `POST /run` | Returns a job id at once. Request up to 10 MB; the result is kept for 30 minutes |
| Job status and result | `GET /status/<job_id>` | States: `IN_QUEUE`, `IN_PROGRESS`, `COMPLETED`, `FAILED`, `CANCELLED`, `TIMED_OUT` |
| Cancel a job | `POST /cancel/<job_id>` | |
| Queue and workers | `GET /health` | Job and worker counts |

### Calling the endpoint

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

The first job after the endpoint scales from zero can take longer than the sync wait, because the worker pulls a ~10 GB image and loads the model. For that first call, and for long texts, use `/run` and poll `/status`:

```bash
JOB_ID=$(curl -s -X POST "https://api.runpod.ai/v2/$ENDPOINT_ID/run" \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"input": {"text": "Bom dia! Seu pedido saiu para entrega.", "emotion": "calm"}}' | jq -r .id)

curl -s "https://api.runpod.ai/v2/$ENDPOINT_ID/status/$JOB_ID" \
  -H "Authorization: Bearer $RUNPOD_API_KEY" | jq '{status, error, metrics: .output.metrics}'
```

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

### Basic TTS (Default Voice)

```json
{
  "input": {
    "text": "Olá! Seja bem-vindo. Como posso ajudar você hoje?"
  }
}
```

Without a reference voice, the worker uses `pt_br_f2`, the female pt-BR voice from Resemble's official demo, with the `neutral` emotion.

### Voice Cloning

```json
{
  "input": {
    "text": "Oi, essa é a minha voz clonada falando.",
    "reference_audio_url": "https://example.com/minha-voz.wav"
  }
}
```

Or with base64 audio:

```json
{
  "input": {
    "text": "Oi, essa é a minha voz clonada falando.",
    "reference_audio_base64": "UklGRi..."
  }
}
```

To build the base64 request from a local file, use `--rawfile`. The base64 of a real clip is longer than the shell allows for a single argument:

```bash
base64 -w0 minha-voz.wav > voice.b64
jq -n --rawfile voice voice.b64 \
  '{input: {text: "Oi, essa é a minha voz clonada falando.", reference_audio_base64: $voice}}' > request.json
curl -s -X POST "https://api.runpod.ai/v2/$ENDPOINT_ID/run" \
  -H "Authorization: Bearer $RUNPOD_API_KEY" -H "Content-Type: application/json" -d @request.json
```

Some rules apply:
- If both fields are sent, the URL wins.
- A `data:audio/...;base64,` prefix and line breaks are accepted.
- The voice is used for that job only, and the next job goes back to the default voice.

See [Reference Audio Requirements](#reference-audio-requirements) for formats, limits and recording tips.

### With Emotion Tags

```json
{
  "input": {
    "text": "[alegre] Consegui! A entrevista foi ótima e eles querem me ver de novo na sexta."
  }
}
```

Supported emotions are `neutral`, `happy`, `excited`, `sad`, `empathetic`, `calm` and `whisper`. Their Portuguese names also work: `neutro`, `alegre`, `animado`, `triste`, `empático`, `calmo` and `sussurro`.

How tags work:
- Only a tag at the very start of the text is read. A tag in the middle is spoken as text.
- Tags contain letters and underscores only (`[A-Za-zÀ-ÿ_]+`), so `[1] Primeiro item` is spoken as written.
- The tag is always removed from the spoken text.
- When the `emotion` field is also sent, the field wins.
- An unknown tag such as `[raiva]` is an error.

### With an Emotion Field

```json
{
  "input": {
    "text": "Infelizmente o voo foi cancelado. Vamos remarcar para amanhã cedo.",
    "emotion": "sad"
  }
}
```

Each emotion is a preset for two parameters:

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

### With Parameters

```json
{
  "input": {
    "text": "Atenção: o embarque foi antecipado para as nove horas!",
    "emotion": "excited",
    "exaggeration": 1.2,
    "cfg_weight": 0.2,
    "temperature": 0.7,
    "seed": 42
  }
}
```

| Parameter | Default | Range | Description |
|---|---|---|---|
| `exaggeration` | from the emotion (0.5 for `neutral`) | 0.25–2.0 | Emotional intensity; overrides the preset |
| `cfg_weight` | from the emotion (0.5 for `neutral`) | 0.0–1.0 | Guidance weight; lower values give slower, more deliberate pacing. Overrides the preset |
| `temperature` | 0.8 | 0.05–2.0 | Sampling randomness; lower values are steadier, higher values more varied |
| `seed` | random | 0 to 2⁶⁴−1 | Reproducible output on the same GPU type; applied before every sentence |
| `speed` | 1.0 | only 1.0 | Not supported; any other value is an error |

All input fields:

| Field | Type | Required | Description |
|---|---|---|---|
| `text` | string | yes | Brazilian Portuguese text, 1–1,000 characters after removing the tag |
| `emotion` | string | no | Emotion preset; `neutral` by default |
| `exaggeration`, `cfg_weight`, `temperature`, `seed`, `speed` | number | no | See the table above |
| `reference_audio_url` | string | no | Public http(s) URL of the voice to clone |
| `reference_audio_base64` | string | no | The voice to clone as base64; ignored if a URL is also sent |
| `health_check` | boolean | no | Returns the worker status without generating audio |

Fields not listed here are ignored.

### Health Check

```json
{
  "input": {
    "health_check": true
  }
}
```

The response is `{"status": "healthy", "model_loaded": true, "gpu": "NVIDIA GeForce RTX 3090"}`. This differs from RunPod's own `GET /health`, which reports queue and worker counts.

## Response Format

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

| Field | Description |
|---|---|
| `audio_base64` | 16-bit PCM mono WAV, base64-encoded |
| `sample_rate` | Always 24000 |
| `duration_seconds` | Audio length, including 150 ms of silence between sentences |
| `text` | The spoken text, with the tag removed |
| `emotion` | Canonical English name of the emotion used |
| `metrics.generation_seconds` | Synthesis time for all sentences; excludes preparing the reference voice and encoding the WAV |
| `metrics.first_sentence_seconds` | Synthesis time of the first sentence; your time to first audio if you play sentence by sentence |
| `metrics.rtf` | Real-time factor, `generation_seconds / duration_seconds`; below 1 means faster than real time |
| `metrics.sentences` | Number of pieces the text was split into |
| `metrics.gpu` | GPU that served the job |
| `metrics.cold_start` | `true` on the first job a worker serves after starting |
| `metrics.model_load_seconds` | Model load time when the worker started |

RunPod wraps the output in its job envelope, as returned by `/status` or `/runsync`:

```json
{
  "id": "4f0a0c1e-...",
  "status": "COMPLETED",
  "delayTime": 1830,
  "executionTime": 4120,
  "output": {"audio_base64": "UklGRi...", "sample_rate": 24000, "...": "..."}
}
```

`delayTime` (queue plus worker start) and `executionTime` are in milliseconds.

### Error Response

On invalid input or a failed generation, the job ends as `FAILED` and the message is in `error`:

```json
{
  "id": "4f0a0c1e-...",
  "status": "FAILED",
  "error": "Unknown emotion: 'raiva'. Accepted: neutral, happy, excited, sad, empathetic, calm, whisper (or in Portuguese: neutro, alegre, animado, triste, empatico, calmo, sussurro)."
}
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

## Cost Estimates

The times below were measured on a local RTX 3060 Ti (8 GB) with the same code and package versions as the image. RunPod's 24 GB GPUs should be as fast or faster. Costs apply RunPod's flex prices of 2026-10-11: **$0.00019/s** for the 24 GB tier (L4, A5000, RTX 3090) and **$0.00034/s** for the 48 GB tier (A40, A6000). Check [runpod.io/pricing](https://www.runpod.io/pricing) for current prices.

| Task | Audio produced | Time (RTX 3060 Ti) | Cost, 24 GB tier | Cost, 48 GB tier |
|---|---|---|---|---|
| TTS, 30 words (173 characters) | ~17 s | ~9 s | ~$0.0017 | ~$0.0031 |
| TTS, 109 words (633 characters) | ~54 s | ~27 s | ~$0.0052 | ~$0.0093 |
| Voice clone setup (27 s reference clip) | | ~0.1 s | ~$0.00002 | ~$0.00003 |
| Cold start (model load + warm-up) | | ~8 s | ~$0.0015 | ~$0.0027 |
| Idle timeout after the last job | | 5 s (default) | ~$0.00095 | ~$0.0017 |

In short, a minute of speech costs about **$0.006** on the 24 GB tier and **$0.010** on the 48 GB tier. Cloning a voice adds almost nothing, apart from the download time when the voice comes by URL.

Billing is per second while a worker runs, not only while it executes jobs. A request that scales the endpoint from zero also pays the cold start, plus the image start. After the last job, the worker also bills the idle timeout. Cloud figures from `scripts/benchmark_endpoint.py` will replace these after the first deployment.

Other measurements on the same GPU:

| Measure | Value |
|---|---|
| Model load (weights on local disk) | 6–7 s |
| GPU memory | ~3.0 GiB of weights, ~3.5 GiB peak |
| Real-time factor | 0.5–0.58, so speech is generated about 2× faster than it plays |
| First sentence of a job | 1.0–2.3 s |
| Pronunciation (Parakeet character error rate) | 0–1.6% |

## Reference Audio Requirements

For best voice cloning results:
- WAV format, 16-bit mono, is preferred. MP3, FLAC and OGG are also accepted.
- Use 5–15 seconds of clear speech. The model uses the first 10 s for the voice timbre and the first 6 s as the speech prompt, so a clean 10 s clip is the sweet spot.
- Minimal background noise and no music.
- A single speaker only.
- A natural speaking pace, in the emotion you want to hear.
- Only clone voices with the speaker's consent.

Limits enforced by the worker:

| Source | Limit |
|---|---|
| `reference_audio_url` | `http(s)` only, at most 10 MB, downloaded within 15 s |
| `reference_audio_url` destination | Must resolve to a public address. `localhost`, private networks and the cloud metadata service are refused, also after redirects |
| `reference_audio_base64` | At most 10 MB once decoded |
| Whole request (RunPod) | 10 MB on `/run` and 20 MB on `/runsync`. Base64 adds a third to the size, so through `/run` you can send about 7 MB of audio; use a URL for bigger files |

The URL must point to the audio file itself, not to a sharing page such as a Google Drive or Dropbox preview.

## Examples by use case

**Meditation narration:**

```json
{
  "input": {
    "text": "[calmo] Bem-vindo à sua pausa de hoje. Encontre uma posição confortável, feche os olhos e respire fundo. Solte o ar devagar.",
    "temperature": 0.6
  }
}
```

**Good news:**

```json
{
  "input": {
    "text": "[alegre] Parabéns! O seu pedido foi aprovado e chega amanhã de manhã."
  }
}
```

**Customer support:**

```json
{
  "input": {
    "text": "Sinto muito pelo transtorno. Já registrei o seu caso e vou acompanhar tudo pessoalmente.",
    "emotion": "empathetic"
  }
}
```

**Whisper in your own voice:**

```json
{
  "input": {
    "text": "Fala baixinho, que o bebê acabou de dormir.",
    "emotion": "whisper",
    "reference_audio_url": "https://example.com/minha-voz-sussurrando.wav"
  }
}
```

**Reproducible output.** The same text, parameters and seed on the same GPU type give the same audio:

```json
{
  "input": {
    "text": "Este áudio sai igual toda vez.",
    "seed": 1234
  }
}
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

## Local Development

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

`scripts/simulate_image.sh` reproduces the image's build and offline start without Docker, which was not available where this worker was developed. With Docker and the NVIDIA Container Toolkit, you can build the image and run one test job. Outside RunPod, the SDK needs `--test_input`:

```bash
docker build -t chatterbox-ptbr-worker .
docker run --rm --gpus all chatterbox-ptbr-worker \
  python -u handler.py --test_input '{"input": {"text": "Teste local da imagem."}}'
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

## Credits and License

- **Model:**
  - [Chatterbox](https://github.com/resemble-ai/chatterbox) by Resemble AI, MIT license, pinned at commit `5de7a54`;
  - the pt-BR language pack [`ResembleAI/Chatterbox-Multilingual-pt-br`](https://huggingface.co/ResembleAI/Chatterbox-Multilingual-pt-br), MIT license.
- **Watermark:** [Perth](https://github.com/resemble-ai/Perth) by Resemble AI, MIT license.
- **API compatibility:** the field names follow [profzeller/chatterbox-wellness-worker](https://github.com/profzeller/chatterbox-wellness-worker), so most requests work unchanged. The exceptions are `speed` values other than 1.0 and the `angry` and `surprised` emotions, which this worker doesn't have. All code here was written from scratch.
- **This repository:** no license has been granted yet.
