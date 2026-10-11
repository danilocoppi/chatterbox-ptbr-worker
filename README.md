# chatterbox-ptbr-worker

Worker serverless (fila) do RunPod que gera fala em **português do Brasil com emoção**, com o Chatterbox Multilingual V3 e o pacote pt-BR da Resemble AI (MIT). Os pesos ficam dentro da imagem, e o worker liga sem internet.

Especificação: `docs/superpowers/specs/2026-10-11-chatterbox-ptbr-worker-design.md`.

## Pedido e resposta

```json
{"input": {"text": "[alegre] Consegui! A entrevista foi ótima.", "seed": 42}}
```

Campos:
- `text` (obrigatório, até 1.000 caracteres);
- `emotion`: neutro, alegre, animado, triste, empatico, calmo ou sussurro, também em inglês;
- `exaggeration`, `cfg_weight`, `temperature` e `seed`;
- `reference_audio_url` ou `reference_audio_base64`, para usar outra voz;
- `health_check`.

A resposta traz `audio_base64` (WAV 24 kHz), `duration_seconds`, `text`, `emotion` e `metrics`. As medidas são tempo de geração, primeira frase, RTF, GPU, arranque frio e carga do modelo. Um erro vem como `{"error": "..."}`.

## Desenvolvimento local

```bash
export PY="env -u LD_LIBRARY_PATH $HOME/Projects/61-Claudinei/chatterbox-tts/.venv/bin/python"
$PY -m pytest tests -q                       # testes sem GPU
$PY baixar_pesos.py                          # pesos e voz padrão em ./modelos
$PY scripts/integracao_local.py              # modelo real na GPU -> saida_local/
env -u LD_LIBRARY_PATH uv run --python 3.11 --with sherpa-onnx --with soundfile python scripts/transcrever.py saida_local
```

A imagem instala as versões exatas do `requirements.lock`. Depois de mudar o `requirements.txt`, gere o lock de novo:

```bash
env -u LD_LIBRARY_PATH uv pip compile requirements.txt --python-version 3.11 -o requirements.lock
```

## Publicar no RunPod

1. **Settings → Connections:** conecte o GitHub e dê acesso a este repositório.
2. **Serverless → New Endpoint → GitHub Repo:** escolha este repositório, o branch `main` e o `Dockerfile` da raiz.
3. **GPU:** uma de 24 GB (RTX 3090, A5000 ou L4). Para comparar, crie outro endpoint com uma de 48 GB (A40 ou A6000).
4. **Workers:** mínimo de ativos em 0 e máximo em 1. **Disco do container:** 20 GB. **CUDA permitido:** 12.4 ou superior.
5. Espere a compilação, que é demorada por causa da imagem de ~10 GB.
6. Teste com uma chave de API que acesse o endpoint, salva em `~/.config/runpod/api_key`:

```bash
python3 scripts/testar_endpoint.py ENDPOINT_ID
env -u LD_LIBRARY_PATH uv run --python 3.11 --with sherpa-onnx --with soundfile python scripts/transcrever.py saida_runpod/ENDPOINT_ID
```

## Observações

- Toda saída leva a marca d'água imperceptível Perth da Resemble.
- A voz padrão `pt_br_f2` é a da demo oficial da Resemble e serve para testes. Para uso contínuo, use uma voz própria com consentimento.
- Não há streaming nem `speed`. O texto é dividido em frases de até 300 caracteres, com 150 ms de silêncio entre elas.
- A `reference_audio_url` precisa apontar para um endereço público. Endereços internos, como `localhost`, redes privadas e o serviço de metadados da nuvem, são recusados, inclusive depois de redirecionamentos. O download tem limite de 15 s e de 10 MB.
- O `runpod` fica fixado em 1.10.0. O chatterbox no commit `5de7a54` exige `gradio==6.8.0`, que pede `tomlkit<0.14`, e o `runpod` a partir da 1.10.1 pede `tomlkit>=0.15`.
