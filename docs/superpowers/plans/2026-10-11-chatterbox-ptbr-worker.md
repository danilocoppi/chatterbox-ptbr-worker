# chatterbox-ptbr-worker: plano de implementação

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Construir um worker serverless de fila no RunPod que gera fala em pt-BR com emoção usando o Chatterbox Multilingual V3 com o pacote pt-BR, com os pesos embutidos na imagem.

**Architecture:** A lógica pura fica em `sintese.py`, sem GPU nem torch: entrada, emoção, frases, áudio e voz de referência. O carregamento do modelo fica em `ptbr.py`, e a classe `Servico` em `handler.py` liga as duas ao RunPod. O modelo entra na `Servico` por injeção, o que permite testá-la com um modelo falso. A imagem Docker baixa os pesos na montagem e roda sem internet.

**Tech Stack:**
- Python 3.11;
- `chatterbox-tts` no commit `5de7a54`, com PyTorch 2.6.0 e CUDA 12.4;
- `runpod` 1.12.0, `soundfile` e `numpy`;
- `pytest` 9.1.1 nos testes;
- `sherpa-onnx` com o Parakeet local, só para conferir a pronúncia.

**Spec:** `docs/superpowers/specs/2026-10-11-chatterbox-ptbr-worker-design.md`

## Global Constraints

- **Raiz e Python local:** todos os comandos rodam na raiz `~/Projects/61-Claudinei/chatterbox-ptbr-worker`, com `export PY="env -u LD_LIBRARY_PATH $HOME/Projects/61-Claudinei/chatterbox-tts/.venv/bin/python"` definido. O `env -u LD_LIBRARY_PATH` é obrigatório nesta máquina.
- **Versões fixas:** Python 3.11; `chatterbox-tts @ git+https://github.com/resemble-ai/chatterbox.git@5de7a54aa4e5e2baadb0182dde554908b48b85c2`; `runpod==1.12.0`.
- **Modelo:**
  - só pt-BR, sempre com `language_id="pt"`;
  - pesos `t3_pt_br.safetensors` e `s3gen_v3.pt` de `ResembleAI/Chatterbox-Multilingual-pt-br`;
  - arquivos base `ve.pt`, `conds.pt`, `grapheme_mtl_merged_expanded_v1.json` e `Cangjie5_TC.json` de `ResembleAI/chatterbox`.
- **Voz padrão:** `https://storage.googleapis.com/chatterbox-demo-samples/mtl-v3-single-language-prompts/pt-br/pt_br_f2.wav`.
- **Campos de entrada:** `text`, `emotion`, `exaggeration`, `cfg_weight`, `temperature`, `seed`, `reference_audio_url`, `reference_audio_base64` e `health_check`.
- **Campos de saída:** `audio_base64`, `sample_rate`, `duration_seconds`, `text`, `emotion` e `metrics`. As medidas são `generation_seconds`, `first_sentence_seconds`, `rtf`, `sentences`, `gpu`, `cold_start` e `model_load_seconds`.
- **Limites:**
  - texto de 1 a 1.000 caracteres, medido depois de tirar a etiqueta;
  - frases de até 300 caracteres, com silêncio de 150 ms entre elas;
  - voz por URL com até 15 s de download e 10 MB;
  - voz em base64 com até 10 MB depois de decodificada.
- **Faixas:** `exaggeration` de 0,25 a 2,0; `cfg_weight` de 0,0 a 1,0; `temperature` de 0,05 a 2,0, com padrão 0,8.
- **Emoções (exaggeration, cfg_weight):**
  - neutro (0,5, 0,5); alegre (0,8, 0,3); animado (0,9, 0,3); triste (0,35, 0,5);
  - empatico (0,4, 0,5); calmo (0,35, 0,5); sussurro (0,3, 0,5);
  - sinônimos: neutral, happy, excited, sad, empathetic, calm, whisper.
- **Erros:** voltam como `{"error": "<mensagem em português>"}`. Uma exceção inesperada nunca derruba o worker.
- **Imagem:** ao ligar, `HF_HUB_OFFLINE=1`.
- **Repositório:** privado, `danilocoppi/chatterbox-ptbr-worker`.
- **Commits:** terminam com o trailer `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus

1. **Base64 com quebras de linha:** a saída do comando `base64` quebra a cada 76 caracteres e precisa decodificar normalmente. O teste fica na Task 3.
2. **Texto que é só a etiqueta** (`"[alegre]"` ou `"[alegre]   "`): precisa dar um erro claro, sem gerar áudio vazio. O teste fica na Task 1.
3. **Uma "palavra" gigante,** como uma URL ou um código colado com mais de 300 caracteres: precisa ser cortada em pedaços de até 300, sem estourar o limite do modelo. O teste fica na Task 2.
4. **Voz que não é áudio,** como um link do Drive que devolve HTML: precisa dar o erro "Não foi possível ler a voz de referência", e o pedido seguinte volta à voz padrão. O teste fica na Task 5.
5. **`emotion` vazio (`""`) com etiqueta no texto:** vale a etiqueta, sem erro. O teste fica na Task 1.

---

### Task 1: Entrada e emoções (`sintese.interpretar`)

**Files:**
- Create: `.gitignore`, `tests/conftest.py`, `sintese.py`
- Test: `tests/test_entrada.py`

**Interfaces:**
- Consumes: nada.
- Produces:
  - `sintese.EntradaInvalida(ValueError)`;
  - `sintese.Pedido`, uma dataclass congelada com `texto: str`, `emocao: str`, `exaggeration: float`, `cfg_weight: float`, `temperature: float`, `seed: Optional[int]`, `voz_url: Optional[str]` e `voz_base64: Optional[str]`;
  - `sintese.interpretar(entrada) -> Pedido`;
  - `sintese.resolver_emocao(nome: str) -> str`;
  - `sintese.EMOCOES: dict[str, tuple[float, float]]`;
  - as constantes `MAX_CARACTERES = 1000` e `TEMPERATURA_PADRAO = 0.8`.

- [ ] **Step 1: Preparar o ambiente e os arquivos de apoio**

```bash
uv pip install --python ~/Projects/61-Claudinei/chatterbox-tts/.venv/bin/python pytest==9.1.1 runpod==1.12.0
mkdir -p tests
```

`.gitignore`:
```
__pycache__/
*.pyc
.pytest_cache/
modelos/
saida_local/
saida_runpod/
```

`tests/conftest.py`:
```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
```

- [ ] **Step 2: Escrever os testes que falham**

`tests/test_entrada.py`:
```python
import re

import pytest

from sintese import EntradaInvalida, interpretar


def test_padroes_sem_emocao():
    p = interpretar({"text": "  Olá, tudo bem?  "})
    assert p.texto == "Olá, tudo bem?" and p.emocao == "neutro"
    assert (p.exaggeration, p.cfg_weight, p.temperature, p.seed) == (0.5, 0.5, 0.8, None)
    assert p.voz_url is None and p.voz_base64 is None


@pytest.mark.parametrize("nome, esperado", [
    ("alegre", "alegre"), ("HAPPY", "alegre"), ("Empático", "empatico"),
    ("empathetic", "empatico"), ("whisper", "sussurro"), ("  calmo ", "calmo"),
])
def test_emocao_aceita_sinonimos_maiusculas_e_acentos(nome, esperado):
    assert interpretar({"text": "Oi.", "emotion": nome}).emocao == esperado


def test_emocao_define_exaggeration_e_cfg():
    p = interpretar({"text": "Oi.", "emotion": "animado"})
    assert (p.exaggeration, p.cfg_weight) == (0.9, 0.3)


def test_numeros_sobrescrevem_a_emocao():
    p = interpretar({"text": "Oi.", "emotion": "triste", "exaggeration": 1.2,
                     "cfg_weight": 0, "temperature": 0.5, "seed": 42})
    assert (p.exaggeration, p.cfg_weight, p.temperature, p.seed) == (1.2, 0.0, 0.5, 42)


def test_etiqueta_no_inicio_define_emocao_e_sai_do_texto():
    p = interpretar({"text": "[triste] Que pena."})
    assert p.emocao == "triste" and p.texto == "Que pena."


def test_campo_tem_precedencia_e_etiqueta_sai_mesmo_assim():
    p = interpretar({"text": "[triste] Que pena.", "emotion": "calmo"})
    assert p.emocao == "calmo" and p.texto == "Que pena."


def test_emotion_vazio_usa_a_etiqueta():
    assert interpretar({"text": "[alegre] Oba!", "emotion": ""}).emocao == "alegre"


def test_colchete_com_numero_e_falado_como_texto():
    p = interpretar({"text": "[1] Primeiro item."})
    assert p.texto == "[1] Primeiro item." and p.emocao == "neutro"


def test_url_tem_precedencia_sobre_base64():
    p = interpretar({"text": "Oi.", "reference_audio_url": "https://x/v.wav",
                     "reference_audio_base64": "QUJD"})
    assert p.voz_url == "https://x/v.wav" and p.voz_base64 is None


def test_speed_1_e_aceito():
    assert interpretar({"text": "Oi.", "speed": 1.0}).texto == "Oi."


@pytest.mark.parametrize("entrada, trecho", [
    ("nao e dict", "objeto JSON"),
    ({}, "Informe o texto"),
    ({"text": "   "}, "Informe o texto"),
    ({"text": "[alegre]"}, "depois da etiqueta"),
    ({"text": "[alegre]    "}, "depois da etiqueta"),
    ({"text": "x" * 1001}, "máximo é 1000"),
    ({"text": "Oi.", "emotion": "raiva"}, "Emoção desconhecida: 'raiva'"),
    ({"text": "[raiva] Oi."}, "Emoção desconhecida"),
    ({"text": "Oi.", "emotion": 3}, "'emotion' precisa ser um texto"),
    ({"text": "Oi.", "exaggeration": "0.8"}, "'exaggeration' precisa ser um número"),
    ({"text": "Oi.", "exaggeration": True}, "'exaggeration' precisa ser um número"),
    ({"text": "Oi.", "exaggeration": 3}, "'exaggeration' precisa estar entre 0.25 e 2.0"),
    ({"text": "Oi.", "cfg_weight": 1.5}, "'cfg_weight' precisa estar entre 0.0 e 1.0"),
    ({"text": "Oi.", "temperature": 0}, "'temperature' precisa estar entre 0.05 e 2.0"),
    ({"text": "Oi.", "speed": 1.5}, "'speed' não é suportado"),
    ({"text": "Oi.", "seed": -1}, "'seed' precisa ser um inteiro"),
    ({"text": "Oi.", "seed": 1.5}, "'seed' precisa ser um inteiro"),
    ({"text": "Oi.", "reference_audio_url": "ftp://x/voz.wav"}, "URL http(s)"),
    ({"text": "Oi.", "reference_audio_base64": 123}, "texto em base64"),
])
def test_entradas_invalidas(entrada, trecho):
    with pytest.raises(EntradaInvalida, match=re.escape(trecho)):
        interpretar(entrada)
```

- [ ] **Step 3: Rodar e ver falhar**

Run: `$PY -m pytest tests/test_entrada.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'sintese'`

- [ ] **Step 4: Implementar o mínimo**

`sintese.py`:
```python
"""Lógica do worker sem GPU: entrada, emoção, frases, áudio e voz de referência."""
import re
import unicodedata
from dataclasses import dataclass
from typing import Optional

MAX_CARACTERES = 1000
TEMPERATURA_PADRAO = 0.8

# emoção -> (exaggeration, cfg_weight)
EMOCOES = {
    "neutro": (0.5, 0.5),
    "alegre": (0.8, 0.3),
    "animado": (0.9, 0.3),
    "triste": (0.35, 0.5),
    "empatico": (0.4, 0.5),
    "calmo": (0.35, 0.5),
    "sussurro": (0.3, 0.5),
}
SINONIMOS = {
    "neutral": "neutro", "happy": "alegre", "excited": "animado", "sad": "triste",
    "empathetic": "empatico", "calm": "calmo", "whisper": "sussurro",
}

_ETIQUETA = re.compile(r"^\s*\[([A-Za-zÀ-ÿ_]+)\]\s*")


class EntradaInvalida(ValueError):
    """Erro de validação; a mensagem é devolvida ao cliente."""


@dataclass(frozen=True)
class Pedido:
    texto: str
    emocao: str
    exaggeration: float
    cfg_weight: float
    temperature: float
    seed: Optional[int]
    voz_url: Optional[str]
    voz_base64: Optional[str]


def _sem_acento(texto: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", texto) if unicodedata.category(c) != "Mn")


def resolver_emocao(nome: str) -> str:
    """Nome canônico da emoção; aceita sinônimos em inglês, maiúsculas e acentos."""
    chave = _sem_acento(str(nome)).strip().lower()
    chave = SINONIMOS.get(chave, chave)
    if chave not in EMOCOES:
        raise EntradaInvalida(f"Emoção desconhecida: '{nome}'. Aceitas: {', '.join(EMOCOES)}.")
    return chave


def _numero(entrada: dict, campo: str, minimo: float, maximo: float, padrao: float) -> float:
    valor = entrada.get(campo)
    if valor is None:
        return padrao
    if isinstance(valor, bool) or not isinstance(valor, (int, float)):
        raise EntradaInvalida(f"'{campo}' precisa ser um número.")
    if not minimo <= valor <= maximo:
        raise EntradaInvalida(f"'{campo}' precisa estar entre {minimo} e {maximo}.")
    return float(valor)


def interpretar(entrada) -> Pedido:
    """Valida a entrada do RunPod e resolve a emoção e os parâmetros."""
    if not isinstance(entrada, dict):
        raise EntradaInvalida("A entrada precisa ser um objeto JSON.")
    texto = entrada.get("text")
    if not isinstance(texto, str) or not texto.strip():
        raise EntradaInvalida("Informe o texto em 'text'.")

    emocao_etiqueta = None
    etiqueta = _ETIQUETA.match(texto)
    if etiqueta:
        emocao_etiqueta = resolver_emocao(etiqueta.group(1))
        texto = texto[etiqueta.end():]
    texto = texto.strip()
    if not texto:
        raise EntradaInvalida("Não há texto para falar depois da etiqueta de emoção.")
    if len(texto) > MAX_CARACTERES:
        raise EntradaInvalida(f"O texto tem {len(texto)} caracteres; o máximo é {MAX_CARACTERES}.")

    campo = entrada.get("emotion")
    if campo is not None and not isinstance(campo, str):
        raise EntradaInvalida("'emotion' precisa ser um texto.")
    emocao = resolver_emocao(campo) if campo else (emocao_etiqueta or "neutro")
    exaggeration_padrao, cfg_padrao = EMOCOES[emocao]

    speed = entrada.get("speed")
    if speed is not None and speed != 1:
        raise EntradaInvalida("'speed' não é suportado; envie 1.0 ou omita o campo.")

    seed = entrada.get("seed")
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int) or seed < 0):
        raise EntradaInvalida("'seed' precisa ser um inteiro maior ou igual a zero.")

    url = entrada.get("reference_audio_url") or None
    b64 = entrada.get("reference_audio_base64") or None
    if url is not None and (not isinstance(url, str) or not url.lower().startswith(("http://", "https://"))):
        raise EntradaInvalida("'reference_audio_url' precisa ser uma URL http(s).")
    if b64 is not None and not isinstance(b64, str):
        raise EntradaInvalida("'reference_audio_base64' precisa ser um texto em base64.")

    return Pedido(
        texto=texto,
        emocao=emocao,
        exaggeration=_numero(entrada, "exaggeration", 0.25, 2.0, exaggeration_padrao),
        cfg_weight=_numero(entrada, "cfg_weight", 0.0, 1.0, cfg_padrao),
        temperature=_numero(entrada, "temperature", 0.05, 2.0, TEMPERATURA_PADRAO),
        seed=seed,
        voz_url=url,
        voz_base64=None if url else b64,
    )
```

- [ ] **Step 5: Rodar e ver passar**

Run: `$PY -m pytest tests/test_entrada.py -q`
Expected: PASS (todos os testes)

- [ ] **Step 6: Commit**

```bash
git add .gitignore tests/conftest.py tests/test_entrada.py sintese.py
git commit -m "feat: interpretação da entrada e das emoções" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Frases e áudio (`dividir_frases`, `juntar`, `wav_base64`)

**Files:**
- Modify: `sintese.py`, para acrescentar imports, constantes e funções
- Test: `tests/test_frases_audio.py`

**Interfaces:**
- Consumes: o módulo `sintese` da Task 1.
- Produces:
  - `sintese.dividir_frases(texto: str) -> list[str]`;
  - `sintese.juntar(trechos: list, sr: int, pausa_s: float = PAUSA_ENTRE_FRASES_S) -> np.ndarray` (float32, 1-D);
  - `sintese.wav_base64(audio: np.ndarray, sr: int) -> str` (WAV PCM 16 bits);
  - as constantes `MAX_FRASE = 300` e `PAUSA_ENTRE_FRASES_S = 0.15`.

- [ ] **Step 1: Escrever os testes que falham**

`tests/test_frases_audio.py`:
```python
import base64
import io

import numpy as np
import soundfile as sf

import sintese


def test_divide_na_pontuacao_final():
    assert sintese.dividir_frases("Oi! Tudo bem? Sim. Até mais… Tchau") == [
        "Oi!", "Tudo bem?", "Sim.", "Até mais…", "Tchau"]


def test_frase_longa_quebra_na_virgula_sem_perder_texto():
    frase = ", ".join(["palavra de teste numero"] * 20) + "."
    pedacos = sintese.dividir_frases(frase)
    assert len(pedacos) > 1 and all(len(p) <= sintese.MAX_FRASE for p in pedacos)
    assert " ".join(pedacos).split() == frase.split()


def test_texto_sem_pontuacao_quebra_entre_palavras():
    texto = " ".join(["palavra"] * 100)
    pedacos = sintese.dividir_frases(texto)
    assert len(pedacos) > 1 and all(len(p) <= sintese.MAX_FRASE for p in pedacos)
    assert " ".join(pedacos).split() == texto.split()


def test_palavra_gigante_e_cortada():
    texto = "Veja " + "x" * 700 + " agora."
    pedacos = sintese.dividir_frases(texto)
    assert all(len(p) <= sintese.MAX_FRASE for p in pedacos)
    assert "".join(pedacos).replace(" ", "") == texto.replace(" ", "")


def test_juntar_coloca_silencio_entre_trechos():
    audio = sintese.juntar([np.ones(10), np.ones(20)], sr=1000, pausa_s=0.15)
    assert audio.dtype == np.float32 and len(audio) == 10 + 150 + 20
    assert audio[10:160].max() == 0.0


def test_wav_base64_ida_e_volta_com_clipping():
    sr = 24000
    audio = np.linspace(-1.5, 1.5, sr // 10, dtype=np.float32)
    lido, sr_lido = sf.read(io.BytesIO(base64.b64decode(sintese.wav_base64(audio, sr))), dtype="float32")
    assert sr_lido == sr and len(lido) == len(audio)
    assert np.allclose(lido, np.clip(audio, -1.0, 1.0), atol=1e-3)
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `$PY -m pytest tests/test_frases_audio.py -q`
Expected: FAIL with `AttributeError: module 'sintese' has no attribute 'dividir_frases'`

- [ ] **Step 3: Implementar o mínimo**

No topo de `sintese.py`, acrescentar os imports (mantendo os existentes):
```python
import base64
import io

import numpy as np
import soundfile as sf
```

Acrescentar as constantes, logo abaixo de `TEMPERATURA_PADRAO`:
```python
MAX_FRASE = 300
PAUSA_ENTRE_FRASES_S = 0.15
```

Acrescentar as expressões, logo abaixo de `_ETIQUETA`:
```python
_FIM_DE_FRASE = re.compile(r"(?<=[.!?…])\s+")
_VIRGULA = re.compile(r"(?<=[,;])\s+")
```

Acrescentar as funções no fim do arquivo:
```python
def _agrupar(unidades: list[str]) -> list[str]:
    """Junta unidades com espaço em pedaços de até MAX_FRASE caracteres."""
    pedacos, atual = [], ""
    for unidade in unidades:
        if atual and len(atual) + 1 + len(unidade) > MAX_FRASE:
            pedacos.append(atual)
            atual = unidade
        else:
            atual = f"{atual} {unidade}" if atual else unidade
    if atual:
        pedacos.append(atual)
    return pedacos


def _palavras(texto: str) -> list[str]:
    """Palavras do texto; uma palavra maior que MAX_FRASE é cortada em pedaços desse tamanho."""
    saida = []
    for palavra in texto.split():
        saida.extend(palavra[i:i + MAX_FRASE] for i in range(0, len(palavra), MAX_FRASE))
    return saida


def _quebrar_longa(frase: str) -> list[str]:
    if len(frase) <= MAX_FRASE:
        return [frase]
    unidades = []
    for parte in _VIRGULA.split(frase):
        unidades.extend([parte] if len(parte) <= MAX_FRASE else _agrupar(_palavras(parte)))
    return _agrupar(unidades)


def dividir_frases(texto: str) -> list[str]:
    """Divide em frases (. ! ? …) e quebra as maiores que MAX_FRASE em vírgulas ou palavras."""
    frases = []
    for frase in _FIM_DE_FRASE.split(texto.strip()):
        frase = frase.strip()
        if frase:
            frases.extend(_quebrar_longa(frase))
    return frases


def juntar(trechos: list, sr: int, pausa_s: float = PAUSA_ENTRE_FRASES_S) -> np.ndarray:
    """Concatena os trechos de áudio com um silêncio entre eles."""
    silencio = np.zeros(int(round(sr * pausa_s)), dtype=np.float32)
    partes = []
    for i, trecho in enumerate(trechos):
        if i:
            partes.append(silencio)
        partes.append(np.asarray(trecho, dtype=np.float32).reshape(-1))
    return np.concatenate(partes) if partes else np.zeros(0, dtype=np.float32)


def wav_base64(audio: np.ndarray, sr: int) -> str:
    """Codifica o áudio como WAV PCM 16 bits mono em base64."""
    buffer = io.BytesIO()
    sinal = np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0)
    sf.write(buffer, sinal, sr, format="WAV", subtype="PCM_16")
    return base64.b64encode(buffer.getvalue()).decode("ascii")
```

- [ ] **Step 4: Rodar e ver passar**

Run: `$PY -m pytest tests -q`
Expected: PASS (Task 1 + Task 2)

- [ ] **Step 5: Commit**

```bash
git add sintese.py tests/test_frases_audio.py
git commit -m "feat: divisão em frases e codificação do áudio" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: Voz de referência (`decodificar_base64`, `baixar_voz`, `obter_voz`, `sufixo_audio`)

**Files:**
- Modify: `sintese.py`
- Test: `tests/test_voz.py`

**Interfaces:**
- Consumes: `sintese.Pedido` e `sintese.EntradaInvalida` (Task 1); o `base64` já importado (Task 2).
- Produces:
  - `sintese.decodificar_base64(dado: str) -> bytes`;
  - `sintese.baixar_voz(url: str, tempo_max_s: float = TEMPO_MAX_DOWNLOAD_S) -> bytes`;
  - `sintese.obter_voz(pedido: Pedido) -> Optional[bytes]`, que devolve `None` para usar a voz padrão;
  - `sintese.sufixo_audio(dados: bytes) -> str`;
  - as constantes `MAX_BYTES_VOZ = 10 * 1024 * 1024` e `TEMPO_MAX_DOWNLOAD_S = 15.0`.

- [ ] **Step 1: Escrever os testes que falham**

`tests/test_voz.py`:
```python
import base64
import dataclasses
import functools
import http.server
import threading

import pytest

import sintese
from sintese import EntradaInvalida


class _Silencioso(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


@pytest.fixture
def servidor(tmp_path):
    (tmp_path / "voz.wav").write_bytes(b"RIFF" + b"\x01" * 1000)
    (tmp_path / "grande.bin").write_bytes(b"\x00" * 2048)
    httpd = http.server.ThreadingHTTPServer(
        ("127.0.0.1", 0), functools.partial(_Silencioso, directory=str(tmp_path)))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()


def test_base64_com_prefixo_data_e_quebras_de_linha():
    dados = b"RIFF" + bytes(range(256)) * 4
    codificado = base64.encodebytes(dados).decode()
    assert "\n" in codificado
    assert sintese.decodificar_base64("data:audio/wav;base64," + codificado) == dados


@pytest.mark.parametrize("dado, trecho", [
    ("isto não é base64!", "não é um base64 válido"),
    ("", "está vazia"),
])
def test_base64_invalido(dado, trecho):
    with pytest.raises(EntradaInvalida, match=trecho):
        sintese.decodificar_base64(dado)


def test_base64_acima_do_limite(monkeypatch):
    monkeypatch.setattr(sintese, "MAX_BYTES_VOZ", 10)
    with pytest.raises(EntradaInvalida, match="passa de 10 MB"):
        sintese.decodificar_base64(base64.b64encode(b"x" * 11).decode())


def test_baixa_voz_por_url(servidor):
    assert sintese.baixar_voz(f"{servidor}/voz.wav") == b"RIFF" + b"\x01" * 1000


def test_url_inexistente_vira_erro(servidor):
    with pytest.raises(EntradaInvalida, match="Não foi possível baixar"):
        sintese.baixar_voz(f"{servidor}/nao-existe.wav")


def test_download_acima_do_limite(servidor, monkeypatch):
    monkeypatch.setattr(sintese, "MAX_BYTES_VOZ", 1024)
    with pytest.raises(EntradaInvalida, match="passa de 10 MB"):
        sintese.baixar_voz(f"{servidor}/grande.bin")


def test_obter_voz_escolhe_a_fonte(servidor):
    base = sintese.Pedido("Oi.", "neutro", 0.5, 0.5, 0.8, None, None, None)
    assert sintese.obter_voz(base) is None
    com_b64 = dataclasses.replace(base, voz_base64=base64.b64encode(b"abc").decode())
    assert sintese.obter_voz(com_b64) == b"abc"
    com_url = dataclasses.replace(base, voz_url=f"{servidor}/voz.wav")
    assert sintese.obter_voz(com_url).startswith(b"RIFF")


@pytest.mark.parametrize("cabecalho, sufixo", [
    (b"RIFF....", ".wav"), (b"fLaC....", ".flac"), (b"OggS....", ".ogg"),
    (b"ID3.....", ".mp3"), (b"\xff\xfb....", ".mp3"), (b"<html>..", ".audio"),
])
def test_sufixo_pelo_conteudo(cabecalho, sufixo):
    assert sintese.sufixo_audio(cabecalho) == sufixo
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `$PY -m pytest tests/test_voz.py -q`
Expected: FAIL with `AttributeError: module 'sintese' has no attribute 'decodificar_base64'`

- [ ] **Step 3: Implementar o mínimo**

No topo de `sintese.py`, acrescentar os imports:
```python
import time
import urllib.request
```

Acrescentar as constantes, junto das demais:
```python
MAX_BYTES_VOZ = 10 * 1024 * 1024
TEMPO_MAX_DOWNLOAD_S = 15.0
```

Acrescentar as funções no fim do arquivo:
```python
def decodificar_base64(dado: str) -> bytes:
    """Decodifica a voz em base64; aceita prefixo data: e quebras de linha."""
    if dado.startswith("data:"):
        dado = dado.split(",", 1)[-1]
    dado = "".join(dado.split())
    try:
        bruto = base64.b64decode(dado, validate=True)
    except ValueError as erro:
        raise EntradaInvalida("'reference_audio_base64' não é um base64 válido.") from erro
    if not bruto:
        raise EntradaInvalida("A voz de referência está vazia.")
    if len(bruto) > MAX_BYTES_VOZ:
        raise EntradaInvalida("A voz de referência passa de 10 MB.")
    return bruto


def baixar_voz(url: str, tempo_max_s: float = TEMPO_MAX_DOWNLOAD_S) -> bytes:
    """Baixa a voz de referência respeitando o tempo e o tamanho máximos."""
    inicio = time.monotonic()
    partes, total = [], 0
    try:
        requisicao = urllib.request.Request(url, headers={"User-Agent": "chatterbox-ptbr-worker"})
        with urllib.request.urlopen(requisicao, timeout=tempo_max_s) as resposta:
            while bloco := resposta.read(64 * 1024):
                total += len(bloco)
                if total > MAX_BYTES_VOZ:
                    raise EntradaInvalida("A voz de referência passa de 10 MB.")
                if time.monotonic() - inicio > tempo_max_s:
                    raise EntradaInvalida(f"O download da voz de referência passou de {tempo_max_s:.0f} s.")
                partes.append(bloco)
    except EntradaInvalida:
        raise
    except (OSError, ValueError) as erro:
        raise EntradaInvalida(f"Não foi possível baixar a voz de referência: {erro}") from erro
    if not partes:
        raise EntradaInvalida("A voz de referência está vazia.")
    return b"".join(partes)


def obter_voz(pedido: Pedido) -> Optional[bytes]:
    """Bytes da voz de referência do pedido, ou None para usar a voz padrão."""
    if pedido.voz_url:
        return baixar_voz(pedido.voz_url)
    if pedido.voz_base64:
        return decodificar_base64(pedido.voz_base64)
    return None


def sufixo_audio(dados: bytes) -> str:
    """Extensão do arquivo temporário a partir do conteúdo, para o leitor de áudio."""
    if dados[:4] == b"RIFF":
        return ".wav"
    if dados[:4] == b"fLaC":
        return ".flac"
    if dados[:4] == b"OggS":
        return ".ogg"
    if dados[:3] == b"ID3" or dados[:2] in (b"\xff\xfb", b"\xff\xf3", b"\xff\xf2"):
        return ".mp3"
    return ".audio"
```

- [ ] **Step 4: Rodar e ver passar**

Run: `$PY -m pytest tests -q`
Expected: PASS (Tasks 1–3)

- [ ] **Step 5: Commit**

```bash
git add sintese.py tests/test_voz.py
git commit -m "feat: voz de referência por URL ou base64" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: Carregamento do modelo (`ptbr.py`) e download dos pesos (`baixar_pesos.py`)

**Files:**
- Create: `ptbr.py`, `baixar_pesos.py`

**Interfaces:**
- Consumes: nada do worker; depende do pacote `chatterbox` instalado.
- Produces:
  - `ptbr.montar_checkpoint() -> Path`;
  - `ptbr.baixar_voz_padrao() -> Path`;
  - `ptbr.carregar(device: str = "cuda") -> ChatterboxMultilingualTTS`;
  - `ptbr.DIR_MODELOS: Path`, configurável pela variável `CHATTERBOX_MODELOS`, com padrão `./modelos`.

- [ ] **Step 1: Escrever `ptbr.py`**

```python
"""Carrega o Chatterbox Multilingual V3 com o pacote pt-BR da Resemble AI."""
import os
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file as load_safetensors

from chatterbox.mtl_tts import ChatterboxMultilingualTTS, Conditionals
from chatterbox.models.s3gen import S3Gen
from chatterbox.models.t3 import T3
from chatterbox.models.t3.modules.t3_config import T3Config
from chatterbox.models.tokenizers import MTLTokenizer
from chatterbox.models.voice_encoder import VoiceEncoder

BASE_REPO = "ResembleAI/chatterbox"
PACOTE_REPO = "ResembleAI/Chatterbox-Multilingual-pt-br"
VOZ_PADRAO_URL = ("https://storage.googleapis.com/chatterbox-demo-samples/"
                  "mtl-v3-single-language-prompts/pt-br/pt_br_f2.wav")
DIR_MODELOS = Path(os.environ.get("CHATTERBOX_MODELOS", Path(__file__).resolve().parent / "modelos"))
VOZ_PADRAO = DIR_MODELOS / "vozes" / "pt_br_f2.wav"
# O s3gen_v3.pt não traz estes buffers fixos, que o construtor do S3Gen já cria.
BUFFERS_AUSENTES_S3GEN = {"tokenizer._mel_filters", "tokenizer.window"}


def montar_checkpoint() -> Path:
    """Junta os arquivos do repositório base com o pacote pt-BR numa pasta (links simbólicos)."""
    base = Path(snapshot_download(BASE_REPO, allow_patterns=[
        "ve.pt", "conds.pt", "grapheme_mtl_merged_expanded_v1.json", "Cangjie5_TC.json"]))
    pacote = Path(snapshot_download(PACOTE_REPO, allow_patterns=[
        "t3_pt_br.safetensors", "s3gen_v3.pt", "grapheme_mtl_merged_expanded_v1.json"]))
    destino = DIR_MODELOS / "pt-br"
    destino.mkdir(parents=True, exist_ok=True)
    links = {
        "ve.pt": base / "ve.pt",
        "conds.pt": base / "conds.pt",
        "Cangjie5_TC.json": base / "Cangjie5_TC.json",
        "grapheme_mtl_merged_expanded_v1.json": pacote / "grapheme_mtl_merged_expanded_v1.json",
        "t3_pt_br.safetensors": pacote / "t3_pt_br.safetensors",
        "s3gen.pt": pacote / "s3gen_v3.pt",
    }
    for nome, alvo in links.items():
        link = destino / nome
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(alvo.resolve())
    return destino


def baixar_voz_padrao() -> Path:
    """Baixa (uma vez) a voz feminina pt-BR da demo oficial da Resemble."""
    if not VOZ_PADRAO.exists():
        VOZ_PADRAO.parent.mkdir(parents=True, exist_ok=True)
        torch.hub.download_url_to_file(VOZ_PADRAO_URL, str(VOZ_PADRAO), progress=False)
    return VOZ_PADRAO


def carregar(device: str = "cuda") -> ChatterboxMultilingualTTS:
    """Igual ao from_local do pacote, com o vocabulário lido do próprio checkpoint pt-BR."""
    ckpt = montar_checkpoint()

    ve = VoiceEncoder()
    ve.load_state_dict(torch.load(ckpt / "ve.pt", weights_only=True, map_location="cpu"))
    ve.to(device).eval()

    estado_t3 = load_safetensors(ckpt / "t3_pt_br.safetensors")
    if "model" in estado_t3.keys():
        estado_t3 = estado_t3["model"][0]
    configuracao = T3Config.multilingual()
    configuracao.text_tokens_dict_size = estado_t3["text_emb.weight"].shape[0]
    t3 = T3(configuracao)
    t3.load_state_dict(estado_t3)
    t3.to(device).eval()

    s3gen = S3Gen()
    faltando, sobrando = s3gen.load_state_dict(
        torch.load(ckpt / "s3gen.pt", weights_only=True, map_location="cpu"), strict=False)
    if set(faltando) - BUFFERS_AUSENTES_S3GEN or sobrando:
        raise RuntimeError(f"Checkpoint S3Gen inesperado: faltando={faltando}, sobrando={sobrando}")
    s3gen.to(device).eval()

    tokenizer = MTLTokenizer(str(ckpt / "grapheme_mtl_merged_expanded_v1.json"))
    conds = Conditionals.load(ckpt / "conds.pt", map_location="cpu").to(device)
    return ChatterboxMultilingualTTS(t3, s3gen, ve, tokenizer, device, conds=conds)
```

- [ ] **Step 2: Escrever `baixar_pesos.py`**

```python
"""Roda na montagem da imagem: baixa os pesos do pacote pt-BR e a voz padrão."""
import ptbr

if __name__ == "__main__":
    print("checkpoint:", ptbr.montar_checkpoint(), flush=True)
    print("voz padrão:", ptbr.baixar_voz_padrao(), flush=True)
```

- [ ] **Step 3: Rodar o download (usa o cache do Hugging Face que já existe)**

Run: `$PY baixar_pesos.py`
Expected: imprime `checkpoint: .../modelos/pt-br` e `voz padrão: .../modelos/vozes/pt_br_f2.wav`; `ls -L modelos/pt-br` lista os 6 arquivos.

- [ ] **Step 4: Carregar na GPU**

Run: `$PY -c "import ptbr; m = ptbr.carregar(); print(type(m).__name__, m.sr)"`
Expected: `ChatterboxMultilingualTTS 24000`

- [ ] **Step 5: Commit**

```bash
git add ptbr.py baixar_pesos.py
git commit -m "feat: carregamento do modelo pt-BR e download dos pesos" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: Serviço e entrada do RunPod (`handler.py`)

**Files:**
- Create: `handler.py`
- Test: `tests/test_handler.py`

**Interfaces:**
- Consumes:
  - de `sintese`: `interpretar`, `obter_voz`, `sufixo_audio`, `dividir_frases`, `juntar`, `wav_base64` e `EntradaInvalida` (Tasks 1–3);
  - de `ptbr`: `carregar` e `baixar_voz_padrao` (Task 4).
- Produces:
  - `handler.Servico(modelo, voz_padrao, carga_s: float, gpu: str)`, com `.atender(entrada) -> dict`;
  - `handler.iniciar(device: str = "cuda") -> Servico`;
  - `handler.handler(job: dict) -> dict`;
  - a global `handler.SERVICO`.
- **Contrato do modelo:**
  - atributos `sr: int` e `conds`;
  - `generate(texto, language_id=..., exaggeration=..., cfg_weight=..., temperature=...)`, que devolve um tensor ou array 1-D ou (1, N);
  - `prepare_conditionals(caminho: str, exaggeration=...)`.

- [ ] **Step 1: Escrever os testes que falham**

`tests/test_handler.py`:
```python
import base64
import io

import numpy as np
import pytest
import soundfile as sf

import handler as worker


class ModeloFalso:
    sr = 24000

    def __init__(self, falhar_em=None):
        self.conds = "padrao"
        self.chamadas = []
        self.vozes = []
        self.falhar_em = falhar_em

    def generate(self, texto, language_id, exaggeration, cfg_weight, temperature):
        if self.falhar_em and self.falhar_em in texto:
            raise RuntimeError("explodiu")
        self.chamadas.append({"texto": texto, "language_id": language_id, "exaggeration": exaggeration,
                              "cfg_weight": cfg_weight, "temperature": temperature, "conds": self.conds})
        return np.full(self.sr // 2, 0.1, dtype=np.float32)

    def prepare_conditionals(self, caminho, exaggeration=0.5):
        with open(caminho, "rb") as arquivo:
            dados = arquivo.read()
        if dados.startswith(b"ruim"):
            raise RuntimeError("formato não reconhecido")
        self.vozes.append(dados)
        self.conds = "propria"


def servico(modelo=None):
    return worker.Servico(modelo or ModeloFalso(), "padrao", carga_s=6.5, gpu="GPU de teste")


def b64(dados: bytes) -> str:
    return base64.b64encode(dados).decode()


def test_health_check_nao_gera_audio():
    s = servico()
    assert s.atender({"health_check": True}) == {"status": "healthy", "model_loaded": True, "gpu": "GPU de teste"}
    assert s.modelo.chamadas == []


def test_gera_wav_com_parametros_da_emocao_e_metricas():
    s = servico()
    r = s.atender({"text": "Primeira frase. Segunda frase!", "emotion": "happy"})
    audio, sr = sf.read(io.BytesIO(base64.b64decode(r["audio_base64"])))
    assert sr == 24000 and r["sample_rate"] == 24000
    assert len(audio) == 12000 + 3600 + 12000
    assert r["duration_seconds"] == pytest.approx(1.15)
    assert r["emotion"] == "alegre" and r["text"] == "Primeira frase. Segunda frase!"
    assert [c["texto"] for c in s.modelo.chamadas] == ["Primeira frase.", "Segunda frase!"]
    assert all(c["language_id"] == "pt" and c["exaggeration"] == 0.8 and c["cfg_weight"] == 0.3
               and c["temperature"] == 0.8 for c in s.modelo.chamadas)
    m = r["metrics"]
    assert m["sentences"] == 2 and m["gpu"] == "GPU de teste" and m["model_load_seconds"] == 6.5
    assert m["cold_start"] is True
    assert 0 <= m["first_sentence_seconds"] <= m["generation_seconds"]
    assert m["rtf"] is not None and m["rtf"] >= 0


def test_cold_start_so_no_primeiro_pedido():
    s = servico()
    assert s.atender({"text": "Oi."})["metrics"]["cold_start"] is True
    assert s.atender({"text": "Oi."})["metrics"]["cold_start"] is False


def test_etiqueta_define_emocao_e_sai_do_texto():
    s = servico()
    r = s.atender({"text": "[triste] Que pena."})
    assert r["emotion"] == "triste" and r["text"] == "Que pena."
    assert s.modelo.chamadas[0]["texto"] == "Que pena." and s.modelo.chamadas[0]["exaggeration"] == 0.35


def test_entrada_invalida_devolve_erro_sem_gerar():
    s = servico()
    r = s.atender({"text": "Oi.", "emotion": "raiva"})
    assert "Emoção desconhecida" in r["error"]
    assert s.modelo.chamadas == []


def test_voz_propria_em_base64_e_volta_para_padrao_depois():
    s = servico()
    s.atender({"text": "Com a minha voz.", "reference_audio_base64": b64(b"RIFF voz")})
    assert s.modelo.vozes == [b"RIFF voz"] and s.modelo.chamadas[0]["conds"] == "propria"
    s.atender({"text": "Com a voz padrão."})
    assert s.modelo.chamadas[1]["conds"] == "padrao"


def test_voz_ilegivel_vira_erro_e_nao_contamina_o_proximo_pedido():
    s = servico()
    r = s.atender({"text": "Oi.", "reference_audio_base64": b64(b"ruim: isto e HTML")})
    assert r["error"].startswith("Não foi possível ler a voz de referência")
    r2 = s.atender({"text": "Oi de novo."})
    assert "error" not in r2 and s.modelo.chamadas[-1]["conds"] == "padrao"


def test_falha_inesperada_nao_derruba_o_worker():
    s = servico(ModeloFalso(falhar_em="explode"))
    assert s.atender({"text": "Isto explode."})["error"].startswith("Falha inesperada na geração")
    assert "error" not in s.atender({"text": "Isto funciona."})


def test_seed_e_aplicada_antes_de_cada_frase(monkeypatch):
    sementes = []
    monkeypatch.setattr(worker.torch, "manual_seed", lambda semente: sementes.append(semente))
    servico().atender({"text": "Uma. Duas.", "seed": 7})
    assert sementes == [7, 7]


def test_handler_repassa_o_input_do_job(monkeypatch):
    monkeypatch.setattr(worker, "SERVICO", servico())
    assert worker.handler({"id": "x", "input": {"health_check": True}})["status"] == "healthy"
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `$PY -m pytest tests/test_handler.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'handler'`

- [ ] **Step 3: Implementar o mínimo**

`handler.py`:
```python
"""Worker serverless do RunPod: fala em pt-BR com emoção (Chatterbox Multilingual V3 + pacote pt-BR)."""
import os
import tempfile
import time
import traceback

import numpy as np
import torch

import sintese

SERVICO = None


def _para_numpy(wav) -> np.ndarray:
    if hasattr(wav, "detach"):
        wav = wav.detach().cpu().numpy()
    return np.asarray(wav, dtype=np.float32).reshape(-1)


class Servico:
    """Atende as tarefas com um modelo já carregado (o real, ou um falso nos testes)."""

    def __init__(self, modelo, voz_padrao, carga_s: float, gpu: str):
        self.modelo = modelo
        self.voz_padrao = voz_padrao
        self.carga_s = carga_s
        self.gpu = gpu
        self.atendidos = 0

    def atender(self, entrada) -> dict:
        if isinstance(entrada, dict) and entrada.get("health_check"):
            return {"status": "healthy", "model_loaded": True, "gpu": self.gpu}
        primeiro = self.atendidos == 0
        self.atendidos += 1
        try:
            pedido = sintese.interpretar(entrada)
            self._preparar_voz(pedido)
            return self._gerar(pedido, primeiro)
        except sintese.EntradaInvalida as erro:
            return {"error": str(erro)}
        except Exception as erro:
            traceback.print_exc()
            return {"error": f"Falha inesperada na geração: {erro}"}

    def _preparar_voz(self, pedido: sintese.Pedido) -> None:
        # Volta sempre à voz padrão; o generate só troca a intensidade dentro dela.
        self.modelo.conds = self.voz_padrao
        dados = sintese.obter_voz(pedido)
        if dados is None:
            return
        with tempfile.NamedTemporaryFile(suffix=sintese.sufixo_audio(dados), delete=False) as arquivo:
            arquivo.write(dados)
        try:
            self.modelo.prepare_conditionals(arquivo.name, exaggeration=pedido.exaggeration)
        except Exception as erro:
            self.modelo.conds = self.voz_padrao
            raise sintese.EntradaInvalida(f"Não foi possível ler a voz de referência: {erro}") from erro
        finally:
            os.unlink(arquivo.name)

    def _gerar(self, pedido: sintese.Pedido, primeiro: bool) -> dict:
        frases = sintese.dividir_frases(pedido.texto)
        trechos, tempos = [], []
        for frase in frases:
            if pedido.seed is not None:
                torch.manual_seed(pedido.seed)
            inicio = time.perf_counter()
            wav = self.modelo.generate(frase, language_id="pt", exaggeration=pedido.exaggeration,
                                       cfg_weight=pedido.cfg_weight, temperature=pedido.temperature)
            tempos.append(time.perf_counter() - inicio)
            trechos.append(_para_numpy(wav))
        sr = self.modelo.sr
        audio = sintese.juntar(trechos, sr)
        duracao = len(audio) / sr
        geracao = sum(tempos)
        return {
            "audio_base64": sintese.wav_base64(audio, sr),
            "sample_rate": sr,
            "duration_seconds": round(duracao, 3),
            "text": pedido.texto,
            "emotion": pedido.emocao,
            "metrics": {
                "generation_seconds": round(geracao, 3),
                "first_sentence_seconds": round(tempos[0], 3),
                "rtf": round(geracao / duracao, 3) if duracao else None,
                "sentences": len(frases),
                "gpu": self.gpu,
                "cold_start": primeiro,
                "model_load_seconds": round(self.carga_s, 2),
            },
        }


def iniciar(device: str = "cuda") -> Servico:
    """Carrega o modelo, prepara a voz padrão e aquece a GPU; uma vez por worker."""
    import ptbr

    inicio = time.perf_counter()
    modelo = ptbr.carregar(device)
    carga_s = time.perf_counter() - inicio
    modelo.prepare_conditionals(str(ptbr.baixar_voz_padrao()), exaggeration=0.5)
    modelo.generate("Aquecendo a placa de vídeo.", language_id="pt")
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    print(f"[worker] modelo pronto em {carga_s:.1f} s na {gpu}", flush=True)
    return Servico(modelo, modelo.conds, carga_s, gpu)


def handler(job: dict) -> dict:
    return SERVICO.atender(job.get("input"))


if __name__ == "__main__":
    import runpod

    SERVICO = iniciar()
    runpod.serverless.start({"handler": handler})
```

- [ ] **Step 4: Rodar e ver passar**

Run: `$PY -m pytest tests -q`
Expected: PASS (Tasks 1–5)

- [ ] **Step 5: Commit**

```bash
git add handler.py tests/test_handler.py
git commit -m "feat: serviço e handler do RunPod" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: Imagem (`requirements.txt`, `Dockerfile`, `.dockerignore`)

**Files:**
- Create: `requirements.txt`, `Dockerfile`, `.dockerignore`

**Interfaces:**
- Consumes: `baixar_pesos.py` (Task 4) e `handler.py` (Task 5).
- Produces: a imagem que o RunPod compila a partir do GitHub, com `CMD ["python", "-u", "handler.py"]`.

- [ ] **Step 1: Escrever os arquivos**

`requirements.txt`:
```
chatterbox-tts @ git+https://github.com/resemble-ai/chatterbox.git@5de7a54aa4e5e2baadb0182dde554908b48b85c2
runpod==1.12.0
soundfile
```

`Dockerfile`:
```dockerfile
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models/hf \
    CHATTERBOX_MODELOS=/models/chatterbox

RUN apt-get update \
    && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

# Pesos e voz padrão dentro da imagem: o worker liga sem internet.
COPY ptbr.py baixar_pesos.py ./
RUN python baixar_pesos.py

COPY sintese.py handler.py ./
ENV HF_HUB_OFFLINE=1
CMD ["python", "-u", "handler.py"]
```

`.dockerignore`:
```
.git
.gitignore
.pytest_cache
__pycache__
docs
tests
scripts
modelos
saida_local
saida_runpod
README.md
```

- [ ] **Step 2: Verificar que as dependências resolvem para Python 3.11**

Run: `uv pip compile requirements.txt --python-version 3.11 -o /tmp/chatterbox-worker-lock.txt -q && grep -E "^(torch|runpod|transformers)==" /tmp/chatterbox-worker-lock.txt`
Expected: `runpod==1.12.0`, `torch==2.6.0` e `transformers==5.2.0`

- [ ] **Step 3: Verificar a partida sem internet, como na imagem**

Run: `HF_HUB_OFFLINE=1 $PY -c "import handler; s = handler.iniciar(); print(s.atender({'health_check': True}))"`
Expected: `{'status': 'healthy', 'model_loaded': True, 'gpu': 'NVIDIA GeForce RTX 3060 Ti'}`

- [ ] **Step 4: Commit**

```bash
git add requirements.txt Dockerfile .dockerignore
git commit -m "build: imagem do worker com os pesos embutidos" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 7: Integração local com o modelo real e conferência da pronúncia

**Files:**
- Create: `scripts/integracao_local.py`, `scripts/transcrever.py`

**Interfaces:**
- Consumes: `handler.iniciar` e `Servico.atender` (Task 5); `ptbr.baixar_voz_padrao` (Task 4).
- Produces:
  - `saida_local/*.wav` e `saida_local/manifesto.json`, uma lista de `{"arquivo", "texto"}`;
  - `scripts/transcrever.py <pasta> [limite]`, que sai com código 0 se o CER médio ficar abaixo do limite e 1 caso contrário.

- [ ] **Step 1: Escrever `scripts/integracao_local.py`**

```python
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
```

- [ ] **Step 2: Escrever `scripts/transcrever.py`**

```python
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
```

- [ ] **Step 3: Rodar a integração na GPU**

Run: `$PY scripts/integracao_local.py`
Expected: três linhas (`01_neutro`, `02_alegre`, `03_voz_base64`) com duração > 0; `saida_local/` com 3 WAVs e `manifesto.json`

- [ ] **Step 4: Conferir a pronúncia**

Run: `env -u LD_LIBRARY_PATH uv run --python 3.11 --with sherpa-onnx --with soundfile python scripts/transcrever.py saida_local`
Expected: `CER médio` abaixo de 5% e código de saída 0

- [ ] **Step 5: Testar a porta de entrada do RunPod localmente**

Run: `$PY handler.py --test_input '{"input": {"text": "[alegre] Teste local do worker."}}' 2>&1 | grep -E -o "emotion.{1,4}alegre" | head -1`
Expected: uma linha com `emotion` seguido de `alegre`. Isso prova que o `runpod.serverless.start` chamou o handler.

- [ ] **Step 6: Commit**

```bash
git add scripts/integracao_local.py scripts/transcrever.py
git commit -m "test: integração local na GPU e conferência da pronúncia" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 8: Documentação, script de teste do endpoint e publicação no GitHub

**Files:**
- Create: `scripts/testar_endpoint.py`, `README.md`

**Interfaces:**
- Consumes: o formato de entrada e saída das Tasks 1–5.
- Produces:
  - `scripts/testar_endpoint.py ENDPOINT_ID`, que grava `saida_runpod/ENDPOINT_ID/*.wav` e `manifesto.json`, no mesmo formato da Task 7;
  - o repositório privado `danilocoppi/chatterbox-ptbr-worker`.

- [ ] **Step 1: Escrever `scripts/testar_endpoint.py`** (só biblioteca padrão)

```python
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
```

Run: `python3 -m py_compile scripts/testar_endpoint.py`
Expected: sem saída (compila)

- [ ] **Step 2: Escrever o `README.md`**

````markdown
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
````

- [ ] **Step 3: Rodar todos os testes**

Run: `$PY -m pytest tests -q`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add scripts/testar_endpoint.py README.md
git commit -m "docs: README e teste do endpoint no RunPod" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

- [ ] **Step 5: Criar o repositório privado e enviar**

Run: `git status --porcelain` (deve estar vazio) e depois `gh repo create danilocoppi/chatterbox-ptbr-worker --private --source . --remote origin --push`
Expected: repositório criado; `gh repo view danilocoppi/chatterbox-ptbr-worker --json visibility,url` mostra `"visibility":"PRIVATE"`

---

### Task 9: Publicação no RunPod e teste na nuvem

**Files:** nenhum arquivo novo. As saídas vão para `saida_runpod/`, que é ignorada pelo git.

**Interfaces:**
- Consumes: o repositório publicado (Task 8), `scripts/testar_endpoint.py` (Task 8) e `scripts/transcrever.py` (Task 7).
- Produces: as medidas da nuvem comparadas com as da RTX 3060 Ti.

- [ ] **Step 1 (usuário): criar o endpoint.** Siga os passos da seção "Publicar no RunPod" do README e informe:
  - o ID do novo endpoint;
  - uma chave de API com acesso a ele, salva em `~/.config/runpod/api_key`. A chave atual só acessa o endpoint antigo.

- [ ] **Step 2: Conferir o estado do endpoint**

Run: `curl -s -H "Authorization: Bearer $(cat ~/.config/runpod/api_key)" https://api.runpod.ai/v2/ENDPOINT_ID/health`
Expected: JSON com `jobs` e `workers`; sem erro de autorização

- [ ] **Step 3: Rodar as tarefas na nuvem**

Run: `python3 scripts/testar_endpoint.py ENDPOINT_ID`
Expected: 6 linhas sem erro (3 tarefas e 3 repetições quentes), com `frio=True` só na primeira

- [ ] **Step 4: Conferir a pronúncia na nuvem**

Run: `env -u LD_LIBRARY_PATH uv run --python 3.11 --with sherpa-onnx --with soundfile python scripts/transcrever.py saida_runpod/ENDPOINT_ID`
Expected: `CER médio` abaixo de 5% e código de saída 0

- [ ] **Step 5: Relatar ao usuário** uma tabela com GPU, arranque frio (fila e arranque), geração, primeira frase e RTF nas repetições quentes, lado a lado com a RTX 3060 Ti (Task 7). Lembrar de deixar o mínimo de workers em 0 e de trocar a chave de API exposta no chat.
