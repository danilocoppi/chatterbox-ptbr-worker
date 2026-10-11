"""Lógica do worker sem GPU: entrada, emoção, frases, áudio e voz de referência."""
import base64
import io
import re
import time
import unicodedata
import urllib.request
from dataclasses import dataclass
from typing import Optional

import numpy as np
import soundfile as sf

MAX_CARACTERES = 1000
TEMPERATURA_PADRAO = 0.8
MAX_FRASE = 300
PAUSA_ENTRE_FRASES_S = 0.15
MAX_BYTES_VOZ = 10 * 1024 * 1024
TEMPO_MAX_DOWNLOAD_S = 15.0

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
_FIM_DE_FRASE = re.compile(r"(?<=[.!?…])\s+")
_VIRGULA = re.compile(r"(?<=[,;])\s+")


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
