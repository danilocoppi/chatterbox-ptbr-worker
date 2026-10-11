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
