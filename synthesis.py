"""GPU-free worker logic: input parsing, emotions, text chunking, audio level and encoding, and the reference voice."""
import base64
import io
import ipaddress
import re
import socket
import time
import unicodedata
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Optional

import numpy as np
import soundfile as sf

MAX_CHARS = 1000
DEFAULT_TEMPERATURE = 0.8
# Nucleus sampling of the official pt-BR demo (Resemble's Space); the multilingual class would use 1.0.
TOP_P = 0.95
# One generation pass; the official demo also caps its text at 300 characters.
MAX_CHUNK_CHARS = 300
CHUNK_PAUSE_S = 0.15
# Speech level of the response: RMS of the voiced part, with the peak kept 1 dB below full scale.
TARGET_DBFS = -20.0
PEAK_LIMIT = 10 ** (-1 / 20)
MAX_VOICE_BYTES = 10 * 1024 * 1024
MAX_DOWNLOAD_S = 15.0

# emotion -> (exaggeration, cfg_weight)
EMOTIONS = {
    "neutral": (0.5, 0.5),
    "happy": (0.8, 0.3),
    "excited": (0.9, 0.3),
    "sad": (0.35, 0.5),
    "empathetic": (0.4, 0.5),
    "calm": (0.35, 0.5),
    "whisper": (0.3, 0.5),
}
# Portuguese names, accepted in the field and in the tag (accents are ignored).
SYNONYMS = {
    "neutro": "neutral", "alegre": "happy", "animado": "excited", "triste": "sad",
    "empatico": "empathetic", "calmo": "calm", "sussurro": "whisper",
}

_TAG = re.compile(r"^\s*\[([A-Za-zÀ-ÿ_]+)\]\s*")
_SENTENCE_END = re.compile(r"(?<=[.!?…])\s+")
_CLAUSE_END = re.compile(r"(?<=[,;])\s+")


class InvalidInput(ValueError):
    """Validation error; the message is returned to the client."""


@dataclass(frozen=True)
class SpeechRequest:
    text: str
    emotion: str
    exaggeration: float
    cfg_weight: float
    temperature: float
    seed: Optional[int]
    voice_url: Optional[str]
    voice_base64: Optional[str]


def _strip_accents(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")


def resolve_emotion(name: str) -> str:
    """Canonical emotion name; accepts the Portuguese names, any letter case and accents."""
    key = _strip_accents(str(name)).strip().lower()
    key = SYNONYMS.get(key, key)
    if key not in EMOTIONS:
        raise InvalidInput(f"Unknown emotion: '{name}'. Accepted: {', '.join(EMOTIONS)} "
                           f"(or in Portuguese: {', '.join(SYNONYMS)}).")
    return key


def _number(job_input: dict, field: str, low: float, high: float, default: float) -> float:
    value = job_input.get(field)
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InvalidInput(f"'{field}' must be a number.")
    if not low <= value <= high:
        raise InvalidInput(f"'{field}' must be between {low} and {high}.")
    return float(value)


def parse_input(job_input) -> SpeechRequest:
    """Validate the RunPod job input and resolve the emotion and the parameters."""
    if not isinstance(job_input, dict):
        raise InvalidInput("The input must be a JSON object.")
    text = job_input.get("text")
    if not isinstance(text, str) or not text.strip():
        raise InvalidInput("Provide the text to speak in 'text'.")

    tag_emotion = None
    tag = _TAG.match(text)
    if tag:
        tag_emotion = resolve_emotion(tag.group(1))
        text = text[tag.end():]
    text = text.strip()
    if not text:
        raise InvalidInput("There is no text to speak after the emotion tag.")
    if len(text) > MAX_CHARS:
        raise InvalidInput(f"The text has {len(text)} characters; the maximum is {MAX_CHARS}.")

    field = job_input.get("emotion")
    if field is not None and not isinstance(field, str):
        raise InvalidInput("'emotion' must be a string.")
    emotion = resolve_emotion(field) if field else (tag_emotion or "neutral")
    default_exaggeration, default_cfg = EMOTIONS[emotion]

    speed = job_input.get("speed")
    if speed is not None and speed != 1:
        raise InvalidInput("'speed' is not supported; send 1.0 or omit the field.")

    seed = job_input.get("seed")
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int) or seed < 0):
        raise InvalidInput("'seed' must be an integer greater than or equal to zero.")

    url = job_input.get("reference_audio_url") or None
    b64 = job_input.get("reference_audio_base64") or None
    if url is not None and (not isinstance(url, str) or not url.lower().startswith(("http://", "https://"))):
        raise InvalidInput("'reference_audio_url' must be an http(s) URL.")
    if b64 is not None and not isinstance(b64, str):
        raise InvalidInput("'reference_audio_base64' must be a base64 string.")

    return SpeechRequest(
        text=text,
        emotion=emotion,
        exaggeration=_number(job_input, "exaggeration", 0.25, 2.0, default_exaggeration),
        cfg_weight=_number(job_input, "cfg_weight", 0.0, 1.0, default_cfg),
        temperature=_number(job_input, "temperature", 0.05, 2.0, DEFAULT_TEMPERATURE),
        seed=seed,
        voice_url=url,
        voice_base64=None if url else b64,
    )


def _group(units: list[str]) -> list[str]:
    """Join units with spaces into pieces of at most MAX_CHUNK_CHARS characters."""
    pieces, current = [], ""
    for unit in units:
        if current and len(current) + 1 + len(unit) > MAX_CHUNK_CHARS:
            pieces.append(current)
            current = unit
        else:
            current = f"{current} {unit}" if current else unit
    if current:
        pieces.append(current)
    return pieces


def _words(text: str) -> list[str]:
    """Words of the text; a word longer than MAX_CHUNK_CHARS is cut into pieces of that size."""
    words = []
    for word in text.split():
        words.extend(word[i:i + MAX_CHUNK_CHARS] for i in range(0, len(word), MAX_CHUNK_CHARS))
    return words


def _split_long(sentence: str) -> list[str]:
    if len(sentence) <= MAX_CHUNK_CHARS:
        return [sentence]
    units = []
    for clause in _CLAUSE_END.split(sentence):
        units.extend([clause] if len(clause) <= MAX_CHUNK_CHARS else _group(_words(clause)))
    return _group(units)


def split_sentences(text: str) -> list[str]:
    """Split on . ! ? … and break pieces longer than MAX_CHUNK_CHARS at commas, then between words."""
    sentences = []
    for sentence in _SENTENCE_END.split(text.strip()):
        sentence = sentence.strip()
        if sentence:
            sentences.extend(_split_long(sentence))
    return sentences


def split_chunks(text: str) -> list[str]:
    """Whole sentences grouped into chunks of at most MAX_CHUNK_CHARS. A text that fits is generated in one pass,
    which keeps the intonation flowing from one sentence to the next."""
    return _group(split_sentences(text))


def join_audio(chunks: list, sr: int, pause_s: float = CHUNK_PAUSE_S) -> np.ndarray:
    """Concatenate the audio chunks with a silence between them."""
    silence = np.zeros(int(round(sr * pause_s)), dtype=np.float32)
    parts = []
    for i, chunk in enumerate(chunks):
        if i:
            parts.append(silence)
        parts.append(np.asarray(chunk, dtype=np.float32).reshape(-1))
    return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)


def normalize_loudness(audio: np.ndarray, sr: int, target_dbfs: float = TARGET_DBFS,
                       peak_limit: float = PEAK_LIMIT) -> np.ndarray:
    """Scale the audio so its voiced part has an RMS of target_dbfs, without letting the peak pass peak_limit.

    The model's raw output varies from take to take and usually sits 5–10 dB below normal speech. The level is
    measured over 20 ms frames, ignoring those more than 20 dB below the loudest (pauses and silence).
    """
    signal = np.asarray(audio, dtype=np.float32).reshape(-1)
    peak = float(np.abs(signal).max()) if signal.size else 0.0
    if peak == 0.0:
        return signal
    frame = max(1, sr // 50)
    frames = np.pad(signal.astype(np.float64), (0, -len(signal) % frame)).reshape(-1, frame)
    rms = np.sqrt((frames ** 2).mean(axis=1))
    voiced = frames[rms > rms.max() * 0.1]
    level = 20 * np.log10(np.sqrt((voiced ** 2).mean()))
    gain = min(10 ** ((target_dbfs - level) / 20), peak_limit / peak)
    return (signal * gain).astype(np.float32)


def wav_base64(audio: np.ndarray, sr: int) -> str:
    """Encode the audio as a 16-bit PCM mono WAV in base64."""
    buffer = io.BytesIO()
    signal = np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0)
    sf.write(buffer, signal, sr, format="WAV", subtype="PCM_16")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def decode_base64(value: str) -> bytes:
    """Decode the base64 reference voice; accepts a data: prefix and line breaks."""
    if value.startswith("data:"):
        value = value.split(",", 1)[-1]
    value = "".join(value.split())
    try:
        data = base64.b64decode(value, validate=True)
    except ValueError as err:
        raise InvalidInput("'reference_audio_base64' is not valid base64.") from err
    if not data:
        raise InvalidInput("The reference voice is empty.")
    if len(data) > MAX_VOICE_BYTES:
        raise InvalidInput("The reference voice exceeds 10 MB.")
    return data


def _check_destination(url: str) -> None:
    """Refuse URLs that are not http(s) or whose host resolves to an internal address."""
    parts = urllib.parse.urlsplit(url)
    if parts.scheme.lower() not in ("http", "https") or not parts.hostname:
        raise InvalidInput("'reference_audio_url' must be an http(s) URL.")
    for *_, address in socket.getaddrinfo(parts.hostname, None):
        ip = ipaddress.ip_address(address[0].split("%")[0])
        if ip.version == 6 and ip.ipv4_mapped:
            ip = ip.ipv4_mapped
        if not ip.is_global or ip.is_multicast:
            raise InvalidInput("'reference_audio_url' points to an internal address, which is not allowed.")


class _SafeRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Check the destination of every redirect before following it."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        _check_destination(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


_OPENER = urllib.request.build_opener(_SafeRedirectHandler)


def download_voice(url: str, max_seconds: float = MAX_DOWNLOAD_S) -> bytes:
    """Download the reference voice within the time and size limits."""
    start = time.monotonic()
    blocks, total = [], 0
    try:
        _check_destination(url)
        request = urllib.request.Request(url, headers={"User-Agent": "chatterbox-ptbr-worker"})
        with _OPENER.open(request, timeout=max_seconds) as response:
            # read1 returns whatever has arrived; read(n) would wait for n bytes and break the time limit.
            while block := response.read1(64 * 1024):
                total += len(block)
                if total > MAX_VOICE_BYTES:
                    raise InvalidInput("The reference voice exceeds 10 MB.")
                if time.monotonic() - start > max_seconds:
                    raise InvalidInput(f"Downloading the reference voice took more than {max_seconds:.0f} s.")
                blocks.append(block)
    except InvalidInput:
        raise
    except (OSError, ValueError) as err:
        raise InvalidInput(f"Could not download the reference voice: {err}") from err
    if not blocks:
        raise InvalidInput("The reference voice is empty.")
    return b"".join(blocks)


def get_voice(request: SpeechRequest) -> Optional[bytes]:
    """Reference voice bytes for the request, or None to use the default voice."""
    if request.voice_url:
        return download_voice(request.voice_url)
    if request.voice_base64:
        return decode_base64(request.voice_base64)
    return None


def audio_suffix(data: bytes) -> str:
    """Temporary file extension chosen from the content, for the audio reader."""
    if data[:4] == b"RIFF":
        return ".wav"
    if data[:4] == b"fLaC":
        return ".flac"
    if data[:4] == b"OggS":
        return ".ogg"
    if data[:3] == b"ID3" or data[:2] in (b"\xff\xfb", b"\xff\xf3", b"\xff\xf2"):
        return ".mp3"
    return ".audio"
