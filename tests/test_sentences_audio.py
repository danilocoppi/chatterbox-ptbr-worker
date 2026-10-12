import base64
import io

import numpy as np
import pytest
import soundfile as sf

import synthesis


def test_splits_on_sentence_punctuation():
    assert synthesis.split_sentences("Oi! Tudo bem? Sim. Até mais… Tchau") == [
        "Oi!", "Tudo bem?", "Sim.", "Até mais…", "Tchau"]


def test_long_sentence_breaks_at_commas_without_losing_text():
    sentence = ", ".join(["palavra de teste numero"] * 20) + "."
    pieces = synthesis.split_sentences(sentence)
    assert len(pieces) > 1 and all(len(p) <= synthesis.MAX_CHUNK_CHARS for p in pieces)
    assert " ".join(pieces).split() == sentence.split()


def test_text_without_punctuation_breaks_between_words():
    text = " ".join(["palavra"] * 100)
    pieces = synthesis.split_sentences(text)
    assert len(pieces) > 1 and all(len(p) <= synthesis.MAX_CHUNK_CHARS for p in pieces)
    assert " ".join(pieces).split() == text.split()


def test_giant_word_is_cut():
    text = "Veja " + "x" * 700 + " agora."
    pieces = synthesis.split_sentences(text)
    assert all(len(p) <= synthesis.MAX_CHUNK_CHARS for p in pieces)
    assert "".join(pieces).replace(" ", "") == text.replace(" ", "")


def test_short_text_is_one_chunk():
    text = "Oi, tudo bem? Hoje o tempo está ótimo pra ir à praia de Copacabana."
    assert synthesis.split_chunks(text) == [text]


def test_long_text_is_grouped_into_chunks_without_losing_text():
    text = " ".join(f"Esta é a frase número {i}, com algumas palavras a mais." for i in range(18))
    chunks = synthesis.split_chunks(text)
    assert 1 < len(chunks) < len(synthesis.split_sentences(text))
    assert all(len(c) <= synthesis.MAX_CHUNK_CHARS for c in chunks)
    assert " ".join(chunks).split() == text.split()


def test_join_audio_puts_silence_between_chunks():
    audio = synthesis.join_audio([np.ones(10), np.ones(20)], sr=1000, pause_s=0.15)
    assert audio.dtype == np.float32 and len(audio) == 10 + 150 + 20
    assert audio[10:160].max() == 0.0


def _tone_then_silence(level_dbfs: float, sr: int = 24000) -> np.ndarray:
    """Half a second of a 220 Hz tone with the given RMS level, then half a second of silence."""
    t = np.arange(sr // 2) / sr
    tone = np.sqrt(2) * 10 ** (level_dbfs / 20) * np.sin(2 * np.pi * 220 * t)
    return np.concatenate([tone, np.zeros(sr // 2)]).astype(np.float32)


def _rms_dbfs(signal: np.ndarray) -> float:
    return 20 * np.log10(np.sqrt(np.mean(signal.astype(np.float64) ** 2)))


@pytest.mark.parametrize("level", [-35.0, -8.0])
def test_speech_is_brought_to_the_target_level(level):
    audio = synthesis.normalize_loudness(_tone_then_silence(level), 24000)
    assert audio.dtype == np.float32
    assert _rms_dbfs(audio[:12000]) == pytest.approx(synthesis.TARGET_DBFS, abs=0.1)
    assert not np.any(audio[12000:])


def test_gain_stops_at_the_peak_limit():
    audio = _tone_then_silence(-35.0)
    audio[100] = 0.5  # a click far louder than the rest
    out = synthesis.normalize_loudness(audio, 24000)
    assert np.abs(out).max() == pytest.approx(synthesis.PEAK_LIMIT, abs=1e-6)


def test_silence_stays_silent():
    out = synthesis.normalize_loudness(np.zeros(2400, dtype=np.float32), 24000)
    assert out.dtype == np.float32 and not np.any(out)


def test_wav_base64_round_trip_with_clipping():
    sr = 24000
    audio = np.linspace(-1.5, 1.5, sr // 10, dtype=np.float32)
    decoded, decoded_sr = sf.read(io.BytesIO(base64.b64decode(synthesis.wav_base64(audio, sr))), dtype="float32")
    assert decoded_sr == sr and len(decoded) == len(audio)
    assert np.allclose(decoded, np.clip(audio, -1.0, 1.0), atol=1e-3)
