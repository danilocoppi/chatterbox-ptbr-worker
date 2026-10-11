import base64
import io

import numpy as np
import soundfile as sf

import synthesis


def test_splits_on_sentence_punctuation():
    assert synthesis.split_sentences("Oi! Tudo bem? Sim. Até mais… Tchau") == [
        "Oi!", "Tudo bem?", "Sim.", "Até mais…", "Tchau"]


def test_long_sentence_breaks_at_commas_without_losing_text():
    sentence = ", ".join(["palavra de teste numero"] * 20) + "."
    pieces = synthesis.split_sentences(sentence)
    assert len(pieces) > 1 and all(len(p) <= synthesis.MAX_SENTENCE_CHARS for p in pieces)
    assert " ".join(pieces).split() == sentence.split()


def test_text_without_punctuation_breaks_between_words():
    text = " ".join(["palavra"] * 100)
    pieces = synthesis.split_sentences(text)
    assert len(pieces) > 1 and all(len(p) <= synthesis.MAX_SENTENCE_CHARS for p in pieces)
    assert " ".join(pieces).split() == text.split()


def test_giant_word_is_cut():
    text = "Veja " + "x" * 700 + " agora."
    pieces = synthesis.split_sentences(text)
    assert all(len(p) <= synthesis.MAX_SENTENCE_CHARS for p in pieces)
    assert "".join(pieces).replace(" ", "") == text.replace(" ", "")


def test_join_audio_puts_silence_between_chunks():
    audio = synthesis.join_audio([np.ones(10), np.ones(20)], sr=1000, pause_s=0.15)
    assert audio.dtype == np.float32 and len(audio) == 10 + 150 + 20
    assert audio[10:160].max() == 0.0


def test_wav_base64_round_trip_with_clipping():
    sr = 24000
    audio = np.linspace(-1.5, 1.5, sr // 10, dtype=np.float32)
    decoded, decoded_sr = sf.read(io.BytesIO(base64.b64decode(synthesis.wav_base64(audio, sr))), dtype="float32")
    assert decoded_sr == sr and len(decoded) == len(audio)
    assert np.allclose(decoded, np.clip(audio, -1.0, 1.0), atol=1e-3)
