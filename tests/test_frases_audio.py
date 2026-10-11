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
