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
