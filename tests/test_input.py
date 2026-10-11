import re

import pytest

from synthesis import InvalidInput, parse_input


def test_defaults_without_emotion():
    r = parse_input({"text": "  Olá, tudo bem?  "})
    assert r.text == "Olá, tudo bem?" and r.emotion == "neutral"
    assert (r.exaggeration, r.cfg_weight, r.temperature, r.seed) == (0.5, 0.5, 0.8, None)
    assert r.voice_url is None and r.voice_base64 is None


@pytest.mark.parametrize("name, expected", [
    ("happy", "happy"), ("HAPPY", "happy"), ("alegre", "happy"), ("Empático", "empathetic"),
    ("empathetic", "empathetic"), ("sussurro", "whisper"), ("  calm ", "calm"),
])
def test_emotion_accepts_portuguese_names_case_and_accents(name, expected):
    assert parse_input({"text": "Oi.", "emotion": name}).emotion == expected


def test_emotion_sets_exaggeration_and_cfg():
    r = parse_input({"text": "Oi.", "emotion": "excited"})
    assert (r.exaggeration, r.cfg_weight) == (0.9, 0.3)


def test_numbers_override_the_emotion():
    r = parse_input({"text": "Oi.", "emotion": "sad", "exaggeration": 1.2,
                     "cfg_weight": 0, "temperature": 0.5, "seed": 42})
    assert (r.exaggeration, r.cfg_weight, r.temperature, r.seed) == (1.2, 0.0, 0.5, 42)


def test_leading_tag_sets_emotion_and_is_removed():
    r = parse_input({"text": "[triste] Que pena."})
    assert r.emotion == "sad" and r.text == "Que pena."


def test_field_wins_over_tag_and_tag_is_still_removed():
    r = parse_input({"text": "[sad] Que pena.", "emotion": "calmo"})
    assert r.emotion == "calm" and r.text == "Que pena."


def test_empty_emotion_field_uses_the_tag():
    assert parse_input({"text": "[happy] Oba!", "emotion": ""}).emotion == "happy"


def test_bracketed_number_is_spoken_as_text():
    r = parse_input({"text": "[1] Primeiro item."})
    assert r.text == "[1] Primeiro item." and r.emotion == "neutral"


def test_url_wins_over_base64():
    r = parse_input({"text": "Oi.", "reference_audio_url": "https://x/v.wav",
                     "reference_audio_base64": "QUJD"})
    assert r.voice_url == "https://x/v.wav" and r.voice_base64 is None


def test_speed_1_is_accepted():
    assert parse_input({"text": "Oi.", "speed": 1.0}).text == "Oi."


@pytest.mark.parametrize("job_input, fragment", [
    ("not a dict", "JSON object"),
    ({}, "Provide the text"),
    ({"text": "   "}, "Provide the text"),
    ({"text": "[happy]"}, "after the emotion tag"),
    ({"text": "[happy]    "}, "after the emotion tag"),
    ({"text": "x" * 1001}, "the maximum is 1000"),
    ({"text": "Oi.", "emotion": "raiva"}, "Unknown emotion: 'raiva'"),
    ({"text": "[raiva] Oi."}, "Unknown emotion"),
    ({"text": "Oi.", "emotion": 3}, "'emotion' must be a string"),
    ({"text": "Oi.", "exaggeration": "0.8"}, "'exaggeration' must be a number"),
    ({"text": "Oi.", "exaggeration": True}, "'exaggeration' must be a number"),
    ({"text": "Oi.", "exaggeration": 3}, "'exaggeration' must be between 0.25 and 2.0"),
    ({"text": "Oi.", "cfg_weight": 1.5}, "'cfg_weight' must be between 0.0 and 1.0"),
    ({"text": "Oi.", "temperature": 0}, "'temperature' must be between 0.05 and 2.0"),
    ({"text": "Oi.", "speed": 1.5}, "'speed' is not supported"),
    ({"text": "Oi.", "seed": -1}, "'seed' must be an integer"),
    ({"text": "Oi.", "seed": 1.5}, "'seed' must be an integer"),
    ({"text": "Oi.", "reference_audio_url": "ftp://x/voz.wav"}, "http(s) URL"),
    ({"text": "Oi.", "reference_audio_base64": 123}, "base64 string"),
])
def test_invalid_inputs(job_input, fragment):
    with pytest.raises(InvalidInput, match=re.escape(fragment)):
        parse_input(job_input)
