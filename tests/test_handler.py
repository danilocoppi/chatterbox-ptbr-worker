import base64
import io

import numpy as np
import pytest
import soundfile as sf

import handler as worker


class FakeModel:
    sr = 24000

    def __init__(self, fail_on=None):
        self.conds = "default"
        self.calls = []
        self.voices = []
        self.fail_on = fail_on

    def generate(self, text, language_id, exaggeration, cfg_weight, temperature):
        if self.fail_on and self.fail_on in text:
            raise RuntimeError("boom")
        self.calls.append({"text": text, "language_id": language_id, "exaggeration": exaggeration,
                           "cfg_weight": cfg_weight, "temperature": temperature, "conds": self.conds})
        return np.full(self.sr // 2, 0.1, dtype=np.float32)

    def prepare_conditionals(self, path, exaggeration=0.5):
        with open(path, "rb") as f:
            data = f.read()
        if data.startswith(b"bad"):
            raise RuntimeError("unrecognized format")
        self.voices.append(data)
        self.conds = "custom"


def service(model=None):
    return worker.Service(model or FakeModel(), "default", load_seconds=6.5, gpu="Test GPU")


def b64(data: bytes) -> str:
    return base64.b64encode(data).decode()


def test_health_check_does_not_generate():
    s = service()
    assert s.handle({"health_check": True}) == {"status": "healthy", "model_loaded": True, "gpu": "Test GPU"}
    assert s.model.calls == []


def test_generates_wav_with_emotion_parameters_and_metrics():
    s = service()
    r = s.handle({"text": "Primeira frase. Segunda frase!", "emotion": "alegre"})
    audio, sr = sf.read(io.BytesIO(base64.b64decode(r["audio_base64"])))
    assert sr == 24000 and r["sample_rate"] == 24000
    assert len(audio) == 12000 + 3600 + 12000
    assert r["duration_seconds"] == pytest.approx(1.15)
    assert r["emotion"] == "happy" and r["text"] == "Primeira frase. Segunda frase!"
    assert [c["text"] for c in s.model.calls] == ["Primeira frase.", "Segunda frase!"]
    assert all(c["language_id"] == "pt" and c["exaggeration"] == 0.8 and c["cfg_weight"] == 0.3
               and c["temperature"] == 0.8 for c in s.model.calls)
    m = r["metrics"]
    assert m["sentences"] == 2 and m["gpu"] == "Test GPU" and m["model_load_seconds"] == 6.5
    assert m["cold_start"] is True
    assert 0 <= m["first_sentence_seconds"] <= m["generation_seconds"]
    assert m["rtf"] is not None and m["rtf"] >= 0


def test_cold_start_only_on_first_request():
    s = service()
    assert s.handle({"text": "Oi."})["metrics"]["cold_start"] is True
    assert s.handle({"text": "Oi."})["metrics"]["cold_start"] is False


def test_tag_sets_emotion_and_is_removed():
    s = service()
    r = s.handle({"text": "[triste] Que pena."})
    assert r["emotion"] == "sad" and r["text"] == "Que pena."
    assert s.model.calls[0]["text"] == "Que pena." and s.model.calls[0]["exaggeration"] == 0.35


def test_invalid_input_returns_error_without_generating():
    s = service()
    r = s.handle({"text": "Oi.", "emotion": "raiva"})
    assert "Unknown emotion" in r["error"]
    assert s.model.calls == []


def test_custom_base64_voice_then_back_to_default():
    s = service()
    s.handle({"text": "Com a minha voz.", "reference_audio_base64": b64(b"RIFF voice")})
    assert s.model.voices == [b"RIFF voice"] and s.model.calls[0]["conds"] == "custom"
    s.handle({"text": "Com a voz padrão."})
    assert s.model.calls[1]["conds"] == "default"


def test_unreadable_voice_is_an_error_and_does_not_leak_into_next_request():
    s = service()
    r = s.handle({"text": "Oi.", "reference_audio_base64": b64(b"bad: this is HTML")})
    assert r["error"].startswith("Could not read the reference voice")
    r2 = s.handle({"text": "Oi de novo."})
    assert "error" not in r2 and s.model.calls[-1]["conds"] == "default"


def test_unexpected_failure_does_not_kill_the_worker():
    s = service(FakeModel(fail_on="explode"))
    assert s.handle({"text": "Isto explode."})["error"].startswith("Unexpected generation failure")
    assert "error" not in s.handle({"text": "Isto funciona."})


def test_seed_is_applied_before_each_sentence(monkeypatch):
    seeds = []
    monkeypatch.setattr(worker.torch, "manual_seed", lambda seed: seeds.append(seed))
    service().handle({"text": "Uma. Duas.", "seed": 7})
    assert seeds == [7, 7]


def test_handler_passes_the_job_input(monkeypatch):
    monkeypatch.setattr(worker, "SERVICE", service())
    assert worker.handler({"id": "x", "input": {"health_check": True}})["status"] == "healthy"
