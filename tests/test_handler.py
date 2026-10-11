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
