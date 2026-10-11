"""Worker serverless do RunPod: fala em pt-BR com emoção (Chatterbox Multilingual V3 + pacote pt-BR)."""
import os
import tempfile
import time
import traceback

import numpy as np
import torch

import sintese

SERVICO = None


def _para_numpy(wav) -> np.ndarray:
    if hasattr(wav, "detach"):
        wav = wav.detach().cpu().numpy()
    return np.asarray(wav, dtype=np.float32).reshape(-1)


class Servico:
    """Atende as tarefas com um modelo já carregado (o real, ou um falso nos testes)."""

    def __init__(self, modelo, voz_padrao, carga_s: float, gpu: str):
        self.modelo = modelo
        self.voz_padrao = voz_padrao
        self.carga_s = carga_s
        self.gpu = gpu
        self.atendidos = 0

    def atender(self, entrada) -> dict:
        if isinstance(entrada, dict) and entrada.get("health_check"):
            return {"status": "healthy", "model_loaded": True, "gpu": self.gpu}
        primeiro = self.atendidos == 0
        self.atendidos += 1
        try:
            pedido = sintese.interpretar(entrada)
            self._preparar_voz(pedido)
            return self._gerar(pedido, primeiro)
        except sintese.EntradaInvalida as erro:
            return {"error": str(erro)}
        except Exception as erro:
            traceback.print_exc()
            return {"error": f"Falha inesperada na geração: {erro}"}

    def _preparar_voz(self, pedido: sintese.Pedido) -> None:
        # Volta sempre à voz padrão; o generate só troca a intensidade dentro dela.
        self.modelo.conds = self.voz_padrao
        dados = sintese.obter_voz(pedido)
        if dados is None:
            return
        with tempfile.NamedTemporaryFile(suffix=sintese.sufixo_audio(dados), delete=False) as arquivo:
            arquivo.write(dados)
        try:
            self.modelo.prepare_conditionals(arquivo.name, exaggeration=pedido.exaggeration)
        except Exception as erro:
            self.modelo.conds = self.voz_padrao
            raise sintese.EntradaInvalida(f"Não foi possível ler a voz de referência: {erro}") from erro
        finally:
            os.unlink(arquivo.name)

    def _gerar(self, pedido: sintese.Pedido, primeiro: bool) -> dict:
        frases = sintese.dividir_frases(pedido.texto)
        trechos, tempos = [], []
        for frase in frases:
            if pedido.seed is not None:
                torch.manual_seed(pedido.seed)
            inicio = time.perf_counter()
            wav = self.modelo.generate(frase, language_id="pt", exaggeration=pedido.exaggeration,
                                       cfg_weight=pedido.cfg_weight, temperature=pedido.temperature)
            tempos.append(time.perf_counter() - inicio)
            trechos.append(_para_numpy(wav))
        sr = self.modelo.sr
        audio = sintese.juntar(trechos, sr)
        duracao = len(audio) / sr
        geracao = sum(tempos)
        return {
            "audio_base64": sintese.wav_base64(audio, sr),
            "sample_rate": sr,
            "duration_seconds": round(duracao, 3),
            "text": pedido.texto,
            "emotion": pedido.emocao,
            "metrics": {
                "generation_seconds": round(geracao, 3),
                "first_sentence_seconds": round(tempos[0], 3),
                "rtf": round(geracao / duracao, 3) if duracao else None,
                "sentences": len(frases),
                "gpu": self.gpu,
                "cold_start": primeiro,
                "model_load_seconds": round(self.carga_s, 2),
            },
        }


def iniciar(device: str = "cuda") -> Servico:
    """Carrega o modelo, prepara a voz padrão e aquece a GPU; uma vez por worker."""
    import ptbr

    inicio = time.perf_counter()
    modelo = ptbr.carregar(device)
    carga_s = time.perf_counter() - inicio
    modelo.prepare_conditionals(str(ptbr.baixar_voz_padrao()), exaggeration=0.5)
    modelo.generate("Aquecendo a placa de vídeo.", language_id="pt")
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    print(f"[worker] modelo pronto em {carga_s:.1f} s na {gpu}", flush=True)
    return Servico(modelo, modelo.conds, carga_s, gpu)


def handler(job: dict) -> dict:
    return SERVICO.atender(job.get("input"))


if __name__ == "__main__":
    import runpod

    SERVICO = iniciar()
    runpod.serverless.start({"handler": handler})
