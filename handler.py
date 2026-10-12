"""RunPod serverless worker: Brazilian Portuguese speech with emotion (Chatterbox Multilingual V3 + pt-BR pack)."""
import os
import tempfile
import time
import traceback

import numpy as np
import torch

import synthesis

SERVICE = None


def _to_numpy(wav) -> np.ndarray:
    if hasattr(wav, "detach"):
        wav = wav.detach().cpu().numpy()
    return np.asarray(wav, dtype=np.float32).reshape(-1)


class Service:
    """Serves jobs with an already loaded model (the real one, or a fake one in the tests)."""

    def __init__(self, model, default_voice, load_seconds: float, gpu: str):
        self.model = model
        self.default_voice = default_voice
        self.load_seconds = load_seconds
        self.gpu = gpu
        self.served = 0

    def handle(self, job_input) -> dict:
        if isinstance(job_input, dict) and job_input.get("health_check"):
            return {"status": "healthy", "model_loaded": True, "gpu": self.gpu}
        first = self.served == 0
        self.served += 1
        try:
            request = synthesis.parse_input(job_input)
            self._prepare_voice(request)
            return self._generate(request, first)
        except synthesis.InvalidInput as err:
            return {"error": str(err)}
        except Exception as err:
            traceback.print_exc()
            return {"error": f"Unexpected generation failure: {err}"}

    def _prepare_voice(self, request: synthesis.SpeechRequest) -> None:
        # Always start from the default voice; generate() only swaps the intensity inside it.
        self.model.conds = self.default_voice
        data = synthesis.get_voice(request)
        if data is None:
            return
        with tempfile.NamedTemporaryFile(suffix=synthesis.audio_suffix(data), delete=False) as tmp:
            tmp.write(data)
        try:
            self.model.prepare_conditionals(tmp.name, exaggeration=request.exaggeration)
        except Exception as err:
            self.model.conds = self.default_voice
            raise synthesis.InvalidInput(f"Could not read the reference voice: {err}") from err
        finally:
            os.unlink(tmp.name)

    def _generate(self, request: synthesis.SpeechRequest, first: bool) -> dict:
        texts = synthesis.split_chunks(request.text)
        chunks, timings = [], []
        for text in texts:
            if request.seed is not None:
                torch.manual_seed(request.seed)
            start = time.perf_counter()
            wav = self.model.generate(text, language_id="pt", exaggeration=request.exaggeration,
                                      cfg_weight=request.cfg_weight, temperature=request.temperature,
                                      top_p=synthesis.TOP_P)
            timings.append(time.perf_counter() - start)
            chunks.append(_to_numpy(wav))
        sr = self.model.sr
        audio = synthesis.normalize_loudness(synthesis.join_audio(chunks, sr), sr)
        duration = len(audio) / sr
        generation = sum(timings)
        return {
            "audio_base64": synthesis.wav_base64(audio, sr),
            "sample_rate": sr,
            "duration_seconds": round(duration, 3),
            "text": request.text,
            "emotion": request.emotion,
            "metrics": {
                "generation_seconds": round(generation, 3),
                "first_chunk_seconds": round(timings[0], 3),
                "rtf": round(generation / duration, 3) if duration else None,
                "chunks": len(texts),
                "gpu": self.gpu,
                "cold_start": first,
                "model_load_seconds": round(self.load_seconds, 2),
            },
        }


def init_service(device: str = "cuda") -> Service:
    """Load the model, prepare the default voice and warm up the GPU; once per worker."""
    import ptbr_model

    start = time.perf_counter()
    model = ptbr_model.load(device)
    load_seconds = time.perf_counter() - start
    model.prepare_conditionals(str(ptbr_model.download_default_voice()), exaggeration=0.5)
    model.generate("Aquecendo a placa de vídeo.", language_id="pt")
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    print(f"[worker] model ready in {load_seconds:.1f} s on {gpu}", flush=True)
    return Service(model, model.conds, load_seconds, gpu)


def handler(job: dict) -> dict:
    return SERVICE.handle(job.get("input"))


if __name__ == "__main__":
    import runpod

    SERVICE = init_service()
    runpod.serverless.start({"handler": handler})
