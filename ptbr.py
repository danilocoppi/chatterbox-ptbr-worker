"""Carrega o Chatterbox Multilingual V3 com o pacote pt-BR da Resemble AI."""
import os
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file as load_safetensors

from chatterbox.mtl_tts import ChatterboxMultilingualTTS, Conditionals
from chatterbox.models.s3gen import S3Gen
from chatterbox.models.t3 import T3
from chatterbox.models.t3.modules.t3_config import T3Config
from chatterbox.models.tokenizers import MTLTokenizer
from chatterbox.models.voice_encoder import VoiceEncoder

BASE_REPO = "ResembleAI/chatterbox"
PACOTE_REPO = "ResembleAI/Chatterbox-Multilingual-pt-br"
VOZ_PADRAO_URL = ("https://storage.googleapis.com/chatterbox-demo-samples/"
                  "mtl-v3-single-language-prompts/pt-br/pt_br_f2.wav")
DIR_MODELOS = Path(os.environ.get("CHATTERBOX_MODELOS", Path(__file__).resolve().parent / "modelos"))
VOZ_PADRAO = DIR_MODELOS / "vozes" / "pt_br_f2.wav"
# O s3gen_v3.pt não traz estes buffers fixos, que o construtor do S3Gen já cria.
BUFFERS_AUSENTES_S3GEN = {"tokenizer._mel_filters", "tokenizer.window"}


def montar_checkpoint() -> Path:
    """Junta os arquivos do repositório base com o pacote pt-BR numa pasta (links simbólicos)."""
    base = Path(snapshot_download(BASE_REPO, allow_patterns=[
        "ve.pt", "conds.pt", "grapheme_mtl_merged_expanded_v1.json", "Cangjie5_TC.json"]))
    pacote = Path(snapshot_download(PACOTE_REPO, allow_patterns=[
        "t3_pt_br.safetensors", "s3gen_v3.pt", "grapheme_mtl_merged_expanded_v1.json"]))
    destino = DIR_MODELOS / "pt-br"
    destino.mkdir(parents=True, exist_ok=True)
    links = {
        "ve.pt": base / "ve.pt",
        "conds.pt": base / "conds.pt",
        "Cangjie5_TC.json": base / "Cangjie5_TC.json",
        "grapheme_mtl_merged_expanded_v1.json": pacote / "grapheme_mtl_merged_expanded_v1.json",
        "t3_pt_br.safetensors": pacote / "t3_pt_br.safetensors",
        "s3gen.pt": pacote / "s3gen_v3.pt",
    }
    for nome, alvo in links.items():
        link = destino / nome
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(alvo.resolve())
    return destino


def baixar_voz_padrao() -> Path:
    """Baixa (uma vez) a voz feminina pt-BR da demo oficial da Resemble."""
    if not VOZ_PADRAO.exists():
        VOZ_PADRAO.parent.mkdir(parents=True, exist_ok=True)
        torch.hub.download_url_to_file(VOZ_PADRAO_URL, str(VOZ_PADRAO), progress=False)
    return VOZ_PADRAO


def carregar_tokenizador(ckpt: Path) -> MTLTokenizer:
    """Cria o tokenizador; na primeira vez ele baixa o segmentador chinês (para PKUSEG_HOME) e o mapa
    Cangjie (para dentro do checkpoint). A montagem da imagem chama isto para que já fiquem lá."""
    return MTLTokenizer(str(ckpt / "grapheme_mtl_merged_expanded_v1.json"))


def carregar(device: str = "cuda") -> ChatterboxMultilingualTTS:
    """Igual ao from_local do pacote, com o vocabulário lido do próprio checkpoint pt-BR."""
    ckpt = montar_checkpoint()

    ve = VoiceEncoder()
    ve.load_state_dict(torch.load(ckpt / "ve.pt", weights_only=True, map_location="cpu"))
    ve.to(device).eval()

    estado_t3 = load_safetensors(ckpt / "t3_pt_br.safetensors")
    if "model" in estado_t3.keys():
        estado_t3 = estado_t3["model"][0]
    configuracao = T3Config.multilingual()
    configuracao.text_tokens_dict_size = estado_t3["text_emb.weight"].shape[0]
    t3 = T3(configuracao)
    t3.load_state_dict(estado_t3)
    t3.to(device).eval()

    s3gen = S3Gen()
    faltando, sobrando = s3gen.load_state_dict(
        torch.load(ckpt / "s3gen.pt", weights_only=True, map_location="cpu"), strict=False)
    if set(faltando) - BUFFERS_AUSENTES_S3GEN or sobrando:
        raise RuntimeError(f"Checkpoint S3Gen inesperado: faltando={faltando}, sobrando={sobrando}")
    s3gen.to(device).eval()

    tokenizer = carregar_tokenizador(ckpt)
    conds = Conditionals.load(ckpt / "conds.pt", map_location="cpu").to(device)
    return ChatterboxMultilingualTTS(t3, s3gen, ve, tokenizer, device, conds=conds)
