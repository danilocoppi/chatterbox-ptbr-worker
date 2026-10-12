"""Loads Chatterbox Multilingual V3 with Resemble AI's pt-BR language pack."""
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
PACK_REPO = "ResembleAI/Chatterbox-Multilingual-pt-br"
DEFAULT_VOICE_URL = ("https://storage.googleapis.com/chatterbox-demo-samples/"
                     "mtl-v3-single-language-prompts/pt-br/pt_br_f2.wav")
MODELS_DIR = Path(os.environ.get("CHATTERBOX_MODELS", Path(__file__).resolve().parent / "models"))
DEFAULT_VOICE = MODELS_DIR / "voices" / "pt_br_f2.wav"
# s3gen_v3.pt lacks these fixed buffers, which the S3Gen constructor already creates.
S3GEN_MISSING_BUFFERS = {"tokenizer._mel_filters", "tokenizer.window"}


def assemble_checkpoint() -> Path:
    """Gather the base repo files and the pt-BR pack into one folder (symbolic links)."""
    base = Path(snapshot_download(BASE_REPO, allow_patterns=[
        "ve.pt", "conds.pt", "grapheme_mtl_merged_expanded_v1.json", "Cangjie5_TC.json"]))
    pack = Path(snapshot_download(PACK_REPO, allow_patterns=[
        "t3_pt_br.safetensors", "s3gen_v3.pt", "grapheme_mtl_merged_expanded_v1.json"]))
    target = MODELS_DIR / "pt-br"
    target.mkdir(parents=True, exist_ok=True)
    links = {
        "ve.pt": base / "ve.pt",
        "conds.pt": base / "conds.pt",
        "Cangjie5_TC.json": base / "Cangjie5_TC.json",
        "grapheme_mtl_merged_expanded_v1.json": pack / "grapheme_mtl_merged_expanded_v1.json",
        "t3_pt_br.safetensors": pack / "t3_pt_br.safetensors",
        "s3gen.pt": pack / "s3gen_v3.pt",
    }
    for name, source in links.items():
        link = target / name
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(source.resolve())
    return target


def download_default_voice() -> Path:
    """Download (once) the female pt-BR voice from Resemble's official demo."""
    if not DEFAULT_VOICE.exists():
        DEFAULT_VOICE.parent.mkdir(parents=True, exist_ok=True)
        torch.hub.download_url_to_file(DEFAULT_VOICE_URL, str(DEFAULT_VOICE), progress=False)
    return DEFAULT_VOICE


class RawTextTokenizer(MTLTokenizer):
    """Feeds the text as written, keeping capitals and accented letters, as Resemble's official pt-BR demo does.
    The multilingual tokenizer would lowercase the text and split each accent from its letter (NFKD)."""

    def text_to_tokens(self, text, language_id=None, lowercase=False, nfkd_normalize=False):
        return super().text_to_tokens(text, language_id=language_id, lowercase=lowercase,
                                      nfkd_normalize=nfkd_normalize)


def load_tokenizer(checkpoint: Path) -> MTLTokenizer:
    """Create the tokenizer. The first time, it downloads the Chinese segmenter (into PKUSEG_HOME) and the
    Cangjie map (into the checkpoint); the image build calls this so both are already there at startup."""
    return RawTextTokenizer(str(checkpoint / "grapheme_mtl_merged_expanded_v1.json"))


def load(device: str = "cuda") -> ChatterboxMultilingualTTS:
    """Same as the package's from_local, with the vocabulary size read from the pt-BR checkpoint itself and the
    text fed as written."""
    checkpoint = assemble_checkpoint()

    ve = VoiceEncoder()
    ve.load_state_dict(torch.load(checkpoint / "ve.pt", weights_only=True, map_location="cpu"))
    ve.to(device).eval()

    t3_state = load_safetensors(checkpoint / "t3_pt_br.safetensors")
    if "model" in t3_state.keys():
        t3_state = t3_state["model"][0]
    config = T3Config.multilingual()
    config.text_tokens_dict_size = t3_state["text_emb.weight"].shape[0]
    t3 = T3(config)
    t3.load_state_dict(t3_state)
    t3.to(device).eval()

    s3gen = S3Gen()
    missing, unexpected = s3gen.load_state_dict(
        torch.load(checkpoint / "s3gen.pt", weights_only=True, map_location="cpu"), strict=False)
    if set(missing) - S3GEN_MISSING_BUFFERS or unexpected:
        raise RuntimeError(f"Unexpected S3Gen checkpoint: missing={missing}, unexpected={unexpected}")
    s3gen.to(device).eval()

    tokenizer = load_tokenizer(checkpoint)
    conds = Conditionals.load(checkpoint / "conds.pt", map_location="cpu").to(device)
    return ChatterboxMultilingualTTS(t3, s3gen, ve, tokenizer, device, conds=conds)
