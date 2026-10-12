from pathlib import Path

from chatterbox.models.tokenizers import MTLTokenizer

import ptbr_model


def test_tokenizer_feeds_the_text_as_written(monkeypatch):
    seen = {}

    def parent(self, text, language_id=None, lowercase=True, nfkd_normalize=True):
        seen.update(text=text, language_id=language_id, lowercase=lowercase, nfkd_normalize=nfkd_normalize)

    monkeypatch.setattr(MTLTokenizer, "text_to_tokens", parent)
    tokenizer = object.__new__(ptbr_model.RawTextTokenizer)  # skip loading the vocabulary
    tokenizer.text_to_tokens("Não é?", language_id="pt")
    assert seen == {"text": "Não é?", "language_id": "pt", "lowercase": False, "nfkd_normalize": False}


def test_load_tokenizer_uses_the_raw_text_tokenizer(monkeypatch):
    monkeypatch.setattr(ptbr_model, "RawTextTokenizer", lambda path: ("raw", path))
    assert ptbr_model.load_tokenizer(Path("/ckpt")) == ("raw", "/ckpt/grapheme_mtl_merged_expanded_v1.json")
