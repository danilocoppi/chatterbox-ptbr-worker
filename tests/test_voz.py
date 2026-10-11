import base64
import dataclasses
import functools
import http.server
import threading

import pytest

import sintese
from sintese import EntradaInvalida


class _Silencioso(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


@pytest.fixture
def servidor(tmp_path):
    (tmp_path / "voz.wav").write_bytes(b"RIFF" + b"\x01" * 1000)
    (tmp_path / "grande.bin").write_bytes(b"\x00" * 2048)
    httpd = http.server.ThreadingHTTPServer(
        ("127.0.0.1", 0), functools.partial(_Silencioso, directory=str(tmp_path)))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()


def test_base64_com_prefixo_data_e_quebras_de_linha():
    dados = b"RIFF" + bytes(range(256)) * 4
    codificado = base64.encodebytes(dados).decode()
    assert "\n" in codificado
    assert sintese.decodificar_base64("data:audio/wav;base64," + codificado) == dados


@pytest.mark.parametrize("dado, trecho", [
    ("isto não é base64!", "não é um base64 válido"),
    ("", "está vazia"),
])
def test_base64_invalido(dado, trecho):
    with pytest.raises(EntradaInvalida, match=trecho):
        sintese.decodificar_base64(dado)


def test_base64_acima_do_limite(monkeypatch):
    monkeypatch.setattr(sintese, "MAX_BYTES_VOZ", 10)
    with pytest.raises(EntradaInvalida, match="passa de 10 MB"):
        sintese.decodificar_base64(base64.b64encode(b"x" * 11).decode())


def test_baixa_voz_por_url(servidor):
    assert sintese.baixar_voz(f"{servidor}/voz.wav") == b"RIFF" + b"\x01" * 1000


def test_url_inexistente_vira_erro(servidor):
    with pytest.raises(EntradaInvalida, match="Não foi possível baixar"):
        sintese.baixar_voz(f"{servidor}/nao-existe.wav")


def test_download_acima_do_limite(servidor, monkeypatch):
    monkeypatch.setattr(sintese, "MAX_BYTES_VOZ", 1024)
    with pytest.raises(EntradaInvalida, match="passa de 10 MB"):
        sintese.baixar_voz(f"{servidor}/grande.bin")


def test_obter_voz_escolhe_a_fonte(servidor):
    base = sintese.Pedido("Oi.", "neutro", 0.5, 0.5, 0.8, None, None, None)
    assert sintese.obter_voz(base) is None
    com_b64 = dataclasses.replace(base, voz_base64=base64.b64encode(b"abc").decode())
    assert sintese.obter_voz(com_b64) == b"abc"
    com_url = dataclasses.replace(base, voz_url=f"{servidor}/voz.wav")
    assert sintese.obter_voz(com_url).startswith(b"RIFF")


@pytest.mark.parametrize("cabecalho, sufixo", [
    (b"RIFF....", ".wav"), (b"fLaC....", ".flac"), (b"OggS....", ".ogg"),
    (b"ID3.....", ".mp3"), (b"\xff\xfb....", ".mp3"), (b"<html>..", ".audio"),
])
def test_sufixo_pelo_conteudo(cabecalho, sufixo):
    assert sintese.sufixo_audio(cabecalho) == sufixo
