import base64
import dataclasses
import functools
import http.server
import threading
import time

import pytest

import sintese
from sintese import EntradaInvalida


class _Silencioso(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        if self.path == "/lento":
            # 40 bytes, um a cada 0,1 s: cada leitura chega antes do timeout do socket.
            self.send_response(200)
            self.send_header("Content-Length", "40")
            self.end_headers()
            try:
                for _ in range(40):
                    self.wfile.write(b"x")
                    time.sleep(0.1)
            except (BrokenPipeError, ConnectionResetError):
                pass
            return
        if self.path == "/redirecionar":
            self.send_response(302)
            self.send_header("Location", "http://169.254.169.254/latest/meta-data/")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        super().do_GET()


@pytest.fixture
def servidor(tmp_path, monkeypatch):
    (tmp_path / "voz.wav").write_bytes(b"RIFF" + b"\x01" * 1000)
    (tmp_path / "grande.bin").write_bytes(b"\x00" * 2048)
    httpd = http.server.ThreadingHTTPServer(
        ("127.0.0.1", 0), functools.partial(_Silencioso, directory=str(tmp_path)))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    endereco = f"http://127.0.0.1:{httpd.server_address[1]}"
    # O servidor de teste é local; qualquer outro destino passa pela conferência real.
    conferir = sintese._conferir_destino
    monkeypatch.setattr(sintese, "_conferir_destino",
                        lambda url: None if url.startswith(endereco + "/") else conferir(url))
    yield endereco
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


@pytest.mark.parametrize("url", [
    "http://127.0.0.1:9/voz.wav",
    "http://localhost:9/voz.wav",
    "http://169.254.169.254/latest/meta-data/",
    "http://10.0.0.5/voz.wav",
    "http://[::1]:9/voz.wav",
    "http://[::ffff:127.0.0.1]:9/voz.wav",
])
def test_url_para_endereco_interno_e_recusada(url):
    with pytest.raises(EntradaInvalida, match="endereço interno"):
        sintese.baixar_voz(url, tempo_max_s=1.0)


def test_redirecionamento_para_endereco_interno_e_recusado(servidor):
    with pytest.raises(EntradaInvalida, match="endereço interno"):
        sintese.baixar_voz(f"{servidor}/redirecionar", tempo_max_s=1.0)


def test_download_lento_respeita_o_tempo_maximo(servidor):
    inicio = time.monotonic()
    with pytest.raises(EntradaInvalida, match="passou de 1 s"):
        sintese.baixar_voz(f"{servidor}/lento", tempo_max_s=1.0)
    assert time.monotonic() - inicio < 2.5


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
