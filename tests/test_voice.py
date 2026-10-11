import base64
import dataclasses
import functools
import http.server
import threading
import time

import pytest

import synthesis
from synthesis import InvalidInput


class _QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        if self.path == "/slow":
            # 40 bytes, one every 0.1 s: every read arrives before the socket timeout.
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
        if self.path == "/redirect":
            self.send_response(302)
            self.send_header("Location", "http://169.254.169.254/latest/meta-data/")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        super().do_GET()


@pytest.fixture
def server(tmp_path, monkeypatch):
    (tmp_path / "voice.wav").write_bytes(b"RIFF" + b"\x01" * 1000)
    (tmp_path / "big.bin").write_bytes(b"\x00" * 2048)
    httpd = http.server.ThreadingHTTPServer(
        ("127.0.0.1", 0), functools.partial(_QuietHandler, directory=str(tmp_path)))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    address = f"http://127.0.0.1:{httpd.server_address[1]}"
    # The test server is local; every other destination goes through the real check.
    check = synthesis._check_destination
    monkeypatch.setattr(synthesis, "_check_destination",
                        lambda url: None if url.startswith(address + "/") else check(url))
    yield address
    httpd.shutdown()


def test_base64_with_data_prefix_and_line_breaks():
    data = b"RIFF" + bytes(range(256)) * 4
    encoded = base64.encodebytes(data).decode()
    assert "\n" in encoded
    assert synthesis.decode_base64("data:audio/wav;base64," + encoded) == data


@pytest.mark.parametrize("value, fragment", [
    ("isto não é base64!", "not valid base64"),
    ("", "is empty"),
])
def test_invalid_base64(value, fragment):
    with pytest.raises(InvalidInput, match=fragment):
        synthesis.decode_base64(value)


def test_base64_over_the_limit(monkeypatch):
    monkeypatch.setattr(synthesis, "MAX_VOICE_BYTES", 10)
    with pytest.raises(InvalidInput, match="exceeds 10 MB"):
        synthesis.decode_base64(base64.b64encode(b"x" * 11).decode())


def test_downloads_voice_by_url(server):
    assert synthesis.download_voice(f"{server}/voice.wav") == b"RIFF" + b"\x01" * 1000


def test_missing_url_becomes_an_error(server):
    with pytest.raises(InvalidInput, match="Could not download"):
        synthesis.download_voice(f"{server}/missing.wav")


def test_download_over_the_limit(server, monkeypatch):
    monkeypatch.setattr(synthesis, "MAX_VOICE_BYTES", 1024)
    with pytest.raises(InvalidInput, match="exceeds 10 MB"):
        synthesis.download_voice(f"{server}/big.bin")


@pytest.mark.parametrize("url", [
    "http://127.0.0.1:9/voz.wav",
    "http://localhost:9/voz.wav",
    "http://169.254.169.254/latest/meta-data/",
    "http://10.0.0.5/voz.wav",
    "http://[::1]:9/voz.wav",
    "http://[::ffff:127.0.0.1]:9/voz.wav",
])
def test_url_to_internal_address_is_refused(url):
    with pytest.raises(InvalidInput, match="internal address"):
        synthesis.download_voice(url, max_seconds=1.0)


def test_redirect_to_internal_address_is_refused(server):
    with pytest.raises(InvalidInput, match="internal address"):
        synthesis.download_voice(f"{server}/redirect", max_seconds=1.0)


def test_slow_download_respects_the_time_limit(server):
    start = time.monotonic()
    with pytest.raises(InvalidInput, match="took more than 1 s"):
        synthesis.download_voice(f"{server}/slow", max_seconds=1.0)
    assert time.monotonic() - start < 2.5


def test_get_voice_picks_the_source(server):
    base = synthesis.SpeechRequest("Oi.", "neutral", 0.5, 0.5, 0.8, None, None, None)
    assert synthesis.get_voice(base) is None
    with_b64 = dataclasses.replace(base, voice_base64=base64.b64encode(b"abc").decode())
    assert synthesis.get_voice(with_b64) == b"abc"
    with_url = dataclasses.replace(base, voice_url=f"{server}/voice.wav")
    assert synthesis.get_voice(with_url).startswith(b"RIFF")


@pytest.mark.parametrize("header, suffix", [
    (b"RIFF....", ".wav"), (b"fLaC....", ".flac"), (b"OggS....", ".ogg"),
    (b"ID3.....", ".mp3"), (b"\xff\xfb....", ".mp3"), (b"<html>..", ".audio"),
])
def test_suffix_from_content(header, suffix):
    assert synthesis.audio_suffix(header) == suffix
