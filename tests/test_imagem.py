"""A imagem instala só versões fixas: as do lock, sem resolver dependências na montagem."""
import re
import shlex
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
_FIXO = re.compile(r"^[A-Za-z0-9._-]+==\S+$")
_GIT_FIXO = re.compile(r"^[A-Za-z0-9._-]+ @ git\+\S+@[0-9a-f]{40}$")


def _requisitos(arquivo: Path) -> list[str]:
    linhas = (linha.split("#")[0].strip() for linha in arquivo.read_text().splitlines())
    return [linha for linha in linhas if linha]


def _instalacoes_pip() -> list[list[str]]:
    texto = (RAIZ / "Dockerfile").read_text().replace("\\\n", " ")
    return [shlex.split(linha[4:]) for linha in texto.splitlines()
            if linha.startswith("RUN ") and "pip install" in linha]


def test_toda_instalacao_da_imagem_usa_so_versoes_fixas():
    instalacoes = _instalacoes_pip()
    assert instalacoes
    for comando in instalacoes:
        assert "--no-deps" in comando, comando
        arquivos = [comando[i + 1] for i, arg in enumerate(comando) if arg == "-r"]
        diretos = [arg for i, arg in enumerate(comando[2:], 2)
                   if not arg.startswith("-") and comando[i - 1] != "-r"]
        assert arquivos and not diretos, comando
        for arquivo in arquivos:
            soltos = [r for r in _requisitos(RAIZ / arquivo) if not (_FIXO.match(r) or _GIT_FIXO.match(r))]
            assert soltos == [], f"{arquivo}: {soltos}"
