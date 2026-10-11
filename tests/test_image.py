"""The image installs pinned versions only: the lock, without resolving dependencies at build time."""
import re
import shlex
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_PINNED = re.compile(r"^[A-Za-z0-9._-]+==\S+$")
_PINNED_GIT = re.compile(r"^[A-Za-z0-9._-]+ @ git\+\S+@[0-9a-f]{40}$")


def _requirements(path: Path) -> list[str]:
    lines = (line.split("#")[0].strip() for line in path.read_text().splitlines())
    return [line for line in lines if line]


def _pip_installs() -> list[list[str]]:
    text = (ROOT / "Dockerfile").read_text().replace("\\\n", " ")
    return [shlex.split(line[4:]) for line in text.splitlines()
            if line.startswith("RUN ") and "pip install" in line]


def test_every_image_install_uses_pinned_versions_only():
    installs = _pip_installs()
    assert installs
    for command in installs:
        assert "--no-deps" in command, command
        files = [command[i + 1] for i, arg in enumerate(command) if arg == "-r"]
        direct = [arg for i, arg in enumerate(command[2:], 2)
                  if not arg.startswith("-") and command[i - 1] != "-r"]
        assert files and not direct, command
        for path in files:
            loose = [r for r in _requirements(ROOT / path) if not (_PINNED.match(r) or _PINNED_GIT.match(r))]
            assert loose == [], f"{path}: {loose}"
