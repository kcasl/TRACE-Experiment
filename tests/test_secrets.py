"""Fail if a real credential prefix is copied into the source tree."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKIP = {".env", ".git", "__pycache__", ".pytest_cache", "results", ".venv"}
NEEDLES = ("sk-" + "svcacct-", "apikey_" + "2409")


def test_source_tree_has_no_credential_prefix():
    offenders = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or any(part in SKIP for part in path.parts):
            continue
        if path.suffix.lower() not in {".py", ".md", ".toml", ".txt", ".json", ".yml", ".yaml"}:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for needle in NEEDLES:
            if needle in text:
                offenders.append(f"{path.relative_to(ROOT)} contains a credential prefix")
    assert offenders == []
