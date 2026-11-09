"""Remove credential material from anything that might be logged or raised."""

from __future__ import annotations

import os
import re
from pathlib import Path

_BEARER = re.compile(r"Bearer\s+\S+", re.IGNORECASE)
_SK = re.compile(r"sk-[A-Za-z0-9_\-]+")
_APIKEY = re.compile(r"apikey_[A-Za-z0-9_]+")


def redact(text: str) -> str:
    hidden = _BEARER.sub("Bearer [REDACTED]", text)
    hidden = _SK.sub("sk-[REDACTED]", hidden)
    hidden = _APIKEY.sub("apikey_[REDACTED]", hidden)
    for name in ("OPENAI_API_KEY", "JEV_API_KEY", "TYPESAFE_API_KEY"):
        value = os.environ.get(name)
        if value:
            hidden = hidden.replace(value, f"${name}")
    return hidden


def load_env(path: Path) -> None:
    """Load KEY=VALUE lines. Existing environment variables win."""

    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def has_env(*names: str) -> bool:
    """True when at least one of the named environment variables is set and non-empty."""

    return any(os.environ.get(name) for name in names)
