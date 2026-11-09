"""Minimal JSON HTTP client with backoff.

Retries cover rate limits and server failures. Authentication and schema
errors are returned to the caller so a bad key is not retried. Response
text is redacted before it can reach a log or an exception message.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Optional

from ..redact import redact

RETRY_STATUS = frozenset({429, 500, 502, 503, 529})


class HttpError(RuntimeError):
    def __init__(self, status: int, body: str):
        self.status = status
        self.body = redact(body)[:500]
        super().__init__(f"HTTP {status}: {self.body}")


def post_json(
    url: str,
    payload: dict,
    headers: dict[str, str],
    *,
    timeout: float,
    max_retries: int = 3,
    backoff: float = 1.5,
    transport=None,
) -> dict:
    data = json.dumps(payload).encode("utf-8")
    return _send(url, data, {**headers, "Content-Type": "application/json"}, timeout, max_retries, backoff, transport)


def post_bytes(
    url: str,
    data: bytes,
    headers: dict[str, str],
    *,
    timeout: float,
    max_retries: int = 3,
    backoff: float = 1.5,
    transport=None,
) -> dict:
    return _send(url, data, headers, timeout, max_retries, backoff, transport)


def _send(url, data: bytes, headers, timeout, max_retries, backoff, transport) -> dict:
    last: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            status, text = (transport or _urllib)(url, data, headers, timeout)
        except Exception as exc:
            last = exc
            time.sleep(backoff ** attempt)
            continue
        if status == 200:
            try:
                return json.loads(text)
            except json.JSONDecodeError as exc:
                raise HttpError(status, redact(text)) from exc
        if status in RETRY_STATUS and attempt + 1 < max_retries:
            time.sleep(backoff ** attempt)
            last = HttpError(status, text)
            continue
        raise HttpError(status, text)
    raise HttpError(0, redact(repr(last)))


def _urllib(url: str, data: bytes, headers: dict[str, str], timeout: float) -> tuple[int, str]:
    request = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", errors="replace")
