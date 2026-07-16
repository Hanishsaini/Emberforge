"""
EMBERFORGE token counting — accurate when possible, honest about fallback.

Uses tiktoken (cl100k_base) when available; falls back to the chars/4 estimate
if tiktoken is missing or cannot load its encoding (e.g. offline first run).
`count_tokens.is_exact` tells callers which one they got, so benchmarks can
label their numbers correctly.
"""
from __future__ import annotations

_ENCODER = None
_TRIED = False


def _get_encoder():
    global _ENCODER, _TRIED
    if _TRIED:
        return _ENCODER
    _TRIED = True
    try:
        import tiktoken
        _ENCODER = tiktoken.get_encoding("cl100k_base")
    except Exception:
        _ENCODER = None
    return _ENCODER


def count_tokens(text: str) -> int:
    """Token count of `text` — exact via tiktoken, else chars/4 estimate."""
    if not text:
        return 0
    enc = _get_encoder()
    if enc is not None:
        try:
            return len(enc.encode(text, disallowed_special=()))
        except Exception:
            pass
    return max(1, len(text) // 4)


def counting_is_exact() -> bool:
    """True if tiktoken is active (benchmark labels depend on this)."""
    return _get_encoder() is not None
