"""Convert HF token strings to the exact byte pieces consumed by grammars."""
from __future__ import annotations

import re

_BYTE_RE = re.compile(r"^<0x([0-9A-Fa-f]{2})>$")


def _gpt2_byte_decoder() -> dict[str, int]:
    # Inverse of tokenizers' ByteLevel bytes_to_unicode table.
    values = (list(range(ord("!"), ord("~") + 1)) +
              list(range(ord("¡"), ord("¬") + 1)) +
              list(range(ord("®"), ord("ÿ") + 1)))
    chars = list(values)
    n = 0
    for value in range(256):
        if value not in values:
            values.append(value)
            chars.append(256 + n)
            n += 1
    return {chr(char): value for value, char in zip(values, chars)}


_BYTELEVEL = _gpt2_byte_decoder()


def uses_bytelevel(tokens: list[str | None]) -> bool:
    """ByteLevel's space/newline sentinels are conclusive and cheap to find."""
    return any(t and ("Ġ" in t or "Ċ" in t) for t in tokens)


def piece_bytes(token: str, bytelevel: bool) -> bytes:
    match = _BYTE_RE.match(token)
    if match:
        return bytes([int(match.group(1), 16)])
    if bytelevel:
        try:
            return bytes(_BYTELEVEL[c] for c in token)
        except KeyError:
            # Added/non-ByteLevel text should normally have been classified as
            # special. UTF-8 is the safest fail-open representation.
            return token.encode()
    return token.replace("▁", " ").encode()
