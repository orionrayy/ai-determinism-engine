from __future__ import annotations

from typing import BinaryIO

DEFAULT_READ_CHUNK_BYTES = 64 * 1024


def read_response_limited(
    response: BinaryIO,
    max_bytes: int,
    *,
    error_message: str = "HTTP response exceeds configured safety limit",
) -> bytes:
    """Read an HTTP response body without allowing an unbounded allocation."""
    limit = int(max_bytes)
    if limit <= 0:
        raise ValueError("max_bytes must be positive")

    output = bytearray()
    while len(output) <= limit:
        remaining = limit - len(output)
        chunk = response.read(min(DEFAULT_READ_CHUNK_BYTES, remaining + 1))
        if not chunk:
            return bytes(output)
        output.extend(chunk)
        if len(output) > limit:
            raise RuntimeError(error_message)
    raise RuntimeError(error_message)
