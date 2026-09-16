"""Minimal readline stub for environments where the native module crashes."""

from __future__ import annotations

backend = "stub"


def parse_and_bind(_: str) -> None:
    return None


def read_init_file(_: str | None = None) -> None:
    return None


def read_history_file(_: str | None = None) -> None:
    return None


def write_history_file(_: str | None = None) -> None:
    return None


def set_history_length(_: int) -> None:
    return None
