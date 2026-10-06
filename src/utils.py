"""Small helpers: JSON I/O and robust JSON extraction from LLM output."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def save_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def extract_json(text: str) -> dict | None:
    """Return the first JSON object found in ``text`` (or None).

    LLMs often wrap JSON in prose or markdown fences; this scans for each ``{``
    and tries to decode a complete object starting there.
    """
    if not text:
        return None
    decoder = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(text[i:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    return None