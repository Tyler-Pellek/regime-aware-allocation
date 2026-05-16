"""Config loader: yaml -> nested dataclass-style dict access."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class Config(dict):
    """Dot-accessible nested dict."""

    def __getattr__(self, key: str) -> Any:
        try:
            value = self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc
        if isinstance(value, dict) and not isinstance(value, Config):
            value = Config(value)
            self[key] = value
        return value

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value


def load_config(path: str | Path) -> Config:
    path = Path(path)
    with path.open("r") as fh:
        raw = yaml.safe_load(fh)
    return _to_config(raw)


def _to_config(obj: Any) -> Any:
    if isinstance(obj, dict):
        return Config({k: _to_config(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_config(v) for v in obj]
    return obj
