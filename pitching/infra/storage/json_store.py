from __future__ import annotations

import dataclasses
import json
from enum import Enum
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1


def _to_serializable(obj: Any) -> Any:
    """dataclass / Enum / tuple を JSON シリアライズ可能な形に再帰変換する。"""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {
            "__type__": type(obj).__name__,
            **{f.name: _to_serializable(getattr(obj, f.name)) for f in dataclasses.fields(obj)},
        }
    if isinstance(obj, Enum):
        return {"__enum__": type(obj).__name__, "value": obj.value}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(v) for v in obj]
    return obj


def _from_serializable(obj: Any, registry: dict[str, type]) -> Any:
    """JSON デシリアライズ時に dataclass / Enum を復元する。"""
    if isinstance(obj, dict):
        if "__enum__" in obj:
            cls = registry[obj["__enum__"]]
            return cls(obj["value"])
        if "__type__" in obj:
            cls = registry[obj["__type__"]]
            fields = dataclasses.fields(cls)
            kwargs = {
                f.name: _from_serializable(obj[f.name], registry)
                for f in fields
                if f.name in obj
            }
            return cls(**kwargs)
        return {k: _from_serializable(v, registry) for k, v in obj.items()}
    if isinstance(obj, list):
        return tuple(_from_serializable(v, registry) for v in obj)
    return obj


def save_json(data: Any, path: Path, registry: dict[str, type] | None = None) -> None:
    """dataclass を JSON ファイルに保存する。"""
    payload = {
        "schema_version": SCHEMA_VERSION,
        "data": _to_serializable(data),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_json(path: Path, root_type: type, registry: dict[str, type]) -> Any:
    """JSON ファイルを dataclass に復元する。"""
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)

    version = payload.get("schema_version", 0)
    if version != SCHEMA_VERSION:
        raise ValueError(f"Schema version mismatch: expected {SCHEMA_VERSION}, got {version}")

    return _from_serializable(payload["data"], registry)
