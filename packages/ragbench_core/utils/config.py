from __future__ import annotations
import os, re, yaml
from pathlib import Path

_env_pattern = re.compile(r"\$\{([^}:]+)(?::([^}]*))?\}")

def _substitute_env(value):
    if isinstance(value, str):
        def repl(match):
            key, default = match.group(1), match.group(2)
            return os.getenv(key, default if default is not None else "")
        return _env_pattern.sub(repl, value)
    if isinstance(value, list):
        return [_substitute_env(v) for v in value]
    if isinstance(value, dict):
        return {k: _substitute_env(v) for k, v in value.items()}
    return value

def load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return _substitute_env(raw)
