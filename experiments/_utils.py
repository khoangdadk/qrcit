"""File helpers shared by the experiment runners."""

import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    """Hash file contents when recording inputs and source versions."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, value):
    """Publish a complete record after its temporary file is written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)
