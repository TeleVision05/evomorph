"""JSON serialization for dataclasses, numpy values and nested containers.

Floats are written with Python's shortest round-trip repr, so a value written and
read back is bit-identical. That is what makes "save a genome, reload it, get the
same metrics" hold exactly.
"""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def to_jsonable(obj: Any) -> Any:
    """Convert dataclasses, numpy arrays/scalars, tuples and paths to JSON-safe values."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: to_jsonable(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [to_jsonable(v) for v in obj.tolist()]
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        value = float(obj)
        # JSON has no NaN/inf; encode them as strings so files stay standard JSON.
        if math.isnan(value):
            return "nan"
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        return value
    if isinstance(obj, Path):
        return str(obj)
    return obj


def dump_json(obj: Any, path: str | Path, indent: int = 2) -> Path:
    """Write ``obj`` as JSON, creating parent folders. Returns the path written."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(to_jsonable(obj), indent=indent) + "\n")
    return out


def load_json(path: str | Path) -> Any:
    """Read a JSON file written by :func:`dump_json`."""
    return json.loads(Path(path).read_text())


def dumps_json(obj: Any) -> str:
    """Compact JSON string of ``obj`` (useful for hashing or logging)."""
    return json.dumps(to_jsonable(obj), sort_keys=True)
