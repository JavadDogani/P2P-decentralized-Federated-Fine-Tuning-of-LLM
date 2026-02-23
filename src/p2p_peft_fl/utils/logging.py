from __future__ import annotations
import json
import os
import time
from dataclasses import asdict
from typing import Any, Dict, Optional

class RunLogger:
    """Writes JSONL events + keeps a human-readable stdout helper."""

    def __init__(self, run_dir: str, config_obj: Any):
        self.run_dir = run_dir
        os.makedirs(run_dir, exist_ok=True)
        self.path = os.path.join(run_dir, "events.jsonl")
        self.t0 = time.time()
        self._write({"type": "run_start", "t": 0.0, "config": asdict(config_obj)})

    def _write(self, obj: Dict[str, Any]) -> None:
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    def event(self, type_: str, **fields: Any) -> None:
        self._write({"type": type_, "t": time.time() - self.t0, **fields})

    def close(self) -> None:
        self._write({"type": "run_end", "t": time.time() - self.t0})
