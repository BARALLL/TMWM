"""
Lightweight experiment tracking.

Writes one `runs/run_<id>.json` per invocation (provenance: full
RunConfig, git commit, what it resumed from if anything) plus appends
one line per epoch to a shared `metrics.jsonl` tagged with `run_id`,
so a resume lineage's history stays in one place while still letting
you tell which invocation produced which rows.

Deliberately behind a small ABC: swapping in TensorBoard/W&B later
means writing another ExperimentLogger, not touching run_pipeline.py.
"""
from __future__ import annotations
import os
import json
import subprocess
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from config import RunConfig


def _git_commit() -> Optional[str]:
    try:
        out = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return None


def make_run_id() -> str:
    return f"{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"


class ExperimentLogger(ABC):
    @abstractmethod
    def log_run_start(self, run_config: RunConfig, resumed_from: Optional[dict]) -> None:
        ...

    @abstractmethod
    def log_epoch(self, epoch: int, metrics: dict[str, Any]) -> None:
        ...


class JsonlExperimentLogger(ExperimentLogger):
    def __init__(self, output_dir: Path, run_id: str):
        self.output_dir = Path(output_dir)
        self.run_id = run_id
        self.runs_dir = self.output_dir / 'runs'
        self.metrics_path = self.output_dir / 'metrics.jsonl'
        self.runs_dir.mkdir(parents=True, exist_ok=True)

    def log_run_start(self, run_config: RunConfig, resumed_from: Optional[dict]) -> None:
        record = {
            'run_id': self.run_id,
            'started_at': datetime.now(timezone.utc).isoformat(),
            'resumed_from': resumed_from,
            'run_config': run_config.to_dict(),
            'git_commit': _git_commit(),
        }
        run_path = self.runs_dir / f'run_{self.run_id}.json'
        with open(run_path, 'w') as f:
            json.dump(record, f, indent=2)
        print(f'  Run record: {run_path}')

    def log_epoch(self, epoch: int, metrics: dict[str, Any]) -> None:
        record = {'run_id': self.run_id, 'epoch': epoch, **metrics}
        with open(self.metrics_path, 'a') as f:
            f.write(json.dumps(record) + '\n')