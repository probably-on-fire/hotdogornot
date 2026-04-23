from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class ConnectorClass:
    id: int
    name: str
    family: str                      # "sma" | "precision"
    gender: str                      # "male" | "female"
    inner_pin_diameter_mm: float
    frequency_ghz_max: float
    impedance_ohms: int
    mating_torque_in_lb: float

    def is_precision(self) -> bool:
        return self.family == "precision"


def load_classes(path: Path | str) -> list[ConnectorClass]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    classes = [ConnectorClass(**entry) for entry in data["classes"]]

    # Invariant: ids must be 0..N-1 with no gaps
    ids = [c.id for c in classes]
    if ids != list(range(len(ids))):
        raise ValueError(f"Class ids must be contiguous from 0; got {ids}")

    return classes
