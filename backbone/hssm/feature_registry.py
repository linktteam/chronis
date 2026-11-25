"""WP8 (owner decision A6, channel 1): a small feature registry, so that a stored scaler and the regime means in raw units are interpretable (steps, beats per minute, pitch in Hz, ...).

A registry is a tuple of `FeatureSpec`, in the column order of the model input. `direction` is the DOMAIN PRIOR for the activity ordering (+1: more activity means a higher value, -1: lower,
0: the feature carries no activity direction). It is never learned from data or from outcome labels. `role` is what the ordering ladder uses to build its rungs.
"""
from __future__ import annotations

from dataclasses import dataclass

SOURCES = ("movement", "heart_rate", "voice", "other")
ROLES = ("movement", "time_moving", "heart_rate", "talking", "other")


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    unit: str
    source: str
    role: str
    direction: int = 0

    def __post_init__(self) -> None:
        if not self.name or not self.unit:
            raise ValueError("a feature needs a name and a unit")
        if self.source not in SOURCES:
            raise ValueError(f"source must be one of {SOURCES}, got {self.source!r}")
        if self.role not in ROLES:
            raise ValueError(f"role must be one of {ROLES}, got {self.role!r}")
        if self.direction not in (-1, 0, 1):
            raise ValueError("direction must be -1, 0 or +1")


def validate_registry(registry, n_features: int | None = None) -> tuple:
    reg = tuple(registry)
    names = [f.name for f in reg]
    if len(set(names)) != len(names):
        raise ValueError("feature names must be unique")
    if n_features is not None and len(reg) != n_features:
        raise ValueError(f"registry has {len(reg)} features, the model input has {n_features}")
    return reg
