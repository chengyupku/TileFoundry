"""The public Schedule boundary and its independently importable families."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from pathlib import Path


@dataclass(frozen=True)
class ScheduleOptions:
    """Solver runtime and debug controls, independent of the selected algorithm."""

    timeout_seconds: float = 60.0
    workers: int = 0
    random_seed: int = 0
    stop_at_first_solution: bool = False
    debug_dump_dir: Path | None = None


_PUBLIC = {
    "PlanVerificationError": ("tilefoundry.schedule.plan", "PlanVerificationError"),
    "ScheduleError": ("tilefoundry.schedule.errors", "ScheduleError"),
    "SchedulePlan": ("tilefoundry.schedule.plan", "SchedulePlan"),
    "ScheduleResult": ("tilefoundry.schedule.api", "ScheduleResult"),
    "schedule": ("tilefoundry.schedule.api", "schedule"),
}


def __getattr__(name: str) -> object:
    """Resolve an established schedule public name on first use."""
    entry = _PUBLIC.get(name)
    if entry is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from tilefoundry import _ensure_ir  # noqa: PLC0415

    _ensure_ir()
    module_name, attribute = entry
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value


__all__ = [
    "PlanVerificationError",
    "ScheduleError",
    "ScheduleOptions",
    "SchedulePlan",
    "ScheduleResult",
    "schedule",
]
