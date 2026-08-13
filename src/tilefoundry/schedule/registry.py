"""Which algorithm schedules which hardware at which level.

Scheduling dispatches on the exact `(Target concrete type, topology name)` pair,
so the support matrix is something you read off the registrations. Two devices
that share a base class do not share a scheduler: what is legal at one level of
one machine says nothing about the same level of another.

An algorithm carries no declared inputs or outputs here. Unlike an analysis, it
owns its whole problem -- its program view, its facts query, its solve, and its
Plan type -- so there is nothing for a registration to promise on its behalf.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from tilefoundry.registry import AlgorithmRegistry

# An algorithm solves for the selected Module and Function at one topology.
ScheduleCallable = Callable[..., "SchedulePlan"]


@dataclass(frozen=True)
class ScheduleAlgorithm:
    """One registered scheduler: the level it solves for, and the solve."""

    topology: str
    solve: ScheduleCallable

    def __post_init__(self) -> None:
        if not self.topology:
            raise ValueError("a schedule algorithm needs a non-empty topology name")


class _ScheduleRegistry(AlgorithmRegistry[ScheduleAlgorithm]):
    """The shared registry, loading builtins only when a caller reads it."""

    def __init__(self) -> None:
        super().__init__("schedule")
        self._builtins_loaded = False

    def _ensure_builtins(self) -> None:
        if self._builtins_loaded:
            return
        from tilefoundry.target import register_schedule_algorithms  # noqa: PLC0415

        register_schedule_algorithms()
        self._builtins_loaded = True

    def resolve(self, target: object, selector: str) -> ScheduleAlgorithm:
        self._ensure_builtins()
        return super().resolve(target, selector)

    def selectors_for(self, target_type: type) -> tuple[str, ...]:
        self._ensure_builtins()
        return super().selectors_for(target_type)

    def registered_pairs(self) -> tuple[tuple[str, str], ...]:
        self._ensure_builtins()
        return super().registered_pairs()


SCHEDULES = _ScheduleRegistry()


def register_schedule(
    target_type: type, topology: str
) -> Callable[[ScheduleCallable], ScheduleCallable]:
    """Register a scheduler for one exact target at one topology level."""

    def bind(solve: ScheduleCallable) -> ScheduleCallable:
        SCHEDULES.register(
            target_type, topology, ScheduleAlgorithm(topology=topology, solve=solve)
        )
        return solve

    return bind


__all__ = [
    "SCHEDULES",
    "ScheduleAlgorithm",
    "ScheduleCallable",
    "register_schedule",
]
