"""Generic inputs and outputs for a finite software-pipeline scheduler."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Optional, Protocol, Tuple


class SolveStatus(str, Enum):
    FEASIBLE = "feasible"
    INFEASIBLE = "infeasible"
    TIMEOUT = "timeout"


@dataclass(frozen=True)
class StageSpec:
    """One schedulable stage instance.

    ``name`` is unique in a problem. ``group`` identifies the source operation
    when a caller expands one operation into multiple instances. ``payload`` is
    opaque to the solver and belongs to the caller's timing oracle.
    """

    name: str
    group: str
    iteration: int = 0
    resources: Tuple[str, ...] = ()
    payload: Any = None


@dataclass(frozen=True)
class Precedence:
    src: str
    dst: str
    delay_ns: float = 0.0


@dataclass(frozen=True)
class ResourceSpec:
    name: str
    capacity: int = 1


@dataclass(frozen=True)
class PipelineHardware:
    resources: Tuple[ResourceSpec, ...] = ()

    def capacity(self, name: str) -> Optional[int]:
        for resource in self.resources:
            if resource.name == name:
                return resource.capacity
        return None


@dataclass(frozen=True)
class PipelineProblem:
    stages: Tuple[StageSpec, ...]
    precedences: Tuple[Precedence, ...] = ()


@dataclass(frozen=True)
class StageTiming:
    duration_ns: float


class TimingOracle(Protocol):
    def estimate(
        self,
        stage: StageSpec,
        *,
        hardware: PipelineHardware,
    ) -> StageTiming:
        ...


@dataclass(frozen=True)
class Placement:
    stage: str
    group: str
    iteration: int
    start_ns: float
    end_ns: float
    resources: Tuple[str, ...] = ()


@dataclass(frozen=True)
class PipelineSolution:
    status: SolveStatus
    makespan_ns: Optional[float] = None
    initiation_interval_ns: Optional[float] = None
    lower_bound_ns: Optional[float] = None
    placements: Tuple[Placement, ...] = ()
    per_group_ns: Mapping[str, float] = field(default_factory=dict)
    critical_path: Tuple[str, ...] = ()
    diagnostics: Tuple[str, ...] = ()

