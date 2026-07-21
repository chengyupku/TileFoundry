"""Generic inputs and outputs for a finite software-pipeline scheduler."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Optional, Protocol, Tuple


class SolveStatus(str, Enum):
    OPTIMAL = "optimal"
    FEASIBLE = "feasible"
    UNSUPPORTED = "unsupported"
    INFEASIBLE = "infeasible"
    TIMEOUT = "timeout"


@dataclass(frozen=True)
class ResourceDemand:
    """Capacity slots occupied by one stage on a named resource."""

    resource: str
    demand: int = 1

    def __post_init__(self) -> None:
        if not self.resource:
            raise ValueError("resource demand name must be non-empty")
        if isinstance(self.demand, bool) or not isinstance(self.demand, int):
            raise ValueError("resource demand must be an integer")
        if self.demand <= 0:
            raise ValueError("resource demand must be positive")


@dataclass(frozen=True)
class StageSpec:
    """One schedulable stage instance.

    ``name`` is unique in a problem. ``group`` identifies the source operation
    when a caller expands one operation into multiple instances. ``payload`` is
    opaque to the solver and belongs to the caller's timing oracle. ``resources``
    is the demand-one compatibility form; ``resource_demands`` is the canonical
    per-resource capacity requirement consumed by solvers.
    """

    name: str
    group: str
    iteration: int = 0
    resources: Tuple[str, ...] = ()
    payload: Any = None
    resource_demands: Tuple[ResourceDemand, ...] = ()

    def __post_init__(self) -> None:
        explicit = tuple(self.resource_demands)
        explicit_by_name = {item.resource: item for item in explicit}
        if len(explicit_by_name) != len(explicit):
            raise ValueError("stage has duplicate resource demands")

        names = tuple(dict.fromkeys((
            *tuple(self.resources),
            *(item.resource for item in explicit),
        )))
        if any(not name for name in names):
            raise ValueError("stage resource names must be non-empty")
        normalized = tuple(
            explicit_by_name.get(name, ResourceDemand(name))
            for name in names
        )
        object.__setattr__(self, "resources", names)
        object.__setattr__(self, "resource_demands", normalized)


@dataclass(frozen=True)
class Precedence:
    src: str
    dst: str
    delay_ns: float = 0.0


@dataclass(frozen=True)
class ResourceSpec:
    name: str
    capacity: int = 1

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("resource name must be non-empty")
        if isinstance(self.capacity, bool) or not isinstance(self.capacity, int):
            raise ValueError("resource capacity must be an integer")
        if self.capacity <= 0:
            raise ValueError("resource capacity must be positive")


@dataclass(frozen=True)
class PipelineHardware:
    resources: Tuple[ResourceSpec, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "resources", tuple(self.resources))
        names = [resource.name for resource in self.resources]
        if len(names) != len(set(names)):
            raise ValueError("pipeline hardware has duplicate resource names")

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
    resource_demands: Tuple[ResourceDemand, ...] = ()


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
