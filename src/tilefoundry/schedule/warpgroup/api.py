"""One public build, solve, export, and verification workflow."""

from __future__ import annotations

from dataclasses import dataclass, field

from .build import build_warpgroup_problem
from .cost import CostLibrary
from .errors import WarpgroupValidationError
from .model import WarpgroupProblem, WarpgroupProgram, WarpgroupSchedule
from .solve import SolveStatus, solve_warpgroup_problem
from .sync import export_warpgroup_schedule
from .verify import verify_warpgroup_schedule


@dataclass(frozen=True, slots=True)
class WarpgroupScheduleResult:
    """A verified schedule with its solve status and derived makespan."""

    status: SolveStatus
    schedule: WarpgroupSchedule
    makespan: int = field(init=False)

    def __post_init__(self) -> None:
        if self.status not in ("OPTIMAL", "FEASIBLE_NOT_PROVEN"):
            raise WarpgroupValidationError(f"invalid warpgroup schedule status {self.status!r}")
        if type(self.schedule) is not WarpgroupSchedule:
            raise WarpgroupValidationError("result schedule must be an exact WarpgroupSchedule")
        object.__setattr__(
            self,
            "makespan",
            max((timed.completion for timed in self.schedule.times), default=0),
        )


def schedule_warpgroups(
    source: WarpgroupProgram | WarpgroupProblem,
    cost_library: CostLibrary | None = None,
    *,
    timeout_seconds: float = 60.0,
) -> WarpgroupScheduleResult:
    """Close costs when needed, solve once, export, and independently verify."""
    if type(source) is WarpgroupProgram:
        if cost_library is None:
            raise WarpgroupValidationError("a warpgroup program requires a cost library")
        problem = build_warpgroup_problem(source, cost_library)
    elif type(source) is WarpgroupProblem:
        if cost_library is not None:
            raise WarpgroupValidationError("a closed warpgroup problem rejects a cost library")
        problem = source
    else:
        raise WarpgroupValidationError(
            "schedule_warpgroups requires an exact WarpgroupProgram or WarpgroupProblem"
        )
    solved = solve_warpgroup_problem(problem, timeout_seconds=timeout_seconds)
    schedule = export_warpgroup_schedule(problem, solved)
    verify_warpgroup_schedule(problem, schedule)
    return WarpgroupScheduleResult(solved.status, schedule)


__all__ = ["WarpgroupScheduleResult", "schedule_warpgroups"]
