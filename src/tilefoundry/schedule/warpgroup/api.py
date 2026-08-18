"""One public build, solve, export, and verification workflow."""

from __future__ import annotations

from dataclasses import dataclass, field

from .build import build_warpgroup_problem
from .cost import CostLibrary
from .errors import WarpgroupValidationError
from .model import WarpgroupProgram, WarpgroupSchedule
from .solve import SolveStatus, solve_warpgroup_problem
from .sync import export_warpgroup_schedule
from .verify import _verify_warpgroup_schedule


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
    program: WarpgroupProgram,
    hardware: CostLibrary,
    *,
    timeout_seconds: float = 60.0,
    search_workers: int = 1,
) -> WarpgroupScheduleResult:
    """Close a program with hardware costs, solve, export, and independently verify.

    ``search_workers`` above one trades the reproducible arrangement for speed:
    CP-SAT returns whichever of several optimal arrangements a racing worker
    proves first. One worker keeps the answer reproducible and is the default.
    """
    if type(program) is not WarpgroupProgram:
        raise WarpgroupValidationError("schedule_warpgroups requires an exact WarpgroupProgram")
    problem = build_warpgroup_problem(program, hardware)
    solved = solve_warpgroup_problem(
        problem, timeout_seconds=timeout_seconds, search_workers=search_workers
    )
    schedule = export_warpgroup_schedule(problem, solved)
    _verify_warpgroup_schedule(problem, schedule)
    return WarpgroupScheduleResult(solved.status, schedule)


__all__ = ["WarpgroupScheduleResult", "schedule_warpgroups"]
