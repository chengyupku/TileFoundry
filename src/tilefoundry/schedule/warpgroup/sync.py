"""Canonical synchronization export from one finite solve result."""

from __future__ import annotations

from .errors import WarpgroupVerificationError
from .model import (
    PROBLEM_FORMAT,
    PROBLEM_FORMAT_V2,
    PROBLEM_FORMAT_V3,
    SCHEDULE_FORMAT,
    SCHEDULE_FORMAT_V2,
    SCHEDULE_FORMAT_V3,
    WarpgroupProblem,
    WarpgroupSchedule,
)
from .solve import WarpgroupSolveResult
from .verify import (
    _completion_event_edges,
    _event_reachable,
    _expand_relation,
    _required_completion_relations,
    verify_warpgroup_schedule,
)


def export_warpgroup_schedule(
    problem: WarpgroupProblem, result: WarpgroupSolveResult
) -> WarpgroupSchedule:
    """Export and transitively reduce the synchronization required by a solve."""
    if type(problem) is not WarpgroupProblem:
        raise WarpgroupVerificationError("export requires an exact WarpgroupProblem")
    if type(result) is not WarpgroupSolveResult:
        raise WarpgroupVerificationError("export requires an exact WarpgroupSolveResult")

    expected = {operation.id for operation in problem.loop.ops}
    lane_by_operation = {
        operation_id: lane_index
        for lane_index, lane in enumerate(result.lanes)
        for operation_id in lane.operations
    }
    if len(result.lanes) != problem.warp_groups or set(lane_by_operation) != expected:
        raise WarpgroupVerificationError(
            "solve result lanes do not exactly cover the problem operations"
        )

    candidates = tuple(
        relation
        for relation in _required_completion_relations(problem)
        if relation.distance < problem.loop.iterations
    )
    schedule_format = {
        PROBLEM_FORMAT: SCHEDULE_FORMAT,
        PROBLEM_FORMAT_V2: SCHEDULE_FORMAT_V2,
        PROBLEM_FORMAT_V3: SCHEDULE_FORMAT_V3,
    }[problem.format]
    complete = WarpgroupSchedule(schedule_format, result.lanes, candidates, result.times)
    verify_warpgroup_schedule(problem, complete)

    retained = list(candidates)
    for candidate in candidates:
        trial = tuple(item for item in retained if item != candidate)
        trial_schedule = WarpgroupSchedule(schedule_format, result.lanes, trial, result.times)
        control = _completion_event_edges(problem, trial_schedule)
        if all(
            _event_reachable(control, after, before)
            for after, before in _expand_relation(candidate, problem.loop.iterations)
        ):
            retained.remove(candidate)

    schedule = WarpgroupSchedule(schedule_format, result.lanes, tuple(retained), result.times)
    verify_warpgroup_schedule(problem, schedule)
    return schedule


__all__ = ["export_warpgroup_schedule"]
