"""Canonical synchronization export from one finite solve result."""

from __future__ import annotations

from .errors import WarpgroupVerificationError
from .model import (
    SCHEDULE_FORMAT,
    WarpgroupProblem,
    WarpgroupSchedule,
)
from .solve import WarpgroupSolveResult
from .verify import (
    _completion_event_edges,
    _event_reachable,
    _expand_relation,
    _required_completion_relations,
    _verify_warpgroup_schedule,
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
    complete = WarpgroupSchedule(SCHEDULE_FORMAT, result.lanes, candidates, result.times)
    _verify_warpgroup_schedule(problem, complete)

    retained = list(candidates)
    for candidate in candidates:
        trial = tuple(item for item in retained if item != candidate)
        trial_schedule = WarpgroupSchedule(SCHEDULE_FORMAT, result.lanes, trial, result.times)
        control = _completion_event_edges(problem, trial_schedule)
        if all(
            _event_reachable(control, after, before)
            for after, before in _expand_relation(candidate, problem.loop.iterations)
        ):
            retained.remove(candidate)

    schedule = WarpgroupSchedule(SCHEDULE_FORMAT, result.lanes, tuple(retained), result.times)
    _verify_warpgroup_schedule(problem, schedule)
    return schedule


__all__ = ["export_warpgroup_schedule"]
