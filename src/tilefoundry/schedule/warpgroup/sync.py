"""Canonical synchronization export from one finite solve result."""

from __future__ import annotations

from .errors import WarpgroupVerificationError
from .model import SCHEDULE_FORMAT, WarpgroupProblem, WarpgroupSchedule
from .solve import WarpgroupSolveResult
from .verify import (
    _control_edges,
    _expand_relation,
    _reachable,
    _required_shared_relations,
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
        for relation in _required_shared_relations(problem)
        if relation.distance < problem.loop.iterations
        and lane_by_operation[relation.after] != lane_by_operation[relation.before]
    )
    complete = WarpgroupSchedule(SCHEDULE_FORMAT, result.lanes, candidates, result.times)
    verify_warpgroup_schedule(problem, complete)

    retained = list(candidates)
    for candidate in candidates:
        trial = tuple(item for item in retained if item != candidate)
        trial_schedule = WarpgroupSchedule(SCHEDULE_FORMAT, result.lanes, trial, result.times)
        control = _control_edges(trial_schedule, problem.loop.iterations)
        if all(
            _reachable(control, after, before)
            for after, before in _expand_relation(candidate, problem.loop.iterations)
        ):
            retained.remove(candidate)

    schedule = WarpgroupSchedule(SCHEDULE_FORMAT, result.lanes, tuple(retained), result.times)
    verify_warpgroup_schedule(problem, schedule)
    return schedule


__all__ = ["export_warpgroup_schedule"]
