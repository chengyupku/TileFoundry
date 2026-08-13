"""Exact finite CP-SAT solving for an already closed warpgroup problem."""

from __future__ import annotations

import importlib
import math
from dataclasses import dataclass
from typing import Any, Literal

from .errors import (
    WarpgroupInfeasibleError,
    WarpgroupModelError,
    WarpgroupNoSolutionError,
    WarpgroupSolveError,
    WarpgroupValidationError,
)
from .expression import value_references
from .model import (
    MemorySpace,
    TimedOperation,
    WarpgroupLane,
    WarpgroupProblem,
)

SolveStatus = Literal["OPTIMAL", "FEASIBLE_NOT_PROVEN"]


@dataclass(frozen=True, slots=True)
class WarpgroupSolveResult:
    """The finite lane program and timing witness produced by M2."""

    status: SolveStatus
    lanes: tuple[WarpgroupLane, ...]
    times: tuple[TimedOperation, ...]

    def __post_init__(self) -> None:
        if self.status not in ("OPTIMAL", "FEASIBLE_NOT_PROVEN"):
            raise WarpgroupValidationError(f"invalid warpgroup solve status {self.status!r}")
        if type(self.lanes) is not tuple or not all(
            type(item) is WarpgroupLane for item in self.lanes
        ):
            raise WarpgroupValidationError("solve lanes must contain exact WarpgroupLane records")
        if type(self.times) is not tuple or not all(
            type(item) is TimedOperation for item in self.times
        ):
            raise WarpgroupValidationError("solve times must contain exact TimedOperation records")

    @property
    def makespan(self) -> int:
        """Derive the finite makespan from the timing witness."""
        return max((item.end for item in self.times), default=0)


def _value_owners(
    problem: WarpgroupProblem,
) -> tuple[dict[str, str], dict[str, MemorySpace]]:
    types = {item.id: item for item in problem.types}
    values = {item.id: types[item.type_id] for item in problem.inputs}
    owner: dict[str, str] = {}
    spaces: dict[str, MemorySpace] = {item.id: values[item.id].space for item in problem.inputs}
    for operation in problem.loop.ops:
        for output in operation.outputs:
            owner[output.id] = operation.id
            values[output.id] = types[output.type_id]
            spaces[output.id] = values[output.id].space
    for item in problem.loop.iter_args:
        values[item.id] = values[item.yield_value.id]
        spaces[item.id] = values[item.id].space
        owner[item.id] = owner[item.yield_value.id]
    return owner, spaces


def _users(problem: WarpgroupProblem) -> dict[str, tuple[str, ...]]:
    users: dict[str, set[str]] = {}
    for operation in problem.loop.ops:
        for output in operation.outputs:
            for value_id in value_references(output.expression):
                users.setdefault(value_id, set()).add(operation.id)
    return {value_id: tuple(sorted(operation_ids)) for value_id, operation_ids in users.items()}


def _register_locality_constraints(
    model: Any,
    problem: WarpgroupProblem,
    lane_vars: dict[str, Any],
) -> None:
    owner, spaces = _value_owners(problem)
    users = _users(problem)
    for value_id, space in spaces.items():
        if space is not MemorySpace.REGISTER:
            continue
        operations = set(users.get(value_id, ()))
        if value_id in owner:
            operations.add(owner[value_id])
        ordered = sorted(operations)
        for operation in ordered[1:]:
            model.Add(lane_vars[ordered[0]] == lane_vars[operation])


def _shared_lifetime_constraints(
    model: Any,
    problem: WarpgroupProblem,
    starts: dict[tuple[int, str], Any],
    ends: dict[tuple[int, str], Any],
) -> None:
    owner, spaces = _value_owners(problem)
    users = _users(problem)
    iterations = problem.loop.iterations
    output_ids = tuple(output.id for operation in problem.loop.ops for output in operation.outputs)
    for value_id in output_ids:
        if spaces[value_id] is not MemorySpace.SHARED:
            continue
        defining_operation = owner[value_id]
        # A fresh definition in iteration i+1 may reuse the allocation only
        # after every ordinary user of iteration i has completed.
        for user in users.get(value_id, ()):
            if user == defining_operation:
                continue
            for iteration in range(iterations - 1):
                model.Add(ends[(iteration, user)] <= starts[(iteration + 1, defining_operation)])

        # Iteration zero reads the boundary-ready external init.  From iteration
        # one onward, the previous body definition remains live through phi users
        # before the same loop-body position may overwrite it.
        for iter_arg in problem.loop.iter_args:
            if iter_arg.yield_value.id != value_id:
                continue
            for user in users.get(iter_arg.id, ()):
                if user == defining_operation:
                    continue
                for iteration in range(1, iterations):
                    model.Add(ends[(iteration, user)] <= starts[(iteration, defining_operation)])


def _solve_model(problem: WarpgroupProblem, timeout_seconds: float) -> WarpgroupSolveResult:
    """Build and solve the integer model; this is the only OR-Tools boundary."""
    cp_model: Any = importlib.import_module("ortools.sat.python.cp_model")

    if (
        type(timeout_seconds) not in (int, float)
        or not math.isfinite(timeout_seconds)
        or timeout_seconds <= 0
    ):
        raise WarpgroupValidationError("timeout_seconds must be positive")
    operations = tuple(sorted(problem.loop.ops, key=lambda item: item.id))
    operation_ids = tuple(operation.id for operation in operations)
    total_work = sum(operation.duration for operation in operations) * problem.loop.iterations
    horizon = max(total_work, 1)
    model = cp_model.CpModel()
    lane_vars = {
        operation_id: model.NewIntVar(0, problem.warp_groups - 1, f"lane_{operation_id}")
        for operation_id in operation_ids
    }
    starts: dict[tuple[int, str], Any] = {}
    ends: dict[tuple[int, str], Any] = {}
    lane_presence: dict[tuple[str, int], Any] = {}
    intervals_by_lane: dict[int, list[Any]] = {lane: [] for lane in range(problem.warp_groups)}
    intervals_by_resource: dict[str, list[Any]] = {
        resource.id: [] for resource in problem.resources
    }
    demands_by_resource: dict[str, list[int]] = {resource.id: [] for resource in problem.resources}
    for operation_id in operation_ids:
        presences = []
        for lane in range(problem.warp_groups):
            presence = model.NewBoolVar(f"placed_{operation_id}_{lane}")
            model.Add(lane_vars[operation_id] == lane).OnlyEnforceIf(presence)
            lane_presence[(operation_id, lane)] = presence
            presences.append(presence)
        model.AddExactlyOne(presences)
    for iteration in range(problem.loop.iterations):
        for operation in operations:
            key = (iteration, operation.id)
            start = model.NewIntVar(
                0, horizon - operation.duration, f"start_{iteration}_{operation.id}"
            )
            end = model.NewIntVar(operation.duration, horizon, f"end_{iteration}_{operation.id}")
            model.Add(end == start + operation.duration)
            starts[key] = start
            ends[key] = end
            for lane in range(problem.warp_groups):
                intervals_by_lane[lane].append(
                    model.NewOptionalIntervalVar(
                        start,
                        operation.duration,
                        end,
                        lane_presence[(operation.id, lane)],
                        f"lane_interval_{iteration}_{operation.id}_{lane}",
                    )
                )
            for demand in operation.resources:
                interval = model.NewIntervalVar(
                    start,
                    operation.duration,
                    end,
                    f"interval_{iteration}_{operation.id}_{demand.resource_id}",
                )
                intervals_by_resource[demand.resource_id].append(interval)
                demands_by_resource[demand.resource_id].append(demand.amount)

    # Each operation occupies its selected lane once per iteration.  This also
    # orders a lane containing only one loop-body operation across iterations.
    for operation_id in operation_ids:
        for iteration in range(problem.loop.iterations - 1):
            model.Add(ends[(iteration, operation_id)] <= starts[(iteration + 1, operation_id)])

    for lane in range(problem.warp_groups):
        model.AddNoOverlap(intervals_by_lane[lane])

    for resource in problem.resources:
        model.AddCumulative(
            intervals_by_resource[resource.id],
            demands_by_resource[resource.id],
            resource.capacity,
        )

    for index, first_id in enumerate(operation_ids):
        for second_id in operation_ids[index + 1 :]:
            same_lane = model.NewBoolVar(f"same_lane_{first_id}_{second_id}")
            first_before = model.NewBoolVar(f"order_{first_id}_{second_id}")
            second_before = model.NewBoolVar(f"order_{second_id}_{first_id}")
            model.Add(lane_vars[first_id] == lane_vars[second_id]).OnlyEnforceIf(same_lane)
            model.Add(lane_vars[first_id] != lane_vars[second_id]).OnlyEnforceIf(same_lane.Not())
            model.Add(first_before + second_before == 1).OnlyEnforceIf(same_lane)
            model.Add(first_before == 0).OnlyEnforceIf(same_lane.Not())
            model.Add(second_before == 0).OnlyEnforceIf(same_lane.Not())
            for iteration in range(problem.loop.iterations):
                model.Add(
                    ends[(iteration, first_id)] <= starts[(iteration, second_id)]
                ).OnlyEnforceIf((same_lane, first_before))
                model.Add(
                    ends[(iteration, second_id)] <= starts[(iteration, first_id)]
                ).OnlyEnforceIf((same_lane, second_before))
                if iteration + 1 < problem.loop.iterations:
                    model.Add(
                        ends[(iteration, second_id)] <= starts[(iteration + 1, first_id)]
                    ).OnlyEnforceIf((same_lane, first_before))
                    model.Add(
                        ends[(iteration, first_id)] <= starts[(iteration + 1, second_id)]
                    ).OnlyEnforceIf((same_lane, second_before))

    for dependency in problem.dependencies():
        for iteration in range(problem.loop.iterations - dependency.distance):
            model.Add(
                ends[(iteration, dependency.after)]
                <= starts[(iteration + dependency.distance, dependency.before)]
            )
    _register_locality_constraints(model, problem, lane_vars)
    _shared_lifetime_constraints(model, problem, starts, ends)
    makespan = model.NewIntVar(0, horizon, "makespan")
    model.AddMaxEquality(makespan, tuple(ends.values()))
    model.Minimize(makespan)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(timeout_seconds)
    solver.parameters.num_search_workers = 1
    solver.parameters.random_seed = 0
    solver.parameters.cp_model_presolve = True
    solver.parameters.symmetry_level = 2
    status = solver.Solve(model)
    if status == cp_model.INFEASIBLE:
        raise WarpgroupInfeasibleError("warpgroup problem is infeasible")
    if status == cp_model.UNKNOWN:
        raise WarpgroupNoSolutionError("warpgroup solver stopped without an incumbent")
    if status == cp_model.MODEL_INVALID:
        raise WarpgroupModelError("warpgroup CP-SAT model is invalid")
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise WarpgroupModelError(f"unexpected warpgroup solver status {status}")

    lane_groups: dict[int, list[str]] = {lane: [] for lane in range(problem.warp_groups)}
    for operation_id in operation_ids:
        lane_groups[solver.Value(lane_vars[operation_id])].append(operation_id)
    for group in lane_groups.values():
        group.sort(key=lambda operation_id: (solver.Value(starts[(0, operation_id)]), operation_id))
    non_empty = sorted(
        (group for group in lane_groups.values() if group),
        key=lambda group: min(operation_ids.index(operation_id) for operation_id in group),
    )
    groups = non_empty + [[] for _ in range(problem.warp_groups - len(non_empty))]
    lanes = tuple(WarpgroupLane(tuple(group)) for group in groups)
    times = tuple(
        TimedOperation(
            iteration,
            operation_id,
            solver.Value(starts[(iteration, operation_id)]),
            solver.Value(ends[(iteration, operation_id)]),
        )
        for iteration in range(problem.loop.iterations)
        for operation_id in operation_ids
    )
    return WarpgroupSolveResult(
        "OPTIMAL" if status == cp_model.OPTIMAL else "FEASIBLE_NOT_PROVEN",
        lanes,
        tuple(sorted(times)),
    )


def solve_warpgroup_problem(
    problem: WarpgroupProblem, *, timeout_seconds: float = 60.0
) -> WarpgroupSolveResult:
    """Solve one closed problem without importing its cost provider or target."""
    if type(problem) is not WarpgroupProblem:
        raise WarpgroupValidationError("solve requires an exact WarpgroupProblem")
    try:
        return _solve_model(problem, timeout_seconds)
    except (WarpgroupSolveError, WarpgroupValidationError):
        raise
    except Exception as error:
        raise WarpgroupModelError(f"warpgroup solve failed: {error}") from error


__all__ = [
    "SolveStatus",
    "WarpgroupSolveResult",
    "solve_warpgroup_problem",
]
