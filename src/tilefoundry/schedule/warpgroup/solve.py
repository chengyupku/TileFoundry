"""Compact periodic CP-SAT solving for a closed warpgroup problem."""

from __future__ import annotations

import importlib
import math
import time
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
    ProblemOperation,
    TimedOperation,
    WarpgroupLane,
    WarpgroupProblem,
)

SolveStatus = Literal["OPTIMAL", "FEASIBLE_NOT_PROVEN"]


@dataclass(frozen=True, slots=True)
class WarpgroupSolveResult:
    """The finite lane program and timing witness produced by the solver."""

    status: SolveStatus
    lanes: tuple[WarpgroupLane, ...]
    times: tuple[TimedOperation, ...]
    #: The steady-state period, for a solver that schedules one. Deriving it
    #: from `times` needs three iterations, which a short loop does not have.
    initiation_interval: int | None = None

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
        if self.initiation_interval is not None and (
            type(self.initiation_interval) is not int or self.initiation_interval <= 0
        ):
            raise WarpgroupValidationError("initiation interval must be a positive integer")

    @property
    def makespan(self) -> int:
        """Derive the finite makespan from the timing witness."""
        return max((item.completion for item in self.times), default=0)


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


def _add_compact_periodic_resource_constraints(
    model: Any,
    problem: WarpgroupProblem,
    operations: tuple[ProblemOperation, ...],
    prologue_starts: dict[str, Any],
    offsets: dict[str, Any],
    initiation_interval: Any,
    horizon: int,
    ii_lower_bound: int,
) -> None:
    """Constrain finite prologue and infinitely repeated body windows."""
    if not problem.resources:
        return

    capacities = {resource.id: resource.capacity for resource in problem.resources}
    copy_bound = math.ceil(horizon / ii_lower_bound)
    periodic_intervals: dict[str, list[Any]] = {resource.id: [] for resource in problem.resources}
    periodic_demands: dict[str, list[int]] = {resource.id: [] for resource in problem.resources}
    boundary_intervals: dict[str, list[Any]] = {resource.id: [] for resource in problem.resources}
    boundary_demands: dict[str, list[int]] = {resource.id: [] for resource in problem.resources}

    def add_window(
        intervals: dict[str, list[Any]],
        demands: dict[str, list[int]],
        operation: ProblemOperation,
        window_index: int,
        start: Any,
        name: str,
        *,
        materialize_start: bool = False,
    ) -> None:
        window = operation.resource_windows[window_index]
        if materialize_start:
            start_variable = model.NewIntVar(
                -copy_bound * horizon,
                (copy_bound + 1) * horizon,
                f"{name}_start",
            )
            model.Add(start_variable == start)
            start = start_variable
        intervals[window.resource_id].append(
            model.NewIntervalVar(start, window.duration, start + window.duration, name)
        )
        demands[window.resource_id].append(window.amount)

    for operation in operations:
        for window_index, window in enumerate(operation.resource_windows):
            prologue_start = prologue_starts[operation.id]
            add_window(
                boundary_intervals,
                boundary_demands,
                operation,
                window_index,
                prologue_start,
                f"prologue_resource_{operation.id}_{window.resource_id}_{window_index}",
            )
            if problem.loop.iterations < 2:
                continue

            base_start = offsets[operation.id]
            # Every base window lies in [0, horizon].  Any infinite-period copy
            # intersecting [0, II) therefore has shift in [-copy_bound, 0].
            for shift in range(-copy_bound, 1):
                add_window(
                    periodic_intervals,
                    periodic_demands,
                    operation,
                    window_index,
                    base_start + shift * initiation_interval,
                    f"periodic_resource_{operation.id}_{window.resource_id}_{window_index}_{shift}",
                    materialize_start=True,
                )

            # Only these earliest real body copies can intersect a prologue
            # window, whose end is also bounded by horizon.
            for iteration in range(1, copy_bound + 1):
                add_window(
                    boundary_intervals,
                    boundary_demands,
                    operation,
                    window_index,
                    base_start + iteration * initiation_interval,
                    f"boundary_resource_{operation.id}_{window.resource_id}_{window_index}_{iteration}",
                    materialize_start=True,
                )

    for resource_id, capacity in capacities.items():
        if periodic_intervals[resource_id]:
            model.AddCumulative(
                periodic_intervals[resource_id], periodic_demands[resource_id], capacity
            )
        if boundary_intervals[resource_id]:
            model.AddCumulative(
                boundary_intervals[resource_id], boundary_demands[resource_id], capacity
            )


def _resource_lower_bound(
    problem: WarpgroupProblem, operations: tuple[ProblemOperation, ...]
) -> int:
    """The period a resource cannot deliver less than, over one iteration."""
    if not problem.resources:
        return 0
    capacity = {item.id: item.capacity for item in problem.resources}
    held: dict[str, int] = {}
    for operation in operations:
        for window in operation.resource_windows:
            held[window.resource_id] = (
                held.get(window.resource_id, 0) + window.duration * window.amount
            )
    return max(
        (-(-total // capacity[resource]) for resource, total in held.items()), default=0
    )


def _materialize_periodic_times(
    operations: tuple[ProblemOperation, ...],
    iterations: int,
    prologue_starts: dict[str, int],
    offsets: dict[str, int],
    initiation_interval: int,
) -> tuple[TimedOperation, ...]:
    """Materialize finite prologue and periodic body rows after solving."""
    times: list[TimedOperation] = []
    for iteration in range(iterations):
        for operation in operations:
            start = (
                prologue_starts[operation.id]
                if iteration == 0
                else offsets[operation.id] + iteration * initiation_interval
            )
            times.append(
                TimedOperation(
                    iteration,
                    operation.id,
                    start,
                    issue_end=start + operation.issue_duration,
                    completion=start + operation.completion_latency,
                )
            )
    return tuple(sorted(times))


def _solve_model(
    problem: WarpgroupProblem, timeout_seconds: float, search_workers: int = 1
) -> WarpgroupSolveResult:
    """Solve with static body timing and finite boundary instances."""
    cp_model: Any = importlib.import_module("ortools.sat.python.cp_model")
    if (
        type(timeout_seconds) not in (int, float)
        or not math.isfinite(timeout_seconds)
        or timeout_seconds <= 0
    ):
        raise WarpgroupValidationError("timeout_seconds must be positive")
    operations = tuple(sorted(problem.loop.ops, key=lambda item: item.id))
    operation_ids = tuple(operation.id for operation in operations)
    operation_by_id = {operation.id: operation for operation in operations}
    static_span = max(
        sum(operation.completion_latency for operation in operations),
        sum(operation.issue_duration for operation in operations),
    ) + max(operation.completion_latency for operation in operations)
    horizon = max(static_span, 1)
    ii_lower_bound = max(
        max(operation.issue_duration for operation in operations),
        _resource_lower_bound(problem, operations),
    )
    model = cp_model.CpModel()
    initiation_interval = model.NewIntVar(1, horizon, "II")
    # The bound sizes the periodic resource model -- `copy_bound` divides by it
    # -- so it has to hold of the variable too, or a raised bound builds fewer
    # window copies than a shorter period would need.
    model.Add(initiation_interval >= ii_lower_bound)
    offsets = {
        operation.id: model.NewIntVar(
            0, horizon - operation.completion_latency, f"body_offset_{operation.id}"
        )
        for operation in operations
    }
    prologue_starts = {
        operation.id: model.NewIntVar(
            0, horizon - operation.completion_latency, f"prologue_start_{operation.id}"
        )
        for operation in operations
    }

    def body_start(operation_id: str, iteration: int) -> Any:
        return offsets[operation_id] + iteration * initiation_interval

    def prologue_completion(operation: ProblemOperation) -> Any:
        return prologue_starts[operation.id] + operation.completion_latency

    def body_completion(operation: ProblemOperation) -> Any:
        return offsets[operation.id] + operation.completion_latency

    for operation in operations:
        model.Add(initiation_interval >= operation.issue_duration)

    lane_order_terms: list[Any] = []
    for index, first_id in enumerate(operation_ids):
        for second_id in operation_ids[index + 1 :]:
            first = operation_by_id[first_id]
            second = operation_by_id[second_id]
            if first.warp_group != second.warp_group:
                continue
            first_before = model.NewBoolVar(f"order_{first_id}_{second_id}")
            second_before = model.NewBoolVar(f"order_{second_id}_{first_id}")
            lane_order_terms.append(first_before)
            model.Add(first_before + second_before == 1)
            model.Add(offsets[first_id] + first.issue_duration <= offsets[second_id]).OnlyEnforceIf(
                first_before
            )
            model.Add(
                offsets[second_id] + second.issue_duration <= offsets[first_id]
            ).OnlyEnforceIf(second_before)
            model.Add(
                offsets[second_id] + second.issue_duration
                <= offsets[first_id] + initiation_interval
            ).OnlyEnforceIf(first_before)
            model.Add(
                offsets[first_id] + first.issue_duration <= offsets[second_id] + initiation_interval
            ).OnlyEnforceIf(second_before)

            model.Add(
                prologue_starts[first_id] + first.issue_duration <= prologue_starts[second_id]
            ).OnlyEnforceIf(first_before)
            model.Add(
                prologue_starts[second_id] + second.issue_duration <= prologue_starts[first_id]
            ).OnlyEnforceIf(second_before)
            if problem.loop.iterations >= 2:
                model.Add(
                    prologue_starts[second_id] + second.issue_duration <= body_start(first_id, 1)
                ).OnlyEnforceIf(first_before)
                model.Add(
                    prologue_starts[first_id] + first.issue_duration <= body_start(second_id, 1)
                ).OnlyEnforceIf(second_before)
    if problem.loop.iterations >= 2:
        for operation in operations:
            model.Add(
                prologue_starts[operation.id] + operation.issue_duration
                <= body_start(operation.id, 1)
            )

    for dependency in problem.dependencies():
        if dependency.distance >= problem.loop.iterations:
            continue
        after = operation_by_id[dependency.after]
        before = operation_by_id[dependency.before]
        destination_iteration = dependency.distance
        if destination_iteration == 0:
            model.Add(prologue_completion(after) <= prologue_starts[before.id])
        else:
            model.Add(prologue_completion(after) <= body_start(before.id, destination_iteration))
        if problem.loop.iterations >= dependency.distance + 2:
            model.Add(
                body_completion(after)
                <= offsets[before.id] + dependency.distance * initiation_interval
            )

    if problem.loop.iterations >= 2:
        owner, spaces = _value_owners(problem)
        users = _users(problem)
        for value_id, space in spaces.items():
            if space is not MemorySpace.SHARED:
                continue
            defining_operation = owner.get(value_id)
            if defining_operation is None:
                continue
            for user in users.get(value_id, ()):
                if user != defining_operation:
                    model.Add(
                        prologue_completion(operation_by_id[user])
                        <= body_start(defining_operation, 1)
                    )
                    if problem.loop.iterations >= 3:
                        model.Add(
                            body_completion(operation_by_id[user])
                            <= offsets[defining_operation] + initiation_interval
                        )
            for iter_arg in problem.loop.iter_args:
                if iter_arg.yield_value.id != value_id:
                    continue
                for user in users.get(iter_arg.id, ()):
                    if user != defining_operation:
                        # This body relation protects every carried reuse after
                        # iteration zero; the external init itself is boundary-ready.
                        model.Add(
                            body_completion(operation_by_id[user]) <= offsets[defining_operation]
                        )

    owner, spaces = _value_owners(problem)
    users = _users(problem)
    for value_id, space in spaces.items():
        if space is not MemorySpace.REGISTER:
            continue
        component = set(users.get(value_id, ()))
        if value_id in owner:
            component.add(owner[value_id])
        groups = {operation_by_id[operation_id].warp_group for operation_id in component}
        if len(groups) > 1:
            raise WarpgroupInfeasibleError(f"register value {value_id!r} crosses fixed warp groups")

    _add_compact_periodic_resource_constraints(
        model,
        problem,
        operations,
        prologue_starts,
        offsets,
        initiation_interval,
        horizon,
        ii_lower_bound,
    )

    makespan_upper = horizon * problem.loop.iterations + max(
        operation.completion_latency for operation in operations
    )
    last_iteration = problem.loop.iterations - 1
    makespan = model.NewIntVar(0, makespan_upper, "makespan")
    completion_exprs = tuple(prologue_completion(operation) for operation in operations)
    if problem.loop.iterations >= 2:
        completion_exprs += tuple(
            body_start(operation.id, last_iteration) + operation.completion_latency
            for operation in operations
        )
    model.AddMaxEquality(makespan, completion_exprs)

    deadline = time.monotonic() + float(timeout_seconds)

    def new_solver() -> Any:
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = max(deadline - time.monotonic(), 0.001)
        solver.parameters.num_search_workers = search_workers
        solver.parameters.random_seed = 0
        solver.parameters.cp_model_presolve = True
        solver.parameters.symmetry_level = 2
        return solver

    model.Minimize(makespan)
    solver = new_solver()
    status = solver.Solve(model)
    if status == cp_model.INFEASIBLE:
        raise WarpgroupInfeasibleError("warpgroup problem is infeasible")
    if status == cp_model.UNKNOWN:
        raise WarpgroupNoSolutionError("warpgroup solver stopped without an incumbent")
    if status == cp_model.MODEL_INVALID:
        raise WarpgroupModelError("warpgroup CP-SAT model is invalid")
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise WarpgroupModelError(f"unexpected warpgroup solver status {status}")

    final_status = status
    if status == cp_model.OPTIMAL:
        best_makespan = solver.Value(makespan)
        model.Add(makespan == best_makespan)
        model.ClearObjective()
        model.Minimize(initiation_interval)
        ii_solver = new_solver()
        ii_status = ii_solver.Solve(model)
        if ii_status == cp_model.UNKNOWN:
            final_status = cp_model.FEASIBLE
        elif ii_status == cp_model.INFEASIBLE:
            raise WarpgroupModelError("warpgroup II tie-break model is infeasible")
        elif ii_status == cp_model.MODEL_INVALID:
            raise WarpgroupModelError("warpgroup II tie-break model is invalid")
        elif ii_status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            raise WarpgroupModelError(f"unexpected warpgroup II tie-break status {ii_status}")
        else:
            solver = ii_solver
            final_status = ii_status
        if ii_status == cp_model.OPTIMAL:
            model.Add(initiation_interval == solver.Value(initiation_interval))
            order_weight = len(lane_order_terms) + 1
            offset_bound = len(operations) * horizon
            tie_break = (
                sum(prologue_starts.values())
                * (offset_bound * order_weight + len(lane_order_terms) + 1)
                + sum(offsets.values()) * order_weight
                + sum(lane_order_terms, 0)
            )
            model.ClearObjective()
            model.Minimize(tie_break)
            tie_solver = new_solver()
            tie_status = tie_solver.Solve(model)
            if tie_status == cp_model.UNKNOWN:
                final_status = cp_model.FEASIBLE
            elif tie_status == cp_model.INFEASIBLE:
                raise WarpgroupModelError("warpgroup tie-break model is infeasible")
            elif tie_status == cp_model.MODEL_INVALID:
                raise WarpgroupModelError("warpgroup tie-break model is invalid")
            elif tie_status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
                raise WarpgroupModelError(f"unexpected warpgroup tie-break status {tie_status}")
            else:
                solver = tie_solver
                final_status = tie_status

    lane_groups: dict[int, list[str]] = {lane: [] for lane in range(problem.warp_groups)}
    for operation in operations:
        lane_groups[operation.warp_group].append(operation.id)
    for group in lane_groups.values():
        group.sort(key=lambda operation_id: (solver.Value(offsets[operation_id]), operation_id))
    lanes = tuple(WarpgroupLane(tuple(lane_groups[lane])) for lane in range(problem.warp_groups))
    ii = solver.Value(initiation_interval)
    times = _materialize_periodic_times(
        operations,
        problem.loop.iterations,
        {operation.id: solver.Value(prologue_starts[operation.id]) for operation in operations},
        {operation.id: solver.Value(offsets[operation.id]) for operation in operations},
        ii,
    )
    return WarpgroupSolveResult(
        "OPTIMAL" if final_status == cp_model.OPTIMAL else "FEASIBLE_NOT_PROVEN",
        lanes,
        tuple(sorted(times)),
        ii,
    )


def solve_warpgroup_problem(
    problem: WarpgroupProblem, *, timeout_seconds: float = 60.0, search_workers: int = 1
) -> WarpgroupSolveResult:
    """Solve one closed problem without importing its cost provider or target.

    ``search_workers`` above one trades the reproducible arrangement for speed:
    CP-SAT returns whichever of several optimal arrangements a racing worker
    proves first. One worker keeps the answer reproducible and is the default.
    """
    if type(search_workers) is not int or search_workers < 1:
        raise WarpgroupValidationError("search_workers must be a positive integer")
    if type(problem) is not WarpgroupProblem:
        raise WarpgroupValidationError("solve requires an exact WarpgroupProblem")
    try:
        return _solve_model(problem, timeout_seconds, search_workers)
    except (WarpgroupSolveError, WarpgroupValidationError):
        raise
    except Exception as error:
        raise WarpgroupModelError(f"warpgroup solve failed: {error}") from error


__all__ = [
    "SolveStatus",
    "WarpgroupSolveResult",
    "solve_warpgroup_problem",
]
