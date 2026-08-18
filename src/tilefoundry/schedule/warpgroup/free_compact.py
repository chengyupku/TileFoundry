"""Compact periodic scheduling with ownership left to the search.

Two solvers already live here and neither covers this. ``_solve_model`` searches
which lane owns each operation but unrolls the loop to a finite makespan;
``_solve_compact_fixed_owner_model`` schedules one steady-state period but takes
ownership as input. A program whose costs come from measurement wants both: the
period is what a long loop actually costs, and the assignment is what nobody
should have to write by hand.

The difference against the fixed-owner model is four changes:

1. a lane variable per operation, as the finite model already builds;
2. the pairwise order disjunction is gated on a reified ``same_lane`` rather
   than filtered by a static owner;
3. register locality becomes a constraint instead of a static check, so a value
   read by two lanes moves its readers together rather than being refused;
4. lanes are read back from the solved variables.

Everything else -- the periodic resource windows, the shared lifetimes, the
dependence relations, the prologue instances -- is the fixed-owner model's, and
this returns the same ``WarpgroupSolveResult``.

Two capacities the problem format does not carry can be supplied by the caller,
because both are properties of the target rather than of the program:

* ``registers`` -- what each loop-carried value costs its owner, so that lane
  allocations are values ``setmaxnreg`` accepts and fit the register file;
* ``shared`` -- shared memory as a capacity in bytes held over each value's
  lifetime. Stating it this way is what makes buffer reuse fall out instead of
  needing a mechanism: two values whose lifetimes do not meet simply fit in one
  allocation's worth of the capacity. It is the capacity half of what
  ``_shared_lifetime_constraints`` states as ordering.
"""

from __future__ import annotations

import importlib
import math
from typing import Any, Mapping, Sequence

from .errors import (WarpgroupInfeasibleError, WarpgroupModelError,
                     WarpgroupNoSolutionError, WarpgroupValidationError)
from .model import (MemorySpace, ProblemOperation, WarpgroupLane,
                    WarpgroupProblem)
from .solve import (WarpgroupSolveResult,
                    _add_compact_periodic_resource_constraints,
                    _materialize_periodic_times, _resource_lower_bound, _users,
                    _value_owners)

#: ``setmaxnreg`` moves registers between warp groups, which is what lets a
#: producer lane be cheap, but ptxas accepts only a multiple of 8 in [24, 256].
#: A lane's allocation is therefore not a free integer but one of thirty values,
#: and the SM's 65536 registers over 128 threads a lane cap the sum at 512.
#:
#: The granularity is why three lanes reach 64512 of 65536: 65536/384 is 170.67
#: and the largest multiple of 8 below it is 168.
REGISTER_STEP = 8
REGISTER_MIN, REGISTER_MAX = 24, 256
LANE_THREADS = 128
REGISTER_BUDGET = 65536 // LANE_THREADS
#: What a lane needs beyond the values it carries across the loop. Short-lived
#: temporaries are not tracked, so this stands in for them.
TEMPORARY_REGISTERS_PER_THREAD = 32


def solve_free_compact(
    problem: WarpgroupProblem,
    *,
    timeout_seconds: float = 60.0,
    search_workers: int = 1,
    registers: Mapping[str, int] | None = None,
    shared: Mapping[str, Any] | None = None,
    pin: Mapping[str, int] | None = None,
) -> WarpgroupSolveResult:
    """Schedule one steady-state period and choose the lane assignment with it.

    ``registers`` maps a loop-carried value's producing operation to the
    registers per thread it holds; ``shared`` is ``{"capacity": bytes,
    "values": {name: (writer, readers, bytes)}}`` with an optional ``movable``
    of ``{name: (readers, bytes, per_thread)}`` for values the search may take
    out of shared memory and into registers. ``pin`` fixes chosen operations to
    chosen lanes, which is how one assignment is priced against the search's.

    ``search_workers`` above one trades the reproducible arrangement for speed,
    as it does for `solve_warpgroup_problem`.
    """
    cp_model: Any = importlib.import_module("ortools.sat.python.cp_model")
    if type(problem) is not WarpgroupProblem:
        raise WarpgroupValidationError("problem must be a WarpgroupProblem")
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

    # Every operation is issued on some lane and a lane issues one at a time, so
    # the total issue work spread over the lanes bounds the period. Independent
    # of the resource bound: one is about engines, this is about issue ports,
    # and either can be the binding one.
    issue_floor = -(-sum(op.issue_duration for op in operations) // problem.warp_groups)
    ii_lower_bound = max(
        max(operation.issue_duration for operation in operations),
        _resource_lower_bound(problem, operations),
        issue_floor,
    )

    model = cp_model.CpModel()
    initiation_interval = model.NewIntVar(1, horizon, "II")
    offsets = {
        operation.id: model.NewIntVar(
            0, horizon - operation.completion_latency, f"start_offset_{operation.id}"
        )
        for operation in operations
    }
    prologue_starts = {
        operation.id: model.NewIntVar(
            0, horizon - operation.completion_latency, f"prologue_start_{operation.id}"
        )
        for operation in operations
    }
    # (1) ownership becomes a decision, exactly as the finite model states it.
    lane_vars = {
        operation.id: model.NewIntVar(0, problem.warp_groups - 1, f"lane_{operation.id}")
        for operation in operations
    }
    for operation_id, lane in (pin or {}).items():
        if operation_id not in lane_vars:
            raise WarpgroupValidationError(f"pinned operation {operation_id!r} is not in the loop")
        if not 0 <= lane < problem.warp_groups:
            raise WarpgroupValidationError(f"pinned lane {lane} is outside the warp groups")
        model.Add(lane_vars[operation_id] == lane)

    def body_start(operation_id: str, iteration: int) -> Any:
        return offsets[operation_id] + iteration * initiation_interval

    def prologue_completion(operation: ProblemOperation) -> Any:
        return prologue_starts[operation.id] + operation.completion_latency

    def body_completion(operation: ProblemOperation) -> Any:
        return offsets[operation.id] + operation.completion_latency

    model.Add(initiation_interval >= ii_lower_bound)

    for index, first_id in enumerate(operation_ids):
        for second_id in operation_ids[index + 1 :]:
            first, second = operation_by_id[first_id], operation_by_id[second_id]
            # (2) the static owner filter becomes a reified literal.
            same_lane = model.NewBoolVar(f"same_lane_{first_id}_{second_id}")
            model.Add(lane_vars[first_id] == lane_vars[second_id]).OnlyEnforceIf(same_lane)
            model.Add(lane_vars[first_id] != lane_vars[second_id]).OnlyEnforceIf(same_lane.Not())
            first_before = model.NewBoolVar(f"order_{first_id}_{second_id}")
            second_before = model.NewBoolVar(f"order_{second_id}_{first_id}")
            model.Add(first_before + second_before == 1).OnlyEnforceIf(same_lane)
            model.Add(first_before == 0).OnlyEnforceIf(same_lane.Not())
            model.Add(second_before == 0).OnlyEnforceIf(same_lane.Not())
            ahead = (same_lane, first_before)
            behind = (same_lane, second_before)
            model.Add(
                offsets[first_id] + first.issue_duration <= offsets[second_id]
            ).OnlyEnforceIf(ahead)
            model.Add(
                offsets[second_id] + second.issue_duration <= offsets[first_id]
            ).OnlyEnforceIf(behind)
            model.Add(
                offsets[second_id] + second.issue_duration
                <= offsets[first_id] + initiation_interval
            ).OnlyEnforceIf(ahead)
            model.Add(
                offsets[first_id] + first.issue_duration
                <= offsets[second_id] + initiation_interval
            ).OnlyEnforceIf(behind)
            model.Add(
                prologue_starts[first_id] + first.issue_duration <= prologue_starts[second_id]
            ).OnlyEnforceIf(ahead)
            model.Add(
                prologue_starts[second_id] + second.issue_duration <= prologue_starts[first_id]
            ).OnlyEnforceIf(behind)
            if problem.loop.iterations >= 2:
                model.Add(
                    prologue_starts[second_id] + second.issue_duration <= body_start(first_id, 1)
                ).OnlyEnforceIf(ahead)
                model.Add(
                    prologue_starts[first_id] + first.issue_duration <= body_start(second_id, 1)
                ).OnlyEnforceIf(behind)

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
        if dependency.distance == 0:
            model.Add(prologue_completion(after) <= prologue_starts[before.id])
        else:
            model.Add(prologue_completion(after) <= body_start(before.id, dependency.distance))
        if problem.loop.iterations >= dependency.distance + 2:
            model.Add(
                body_completion(after)
                <= offsets[before.id] + dependency.distance * initiation_interval
            )

    if problem.loop.iterations >= 2:
        _periodic_shared_lifetimes(
            model, problem, operation_by_id, offsets, prologue_starts, initiation_interval
        )

    # (3) the static cross-group check becomes the constraint that prevents it.
    _register_locality_constraints(model, problem, lane_vars)

    on_lane: dict[tuple[str, int], Any] = {}

    def placed(operation_id: str, lane: int) -> Any:
        """A literal for `operation_id runs on lane`, built once."""
        key = (operation_id, lane)
        if key not in on_lane:
            flag = model.NewBoolVar(f"{operation_id}@{lane}")
            model.Add(lane_vars[operation_id] == lane).OnlyEnforceIf(flag)
            model.Add(lane_vars[operation_id] != lane).OnlyEnforceIf(flag.Not())
            on_lane[key] = flag
        return on_lane[key]

    # The average over lanes is a bound; per lane it is a constraint, and the
    # two differ whenever the work does not divide evenly.
    for lane in range(problem.warp_groups):
        model.Add(
            sum(operation.issue_duration * placed(operation.id, lane) for operation in operations)
            <= initiation_interval
        )

    register_load: list[tuple[int, int, Any]] = []
    if shared:
        register_load = _shared_capacity_constraints(
            model, cp_model, problem, operations, offsets, horizon, shared, placed
        )
    if registers:
        _register_capacity_constraints(
            model, cp_model, problem, registers, register_load, placed
        )

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
    makespan = model.NewIntVar(0, makespan_upper, "makespan")
    completion_exprs = tuple(prologue_completion(operation) for operation in operations)
    if problem.loop.iterations >= 2:
        last = problem.loop.iterations - 1
        completion_exprs += tuple(
            body_start(operation.id, last) + operation.completion_latency
            for operation in operations
        )
    model.AddMaxEquality(makespan, completion_exprs)
    model.Minimize(makespan)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(timeout_seconds)
    solver.parameters.num_search_workers = search_workers
    solver.parameters.random_seed = 0
    solver.parameters.symmetry_level = 2
    status = solver.Solve(model)
    if status == cp_model.INFEASIBLE:
        raise WarpgroupInfeasibleError("warpgroup problem is infeasible")
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        if status == cp_model.MODEL_INVALID:
            raise WarpgroupModelError("warpgroup CP-SAT model is invalid")
        raise WarpgroupNoSolutionError("warpgroup solver stopped without an incumbent")

    lane_groups: dict[int, list[str]] = {lane: [] for lane in range(problem.warp_groups)}
    for operation in operations:
        lane_groups[solver.Value(lane_vars[operation.id])].append(operation.id)
    order = {operation.id: solver.Value(offsets[operation.id]) for operation in operations}
    for members in lane_groups.values():
        members.sort(key=lambda operation_id: (order[operation_id], operation_id))
    lanes = tuple(WarpgroupLane(tuple(lane_groups[lane])) for lane in range(problem.warp_groups))
    period = solver.Value(initiation_interval)
    times = _materialize_periodic_times(
        operations,
        problem.loop.iterations,
        {
            operation.id: solver.Value(prologue_starts[operation.id])
            for operation in operations
        },
        order,
        period,
    )
    return WarpgroupSolveResult(
        "OPTIMAL" if status == cp_model.OPTIMAL else "FEASIBLE_NOT_PROVEN",
        lanes,
        tuple(sorted(times)),
        period,
    )


def _register_locality_constraints(
    model: Any, problem: WarpgroupProblem, lane_vars: dict[str, Any]
) -> None:
    """A register value and every operation touching it share one lane.

    `_solve_model` states this as a static check, which is available to it
    because its ownership is given. Searched ownership has to constrain it
    instead: a value read by two lanes does not fail, it moves its readers
    together.
    """
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


def _periodic_shared_lifetimes(
    model: Any,
    problem: WarpgroupProblem,
    operation_by_id: dict[str, ProblemOperation],
    offsets: dict[str, Any],
    prologue_starts: dict[str, Any],
    initiation_interval: Any,
) -> None:
    """`_shared_lifetime_constraints`, against a period rather than an unrolling."""
    owner, spaces = _value_owners(problem)
    users = _users(problem)
    for value_id, space in spaces.items():
        if space is not MemorySpace.SHARED:
            continue
        defining = owner.get(value_id)
        if defining is None:
            continue
        for user in users.get(value_id, ()):
            if user == defining:
                continue
            model.Add(
                prologue_starts[user] + operation_by_id[user].completion_latency
                <= offsets[defining] + initiation_interval
            )
            if problem.loop.iterations >= 3:
                model.Add(
                    offsets[user] + operation_by_id[user].completion_latency
                    <= offsets[defining] + initiation_interval
                )
        for iter_arg in problem.loop.iter_args:
            if iter_arg.yield_value.id != value_id:
                continue
            for user in users.get(iter_arg.id, ()):
                if user == defining:
                    continue
                model.Add(
                    offsets[user] + operation_by_id[user].completion_latency
                    <= offsets[defining]
                )


def _shared_capacity_constraints(
    model: Any,
    cp_model: Any,
    problem: WarpgroupProblem,
    operations: Sequence[ProblemOperation],
    offsets: dict[str, Any],
    horizon: int,
    shared: Mapping[str, Any],
    placed: Any,
) -> list[tuple[int, int, Any]]:
    """Shared memory as bytes held over each value's lifetime.

    Returns what the movable values cost in registers, which the register
    constraint charges to whichever lane ends up reading them.
    """
    capacity = shared["capacity"]
    issue_of = {operation.id: operation.issue_duration for operation in operations}
    intervals, demands = [], []

    for name, (writer, readers, size) in shared["values"].items():
        start = offsets[writer] if writer is not None else model.NewConstant(0)
        if readers:
            end = model.NewIntVar(0, horizon, f"dead_{name}")
            model.AddMaxEquality(end, [offsets[r] + issue_of[r] for r in readers])
        else:
            end = start
        span = model.NewIntVar(0, horizon, f"live_{name}")
        model.Add(span == end - start)
        intervals.append(model.NewIntervalVar(start, span, end, f"smem_{name}"))
        demands.append(size)

    # A value the caller lets the search move out of shared memory. It costs
    # registers on whichever lane reads it and frees its bytes for the whole
    # period. FlashMLA fits its own body this way: one tile of Q lives in
    # registers so that a second probability buffer has room.
    register_load: list[tuple[int, int, Any]] = []
    for name, (readers, size, per_thread) in shared.get("movable", {}).items():
        stays = model.NewBoolVar(f"{name}_in_shared")
        intervals.append(model.NewOptionalIntervalVar(0, horizon, horizon, stays, f"smem_{name}"))
        demands.append(size)
        for lane in range(problem.warp_groups):
            for reader in readers:
                here = model.NewBoolVar(f"{name}_reg_{lane}_{reader}")
                model.AddMinEquality(here, [placed(reader, lane), stays.Not()])
                register_load.append((lane, per_thread, here))

    model.AddCumulative(intervals, demands, capacity)
    return register_load


def _register_capacity_constraints(
    model: Any,
    cp_model: Any,
    problem: WarpgroupProblem,
    registers: Mapping[str, int],
    register_load: Sequence[tuple[int, int, Any]],
    placed: Any,
) -> None:
    """Lane allocations that `setmaxnreg` accepts and that fit the register file.

    A value carried across the loop occupies its owner's registers for the whole
    period, and which lane owns it is already forced by locality, so the demand
    follows the operation that produces it.
    """
    choices = list(range(REGISTER_MIN, REGISTER_MAX + 1, REGISTER_STEP))
    extra: dict[int, list[Any]] = {}
    for lane, per_thread, literal in register_load:
        extra.setdefault(lane, []).append(LANE_THREADS * per_thread * literal)
    allocations = []
    for lane in range(problem.warp_groups):
        carried = sum(
            size * placed(operation_id, lane) for operation_id, size in registers.items()
        ) + sum(extra.get(lane, []))
        allocated = model.NewIntVarFromDomain(
            cp_model.Domain.FromValues(choices), f"nreg_{lane}"
        )
        # Held values plus room to compute with them, per thread.
        model.Add(
            LANE_THREADS * allocated
            >= carried + LANE_THREADS * TEMPORARY_REGISTERS_PER_THREAD
        )
        allocations.append(allocated)
    model.Add(sum(allocations) <= REGISTER_BUDGET)


__all__ = ["solve_free_compact"]
