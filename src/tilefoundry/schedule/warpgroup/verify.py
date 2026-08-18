"""Independent verification of a finite warpgroup schedule."""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Literal

from .build import build_warpgroup_problem
from .cost import CostLibrary
from .errors import WarpgroupVerificationError
from .expression import value_references
from .model import (
    MemorySpace,
    SynchronizationEdge,
    TimedOperation,
    WarpgroupProblem,
    WarpgroupProgram,
    WarpgroupSchedule,
)

type _Instance = tuple[int, str]
type _ExpandedEdge = tuple[_Instance, _Instance]
type _EventPhase = Literal["start", "issue_end", "completion"]
type _Event = tuple[int, str, _EventPhase]
type _EventEdge = tuple[_Event, _Event]


def _fail(message: str) -> None:
    raise WarpgroupVerificationError(message)


def _semantic_tables(
    problem: WarpgroupProblem,
) -> tuple[dict[str, str], dict[str, MemorySpace], dict[str, tuple[str, ...]]]:
    types = {item.id: item for item in problem.types}
    owner: dict[str, str] = {}
    spaces = {item.id: types[item.type_id].space for item in problem.inputs}
    users: dict[str, set[str]] = defaultdict(set)
    for operation in problem.loop.ops:
        for output in operation.outputs:
            owner[output.id] = operation.id
            spaces[output.id] = types[output.type_id].space
            for value_id in value_references(output.expression):
                users[value_id].add(operation.id)
    for iter_arg in problem.loop.iter_args:
        owner[iter_arg.id] = owner[iter_arg.yield_value.id]
        spaces[iter_arg.id] = spaces[iter_arg.yield_value.id]
    return (
        owner,
        spaces,
        {value_id: tuple(sorted(operation_ids)) for value_id, operation_ids in users.items()},
    )


def _required_completion_relations(
    problem: WarpgroupProblem,
) -> tuple[SynchronizationEdge, ...]:
    owner, spaces, users = _semantic_tables(problem)
    relations: set[SynchronizationEdge] = set()
    relations.update(
        SynchronizationEdge(item.after, item.before, item.distance)
        for item in problem.dependencies()
    )
    for operation in problem.loop.ops:
        for output in operation.outputs:
            if spaces[output.id] is not MemorySpace.SHARED:
                continue
            for user in users.get(output.id, ()):
                if user != operation.id:
                    relations.add(SynchronizationEdge(user, operation.id, 1))
    return tuple(sorted(relations))


def _carried_shared_lifetime_edges(problem: WarpgroupProblem) -> set[_ExpandedEdge]:
    owner, spaces, users = _semantic_tables(problem)
    edges: set[_ExpandedEdge] = set()
    for iter_arg in problem.loop.iter_args:
        if spaces[iter_arg.id] is not MemorySpace.SHARED:
            continue
        defining_operation = owner[iter_arg.id]
        for user in users.get(iter_arg.id, ()):
            if user == defining_operation:
                continue
            for iteration in range(1, problem.loop.iterations):
                edges.add(((iteration, user), (iteration, defining_operation)))
    return edges


def _expand_relation(edge: SynchronizationEdge, iterations: int) -> tuple[_ExpandedEdge, ...]:
    return tuple(
        (
            (iteration, edge.after),
            (iteration + edge.distance, edge.before),
        )
        for iteration in range(max(iterations - edge.distance, 0))
    )


def _lane_edges(schedule: WarpgroupSchedule, iterations: int) -> set[_ExpandedEdge]:
    edges: set[_ExpandedEdge] = set()
    for lane in schedule.lanes:
        for iteration in range(iterations):
            for after, before in zip(lane.operations, lane.operations[1:]):
                edges.add(((iteration, after), (iteration, before)))
            if lane.operations and iteration + 1 < iterations:
                edges.add(
                    (
                        (iteration, lane.operations[-1]),
                        (iteration + 1, lane.operations[0]),
                    )
                )
    return edges


def _control_edges(schedule: WarpgroupSchedule, iterations: int) -> set[_ExpandedEdge]:
    edges = _lane_edges(schedule, iterations)
    for edge in schedule.sync:
        edges.update(_expand_relation(edge, iterations))
    return edges


def _completion_event_edges(
    problem: WarpgroupProblem,
    schedule: WarpgroupSchedule,
) -> set[_EventEdge]:
    edges: set[_EventEdge] = set()
    for iteration in range(problem.loop.iterations):
        for operation in problem.loop.ops:
            start: _Event = (iteration, operation.id, "start")
            issue_end: _Event = (iteration, operation.id, "issue_end")
            completion: _Event = (iteration, operation.id, "completion")
            edges.add((start, issue_end))
            edges.add((issue_end, completion))
            if operation.issue_duration == operation.completion_latency:
                edges.add((completion, issue_end))
    for after, before in _lane_edges(schedule, problem.loop.iterations):
        edges.add(((*after, "issue_end"), (*before, "start")))
    for edge in schedule.sync:
        for after, before in _expand_relation(edge, problem.loop.iterations):
            edges.add(((*after, "completion"), (*before, "start")))
    return edges


def _event_reachable(edges: set[_EventEdge], source: _Instance, target: _Instance) -> bool:
    successors: dict[_Event, set[_Event]] = defaultdict(set)
    for after, before in edges:
        successors[after].add(before)
    source_event: _Event = (*source, "completion")
    target_event: _Event = (*target, "start")
    pending = [source_event]
    visited = {source_event}
    while pending:
        current = pending.pop()
        if current == target_event:
            return True
        for successor in successors.get(current, ()):
            if successor not in visited:
                visited.add(successor)
                pending.append(successor)
    return False


def _require_acyclic(nodes: set[_Instance], edges: set[_ExpandedEdge]) -> None:
    successors: dict[_Instance, set[_Instance]] = defaultdict(set)
    indegree = {node: 0 for node in nodes}
    for after, before in edges:
        if before not in successors[after]:
            successors[after].add(before)
            indegree[before] += 1
    ready = deque(sorted(node for node, degree in indegree.items() if degree == 0))
    visited = 0
    while ready:
        current = ready.popleft()
        visited += 1
        for successor in sorted(successors.get(current, ())):
            indegree[successor] -= 1
            if indegree[successor] == 0:
                ready.append(successor)
    if visited != len(nodes):
        _fail("expanded lane, sync, and semantic graph contains a cycle")


def _verify_resources(problem: WarpgroupProblem, times: dict[_Instance, TimedOperation]) -> None:
    operations = {item.id: item for item in problem.loop.ops}
    for resource in problem.resources:
        changes: dict[int, int] = defaultdict(int)
        for instance, timed in times.items():
            operation = operations[instance[1]]
            for window in operation.resource_windows:
                if window.resource_id != resource.id:
                    continue
                start = timed.start
                end = start + window.duration
                changes[start] += window.amount
                changes[end] -= window.amount
        demand = 0
        for timestamp in sorted(changes):
            demand += changes[timestamp]
            if demand > resource.capacity:
                _fail(f"resource {resource.id!r} exceeds capacity at time {timestamp}")


def _region_ids(problem: WarpgroupProblem) -> set[str]:
    return {operation.id for operation in (*problem.prologue, *problem.epilogue)}


def _verify_regions(problem: WarpgroupProblem, schedule: WarpgroupSchedule) -> None:
    """Every region operation on its declared lane, in its declared order.

    There is nothing to search here and therefore nothing to check a search
    against: what a schedule may get wrong about a region is dropping one,
    inventing one, or reordering a run whose order is its whole meaning.
    """
    for region in ("prologue", "epilogue"):
        declared: dict[int, list[str]] = {lane: [] for lane in range(problem.warp_groups)}
        for operation in getattr(problem, region):
            declared[operation.warp_group].append(operation.id)
        for lane_index, lane in enumerate(schedule.lanes):
            found = list(getattr(lane, region))
            if found != declared[lane_index]:
                _fail(
                    f"schedule lane {lane_index} {region} is {found!r}, "
                    f"expected {declared[lane_index]!r}"
                )


def _region_semantic_tables(
    problem: WarpgroupProblem,
) -> tuple[dict[str, str], dict[str, MemorySpace], dict[str, tuple[str, ...]]]:
    """The body's tables widened by the regions, for register locality only.

    Register locality is the one rule that has to span the loop boundary: an
    epilogue that reduces a carried accumulator reads a register file belonging
    to one warp group, and reading it from another is not slow but impossible.
    The completion relations stay body-only -- a region has no periodic instance
    for an edge to name.
    """
    types = {item.id: item for item in problem.types}
    owner, spaces, users = _semantic_tables(problem)
    widened = {value_id: set(operation_ids) for value_id, operation_ids in users.items()}
    for operation in (*problem.prologue, *problem.epilogue):
        for output in operation.outputs:
            owner[output.id] = operation.id
            spaces[output.id] = types[output.type_id].space
            for value_id in value_references(output.expression):
                widened.setdefault(value_id, set()).add(operation.id)
    return (
        owner,
        spaces,
        {value_id: tuple(sorted(operation_ids)) for value_id, operation_ids in widened.items()},
    )


def _verify_warpgroup_schedule(problem: WarpgroupProblem, schedule: WarpgroupSchedule) -> None:
    """Verify one schedule against an internal closed problem."""
    if type(problem) is not WarpgroupProblem:
        _fail("verification requires an exact WarpgroupProblem")
    if type(schedule) is not WarpgroupSchedule:
        _fail("verification requires an exact WarpgroupSchedule")
    operation_by_id = {item.id: item for item in problem.loop.ops}
    expected_operations = set(operation_by_id)
    if len(schedule.lanes) != problem.warp_groups:
        _fail("schedule lane count does not match warp_groups")
    lane_operations = tuple(
        operation_id for lane in schedule.lanes for operation_id in lane.operations
    )
    if len(lane_operations) != len(set(lane_operations)):
        _fail("schedule repeats an operation across lanes")
    lane_by_operation = {
        operation_id: lane_index
        for lane_index, lane in enumerate(schedule.lanes)
        for operation_id in lane.operations
    }
    _verify_regions(problem, schedule)
    for lane_index, lane in enumerate(schedule.lanes):
        for operation_id in lane.prologue + lane.epilogue:
            lane_by_operation[operation_id] = lane_index
    if set(lane_by_operation) - _region_ids(problem) != expected_operations:
        missing = sorted(expected_operations - set(lane_by_operation))
        unknown = sorted(set(lane_by_operation) - expected_operations - _region_ids(problem))
        _fail(f"schedule lane coverage differs: missing={missing!r}, unknown={unknown!r}")
    for operation in problem.loop.ops:
        if operation.warp_group is None:
            # Nothing was declared, so there is nothing to contradict.
            continue
        if lane_by_operation[operation.id] != operation.warp_group:
            _fail(
                f"operation {operation.id!r} is scheduled on lane "
                f"{lane_by_operation[operation.id]}, expected warp_group "
                f"{operation.warp_group}"
            )

    expected_instances = {
        (iteration, operation_id)
        for iteration in range(problem.loop.iterations)
        for operation_id in expected_operations
    }
    times = {(item.iteration, item.operation_id): item for item in schedule.times}
    if len(times) != len(schedule.times):
        _fail("schedule contains duplicate timed instances")
    if set(times) != expected_instances:
        missing_instances = sorted(expected_instances - set(times))
        extra_instances = sorted(set(times) - expected_instances)
        _fail(
            "schedule time coverage differs: "
            f"missing={missing_instances!r}, extra={extra_instances!r}"
        )
    for instance, timed in times.items():
        operation = operation_by_id[instance[1]]
        issue_end = timed.issue_end
        if issue_end - timed.start != operation.issue_duration:
            _fail(f"timed operation {instance!r} has the wrong duration: issue duration")
        if timed.completion - timed.start != operation.completion_latency:
            _fail(f"timed operation {instance!r} has the wrong duration: completion latency")
    if problem.loop.iterations >= 3:
        first_operation = min(expected_operations)
        initiation_interval = times[(2, first_operation)].start - times[(1, first_operation)].start
        if initiation_interval <= 0:
            _fail("periodic initiation interval must be positive")
        for operation_id in sorted(expected_operations):
            for iteration in range(1, problem.loop.iterations - 1):
                actual = (
                    times[(iteration + 1, operation_id)].start
                    - times[(iteration, operation_id)].start
                )
                if actual != initiation_interval:
                    _fail(
                        f"periodic initiation interval differs for "
                        f"{operation_id!r} at iteration {iteration}: "
                        f"expected {initiation_interval}, got {actual}"
                    )

    lane_edges = _lane_edges(schedule, problem.loop.iterations)
    for after, before in lane_edges:
        issue_end = times[after].issue_end
        if issue_end > times[before].start:
            _fail(f"lane order {after!r} -> {before!r} is not respected")

    for edge in schedule.sync:
        if edge.after not in expected_operations or edge.before not in expected_operations:
            _fail(f"sync edge names an unknown operation: {edge!r}")
        expanded = _expand_relation(edge, problem.loop.iterations)
        if not expanded:
            _fail(f"sync edge has no finite instance: {edge!r}")
        for after, before in expanded:
            if times[after].completion > times[before].start:
                _fail(f"sync inequality {after!r} -> {before!r} is not respected")

    semantic_edges: set[_ExpandedEdge] = set()

    owner, spaces, users = _region_semantic_tables(problem)
    for value_id, space in spaces.items():
        if space is not MemorySpace.REGISTER:
            continue
        component = set(users.get(value_id, ()))
        if value_id in owner:
            component.add(owner[value_id])
        if len({lane_by_operation[item] for item in component}) > 1:
            _fail(f"register value {value_id!r} crosses warpgroup lanes")

    control_edges = _control_edges(schedule, problem.loop.iterations)
    completion_edges = _completion_event_edges(problem, schedule)
    for relation in _required_completion_relations(problem):
        for after, before in _expand_relation(relation, problem.loop.iterations):
            semantic_edges.add((after, before))
            if times[after].completion > times[before].start:
                _fail(f"completion relation {after!r} -> {before!r} is not respected")
            if not _event_reachable(completion_edges, after, before):
                _fail(f"completion relation {after!r} -> {before!r} has no lane/sync path")
    for after, before in _carried_shared_lifetime_edges(problem):
        semantic_edges.add((after, before))
        if times[after].completion > times[before].start:
            _fail(f"carried shared lifetime {after!r} -> {before!r} is not respected")

    _verify_resources(problem, times)
    nodes = set(expected_instances)
    _require_acyclic(nodes, control_edges | semantic_edges)


def verify_warpgroup_schedule(
    program: WarpgroupProgram,
    hardware: CostLibrary,
    schedule: WarpgroupSchedule,
) -> None:
    """Verify one schedule against its public program and hardware inputs."""
    _verify_warpgroup_schedule(build_warpgroup_problem(program, hardware), schedule)


__all__ = ["verify_warpgroup_schedule"]
