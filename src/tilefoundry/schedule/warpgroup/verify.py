"""Independent verification of a finite warpgroup schedule."""

from __future__ import annotations

from collections import defaultdict, deque

from .errors import WarpgroupVerificationError
from .expression import value_references
from .model import (
    MemorySpace,
    SynchronizationEdge,
    TimedOperation,
    WarpgroupProblem,
    WarpgroupSchedule,
)

type _Instance = tuple[int, str]
type _ExpandedEdge = tuple[_Instance, _Instance]


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


def _required_shared_relations(
    problem: WarpgroupProblem,
) -> tuple[SynchronizationEdge, ...]:
    owner, spaces, users = _semantic_tables(problem)
    relations: set[SynchronizationEdge] = set()
    iter_args = {item.id for item in problem.loop.iter_args}
    for operation in problem.loop.ops:
        for output in operation.outputs:
            for value_id in value_references(output.expression):
                if value_id not in owner or spaces[value_id] is not MemorySpace.SHARED:
                    continue
                distance = 1 if value_id in iter_args else 0
                if owner[value_id] != operation.id or distance:
                    relations.add(SynchronizationEdge(owner[value_id], operation.id, distance))

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


def _reachable(edges: set[_ExpandedEdge], source: _Instance, target: _Instance) -> bool:
    successors: dict[_Instance, set[_Instance]] = defaultdict(set)
    for after, before in edges:
        successors[after].add(before)
    pending = [source]
    visited = {source}
    while pending:
        current = pending.pop()
        if current == target:
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
            amount = next(
                (
                    demand.amount
                    for demand in operation.resources
                    if demand.resource_id == resource.id
                ),
                0,
            )
            changes[timed.start] += amount
            changes[timed.end] -= amount
        demand = 0
        for timestamp in sorted(changes):
            demand += changes[timestamp]
            if demand > resource.capacity:
                _fail(f"resource {resource.id!r} exceeds capacity at time {timestamp}")


def verify_warpgroup_schedule(problem: WarpgroupProblem, schedule: WarpgroupSchedule) -> None:
    """Verify one schedule against its problem without solving it again."""
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
    if set(lane_by_operation) != expected_operations:
        missing = sorted(expected_operations - set(lane_by_operation))
        unknown = sorted(set(lane_by_operation) - expected_operations)
        _fail(f"schedule lane coverage differs: missing={missing!r}, unknown={unknown!r}")

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
        duration = operation_by_id[instance[1]].duration
        if timed.end - timed.start != duration:
            _fail(f"timed operation {instance!r} has the wrong duration")

    lane_edges = _lane_edges(schedule, problem.loop.iterations)
    for after, before in lane_edges:
        if times[after].end > times[before].start:
            _fail(f"lane order {after!r} -> {before!r} is not respected")

    for edge in schedule.sync:
        if edge.after not in expected_operations or edge.before not in expected_operations:
            _fail(f"sync edge names an unknown operation: {edge!r}")
        expanded = _expand_relation(edge, problem.loop.iterations)
        if not expanded:
            _fail(f"sync edge has no finite instance: {edge!r}")
        for after, before in expanded:
            if times[after].end > times[before].start:
                _fail(f"sync inequality {after!r} -> {before!r} is not respected")

    semantic_edges: set[_ExpandedEdge] = set()
    for dependency in problem.dependencies():
        for after, before in _expand_relation(
            SynchronizationEdge(dependency.after, dependency.before, dependency.distance),
            problem.loop.iterations,
        ):
            semantic_edges.add((after, before))
            if times[after].end > times[before].start:
                _fail(f"SSA dependency {after!r} -> {before!r} is not respected")

    owner, spaces, users = _semantic_tables(problem)
    for value_id, space in spaces.items():
        if space is not MemorySpace.REGISTER:
            continue
        component = set(users.get(value_id, ()))
        if value_id in owner:
            component.add(owner[value_id])
        if len({lane_by_operation[item] for item in component}) > 1:
            _fail(f"register value {value_id!r} crosses warpgroup lanes")

    control_edges = _control_edges(schedule, problem.loop.iterations)
    for relation in _required_shared_relations(problem):
        for after, before in _expand_relation(relation, problem.loop.iterations):
            semantic_edges.add((after, before))
            if times[after].end > times[before].start:
                _fail(f"shared relation {after!r} -> {before!r} is not respected")
            if not _reachable(control_edges, after, before):
                _fail(f"shared relation {after!r} -> {before!r} has no lane/sync path")
    for after, before in _carried_shared_lifetime_edges(problem):
        semantic_edges.add((after, before))
        if times[after].end > times[before].start:
            _fail(f"carried shared lifetime {after!r} -> {before!r} is not respected")

    _verify_resources(problem, times)
    nodes = set(expected_instances)
    _require_acyclic(nodes, control_edges | semantic_edges)


__all__ = ["verify_warpgroup_schedule"]
