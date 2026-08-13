"""Deterministic resource-constrained list scheduler."""

from __future__ import annotations

import heapq
import math
from collections import defaultdict
from collections.abc import Mapping
from typing import Dict, List, Optional, Tuple

from .model import (
    PipelineHardware,
    PipelineProblem,
    PipelineSolution,
    Placement,
    SolveStatus,
    TimingOracle,
)


class ListPipelineSolver:
    """Produce a feasible finite schedule with critical-path priority.

    This solver intentionally reports ``FEASIBLE``: list scheduling does not
    prove optimality. The problem/oracle/hardware contract is independent of the
    algorithm, so a CP-SAT backend can replace it without changing callers.
    """

    def solve(
        self,
        problem: PipelineProblem,
        oracle: TimingOracle,
        hardware: PipelineHardware,
    ) -> PipelineSolution:
        stages = {stage.name: stage for stage in problem.stages}
        if len(stages) != len(problem.stages):
            return self._infeasible("stage names must be unique")

        durations: Dict[str, float] = {}
        for stage in problem.stages:
            timing = oracle.estimate(stage, hardware=hardware)
            duration = float(timing.duration_ns)
            if not math.isfinite(duration) or duration < 0:
                return self._infeasible(f"stage {stage.name!r} has invalid duration {duration!r}")
            durations[stage.name] = duration
            for resource_demand in stage.resource_demands:
                resource = resource_demand.resource
                capacity = hardware.capacity(resource)
                if capacity is None:
                    return self._infeasible(
                        f"stage {stage.name!r} references unknown resource {resource!r}"
                    )
                if capacity <= 0:
                    return self._infeasible(
                        f"resource {resource!r} has non-positive capacity {capacity}"
                    )
                if resource_demand.demand > capacity:
                    return self._infeasible(
                        f"stage {stage.name!r} demand {resource_demand.demand} "
                        f"exceeds resource {resource!r} capacity {capacity}"
                    )

        successors: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
        predecessors: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
        indegree = {name: 0 for name in stages}
        seen_edges = set()
        for edge in problem.precedences:
            if edge.src not in stages or edge.dst not in stages:
                return self._infeasible(
                    f"precedence {edge.src!r}->{edge.dst!r} references a missing stage"
                )
            delay = float(edge.delay_ns)
            if not math.isfinite(delay) or delay < 0:
                return self._infeasible(
                    f"precedence {edge.src!r}->{edge.dst!r} has invalid delay {delay!r}"
                )
            key = (edge.src, edge.dst, delay)
            if key in seen_edges:
                continue
            seen_edges.add(key)
            successors[edge.src].append((edge.dst, delay))
            predecessors[edge.dst].append((edge.src, delay))
            indegree[edge.dst] += 1

        topo = self._topological_order(stages, successors, indegree)
        if topo is None:
            return self._infeasible("precedence graph contains a cycle")

        bottom_level = self._bottom_levels(topo, successors, durations)
        remaining = dict(indegree)
        ready: List[Tuple[float, str]] = []
        for name, degree in remaining.items():
            if degree == 0:
                heapq.heappush(ready, (-bottom_level[name], name))

        lanes: Dict[str, List[Tuple[float, Optional[str]]]] = {}
        for resource_spec in hardware.resources:
            lanes[resource_spec.name] = [(0.0, None) for _ in range(resource_spec.capacity)]

        start_ns: Dict[str, float] = {}
        end_ns: Dict[str, float] = {}
        critical_predecessor: Dict[str, Optional[str]] = {}
        placements: List[Placement] = []

        while ready:
            _, name = heapq.heappop(ready)
            stage = stages[name]
            dependency_ready = 0.0
            dependency_pred: Optional[str] = None
            for pred, delay in predecessors.get(name, ()):
                candidate = end_ns[pred] + delay
                if candidate >= dependency_ready:
                    dependency_ready = candidate
                    dependency_pred = pred

            selected_lanes: List[Tuple[str, int, float, Optional[str]]] = []
            resource_ready = 0.0
            resource_pred: Optional[str] = None
            for resource_demand in stage.resource_demands:
                resource = resource_demand.resource
                candidates = sorted(
                    enumerate(lanes[resource]),
                    key=lambda item: (item[1][0], item[0]),
                )[: resource_demand.demand]
                selected_lanes.extend(
                    (resource, lane_index, available, previous)
                    for lane_index, (available, previous) in candidates
                )
                available, previous = max(
                    (lane[1] for lane in candidates),
                    key=lambda item: item[0],
                )
                if available >= resource_ready:
                    resource_ready = available
                    resource_pred = previous

            start = max(dependency_ready, resource_ready)
            end = start + durations[name]
            start_ns[name] = start
            end_ns[name] = end
            critical_predecessor[name] = (
                resource_pred if resource_ready > dependency_ready else dependency_pred
            )
            for resource, lane_index, _, _ in selected_lanes:
                lanes[resource][lane_index] = (end, name)
            placements.append(
                Placement(
                    stage=name,
                    group=stage.group,
                    iteration=stage.iteration,
                    start_ns=start,
                    end_ns=end,
                    resources=stage.resources,
                    resource_demands=stage.resource_demands,
                )
            )

            for successor, _ in successors.get(name, ()):
                remaining[successor] -= 1
                if remaining[successor] == 0:
                    heapq.heappush(ready, (-bottom_level[successor], successor))

        makespan = max(end_ns.values(), default=0.0)
        end_stage = max(end_ns, key=lambda name: end_ns[name]) if end_ns else None
        critical_path = self._critical_path(end_stage, critical_predecessor)
        per_group: Dict[str, float] = defaultdict(float)
        for name, stage in stages.items():
            per_group[stage.group] += durations[name]

        precedence_bound = max(bottom_level.values(), default=0.0)
        resource_bound = 0.0
        for resource_spec in hardware.resources:
            busy = sum(
                durations[stage.name] * resource_demand.demand
                for stage in problem.stages
                for resource_demand in stage.resource_demands
                if resource_demand.resource == resource_spec.name
            )
            resource_bound = max(resource_bound, busy / resource_spec.capacity)

        return PipelineSolution(
            status=SolveStatus.FEASIBLE,
            makespan_ns=makespan,
            lower_bound_ns=max(precedence_bound, resource_bound),
            placements=tuple(sorted(placements, key=lambda p: (p.start_ns, p.stage))),
            per_group_ns=dict(per_group),
            critical_path=critical_path,
        )

    @staticmethod
    def _topological_order(
        stages: Mapping[str, object],
        successors: Mapping[str, List[Tuple[str, float]]],
        indegree: Mapping[str, int],
    ) -> Optional[List[str]]:
        pending = [name for name, degree in indegree.items() if degree == 0]
        heapq.heapify(pending)
        degrees = dict(indegree)
        out: List[str] = []
        while pending:
            name = heapq.heappop(pending)
            out.append(name)
            for successor, _ in successors.get(name, ()):
                degrees[successor] -= 1
                if degrees[successor] == 0:
                    heapq.heappush(pending, successor)
        return out if len(out) == len(stages) else None

    @staticmethod
    def _bottom_levels(
        topo: List[str],
        successors: Mapping[str, List[Tuple[str, float]]],
        durations: Mapping[str, float],
    ) -> Dict[str, float]:
        levels: Dict[str, float] = {}
        for name in reversed(topo):
            tail = max(
                (delay + levels[successor] for successor, delay in successors.get(name, ())),
                default=0.0,
            )
            levels[name] = durations[name] + tail
        return levels

    @staticmethod
    def _critical_path(
        end_stage: Optional[str],
        predecessor: Dict[str, Optional[str]],
    ) -> Tuple[str, ...]:
        path: List[str] = []
        seen = set()
        current = end_stage
        while current is not None and current not in seen:
            path.append(current)
            seen.add(current)
            current = predecessor.get(current)
        path.reverse()
        return tuple(path)

    @staticmethod
    def _infeasible(message: str) -> PipelineSolution:
        return PipelineSolution(
            status=SolveStatus.INFEASIBLE,
            diagnostics=(message,),
        )
