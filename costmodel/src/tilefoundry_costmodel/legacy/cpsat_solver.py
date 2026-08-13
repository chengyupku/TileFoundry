"""Optional exact CP-SAT backend for finite resource-constrained pipelines."""

from __future__ import annotations

import math
from collections import defaultdict
from decimal import ROUND_CEILING, Decimal
from importlib import import_module
from typing import Any, Dict, List, Mapping, Optional, Tuple, cast

from .model import (
    PipelineHardware,
    PipelineProblem,
    PipelineSolution,
    Placement,
    SolveStatus,
    TimingOracle,
)


class CpSatPipelineSolver:
    """Minimize finite-pipeline makespan with OR-Tools CP-SAT.

    Floating-point nanoseconds are rounded up to integer ticks, so the discretized
    problem never shortens a duration or precedence delay. The default one-picosecond
    resolution keeps the public problem API in nanoseconds.
    """

    def __init__(
        self,
        *,
        time_limit_s: Optional[float] = 30.0,
        time_resolution_ps: int = 1,
        num_workers: int = 1,
        random_seed: int = 0,
    ):
        if time_limit_s is not None and time_limit_s <= 0:
            raise ValueError("time_limit_s must be positive or None")
        if time_resolution_ps <= 0:
            raise ValueError("time_resolution_ps must be positive")
        if num_workers <= 0:
            raise ValueError("num_workers must be positive")
        self.time_limit_s = time_limit_s
        self.time_resolution_ps = int(time_resolution_ps)
        self.num_workers = int(num_workers)
        self.random_seed = int(random_seed)

    @property
    def _tick_ns(self) -> float:
        return self.time_resolution_ps / 1000.0

    def _ticks(self, value_ns: float) -> int:
        ticks = Decimal(str(value_ns)) * Decimal(1000) / Decimal(self.time_resolution_ps)
        return int(ticks.to_integral_value(rounding=ROUND_CEILING))

    def _ns(self, ticks: int | float) -> float:
        return float(ticks) * self._tick_ns

    def solve(
        self,
        problem: PipelineProblem,
        oracle: TimingOracle,
        hardware: PipelineHardware,
    ) -> PipelineSolution:
        try:
            cp_model = cast(Any, import_module("ortools.sat.python.cp_model"))
        except ImportError:
            return PipelineSolution(
                status=SolveStatus.UNSUPPORTED,
                diagnostics=("OR-Tools is not installed; install tilefoundry-costmodel[cpsat]",),
            )

        stages = {stage.name: stage for stage in problem.stages}
        if len(stages) != len(problem.stages):
            return self._infeasible("stage names must be unique")
        if not stages:
            return PipelineSolution(
                status=SolveStatus.OPTIMAL,
                makespan_ns=0.0,
                lower_bound_ns=0.0,
                diagnostics=(f"solver=cp-sat resolution_ps={self.time_resolution_ps}",),
            )

        durations: Dict[str, float] = {}
        duration_ticks: Dict[str, int] = {}
        for stage in problem.stages:
            duration = float(oracle.estimate(stage, hardware=hardware).duration_ns)
            if not math.isfinite(duration) or duration < 0:
                return self._infeasible(f"stage {stage.name!r} has invalid duration {duration!r}")
            durations[stage.name] = duration
            duration_ticks[stage.name] = self._ticks(duration)
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

        edges = []
        seen_edges = set()
        successors: Dict[str, List[str]] = defaultdict(list)
        indegree = {name: 0 for name in stages}
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
            edges.append((edge.src, edge.dst, delay, self._ticks(delay)))
            successors[edge.src].append(edge.dst)
            indegree[edge.dst] += 1
        if not self._is_acyclic(successors, indegree):
            return self._infeasible("precedence graph contains a cycle")

        horizon = max(
            1, sum(duration_ticks.values()) + sum(delay_ticks for _, _, _, delay_ticks in edges)
        )
        model = cp_model.CpModel()
        starts = {}
        ends = {}
        intervals = {}
        for stage in problem.stages:
            start = model.new_int_var(0, horizon, f"{stage.name}_start")
            end = model.new_int_var(0, horizon, f"{stage.name}_end")
            interval = model.new_interval_var(
                start, duration_ticks[stage.name], end, f"{stage.name}_interval"
            )
            starts[stage.name] = start
            ends[stage.name] = end
            intervals[stage.name] = interval

        for src, dst, _, delay_ticks in edges:
            model.add(starts[dst] >= ends[src] + delay_ticks)

        by_resource = defaultdict(list)
        for stage in problem.stages:
            for resource_demand in stage.resource_demands:
                by_resource[resource_demand.resource].append(
                    (intervals[stage.name], resource_demand.demand)
                )
        for resource, reservations in by_resource.items():
            capacity = hardware.capacity(resource)
            if capacity is None:
                return self._infeasible(f"resource {resource!r} is not declared")
            resource_intervals = [interval for interval, _ in reservations]
            demands = [demand for _, demand in reservations]
            if capacity == 1 and all(demand == 1 for demand in demands):
                model.add_no_overlap(resource_intervals)
            else:
                model.add_cumulative(
                    resource_intervals,
                    demands,
                    int(capacity),
                )

        makespan = model.new_int_var(0, horizon, "makespan")
        model.add_max_equality(makespan, tuple(ends.values()))
        model.minimize(makespan)

        solver = cp_model.CpSolver()
        solver.parameters.num_search_workers = self.num_workers
        solver.parameters.random_seed = self.random_seed
        if self.time_limit_s is not None:
            solver.parameters.max_time_in_seconds = float(self.time_limit_s)
        status = solver.solve(model)
        if status == cp_model.INFEASIBLE:
            return self._infeasible("CP-SAT proved the pipeline infeasible")
        if status == cp_model.MODEL_INVALID:
            return self._infeasible("CP-SAT rejected the pipeline model")
        if status == cp_model.UNKNOWN:
            return PipelineSolution(
                status=SolveStatus.TIMEOUT,
                lower_bound_ns=self._ns(solver.best_objective_bound),
                diagnostics=("CP-SAT returned no feasible schedule",),
            )

        solved_starts = {name: solver.value(starts[name]) for name in stages}
        canonical_starts = dict(solved_starts)
        predecessor_edges: Dict[str, List[Tuple[str, int]]] = defaultdict(list)
        canonical_indegree = {name: 0 for name in stages}
        canonical_successors: Dict[str, List[str]] = defaultdict(list)
        for src, dst, _, delay_ticks in edges:
            predecessor_edges[dst].append((src, delay_ticks))
            canonical_successors[src].append(dst)
            canonical_indegree[dst] += 1
        ready = sorted(name for name, degree in canonical_indegree.items() if degree == 0)
        while ready:
            name = ready.pop(0)
            if duration_ticks[name] == 0:
                canonical_starts[name] = max(
                    (
                        canonical_starts[src] + duration_ticks[src] + delay
                        for src, delay in predecessor_edges.get(name, ())
                    ),
                    default=0,
                )
            for successor in sorted(canonical_successors.get(name, ())):
                canonical_indegree[successor] -= 1
                if canonical_indegree[successor] == 0:
                    ready.append(successor)
                    ready.sort()

        placements = tuple(
            sorted(
                (
                    Placement(
                        stage=stage.name,
                        group=stage.group,
                        iteration=stage.iteration,
                        start_ns=self._ns(canonical_starts[stage.name]),
                        end_ns=self._ns(canonical_starts[stage.name] + duration_ticks[stage.name]),
                        resources=stage.resources,
                        resource_demands=stage.resource_demands,
                    )
                    for stage in problem.stages
                ),
                key=lambda item: (item.start_ns, item.stage),
            )
        )
        placement_by_name = {placement.stage: placement for placement in placements}
        critical_path = self._critical_path(placement_by_name, edges, hardware)
        per_group: Dict[str, float] = defaultdict(float)
        for stage in problem.stages:
            per_group[stage.group] += durations[stage.name]
        result_status = SolveStatus.OPTIMAL if status == cp_model.OPTIMAL else SolveStatus.FEASIBLE
        return PipelineSolution(
            status=result_status,
            makespan_ns=self._ns(solver.value(makespan)),
            lower_bound_ns=self._ns(solver.best_objective_bound),
            placements=placements,
            per_group_ns=dict(per_group),
            critical_path=critical_path,
            diagnostics=(f"solver=cp-sat resolution_ps={self.time_resolution_ps}",),
        )

    @staticmethod
    def _is_acyclic(
        successors: Mapping[str, List[str]],
        indegree: Mapping[str, int],
    ) -> bool:
        pending = [name for name, degree in indegree.items() if degree == 0]
        degrees = dict(indegree)
        visited = 0
        while pending:
            name = pending.pop()
            visited += 1
            for successor in successors.get(name, ()):
                degrees[successor] -= 1
                if degrees[successor] == 0:
                    pending.append(successor)
        return visited == len(indegree)

    @staticmethod
    def _critical_path(
        placements: Mapping[str, Placement],
        edges: List[Tuple[str, str, float, int]],
        hardware: PipelineHardware,
    ) -> Tuple[str, ...]:
        if not placements:
            return ()
        predecessors: Dict[str, List[Tuple[float, str]]] = defaultdict(list)
        for src, dst, delay, _ in edges:
            predecessors[dst].append((placements[src].end_ns + delay, src))
        for stage, placement in placements.items():
            for resource in placement.resources:
                if hardware.capacity(resource) != 1:
                    continue
                for candidate, previous in placements.items():
                    if candidate == stage or resource not in previous.resources:
                        continue
                    if previous.end_ns <= placement.start_ns + 1e-9:
                        predecessors[stage].append((previous.end_ns, candidate))

        current = max(placements, key=lambda name: placements[name].end_ns)
        path = []
        seen = set()
        while current not in seen:
            path.append(current)
            seen.add(current)
            start = placements[current].start_ns
            candidates = [
                (ready, name)
                for ready, name in predecessors.get(current, ())
                if abs(ready - start) <= 1e-6
            ]
            if not candidates:
                break
            current = max(candidates)[1]
        path.reverse()
        return tuple(path)

    @staticmethod
    def _infeasible(message: str) -> PipelineSolution:
        return PipelineSolution(
            status=SolveStatus.INFEASIBLE,
            diagnostics=(message,),
        )
