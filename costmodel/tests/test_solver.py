from __future__ import annotations

import pytest

from tilefoundry_costmodel.legacy import (
    CpSatPipelineSolver,
    ListPipelineSolver,
    PipelineHardware,
    PipelineProblem,
    Precedence,
    ResourceDemand,
    ResourceSpec,
    SolveStatus,
    StageSpec,
    StageTiming,
)


class Oracle:
    def estimate(self, stage, *, hardware):
        del hardware
        return StageTiming(stage.payload)


def _solve(solver, stages, *, precedences=(), resources=()):
    return solver.solve(
        PipelineProblem(tuple(stages), tuple(precedences)),
        Oracle(),
        PipelineHardware(tuple(resources)),
    )


def test_pipeline_hardware_rejects_invalid_resource_capacities():
    with pytest.raises(ValueError, match="positive"):
        ResourceSpec("pipe", capacity=0)
    with pytest.raises(ValueError, match="duplicate"):
        PipelineHardware((ResourceSpec("pipe"), ResourceSpec("pipe")))
    with pytest.raises(ValueError, match="duplicate"):
        StageSpec(
            "duplicate-demand",
            "work",
            resource_demands=(ResourceDemand("pipe"), ResourceDemand("pipe")),
        )


def test_list_solver_respects_precedence_and_resource_capacity():
    result = _solve(
        ListPipelineSolver(),
        (
            StageSpec("load_a", "load", resources=("copy",), payload=3.0),
            StageSpec("load_b", "load", resources=("copy",), payload=2.0),
            StageSpec("compute", "compute", resources=("compute",), payload=4.0),
        ),
        precedences=(Precedence("load_a", "compute"),),
        resources=(ResourceSpec("copy", capacity=1), ResourceSpec("compute", capacity=1)),
    )
    placements = {placement.stage: placement for placement in result.placements}

    assert result.status is SolveStatus.FEASIBLE
    assert result.makespan_ns == 7.0
    assert result.per_group_ns == {"load": 5.0, "compute": 4.0}
    assert placements["load_a"].end_ns <= placements["compute"].start_ns
    assert all(
        placement.start_ns >= 0.0 and placement.end_ns >= placement.start_ns
        for placement in placements.values()
    )
    assert not (
        placements["load_a"].start_ns < placements["load_b"].end_ns
        and placements["load_b"].start_ns < placements["load_a"].end_ns
    )


def test_resource_capacity_allows_parallel_instances():
    result = _solve(
        ListPipelineSolver(),
        (
            StageSpec("a", "work", resources=("pipe",), payload=3.0),
            StageSpec("b", "work", resources=("pipe",), payload=2.0),
        ),
        resources=(ResourceSpec("pipe", capacity=2),),
    )
    placements = {placement.stage: placement for placement in result.placements}
    assert result.status is SolveStatus.FEASIBLE
    assert result.makespan_ns == 3.0
    assert placements["a"].start_ns == placements["b"].start_ns == 0.0
    assert placements["a"].end_ns == 3.0
    assert placements["b"].end_ns == 2.0


@pytest.mark.parametrize("solver", [ListPipelineSolver(), CpSatPipelineSolver()])
def test_resource_demand_consumes_multiple_capacity_slots(solver):
    if isinstance(solver, CpSatPipelineSolver):
        pytest.importorskip("ortools")
    result = _solve(
        solver,
        (
            StageSpec(
                "wide",
                "work",
                payload=5.0,
                resource_demands=(ResourceDemand("pipe", 2),),
            ),
            StageSpec(
                "narrow",
                "work",
                payload=5.0,
                resource_demands=(ResourceDemand("pipe", 1),),
            ),
        ),
        resources=(ResourceSpec("pipe", capacity=2),),
    )
    placements = {placement.stage: placement for placement in result.placements}

    assert result.status in (SolveStatus.FEASIBLE, SolveStatus.OPTIMAL)
    assert result.makespan_ns == 10.0
    assert placements["wide"].resource_demands == (ResourceDemand("pipe", 2),)
    assert {placements["wide"].start_ns, placements["narrow"].start_ns} == {0.0, 5.0}
    assert all(placement.end_ns == placement.start_ns + 5.0 for placement in placements.values())


@pytest.mark.parametrize("solver", [ListPipelineSolver(), CpSatPipelineSolver()])
def test_resource_demand_larger_than_capacity_is_infeasible(solver):
    if isinstance(solver, CpSatPipelineSolver):
        pytest.importorskip("ortools")
    result = _solve(
        solver,
        (
            StageSpec(
                "wide",
                "work",
                payload=5.0,
                resource_demands=(ResourceDemand("pipe", 3),),
            ),
        ),
        resources=(ResourceSpec("pipe", capacity=2),),
    )

    assert result.status is SolveStatus.INFEASIBLE
    assert "demand" in result.diagnostics[0]


def test_cycle_is_reported_as_infeasible():
    result = _solve(
        ListPipelineSolver(),
        (StageSpec("a", "a", payload=1.0), StageSpec("b", "b", payload=1.0)),
        precedences=(Precedence("a", "b"), Precedence("b", "a")),
    )

    assert result.status is SolveStatus.INFEASIBLE
    assert "cycle" in result.diagnostics[0]


def test_cpsat_solver_proves_optimal_schedule_with_delay_and_capacity():
    pytest.importorskip("ortools")
    result = _solve(
        CpSatPipelineSolver(),
        (
            StageSpec("load_a", "load", resources=("copy",), payload=3.0),
            StageSpec("load_b", "load", resources=("copy",), payload=2.0),
            StageSpec("compute", "compute", resources=("compute",), payload=4.0),
        ),
        precedences=(Precedence("load_a", "compute", delay_ns=1.0),),
        resources=(ResourceSpec("copy", capacity=2), ResourceSpec("compute", capacity=1)),
    )
    placements = {placement.stage: placement for placement in result.placements}

    assert result.status is SolveStatus.OPTIMAL
    assert result.makespan_ns == 8.0
    assert result.lower_bound_ns == 8.0
    assert placements["load_a"].end_ns + 1.0 <= placements["compute"].start_ns
    assert placements["load_a"].start_ns == placements["load_b"].start_ns == 0.0
    assert result.per_group_ns == {"load": 5.0, "compute": 4.0}
    assert placements["load_a"].resource_demands == (ResourceDemand("copy"),)
    assert placements["compute"].resource_demands == (ResourceDemand("compute"),)
    assert all(
        placement.start_ns >= 0.0 and placement.end_ns >= placement.start_ns
        for placement in placements.values()
    )


def test_cpsat_solver_preserves_picosecond_resolution():
    pytest.importorskip("ortools")
    result = _solve(
        CpSatPipelineSolver(time_resolution_ps=1),
        (
            StageSpec("a", "a", payload=0.123),
            StageSpec("b", "b", payload=0.456),
        ),
        precedences=(Precedence("a", "b"),),
    )

    assert result.status is SolveStatus.OPTIMAL
    assert result.makespan_ns == 0.579


def test_cpsat_solver_rounds_durations_and_delays_upward():
    pytest.importorskip("ortools")
    result = _solve(
        CpSatPipelineSolver(time_resolution_ps=100),
        (
            StageSpec("a", "a", payload=0.149),
            StageSpec("b", "b", payload=0.149),
        ),
        precedences=(Precedence("a", "b", delay_ns=0.149),),
    )
    placements = {placement.stage: placement for placement in result.placements}

    assert result.status is SolveStatus.OPTIMAL
    assert result.makespan_ns == pytest.approx(0.6)
    assert placements["a"].end_ns >= 0.149
    assert placements["b"].start_ns - placements["a"].end_ns >= 0.149


def test_cpsat_solver_accepts_empty_pipeline():
    pytest.importorskip("ortools")
    result = _solve(CpSatPipelineSolver(), ())

    assert result.status is SolveStatus.OPTIMAL
    assert result.makespan_ns == result.lower_bound_ns == 0.0
