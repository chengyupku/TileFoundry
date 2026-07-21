import pytest

from tilefoundry_costmodel import (
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


class _Oracle:
    def estimate(self, stage, *, hardware):
        del hardware
        return StageTiming(stage.payload)


def test_pipeline_hardware_rejects_invalid_resource_capacities():
    with pytest.raises(ValueError, match="positive"):
        ResourceSpec("pipe", capacity=0)
    with pytest.raises(ValueError, match="duplicate"):
        PipelineHardware((ResourceSpec("pipe"), ResourceSpec("pipe")))


def test_list_solver_respects_precedence_and_resource_capacity():
    problem = PipelineProblem(
        stages=(
            StageSpec("load_a", "load", resources=("copy",), payload=3.0),
            StageSpec("load_b", "load", resources=("copy",), payload=2.0),
            StageSpec("compute", "compute", resources=("compute",), payload=4.0),
        ),
        precedences=(Precedence("load_a", "compute"),),
    )
    hardware = PipelineHardware(resources=(
        ResourceSpec("copy", capacity=1),
        ResourceSpec("compute", capacity=1),
    ))

    result = ListPipelineSolver().solve(problem, _Oracle(), hardware)

    assert result.status is SolveStatus.FEASIBLE
    assert result.makespan_ns == 7.0
    assert result.per_group_ns == {"load": 5.0, "compute": 4.0}
    placements = {placement.stage: placement for placement in result.placements}
    assert placements["load_a"].end_ns <= placements["compute"].start_ns
    assert not (
        placements["load_a"].start_ns < placements["load_b"].end_ns
        and placements["load_b"].start_ns < placements["load_a"].end_ns
    )


def test_resource_capacity_allows_parallel_instances():
    problem = PipelineProblem(stages=(
        StageSpec("a", "work", resources=("pipe",), payload=3.0),
        StageSpec("b", "work", resources=("pipe",), payload=2.0),
    ))
    hardware = PipelineHardware(resources=(ResourceSpec("pipe", capacity=2),))

    result = ListPipelineSolver().solve(problem, _Oracle(), hardware)

    assert result.status is SolveStatus.FEASIBLE
    assert result.makespan_ns == 3.0


@pytest.mark.parametrize("solver", [ListPipelineSolver(), CpSatPipelineSolver()])
def test_resource_demand_consumes_multiple_capacity_slots(solver):
    if isinstance(solver, CpSatPipelineSolver):
        pytest.importorskip("ortools")
    problem = PipelineProblem(stages=(
        StageSpec(
            "wide",
            "work",
            resource_demands=(ResourceDemand("pipe", 2),),
            payload=5.0,
        ),
        StageSpec("narrow", "work", resources=("pipe",), payload=5.0),
    ))
    hardware = PipelineHardware(resources=(ResourceSpec("pipe", capacity=2),))

    result = solver.solve(problem, _Oracle(), hardware)

    assert result.status in (SolveStatus.FEASIBLE, SolveStatus.OPTIMAL)
    assert result.makespan_ns == 10.0
    placements = {placement.stage: placement for placement in result.placements}
    assert placements["wide"].resource_demands == (ResourceDemand("pipe", 2),)


@pytest.mark.parametrize("solver", [ListPipelineSolver(), CpSatPipelineSolver()])
def test_resource_demand_larger_than_capacity_is_infeasible(solver):
    if isinstance(solver, CpSatPipelineSolver):
        pytest.importorskip("ortools")
    problem = PipelineProblem(stages=(StageSpec(
        "wide",
        "work",
        resource_demands=(ResourceDemand("pipe", 3),),
        payload=5.0,
    ),))
    hardware = PipelineHardware(resources=(ResourceSpec("pipe", capacity=2),))

    result = solver.solve(problem, _Oracle(), hardware)

    assert result.status is SolveStatus.INFEASIBLE
    assert "demand" in result.diagnostics[0]


def test_cycle_is_reported_as_infeasible():
    problem = PipelineProblem(
        stages=(
            StageSpec("a", "a", payload=1.0),
            StageSpec("b", "b", payload=1.0),
        ),
        precedences=(Precedence("a", "b"), Precedence("b", "a")),
    )

    result = ListPipelineSolver().solve(problem, _Oracle(), PipelineHardware())

    assert result.status is SolveStatus.INFEASIBLE
    assert "cycle" in result.diagnostics[0]


def test_cpsat_solver_proves_optimal_schedule_with_delay_and_capacity():
    pytest.importorskip("ortools")
    problem = PipelineProblem(
        stages=(
            StageSpec("load_a", "load", resources=("copy",), payload=3.0),
            StageSpec("load_b", "load", resources=("copy",), payload=2.0),
            StageSpec("compute", "compute", resources=("compute",), payload=4.0),
        ),
        precedences=(Precedence("load_a", "compute", delay_ns=1.0),),
    )
    hardware = PipelineHardware(resources=(
        ResourceSpec("copy", capacity=2),
        ResourceSpec("compute", capacity=1),
    ))

    result = CpSatPipelineSolver().solve(problem, _Oracle(), hardware)

    assert result.status is SolveStatus.OPTIMAL
    assert result.makespan_ns == 8.0
    assert result.lower_bound_ns == 8.0
    placements = {placement.stage: placement for placement in result.placements}
    assert placements["load_a"].end_ns + 1.0 <= placements["compute"].start_ns
    assert placements["load_a"].start_ns == placements["load_b"].start_ns == 0.0


def test_cpsat_solver_preserves_picosecond_resolution():
    pytest.importorskip("ortools")
    problem = PipelineProblem(
        stages=(
            StageSpec("a", "a", payload=0.123),
            StageSpec("b", "b", payload=0.456),
        ),
        precedences=(Precedence("a", "b"),),
    )

    result = CpSatPipelineSolver(time_resolution_ps=1).solve(
        problem, _Oracle(), PipelineHardware())

    assert result.status is SolveStatus.OPTIMAL
    assert result.makespan_ns == 0.579


def test_cpsat_solver_rounds_durations_and_delays_upward():
    pytest.importorskip("ortools")
    problem = PipelineProblem(
        stages=(
            StageSpec("a", "a", payload=0.149),
            StageSpec("b", "b", payload=0.149),
        ),
        precedences=(Precedence("a", "b", delay_ns=0.149),),
    )

    result = CpSatPipelineSolver(time_resolution_ps=100).solve(
        problem, _Oracle(), PipelineHardware())

    assert result.status is SolveStatus.OPTIMAL
    assert result.makespan_ns == pytest.approx(0.6)
    placements = {placement.stage: placement for placement in result.placements}
    assert placements["a"].end_ns >= 0.149
    assert placements["b"].start_ns - placements["a"].end_ns >= 0.149


def test_cpsat_solver_accepts_empty_pipeline():
    pytest.importorskip("ortools")

    result = CpSatPipelineSolver().solve(
        PipelineProblem(()), _Oracle(), PipelineHardware())

    assert result.status is SolveStatus.OPTIMAL
    assert result.makespan_ns == result.lower_bound_ns == 0.0
