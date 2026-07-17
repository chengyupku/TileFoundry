from tilefoundry_costmodel import (
    ListPipelineSolver,
    PipelineHardware,
    PipelineProblem,
    Precedence,
    ResourceSpec,
    SolveStatus,
    StageSpec,
    StageTiming,
)


class _Oracle:
    def estimate(self, stage, *, hardware):
        del hardware
        return StageTiming(stage.payload)


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
