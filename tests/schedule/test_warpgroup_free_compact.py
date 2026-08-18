"""Compact periodic scheduling that also chooses the lane assignment."""

from __future__ import annotations

import json

import pytest

from tilefoundry.schedule.warpgroup import (
    PROGRAM_FORMAT,
    operation_signature,
    solve_free_compact,
    warpgroup_program_from_json,
)
from tilefoundry.schedule.warpgroup.build import build_warpgroup_problem
from tilefoundry.schedule.warpgroup.cost import (
    HARDWARE_FORMAT,
    OperationCost,
    OperationCostEntry,
    OperationKind,
    WarpgroupHardware,
)
from tilefoundry.schedule.warpgroup.errors import (
    WarpgroupInfeasibleError,
    WarpgroupValidationError,
)
from tilefoundry.schedule.warpgroup.model import ResourceCapacity, ResourceWindow

# Two loads feeding one matmul, over four trips. Small enough to solve in
# milliseconds and still the shape that matters: a shared value written by one
# operation and read by another, so ownership is a real choice.
PROGRAM = {
    "format": "tilefoundry.warpgroup_program",
    "warp_groups": 2,
    "types": {
        "source": {"shape": [4, 2, 64, 64], "dtype": "bf16", "space": "global"},
        "tile": {"shape": [64, 64], "dtype": "bf16", "space": "shared"},
        "accumulator": {"shape": [64, 64], "dtype": "fp32", "space": "register"},
    },
    "inputs": [{"id": "%source", "type": "source"}],
    "loop": {
        "index": "%i",
        "iterations": 4,
        "iter_args": [{"id": "%accumulator", "init": 0.0, "yield": "%next"}],
        "ops": [
            {
                "id": "load_a",
                "warp_group": 1,
                "outputs": [
                    {
                        "id": "%a",
                        "type": "tile",
                        "expr": ["copy", ["index", "%source", "%i", 0]],
                    }
                ],
            },
            {
                "id": "load_b",
                "warp_group": 1,
                "outputs": [
                    {
                        "id": "%b",
                        "type": "tile",
                        "expr": ["copy", ["index", "%source", "%i", 1]],
                    }
                ],
            },
            {
                "id": "matmul",
                "warp_group": 0,
                "outputs": [
                    {
                        "id": "%next",
                        "type": "accumulator",
                        "expr": [
                            "add",
                            "%accumulator",
                            ["matmul", "%a", ["transpose", "%b"]],
                        ],
                    }
                ],
            },
        ],
    },
}

TILE_BYTES = 64 * 64 * 2


def _program():
    return warpgroup_program_from_json(json.dumps(PROGRAM))


def _library(program, *, copy_cost=8, matmul_cost=12):
    """One engine, held by the copies, so the period has a resource floor."""
    signatures = sorted(
        {operation_signature(program, operation) for operation in program.loop.ops},
        key=lambda item: item.canonical_key,
    )
    return WarpgroupHardware(
        HARDWARE_FORMAT,
        "cycle",
        (ResourceCapacity("engine", 1),),
        tuple(
            OperationCostEntry(
                signature,
                OperationCost(
                    copy_cost if signature.kind is OperationKind.COPY else matmul_cost,
                    copy_cost if signature.kind is OperationKind.COPY else matmul_cost,
                    (ResourceWindow("engine", 1, copy_cost),)
                    if signature.kind is OperationKind.COPY
                    else (),
                ),
            )
            for signature in signatures
        ),
    )


def _solved(**kwargs):
    program = _program()
    problem = build_warpgroup_problem(program, _library(program))
    return solve_free_compact(problem, timeout_seconds=30.0, **kwargs)


def test_every_operation_lands_on_some_lane():
    result = _solved()
    placed = [item for lane in result.lanes for item in lane.operations]
    assert sorted(placed) == ["load_a", "load_b", "matmul"]
    assert len(placed) == len(set(placed))


def test_the_result_carries_the_period_it_solved_for():
    """A compact model produces a period, and three iterations are not always
    available to recover it from the timing witness."""
    result = _solved()
    assert result.initiation_interval is not None
    assert result.initiation_interval >= 8          # the widest issue duration


def test_the_period_is_at_least_what_the_engine_can_deliver():
    """Two copies of eight cycles share one engine, so no period is under 16."""
    result = _solved()
    assert result.initiation_interval >= 16


def test_pinning_an_operation_puts_it_on_that_lane():
    result = _solved(pin={"matmul": 1})
    assert "matmul" in result.lanes[1].operations


def test_pinning_is_checked_against_the_program_and_the_warp_groups():
    program = _program()
    problem = build_warpgroup_problem(program, _library(program))
    with pytest.raises(WarpgroupValidationError):
        solve_free_compact(problem, timeout_seconds=30.0, pin={"absent": 0})
    with pytest.raises(WarpgroupValidationError):
        solve_free_compact(problem, timeout_seconds=30.0, pin={"matmul": 7})


@pytest.mark.parametrize("timeout", [0, -1.0, float("inf"), "30"])
def test_the_timeout_must_be_a_positive_finite_number(timeout: object):
    program = _program()
    problem = build_warpgroup_problem(program, _library(program))
    with pytest.raises(WarpgroupValidationError):
        solve_free_compact(problem, timeout_seconds=timeout)


# --- the capacities the problem format does not carry ----------------------

def _shared(capacity):
    return {
        "capacity": capacity,
        "values": {
            "%a": ("load_a", ["matmul"], TILE_BYTES),
            "%b": ("load_b", ["matmul"], TILE_BYTES),
        },
    }


def test_shared_memory_holds_a_value_from_its_write_to_its_last_read():
    """Both tiles are live at the matmul, so both have to fit at once."""
    result = _solved(shared=_shared(2 * TILE_BYTES))
    assert result.initiation_interval is not None


def test_a_body_that_cannot_fit_is_refused_rather_than_returned():
    """Without this the model happily returns a schedule nothing can build.

    The two tiles overlap by construction -- one matmul reads both -- so a
    capacity of one tile is not a packing problem, it is infeasible.
    """
    program = _program()
    problem = build_warpgroup_problem(program, _library(program))
    with pytest.raises(WarpgroupInfeasibleError):
        solve_free_compact(
            problem, timeout_seconds=30.0, shared=_shared(TILE_BYTES)
        )


def test_registers_carried_across_the_loop_have_to_fit_the_file():
    """`setmaxnreg` takes a multiple of eight in [24, 256] and the lanes share
    one register file, so a demand past it is infeasible rather than rounded."""
    program = _program()
    problem = build_warpgroup_problem(program, _library(program))
    assert solve_free_compact(
        problem, timeout_seconds=30.0, registers={"matmul": 128 * 64}
    ).initiation_interval is not None
    with pytest.raises(WarpgroupInfeasibleError):
        solve_free_compact(
            problem, timeout_seconds=30.0, registers={"matmul": 128 * 4096}
        )


# --- an operation whose lane is not decided --------------------------------

def _unassigned():
    """The same program with every `warp_group` removed."""
    import copy

    document = copy.deepcopy(PROGRAM)
    for operation in document["loop"]["ops"]:
        operation.pop("warp_group", None)
    return document


def test_an_operation_may_arrive_without_a_lane():
    program = warpgroup_program_from_json(json.dumps(_unassigned()))
    assert all(operation.warp_group is None for operation in program.loop.ops)


def test_absent_is_not_written_back_as_a_lane():
    """A round trip must not invent an owner for an operation that had none."""
    from tilefoundry.schedule.warpgroup import warpgroup_program_to_json

    document = _unassigned()
    program = warpgroup_program_from_json(json.dumps(document))
    written = json.loads(warpgroup_program_to_json(program))
    assert all("warp_group" not in operation for operation in written["loop"]["ops"])
    assert warpgroup_program_from_json(json.dumps(written)) == program


def test_the_search_places_an_operation_that_arrived_without_a_lane():
    program = warpgroup_program_from_json(json.dumps(_unassigned()))
    problem = build_warpgroup_problem(program, _library(program))
    result = solve_free_compact(problem, timeout_seconds=30.0)
    placed = sorted(item for lane in result.lanes for item in lane.operations)
    assert placed == ["load_a", "load_b", "matmul"]


def test_the_search_replaces_a_lane_that_was_declared():
    """Declared ownership is an input to the fixed-owner model, not to this one.

    Pinning every operation to lane 0 and solving anyway must not return that
    arrangement -- three operations on one lane of two is not what a search
    minimising the makespan would choose.
    """
    import copy

    document = copy.deepcopy(PROGRAM)
    for operation in document["loop"]["ops"]:
        operation["warp_group"] = 0
    program = warpgroup_program_from_json(json.dumps(document))
    problem = build_warpgroup_problem(program, _library(program))
    result = solve_free_compact(problem, timeout_seconds=30.0)
    assert any(lane.operations for lane in result.lanes[1:])


def test_the_fixed_owner_model_refuses_what_it_cannot_decide():
    """It schedules an assignment; it does not invent one, and it says which."""
    from tilefoundry.schedule.warpgroup.solve import solve_warpgroup_problem

    program = warpgroup_program_from_json(json.dumps(_unassigned()))
    problem = build_warpgroup_problem(program, _library(program))
    with pytest.raises(WarpgroupValidationError) as error:
        solve_warpgroup_problem(problem)
    assert "load_a" in str(error.value)
