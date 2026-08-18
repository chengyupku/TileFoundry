"""Solver parallelism is a caller choice, not a constant."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tilefoundry.schedule.warpgroup import (
    operation_signature,
    schedule_warpgroups,
    warpgroup_program_from_json,
)
from tilefoundry.schedule.warpgroup.solve import solve_warpgroup_problem
from tilefoundry.schedule.warpgroup.build import build_warpgroup_problem
from tilefoundry.schedule.warpgroup.cost import (
    HARDWARE_FORMAT,
    OperationCost,
    OperationCostEntry,
    OperationKind,
    WarpgroupHardware,
)
from tilefoundry.schedule.warpgroup.errors import WarpgroupValidationError
from tilefoundry.schedule.warpgroup.model import ResourceCapacity, ResourceWindow

# Two loads feeding one matmul. Written here rather than read from a document,
# because the reference documents this test used to open have been removed and a
# test that a refactor can delete the input of is not a test.
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
            {"id": "load_a", "warp_group": 1, "outputs": [{"id": "%a", "type": "tile",
              "expr": ["copy", ["index", "%source", "%i", 0]]}]},
            {"id": "load_b", "warp_group": 1, "outputs": [{"id": "%b", "type": "tile",
              "expr": ["copy", ["index", "%source", "%i", 1]]}]},
            {"id": "matmul", "warp_group": 0, "outputs": [{"id": "%next", "type": "accumulator",
              "expr": ["add", "%accumulator",
                       ["matmul", "%a", ["transpose", "%b"]]]}]},
        ],
    },
}


def _program():
    return warpgroup_program_from_json(json.dumps(PROGRAM))


def _library(program):
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
                    index + 1,
                    index + 1,
                    (ResourceWindow("engine", 1, index + 1),)
                    if signature.kind is OperationKind.COPY
                    else (),
                ),
            )
            for index, signature in enumerate(signatures)
        ),
    )


@pytest.mark.parametrize("workers", [1, 4])
def test_search_workers_schedules_every_operation(workers: int) -> None:
    program = _program()
    result = schedule_warpgroups(program, _library(program), search_workers=workers)
    placed = {item for lane in result.schedule.lanes for item in lane.operations}
    assert placed == {operation.id for operation in program.loop.ops}


def test_one_worker_is_the_default_and_reproduces_its_arrangement() -> None:
    program = _program()
    library = _library(program)
    first = schedule_warpgroups(program, library)
    second = schedule_warpgroups(program, library, search_workers=1)
    assert [lane.operations for lane in first.schedule.lanes] == [
        lane.operations for lane in second.schedule.lanes
    ]


@pytest.mark.parametrize("workers", [0, -1, 1.0, "2"])
def test_search_workers_rejects_anything_but_a_positive_integer(workers: object) -> None:
    program = _program()
    problem = build_warpgroup_problem(program, _library(program))
    with pytest.raises(WarpgroupValidationError):
        solve_warpgroup_problem(problem, search_workers=workers)
