"""The prologue and the epilogue: what a program says about running once."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from tilefoundry.schedule.warpgroup import (
    HARDWARE_FORMAT,
    PROGRAM_FORMAT,
    CopyExpression,
    DType,
    ElementwiseExpression,
    ElementwiseOperator,
    IndexExpression,
    LoopIndexRef,
    MemorySpace,
    OperationCost,
    OperationCostEntry,
    OperationOutput,
    OperationSignature,
    ProgramInput,
    ProgramLoop,
    ProgramOperation,
    RegionOperation,
    ResourceCapacity,
    ScalarLiteral,
    TensorType,
    ValueRef,
    WarpgroupHardware,
    WarpgroupLane,
    WarpgroupProgram,
    WarpgroupValidationError,
    WarpgroupVerificationError,
    operation_signature,
    schedule_warpgroups,
    warpgroup_program_from_json,
    warpgroup_program_to_json,
    warpgroup_schedule_from_json,
    warpgroup_schedule_to_json,
)
from tilefoundry.schedule.warpgroup.build import build_warpgroup_problem
from tilefoundry.schedule.warpgroup.verify import _verify_warpgroup_schedule

SCHEMAS = Path(__file__).parents[2] / "schemas"

TYPES = (
    TensorType("stream", (3, 1), DType.FP32, MemorySpace.GLOBAL),
    TensorType("tile", (1,), DType.FP32, MemorySpace.SHARED),
    TensorType("value", (1,), DType.FP32, MemorySpace.REGISTER),
    TensorType("written", (1,), DType.FP32, MemorySpace.GLOBAL),
    TensorType("counter", (1,), DType.I32, MemorySpace.REGISTER),
)

BODY = (
    ProgramOperation(
        "load",
        (
            OperationOutput(
                "%tile",
                "tile",
                CopyExpression(IndexExpression(ValueRef("%stream"), (LoopIndexRef("%trip"),))),
            ),
        ),
        0,
    ),
    ProgramOperation(
        "compute",
        (
            OperationOutput(
                "%value",
                "value",
                ElementwiseExpression(ElementwiseOperator.EXP, (ValueRef("%tile"),)),
            ),
        ),
        1,
    ),
)

#: A transfer that makes a tile resident before the first trip, on the lane that
#: will read it, and a write-back of the last value the loop produced.
PROLOGUE = (
    RegionOperation(
        "stage",
        1,
        (
            OperationOutput(
                "%resident",
                "tile",
                CopyExpression(IndexExpression(ValueRef("%stream"), (0,))),
            ),
        ),
    ),
)
EPILOGUE = (
    RegionOperation(
        "store",
        1,
        (OperationOutput("%written", "written", CopyExpression(ValueRef("%value"))),),
    ),
)


def _validator(name: str) -> Draft202012Validator:
    return Draft202012Validator(json.loads((SCHEMAS / name).read_text(encoding="utf-8")))


def _program(
    *,
    prologue: tuple[RegionOperation, ...] = (),
    epilogue: tuple[RegionOperation, ...] = (),
    body: tuple[ProgramOperation, ...] = BODY,
) -> WarpgroupProgram:
    return WarpgroupProgram(
        PROGRAM_FORMAT,
        3,
        TYPES,
        (ProgramInput("%stream", "stream"),),
        ProgramLoop("%trip", 3, (), body),
        prologue,
        epilogue,
    )


def _hardware(program: WarpgroupProgram) -> WarpgroupHardware:
    costs: dict[OperationSignature, OperationCost] = {}
    for operation in program.loop.ops:
        costs[operation_signature(program, operation)] = OperationCost(1, 2, ())
    entries = tuple(
        OperationCostEntry(signature, cost)
        for signature, cost in sorted(costs.items(), key=lambda item: item[0].canonical_key)
    )
    return WarpgroupHardware(HARDWARE_FORMAT, "cycle", (ResourceCapacity("engine", 1),), entries)


def test_a_program_with_no_region_writes_no_region_field() -> None:
    program = _program()
    document = json.loads(warpgroup_program_to_json(program))
    assert "prologue" not in document and "epilogue" not in document

    schedule = schedule_warpgroups(program, _hardware(program)).schedule
    lanes = json.loads(warpgroup_schedule_to_json(schedule))["lanes"]
    assert all(type(lane) is list for lane in lanes)


def test_both_regions_round_trip_through_serialization_unchanged() -> None:
    program = _program(prologue=PROLOGUE, epilogue=EPILOGUE)
    text = warpgroup_program_to_json(program)
    document = json.loads(text)
    _validator("warpgroup-program.schema.json").validate(document)
    assert document["prologue"][0]["warp_group"] == 1
    assert warpgroup_program_from_json(text) == program

    schedule = schedule_warpgroups(program, _hardware(program)).schedule
    schedule_text = warpgroup_schedule_to_json(schedule)
    _validator("warpgroup-schedule.schema.json").validate(json.loads(schedule_text))
    assert warpgroup_schedule_from_json(schedule_text) == schedule
    assert json.loads(schedule_text)["lanes"][1] == {
        "operations": ["compute"],
        "prologue": ["stage"],
        "epilogue": ["store"],
    }


def test_adding_a_region_leaves_the_loop_and_its_solve_alone() -> None:
    """A region runs once, so nothing about it may reach the periodic model."""
    plain = _program()
    with_regions = _program(prologue=PROLOGUE, epilogue=EPILOGUE)
    hardware = _hardware(plain)
    assert build_warpgroup_problem(with_regions, hardware).dependencies() == (
        build_warpgroup_problem(plain, hardware).dependencies()
    )

    first = schedule_warpgroups(plain, hardware)
    second = schedule_warpgroups(with_regions, hardware)
    assert first.status == second.status
    assert first.makespan == second.makespan
    assert first.schedule.times == second.schedule.times
    assert first.schedule.sync == second.schedule.sync
    assert tuple(lane.operations for lane in first.schedule.lanes) == tuple(
        lane.operations for lane in second.schedule.lanes
    )


def test_a_region_operation_must_name_a_lane_the_program_declares() -> None:
    stray = replace(PROLOGUE[0], warp_group=3)
    with pytest.raises(WarpgroupValidationError, match="warp_group is out of range"):
        _program(prologue=(stray,))


def test_a_region_operation_may_not_reuse_a_body_operation_id() -> None:
    with pytest.raises(WarpgroupValidationError, match="used more than once"):
        _program(epilogue=(replace(EPILOGUE[0], id="load"),))


def test_a_region_reads_only_what_the_program_has_already_defined() -> None:
    absent = replace(
        EPILOGUE[0],
        outputs=(OperationOutput("%written", "written", CopyExpression(ValueRef("%absent"))),),
    )
    with pytest.raises(WarpgroupValidationError, match="undefined SSA value '%absent'"):
        _program(epilogue=(absent,))

    # A prologue runs before the loop, so a body definition does not exist yet.
    early = replace(
        PROLOGUE[0],
        outputs=(OperationOutput("%resident", "tile", CopyExpression(ValueRef("%value"))),),
    )
    with pytest.raises(WarpgroupValidationError, match="undefined SSA value '%value'"):
        _program(prologue=(early,))

    indexed = replace(
        EPILOGUE[0],
        outputs=(
            OperationOutput(
                "%written",
                "written",
                CopyExpression(IndexExpression(ValueRef("%stream"), (LoopIndexRef("%trip"),))),
            ),
        ),
    )
    with pytest.raises(WarpgroupValidationError, match="loop index"):
        _program(epilogue=(indexed,))


def test_only_the_epilogue_may_define_global_storage() -> None:
    """A kernel writes its result once, after the last trip, and nowhere else."""
    assert _program(epilogue=EPILOGUE).epilogue == EPILOGUE
    written_early = replace(
        PROLOGUE[0],
        outputs=(OperationOutput("%early", "written", CopyExpression(ValueRef("%stream"))),),
    )
    with pytest.raises(WarpgroupValidationError, match="global external storage"):
        _program(prologue=(written_early,))


def test_the_schedule_carries_each_region_on_the_lane_that_declared_it() -> None:
    program = _program(prologue=PROLOGUE, epilogue=EPILOGUE)
    hardware = _hardware(program)
    problem = build_warpgroup_problem(program, hardware)
    schedule = schedule_warpgroups(program, hardware).schedule
    assert schedule.lanes[1].prologue == ("stage",)
    assert schedule.lanes[1].epilogue == ("store",)
    assert schedule.lanes[0].prologue == () and schedule.lanes[0].epilogue == ()

    moved = list(schedule.lanes)
    moved[0] = replace(moved[0], epilogue=("store",))
    moved[1] = replace(moved[1], epilogue=())
    with pytest.raises(WarpgroupVerificationError, match="lane 0 epilogue"):
        _verify_warpgroup_schedule(problem, replace(schedule, lanes=tuple(moved)))

    dropped = list(schedule.lanes)
    dropped[1] = replace(dropped[1], prologue=())
    with pytest.raises(WarpgroupVerificationError, match="lane 1 prologue"):
        _verify_warpgroup_schedule(problem, replace(schedule, lanes=tuple(dropped)))


def test_a_region_keeps_the_order_it_was_written_in() -> None:
    second = RegionOperation(
        "store_again",
        1,
        (OperationOutput("%written_again", "written", CopyExpression(ValueRef("%written"))),),
    )
    program = _program(epilogue=(*EPILOGUE, second))
    hardware = _hardware(program)
    problem = build_warpgroup_problem(program, hardware)
    schedule = schedule_warpgroups(program, hardware).schedule
    assert schedule.lanes[1].epilogue == ("store", "store_again")

    reversed_lanes = list(schedule.lanes)
    reversed_lanes[1] = replace(reversed_lanes[1], epilogue=("store_again", "store"))
    with pytest.raises(WarpgroupVerificationError, match="lane 1 epilogue"):
        _verify_warpgroup_schedule(problem, replace(schedule, lanes=tuple(reversed_lanes)))


def test_an_epilogue_may_not_read_another_lanes_register_file() -> None:
    program = _program(epilogue=(replace(EPILOGUE[0], warp_group=0),))
    with pytest.raises(WarpgroupVerificationError, match="crosses warpgroup lanes"):
        schedule_warpgroups(program, _hardware(program))


def test_a_lane_names_an_operation_in_one_region_only() -> None:
    with pytest.raises(WarpgroupValidationError, match="more than one region"):
        WarpgroupLane(("load",), prologue=("load",))


def test_an_epilogue_divides_by_what_the_loop_accumulated() -> None:
    """The rescale after the last trip, and the operator it needed.

    An attention output is divided by the denominator its loop summed. There is
    no writing of that as a multiply without a reciprocal the grammar also
    lacks, so admitting a region and refusing division would have admitted a
    region nobody can fill.
    """
    rescale = RegionOperation(
        "rescale",
        1,
        (
            OperationOutput(
                "%scaled",
                "value",
                ElementwiseExpression(
                    ElementwiseOperator.DIV, (ValueRef("%value"), ValueRef("%value"))
                ),
            ),
        ),
    )
    program = _program(epilogue=(rescale,))
    text = warpgroup_program_to_json(program)
    _validator("warpgroup-program.schema.json").validate(json.loads(text))
    assert warpgroup_program_from_json(text) == program

    with pytest.raises(WarpgroupValidationError, match="exactly two operands"):
        ElementwiseExpression(
            ElementwiseOperator.DIV,
            (ValueRef("%value"), ValueRef("%value"), ValueRef("%value")),
        )

    counted = (
        *BODY,
        ProgramOperation("count", (OperationOutput("%count", "counter", ScalarLiteral(1)),), 2),
    )
    integer = replace(
        rescale,
        outputs=(
            OperationOutput(
                "%scaled",
                "counter",
                ElementwiseExpression(
                    ElementwiseOperator.DIV, (ValueRef("%count"), ValueRef("%count"))
                ),
            ),
        ),
    )
    with pytest.raises(WarpgroupValidationError, match="div requires a floating dtype"):
        _program(body=counted, epilogue=(integer,))
