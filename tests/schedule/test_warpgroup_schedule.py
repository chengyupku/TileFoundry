"""Contract and workflow tests for canonical warpgroup scheduling."""

from __future__ import annotations

import importlib
import json
import subprocess
import sys
from collections.abc import Callable
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
    LoopIterArg,
    MemorySpace,
    OperationCost,
    OperationCostEntry,
    OperationOutput,
    OperationSignature,
    ProgramInput,
    ProgramLoop,
    ProgramOperation,
    ResourceCapacity,
    ResourceWindow,
    ScalarLiteral,
    SynchronizationEdge,
    TensorType,
    TimedOperation,
    ValueRef,
    WarpgroupCostAmbiguityError,
    WarpgroupHardware,
    WarpgroupLane,
    WarpgroupMissingSignaturesError,
    WarpgroupProgram,
    WarpgroupSchedule,
    WarpgroupSerializationError,
    WarpgroupVerificationError,
    operation_signature,
    schedule_warpgroups,
    verify_warpgroup_schedule,
    warpgroup_hardware_from_json,
    warpgroup_hardware_to_json,
    warpgroup_program_from_json,
    warpgroup_program_to_json,
    warpgroup_schedule_from_json,
    warpgroup_schedule_to_json,
)
from tilefoundry.schedule.warpgroup.build import build_warpgroup_problem
from tilefoundry.schedule.warpgroup.model import ProblemLoop, ProblemOperation, WarpgroupProblem
from tilefoundry.schedule.warpgroup.solve import solve_warpgroup_problem
from tilefoundry.schedule.warpgroup.sync import export_warpgroup_schedule
from tilefoundry.schedule.warpgroup.verify import _verify_warpgroup_schedule

ROOT = Path(__file__).parents[2]
SCHEMAS = ROOT / "schemas"
EXAMPLE = ROOT / "examples" / "mla-schedule"


def _validator(name: str) -> Draft202012Validator:
    schema = json.loads((SCHEMAS / name).read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def _program(*, iterations: int = 3) -> WarpgroupProgram:
    types = (
        TensorType("source", (iterations, 1), DType.FP32, MemorySpace.GLOBAL),
        TensorType("shared", (1,), DType.FP32, MemorySpace.SHARED),
        TensorType("value", (1,), DType.FP32, MemorySpace.REGISTER),
    )
    operations = (
        ProgramOperation(
            "load",
            (
                OperationOutput(
                    "%shared",
                    "shared",
                    CopyExpression(
                        IndexExpression(ValueRef("%source"), (LoopIndexRef("%iteration"),))
                    ),
                ),
            ),
            0,
        ),
        ProgramOperation(
            "compute",
            (
                OperationOutput(
                    "%result",
                    "value",
                    ElementwiseExpression(ElementwiseOperator.EXP, (ValueRef("%shared"),)),
                ),
            ),
            1,
        ),
        ProgramOperation(
            "independent",
            (OperationOutput("%independent", "value", ScalarLiteral(1.0)),),
            2,
        ),
    )
    return WarpgroupProgram(
        PROGRAM_FORMAT,
        4,
        types,
        (ProgramInput("%source", "source"),),
        ProgramLoop("%iteration", iterations, (), operations),
    )


def _hardware(program: WarpgroupProgram, *, capacity: int = 1) -> WarpgroupHardware:
    costs: dict[OperationSignature, OperationCost] = {}
    for operation in program.loop.ops:
        signature = operation_signature(program, operation)
        if operation.id == "load":
            cost = OperationCost(1, 3, ())
        elif operation.id == "compute":
            cost = OperationCost(1, 2, (ResourceWindow("engine", 1, 0, 2),))
        else:
            cost = OperationCost(1, 1, (ResourceWindow("engine", 1, 0, 1),))
        costs[signature] = cost
    entries = tuple(
        OperationCostEntry(signature, cost)
        for signature, cost in sorted(
            costs.items(),
            key=lambda item: item[0].canonical_key,
        )
    )
    return WarpgroupHardware(
        HARDWARE_FORMAT,
        "cycle",
        (ResourceCapacity("engine", capacity),),
        entries,
    )


def _time_map(schedule: WarpgroupSchedule) -> dict[tuple[int, str], TimedOperation]:
    return {(item.iteration, item.operation_id): item for item in schedule.times}


def test_only_one_schema_exists_for_each_public_document() -> None:
    assert {path.name for path in SCHEMAS.glob("*.schema.json")} == {
        "warpgroup-program.schema.json",
        "warpgroup-hardware.schema.json",
        "warpgroup-schedule.schema.json",
    }
    for path in SCHEMAS.glob("*.schema.json"):
        Draft202012Validator.check_schema(json.loads(path.read_text(encoding="utf-8")))


def test_program_hardware_and_schedule_share_canonical_json_boundaries() -> None:
    program = _program()
    hardware = _hardware(program)
    result = schedule_warpgroups(program, hardware)

    program_json = warpgroup_program_to_json(program)
    hardware_json = warpgroup_hardware_to_json(hardware)
    schedule_json = warpgroup_schedule_to_json(result.schedule)
    _validator("warpgroup-program.schema.json").validate(json.loads(program_json))
    _validator("warpgroup-hardware.schema.json").validate(json.loads(hardware_json))
    _validator("warpgroup-schedule.schema.json").validate(json.loads(schedule_json))
    assert warpgroup_program_from_json(program_json) == program
    assert warpgroup_hardware_from_json(hardware_json) == hardware
    assert warpgroup_schedule_from_json(schedule_json) == result.schedule


def test_legacy_formats_are_rejected() -> None:
    program = _program()
    hardware = _hardware(program)
    schedule = schedule_warpgroups(program, hardware).schedule
    cases: tuple[tuple[Callable[[str], object], str, str], ...] = (
        (
            warpgroup_program_from_json,
            warpgroup_program_to_json(program),
            "tilefoundry.warpgroup_program.v2",
        ),
        (
            warpgroup_hardware_from_json,
            warpgroup_hardware_to_json(hardware),
            "tilefoundry.warpgroup_problem.v3",
        ),
        (
            warpgroup_schedule_from_json,
            warpgroup_schedule_to_json(schedule),
            "tilefoundry.warpgroup_schedule.v3",
        ),
    )
    for decoder, text, legacy_format in cases:
        payload = json.loads(text)
        payload["format"] = legacy_format
        with pytest.raises(WarpgroupSerializationError):
            decoder(json.dumps(payload))


@pytest.mark.parametrize(
    "invalid_expression",
    (
        ["copy", "%source", "%source"],
        ["sub", "%source"],
        ["reduce", "invalid", 0, "%source"],
        ["select", "%source", "%source"],
    ),
)
def test_schema_and_decoder_reject_invalid_expression_arity(
    invalid_expression: list[object],
) -> None:
    document = json.loads(warpgroup_program_to_json(_program()))
    document["loop"]["ops"][0]["outputs"][0]["expr"] = invalid_expression
    assert list(_validator("warpgroup-program.schema.json").iter_errors(document))
    with pytest.raises(WarpgroupSerializationError):
        warpgroup_program_from_json(json.dumps(document))


def test_hardware_matches_complete_identity_free_signatures() -> None:
    program = _program()
    hardware = _hardware(program)
    renamed_ops = tuple(
        replace(operation, id=f"renamed_{index}")
        for index, operation in enumerate(program.loop.ops)
    )
    renamed = replace(program, loop=replace(program.loop, ops=renamed_ops))
    assert {operation_signature(program, operation) for operation in program.loop.ops} == {
        operation_signature(renamed, operation) for operation in renamed.loop.ops
    }
    assert build_warpgroup_problem(renamed, hardware)

    missing = WarpgroupHardware(HARDWARE_FORMAT, "cycle", hardware.resources, hardware.entries[:-1])
    with pytest.raises(WarpgroupMissingSignaturesError):
        build_warpgroup_problem(program, missing)
    with pytest.raises(WarpgroupCostAmbiguityError):
        WarpgroupHardware(
            HARDWARE_FORMAT,
            "cycle",
            hardware.resources,
            (*hardware.entries, hardware.entries[0]),
        )


def test_public_workflow_is_periodic_fixed_owner_and_deterministic() -> None:
    program = _program(iterations=4)
    hardware = _hardware(program)
    first = schedule_warpgroups(program, hardware)
    second = schedule_warpgroups(program, hardware)
    assert first == second
    assert warpgroup_schedule_to_json(first.schedule) == warpgroup_schedule_to_json(second.schedule)
    assert len(first.schedule.times) == len(program.loop.ops) * program.loop.iterations
    assert first.schedule.lanes[0].operations == ("load",)
    assert first.schedule.lanes[1].operations == ("compute",)
    assert first.schedule.lanes[2].operations == ("independent",)
    assert first.schedule.lanes[3].operations == ()
    times = _time_map(first.schedule)
    intervals = {
        times[(iteration + 1, operation.id)].start - times[(iteration, operation.id)].start
        for operation in program.loop.ops
        for iteration in range(1, program.loop.iterations - 1)
    }
    assert len(intervals) == 1
    verify_warpgroup_schedule(program, hardware, first.schedule)


def test_verifier_requires_fixed_ownership_and_cross_lane_completion_sync() -> None:
    program = _program()
    hardware = _hardware(program)
    problem = build_warpgroup_problem(program, hardware)
    schedule = schedule_warpgroups(program, hardware).schedule
    required = SynchronizationEdge("load", "compute", 0)
    assert required in schedule.sync

    without_sync = replace(schedule, sync=tuple(edge for edge in schedule.sync if edge != required))
    with pytest.raises(WarpgroupVerificationError, match="no lane/sync path"):
        _verify_warpgroup_schedule(problem, without_sync)

    moved = list(schedule.lanes)
    moved[0] = WarpgroupLane(())
    moved[1] = WarpgroupLane(("load", "compute"))
    with pytest.raises(WarpgroupVerificationError, match="expected warp_group"):
        _verify_warpgroup_schedule(problem, replace(schedule, lanes=tuple(moved)))


def _periodic_resource_problem(
    *, capacity: int, iterations: int, window_duration: int = 2
) -> WarpgroupProblem:
    value_type = TensorType("result", (1,), DType.FP32, MemorySpace.REGISTER)
    return WarpgroupProblem(
        "cycle",
        2,
        (ResourceCapacity("engine", capacity),),
        (value_type,),
        (),
        ProblemLoop(
            "%iteration",
            iterations,
            (),
            (
                ProblemOperation(
                    "early",
                    (OperationOutput("%early", "result", ScalarLiteral(1.0)),),
                    0,
                    1,
                    window_duration,
                    (ResourceWindow("engine", 1, 0, window_duration),),
                ),
                ProblemOperation(
                    "late",
                    (OperationOutput("%late", "result", ScalarLiteral(2.0)),),
                    1,
                    1,
                    window_duration + 1,
                    (ResourceWindow("engine", 1, 1, window_duration),),
                ),
            ),
        ),
    )


def test_periodic_resource_capacity_and_self_overlap() -> None:
    for capacity, expected_ii in ((1, 4), (2, 2)):
        problem = _periodic_resource_problem(capacity=capacity, iterations=3)
        result = solve_warpgroup_problem(problem)
        schedule = export_warpgroup_schedule(problem, result)
        _verify_warpgroup_schedule(problem, schedule)
        times = _time_map(schedule)
        assert times[(2, "early")].start - times[(1, "early")].start == expected_ii

    for capacity, expected_ii in ((4, 3), (5, 2)):
        problem = _periodic_resource_problem(
            capacity=capacity,
            iterations=3,
            window_duration=5,
        )
        schedule = export_warpgroup_schedule(problem, solve_warpgroup_problem(problem))
        times = _time_map(schedule)
        initiation_interval = times[(2, "early")].start - times[(1, "early")].start
        assert initiation_interval == expected_ii
        if capacity == 5:
            assert (5 + initiation_interval - 1) // initiation_interval == 3


def test_periodic_model_size_does_not_depend_on_requested_iterations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cp_model = importlib.import_module("ortools.sat.python.cp_model")
    original_model = cp_model.CpModel
    models: list[object] = []

    class CountingModel:
        def __init__(self) -> None:
            self.inner = original_model()
            models.append(self.inner)

        def NewIntVar(self, *args: object, **kwargs: object) -> object:
            return self.inner.NewIntVar(*args, **kwargs)

        def NewBoolVar(self, *args: object, **kwargs: object) -> object:
            return self.inner.NewBoolVar(*args, **kwargs)

        def __getattr__(self, name: str) -> object:
            return getattr(self.inner, name)

    monkeypatch.setattr(cp_model, "CpModel", CountingModel)
    shapes = []
    for iterations in (3, 64):
        problem = _periodic_resource_problem(capacity=1, iterations=iterations)
        schedule = export_warpgroup_schedule(problem, solve_warpgroup_problem(problem))
        _verify_warpgroup_schedule(problem, schedule)
        proto = getattr(models[-1], "Proto")()
        shapes.append((len(proto.variables), len(proto.constraints)))
        assert len(schedule.times) == iterations * len(problem.loop.ops)
    assert shapes[0] == shapes[1]


def test_external_shared_init_overlaps_prologue_and_protects_body_reuse() -> None:
    types = (
        TensorType("source", (2, 1), DType.FP32, MemorySpace.GLOBAL),
        TensorType("shared", (1,), DType.FP32, MemorySpace.SHARED),
        TensorType("result", (1,), DType.FP32, MemorySpace.REGISTER),
    )
    problem = WarpgroupProblem(
        "cycle",
        2,
        (),
        types,
        (ProgramInput("%source", "source"), ProgramInput("%initial", "shared")),
        ProblemLoop(
            "%iteration",
            2,
            (LoopIterArg("%carry", ValueRef("%initial"), ValueRef("%shared")),),
            (
                ProblemOperation(
                    "publish",
                    (
                        OperationOutput(
                            "%shared",
                            "shared",
                            CopyExpression(
                                IndexExpression(
                                    ValueRef("%source"),
                                    (LoopIndexRef("%iteration"),),
                                )
                            ),
                        ),
                    ),
                    0,
                    1,
                    1,
                    (),
                ),
                ProblemOperation(
                    "consume",
                    (OperationOutput("%result", "result", CopyExpression(ValueRef("%carry"))),),
                    1,
                    4,
                    4,
                    (),
                ),
            ),
        ),
    )
    schedule = export_warpgroup_schedule(problem, solve_warpgroup_problem(problem))
    times = _time_map(schedule)
    consume_zero = times[(0, "consume")]
    publish_zero = times[(0, "publish")]
    assert max(consume_zero.start, publish_zero.start) < min(
        consume_zero.completion,
        publish_zero.completion,
    )
    assert publish_zero.completion <= times[(1, "consume")].start
    assert times[(1, "consume")].completion <= times[(1, "publish")].start
    assert SynchronizationEdge("publish", "consume", 1) in schedule.sync
    _verify_warpgroup_schedule(problem, schedule)


def test_cli_requires_program_and_hardware_and_matches_public_api(tmp_path: Path) -> None:
    program = _program()
    hardware = _hardware(program)
    program_path = tmp_path / "program.json"
    hardware_path = tmp_path / "hardware.json"
    program_path.write_text(warpgroup_program_to_json(program), encoding="utf-8")
    hardware_path.write_text(warpgroup_hardware_to_json(hardware), encoding="utf-8")
    command = [
        sys.executable,
        "-m",
        "tilefoundry.cli",
        "schedule",
        "--program",
        str(program_path),
        "--hardware",
        str(hardware_path),
        "--json",
    ]
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stderr
    expected = warpgroup_schedule_to_json(schedule_warpgroups(program, hardware).schedule)
    assert completed.stdout.strip() == expected

    missing = subprocess.run(
        command[:-3] + ["--json"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert missing.returncode == 2
    assert "--hardware" in missing.stderr


def test_mla_example_is_one_complete_three_document_workflow() -> None:
    program_text = (EXAMPLE / "program.json").read_text(encoding="utf-8")
    hardware_text = (EXAMPLE / "hardware.json").read_text(encoding="utf-8")
    schedule_text = (EXAMPLE / "schedule.json").read_text(encoding="utf-8")
    program = warpgroup_program_from_json(program_text)
    hardware = warpgroup_hardware_from_json(hardware_text)
    expected = warpgroup_schedule_from_json(schedule_text)
    _validator("warpgroup-program.schema.json").validate(json.loads(program_text))
    _validator("warpgroup-hardware.schema.json").validate(json.loads(hardware_text))
    _validator("warpgroup-schedule.schema.json").validate(json.loads(schedule_text))
    verify_warpgroup_schedule(program, hardware, expected)


def test_typed_and_verifier_imports_do_not_load_solver_or_device_modules() -> None:
    script = """
import sys
from tilefoundry.schedule.warpgroup import (
    WarpgroupHardware,
    WarpgroupProgram,
    verify_warpgroup_schedule,
)
assert 'ortools' not in sys.modules
assert not any(name.startswith(('cuda', 'tileprof', 'tilefoundry_costmodel')) for name in sys.modules)
print(WarpgroupProgram.__name__, WarpgroupHardware.__name__, verify_warpgroup_schedule.__name__)
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert (
        completed.stdout.strip() == "WarpgroupProgram WarpgroupHardware verify_warpgroup_schedule"
    )
