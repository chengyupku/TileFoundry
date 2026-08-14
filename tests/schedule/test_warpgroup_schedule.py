"""Strict typed boundary for finite warpgroup scheduling."""

from __future__ import annotations

import dataclasses
import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Callable

import pytest
from jsonschema import Draft202012Validator
from referencing import Registry, Resource

from tilefoundry.schedule.warpgroup import (
    PROBLEM_FORMAT,
    PROBLEM_FORMAT_V2,
    PROBLEM_FORMAT_V3,
    PROGRAM_FORMAT,
    PROGRAM_FORMAT_V2,
    SCHEDULE_FORMAT,
    SCHEDULE_FORMAT_V2,
    SCHEDULE_FORMAT_V3,
    DefUseDependency,
    DType,
    ElementwiseExpression,
    ElementwiseOperator,
    LoopIndexRef,
    LoopIterArg,
    MemorySpace,
    OperationCost,
    OperationCostEntry,
    OperationCostLibrary,
    OperationKind,
    OperationOutput,
    OperationSignature,
    ProblemLoop,
    ProblemOperation,
    ProgramInput,
    ProgramLoop,
    ProgramOperation,
    ResourceCapacity,
    ResourceDemand,
    ResourceWindow,
    ScalarLiteral,
    SynchronizationEdge,
    TensorType,
    TimedOperation,
    ValueRef,
    WarpgroupCostAmbiguityError,
    WarpgroupCostMissingError,
    WarpgroupLane,
    WarpgroupMissingSignaturesError,
    WarpgroupProblem,
    WarpgroupProgram,
    WarpgroupSchedule,
    WarpgroupScheduleResult,
    WarpgroupSerializationError,
    WarpgroupSolveResult,
    WarpgroupValidationError,
    build_warpgroup_problem,
    export_warpgroup_schedule,
    operation_signature,
    schedule_warpgroups,
    solve_warpgroup_problem,
    verify_warpgroup_schedule,
    warpgroup_problem_from_json,
    warpgroup_problem_to_json,
    warpgroup_program_from_json,
    warpgroup_program_to_json,
    warpgroup_schedule_from_json,
    warpgroup_schedule_to_json,
)
from tilefoundry.schedule.warpgroup.errors import WarpgroupVerificationError
from tilefoundry.schedule.warpgroup.expression import (
    CopyExpression,
    IndexExpression,
    value_references,
)
from tilefoundry.target.cuda.warpgroup_costs import (
    B200CalibrationMissingError,
    B200OperationFamily,
    CalibrationStatus,
    b200_global_to_shared_copy_signature,
    b200_warpgroup_coverage_matrix,
)

ROOT = Path(__file__).parents[2]
DESIGN = ROOT / "docs" / "design"
SCHEMAS = ROOT / "schemas"


def _document(name: str) -> str:
    return (DESIGN / name).read_text(encoding="utf-8")


def _validator(name: str) -> Draft202012Validator:
    path = SCHEMAS / name
    schema = json.loads(path.read_text(encoding="utf-8"))
    resources = []
    for schema_path in SCHEMAS.glob("warpgroup-*.schema.json"):
        document = json.loads(schema_path.read_text(encoding="utf-8"))
        resources.append((document["$id"], Resource.from_contents(document)))
    return Draft202012Validator(schema, registry=Registry().with_resources(resources))


def _generic_values() -> tuple[WarpgroupProgram, WarpgroupProblem, WarpgroupSchedule]:
    types = (
        TensorType("external", (2, 4), DType.FP32, MemorySpace.GLOBAL),
        TensorType("tile", (4,), DType.FP32, MemorySpace.SHARED),
        TensorType("state", (4,), DType.FP32, MemorySpace.REGISTER),
    )
    inputs = (ProgramInput("%external", "external"),)
    iter_args = (LoopIterArg("%state", ScalarLiteral(0.0), ValueRef("%next")),)
    load_outputs = (
        OperationOutput(
            "%tile",
            "tile",
            CopyExpression(IndexExpression(ValueRef("%external"), (LoopIndexRef("%i"),))),
        ),
    )
    independent_outputs = (OperationOutput("%independent", "state", ScalarLiteral(1.0)),)
    advance_outputs = (
        OperationOutput(
            "%next",
            "state",
            ElementwiseExpression(
                ElementwiseOperator.ADD,
                (ValueRef("%state"), ValueRef("%tile")),
            ),
        ),
    )
    program_ops = (
        ProgramOperation("load", load_outputs),
        ProgramOperation("independent", independent_outputs),
        ProgramOperation("advance", advance_outputs),
    )
    problem_ops = (
        ProblemOperation("load", load_outputs, 1, (ResourceDemand("copy", 1),)),
        ProblemOperation("independent", independent_outputs, 1, ()),
        ProblemOperation("advance", advance_outputs, 1, ()),
    )
    program = WarpgroupProgram(
        PROGRAM_FORMAT,
        2,
        types,
        inputs,
        ProgramLoop("%i", 2, iter_args, program_ops),
    )
    problem = WarpgroupProblem(
        PROBLEM_FORMAT,
        "cycle",
        2,
        (ResourceCapacity("copy", 1),),
        types,
        inputs,
        ProblemLoop("%i", 2, iter_args, problem_ops),
    )
    schedule = WarpgroupSchedule(
        SCHEDULE_FORMAT,
        (
            WarpgroupLane(("load", "independent")),
            WarpgroupLane(("advance",)),
        ),
        (SynchronizationEdge("load", "advance", 0),),
        (
            TimedOperation(0, "load", 0, 1),
            TimedOperation(0, "independent", 1, 2),
            TimedOperation(0, "advance", 1, 2),
            TimedOperation(1, "load", 2, 3),
            TimedOperation(1, "independent", 3, 4),
            TimedOperation(1, "advance", 3, 4),
        ),
    )
    return program, problem, schedule


def _canonical_document(value: object, encode: Callable[[object], str]) -> dict[str, object]:
    document = json.loads(encode(value))
    assert isinstance(document, dict)
    return document


def _signatures(program: WarpgroupProgram) -> tuple[OperationSignature, ...]:
    return tuple(operation_signature(program, item) for item in program.loop.ops)


def _library(program: WarpgroupProgram) -> OperationCostLibrary:
    entries = tuple(
        OperationCostEntry(
            signature,
            OperationCost(
                index + 1,
                (ResourceDemand("engine", 1),) if signature.kind is OperationKind.COPY else (),
            ),
        )
        for index, signature in enumerate(
            sorted(set(_signatures(program)), key=lambda item: item.canonical_key)
        )
    )
    return OperationCostLibrary("cycle", (ResourceCapacity("engine", 1),), entries)


def test_m5_b200_coverage_is_exact_and_has_no_fallback_measurement() -> None:
    signature = b200_global_to_shared_copy_signature()
    assert (
        signature.canonical_key
        == (
            ROOT / "costmodel" / "benchmarks" / "warpgroup" / "global-to-shared-copy.signature.json"
        )
        .read_text(encoding="utf-8")
        .strip()
    )
    matrix = b200_warpgroup_coverage_matrix((signature,))
    ready = tuple(
        item for item in matrix.entries if item.status is CalibrationStatus.PROVIDER_READY
    )
    assert len(ready) == 1
    assert not any(hasattr(item, "duration") for item in matrix.entries)
    entry = ready[0]
    assert entry.family is B200OperationFamily.GLOBAL_TO_SHARED_COPY
    assert entry.signature.kind is OperationKind.COPY
    assert entry.signature.operands[0].shape == (64, 64)
    assert entry.signature.operands[0].dtype is DType.BF16
    assert entry.signature.operands[0].space is MemorySpace.GLOBAL
    assert entry.signature.outputs[0].type.space is MemorySpace.SHARED
    assert {item.name for item in entry.conditions} == {
        "cuda_arch",
        "hardware",
        "pipeline_depth",
    }
    assert {item.resource_id for item in entry.resources} == {
        "b200.cuda_core",
        "b200.gmem_read",
        "b200.smem_write",
        "b200.warp_issue",
    }
    for entry in matrix.entries:
        with pytest.raises(B200CalibrationMissingError, match="missing correctness-checked"):
            matrix.require_measured(entry.signature)
    with pytest.raises(B200CalibrationMissingError, match="missing correctness-checked"):
        matrix.lookup(dataclasses.replace(signature, kind=OperationKind.COMPUTE))


def _rename_program(program: WarpgroupProgram) -> WarpgroupProgram:
    document = _canonical_document(program, warpgroup_program_to_json)
    replacements = {
        "%external": "%source",
        "%i": "%step",
        "%independent": "%aside",
        "%next": "%updated",
        "%state": "%carry",
        "%tile": "%staged",
        "advance": "combine",
        "external": "source_type",
        "independent": "aside_op",
        "load": "stage_op",
        "state": "register_type",
        "tile": "shared_type",
    }

    def rename(value: object) -> object:
        if type(value) is str:
            return replacements.get(value, value)
        if type(value) is list:
            return [rename(item) for item in value]
        if type(value) is dict:
            return {replacements.get(key, key): rename(item) for key, item in value.items()}
        return value

    return warpgroup_program_from_json(json.dumps(rename(document)))


def test_generic_python_and_json_share_one_canonical_immutable_boundary() -> None:
    program, problem, schedule = _generic_values()
    codecs = (
        (program, warpgroup_program_to_json, warpgroup_program_from_json),
        (problem, warpgroup_problem_to_json, warpgroup_problem_from_json),
        (schedule, warpgroup_schedule_to_json, warpgroup_schedule_from_json),
    )

    for value, encode, decode in codecs:
        canonical = encode(value)
        decoded = decode(canonical)
        assert decoded == value
        assert encode(decoded) == canonical
        assert json.dumps(json.loads(canonical), sort_keys=True, separators=(",", ":")) == canonical

    assert (
        program.dependencies()
        == problem.dependencies()
        == (
            DefUseDependency("advance", "advance", 1),
            DefUseDependency("load", "advance", 0),
        )
    )
    assert "independent" not in {
        endpoint for edge in program.dependencies() for endpoint in (edge.after, edge.before)
    }
    nested = (
        program.types[0].shape,
        program.loop.ops[0].outputs,
        program.loop.ops[0].outputs[0].expression,
        problem.resources,
        problem.loop.ops[0].resources,
        schedule.lanes,
        schedule.sync,
        schedule.times,
    )
    assert not any(isinstance(item, (dict, list)) for item in nested)
    for value in (program, program.loop, problem, schedule):
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError, TypeError)):
            value.format = "changed"  # type: ignore[attr-defined]
    load = next(item for item in program.loop.ops if item.id == "load")
    with pytest.raises((dataclasses.FrozenInstanceError, TypeError)):
        load.outputs[0].expression.source = ValueRef("%changed")  # type: ignore[misc,union-attr]
    with pytest.raises(TypeError):
        program.types[0].shape[0] = 8  # type: ignore[index]
    with pytest.raises(TypeError):
        problem.loop.ops[0].resources[0] = ResourceDemand("copy", 1)  # type: ignore[index]
    with pytest.raises(TypeError):
        schedule.lanes[0].operations[0] = "changed"  # type: ignore[index]
    with pytest.raises(dataclasses.FrozenInstanceError):
        schedule.sync[0].distance = 1  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        schedule.times[0].start = 1  # type: ignore[misc]


def test_generic_input_and_operation_order_have_no_semantic_meaning() -> None:
    program, _, _ = _generic_values()
    original = _canonical_document(program, warpgroup_program_to_json)
    reordered = deepcopy(original)
    reordered["inputs"] = list(reversed(reordered["inputs"]))
    reordered["loop"]["ops"] = list(reversed(reordered["loop"]["ops"]))

    decoded = warpgroup_program_from_json(json.dumps(reordered))
    assert decoded == program
    assert decoded.dependencies() == program.dependencies()
    assert warpgroup_program_to_json(decoded) == warpgroup_program_to_json(program)


@pytest.mark.parametrize(
    ("value_index", "schema_name", "encode", "decode"),
    (
        (
            0,
            "warpgroup-program-v1.schema.json",
            warpgroup_program_to_json,
            warpgroup_program_from_json,
        ),
        (
            1,
            "warpgroup-problem-v1.schema.json",
            warpgroup_problem_to_json,
            warpgroup_problem_from_json,
        ),
        (
            2,
            "warpgroup-schedule-v1.schema.json",
            warpgroup_schedule_to_json,
            warpgroup_schedule_from_json,
        ),
    ),
)
def test_schema_and_decoder_agree_on_structural_documents(
    value_index: int,
    schema_name: str,
    encode: Callable[[object], str],
    decode: Callable[[str], object],
) -> None:
    value = _generic_values()[value_index]
    document = _canonical_document(value, encode)
    validator = _validator(schema_name)
    validator.validate(document)
    assert decode(json.dumps(document)) == value

    unknown = deepcopy(document)
    unknown["unknown"] = True
    assert list(validator.iter_errors(unknown))
    with pytest.raises(WarpgroupSerializationError, match="unknown field"):
        decode(json.dumps(unknown))

    invalid_id = deepcopy(document)
    if value_index == 0:
        invalid_id["loop"]["ops"][0]["id"] += "\n"
    elif value_index == 1:
        invalid_id["time_unit"] += "\n"
    else:
        invalid_id["lanes"][0][0] += "\n"
    assert list(validator.iter_errors(invalid_id))
    with pytest.raises(WarpgroupSerializationError, match="ASCII identifier"):
        decode(json.dumps(invalid_id))


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (
            lambda value: value["lanes"][1].append(value["lanes"][0][0]),
            "more than one lane",
        ),
        (lambda value: value["times"][0].__setitem__(2, 2), "greater than start"),
        (lambda value: value["times"].append(deepcopy(value["times"][0])), "duplicate timed"),
    ),
)
def test_schedule_cross_record_semantics_belong_to_decoder_model(
    mutation: Callable[[dict[str, object]], object], message: str
) -> None:
    document = _canonical_document(_generic_values()[2], warpgroup_schedule_to_json)
    mutation(document)

    _validator("warpgroup-schedule-v1.schema.json").validate(document)
    with pytest.raises(WarpgroupSerializationError, match=message):
        warpgroup_schedule_from_json(json.dumps(document))


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (
            lambda value: value["loop"]["ops"][0]["outputs"][0].update(
                {"expr": ["mystery", "%external"]}
            ),
            "unknown expression operator",
        ),
        (
            lambda value: value["loop"]["ops"].append(deepcopy(value["loop"]["ops"][0])),
            "duplicate operation",
        ),
        (
            lambda value: value["loop"]["ops"][1]["outputs"][0].update({"id": "%tile"}),
            "duplicate SSA",
        ),
        (
            lambda value: value["loop"]["ops"][0]["outputs"][0].update(
                {"expr": ["copy", "%missing"]}
            ),
            "undefined SSA",
        ),
        (lambda value: value["loop"]["iter_args"][0].update({"yield": "%missing"}), "yield"),
        (
            lambda value: value["loop"]["iter_args"][0].update({"init": "%missing"}),
            "external input",
        ),
        (
            lambda value: value["loop"]["ops"][0]["outputs"][0].update(
                {"expr": ["copy", ["index", "%external", 2]]}
            ),
            "out of bounds",
        ),
        (lambda value: value["types"]["tile"].update({"shape": [5]}), "shape"),
        (lambda value: value["types"]["tile"].update({"dtype": "i32"}), "dtype"),
        (
            lambda value: value["types"]["state"].update({"shape": [3]}),
            "singleton broadcast",
        ),
        (lambda value: value["types"]["tile"].update({"space": "texture"}), "memory space"),
    ),
)
def test_generic_program_rejects_externally_reachable_static_errors(
    mutation: Callable[[dict[str, object]], object], message: str
) -> None:
    document = _canonical_document(_generic_values()[0], warpgroup_program_to_json)
    mutation(document)
    with pytest.raises(WarpgroupSerializationError, match=message):
        warpgroup_program_from_json(json.dumps(document))


def test_computation_is_register_local_and_shared_publication_is_explicit_copy() -> None:
    document = _canonical_document(_generic_values()[0], warpgroup_program_to_json)
    advance = next(item for item in document["loop"]["ops"] if item["id"] == "advance")
    advance["outputs"][0]["type"] = "tile"
    with pytest.raises(WarpgroupSerializationError, match="computed expressions.*register"):
        warpgroup_program_from_json(json.dumps(document))

    document = _canonical_document(_generic_values()[0], warpgroup_program_to_json)
    document["loop"]["ops"].append(
        {
            "id": "publish",
            "outputs": [{"id": "%published", "type": "tile", "expr": ["copy", "%next"]}],
        }
    )
    assert warpgroup_program_from_json(json.dumps(document))


def test_problem_rejects_invalid_numeric_facts_before_solving() -> None:
    problem = _generic_values()[1]

    document = _canonical_document(problem, warpgroup_problem_to_json)
    document["loop"]["ops"][0]["duration"] = 0
    with pytest.raises(WarpgroupSerializationError, match="positive integer"):
        warpgroup_problem_from_json(json.dumps(document))

    document = _canonical_document(problem, warpgroup_problem_to_json)
    document["loop"]["ops"][0]["resources"] = {"missing": 1}
    with pytest.raises(WarpgroupSerializationError, match="undefined resource"):
        warpgroup_problem_from_json(json.dumps(document))

    document = _canonical_document(problem, warpgroup_problem_to_json)
    document["loop"]["ops"][0]["resources"] = {"copy": 2}
    with pytest.raises(WarpgroupSerializationError, match="exceeds capacity"):
        warpgroup_problem_from_json(json.dumps(document))


def test_serializers_require_exact_typed_roots_and_nested_records() -> None:
    program, problem, schedule = _generic_values()
    with pytest.raises(WarpgroupSerializationError, match="exact WarpgroupProgram"):
        warpgroup_program_to_json({"format": PROGRAM_FORMAT})  # type: ignore[arg-type]
    with pytest.raises(WarpgroupSerializationError, match="exact WarpgroupProblem"):
        warpgroup_problem_to_json({"format": PROBLEM_FORMAT})  # type: ignore[arg-type]
    with pytest.raises(WarpgroupSerializationError, match="exact WarpgroupSchedule"):
        warpgroup_schedule_to_json({"format": SCHEDULE_FORMAT})  # type: ignore[arg-type]

    object.__setattr__(program, "types", ({"shape": [4]},))
    with pytest.raises(WarpgroupSerializationError, match="exact TensorType"):
        warpgroup_program_to_json(program)
    object.__setattr__(problem.loop.ops[0], "resources", ({"copy": 1},))
    with pytest.raises(WarpgroupSerializationError, match="exact ResourceDemand"):
        warpgroup_problem_to_json(problem)
    object.__setattr__(schedule, "sync", ({"after": "load"},))
    with pytest.raises(WarpgroupSerializationError, match="exact SynchronizationEdge"):
        warpgroup_schedule_to_json(schedule)


def test_typed_boundary_import_has_no_solver_cuda_costmodel_or_target_side_effect() -> None:
    code = """
import sys
from tilefoundry.schedule.warpgroup import WarpgroupProgram
forbidden = ('ortools', 'cuda', 'tilefoundry_costmodel', 'tilefoundry.target.amx', 'tilefoundry.target.cuda')
assert WarpgroupProgram.__name__ == 'WarpgroupProgram'
assert not [name for name in sys.modules if any(name == root or name.startswith(root + '.') for root in forbidden)]
"""
    subprocess.run([sys.executable, "-c", code], check=True, cwd=ROOT)


def test_m1_build_closes_generic_program_and_problem_replays_without_library() -> None:
    program = _generic_values()[0]
    library = _library(program)

    problem = build_warpgroup_problem(program, library)

    assert problem.time_unit == "cycle"
    assert problem.dependencies() == program.dependencies()
    assert not any(field.name == "cost_library" for field in dataclasses.fields(problem))
    costs = {entry.signature: entry.cost for entry in library.entries}
    assert all(
        (operation.duration, operation.resources)
        == (
            costs[operation_signature(program, source)].duration,
            costs[operation_signature(program, source)].resources,
        )
        for operation, source in zip(problem.loop.ops, program.loop.ops, strict=True)
    )

    payload = warpgroup_problem_to_json(problem)
    code = f"""
import sys
from tilefoundry.schedule.warpgroup import warpgroup_problem_from_json
problem = warpgroup_problem_from_json({payload!r})
assert problem.time_unit == 'cycle'
forbidden = ('ortools', 'cuda', 'tilefoundry_costmodel', 'tilefoundry.target.amx', 'tilefoundry.target.cuda')
assert not [name for name in sys.modules if any(name == root or name.startswith(root + '.') for root in forbidden)]
"""
    subprocess.run([sys.executable, "-c", code], check=True, cwd=ROOT)


def test_m1_signature_erases_operation_ssa_loop_index_and_type_alias_names() -> None:
    program = _generic_values()[0]
    renamed = _rename_program(program)

    assert sorted(item.canonical_key for item in _signatures(renamed)) == sorted(
        item.canonical_key for item in _signatures(program)
    )
    assert build_warpgroup_problem(renamed, _library(program))


def _operation_document(document: dict[str, object], operation_id: str) -> dict[str, object]:
    return next(item for item in document["loop"]["ops"] if item["id"] == operation_id)


def test_m1_signature_changes_with_cost_relevant_semantics() -> None:
    program = _generic_values()[0]
    baseline = {item.id: operation_signature(program, item) for item in program.loop.ops}

    mutations: tuple[tuple[str, Callable[[dict[str, object]], None]], ...] = (
        (
            "advance",
            lambda value: _operation_document(value, "advance")["outputs"][0].update(
                {"expr": ["mul", "%state", "%tile"]}
            ),
        ),
        (
            "independent",
            lambda value: _operation_document(value, "independent")["outputs"][0].update(
                {"expr": 2.0}
            ),
        ),
        (
            "load",
            lambda value: value["types"].update(
                {
                    "external": {"shape": [2, 5], "dtype": "fp32", "space": "global"},
                    "tile": {"shape": [5], "dtype": "fp32", "space": "shared"},
                    "state": {"shape": [5], "dtype": "fp32", "space": "register"},
                }
            ),
        ),
        (
            "load",
            lambda value: [item.update({"dtype": "fp16"}) for item in value["types"].values()],
        ),
        (
            "load",
            lambda value: value["types"]["tile"].update({"space": "register"}),
        ),
        (
            "independent",
            lambda value: _operation_document(value, "independent")["outputs"].append(
                {"id": "%second", "type": "state", "expr": 2.0}
            ),
        ),
    )
    for operation_id, mutate in mutations:
        document = _canonical_document(program, warpgroup_program_to_json)
        mutate(document)
        changed = warpgroup_program_from_json(json.dumps(document))
        operation = next(item for item in changed.loop.ops if item.id == operation_id)
        assert operation_signature(changed, operation) != baseline[operation_id]


def test_m1_signature_preserves_operand_aliasing_and_all_atomic_outputs() -> None:
    document = _canonical_document(_generic_values()[0], warpgroup_program_to_json)
    operation = _operation_document(document, "advance")
    operation["outputs"].append(
        {"id": "%also_next", "type": "state", "expr": ["add", "%state", "%state"]}
    )
    program = warpgroup_program_from_json(json.dumps(document))
    signature = operation_signature(
        program, next(item for item in program.loop.ops if item.id == "advance")
    )

    assert len(signature.outputs) == 2
    assert len(signature.operands) == 2
    assert signature.outputs[0].expression != signature.outputs[1].expression


def _single_operation_program(
    expression: object,
    *,
    inputs: list[dict[str, str]],
    types: dict[str, dict[str, object]],
    output_type: str,
) -> WarpgroupProgram:
    return warpgroup_program_from_json(
        json.dumps(
            {
                "format": PROGRAM_FORMAT,
                "warp_groups": 1,
                "types": types,
                "inputs": inputs,
                "loop": {
                    "index": "%iteration",
                    "iterations": 1,
                    "iter_args": [],
                    "ops": [
                        {
                            "id": "operation",
                            "outputs": [{"id": "%result", "type": output_type, "expr": expression}],
                        }
                    ],
                },
            }
        )
    )


def test_m1_axis_and_work_class_are_explicit_signature_semantics() -> None:
    reduce0 = _single_operation_program(
        ["reduce", "sum", 0, "%source"],
        inputs=[{"id": "%source", "type": "source"}],
        types={
            "source": {"shape": [1, 1], "dtype": "fp32", "space": "register"},
            "result": {"shape": [1, 1], "dtype": "fp32", "space": "register"},
        },
        output_type="result",
    )
    reduce1 = _single_operation_program(
        ["reduce", "sum", 1, "%source"],
        inputs=[{"id": "%source", "type": "source"}],
        types={
            "source": {"shape": [1, 1], "dtype": "fp32", "space": "register"},
            "result": {"shape": [1, 1], "dtype": "fp32", "space": "register"},
        },
        output_type="result",
    )
    transpose = _single_operation_program(
        ["transpose", "%source"],
        inputs=[{"id": "%source", "type": "source"}],
        types={
            "source": {"shape": [2, 3], "dtype": "fp32", "space": "register"},
            "result": {"shape": [3, 2], "dtype": "fp32", "space": "register"},
        },
        output_type="result",
    )
    concat = _single_operation_program(
        ["concat", 0, "%source", "%source"],
        inputs=[{"id": "%source", "type": "source"}],
        types={
            "source": {"shape": [1], "dtype": "fp32", "space": "register"},
            "result": {"shape": [2], "dtype": "fp32", "space": "register"},
        },
        output_type="result",
    )
    cast = _single_operation_program(
        ["cast", "%source"],
        inputs=[{"id": "%source", "type": "source"}],
        types={
            "source": {"shape": [1], "dtype": "fp32", "space": "register"},
            "result": {"shape": [1], "dtype": "fp16", "space": "register"},
        },
        output_type="result",
    )

    reduce0_signature = _signatures(reduce0)[0]
    assert reduce0_signature != _signatures(reduce1)[0]
    assert reduce0_signature.kind is OperationKind.COMPUTE
    assert _signatures(transpose)[0].kind is OperationKind.VIEW
    assert _signatures(concat)[0].kind is OperationKind.VIEW
    assert _signatures(cast)[0].kind is OperationKind.COMPUTE
    assert (
        next(
            operation_signature(_generic_values()[0], item)
            for item in _generic_values()[0].loop.ops
            if item.id == "load"
        ).kind
        is OperationKind.COPY
    )


def test_m1_matmul_result_dtype_is_declared_without_implicit_fp32() -> None:
    def program(dtype: str) -> WarpgroupProgram:
        return _single_operation_program(
            ["matmul", "%lhs", "%rhs"],
            inputs=[{"id": "%lhs", "type": "lhs"}, {"id": "%rhs", "type": "rhs"}],
            types={
                "lhs": {"shape": [2, 3], "dtype": "fp16", "space": "register"},
                "rhs": {"shape": [3, 4], "dtype": "fp16", "space": "register"},
                "result": {"shape": [2, 4], "dtype": dtype, "space": "register"},
            },
            output_type="result",
        )

    fp16 = program("fp16")
    fp32 = program("fp32")
    assert _signatures(fp16)[0] != _signatures(fp32)[0]
    with pytest.raises(WarpgroupSerializationError, match="floating result dtype"):
        program("i32")


class _TrackingLibrary:
    def __init__(
        self,
        costs: dict[OperationSignature, OperationCost],
    ) -> None:
        self.time_unit = "cycle"
        self.resources: tuple[ResourceCapacity, ...] = ()
        self.costs = costs
        self.calls: list[OperationSignature] = []

    def lookup(self, signature: OperationSignature) -> OperationCost:
        self.calls.append(signature)
        try:
            return self.costs[signature]
        except KeyError:
            raise WarpgroupCostMissingError(signature) from None


def test_m1_build_queries_each_unique_signature_once() -> None:
    document = _canonical_document(_generic_values()[0], warpgroup_program_to_json)
    document["loop"]["ops"].append(
        {
            "id": "second_independent",
            "outputs": [{"id": "%second", "type": "state", "expr": 1.0}],
        }
    )
    program = warpgroup_program_from_json(json.dumps(document))
    costs = {signature: OperationCost(1, ()) for signature in _signatures(program)}
    library = _TrackingLibrary(costs)

    problem = build_warpgroup_problem(program, library)

    assert len(library.calls) == len(set(_signatures(program)))
    assert len(problem.loop.ops) == len(program.loop.ops)


def test_m1_build_aggregates_missing_signatures_without_partial_problem() -> None:
    program = _generic_values()[0]
    signatures = set(_signatures(program))
    library = _TrackingLibrary({})

    with pytest.raises(WarpgroupMissingSignaturesError) as failure:
        build_warpgroup_problem(program, library)

    assert set(failure.value.signatures) == signatures
    assert set(library.calls) == signatures
    assert len(library.calls) == len(signatures)


def test_m1_exact_library_rejects_ambiguous_entries() -> None:
    signature = _signatures(_generic_values()[0])[0]
    entry = OperationCostEntry(signature, OperationCost(1, ()))
    with pytest.raises(WarpgroupCostAmbiguityError, match="duplicate signatures"):
        OperationCostLibrary("cycle", (), (entry, entry))


def _independent_solve_problem(
    *, warp_groups: int, iterations: int = 1, capacity: int | None = None
) -> WarpgroupProblem:
    value_type = TensorType("value", (1,), DType.FP32, MemorySpace.REGISTER)
    resources = () if capacity is None else (ResourceCapacity("engine", capacity),)
    demands = () if capacity is None else (ResourceDemand("engine", 1),)
    operations = tuple(
        ProblemOperation(
            operation_id,
            (OperationOutput(f"%{operation_id}", "value", ScalarLiteral(1.0)),),
            2,
            demands,
        )
        for operation_id in ("alpha", "beta")
    )
    return WarpgroupProblem(
        PROBLEM_FORMAT,
        "cycle",
        warp_groups,
        resources,
        (value_type,),
        (),
        ProblemLoop("%iteration", iterations, (), operations),
    )


def _time_map(result: WarpgroupSolveResult) -> dict[tuple[int, str], TimedOperation]:
    return {(item.iteration, item.operation_id): item for item in result.times}


def _lane_map(result: WarpgroupSolveResult) -> dict[str, int]:
    return {
        operation_id: lane
        for lane, program in enumerate(result.lanes)
        for operation_id in program.operations
    }


def test_m2_independent_work_overlaps_on_two_lanes_and_serializes_on_one() -> None:
    parallel = solve_warpgroup_problem(_independent_solve_problem(warp_groups=2))
    serial = solve_warpgroup_problem(_independent_solve_problem(warp_groups=1))

    parallel_times = _time_map(parallel)
    assert parallel.status == serial.status == "OPTIMAL"
    assert parallel.makespan == 2
    assert parallel_times[(0, "alpha")].start == parallel_times[(0, "beta")].start == 0
    assert len(set(_lane_map(parallel).values())) == 2
    assert serial.makespan == 4


def test_m2_register_locality_and_shared_handoff_have_distinct_lane_rules() -> None:
    register_type = TensorType("register", (1,), DType.FP32, MemorySpace.REGISTER)
    register_problem = WarpgroupProblem(
        PROBLEM_FORMAT,
        "cycle",
        2,
        (),
        (register_type,),
        (),
        ProblemLoop(
            "%iteration",
            1,
            (),
            (
                ProblemOperation(
                    "define",
                    (
                        OperationOutput("%value", "register", ScalarLiteral(1.0)),
                        OperationOutput("%define_token", "register", ScalarLiteral(0.0)),
                    ),
                    2,
                    (),
                ),
                ProblemOperation(
                    "use",
                    (
                        OperationOutput(
                            "%use_token", "register", CopyExpression(ValueRef("%value"))
                        ),
                    ),
                    3,
                    (),
                ),
                ProblemOperation(
                    "define_tail",
                    (
                        OperationOutput(
                            "%define_done",
                            "register",
                            CopyExpression(ValueRef("%define_token")),
                        ),
                    ),
                    4,
                    (),
                ),
                ProblemOperation(
                    "use_tail",
                    (
                        OperationOutput(
                            "%use_done",
                            "register",
                            CopyExpression(ValueRef("%use_token")),
                        ),
                    ),
                    4,
                    (),
                ),
            ),
        ),
    )
    register_result = solve_warpgroup_problem(register_problem)
    assert len(set(_lane_map(register_result).values())) == 1
    assert register_result.makespan == 13

    types = (
        TensorType("global", (1,), DType.FP32, MemorySpace.GLOBAL),
        TensorType("shared", (1,), DType.FP32, MemorySpace.SHARED),
        register_type,
    )
    shared_problem = WarpgroupProblem(
        PROBLEM_FORMAT,
        "cycle",
        2,
        (),
        types,
        (ProgramInput("%source", "global"),),
        ProblemLoop(
            "%iteration",
            1,
            (),
            (
                ProblemOperation(
                    "publish",
                    (
                        OperationOutput("%shared", "shared", CopyExpression(ValueRef("%source"))),
                        OperationOutput("%publish_token", "register", ScalarLiteral(0.0)),
                    ),
                    2,
                    (),
                ),
                ProblemOperation(
                    "publish_tail",
                    (
                        OperationOutput(
                            "%publish_done",
                            "register",
                            CopyExpression(ValueRef("%publish_token")),
                        ),
                    ),
                    4,
                    (),
                ),
                ProblemOperation(
                    "consume",
                    (
                        OperationOutput(
                            "%consume_token",
                            "register",
                            CopyExpression(ValueRef("%shared")),
                        ),
                    ),
                    3,
                    (),
                ),
                ProblemOperation(
                    "consume_tail",
                    (
                        OperationOutput(
                            "%consume_done",
                            "register",
                            CopyExpression(ValueRef("%consume_token")),
                        ),
                    ),
                    4,
                    (),
                ),
            ),
        ),
    )
    shared_result = solve_warpgroup_problem(shared_problem)
    shared_lanes = _lane_map(shared_result)
    shared_times = _time_map(shared_result)
    assert shared_lanes["publish"] != shared_lanes["consume"]
    assert shared_times[(0, "publish")].end <= shared_times[(0, "consume")].start
    assert shared_result.makespan == 9


def test_m2_lane_assignment_and_body_order_are_stable_across_iterations() -> None:
    result = solve_warpgroup_problem(_independent_solve_problem(warp_groups=1, iterations=3))
    times = _time_map(result)
    lane = result.lanes[0].operations

    assert set(lane) == {"alpha", "beta"}
    for iteration in range(3):
        assert times[(iteration, lane[0])].end <= times[(iteration, lane[1])].start
    for iteration in range(2):
        assert times[(iteration, lane[-1])].end <= times[(iteration + 1, lane[0])].start


def test_m2_cumulative_capacity_controls_legal_overlap() -> None:
    serial = solve_warpgroup_problem(_independent_solve_problem(warp_groups=2, capacity=1))
    parallel = solve_warpgroup_problem(_independent_solve_problem(warp_groups=2, capacity=2))

    assert serial.makespan == 4
    assert parallel.makespan == 2


def test_m2_shared_allocation_reuse_waits_for_every_previous_use() -> None:
    types = (
        TensorType("source", (2, 1), DType.FP32, MemorySpace.GLOBAL),
        TensorType("shared", (1,), DType.FP32, MemorySpace.SHARED),
        TensorType("result", (1,), DType.FP32, MemorySpace.REGISTER),
    )
    problem = WarpgroupProblem(
        PROBLEM_FORMAT,
        "cycle",
        2,
        (),
        types,
        (ProgramInput("%source", "source"),),
        ProblemLoop(
            "%iteration",
            2,
            (),
            (
                ProblemOperation(
                    "publish",
                    (
                        OperationOutput(
                            "%shared",
                            "shared",
                            CopyExpression(
                                IndexExpression(ValueRef("%source"), (LoopIndexRef("%iteration"),))
                            ),
                        ),
                    ),
                    1,
                    (),
                ),
                ProblemOperation(
                    "consume",
                    (OperationOutput("%result", "result", CopyExpression(ValueRef("%shared"))),),
                    4,
                    (),
                ),
            ),
        ),
    )
    result = solve_warpgroup_problem(problem)
    times = _time_map(result)

    assert times[(0, "consume")].end <= times[(1, "publish")].start
    assert result.makespan == 10


def test_m2_m3_loop_carried_shared_init_and_reuse_boundary() -> None:
    """M2/M3 lacked a boundary test separating external init from body reuse."""
    types = (
        TensorType("source", (2, 1), DType.FP32, MemorySpace.GLOBAL),
        TensorType("shared", (1,), DType.FP32, MemorySpace.SHARED),
        TensorType("result", (1,), DType.FP32, MemorySpace.REGISTER),
    )
    problem = WarpgroupProblem(
        PROBLEM_FORMAT,
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
                                IndexExpression(ValueRef("%source"), (LoopIndexRef("%iteration"),))
                            ),
                        ),
                    ),
                    1,
                    (),
                ),
                ProblemOperation(
                    "consume",
                    (OperationOutput("%result", "result", CopyExpression(ValueRef("%carry"))),),
                    4,
                    (),
                ),
            ),
        ),
    )
    result = solve_warpgroup_problem(problem)
    times = _time_map(result)
    consume_zero = times[(0, "consume")]
    publish_zero = times[(0, "publish")]

    assert max(consume_zero.start, publish_zero.start) < min(consume_zero.end, publish_zero.end)
    assert publish_zero.end <= times[(1, "consume")].start
    assert times[(1, "consume")].end <= times[(1, "publish")].start
    assert result.makespan == 9

    schedule = export_warpgroup_schedule(problem, result)
    verify_warpgroup_schedule(problem, schedule)
    assert SynchronizationEdge("publish", "consume", 1) in schedule.sync
    assert SynchronizationEdge("consume", "publish", 0) not in schedule.sync

    unsafe_times = tuple(
        TimedOperation(1, "publish", 4, 5)
        if (item.iteration, item.operation_id) == (1, "publish")
        else item
        for item in schedule.times
    )
    with pytest.raises(WarpgroupVerificationError, match="carried shared lifetime"):
        verify_warpgroup_schedule(
            problem,
            WarpgroupSchedule(SCHEDULE_FORMAT, schedule.lanes, schedule.sync, unsafe_times),
        )


def test_m2_repeated_deterministic_solves_are_identical() -> None:
    problem = _generic_values()[1]
    assert solve_warpgroup_problem(problem) == solve_warpgroup_problem(problem)


def _fixed_owner_program() -> WarpgroupProgram:
    program = _generic_values()[0]
    operations = tuple(
        ProgramOperation(operation.id, operation.outputs, 1 if operation.id == "independent" else 0)
        for operation in program.loop.ops
    )
    return WarpgroupProgram(
        PROGRAM_FORMAT_V2,
        3,
        program.types,
        program.inputs,
        ProgramLoop(
            program.loop.index, program.loop.iterations, program.loop.iter_args, operations
        ),
    )


def test_m6_fixed_owner_build_round_trip_keeps_empty_group() -> None:
    program = _fixed_owner_program()
    problem = build_warpgroup_problem(program, _library(program))

    assert problem.format == PROBLEM_FORMAT_V3
    assert {item.id: item.warp_group for item in problem.loop.ops} == {
        "advance": 0,
        "independent": 1,
        "load": 0,
    }
    encoded_program = warpgroup_program_to_json(program)
    encoded_problem = warpgroup_problem_to_json(problem)
    assert warpgroup_program_from_json(encoded_program) == program
    assert warpgroup_problem_from_json(encoded_problem) == problem
    _validator("warpgroup-program-v2.schema.json").validate(json.loads(encoded_program))
    _validator("warpgroup-problem-v3.schema.json").validate(json.loads(encoded_problem))

    result = schedule_warpgroups(problem)
    assert result.schedule.format == SCHEDULE_FORMAT_V3
    assert len(result.schedule.lanes) == 3
    assert result.schedule.lanes[2].operations == ()
    assert set(result.schedule.lanes[0].operations) == {"load", "advance"}
    assert result.schedule.lanes[1].operations == ("independent",)
    verify_warpgroup_schedule(problem, result.schedule)
    encoded_schedule = warpgroup_schedule_to_json(result.schedule)
    assert warpgroup_schedule_from_json(encoded_schedule) == result.schedule
    _validator("warpgroup-schedule-v3.schema.json").validate(json.loads(encoded_schedule))
    assert result == schedule_warpgroups(problem)


def test_m6_fixed_owner_searches_same_group_order_without_migration() -> None:
    value_type = TensorType("value", (1,), DType.FP32, MemorySpace.REGISTER)
    problem = WarpgroupProblem(
        PROBLEM_FORMAT_V3,
        "cycle",
        3,
        (),
        (value_type,),
        (),
        ProblemLoop(
            "%iteration",
            1,
            (),
            (
                ProblemOperation(
                    "short",
                    (OperationOutput("%short", "value", ScalarLiteral(1.0)),),
                    2,
                    (),
                    warp_group=0,
                ),
                ProblemOperation(
                    "long",
                    (OperationOutput("%long", "value", ScalarLiteral(1.0)),),
                    3,
                    (),
                    warp_group=0,
                ),
                ProblemOperation(
                    "other",
                    (OperationOutput("%other", "value", ScalarLiteral(1.0)),),
                    1,
                    (),
                    warp_group=1,
                ),
            ),
        ),
    )
    result = solve_warpgroup_problem(problem)
    assert result.makespan == 5
    assert result.lanes[0].operations in (("short", "long"), ("long", "short"))
    assert set(result.lanes[0].operations) == {"short", "long"}
    assert result.lanes[1].operations == ("other",)
    assert result.lanes[2].operations == ()


def test_m6_fixed_owner_verifier_rejects_cross_group_move() -> None:
    problem = build_warpgroup_problem(_fixed_owner_program(), _library(_fixed_owner_program()))
    schedule = schedule_warpgroups(problem).schedule
    moved = tuple(
        WarpgroupLane(("independent", "load"))
        if index == 1
        else WarpgroupLane(tuple(operation for operation in lane.operations if operation != "load"))
        if index == 0
        else lane
        for index, lane in enumerate(schedule.lanes)
    )
    moved_schedule = WarpgroupSchedule(SCHEDULE_FORMAT_V3, moved, schedule.sync, schedule.times)
    with pytest.raises(WarpgroupVerificationError, match="expected warp_group"):
        verify_warpgroup_schedule(problem, moved_schedule)


@pytest.mark.parametrize(
    ("problem_version", "schedule_version", "valid"),
    (
        ("v1", "v1", True),
        ("v2", "v2", True),
        ("v3", "v3", True),
        ("v1", "v3", False),
        ("v2", "v3", False),
        ("v3", "v1", False),
        ("v3", "v2", False),
    ),
)
def test_m6_problem_schedule_formats_are_strictly_paired(
    problem_version: str, schedule_version: str, valid: bool
) -> None:
    generic_problem = _generic_values()[1]
    async_program = _async_program()
    async_problem = build_warpgroup_problem(
        async_program, _async_library(async_program, resource_windows=False)
    )
    fixed_program = _fixed_owner_program()
    fixed_problem = build_warpgroup_problem(fixed_program, _library(fixed_program))
    problems = {"v1": generic_problem, "v2": async_problem, "v3": fixed_problem}
    schedules = {
        "v1": schedule_warpgroups(generic_problem).schedule,
        "v2": schedule_warpgroups(async_problem).schedule,
        "v3": schedule_warpgroups(fixed_problem).schedule,
    }
    problem = problems[problem_version]
    schedule = schedules[schedule_version]
    if valid:
        verify_warpgroup_schedule(problem, schedule)
    else:
        with pytest.raises(WarpgroupVerificationError, match="requires schedule"):
            verify_warpgroup_schedule(problem, schedule)


@pytest.mark.parametrize(
    "invalid_expression",
    (
        ["copy", "%x", "%y"],
        ["sub", "%x"],
        ["reduce", "invalid", 0, "%x"],
        ["select", "%condition", "%x"],
    ),
)
def test_m6_new_schema_and_decoder_reject_invalid_expression_arity(
    invalid_expression: list[object],
) -> None:
    program = _fixed_owner_program()
    program_document = json.loads(warpgroup_program_to_json(program))
    problem = build_warpgroup_problem(program, _library(program))
    problem_document = json.loads(warpgroup_problem_to_json(problem))
    for document, schema_name, decoder in (
        (program_document, "warpgroup-program-v2.schema.json", warpgroup_program_from_json),
        (problem_document, "warpgroup-problem-v3.schema.json", warpgroup_problem_from_json),
    ):
        document["loop"]["ops"][0]["outputs"][0]["expr"] = invalid_expression
        assert list(_validator(schema_name).iter_errors(document))
        with pytest.raises(WarpgroupSerializationError):
            decoder(json.dumps(document))


def _periodic_problem() -> WarpgroupProblem:
    value_type = TensorType("value", (1,), DType.FP32, MemorySpace.REGISTER)
    return WarpgroupProblem(
        PROBLEM_FORMAT_V3,
        "cycle",
        2,
        (),
        (value_type,),
        (),
        ProblemLoop(
            "%iteration",
            3,
            (),
            (
                ProblemOperation(
                    "a",
                    (OperationOutput("%a", "value", ScalarLiteral(1.0)),),
                    2,
                    (),
                    warp_group=0,
                ),
                ProblemOperation(
                    "b",
                    (OperationOutput("%b", "value", ScalarLiteral(1.0)),),
                    1,
                    (),
                    warp_group=0,
                ),
                ProblemOperation(
                    "c",
                    (OperationOutput("%c", "value", ScalarLiteral(1.0)),),
                    1,
                    (),
                    warp_group=1,
                ),
            ),
        ),
    )


def test_m6_1_fixed_owner_periodic_finite_solve_is_deterministic() -> None:
    problem = _periodic_problem()
    first = solve_warpgroup_problem(problem)
    second = solve_warpgroup_problem(problem)
    assert first == second
    schedule = export_warpgroup_schedule(problem, first)
    verify_warpgroup_schedule(problem, schedule)
    times = _time_map(first)
    initiation_intervals = {
        times[(iteration + 1, operation_id)].start - times[(iteration, operation_id)].start
        for operation_id in ("a", "b", "c")
        for iteration in range(2)
    }
    assert initiation_intervals == {3}
    assert first.makespan == 9
    assert set(first.lanes[0].operations) == {"a", "b"}
    assert first.lanes[1].operations == ("c",)
    assert warpgroup_schedule_to_json(schedule) == warpgroup_schedule_to_json(
        export_warpgroup_schedule(problem, second)
    )


def test_m6_1_verifier_rejects_one_iteration_period_shift() -> None:
    problem = _periodic_problem()
    schedule = export_warpgroup_schedule(problem, solve_warpgroup_problem(problem))
    shifted_times = tuple(
        TimedOperation(
            item.iteration,
            item.operation_id,
            item.start + 1,
            issue_end=item.issue_end + 1,
            completion=item.completion + 1,
        )
        if (item.iteration, item.operation_id) == (1, "c")
        else item
        for item in schedule.times
    )
    shifted = WarpgroupSchedule(SCHEDULE_FORMAT_V3, schedule.lanes, schedule.sync, shifted_times)
    with pytest.raises(WarpgroupVerificationError, match="periodic initiation interval"):
        verify_warpgroup_schedule(problem, shifted)


def _async_program() -> WarpgroupProgram:
    types = (
        TensorType("source", (2, 1), DType.FP32, MemorySpace.GLOBAL),
        TensorType("shared", (1,), DType.FP32, MemorySpace.SHARED),
        TensorType("value", (1,), DType.FP32, MemorySpace.REGISTER),
    )
    return WarpgroupProgram(
        PROGRAM_FORMAT,
        1,
        types,
        (ProgramInput("%source", "source"), ProgramInput("%initial", "shared")),
        ProgramLoop(
            "%iteration",
            2,
            (LoopIterArg("%carry", ValueRef("%initial"), ValueRef("%shared")),),
            (
                ProgramOperation(
                    "produce",
                    (OperationOutput("%produced", "value", ScalarLiteral(1.0)),),
                ),
                ProgramOperation(
                    "use",
                    (OperationOutput("%used", "value", CopyExpression(ValueRef("%produced"))),),
                ),
                ProgramOperation(
                    "publish",
                    (
                        OperationOutput(
                            "%shared",
                            "shared",
                            CopyExpression(
                                IndexExpression(ValueRef("%source"), (LoopIndexRef("%iteration"),))
                            ),
                        ),
                    ),
                ),
                ProgramOperation(
                    "consume",
                    (OperationOutput("%consumed", "value", CopyExpression(ValueRef("%carry"))),),
                ),
            ),
        ),
    )


def _async_library(program: WarpgroupProgram, *, resource_windows: bool) -> OperationCostLibrary:
    costs = {
        "produce": (2, 6),
        "use": (1, 1),
        "publish": (2, 6),
        "consume": (1, 3),
    }
    entries = tuple(
        OperationCostEntry(
            operation_signature(program, operation),
            OperationCost(
                issue_duration=costs[operation.id][0],
                completion_latency=costs[operation.id][1],
                resource_windows=(ResourceWindow("engine", 1, 2, 4),)
                if resource_windows and operation.id in {"produce", "publish"}
                else (),
            ),
        )
        for operation in program.loop.ops
    )
    return OperationCostLibrary(
        "cycle",
        (ResourceCapacity("engine", 1),) if resource_windows else (),
        entries,
    )


def test_m5_async_timing_resource_windows_and_v1_replay_share_one_workflow() -> None:
    program = _async_program()
    library = _async_library(program, resource_windows=False)
    problem = build_warpgroup_problem(program, library)
    assert problem.format == PROBLEM_FORMAT_V2
    assert warpgroup_problem_from_json(warpgroup_problem_to_json(problem)) == problem
    _validator("warpgroup-problem-v2.schema.json").validate(
        json.loads(warpgroup_problem_to_json(problem))
    )

    solved = schedule_warpgroups(program, library)
    assert solved == schedule_warpgroups(problem)
    schedule = solved.schedule
    times = _time_map(solved.schedule)
    produce = times[(0, "produce")]
    use = times[(0, "use")]
    publish = times[(0, "publish")]
    assert produce.issue_end < produce.completion <= use.start
    assert publish.completion <= times[(1, "consume")].start
    assert times[(1, "consume")].completion <= times[(1, "publish")].start
    assert solved.makespan == max(item.completion for item in solved.schedule.times)

    assert schedule.format == SCHEDULE_FORMAT_V2
    assert SynchronizationEdge("produce", "use", 0) in schedule.sync
    assert SynchronizationEdge("consume", "publish", 1) not in schedule.sync
    verify_warpgroup_schedule(problem, schedule)
    encoded = warpgroup_schedule_to_json(schedule)
    document = json.loads(encoded)
    assert set(document) == {"format", "lanes", "sync", "times"}
    assert all(len(row) == 5 for row in document["times"])
    _validator("warpgroup-schedule-v2.schema.json").validate(document)
    assert warpgroup_schedule_to_json(warpgroup_schedule_from_json(encoded)) == encoded

    lane_overlap = tuple(
        TimedOperation(
            item.iteration,
            item.operation_id,
            produce.start + 1,
            issue_end=produce.start + 2,
            completion=produce.start + 2,
        )
        if item.operation_id == "use"
        else item
        for item in schedule.times
    )
    with pytest.raises(WarpgroupVerificationError, match="lane order"):
        verify_warpgroup_schedule(
            problem,
            WarpgroupSchedule(SCHEDULE_FORMAT_V2, schedule.lanes, schedule.sync, lane_overlap),
        )

    resource_program = _async_program()
    resource_library = _async_library(resource_program, resource_windows=True)
    resource_problem = build_warpgroup_problem(resource_program, resource_library)
    with pytest.raises(WarpgroupVerificationError, match="exceeds capacity"):
        verify_warpgroup_schedule(resource_problem, schedule)
    constrained = schedule_warpgroups(resource_problem)
    verify_warpgroup_schedule(resource_problem, constrained.schedule)
    assert constrained.makespan > solved.makespan
    assert constrained.makespan == max(item.completion for item in constrained.schedule.times)

    missing_register_sync = WarpgroupSchedule(
        SCHEDULE_FORMAT_V2,
        schedule.lanes,
        tuple(edge for edge in schedule.sync if edge != SynchronizationEdge("produce", "use", 0)),
        schedule.times,
    )
    with pytest.raises(WarpgroupVerificationError, match="no lane/sync path"):
        verify_warpgroup_schedule(problem, missing_register_sync)

    v1_problem = _generic_values()[1]
    v1_schedule = export_warpgroup_schedule(v1_problem, solve_warpgroup_problem(v1_problem))
    assert v1_schedule.format == SCHEDULE_FORMAT
    assert all(item.issue_end == item.end for item in v1_schedule.times)
    assert all(
        len(row) == 4 for row in json.loads(warpgroup_schedule_to_json(v1_schedule))["times"]
    )


def test_m5_completion_event_graph_reduces_through_an_async_middle_operation() -> None:
    value_type = TensorType("value", (1,), DType.FP32, MemorySpace.REGISTER)
    problem = WarpgroupProblem(
        PROBLEM_FORMAT_V2,
        "cycle",
        1,
        (),
        (value_type,),
        (),
        ProblemLoop(
            "%iteration",
            1,
            (),
            (
                ProblemOperation(
                    "a",
                    (OperationOutput("%a", "value", ScalarLiteral(1.0)),),
                    issue_duration=1,
                    completion_latency=1,
                ),
                ProblemOperation(
                    "x",
                    (OperationOutput("%x", "value", CopyExpression(ValueRef("%a"))),),
                    issue_duration=1,
                    completion_latency=4,
                ),
                ProblemOperation(
                    "b",
                    (
                        OperationOutput(
                            "%b",
                            "value",
                            ElementwiseExpression(
                                ElementwiseOperator.ADD,
                                (ValueRef("%a"), ValueRef("%x")),
                            ),
                        ),
                    ),
                    issue_duration=1,
                    completion_latency=1,
                ),
            ),
        ),
    )
    schedule = export_warpgroup_schedule(problem, solve_warpgroup_problem(problem))

    verify_warpgroup_schedule(problem, schedule)
    assert schedule.lanes[0].operations == ("a", "x", "b")
    assert SynchronizationEdge("x", "b", 0) in schedule.sync
    assert SynchronizationEdge("a", "b", 0) not in schedule.sync


def test_lsy_reference_documents_and_complete_workflow_are_one_smoke_test() -> None:
    program = warpgroup_program_from_json(_document("lsy-schedule-input.json"))
    reference_problem = warpgroup_problem_from_json(_document("warpgroup-closed-problem.json"))
    reference_schedule = warpgroup_schedule_from_json(_document("lsy-schedule-output.json"))
    problem = build_warpgroup_problem(program, _library(program))
    result = schedule_warpgroups(program, _library(program))
    schedule = result.schedule
    reference_signatures = tuple(
        sorted(
            {operation_signature(program, operation) for operation in program.loop.ops},
            key=lambda item: item.canonical_key,
        )
    )
    coverage = b200_warpgroup_coverage_matrix(reference_signatures)

    verify_warpgroup_schedule(problem, schedule)
    assert {item.family for item in coverage.entries} == set(B200OperationFamily)
    assert all(item.status is CalibrationStatus.MISSING for item in coverage.entries)
    assert warpgroup_program_from_json(warpgroup_program_to_json(program)) == program
    assert warpgroup_problem_from_json(warpgroup_problem_to_json(problem)) == problem
    assert warpgroup_schedule_from_json(warpgroup_schedule_to_json(schedule)) == schedule
    assert (
        warpgroup_problem_from_json(warpgroup_problem_to_json(reference_problem))
        == reference_problem
    )
    assert (
        warpgroup_schedule_from_json(warpgroup_schedule_to_json(reference_schedule))
        == reference_schedule
    )
    assert {item for lane in schedule.lanes for item in lane.operations} == {
        operation.id for operation in problem.loop.ops
    }
    lane_by_operation = {
        operation_id: lane_index
        for lane_index, lane in enumerate(schedule.lanes)
        for operation_id in lane.operations
    }
    kind_by_operation = {
        operation.id: operation_signature(program, operation).kind for operation in program.loop.ops
    }
    times = {(timed.iteration, timed.operation_id): timed for timed in schedule.times}
    assert (
        len(
            {
                lane_by_operation[operation_id]
                for operation_id, kind in kind_by_operation.items()
                if kind is not OperationKind.COPY
            }
        )
        > 1
    )
    assert any(
        lane_by_operation[left.operation_id] != lane_by_operation[right.operation_id]
        and kind_by_operation[left.operation_id] is OperationKind.COPY
        and kind_by_operation[right.operation_id] is not OperationKind.COPY
        and max(left.start, right.start) < min(left.end, right.end)
        for left in schedule.times
        for right in schedule.times
        if left.iteration == right.iteration
    )
    assert schedule.sync
    assert any(
        kind_by_operation[operation.id] is OperationKind.COPY
        and any(
            kind_by_operation[previous.operation_id] is not OperationKind.COPY
            and lane_by_operation[operation.id] != lane_by_operation[previous.operation_id]
            and times[(1, operation.id)].start < previous.end
            for previous in schedule.times
            if previous.iteration == 0
        )
        for operation in program.loop.ops
    )

    types = {item.id: item for item in program.types}
    value_types = {item.id: types[item.type_id] for item in program.inputs}
    output_owner = {
        output.id: operation.id for operation in program.loop.ops for output in operation.outputs
    }
    outputs = {output.id: output for operation in program.loop.ops for output in operation.outputs}
    value_types.update({output.id: types[output.type_id] for output in outputs.values()})
    for iter_arg in program.loop.iter_args:
        value_types[iter_arg.id] = value_types[iter_arg.yield_value.id]

    probability_handoffs: list[tuple[str, str]] = []
    for output in outputs.values():
        if output.type_id != "shared_probability":
            continue
        assert type(output.expression) is CopyExpression
        assert type(output.expression.source) is ValueRef
        assert value_types[output.expression.source.id].space is MemorySpace.REGISTER
        consumers = tuple(
            operation.id
            for operation in program.loop.ops
            if any(
                output.id in value_references(candidate.expression)
                for candidate in operation.outputs
            )
        )
        assert len(consumers) == 1
        producer = output_owner[output.id]
        consumer = consumers[0]
        assert lane_by_operation[producer] != lane_by_operation[consumer]
        assert SynchronizationEdge(producer, consumer, 0) in schedule.sync
        probability_handoffs.append((producer, consumer))
    assert len(probability_handoffs) == 2

    output_half_lanes: list[int] = []
    for iter_arg in program.loop.iter_args:
        if outputs[iter_arg.yield_value.id].type_id != "output_half":
            continue
        reached = {iter_arg.id}
        component: set[str] = set()
        changed = True
        while changed:
            changed = False
            for output in outputs.values():
                if output.type_id != "output_half" or output.id in reached:
                    continue
                if reached.isdisjoint(value_references(output.expression)):
                    continue
                reached.add(output.id)
                component.add(output_owner[output.id])
                changed = True
        assert iter_arg.yield_value.id in reached
        lanes = {lane_by_operation[operation_id] for operation_id in component}
        assert len(lanes) == 1
        output_half_lanes.extend(lanes)
    assert len(output_half_lanes) == 2
    assert len(set(output_half_lanes)) == 2


def test_m4_python_program_and_closed_problem_share_one_verified_workflow() -> None:
    program = _generic_values()[0]
    library = _library(program)
    problem = build_warpgroup_problem(program, library)

    from_program = schedule_warpgroups(program, library)
    from_problem = schedule_warpgroups(problem)

    assert type(from_program) is WarpgroupScheduleResult
    assert from_program == from_problem
    assert from_program.makespan == max(item.end for item in from_program.schedule.times)
    verify_warpgroup_schedule(problem, from_program.schedule)

    with pytest.raises(WarpgroupValidationError, match="requires a cost library"):
        schedule_warpgroups(program)
    with pytest.raises(WarpgroupValidationError, match="rejects a cost library"):
        schedule_warpgroups(problem, library)
    with pytest.raises(WarpgroupValidationError, match="exact WarpgroupProgram"):
        schedule_warpgroups({})  # type: ignore[arg-type]


def _run_warpgroup_cli(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "tilefoundry.cli", "schedule", *arguments],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_m4_cli_program_problem_json_and_text_share_one_schedule(tmp_path: Path) -> None:
    program, problem, _ = _generic_values()
    program_path = tmp_path / "program.json"
    problem_path = tmp_path / "problem.json"
    program_path.write_text(warpgroup_program_to_json(program), encoding="utf-8")
    problem_path.write_text(warpgroup_problem_to_json(problem), encoding="utf-8")

    program_json = _run_warpgroup_cli(
        "--warpgroup-program", str(program_path), "--fixture-costs", "--json"
    )
    problem_json = _run_warpgroup_cli("--warpgroup-problem", str(problem_path), "--json")
    text = _run_warpgroup_cli("--warpgroup-problem", str(problem_path))

    assert program_json.returncode == problem_json.returncode == text.returncode == 0
    assert program_json.stderr == problem_json.stderr == text.stderr == ""
    assert program_json.stdout == problem_json.stdout
    schedule = warpgroup_schedule_from_json(problem_json.stdout)
    assert set(json.loads(problem_json.stdout)) == {"format", "lanes", "sync", "times"}
    assert f"makespan={max(item.end for item in schedule.times)}" in text.stdout
    for lane_index, lane in enumerate(schedule.lanes):
        assert f"lane {lane_index}: {' -> '.join(lane.operations) or '(empty)'}" in text.stdout
    for timed in schedule.times:
        assert f"{timed.operation_id}[{timed.start},{timed.end})" in text.stdout
    for edge in schedule.sync:
        assert f"{edge.after} -> {edge.before} distance={edge.distance}" in text.stdout

    missing_costs = _run_warpgroup_cli("--warpgroup-program", str(program_path))
    extra_costs = _run_warpgroup_cli("--warpgroup-problem", str(problem_path), "--fixture-costs")
    assert missing_costs.returncode == extra_costs.returncode == 1
    assert "requires --fixture-costs" in missing_costs.stderr
    assert "rejects --fixture-costs" in extra_costs.stderr


def test_m4_closed_problem_replays_without_library_and_loads_only_solver_backend() -> None:
    problem = _generic_values()[1]
    expected = warpgroup_schedule_to_json(schedule_warpgroups(problem).schedule)
    code = f"""
import sys
from tilefoundry.schedule.warpgroup import (
    schedule_warpgroups,
    warpgroup_problem_from_json,
    warpgroup_schedule_to_json,
)
forbidden = ('ortools', 'cuda', 'tilefoundry_costmodel', 'tilefoundry.target.amx', 'tilefoundry.target.cuda')
assert not [name for name in sys.modules if any(name == root or name.startswith(root + '.') for root in forbidden)]
problem = warpgroup_problem_from_json({warpgroup_problem_to_json(problem)!r})
result = schedule_warpgroups(problem)
assert any(name == 'ortools' or name.startswith('ortools.') for name in sys.modules)
forbidden = forbidden[1:]
assert not [name for name in sys.modules if any(name == root or name.startswith(root + '.') for root in forbidden)]
print(warpgroup_schedule_to_json(result.schedule))
"""
    replay = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    assert replay.stdout.strip() == expected

    import_code = """
import sys
import tilefoundry.cli
import tilefoundry.schedule.warpgroup
forbidden = ('ortools', 'cuda', 'tilefoundry_costmodel', 'tilefoundry.target.amx', 'tilefoundry.target.cuda')
assert not [name for name in sys.modules if any(name == root or name.startswith(root + '.') for root in forbidden)]
"""
    subprocess.run([sys.executable, "-c", import_code], cwd=ROOT, check=True)


def _m3_pipeline_problem() -> WarpgroupProblem:
    types = (
        TensorType("source", (2, 1), DType.FP32, MemorySpace.GLOBAL),
        TensorType("shared", (1,), DType.FP32, MemorySpace.SHARED),
        TensorType("register", (1,), DType.FP32, MemorySpace.REGISTER),
    )
    return WarpgroupProblem(
        PROBLEM_FORMAT,
        "cycle",
        2,
        (),
        types,
        (ProgramInput("%source", "source"),),
        ProblemLoop(
            "%iteration",
            2,
            (),
            (
                ProblemOperation(
                    "publish",
                    (
                        OperationOutput(
                            "%shared",
                            "shared",
                            CopyExpression(
                                IndexExpression(ValueRef("%source"), (LoopIndexRef("%iteration"),))
                            ),
                        ),
                        OperationOutput("%publish_token", "register", ScalarLiteral(0.0)),
                    ),
                    2,
                    (),
                ),
                ProblemOperation(
                    "publish_tail",
                    (
                        OperationOutput(
                            "%publish_done",
                            "register",
                            CopyExpression(ValueRef("%publish_token")),
                        ),
                    ),
                    6,
                    (),
                ),
                ProblemOperation(
                    "consume",
                    (
                        OperationOutput(
                            "%consume_token",
                            "register",
                            CopyExpression(ValueRef("%shared")),
                        ),
                    ),
                    2,
                    (),
                ),
                ProblemOperation(
                    "consume_again",
                    (
                        OperationOutput(
                            "%consume_done",
                            "register",
                            ElementwiseExpression(
                                ElementwiseOperator.ADD,
                                (ValueRef("%consume_token"), ValueRef("%shared")),
                            ),
                        ),
                    ),
                    2,
                    (),
                ),
            ),
        ),
    )


def _m3_schedule() -> tuple[WarpgroupProblem, WarpgroupSchedule]:
    problem = _m3_pipeline_problem()
    return problem, export_warpgroup_schedule(problem, solve_warpgroup_problem(problem))


def test_m3_export_reduces_sync_and_round_trips_the_four_field_schedule() -> None:
    """M2 had no workflow covering cross-lane control reachability."""
    problem, schedule = _m3_schedule()

    verify_warpgroup_schedule(problem, schedule)
    assert schedule.sync == (
        SynchronizationEdge("consume_again", "publish", 1),
        SynchronizationEdge("publish", "consume", 0),
    )
    assert SynchronizationEdge("publish", "consume_again", 0) not in schedule.sync
    assert SynchronizationEdge("consume", "publish", 1) not in schedule.sync

    encoded = warpgroup_schedule_to_json(schedule)
    assert set(json.loads(encoded)) == {"format", "lanes", "sync", "times"}
    assert warpgroup_schedule_to_json(warpgroup_schedule_from_json(encoded)) == encoded
    assert export_warpgroup_schedule(problem, solve_warpgroup_problem(problem)) == schedule

    without_handoff = WarpgroupSchedule(
        SCHEDULE_FORMAT,
        schedule.lanes,
        tuple(
            edge for edge in schedule.sync if edge != SynchronizationEdge("publish", "consume", 0)
        ),
        schedule.times,
    )
    with pytest.raises(WarpgroupVerificationError, match="no lane/sync path"):
        verify_warpgroup_schedule(problem, without_handoff)


def test_m3_verifier_rejects_mutated_witness_contracts() -> None:
    """M2 tests observed solves but could not reject a modified exported witness."""
    problem, schedule = _m3_schedule()
    times = list(schedule.times)
    first = times[0]
    times[0] = TimedOperation(first.iteration, first.operation_id, first.start, first.end + 1)
    with pytest.raises(WarpgroupVerificationError, match="wrong duration"):
        verify_warpgroup_schedule(
            problem,
            WarpgroupSchedule(SCHEDULE_FORMAT, schedule.lanes, schedule.sync, tuple(times)),
        )

    with pytest.raises(WarpgroupVerificationError, match="time coverage"):
        verify_warpgroup_schedule(
            problem,
            WarpgroupSchedule(SCHEDULE_FORMAT, schedule.lanes, schedule.sync, schedule.times[:-1]),
        )

    reversed_lanes = tuple(
        WarpgroupLane(tuple(reversed(lane.operations))) for lane in schedule.lanes
    )
    with pytest.raises(WarpgroupVerificationError, match="lane order"):
        verify_warpgroup_schedule(
            problem,
            WarpgroupSchedule(SCHEDULE_FORMAT, reversed_lanes, schedule.sync, schedule.times),
        )

    without_lifetime = WarpgroupSchedule(
        SCHEDULE_FORMAT,
        schedule.lanes,
        tuple(edge for edge in schedule.sync if edge.distance == 0),
        schedule.times,
    )
    with pytest.raises(WarpgroupVerificationError, match="no lane/sync path"):
        verify_warpgroup_schedule(problem, without_lifetime)

    resource_problem = _independent_solve_problem(warp_groups=2, capacity=1)
    overlapping = WarpgroupSchedule(
        SCHEDULE_FORMAT,
        (WarpgroupLane(("alpha",)), WarpgroupLane(("beta",))),
        (),
        (
            TimedOperation(0, "alpha", 0, 2),
            TimedOperation(0, "beta", 0, 2),
        ),
    )

    extra_time = TimedOperation(0, "unknown", 0, 1)
    with pytest.raises(WarpgroupVerificationError, match="time coverage"):
        verify_warpgroup_schedule(
            problem,
            WarpgroupSchedule(
                SCHEDULE_FORMAT,
                schedule.lanes,
                schedule.sync,
                schedule.times + (extra_time,),
            ),
        )

    duplicate_times = WarpgroupSchedule(
        SCHEDULE_FORMAT, schedule.lanes, schedule.sync, schedule.times
    )
    object.__setattr__(duplicate_times, "times", schedule.times + (schedule.times[0],))
    with pytest.raises(WarpgroupVerificationError, match="duplicate timed"):
        verify_warpgroup_schedule(problem, duplicate_times)
    with pytest.raises(WarpgroupVerificationError, match="exceeds capacity"):
        verify_warpgroup_schedule(resource_problem, overlapping)

    cyclic = WarpgroupSchedule(
        SCHEDULE_FORMAT,
        overlapping.lanes,
        (
            SynchronizationEdge("alpha", "beta", 0),
            SynchronizationEdge("beta", "alpha", 0),
        ),
        overlapping.times,
    )
    with pytest.raises(WarpgroupVerificationError, match="sync inequality|cycle"):
        verify_warpgroup_schedule(_independent_solve_problem(warp_groups=2, capacity=2), cyclic)

    register_type = TensorType("register", (1,), DType.FP32, MemorySpace.REGISTER)
    register_problem = WarpgroupProblem(
        PROBLEM_FORMAT,
        "cycle",
        2,
        (),
        (register_type,),
        (),
        ProblemLoop(
            "%iteration",
            1,
            (),
            (
                ProblemOperation(
                    "define",
                    (OperationOutput("%value", "register", ScalarLiteral(1.0)),),
                    1,
                    (),
                ),
                ProblemOperation(
                    "use",
                    (
                        OperationOutput(
                            "%result",
                            "register",
                            CopyExpression(ValueRef("%value")),
                        ),
                    ),
                    1,
                    (),
                ),
            ),
        ),
    )
    cross_lane_register = WarpgroupSchedule(
        SCHEDULE_FORMAT,
        (WarpgroupLane(("define",)), WarpgroupLane(("use",))),
        (),
        (
            TimedOperation(0, "define", 0, 1),
            TimedOperation(0, "use", 1, 2),
        ),
    )
    with pytest.raises(WarpgroupVerificationError, match="crosses warpgroup lanes"):
        verify_warpgroup_schedule(register_problem, cross_lane_register)


def test_m3_verifier_process_does_not_load_ortools() -> None:
    problem, schedule = _m3_schedule()
    code = f"""
import sys
from tilefoundry.schedule.warpgroup import (
    verify_warpgroup_schedule,
    warpgroup_problem_from_json,
    warpgroup_schedule_from_json,
)
problem = warpgroup_problem_from_json({warpgroup_problem_to_json(problem)!r})
schedule = warpgroup_schedule_from_json({warpgroup_schedule_to_json(schedule)!r})
verify_warpgroup_schedule(problem, schedule)
forbidden = ('ortools', 'cuda', 'tilefoundry_costmodel', 'tilefoundry.target.amx', 'tilefoundry.target.cuda')
assert not [name for name in sys.modules if any(name == root or name.startswith(root + '.') for root in forbidden)]
"""
    subprocess.run([sys.executable, "-c", code], check=True, cwd=ROOT)
