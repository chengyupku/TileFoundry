"""Strict canonical JSON codecs for the warpgroup scheduling documents."""

from __future__ import annotations

import json
import math
from typing import cast

from .cost import (
    CanonicalExpression,
    OperationCost,
    OperationCostEntry,
    OperationKind,
    OperationSignature,
    SignatureOutput,
    SignatureValueType,
    WarpgroupHardware,
)
from .errors import WarpgroupSerializationError, WarpgroupValidationError
from .expression import (
    CastExpression,
    ConcatExpression,
    CopyExpression,
    ElementwiseExpression,
    ElementwiseOperator,
    ExpressionAttribute,
    ExpressionValue,
    IndexExpression,
    LoopIndexRef,
    MatmulExpression,
    NegativeInfinity,
    ReduceExpression,
    ReductionOperator,
    ScalarLiteral,
    SelectExpression,
    TransposeExpression,
    ValueRef,
    fold_expression,
)
from .model import (
    DType,
    LoopIterArg,
    MemorySpace,
    OperationOutput,
    ProgramInput,
    ProgramLoop,
    ProgramOperation,
    ResourceCapacity,
    ResourceWindow,
    SynchronizationEdge,
    TensorType,
    TimedOperation,
    WarpgroupLane,
    WarpgroupProgram,
    WarpgroupSchedule,
)


def _duplicate_fields(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field {key!r}")
        result[key] = value
    return result


def _finite_float(value: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("non-finite JSON number")
    return result


def _loads(text: str) -> object:
    if type(text) is not str:
        raise WarpgroupSerializationError("JSON input must be text")
    try:
        return json.loads(
            text,
            object_pairs_hook=_duplicate_fields,
            parse_float=_finite_float,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"invalid JSON constant {token!r}")
            ),
        )
    except (json.JSONDecodeError, TypeError, ValueError) as error:
        raise WarpgroupSerializationError(f"invalid JSON: {error}") from error


def _canonical(value: object) -> str:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as error:
        raise WarpgroupSerializationError(f"value is not canonical JSON: {error}") from error


def _object(
    value: object,
    *,
    fields: frozenset[str],
    label: str,
) -> dict[str, object]:
    if type(value) is not dict:
        raise WarpgroupSerializationError(f"{label} must be a JSON object")
    result = cast(dict[str, object], value)
    unknown = sorted(set(result) - fields)
    missing = sorted(fields - set(result))
    if unknown:
        raise WarpgroupSerializationError(
            f"{label} contains unknown field(s): {', '.join(unknown)}"
        )
    if missing:
        raise WarpgroupSerializationError(
            f"{label} is missing required field(s): {', '.join(missing)}"
        )
    return result


def _array(value: object, label: str) -> tuple[object, ...]:
    if type(value) is not list:
        raise WarpgroupSerializationError(f"{label} must be a JSON array")
    return tuple(cast(list[object], value))


def _string(value: object, label: str) -> str:
    if type(value) is not str:
        raise WarpgroupSerializationError(f"{label} must be a string")
    return value


def _integer(value: object, label: str) -> int:
    if type(value) is not int:
        raise WarpgroupSerializationError(f"{label} must be an integer")
    return value


def _number(value: object, label: str) -> int | float:
    if type(value) not in (int, float) or not math.isfinite(cast(float, value)):
        raise WarpgroupSerializationError(f"{label} must be a finite JSON number")
    return cast(int | float, value)


def _enum(enum_type: type, value: object, label: str) -> object:
    text = _string(value, label)
    try:
        return enum_type(text)
    except ValueError as error:
        raise WarpgroupSerializationError(f"{label} has unknown value {text!r}") from error


def _exact(value: object, expected: type, label: str) -> None:
    if type(value) is not expected:
        raise WarpgroupSerializationError(
            f"{label} must be exact {expected.__name__}, got {type(value).__name__}"
        )


def _decode_expression(value: object, loop_index: str) -> ExpressionValue:
    if type(value) in (int, float):
        return ScalarLiteral(_number(value, "scalar literal"))
    if type(value) is str:
        text = value
        if text == NegativeInfinity.VALUE.value:
            return ScalarLiteral(NegativeInfinity.VALUE)
        return ValueRef(text)
    items = _array(value, "expression")
    if not items:
        raise WarpgroupSerializationError("expression must not be empty")
    operator = _string(items[0], "expression operator")
    operands = items[1:]
    if operator == "index":
        if len(operands) < 2:
            raise WarpgroupSerializationError("index requires a source and at least one index")
        source = ValueRef(_string(operands[0], "index source"))
        indices: list[int | LoopIndexRef] = []
        for operand in operands[1:]:
            if type(operand) is int:
                indices.append(operand)
            elif type(operand) is str:
                index = _string(operand, "index operand")
                if index != loop_index:
                    raise WarpgroupSerializationError(
                        f"index operand must name loop index {loop_index!r}"
                    )
                indices.append(LoopIndexRef(index))
            else:
                raise WarpgroupSerializationError(
                    "index operand must be a non-negative integer or loop index"
                )
        return IndexExpression(source, tuple(indices))
    if operator in {"copy", "cast", "transpose"}:
        if len(operands) != 1:
            raise WarpgroupSerializationError(f"{operator} requires exactly one operand")
        unary_source = _decode_expression(operands[0], loop_index)
        if operator == "copy":
            return CopyExpression(unary_source)
        if operator == "cast":
            return CastExpression(unary_source)
        return TransposeExpression(unary_source)
    if operator == "matmul":
        if len(operands) != 2:
            raise WarpgroupSerializationError("matmul requires exactly two operands")
        return MatmulExpression(
            _decode_expression(operands[0], loop_index),
            _decode_expression(operands[1], loop_index),
        )
    if operator == "concat":
        if len(operands) < 3:
            raise WarpgroupSerializationError("concat requires an axis and at least two values")
        return ConcatExpression(
            _integer(operands[0], "concat axis"),
            tuple(_decode_expression(item, loop_index) for item in operands[1:]),
        )
    if operator == "select":
        if len(operands) != 3:
            raise WarpgroupSerializationError("select requires exactly three operands")
        return SelectExpression(
            _decode_expression(operands[0], loop_index),
            _decode_expression(operands[1], loop_index),
            _decode_expression(operands[2], loop_index),
        )
    if operator == "reduce":
        if len(operands) != 3:
            raise WarpgroupSerializationError("reduce requires an operator, axis, and source")
        reduction = cast(
            ReductionOperator,
            _enum(ReductionOperator, operands[0], "reduce operator"),
        )
        return ReduceExpression(
            reduction,
            _integer(operands[1], "reduce axis"),
            _decode_expression(operands[2], loop_index),
        )
    try:
        elementwise = ElementwiseOperator(operator)
    except ValueError as error:
        raise WarpgroupSerializationError(f"unknown expression operator {operator!r}") from error
    return ElementwiseExpression(
        elementwise,
        tuple(_decode_expression(item, loop_index) for item in operands),
    )


def _encode_expression(value: ExpressionValue) -> object:
    def scalar(item: ScalarLiteral) -> object:
        literal = item.value
        return literal.value if type(literal) is NegativeInfinity else literal

    def compose(
        operator: str,
        attributes: tuple[ExpressionAttribute, ...],
        children: tuple[object, ...],
    ) -> object:
        encoded = tuple(item.id if type(item) is LoopIndexRef else item for item in attributes)
        if operator == "index":
            return [operator, *children, *encoded]
        return [operator, *encoded, *children]

    return fold_expression(
        value,
        reference=lambda item: item.id,
        scalar=scalar,
        compose=compose,
    )


def _decode_types(value: object) -> tuple[TensorType, ...]:
    data = _object_map(value, "types")
    result: list[TensorType] = []
    for type_id, raw in data.items():
        item = _object(raw, fields=frozenset({"shape", "dtype", "space"}), label="type")
        result.append(
            TensorType(
                type_id,
                tuple(
                    _integer(extent, "shape extent") for extent in _array(item["shape"], "shape")
                ),
                cast(DType, _enum(DType, item["dtype"], "dtype")),
                cast(MemorySpace, _enum(MemorySpace, item["space"], "memory space")),
            )
        )
    return tuple(result)


def _encode_types(types: tuple[TensorType, ...]) -> dict[str, object]:
    result: dict[str, object] = {}
    for item in types:
        _exact(item, TensorType, "type")
        result[item.id] = {
            "shape": list(item.shape),
            "dtype": item.dtype.value,
            "space": item.space.value,
        }
    return result


def _object_map(value: object, label: str) -> dict[str, object]:
    if type(value) is not dict:
        raise WarpgroupSerializationError(f"{label} must be a JSON object")
    return cast(dict[str, object], value)


def _decode_inputs(value: object) -> tuple[ProgramInput, ...]:
    result: list[ProgramInput] = []
    for raw in _array(value, "inputs"):
        item = _object(raw, fields=frozenset({"id", "type"}), label="input")
        result.append(
            ProgramInput(_string(item["id"], "input ID"), _string(item["type"], "input type"))
        )
    return tuple(result)


def _encode_inputs(inputs: tuple[ProgramInput, ...]) -> list[object]:
    result: list[object] = []
    for item in inputs:
        _exact(item, ProgramInput, "input")
        result.append({"id": item.id, "type": item.type_id})
    return result


def _decode_iter_args(value: object) -> tuple[LoopIterArg, ...]:
    result: list[LoopIterArg] = []
    for raw in _array(value, "iter_args"):
        item = _object(raw, fields=frozenset({"id", "init", "yield"}), label="iter_arg")
        init_raw = item["init"]
        init: ValueRef | ScalarLiteral
        if init_raw == NegativeInfinity.VALUE.value:
            init = ScalarLiteral(NegativeInfinity.VALUE)
        elif type(init_raw) is str:
            init = ValueRef(_string(init_raw, "iter_arg init"))
        else:
            init = ScalarLiteral(_number(init_raw, "iter_arg init"))
        result.append(
            LoopIterArg(
                _string(item["id"], "iter_arg ID"),
                init,
                ValueRef(_string(item["yield"], "iter_arg yield")),
            )
        )
    return tuple(result)


def _encode_iter_args(iter_args: tuple[LoopIterArg, ...]) -> list[object]:
    result: list[object] = []
    for item in iter_args:
        _exact(item, LoopIterArg, "iter_arg")
        _exact(item.yield_value, ValueRef, "iter_arg yield")
        if type(item.init) is ValueRef:
            init: object = item.init.id
        elif type(item.init) is ScalarLiteral:
            init = _encode_expression(item.init)
        else:
            raise WarpgroupSerializationError("iter_arg init must be exact typed value")
        result.append({"id": item.id, "init": init, "yield": item.yield_value.id})
    return result


def _decode_outputs(value: object, loop_index: str) -> tuple[OperationOutput, ...]:
    result: list[OperationOutput] = []
    for raw in _array(value, "outputs"):
        item = _object(
            raw,
            fields=frozenset({"id", "type", "expr"}),
            label="operation output",
        )
        result.append(
            OperationOutput(
                _string(item["id"], "output ID"),
                _string(item["type"], "output type"),
                _decode_expression(item["expr"], loop_index),
            )
        )
    return tuple(result)


def _encode_outputs(outputs: tuple[OperationOutput, ...]) -> list[object]:
    result: list[object] = []
    for item in outputs:
        _exact(item, OperationOutput, "operation output")
        result.append(
            {"id": item.id, "type": item.type_id, "expr": _encode_expression(item.expression)}
        )
    return result


def _decode_program_loop(value: object) -> ProgramLoop:
    data = _object(
        value,
        fields=frozenset({"index", "iterations", "iter_args", "ops"}),
        label="loop",
    )
    index = _string(data["index"], "loop index")
    operations: list[ProgramOperation] = []
    for raw in _array(data["ops"], "loop ops"):
        item = _object(
            raw,
            fields=frozenset({"id", "warp_group", "outputs"}),
            label="operation",
        )
        operations.append(
            ProgramOperation(
                _string(item["id"], "operation ID"),
                _decode_outputs(item["outputs"], index),
                _integer(item["warp_group"], "operation warp_group"),
            )
        )
    return ProgramLoop(
        index,
        _integer(data["iterations"], "loop iterations"),
        _decode_iter_args(data["iter_args"]),
        tuple(operations),
    )


def _encode_program_loop(loop: ProgramLoop) -> dict[str, object]:
    _exact(loop, ProgramLoop, "program loop")
    operations: list[object] = []
    for item in loop.ops:
        _exact(item, ProgramOperation, "program operation")
        operations.append(
            {
                "id": item.id,
                "warp_group": item.warp_group,
                "outputs": _encode_outputs(item.outputs),
            }
        )
    return {
        "index": loop.index,
        "iterations": loop.iterations,
        "iter_args": _encode_iter_args(loop.iter_args),
        "ops": operations,
    }


def _decode_resources(value: object) -> tuple[ResourceCapacity, ...]:
    return tuple(
        ResourceCapacity(resource_id, _integer(capacity, "resource capacity"))
        for resource_id, capacity in _object_map(value, "resources").items()
    )


def _encode_resources(resources: tuple[ResourceCapacity, ...]) -> dict[str, object]:
    result: dict[str, object] = {}
    for item in resources:
        _exact(item, ResourceCapacity, "resource capacity")
        result[item.id] = item.capacity
    return result


def _decode_windows(value: object) -> tuple[ResourceWindow, ...]:
    windows: list[ResourceWindow] = []
    for raw in _array(value, "resource windows"):
        item = _object(
            raw,
            fields=frozenset({"resource_id", "amount", "start_offset", "duration"}),
            label="resource window",
        )
        windows.append(
            ResourceWindow(
                _string(item["resource_id"], "resource window ID"),
                _integer(item["amount"], "resource window amount"),
                _integer(item["start_offset"], "resource window start offset"),
                _integer(item["duration"], "resource window duration"),
            )
        )
    return tuple(windows)


def _encode_windows(windows: tuple[ResourceWindow, ...]) -> list[object]:
    encoded: list[object] = []
    for item in windows:
        _exact(item, ResourceWindow, "resource window")
        encoded.append(
            {
                "resource_id": item.resource_id,
                "amount": item.amount,
                "start_offset": item.start_offset,
                "duration": item.duration,
            }
        )
    return encoded


def _decode_signature_type(value: object) -> SignatureValueType:
    item = _object(
        value,
        fields=frozenset({"shape", "dtype", "space"}),
        label="signature type",
    )
    return SignatureValueType(
        tuple(
            _integer(extent, "signature shape extent")
            for extent in _array(item["shape"], "signature shape")
        ),
        cast(DType, _enum(DType, item["dtype"], "signature dtype")),
        cast(MemorySpace, _enum(MemorySpace, item["space"], "signature memory space")),
    )


def _encode_signature_type(value: SignatureValueType) -> dict[str, object]:
    _exact(value, SignatureValueType, "signature type")
    return {"shape": list(value.shape), "dtype": value.dtype.value, "space": value.space.value}


def _decode_canonical_expression(value: object) -> CanonicalExpression:
    items = _array(value, "canonical expression")
    if not items:
        raise WarpgroupSerializationError("canonical expression must not be empty")
    result: list[str | int | float | CanonicalExpression] = []
    for item in items:
        if type(item) is list:
            result.append(_decode_canonical_expression(item))
        elif type(item) is str:
            result.append(item)
        else:
            result.append(_number(item, "canonical expression atom"))
    return tuple(result)


def _encode_canonical_expression(value: CanonicalExpression) -> list[object]:
    return [_encode_canonical_expression(item) if type(item) is tuple else item for item in value]


def _decode_signature(value: object) -> OperationSignature:
    item = _object(
        value,
        fields=frozenset({"kind", "operands", "outputs"}),
        label="operation signature",
    )
    outputs: list[SignatureOutput] = []
    for raw in _array(item["outputs"], "signature outputs"):
        output = _object(
            raw,
            fields=frozenset({"type", "expr"}),
            label="signature output",
        )
        outputs.append(
            SignatureOutput(
                _decode_signature_type(output["type"]),
                _decode_canonical_expression(output["expr"]),
            )
        )
    return OperationSignature(
        cast(OperationKind, _enum(OperationKind, item["kind"], "operation kind")),
        tuple(
            _decode_signature_type(raw) for raw in _array(item["operands"], "signature operands")
        ),
        tuple(outputs),
    )


def _encode_signature(value: OperationSignature) -> dict[str, object]:
    _exact(value, OperationSignature, "operation signature")
    return {
        "kind": value.kind.value,
        "operands": [_encode_signature_type(item) for item in value.operands],
        "outputs": [
            {
                "type": _encode_signature_type(item.type),
                "expr": _encode_canonical_expression(item.expression),
            }
            for item in value.outputs
        ],
    }


def warpgroup_program_from_json(text: str) -> WarpgroupProgram:
    """Decode and validate one strict authored program document."""
    try:
        data = _object(
            _loads(text),
            fields=frozenset({"format", "warp_groups", "types", "inputs", "loop"}),
            label="warpgroup program",
        )
        return WarpgroupProgram(
            _string(data["format"], "program format"),
            _integer(data["warp_groups"], "warp_groups"),
            _decode_types(data["types"]),
            _decode_inputs(data["inputs"]),
            _decode_program_loop(data["loop"]),
        )
    except WarpgroupSerializationError:
        raise
    except WarpgroupValidationError as error:
        raise WarpgroupSerializationError(str(error)) from error


def warpgroup_program_to_json(program: WarpgroupProgram) -> str:
    """Validate an exact typed program and return canonical JSON."""
    _exact(program, WarpgroupProgram, "program")
    payload = {
        "format": program.format,
        "warp_groups": program.warp_groups,
        "types": _encode_types(program.types),
        "inputs": _encode_inputs(program.inputs),
        "loop": _encode_program_loop(program.loop),
    }
    decoded = warpgroup_program_from_json(_canonical(payload))
    if decoded != program:
        raise WarpgroupSerializationError("program canonical round-trip changed the value")
    return _canonical(payload)


def warpgroup_hardware_from_json(text: str) -> WarpgroupHardware:
    """Decode and validate one strict hardware cost document."""
    try:
        data = _object(
            _loads(text),
            fields=frozenset({"format", "time_unit", "resources", "costs"}),
            label="warpgroup hardware",
        )
        entries: list[OperationCostEntry] = []
        for raw in _array(data["costs"], "hardware costs"):
            item = _object(
                raw,
                fields=frozenset(
                    {
                        "signature",
                        "issue_duration",
                        "completion_latency",
                        "resource_windows",
                    }
                ),
                label="hardware cost",
            )
            entries.append(
                OperationCostEntry(
                    _decode_signature(item["signature"]),
                    OperationCost(
                        _integer(item["issue_duration"], "issue duration"),
                        _integer(item["completion_latency"], "completion latency"),
                        _decode_windows(item["resource_windows"]),
                    ),
                )
            )
        return WarpgroupHardware(
            _string(data["format"], "hardware format"),
            _string(data["time_unit"], "time_unit"),
            _decode_resources(data["resources"]),
            tuple(entries),
        )
    except WarpgroupSerializationError:
        raise
    except WarpgroupValidationError as error:
        raise WarpgroupSerializationError(str(error)) from error


def warpgroup_hardware_to_json(hardware: WarpgroupHardware) -> str:
    """Validate exact typed hardware and return canonical JSON."""
    _exact(hardware, WarpgroupHardware, "hardware")
    payload = {
        "format": hardware.format,
        "time_unit": hardware.time_unit,
        "resources": _encode_resources(hardware.resources),
        "costs": [
            {
                "signature": _encode_signature(entry.signature),
                "issue_duration": entry.cost.issue_duration,
                "completion_latency": entry.cost.completion_latency,
                "resource_windows": _encode_windows(entry.cost.resource_windows),
            }
            for entry in hardware.entries
        ],
    }
    decoded = warpgroup_hardware_from_json(_canonical(payload))
    if decoded != hardware:
        raise WarpgroupSerializationError("hardware canonical round-trip changed the value")
    return _canonical(payload)


def warpgroup_schedule_from_json(text: str) -> WarpgroupSchedule:
    """Decode and validate one strict successful schedule document."""
    try:
        data = _object(
            _loads(text),
            fields=frozenset({"format", "lanes", "sync", "times"}),
            label="warpgroup schedule",
        )
        format_value = _string(data["format"], "schedule format")
        lanes = tuple(
            WarpgroupLane(tuple(_string(item, "lane operation") for item in _array(raw, "lane")))
            for raw in _array(data["lanes"], "lanes")
        )
        sync: list[SynchronizationEdge] = []
        for raw in _array(data["sync"], "sync"):
            item = _object(
                raw,
                fields=frozenset({"after", "before", "distance"}),
                label="sync edge",
            )
            sync.append(
                SynchronizationEdge(
                    _string(item["after"], "sync after"),
                    _string(item["before"], "sync before"),
                    _integer(item["distance"], "sync distance"),
                )
            )
        times: list[TimedOperation] = []
        for raw in _array(data["times"], "times"):
            row = _array(raw, "time row")
            if len(row) != 5:
                raise WarpgroupSerializationError("time row must contain exactly 5 items")
            iteration = _integer(row[0], "time iteration")
            operation_id = _string(row[1], "timed operation ID")
            start = _integer(row[2], "time start")
            times.append(
                TimedOperation(
                    iteration,
                    operation_id,
                    start,
                    _integer(row[3], "time issue end"),
                    _integer(row[4], "time completion"),
                )
            )
        return WarpgroupSchedule(format_value, lanes, tuple(sync), tuple(times))
    except WarpgroupSerializationError:
        raise
    except WarpgroupValidationError as error:
        raise WarpgroupSerializationError(str(error)) from error


def warpgroup_schedule_to_json(schedule: WarpgroupSchedule) -> str:
    """Validate an exact typed successful schedule and return canonical JSON."""
    _exact(schedule, WarpgroupSchedule, "schedule")
    lanes: list[object] = []
    for lane in schedule.lanes:
        _exact(lane, WarpgroupLane, "schedule lane")
        lanes.append(list(lane.operations))
    sync: list[object] = []
    for edge in schedule.sync:
        _exact(edge, SynchronizationEdge, "synchronization edge")
        sync.append({"after": edge.after, "before": edge.before, "distance": edge.distance})
    times: list[object] = []
    for timed in schedule.times:
        _exact(timed, TimedOperation, "timed operation")
        times.append(
            [
                timed.iteration,
                timed.operation_id,
                timed.start,
                timed.issue_end,
                timed.completion,
            ]
        )
    payload = {"format": schedule.format, "lanes": lanes, "sync": sync, "times": times}
    decoded = warpgroup_schedule_from_json(_canonical(payload))
    if decoded != schedule:
        raise WarpgroupSerializationError("schedule canonical round-trip changed the value")
    return _canonical(payload)


__all__ = [
    "warpgroup_hardware_from_json",
    "warpgroup_hardware_to_json",
    "warpgroup_program_from_json",
    "warpgroup_program_to_json",
    "warpgroup_schedule_from_json",
    "warpgroup_schedule_to_json",
]
