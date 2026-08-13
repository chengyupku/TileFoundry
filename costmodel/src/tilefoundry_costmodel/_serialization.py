"""Strict, deterministic JSON codecs for the M0 public records."""

# JSON decoding is intentionally boundary-typed: values are checked at runtime
# before entering immutable records, so a few constructor calls necessarily
# cross the static ``object`` representation used by the decoder.
# mypy: disable-error-code="arg-type,call-arg,misc,assignment"

from __future__ import annotations

import base64
import json
import math
from dataclasses import fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

from .constants import (
    HARDWARE_SCHEMA_VERSION,
    PLAN_SCHEMA_VERSION,
    PROFILE_SCHEMA_VERSION,
    PROGRAM_SCHEMA_VERSION,
    REQUEST_SCHEMA_VERSION,
    RESULT_SCHEMA_VERSION,
    SEARCH_PROBLEM_SCHEMA_VERSION,
)
from .errors import (
    HardwareSpecError,
    InvalidRequestError,
    ProfileStoreError,
    SearchProblemError,
    WorkloadError,
)
from .hardware.model import (
    FactOrigin,
    FactProvenance,
    HardwareSpec,
    StaticResourceSpec,
    StaticUnit,
    TemporalResourceSpec,
)
from .model import (
    ActivationKind,
    AxisExtent,
    DType,
    EpilogueKind,
    FlashAttentionSpec,
    GemmSpec,
    GqaDecodeSpec,
    HardwareSpecRef,
    MlpSpec,
    NamedShape,
    TensorDescriptor,
    TensorLayout,
    TimingMetric,
    WorkloadKind,
)
from .profiles.model import (
    MeasurementOrigin,
    ProfileEnvironment,
    ProfileMeasurement,
    ProfileSnapshot,
)
from .program import (
    AlignedRelation,
    CopyOp,
    DependencyRelationKind,
    ElementwiseKind,
    ElementwiseOp,
    EndpointRelation,
    GemmOp,
    InstanceEndpoint,
    LoopBarrier,
    MemorySpace,
    OpIterationDomain,
    ReduceOp,
    ReductionKind,
    TileCandidate,
    TileDependency,
    TileLoop,
    TileOpKind,
    TileProgram,
    TileValue,
    TileValueType,
)
from .request import (
    CostModelRequest,
    ProfileMode,
    ProfileSelection,
    ProfileSnapshotRef,
    SearchSpace,
    SolverOptions,
    TimingStatistic,
    WarpConfig,
    WarpRole,
    WarpRoleAssignment,
)
from .result import (
    BufferAllocation,
    CostModelPlan,
    CostModelResult,
    Diagnostic,
    DiagnosticCode,
    EvaluationStatus,
    LoopTiming,
    PhasePlacement,
    ProfileProvenance,
    RejectedCandidate,
    ResourceReservation,
    ResourceUtilization,
    SelectedConfiguration,
    SelectedImplementation,
    SelectedStaticDemand,
    SolveProof,
    TimelineRegion,
)
from .solver.model import (
    BufferTemplate,
    Configuration,
    LoopTemplate,
    OpImplementationSelection,
    Phase,
    PhaseDependency,
    PhaseIterationDomain,
    PhaseStartAlignment,
    SearchProblem,
    StaticDemand,
    TemporalDemand,
)
from .tileop import (
    BenchmarkFingerprint,
    CanonicalAttribute,
    TileOpProfileKey,
    TileOpProfileQuery,
    TileOpSignature,
)


def canonical_json(value: object) -> str:
    """Return compact sorted-key JSON with no non-finite numbers."""

    try:
        encoded = _encode(value)
        return json.dumps(
            encoded,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        if isinstance(exc, (InvalidRequestError, WorkloadError, SearchProblemError)):
            raise
        raise InvalidRequestError(f"value is not JSON serializable: {exc}") from exc


def _encode(value: object) -> object:
    if isinstance(value, Enum):
        return _encode(value.value)
    if is_dataclass(value):
        return {field.name: _encode(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        result: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("JSON object keys must be strings")
            result[key] = _encode(item)
        return result
    if isinstance(value, (tuple, list)):
        return [_encode(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return [_encode(item) for item in sorted(value, key=repr)]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bytes):
        return base64.b64encode(value).decode("ascii")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("non-finite floating point value")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"unsupported value type {type(value).__name__}")


def _pairs_no_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field {key!r}")
        result[key] = value
    return result


def _loads(text: str, error_type: type[Exception]) -> object:
    if not isinstance(text, str):
        raise error_type("JSON input must be text")
    try:
        return json.loads(
            text,
            object_pairs_hook=_pairs_no_duplicates,
            parse_float=_finite_json_float,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"invalid JSON constant {value}")
            ),
        )
    except (json.JSONDecodeError, ValueError, TypeError) as exc:
        raise error_type(f"invalid JSON: {exc}") from exc


def _finite_json_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError("non-finite floating point value")
    return parsed


def _object(
    value: object,
    *,
    allowed: set[str],
    required: set[str],
    error_type: type[Exception],
    label: str,
) -> dict[str, object]:
    if not isinstance(value, dict):
        raise error_type(f"{label} must be a JSON object")
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise error_type(f"{label} contains unknown field(s): {', '.join(unknown)}")
    missing = sorted(required - set(value))
    if missing:
        raise error_type(f"{label} is missing required field(s): {', '.join(missing)}")
    return value


def _version(
    data: dict[str, object],
    expected: int,
    *,
    error_type: type[Exception],
    label: str,
) -> None:
    value = data.get("schema_version")
    if isinstance(value, bool) or not isinstance(value, int) or value != expected:
        raise error_type(f"unsupported {label} schema version: {value!r}")


def _sequence(value: object, *, error_type: type[Exception], label: str) -> tuple[object, ...]:
    if not isinstance(value, list):
        raise error_type(f"{label} must be a JSON array")
    return tuple(value)


def _string(value: object, *, error_type: type[Exception], label: str) -> str:
    if not isinstance(value, str):
        raise error_type(f"{label} must be a string")
    return value


def _int(value: object, *, error_type: type[Exception], label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise error_type(f"{label} must be an integer")
    return value


def _optional_int(value: object, *, error_type: type[Exception], label: str) -> int | None:
    if value is None:
        return None
    return _int(value, error_type=error_type, label=label)


def _non_empty_string(value: object, label: str, error_type: type[Exception]) -> str:
    result = _string(value, error_type=error_type, label=label)
    if not result:
        raise error_type(f"{label} must be non-empty")
    return result


def _ascii_identifier_string(value: object, label: str, error_type: type[Exception]) -> str:
    result = _non_empty_string(value, label, error_type)
    try:
        result.encode("ascii")
    except UnicodeEncodeError as exc:
        raise error_type(f"{label} must be a non-empty ASCII string") from exc
    return result


def _decode_enum(
    value: object, enum_type: type[Enum], *, error_type: type[Exception], label: str
) -> Any:
    try:
        return enum_type(value)
    except (TypeError, ValueError) as exc:
        raise error_type(f"invalid {label}: {value!r}") from exc


def _decode_axis(value: object) -> AxisExtent:
    data = _object(
        value,
        allowed={"name", "extent"},
        required={"name", "extent"},
        error_type=WorkloadError,
        label="axis",
    )
    try:
        return AxisExtent(
            _string(data["name"], error_type=WorkloadError, label="axis name"),
            _int(data["extent"], error_type=WorkloadError, label="axis extent"),
        )
    except InvalidRequestError as exc:
        raise WorkloadError(str(exc)) from exc


def _decode_shape(value: object) -> NamedShape:
    data = _object(
        value, allowed={"axes"}, required={"axes"}, error_type=WorkloadError, label="shape"
    )
    try:
        return NamedShape(
            tuple(
                _decode_axis(item)
                for item in _sequence(data["axes"], error_type=WorkloadError, label="shape axes")
            )
        )
    except InvalidRequestError as exc:
        raise WorkloadError(str(exc)) from exc


def _decode_tensor(value: object) -> TensorDescriptor:
    data = _object(
        value,
        allowed={"shape", "dtype", "layout", "strides_elements"},
        required={"shape", "dtype", "layout"},
        error_type=WorkloadError,
        label="tensor",
    )
    strides_value = data.get("strides_elements")
    strides = (
        None
        if strides_value is None
        else tuple(
            _int(item, error_type=WorkloadError, label="stride")
            for item in _sequence(strides_value, error_type=WorkloadError, label="strides")
        )
    )
    try:
        return TensorDescriptor(
            _decode_shape(data["shape"]),
            _decode_enum(data["dtype"], DType, error_type=WorkloadError, label="dtype"),
            _decode_enum(data["layout"], TensorLayout, error_type=WorkloadError, label="layout"),
            strides,
        )
    except InvalidRequestError as exc:
        raise WorkloadError(str(exc)) from exc


def _decode_workload(value: object) -> object:
    data = _object(
        value,
        allowed={
            "kind",
            "m",
            "n",
            "k",
            "dtype_a",
            "dtype_b",
            "dtype_accumulator",
            "dtype_output",
            "layout_a",
            "layout_b",
            "epilogue",
            "batch_size",
            "query_heads",
            "kv_heads",
            "head_dim",
            "context_len",
            "dtype_query",
            "dtype_kv",
            "heads",
            "query_len",
            "key_len",
            "causal",
            "rows",
            "input_dim",
            "hidden_dim",
            "output_dim",
            "dtype_input",
            "dtype_weight",
            "activation",
        },
        required={"kind"},
        error_type=WorkloadError,
        label="workload",
    )
    kind = _decode_enum(data["kind"], WorkloadKind, error_type=WorkloadError, label="workload kind")
    if kind is WorkloadKind.GEMM:
        _reject_workload_extras(
            data,
            {
                "kind",
                "m",
                "n",
                "k",
                "dtype_a",
                "dtype_b",
                "dtype_accumulator",
                "dtype_output",
                "layout_a",
                "layout_b",
                "epilogue",
            },
            "GEMM workload",
        )
        required = {
            "kind",
            "m",
            "n",
            "k",
            "dtype_a",
            "dtype_b",
            "dtype_accumulator",
            "dtype_output",
            "layout_a",
            "layout_b",
        }
        _require_subset(data, required, WorkloadError, "GEMM workload")
        return GemmSpec(
            kind,
            _int(data["m"], error_type=WorkloadError, label="m"),
            _int(data["n"], error_type=WorkloadError, label="n"),
            _int(data["k"], error_type=WorkloadError, label="k"),
            _decode_enum(data["dtype_a"], DType, error_type=WorkloadError, label="dtype_a"),
            _decode_enum(data["dtype_b"], DType, error_type=WorkloadError, label="dtype_b"),
            _decode_enum(
                data["dtype_accumulator"],
                DType,
                error_type=WorkloadError,
                label="dtype_accumulator",
            ),
            _decode_enum(
                data["dtype_output"], DType, error_type=WorkloadError, label="dtype_output"
            ),
            _decode_enum(
                data["layout_a"], TensorLayout, error_type=WorkloadError, label="layout_a"
            ),
            _decode_enum(
                data["layout_b"], TensorLayout, error_type=WorkloadError, label="layout_b"
            ),
            _decode_enum(
                data.get("epilogue", EpilogueKind.NONE.value),
                EpilogueKind,
                error_type=WorkloadError,
                label="epilogue",
            ),
        )
    if kind is WorkloadKind.GQA_DECODE:
        _reject_workload_extras(
            data,
            {
                "kind",
                "batch_size",
                "query_heads",
                "kv_heads",
                "head_dim",
                "context_len",
                "dtype_query",
                "dtype_kv",
                "dtype_accumulator",
                "dtype_output",
            },
            "GQA workload",
        )
        required = {
            "kind",
            "batch_size",
            "query_heads",
            "kv_heads",
            "head_dim",
            "context_len",
            "dtype_query",
            "dtype_kv",
            "dtype_accumulator",
            "dtype_output",
        }
        _require_subset(data, required, WorkloadError, "GQA workload")
        return GqaDecodeSpec(
            kind,
            *(
                _int(data[name], error_type=WorkloadError, label=name)
                for name in ("batch_size", "query_heads", "kv_heads", "head_dim", "context_len")
            ),
            *(
                _decode_enum(data[name], DType, error_type=WorkloadError, label=name)
                for name in ("dtype_query", "dtype_kv", "dtype_accumulator", "dtype_output")
            ),
        )
    if kind is WorkloadKind.FLASH_ATTENTION:
        _reject_workload_extras(
            data,
            {
                "kind",
                "batch_size",
                "heads",
                "query_len",
                "key_len",
                "head_dim",
                "dtype_query",
                "dtype_kv",
                "dtype_accumulator",
                "dtype_output",
                "causal",
            },
            "FlashAttention workload",
        )
        required = {
            "kind",
            "batch_size",
            "heads",
            "query_len",
            "key_len",
            "head_dim",
            "dtype_query",
            "dtype_kv",
            "dtype_accumulator",
            "dtype_output",
            "causal",
        }
        _require_subset(data, required, WorkloadError, "FlashAttention workload")
        if not isinstance(data["causal"], bool):
            raise WorkloadError("causal must be boolean")
        return FlashAttentionSpec(
            kind,
            *(
                _int(data[name], error_type=WorkloadError, label=name)
                for name in ("batch_size", "heads", "query_len", "key_len", "head_dim")
            ),
            *(
                _decode_enum(data[name], DType, error_type=WorkloadError, label=name)
                for name in ("dtype_query", "dtype_kv", "dtype_accumulator", "dtype_output")
            ),
            data["causal"],
        )
    _reject_workload_extras(
        data,
        {
            "kind",
            "rows",
            "input_dim",
            "hidden_dim",
            "output_dim",
            "dtype_input",
            "dtype_weight",
            "dtype_accumulator",
            "dtype_output",
            "activation",
        },
        "MLP workload",
    )
    required = {
        "kind",
        "rows",
        "input_dim",
        "hidden_dim",
        "output_dim",
        "dtype_input",
        "dtype_weight",
        "dtype_accumulator",
        "dtype_output",
        "activation",
    }
    _require_subset(data, required, WorkloadError, "MLP workload")
    return MlpSpec(
        kind,
        *(
            _int(data[name], error_type=WorkloadError, label=name)
            for name in ("rows", "input_dim", "hidden_dim", "output_dim")
        ),
        *(
            _decode_enum(data[name], DType, error_type=WorkloadError, label=name)
            for name in ("dtype_input", "dtype_weight", "dtype_accumulator", "dtype_output")
        ),
        _decode_enum(
            data["activation"], ActivationKind, error_type=WorkloadError, label="activation"
        ),
    )


def _decode_tile_value_type(value: object) -> TileValueType:
    data = _object(
        value,
        allowed={"tensor", "memory_space"},
        required={"tensor", "memory_space"},
        error_type=WorkloadError,
        label="value_type",
    )
    return TileValueType(
        _decode_tensor(data["tensor"]),
        _decode_enum(
            data["memory_space"], MemorySpace, error_type=WorkloadError, label="memory_space"
        ),
    )


def _decode_tile_value(value: object) -> TileValue:
    data = _object(
        value,
        allowed={"value_id", "value_type"},
        required={"value_id", "value_type"},
        error_type=WorkloadError,
        label="value",
    )
    return TileValue(
        _string(data["value_id"], error_type=WorkloadError, label="value_id"),
        _decode_tile_value_type(data["value_type"]),
    )


def _decode_domain(value: object) -> OpIterationDomain:
    data = _object(
        value,
        allowed={"loop_id", "first_iteration", "iteration_count"},
        required={"loop_id", "first_iteration", "iteration_count"},
        error_type=WorkloadError,
        label="operation domain",
    )
    loop_id = (
        None
        if data["loop_id"] is None
        else _string(data["loop_id"], error_type=WorkloadError, label="loop_id")
    )
    return OpIterationDomain(
        loop_id,
        _int(data["first_iteration"], error_type=WorkloadError, label="first_iteration"),
        _int(data["iteration_count"], error_type=WorkloadError, label="iteration_count"),
    )


def _decode_op(value: object) -> object:
    if not isinstance(value, dict):
        raise WorkloadError("operation must be an object")
    kind = _decode_enum(
        value.get("kind"), TileOpKind, error_type=WorkloadError, label="operation kind"
    )
    if kind is TileOpKind.COPY:
        data = _object(
            value,
            allowed={"kind", "op_id", "source", "destination", "domain"},
            required={"kind", "op_id", "source", "destination", "domain"},
            error_type=WorkloadError,
            label="copy operation",
        )
        return CopyOp(
            kind,
            _string(data["op_id"], error_type=WorkloadError, label="op_id"),
            _string(data["source"], error_type=WorkloadError, label="source"),
            _string(data["destination"], error_type=WorkloadError, label="destination"),
            _decode_domain(data["domain"]),
        )
    if kind is TileOpKind.GEMM:
        data = _object(
            value,
            allowed={
                "kind",
                "op_id",
                "lhs",
                "rhs",
                "accumulator",
                "result",
                "m_axis",
                "n_axis",
                "k_axis",
                "domain",
            },
            required={
                "kind",
                "op_id",
                "lhs",
                "rhs",
                "accumulator",
                "result",
                "m_axis",
                "n_axis",
                "k_axis",
                "domain",
            },
            error_type=WorkloadError,
            label="gemm operation",
        )
        return GemmOp(
            kind,
            _string(data["op_id"], error_type=WorkloadError, label="op_id"),
            *(
                _string(data[name], error_type=WorkloadError, label=name)
                for name in ("lhs", "rhs", "accumulator", "result")
            ),
            *(
                _string(data[name], error_type=WorkloadError, label=name)
                for name in ("m_axis", "n_axis", "k_axis")
            ),
            _decode_domain(data["domain"]),
        )
    if kind is TileOpKind.REDUCE:
        data = _object(
            value,
            allowed={"kind", "op_id", "source", "result", "axes", "reduction", "domain"},
            required={"kind", "op_id", "source", "result", "axes", "reduction", "domain"},
            error_type=WorkloadError,
            label="reduce operation",
        )
        return ReduceOp(
            kind,
            _string(data["op_id"], error_type=WorkloadError, label="op_id"),
            _string(data["source"], error_type=WorkloadError, label="source"),
            _string(data["result"], error_type=WorkloadError, label="result"),
            tuple(
                _string(item, error_type=WorkloadError, label="axis")
                for item in _sequence(data["axes"], error_type=WorkloadError, label="axes")
            ),
            _decode_enum(
                data["reduction"], ReductionKind, error_type=WorkloadError, label="reduction"
            ),
            _decode_domain(data["domain"]),
        )
    data = _object(
        value,
        allowed={"kind", "op_id", "inputs", "result", "function", "domain"},
        required={"kind", "op_id", "inputs", "result", "function", "domain"},
        error_type=WorkloadError,
        label="elementwise operation",
    )
    return ElementwiseOp(
        kind,
        _string(data["op_id"], error_type=WorkloadError, label="op_id"),
        tuple(
            _string(item, error_type=WorkloadError, label="input")
            for item in _sequence(data["inputs"], error_type=WorkloadError, label="inputs")
        ),
        _string(data["result"], error_type=WorkloadError, label="result"),
        _decode_enum(data["function"], ElementwiseKind, error_type=WorkloadError, label="function"),
        _decode_domain(data["domain"]),
    )


def _decode_relation(value: object) -> object:
    if not isinstance(value, dict):
        raise WorkloadError("dependency relation must be an object")
    kind = _decode_enum(
        value.get("kind"), DependencyRelationKind, error_type=WorkloadError, label="relation kind"
    )
    if kind is DependencyRelationKind.ALIGNED:
        data = _object(
            value,
            allowed={"kind", "iteration_distance"},
            required={"kind", "iteration_distance"},
            error_type=WorkloadError,
            label="aligned relation",
        )
        return AlignedRelation(
            kind,
            _int(data["iteration_distance"], error_type=WorkloadError, label="iteration_distance"),
        )
    data = _object(
        value,
        allowed={"kind", "src_endpoint", "dst_endpoint"},
        required={"kind", "src_endpoint", "dst_endpoint"},
        error_type=WorkloadError,
        label="endpoint relation",
    )
    return EndpointRelation(
        kind,
        _decode_enum(
            data["src_endpoint"], InstanceEndpoint, error_type=WorkloadError, label="src_endpoint"
        ),
        _decode_enum(
            data["dst_endpoint"], InstanceEndpoint, error_type=WorkloadError, label="dst_endpoint"
        ),
    )


def _decode_program_value(value: object) -> TileProgram:
    data = _object(
        value,
        allowed={
            "schema_version",
            "program_id",
            "workload_kind",
            "tile",
            "values",
            "loops",
            "operations",
            "dependencies",
            "loop_barriers",
            "inputs",
            "outputs",
        },
        required={
            "schema_version",
            "program_id",
            "workload_kind",
            "tile",
            "values",
            "loops",
            "operations",
            "dependencies",
            "loop_barriers",
            "inputs",
            "outputs",
        },
        error_type=WorkloadError,
        label="program",
    )
    _version(data, PROGRAM_SCHEMA_VERSION, error_type=WorkloadError, label="program")
    tile_data = _object(
        data["tile"],
        allowed={"tile_id", "shape"},
        required={"tile_id", "shape"},
        error_type=WorkloadError,
        label="tile",
    )
    tile = TileCandidate(
        _string(tile_data["tile_id"], error_type=WorkloadError, label="tile_id"),
        _decode_shape(tile_data["shape"]),
    )
    loops = tuple(
        TileLoop(
            _string(item["loop_id"], error_type=WorkloadError, label="loop_id"),
            _int(item["iterations"], error_type=WorkloadError, label="iterations"),
        )
        for item in (
            _object(
                raw,
                allowed={"loop_id", "iterations"},
                required={"loop_id", "iterations"},
                error_type=WorkloadError,
                label="loop",
            )
            for raw in _sequence(data["loops"], error_type=WorkloadError, label="loops")
        )
    )
    barriers = tuple(
        LoopBarrier(
            _string(item["barrier_id"], error_type=WorkloadError, label="barrier_id"),
            _string(item["src_loop_id"], error_type=WorkloadError, label="src_loop_id"),
            _string(item["dst_loop_id"], error_type=WorkloadError, label="dst_loop_id"),
        )
        for item in (
            _object(
                raw,
                allowed={"barrier_id", "src_loop_id", "dst_loop_id"},
                required={"barrier_id", "src_loop_id", "dst_loop_id"},
                error_type=WorkloadError,
                label="loop barrier",
            )
            for raw in _sequence(
                data["loop_barriers"], error_type=WorkloadError, label="loop_barriers"
            )
        )
    )
    dependencies = tuple(
        TileDependency(
            _string(item["value_id"], error_type=WorkloadError, label="value_id"),
            _string(item["src_op_id"], error_type=WorkloadError, label="src_op_id"),
            _string(item["dst_op_id"], error_type=WorkloadError, label="dst_op_id"),
            _decode_relation(item["relation"]),
        )
        for item in (
            _object(
                raw,
                allowed={"value_id", "src_op_id", "dst_op_id", "relation"},
                required={"value_id", "src_op_id", "dst_op_id", "relation"},
                error_type=WorkloadError,
                label="dependency",
            )
            for raw in _sequence(
                data["dependencies"], error_type=WorkloadError, label="dependencies"
            )
        )
    )
    return TileProgram(
        _int(data["schema_version"], error_type=WorkloadError, label="schema_version"),
        _string(data["program_id"], error_type=WorkloadError, label="program_id"),
        _decode_enum(
            data["workload_kind"], WorkloadKind, error_type=WorkloadError, label="workload_kind"
        ),
        tile,
        tuple(
            _decode_tile_value(raw)
            for raw in _sequence(data["values"], error_type=WorkloadError, label="values")
        ),
        loops,
        tuple(
            _decode_op(raw)
            for raw in _sequence(data["operations"], error_type=WorkloadError, label="operations")
        ),
        dependencies,
        barriers,
        tuple(
            _string(raw, error_type=WorkloadError, label="input")
            for raw in _sequence(data["inputs"], error_type=WorkloadError, label="inputs")
        ),
        tuple(
            _string(raw, error_type=WorkloadError, label="output")
            for raw in _sequence(data["outputs"], error_type=WorkloadError, label="outputs")
        ),
    )


def _decode_hardware_ref(
    value: object,
    *,
    error_type: type[Exception] = InvalidRequestError,
    label: str = "hardware reference",
) -> HardwareSpecRef:
    data = _object(
        value,
        allowed={"hardware_id", "schema_version", "calibration_id"},
        required={"hardware_id", "schema_version", "calibration_id"},
        error_type=error_type,
        label=label,
    )
    try:
        return HardwareSpecRef(
            _string(data["hardware_id"], error_type=error_type, label="hardware_id"),
            _int(data["schema_version"], error_type=error_type, label="schema_version"),
            _string(data["calibration_id"], error_type=error_type, label="calibration_id"),
        )
    except InvalidRequestError as exc:
        if error_type is InvalidRequestError:
            raise
        raise error_type(str(exc)) from exc


def _decode_hardware_spec(
    value: object,
    *,
    error_type: type[Exception] = SearchProblemError,
) -> HardwareSpec:
    data = _object(
        value,
        allowed={
            "schema_version",
            "ref",
            "architecture",
            "temporal_resources",
            "static_resources",
            "supported_dtypes",
            "supported_implementation_ids",
        },
        required={
            "schema_version",
            "ref",
            "architecture",
            "temporal_resources",
            "static_resources",
            "supported_dtypes",
            "supported_implementation_ids",
        },
        error_type=error_type,
        label="hardware",
    )
    _version(data, HARDWARE_SCHEMA_VERSION, error_type=error_type, label="hardware")
    temporal_resources = tuple(
        _decode_temporal_resource(item, error_type=error_type)
        for item in _sequence(
            data["temporal_resources"], error_type=error_type, label="temporal_resources"
        )
    )
    static_resources = tuple(
        _decode_static_resource(item, error_type=error_type)
        for item in _sequence(
            data["static_resources"], error_type=error_type, label="static_resources"
        )
    )
    try:
        return HardwareSpec(
            _decode_hardware_ref(data["ref"], error_type=error_type),
            _string(data["architecture"], error_type=error_type, label="architecture"),
            temporal_resources,
            static_resources,
            tuple(
                _decode_enum(item, DType, error_type=error_type, label="dtype")
                for item in _sequence(
                    data["supported_dtypes"], error_type=error_type, label="supported_dtypes"
                )
            ),
            tuple(
                _string(item, error_type=error_type, label="implementation_id")
                for item in _sequence(
                    data["supported_implementation_ids"],
                    error_type=error_type,
                    label="supported_implementation_ids",
                )
            ),
            _int(data["schema_version"], error_type=error_type, label="schema_version"),
        )
    except HardwareSpecError as exc:
        if error_type is HardwareSpecError:
            raise
        raise error_type(str(exc)) from exc


def _decode_fact_provenance(value: object, *, error_type: type[Exception]) -> FactProvenance:
    data = _object(
        value,
        allowed={"origin", "source", "conditions"},
        required={"origin", "source", "conditions"},
        error_type=error_type,
        label="hardware provenance",
    )
    try:
        return FactProvenance(
            _decode_enum(data["origin"], FactOrigin, error_type=error_type, label="origin"),
            _string(data["source"], error_type=error_type, label="source"),
            _string(data["conditions"], error_type=error_type, label="conditions"),
        )
    except HardwareSpecError as exc:
        if error_type is HardwareSpecError:
            raise
        raise error_type(str(exc)) from exc


def _decode_temporal_resource(
    value: object, *, error_type: type[Exception]
) -> TemporalResourceSpec:
    data = _object(
        value,
        allowed={"resource_id", "capacity_slots", "description", "provenance"},
        required={"resource_id", "capacity_slots", "description", "provenance"},
        error_type=error_type,
        label="temporal resource",
    )
    try:
        return TemporalResourceSpec(
            _string(data["resource_id"], error_type=error_type, label="resource_id"),
            _int(data["capacity_slots"], error_type=error_type, label="capacity_slots"),
            _string(data["description"], error_type=error_type, label="description"),
            _decode_fact_provenance(data["provenance"], error_type=error_type),
        )
    except HardwareSpecError as exc:
        if error_type is HardwareSpecError:
            raise
        raise error_type(str(exc)) from exc


def _decode_static_resource(value: object, *, error_type: type[Exception]) -> StaticResourceSpec:
    data = _object(
        value,
        allowed={"resource_id", "capacity_units", "unit", "description", "provenance"},
        required={"resource_id", "capacity_units", "unit", "description", "provenance"},
        error_type=error_type,
        label="static resource",
    )
    try:
        return StaticResourceSpec(
            _string(data["resource_id"], error_type=error_type, label="resource_id"),
            _int(data["capacity_units"], error_type=error_type, label="capacity_units"),
            _decode_enum(data["unit"], StaticUnit, error_type=error_type, label="unit"),
            _string(data["description"], error_type=error_type, label="description"),
            _decode_fact_provenance(data["provenance"], error_type=error_type),
        )
    except HardwareSpecError as exc:
        if error_type is HardwareSpecError:
            raise
        raise error_type(str(exc)) from exc


def _decode_tile_candidate(value: object) -> TileCandidate | object:
    if not isinstance(value, dict):
        return value
    data = _object(
        value,
        allowed={"tile_id", "shape"},
        required={"tile_id", "shape"},
        error_type=InvalidRequestError,
        label="tile",
    )
    try:
        shape = _decode_shape(data["shape"])
    except WorkloadError as exc:
        raise InvalidRequestError(str(exc)) from exc
    return TileCandidate(
        _string(data["tile_id"], error_type=InvalidRequestError, label="tile_id"), shape
    )


def _decode_warp_config(
    value: object, *, error_type: type[Exception] = InvalidRequestError
) -> WarpConfig:
    data = _object(
        value,
        allowed={"config_id", "total_warps", "roles"},
        required={"config_id", "total_warps", "roles"},
        error_type=error_type,
        label="warp config",
    )
    roles = []
    for raw in _sequence(data["roles"], error_type=error_type, label="roles"):
        item = _object(
            raw,
            allowed={"role", "warp_ids"},
            required={"role", "warp_ids"},
            error_type=error_type,
            label="warp role",
        )
        try:
            roles.append(
                WarpRoleAssignment(
                    _decode_enum(item["role"], WarpRole, error_type=error_type, label="role"),
                    tuple(
                        _int(warp, error_type=error_type, label="warp_id")
                        for warp in _sequence(
                            item["warp_ids"], error_type=error_type, label="warp_ids"
                        )
                    ),
                )
            )
        except InvalidRequestError as exc:
            if error_type is InvalidRequestError:
                raise
            raise error_type(str(exc)) from exc
    try:
        return WarpConfig(
            _string(data["config_id"], error_type=error_type, label="config_id"),
            _int(data["total_warps"], error_type=error_type, label="total_warps"),
            tuple(roles),
        )
    except InvalidRequestError as exc:
        if error_type is InvalidRequestError:
            raise
        raise error_type(str(exc)) from exc


def _decode_search_space(value: object) -> SearchSpace:
    data = _object(
        value,
        allowed={
            "implementation_ids",
            "warp_configs",
            "pipeline_depths",
            "layout_variant_ids",
            "max_candidates",
        },
        required={"implementation_ids", "warp_configs", "pipeline_depths"},
        error_type=InvalidRequestError,
        label="search_space",
    )
    return SearchSpace(
        tuple(
            _string(item, error_type=InvalidRequestError, label="implementation_id")
            for item in _sequence(
                data["implementation_ids"],
                error_type=InvalidRequestError,
                label="implementation_ids",
            )
        ),
        tuple(
            _decode_warp_config(item)
            for item in _sequence(
                data["warp_configs"], error_type=InvalidRequestError, label="warp_configs"
            )
        ),
        tuple(
            _int(item, error_type=InvalidRequestError, label="pipeline_depth")
            for item in _sequence(
                data["pipeline_depths"], error_type=InvalidRequestError, label="pipeline_depths"
            )
        ),
        tuple(
            _string(item, error_type=InvalidRequestError, label="layout_variant_id")
            for item in _sequence(
                data.get("layout_variant_ids", []),
                error_type=InvalidRequestError,
                label="layout_variant_ids",
            )
        ),
        _int(
            data.get("max_candidates", 10_000),
            error_type=InvalidRequestError,
            label="max_candidates",
        ),
    )


def _decode_profiles(value: object) -> ProfileSelection:
    data = _object(
        value,
        allowed={"snapshot", "mode", "timing_statistic"},
        required={"snapshot"},
        error_type=InvalidRequestError,
        label="profiles",
    )
    snapshot_data = _object(
        data["snapshot"],
        allowed={"snapshot_id", "revision"},
        required={"snapshot_id", "revision"},
        error_type=InvalidRequestError,
        label="profile snapshot",
    )
    return ProfileSelection(
        ProfileSnapshotRef(
            _string(
                snapshot_data["snapshot_id"], error_type=InvalidRequestError, label="snapshot_id"
            ),
            _int(snapshot_data["revision"], error_type=InvalidRequestError, label="revision"),
        ),
        _decode_enum(
            data.get("mode", ProfileMode.REQUIRE.value),
            ProfileMode,
            error_type=InvalidRequestError,
            label="mode",
        ),
        _decode_enum(
            data.get("timing_statistic", TimingStatistic.P50.value),
            TimingStatistic,
            error_type=InvalidRequestError,
            label="timing_statistic",
        ),
    )


def _decode_solver_options(value: object) -> SolverOptions:
    data = _object(
        value,
        allowed={
            "candidate_timeout_s",
            "search_timeout_s",
            "time_resolution_ps",
            "ortools_workers",
            "candidate_workers",
            "random_seed",
            "finite_unroll_limit",
            "stop_after_first_solution",
            "deterministic",
        },
        required=set(),
        error_type=InvalidRequestError,
        label="solver",
    )
    return SolverOptions(
        **{
            name: data[name]
            for name in (
                "candidate_timeout_s",
                "search_timeout_s",
                "time_resolution_ps",
                "ortools_workers",
                "candidate_workers",
                "random_seed",
                "finite_unroll_limit",
                "stop_after_first_solution",
                "deterministic",
            )
            if name in data
        }
    )


def request_from_json(text: str) -> CostModelRequest:
    data = _object(
        _loads(text, InvalidRequestError),
        allowed={
            "schema_version",
            "request_id",
            "workload",
            "programs",
            "hardware",
            "search_space",
            "profiles",
            "solver",
        },
        required={
            "schema_version",
            "request_id",
            "workload",
            "programs",
            "hardware",
            "search_space",
            "profiles",
        },
        error_type=InvalidRequestError,
        label="request",
    )
    _version(data, REQUEST_SCHEMA_VERSION, error_type=InvalidRequestError, label="request")
    return CostModelRequest(
        _int(data["schema_version"], error_type=InvalidRequestError, label="schema_version"),
        _string(data["request_id"], error_type=InvalidRequestError, label="request_id"),
        _decode_workload(data["workload"]),
        tuple(
            _decode_program_value(item)
            for item in _sequence(
                data["programs"], error_type=InvalidRequestError, label="programs"
            )
        ),
        _decode_hardware_ref(data["hardware"]),
        _decode_search_space(data["search_space"]),
        _decode_profiles(data["profiles"]),
        _decode_solver_options(data.get("solver", {})),
    )


def request_to_json(request: CostModelRequest) -> str:
    if type(request) is not CostModelRequest:
        raise InvalidRequestError("request must be CostModelRequest")
    _assert_version(request, REQUEST_SCHEMA_VERSION, "request")
    payload = _encode(request)
    text = canonical_json(payload)
    request_from_json(text)
    return text


def _decode_profile_snapshot(value: object) -> ProfileSnapshot:
    data = _object(
        value,
        allowed={"schema_version", "snapshot_id", "revision", "hardware", "measurements"},
        required={"schema_version", "snapshot_id", "revision", "hardware", "measurements"},
        error_type=ProfileStoreError,
        label="profile snapshot",
    )
    _version(data, PROFILE_SCHEMA_VERSION, error_type=ProfileStoreError, label="profile snapshot")
    snapshot = ProfileSnapshot(
        _int(data["schema_version"], error_type=ProfileStoreError, label="schema_version"),
        _string(data["snapshot_id"], error_type=ProfileStoreError, label="snapshot_id"),
        _int(data["revision"], error_type=ProfileStoreError, label="revision"),
        _decode_hardware_spec(data["hardware"], error_type=ProfileStoreError),
        tuple(
            _decode_profile_measurement(item)
            for item in _sequence(
                data["measurements"], error_type=ProfileStoreError, label="measurements"
            )
        ),
    )
    return snapshot


def _decode_profile_measurement(value: object) -> ProfileMeasurement:
    fields_allowed = {
        "measurement_id",
        "key",
        "environment",
        "origin",
        "latency_p50_ps",
        "latency_p90_ps",
        "initiation_interval_p50_ps",
        "initiation_interval_p90_ps",
        "warmup_runs",
        "sample_count",
        "latency_repetitions_per_sample",
        "initiation_interval_repetitions_per_sample",
        "target_sample_ns",
        "relative_iqr_ppm",
        "raw_samples_retained",
        "raw_latency_samples_ps",
        "raw_initiation_interval_samples_ps",
        "measured_at_utc",
    }
    data = _object(
        value,
        allowed=fields_allowed,
        required=fields_allowed,
        error_type=ProfileStoreError,
        label="profile measurement",
    )
    key = _decode_profile_key(data["key"])
    environment = _decode_profile_environment(data["environment"])
    raw_retained = data["raw_samples_retained"]
    if not isinstance(raw_retained, bool):
        raise ProfileStoreError("raw_samples_retained must be boolean")
    return ProfileMeasurement(
        _string(data["measurement_id"], error_type=ProfileStoreError, label="measurement_id"),
        key,
        environment,
        _decode_enum(
            data["origin"],
            MeasurementOrigin,
            error_type=ProfileStoreError,
            label="measurement origin",
        ),
        _int(data["latency_p50_ps"], error_type=ProfileStoreError, label="latency_p50_ps"),
        _int(data["latency_p90_ps"], error_type=ProfileStoreError, label="latency_p90_ps"),
        _optional_int(
            data["initiation_interval_p50_ps"],
            error_type=ProfileStoreError,
            label="initiation_interval_p50_ps",
        ),
        _optional_int(
            data["initiation_interval_p90_ps"],
            error_type=ProfileStoreError,
            label="initiation_interval_p90_ps",
        ),
        _int(data["warmup_runs"], error_type=ProfileStoreError, label="warmup_runs"),
        _int(data["sample_count"], error_type=ProfileStoreError, label="sample_count"),
        _int(
            data["latency_repetitions_per_sample"],
            error_type=ProfileStoreError,
            label="latency_repetitions_per_sample",
        ),
        _optional_int(
            data["initiation_interval_repetitions_per_sample"],
            error_type=ProfileStoreError,
            label="initiation_interval_repetitions_per_sample",
        ),
        _int(data["target_sample_ns"], error_type=ProfileStoreError, label="target_sample_ns"),
        _int(data["relative_iqr_ppm"], error_type=ProfileStoreError, label="relative_iqr_ppm"),
        raw_retained,
        tuple(
            _int(item, error_type=ProfileStoreError, label="raw latency sample")
            for item in _sequence(
                data["raw_latency_samples_ps"],
                error_type=ProfileStoreError,
                label="raw_latency_samples_ps",
            )
        ),
        tuple(
            _int(item, error_type=ProfileStoreError, label="raw initiation-interval sample")
            for item in _sequence(
                data["raw_initiation_interval_samples_ps"],
                error_type=ProfileStoreError,
                label="raw_initiation_interval_samples_ps",
            )
        ),
        _string(data["measured_at_utc"], error_type=ProfileStoreError, label="measured_at_utc"),
    )


def _decode_profile_key(value: object) -> TileOpProfileKey:
    data = _object(
        value,
        allowed={"schema_version", "query", "fingerprint"},
        required={"schema_version", "query", "fingerprint"},
        error_type=ProfileStoreError,
        label="profile key",
    )
    _version(data, PROFILE_SCHEMA_VERSION, error_type=ProfileStoreError, label="profile key")
    query = _object(
        data["query"],
        allowed={
            "hardware",
            "operation",
            "implementation_id",
            "component_id",
            "tile_shape",
            "warp_config_id",
            "pipeline_depth",
            "layout_variant_id",
            "conditions",
        },
        required={
            "hardware",
            "operation",
            "implementation_id",
            "component_id",
            "tile_shape",
            "warp_config_id",
            "pipeline_depth",
            "layout_variant_id",
            "conditions",
        },
        error_type=ProfileStoreError,
        label="profile query",
    )
    operation = _object(
        query["operation"],
        allowed={"op_kind", "operands", "results", "semantic_attributes"},
        required={"op_kind", "operands", "results", "semantic_attributes"},
        error_type=ProfileStoreError,
        label="operation signature",
    )
    try:
        signature = TileOpSignature(
            _decode_enum(
                operation["op_kind"],
                TileOpKind,
                error_type=ProfileStoreError,
                label="op_kind",
            ),
            tuple(
                _decode_tile_value_type(item)
                for item in _sequence(
                    operation["operands"],
                    error_type=ProfileStoreError,
                    label="operands",
                )
            ),
            tuple(
                _decode_tile_value_type(item)
                for item in _sequence(
                    operation["results"],
                    error_type=ProfileStoreError,
                    label="results",
                )
            ),
            _decode_canonical_attributes(operation["semantic_attributes"], "semantic_attributes"),
        )
    except (InvalidRequestError, WorkloadError) as exc:
        raise ProfileStoreError(str(exc)) from exc
    for field_name in (
        "implementation_id",
        "component_id",
        "warp_config_id",
        "layout_variant_id",
    ):
        _ascii_identifier_string(query[field_name], field_name, ProfileStoreError)
    pipeline_depth = _int(
        query["pipeline_depth"], error_type=ProfileStoreError, label="pipeline_depth"
    )
    if pipeline_depth <= 0:
        raise ProfileStoreError("pipeline_depth must be positive")
    try:
        typed_query = TileOpProfileQuery(
            _decode_hardware_ref(query["hardware"], error_type=ProfileStoreError),
            signature,
            _string(
                query["implementation_id"],
                error_type=ProfileStoreError,
                label="implementation_id",
            ),
            _string(query["component_id"], error_type=ProfileStoreError, label="component_id"),
            _decode_shape(query["tile_shape"]),
            _string(
                query["warp_config_id"],
                error_type=ProfileStoreError,
                label="warp_config_id",
            ),
            pipeline_depth,
            _string(
                query["layout_variant_id"],
                error_type=ProfileStoreError,
                label="layout_variant_id",
            ),
            _decode_canonical_attributes(query["conditions"], "conditions"),
        )
    except (InvalidRequestError, WorkloadError) as exc:
        raise ProfileStoreError(str(exc)) from exc

    fingerprint = _object(
        data["fingerprint"],
        allowed={
            "provider_id",
            "provider_version",
            "benchmark_abi_version",
            "source_sha256",
            "compile_options_sha256",
        },
        required={
            "provider_id",
            "provider_version",
            "benchmark_abi_version",
            "source_sha256",
            "compile_options_sha256",
        },
        error_type=ProfileStoreError,
        label="benchmark fingerprint",
    )
    typed_fingerprint = BenchmarkFingerprint(
        _string(fingerprint["provider_id"], error_type=ProfileStoreError, label="provider_id"),
        _string(
            fingerprint["provider_version"],
            error_type=ProfileStoreError,
            label="provider_version",
        ),
        _int(
            fingerprint["benchmark_abi_version"],
            error_type=ProfileStoreError,
            label="benchmark_abi_version",
        ),
        _string(
            fingerprint["source_sha256"],
            error_type=ProfileStoreError,
            label="source_sha256",
        ),
        _string(
            fingerprint["compile_options_sha256"],
            error_type=ProfileStoreError,
            label="compile_options_sha256",
        ),
    )
    return TileOpProfileKey(
        _int(data["schema_version"], error_type=ProfileStoreError, label="schema_version"),
        typed_query,
        typed_fingerprint,
    )


def _decode_profile_environment(value: object) -> ProfileEnvironment:
    fields_allowed = {
        "environment_id",
        "device_uuid",
        "hardware",
        "cuda_arch",
        "driver_version",
        "runtime_version",
        "nvrtc_version",
        "device_clock_khz",
        "memory_clock_khz",
        "power_limit_mw",
    }
    data = _object(
        value,
        allowed=fields_allowed,
        required=fields_allowed,
        error_type=ProfileStoreError,
        label="profile environment",
    )
    for field_name in (
        "environment_id",
        "device_uuid",
        "cuda_arch",
        "driver_version",
        "runtime_version",
        "nvrtc_version",
    ):
        _ascii_identifier_string(data[field_name], field_name, ProfileStoreError)
    hardware = _decode_hardware_ref(data["hardware"], error_type=ProfileStoreError)
    for field_name in ("device_clock_khz", "memory_clock_khz", "power_limit_mw"):
        quantity = _optional_int(data[field_name], error_type=ProfileStoreError, label=field_name)
        if quantity is not None and quantity <= 0:
            raise ProfileStoreError(f"{field_name} must be positive or null")
    try:
        return ProfileEnvironment(
            _string(data["environment_id"], error_type=ProfileStoreError, label="environment_id"),
            _string(data["device_uuid"], error_type=ProfileStoreError, label="device_uuid"),
            hardware,
            _string(data["cuda_arch"], error_type=ProfileStoreError, label="cuda_arch"),
            _string(data["driver_version"], error_type=ProfileStoreError, label="driver_version"),
            _string(data["runtime_version"], error_type=ProfileStoreError, label="runtime_version"),
            _string(data["nvrtc_version"], error_type=ProfileStoreError, label="nvrtc_version"),
            _optional_int(
                data["device_clock_khz"],
                error_type=ProfileStoreError,
                label="device_clock_khz",
            ),
            _optional_int(
                data["memory_clock_khz"],
                error_type=ProfileStoreError,
                label="memory_clock_khz",
            ),
            _optional_int(
                data["power_limit_mw"],
                error_type=ProfileStoreError,
                label="power_limit_mw",
            ),
        )
    except (InvalidRequestError, ProfileStoreError) as exc:
        raise ProfileStoreError(str(exc)) from exc


def _decode_canonical_attributes(value: object, label: str) -> tuple[CanonicalAttribute, ...]:
    names: list[str] = []
    attributes: list[tuple[str, str]] = []
    for raw in _sequence(value, error_type=ProfileStoreError, label=label):
        data = _object(
            raw,
            allowed={"name", "value"},
            required={"name", "value"},
            error_type=ProfileStoreError,
            label="canonical attribute",
        )
        name = _ascii_identifier_string(data["name"], "attribute name", ProfileStoreError)
        attribute_value = _string(
            data["value"], error_type=ProfileStoreError, label="attribute value"
        )
        names.append(name)
        attributes.append((name, attribute_value))
    if len(names) != len(set(names)):
        raise ProfileStoreError(f"{label} names must be unique")
    if attributes != sorted(attributes):
        raise ProfileStoreError(f"{label} must be sorted by name and value")
    return tuple(CanonicalAttribute(name, attribute_value) for name, attribute_value in attributes)


def hardware_from_json(text: str) -> HardwareSpec:
    """Parse a strict immutable hardware-v1 document."""

    return _decode_hardware_spec(_loads(text, HardwareSpecError), error_type=HardwareSpecError)


def hardware_to_json(document: Mapping[str, object] | object) -> str:
    """Serialize a strict hardware-v1 document deterministically."""

    if type(document) is HardwareSpec:
        # Re-enter the decoder even for an already typed value.  This keeps the
        # serializer boundary strict if a caller forged or mutated a frozen
        # record through a subclass or ``object.__setattr__``.
        spec = _decode_hardware_spec(_encode(document), error_type=HardwareSpecError)
    elif isinstance(document, HardwareSpec):
        raise HardwareSpecError("hardware document must be HardwareSpec")
    else:
        try:
            encoded = _document_mapping(document)
        except (TypeError, ValueError) as exc:
            raise HardwareSpecError(f"invalid hardware document: {exc}") from exc
        spec = _decode_hardware_spec(encoded, error_type=HardwareSpecError)
    return canonical_json(spec)


def profile_snapshot_from_json(text: str) -> ProfileSnapshot:
    """Parse a strict profile-snapshot-v1 document."""

    return _decode_profile_snapshot(_loads(text, ProfileStoreError))


def profile_snapshot_to_json(document: Mapping[str, object] | object) -> str:
    """Serialize a strict profile-snapshot-v1 document deterministically."""

    if type(document) is ProfileSnapshot:
        snapshot = _decode_profile_snapshot(_encode(document))
    elif isinstance(document, ProfileSnapshot):
        raise ProfileStoreError("profile snapshot must be ProfileSnapshot")
    else:
        try:
            encoded = _document_mapping(document)
        except (TypeError, ValueError) as exc:
            raise ProfileStoreError(f"invalid profile snapshot: {exc}") from exc
        snapshot = _decode_profile_snapshot(encoded)
    return canonical_json(snapshot)


def program_from_json(text: str) -> TileProgram:
    return _decode_program_value(_loads(text, WorkloadError))


def program_to_json(program: TileProgram) -> str:
    if type(program) is not TileProgram:
        raise WorkloadError("program must be TileProgram")
    _assert_version(program, PROGRAM_SCHEMA_VERSION, "program")
    payload = _encode(program)
    _decode_program_value(payload)
    return canonical_json(payload)


def _decode_phase_domain(value: object) -> PhaseIterationDomain:
    data = _object(
        value,
        allowed={"loop_id", "first_iteration", "iteration_count"},
        required={"loop_id", "first_iteration", "iteration_count"},
        error_type=SearchProblemError,
        label="phase domain",
    )
    loop_id = (
        None
        if data["loop_id"] is None
        else _string(data["loop_id"], error_type=SearchProblemError, label="loop_id")
    )
    return PhaseIterationDomain(
        loop_id,
        _int(data["first_iteration"], error_type=SearchProblemError, label="first_iteration"),
        _int(data["iteration_count"], error_type=SearchProblemError, label="iteration_count"),
    )


def _decode_temporal_demand(value: object) -> TemporalDemand:
    data = _object(
        value,
        allowed={"resource_id", "slots"},
        required={"resource_id", "slots"},
        error_type=SearchProblemError,
        label="temporal demand",
    )
    return TemporalDemand(
        _string(data["resource_id"], error_type=SearchProblemError, label="resource_id"),
        _int(data["slots"], error_type=SearchProblemError, label="slots"),
    )


def _decode_static_demand(value: object) -> StaticDemand:
    data = _object(
        value,
        allowed={"resource_id", "units"},
        required={"resource_id", "units"},
        error_type=SearchProblemError,
        label="static demand",
    )
    return StaticDemand(
        _string(data["resource_id"], error_type=SearchProblemError, label="resource_id"),
        _int(data["units"], error_type=SearchProblemError, label="units"),
    )


def _decode_configuration_phase(value: object) -> Phase:
    allowed = {
        "phase_id",
        "source_op_id",
        "implementation_id",
        "phase_name",
        "component_id",
        "domain",
        "duration_ticks",
        "sensitivity_duration_ticks",
        "warp_ids",
        "temporal_demands",
        "measurement_id",
        "profile_key_id",
        "environment_id",
        "timing_metric",
        "timing_statistic",
        "sensitivity_timing_statistic",
    }
    data = _object(
        value,
        allowed=allowed,
        required=allowed,
        error_type=SearchProblemError,
        label="configuration phase",
    )
    return Phase(
        _string(data["phase_id"], error_type=SearchProblemError, label="phase_id"),
        _string(data["source_op_id"], error_type=SearchProblemError, label="source_op_id"),
        _string(
            data["implementation_id"], error_type=SearchProblemError, label="implementation_id"
        ),
        _string(data["phase_name"], error_type=SearchProblemError, label="phase_name"),
        _string(data["component_id"], error_type=SearchProblemError, label="component_id"),
        _decode_phase_domain(data["domain"]),
        _int(data["duration_ticks"], error_type=SearchProblemError, label="duration_ticks"),
        _int(
            data["sensitivity_duration_ticks"],
            error_type=SearchProblemError,
            label="sensitivity_duration_ticks",
        ),
        tuple(
            _int(item, error_type=SearchProblemError, label="warp_id")
            for item in _sequence(data["warp_ids"], error_type=SearchProblemError, label="warp_ids")
        ),
        tuple(
            _decode_temporal_demand(item)
            for item in _sequence(
                data["temporal_demands"], error_type=SearchProblemError, label="temporal_demands"
            )
        ),
        _string(data["measurement_id"], error_type=SearchProblemError, label="measurement_id"),
        _string(data["profile_key_id"], error_type=SearchProblemError, label="profile_key_id"),
        _string(data["environment_id"], error_type=SearchProblemError, label="environment_id"),
        _decode_enum(
            data["timing_metric"],
            TimingMetric,
            error_type=SearchProblemError,
            label="timing_metric",
        ),
        _decode_enum(
            data["timing_statistic"],
            TimingStatistic,
            error_type=SearchProblemError,
            label="timing_statistic",
        ),
        _decode_enum(
            data["sensitivity_timing_statistic"],
            TimingStatistic,
            error_type=SearchProblemError,
            label="sensitivity_timing_statistic",
        ),
    )


def _decode_configuration_dependency(value: object) -> PhaseDependency:
    data = _object(
        value,
        allowed={"src_phase_id", "dst_phase_id", "relation", "delay_ps"},
        required={"src_phase_id", "dst_phase_id", "relation"},
        error_type=SearchProblemError,
        label="phase dependency",
    )
    try:
        relation = _decode_relation(data["relation"])
    except WorkloadError as exc:
        raise SearchProblemError(str(exc)) from exc
    return PhaseDependency(
        _string(data["src_phase_id"], error_type=SearchProblemError, label="src_phase_id"),
        _string(data["dst_phase_id"], error_type=SearchProblemError, label="dst_phase_id"),
        relation,
        _int(data.get("delay_ps", 0), error_type=SearchProblemError, label="delay_ps"),
    )


def _decode_configuration_alignment(value: object) -> PhaseStartAlignment:
    data = _object(
        value,
        allowed={"src_phase_id", "dst_phase_id", "offset_ps"},
        required={"src_phase_id", "dst_phase_id"},
        error_type=SearchProblemError,
        label="phase start alignment",
    )
    return PhaseStartAlignment(
        _string(data["src_phase_id"], error_type=SearchProblemError, label="src_phase_id"),
        _string(data["dst_phase_id"], error_type=SearchProblemError, label="dst_phase_id"),
        _int(data.get("offset_ps", 0), error_type=SearchProblemError, label="offset_ps"),
    )


def _decode_configuration(value: object) -> Configuration:
    allowed = {
        "configuration_id",
        "program_id",
        "workload_kind",
        "tile",
        "implementations",
        "warps",
        "pipeline_depth",
        "layout_variant_id",
        "loops",
        "phases",
        "dependencies",
        "loop_barriers",
        "start_alignments",
        "buffers",
        "static_demands",
    }
    data = _object(
        value,
        allowed=allowed,
        required=allowed,
        error_type=SearchProblemError,
        label="configuration",
    )
    tile_data = _object(
        data["tile"],
        allowed={"tile_id", "shape"},
        required={"tile_id", "shape"},
        error_type=SearchProblemError,
        label="configuration tile",
    )
    try:
        tile = TileCandidate(
            _string(tile_data["tile_id"], error_type=SearchProblemError, label="tile_id"),
            _decode_shape(tile_data["shape"]),
        )
    except (InvalidRequestError, WorkloadError) as exc:
        raise SearchProblemError(str(exc)) from exc

    implementations = tuple(
        _decode_configuration_implementation(item)
        for item in _sequence(
            data["implementations"], error_type=SearchProblemError, label="implementations"
        )
    )
    loops = tuple(
        _decode_configuration_loop(item)
        for item in _sequence(data["loops"], error_type=SearchProblemError, label="loops")
    )
    phases = tuple(
        _decode_configuration_phase(item)
        for item in _sequence(data["phases"], error_type=SearchProblemError, label="phases")
    )
    dependencies = tuple(
        _decode_configuration_dependency(item)
        for item in _sequence(
            data["dependencies"], error_type=SearchProblemError, label="dependencies"
        )
    )
    barriers = tuple(
        _decode_configuration_barrier(item)
        for item in _sequence(
            data["loop_barriers"], error_type=SearchProblemError, label="loop_barriers"
        )
    )
    alignments = tuple(
        _decode_configuration_alignment(item)
        for item in _sequence(
            data["start_alignments"], error_type=SearchProblemError, label="start_alignments"
        )
    )
    buffers = tuple(
        _decode_configuration_buffer(item)
        for item in _sequence(data["buffers"], error_type=SearchProblemError, label="buffers")
    )
    static_demands = tuple(
        _decode_static_demand(item)
        for item in _sequence(
            data["static_demands"], error_type=SearchProblemError, label="static_demands"
        )
    )
    try:
        return Configuration(
            _string(
                data["configuration_id"],
                error_type=SearchProblemError,
                label="configuration_id",
            ),
            _string(data["program_id"], error_type=SearchProblemError, label="program_id"),
            _decode_enum(
                data["workload_kind"],
                WorkloadKind,
                error_type=SearchProblemError,
                label="workload_kind",
            ),
            tile,
            implementations,
            _decode_warp_config(data["warps"], error_type=SearchProblemError),
            _int(data["pipeline_depth"], error_type=SearchProblemError, label="pipeline_depth"),
            _string(
                data["layout_variant_id"],
                error_type=SearchProblemError,
                label="layout_variant_id",
            ),
            loops,
            phases,
            dependencies,
            barriers,
            alignments,
            buffers,
            static_demands,
        )
    except (InvalidRequestError, WorkloadError) as exc:
        raise SearchProblemError(str(exc)) from exc


def _decode_configuration_implementation(value: object) -> OpImplementationSelection:
    data = _object(
        value,
        allowed={"op_id", "implementation_id"},
        required={"op_id", "implementation_id"},
        error_type=SearchProblemError,
        label="implementation selection",
    )
    return OpImplementationSelection(
        _string(data["op_id"], error_type=SearchProblemError, label="op_id"),
        _string(
            data["implementation_id"], error_type=SearchProblemError, label="implementation_id"
        ),
    )


def _decode_configuration_loop(value: object) -> LoopTemplate:
    data = _object(
        value,
        allowed={"loop_id", "iterations"},
        required={"loop_id", "iterations"},
        error_type=SearchProblemError,
        label="configuration loop",
    )
    return LoopTemplate(
        _string(data["loop_id"], error_type=SearchProblemError, label="loop_id"),
        _int(data["iterations"], error_type=SearchProblemError, label="iterations"),
    )


def _decode_configuration_barrier(value: object) -> LoopBarrier:
    data = _object(
        value,
        allowed={"barrier_id", "src_loop_id", "dst_loop_id"},
        required={"barrier_id", "src_loop_id", "dst_loop_id"},
        error_type=SearchProblemError,
        label="configuration loop barrier",
    )
    try:
        return LoopBarrier(
            _string(data["barrier_id"], error_type=SearchProblemError, label="barrier_id"),
            _string(data["src_loop_id"], error_type=SearchProblemError, label="src_loop_id"),
            _string(data["dst_loop_id"], error_type=SearchProblemError, label="dst_loop_id"),
        )
    except (InvalidRequestError, WorkloadError) as exc:
        raise SearchProblemError(str(exc)) from exc


def _decode_configuration_buffer(value: object) -> BufferTemplate:
    allowed = {
        "buffer_id",
        "value_id",
        "storage_resource_id",
        "bytes_per_slot",
        "slot_count",
        "producer_phase_id",
        "release_phase_ids",
    }
    data = _object(
        value,
        allowed=allowed,
        required=allowed,
        error_type=SearchProblemError,
        label="configuration buffer",
    )
    return BufferTemplate(
        _string(data["buffer_id"], error_type=SearchProblemError, label="buffer_id"),
        _string(data["value_id"], error_type=SearchProblemError, label="value_id"),
        _string(
            data["storage_resource_id"],
            error_type=SearchProblemError,
            label="storage_resource_id",
        ),
        _int(data["bytes_per_slot"], error_type=SearchProblemError, label="bytes_per_slot"),
        _int(data["slot_count"], error_type=SearchProblemError, label="slot_count"),
        _string(
            data["producer_phase_id"],
            error_type=SearchProblemError,
            label="producer_phase_id",
        ),
        tuple(
            _string(item, error_type=SearchProblemError, label="release_phase_id")
            for item in _sequence(
                data["release_phase_ids"],
                error_type=SearchProblemError,
                label="release_phase_ids",
            )
        ),
    )


def _decode_snapshot(value: object) -> ProfileSnapshotRef:
    data = _object(
        value,
        allowed={"snapshot_id", "revision"},
        required={"snapshot_id", "revision"},
        error_type=SearchProblemError,
        label="profile snapshot",
    )
    try:
        return ProfileSnapshotRef(
            _string(data["snapshot_id"], error_type=SearchProblemError, label="snapshot_id"),
            _int(data["revision"], error_type=SearchProblemError, label="revision"),
        )
    except InvalidRequestError as exc:
        raise SearchProblemError(str(exc)) from exc


def _decode_problem_workload(value: object) -> object:
    try:
        return _decode_workload(value)
    except WorkloadError as exc:
        raise SearchProblemError(str(exc)) from exc


def _decode_problem_program(value: object) -> TileProgram:
    try:
        return _decode_program_value(value)
    except (InvalidRequestError, WorkloadError) as exc:
        raise SearchProblemError(str(exc)) from exc


def _decode_problem_solver_options(value: object) -> SolverOptions:
    try:
        return _decode_solver_options(value)
    except InvalidRequestError as exc:
        raise SearchProblemError(str(exc)) from exc


def problem_from_json(text: str) -> SearchProblem:
    data = _object(
        _loads(text, SearchProblemError),
        allowed={
            "schema_version",
            "request_id",
            "hardware",
            "workload",
            "programs",
            "profile_snapshot",
            "solver_options",
            "configurations",
            "rejected_before_solve",
        },
        required={
            "schema_version",
            "request_id",
            "hardware",
            "workload",
            "programs",
            "profile_snapshot",
            "solver_options",
            "configurations",
            "rejected_before_solve",
        },
        error_type=SearchProblemError,
        label="search problem",
    )
    _version(
        data, SEARCH_PROBLEM_SCHEMA_VERSION, error_type=SearchProblemError, label="search problem"
    )
    return SearchProblem(
        _int(data["schema_version"], error_type=SearchProblemError, label="schema_version"),
        _string(data["request_id"], error_type=SearchProblemError, label="request_id"),
        _decode_hardware_spec(data["hardware"]),
        _decode_problem_workload(data["workload"]),
        tuple(
            _decode_problem_program(item)
            for item in _sequence(data["programs"], error_type=SearchProblemError, label="programs")
        ),
        _decode_snapshot(data["profile_snapshot"]),
        _decode_problem_solver_options(data["solver_options"]),
        tuple(
            _decode_configuration(item)
            for item in _sequence(
                data["configurations"], error_type=SearchProblemError, label="configurations"
            )
        ),
        tuple(
            _decode_rejected(item, error_type=SearchProblemError)
            for item in _sequence(
                data["rejected_before_solve"],
                error_type=SearchProblemError,
                label="rejected_before_solve",
            )
        ),
    )


def problem_to_json(problem: SearchProblem) -> str:
    if type(problem) is not SearchProblem:
        raise SearchProblemError("problem must be SearchProblem")
    _assert_version(problem, SEARCH_PROBLEM_SCHEMA_VERSION, "search problem")
    payload = _encode(problem)
    text = canonical_json(payload)
    problem_from_json(text)
    return text


def _decode_diagnostic(value: object) -> Diagnostic:
    data = _object(
        value,
        allowed={"code", "message", "subject_id"},
        required={"code", "message"},
        error_type=InvalidRequestError,
        label="diagnostic",
    )
    return Diagnostic(
        _decode_enum(data["code"], DiagnosticCode, error_type=InvalidRequestError, label="code"),
        _string(data["message"], error_type=InvalidRequestError, label="message"),
        None
        if data.get("subject_id") is None
        else _string(data["subject_id"], error_type=InvalidRequestError, label="subject_id"),
    )


def _decode_rejected(
    value: object, *, error_type: type[Exception] = InvalidRequestError
) -> RejectedCandidate:
    data = _object(
        value,
        allowed={"configuration_id", "code", "message"},
        required={"code", "message"},
        error_type=error_type,
        label="rejected candidate",
    )
    configuration_id = data.get("configuration_id")
    try:
        return RejectedCandidate(
            None
            if configuration_id is None
            else _string(configuration_id, error_type=error_type, label="configuration_id"),
            _decode_enum(data["code"], DiagnosticCode, error_type=error_type, label="code"),
            _string(data["message"], error_type=error_type, label="message"),
        )
    except InvalidRequestError as exc:
        if error_type is InvalidRequestError:
            raise
        raise error_type(str(exc)) from exc


def _decode_proof(value: object) -> SolveProof:
    data = _object(
        value,
        allowed={
            "status",
            "objective_ps",
            "lower_bound_ps",
            "sensitivity_ps",
            "optimality_gap_ppm",
            "solver_name",
            "solver_version",
            "candidate_count",
            "solved_candidate_count",
            "rejected_candidate_count",
        },
        required={
            "status",
            "objective_ps",
            "lower_bound_ps",
            "sensitivity_ps",
            "optimality_gap_ppm",
            "solver_name",
            "solver_version",
            "candidate_count",
            "solved_candidate_count",
            "rejected_candidate_count",
        },
        error_type=InvalidRequestError,
        label="proof",
    )
    return SolveProof(
        _decode_enum(
            data["status"], EvaluationStatus, error_type=InvalidRequestError, label="status"
        ),
        *(
            _int(data[name], error_type=InvalidRequestError, label=name)
            for name in ("objective_ps", "lower_bound_ps", "sensitivity_ps", "optimality_gap_ppm")
        ),
        _string(data["solver_name"], error_type=InvalidRequestError, label="solver_name"),
        _string(data["solver_version"], error_type=InvalidRequestError, label="solver_version"),
        *(
            _int(data[name], error_type=InvalidRequestError, label=name)
            for name in ("candidate_count", "solved_candidate_count", "rejected_candidate_count")
        ),
    )


def _decode_loop_timing(value: object) -> LoopTiming:
    data = _object(
        value,
        allowed={"loop_id", "initiation_interval_ps", "prologue_ps", "epilogue_ps", "span_ps"},
        required={"loop_id", "prologue_ps", "epilogue_ps", "span_ps"},
        error_type=InvalidRequestError,
        label="loop timing",
    )
    return LoopTiming(
        _string(data["loop_id"], error_type=InvalidRequestError, label="loop_id"),
        None
        if data.get("initiation_interval_ps") is None
        else _int(
            data["initiation_interval_ps"],
            error_type=InvalidRequestError,
            label="initiation_interval_ps",
        ),
        _int(data["prologue_ps"], error_type=InvalidRequestError, label="prologue_ps"),
        _int(data["epilogue_ps"], error_type=InvalidRequestError, label="epilogue_ps"),
        _int(data["span_ps"], error_type=InvalidRequestError, label="span_ps"),
    )


def _decode_resource_reservation(value: object) -> ResourceReservation:
    data = _object(
        value,
        allowed={"resource_id", "slots"},
        required={"resource_id", "slots"},
        error_type=InvalidRequestError,
        label="resource reservation",
    )
    return ResourceReservation(
        _string(data["resource_id"], error_type=InvalidRequestError, label="resource_id"),
        _int(data["slots"], error_type=InvalidRequestError, label="slots"),
    )


def _decode_placement(value: object) -> PhasePlacement:
    data = _object(
        value,
        allowed={
            "phase_id",
            "source_op_id",
            "loop_id",
            "region",
            "iteration",
            "start_ps",
            "end_ps",
            "warp_ids",
            "resources",
        },
        required={
            "phase_id",
            "source_op_id",
            "loop_id",
            "region",
            "iteration",
            "start_ps",
            "end_ps",
            "warp_ids",
            "resources",
        },
        error_type=InvalidRequestError,
        label="placement",
    )
    return PhasePlacement(
        _string(data["phase_id"], error_type=InvalidRequestError, label="phase_id"),
        _string(data["source_op_id"], error_type=InvalidRequestError, label="source_op_id"),
        None
        if data["loop_id"] is None
        else _string(data["loop_id"], error_type=InvalidRequestError, label="loop_id"),
        _decode_enum(
            data["region"], TimelineRegion, error_type=InvalidRequestError, label="region"
        ),
        _int(data["iteration"], error_type=InvalidRequestError, label="iteration"),
        _int(data["start_ps"], error_type=InvalidRequestError, label="start_ps"),
        _int(data["end_ps"], error_type=InvalidRequestError, label="end_ps"),
        tuple(
            _int(item, error_type=InvalidRequestError, label="warp_id")
            for item in _sequence(
                data["warp_ids"], error_type=InvalidRequestError, label="warp_ids"
            )
        ),
        tuple(
            _decode_resource_reservation(item)
            for item in _sequence(
                data["resources"], error_type=InvalidRequestError, label="resources"
            )
        ),
    )


def _decode_buffer(value: object) -> BufferAllocation:
    data = _object(
        value,
        allowed={
            "buffer_id",
            "value_id",
            "storage_resource_id",
            "bytes_per_slot",
            "slot_count",
            "total_bytes",
        },
        required={
            "buffer_id",
            "value_id",
            "storage_resource_id",
            "bytes_per_slot",
            "slot_count",
            "total_bytes",
        },
        error_type=InvalidRequestError,
        label="buffer",
    )
    return BufferAllocation(
        *(
            _string(data[name], error_type=InvalidRequestError, label=name)
            for name in ("buffer_id", "value_id", "storage_resource_id")
        ),
        *(
            _int(data[name], error_type=InvalidRequestError, label=name)
            for name in ("bytes_per_slot", "slot_count", "total_bytes")
        ),
    )


def _decode_utilization(value: object) -> ResourceUtilization:
    data = _object(
        value,
        allowed={"resource_id", "capacity_slots", "busy_slot_ps", "horizon_ps"},
        required={"resource_id", "capacity_slots", "busy_slot_ps", "horizon_ps"},
        error_type=InvalidRequestError,
        label="utilization",
    )
    return ResourceUtilization(
        _string(data["resource_id"], error_type=InvalidRequestError, label="resource_id"),
        _int(data["capacity_slots"], error_type=InvalidRequestError, label="capacity_slots"),
        _int(data["busy_slot_ps"], error_type=InvalidRequestError, label="busy_slot_ps"),
        _int(data["horizon_ps"], error_type=InvalidRequestError, label="horizon_ps"),
    )


def _decode_profile_provenance(value: object) -> ProfileProvenance:
    data = _object(
        value,
        allowed={
            "phase_id",
            "source_op_id",
            "implementation_id",
            "phase_name",
            "component_id",
            "measurement_id",
            "profile_key_id",
            "environment_id",
            "timing_metric",
            "statistic",
            "sensitivity_statistic",
        },
        required={
            "phase_id",
            "source_op_id",
            "implementation_id",
            "phase_name",
            "component_id",
            "measurement_id",
            "profile_key_id",
            "environment_id",
            "timing_metric",
            "statistic",
            "sensitivity_statistic",
        },
        error_type=InvalidRequestError,
        label="profile provenance",
    )
    return ProfileProvenance(
        *(
            str(_string(data[name], error_type=InvalidRequestError, label=name))
            for name in (
                "phase_id",
                "source_op_id",
                "implementation_id",
                "phase_name",
                "component_id",
                "measurement_id",
                "profile_key_id",
                "environment_id",
            )
        ),
        _decode_enum(
            data["timing_metric"],
            TimingMetric,
            error_type=InvalidRequestError,
            label="timing_metric",
        ),
        _decode_enum(
            data["statistic"],
            TimingStatistic,
            error_type=InvalidRequestError,
            label="statistic",
        ),
        _decode_enum(
            data["sensitivity_statistic"],
            TimingStatistic,
            error_type=InvalidRequestError,
            label="sensitivity_statistic",
        ),
    )


def _decode_selected(value: object) -> SelectedConfiguration:
    data = _object(
        value,
        allowed={
            "configuration_id",
            "program_id",
            "tile",
            "implementations",
            "warps",
            "pipeline_depth",
            "layout_variant_id",
            "static_demands",
        },
        required={
            "configuration_id",
            "program_id",
            "tile",
            "implementations",
            "warps",
            "pipeline_depth",
            "layout_variant_id",
            "static_demands",
        },
        error_type=InvalidRequestError,
        label="selected configuration",
    )
    warps = data["warps"]
    if isinstance(warps, dict):
        warps = _decode_warp_config(warps)
    return SelectedConfiguration(
        _string(data["configuration_id"], error_type=InvalidRequestError, label="configuration_id"),
        _string(data["program_id"], error_type=InvalidRequestError, label="program_id"),
        _decode_tile_candidate(data["tile"]),
        tuple(
            _decode_selected_implementation(item)
            for item in _sequence(
                data["implementations"], error_type=InvalidRequestError, label="implementations"
            )
        ),
        warps,
        _int(data["pipeline_depth"], error_type=InvalidRequestError, label="pipeline_depth"),
        _string(
            data["layout_variant_id"], error_type=InvalidRequestError, label="layout_variant_id"
        ),
        tuple(
            _decode_selected_static_demand(item)
            for item in _sequence(
                data["static_demands"], error_type=InvalidRequestError, label="static_demands"
            )
        ),
    )


def _decode_selected_implementation(value: object) -> SelectedImplementation:
    data = _object(
        value,
        allowed={"op_id", "implementation_id"},
        required={"op_id", "implementation_id"},
        error_type=InvalidRequestError,
        label="selected implementation",
    )
    return SelectedImplementation(
        _string(data["op_id"], error_type=InvalidRequestError, label="op_id"),
        _string(
            data["implementation_id"],
            error_type=InvalidRequestError,
            label="implementation_id",
        ),
    )


def _decode_selected_static_demand(value: object) -> SelectedStaticDemand:
    data = _object(
        value,
        allowed={"resource_id", "units"},
        required={"resource_id", "units"},
        error_type=InvalidRequestError,
        label="selected static demand",
    )
    return SelectedStaticDemand(
        _string(data["resource_id"], error_type=InvalidRequestError, label="resource_id"),
        _int(data["units"], error_type=InvalidRequestError, label="units"),
    )


def _decode_plan(value: object) -> CostModelPlan:
    data = _object(
        value,
        allowed={
            "schema_version",
            "request_id",
            "hardware",
            "profile_snapshot",
            "workload",
            "program",
            "selected",
            "end_to_end_ps",
            "loop_timings",
            "placements",
            "buffers",
            "utilization",
            "profiles",
            "proof",
        },
        required={
            "schema_version",
            "request_id",
            "hardware",
            "profile_snapshot",
            "workload",
            "program",
            "selected",
            "end_to_end_ps",
            "loop_timings",
            "placements",
            "buffers",
            "utilization",
            "profiles",
            "proof",
        },
        error_type=InvalidRequestError,
        label="plan",
    )
    _version(data, PLAN_SCHEMA_VERSION, error_type=InvalidRequestError, label="plan")
    profile_snapshot = data["profile_snapshot"]
    if isinstance(profile_snapshot, dict):
        profile_snapshot = _decode_snapshot(profile_snapshot)
    program = (
        _decode_program_value(data["program"])
        if isinstance(data["program"], dict)
        else data["program"]
    )
    hardware = (
        _decode_hardware_ref(data["hardware"])
        if isinstance(data["hardware"], dict)
        and set(data["hardware"]) == {"hardware_id", "schema_version", "calibration_id"}
        else data["hardware"]
    )
    workload = (
        _decode_workload(data["workload"])
        if isinstance(data["workload"], dict)
        else data["workload"]
    )
    return CostModelPlan(
        _int(data["schema_version"], error_type=InvalidRequestError, label="schema_version"),
        _string(data["request_id"], error_type=InvalidRequestError, label="request_id"),
        hardware,
        profile_snapshot,
        workload,
        program,
        _decode_selected(data["selected"]),
        _int(data["end_to_end_ps"], error_type=InvalidRequestError, label="end_to_end_ps"),
        tuple(
            _decode_loop_timing(item)
            for item in _sequence(
                data["loop_timings"], error_type=InvalidRequestError, label="loop_timings"
            )
        ),
        tuple(
            _decode_placement(item)
            for item in _sequence(
                data["placements"], error_type=InvalidRequestError, label="placements"
            )
        ),
        tuple(
            _decode_buffer(item)
            for item in _sequence(data["buffers"], error_type=InvalidRequestError, label="buffers")
        ),
        tuple(
            _decode_utilization(item)
            for item in _sequence(
                data["utilization"], error_type=InvalidRequestError, label="utilization"
            )
        ),
        tuple(
            _decode_profile_provenance(item)
            for item in _sequence(
                data["profiles"], error_type=InvalidRequestError, label="profiles"
            )
        ),
        _decode_proof(data["proof"]),
    )


def plan_from_json(text: str) -> CostModelPlan:
    return _decode_plan(_loads(text, InvalidRequestError))


def plan_to_json(plan: CostModelPlan) -> str:
    if type(plan) is not CostModelPlan:
        raise InvalidRequestError("plan must be CostModelPlan")
    _assert_version(plan, PLAN_SCHEMA_VERSION, "plan")
    payload = _encode(plan)
    if isinstance(payload, dict):
        _sort_plan_payload(payload)
    _decode_plan(payload)
    return canonical_json(payload)


def result_from_json(text: str) -> CostModelResult:
    data = _object(
        _loads(text, InvalidRequestError),
        allowed={
            "schema_version",
            "status",
            "plan",
            "missing_profiles",
            "rejected_candidates",
            "diagnostics",
        },
        required={
            "schema_version",
            "status",
            "plan",
            "missing_profiles",
            "rejected_candidates",
            "diagnostics",
        },
        error_type=InvalidRequestError,
        label="result",
    )
    _version(data, RESULT_SCHEMA_VERSION, error_type=InvalidRequestError, label="result")
    plan = None if data["plan"] is None else _decode_plan(data["plan"])
    return CostModelResult(
        _int(data["schema_version"], error_type=InvalidRequestError, label="schema_version"),
        _decode_enum(
            data["status"], EvaluationStatus, error_type=InvalidRequestError, label="status"
        ),
        plan,
        tuple(
            _string(item, error_type=InvalidRequestError, label="profile key ID")
            for item in _sequence(
                data["missing_profiles"], error_type=InvalidRequestError, label="missing_profiles"
            )
        ),
        tuple(
            _decode_rejected(item)
            for item in _sequence(
                data["rejected_candidates"],
                error_type=InvalidRequestError,
                label="rejected_candidates",
            )
        ),
        tuple(
            _decode_diagnostic(item)
            for item in _sequence(
                data["diagnostics"], error_type=InvalidRequestError, label="diagnostics"
            )
        ),
    )


def result_to_json(result: CostModelResult) -> str:
    if type(result) is not CostModelResult:
        raise InvalidRequestError("result must be CostModelResult")
    _assert_version(result, RESULT_SCHEMA_VERSION, "result")
    payload = _encode(result)
    if isinstance(payload, dict):
        _sort_result_payload(payload)
    text = canonical_json(payload)
    result_from_json(text)
    return text


def render_timeline(plan: CostModelPlan) -> str:
    """Render a stable compact timeline for a selected plan."""

    if type(plan) is not CostModelPlan:
        raise InvalidRequestError("plan must be CostModelPlan")
    rows = sorted(
        plan.placements,
        key=lambda item: (
            str(item.loop_id) if item.loop_id is not None else "",
            item.region.value,
            item.iteration,
            item.start_ps,
            str(item.phase_id),
        ),
    )
    return "\n".join(
        f"{placement.start_ps}..{placement.end_ps} {placement.phase_id}" for placement in rows
    )


def _assert_version(value: object, expected: int, label: str) -> None:
    version = getattr(value, "schema_version", None)
    if version != expected:
        error_type: type[Exception] = (
            SearchProblemError
            if label == "search problem"
            else (WorkloadError if label == "program" else InvalidRequestError)
        )
        raise error_type(f"unsupported {label} schema version: {version!r}")


def _document_mapping(value: object) -> dict[str, object]:
    # Mapping inputs are accepted as already-decoded JSON documents for the
    # hardware/profile interchange helpers.  A different dataclass is not a
    # schema-owned record, even when its field names happen to line up.
    if is_dataclass(value):
        raise TypeError("schema document must be an owning typed record or a mapping")
    encoded = _encode(value)
    if not isinstance(encoded, dict):
        raise InvalidRequestError("schema document must encode to a JSON object")
    return encoded


def _sort_plan_payload(payload: dict[str, object]) -> None:
    loop_timings = payload.get("loop_timings")
    if isinstance(loop_timings, list):
        loop_timings.sort(
            key=lambda item: str(item.get("loop_id", "")) if isinstance(item, dict) else ""
        )
    placements = payload.get("placements")
    if isinstance(placements, list):
        placements.sort(
            key=lambda item: (
                str(item.get("loop_id") or "") if isinstance(item, dict) else "",
                str(item.get("region", "")) if isinstance(item, dict) else "",
                int(item.get("iteration", 0)) if isinstance(item, dict) else 0,
                int(item.get("start_ps", 0)) if isinstance(item, dict) else 0,
                str(item.get("phase_id", "")) if isinstance(item, dict) else "",
            )
        )
    for field, key in (
        ("buffers", "buffer_id"),
        ("utilization", "resource_id"),
        ("profiles", "phase_id"),
    ):
        values = payload.get(field)
        if isinstance(values, list):
            values.sort(key=lambda item: str(item.get(key, "")) if isinstance(item, dict) else "")


def _sort_result_payload(payload: dict[str, object]) -> None:
    missing = payload.get("missing_profiles")
    if isinstance(missing, list):
        missing.sort(key=str)
    rejected = payload.get("rejected_candidates")
    if isinstance(rejected, list):
        rejected.sort(
            key=lambda item: (
                str(item.get("configuration_id") or "") if isinstance(item, dict) else "",
                str(item.get("code", "")) if isinstance(item, dict) else "",
                str(item.get("message", "")) if isinstance(item, dict) else "",
            )
        )
    diagnostics = payload.get("diagnostics")
    if isinstance(diagnostics, list):
        diagnostics.sort(
            key=lambda item: (
                str(item.get("code", "")) if isinstance(item, dict) else "",
                str(item.get("subject_id") or "") if isinstance(item, dict) else "",
                str(item.get("message", "")) if isinstance(item, dict) else "",
            )
        )
    plan = payload.get("plan")
    if isinstance(plan, dict):
        _sort_plan_payload(plan)


def _require_subset(
    data: Mapping[str, object], required: set[str], error_type: type[Exception], label: str
) -> None:
    missing = sorted(required - set(data))
    if missing:
        raise error_type(f"{label} is missing required field(s): {', '.join(missing)}")


def _reject_workload_extras(data: Mapping[str, object], allowed: set[str], label: str) -> None:
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise WorkloadError(f"{label} contains unknown field(s): {', '.join(unknown)}")


__all__ = [
    "canonical_json",
    "hardware_from_json",
    "hardware_to_json",
    "plan_from_json",
    "plan_to_json",
    "problem_from_json",
    "problem_to_json",
    "program_from_json",
    "program_to_json",
    "profile_snapshot_from_json",
    "profile_snapshot_to_json",
    "render_timeline",
    "request_from_json",
    "request_to_json",
    "result_from_json",
    "result_to_json",
]
