"""Dependency-free shared values for the version 2 cost-model boundary.

These records provide stable request and schema values without importing a
compiler, solver backend, profiler, or optional dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import NewType, TypeAlias

from .constants import HARDWARE_SCHEMA_VERSION
from .errors import InvalidRequestError, WorkloadError

ConfigurationId = NewType("ConfigurationId", str)
ProgramId = NewType("ProgramId", str)
OpId = NewType("OpId", str)
ValueId = NewType("ValueId", str)
LoopId = NewType("LoopId", str)
PhaseId = NewType("PhaseId", str)
BufferId = NewType("BufferId", str)
ResourceId = NewType("ResourceId", str)
MeasurementId = NewType("MeasurementId", str)
ProfileKeyId = NewType("ProfileKeyId", str)


def validate_identifier(value: str, *, label: str = "identifier") -> str:
    """Validate one stable ASCII identifier and return it unchanged."""

    if not isinstance(value, str) or not value:
        raise InvalidRequestError(f"{label} must be a non-empty ASCII string")
    try:
        value.encode("ascii")
    except UnicodeEncodeError as exc:
        raise InvalidRequestError(f"{label} must be a non-empty ASCII string") from exc
    return value


class DType(str, Enum):
    """Name a calibrated tensor element type."""

    BF16 = "bf16"
    FP16 = "fp16"
    FP32 = "fp32"
    FP8_E4M3 = "fp8_e4m3"
    FP8_E5M2 = "fp8_e5m2"


class TensorLayout(str, Enum):
    """Name a logical dense matrix layout."""

    ROW_MAJOR = "row_major"
    COLUMN_MAJOR = "column_major"


class TimingMetric(str, Enum):
    """Select one measured timing dimension."""

    LATENCY = "latency"
    INITIATION_INTERVAL = "initiation_interval"


@dataclass(frozen=True, slots=True)
class AxisExtent:
    """Describe one positive named axis."""

    name: str
    extent: int

    def __post_init__(self) -> None:
        validate_identifier(self.name, label="axis name")
        if isinstance(self.extent, bool) or not isinstance(self.extent, int) or self.extent <= 0:
            raise InvalidRequestError("axis extent must be a positive integer")


@dataclass(frozen=True, slots=True)
class NamedShape:
    """Describe a tensor shape with unique axis names."""

    axes: tuple[AxisExtent, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.axes, (tuple, list)):
            raise InvalidRequestError("shape axes must be a sequence")
        axes = tuple(self.axes)
        object.__setattr__(self, "axes", axes)
        if not all(type(axis) is AxisExtent for axis in axes):
            raise InvalidRequestError("shape axes must contain AxisExtent records")
        names = tuple(axis.name for axis in axes)
        if len(names) != len(set(names)):
            raise InvalidRequestError("shape axis names must be unique")

    def axis(self, name: str) -> AxisExtent:
        """Resolve one named axis exactly."""

        validate_identifier(name, label="axis name")
        for axis in self.axes:
            if axis.name == name:
                return axis
        raise InvalidRequestError(f"shape has no axis named {name!r}")

    def extent(self, name: str) -> int:
        """Return one named axis extent."""

        return self.axis(name).extent


@dataclass(frozen=True, slots=True)
class TensorDescriptor:
    """Describe one concrete tensor view."""

    shape: NamedShape
    dtype: DType
    layout: TensorLayout
    strides_elements: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if type(self.shape) is not NamedShape:
            raise InvalidRequestError("tensor shape must be NamedShape")
        if not isinstance(self.dtype, DType):
            try:
                object.__setattr__(self, "dtype", DType(self.dtype))
            except (TypeError, ValueError) as exc:
                raise InvalidRequestError("unknown tensor dtype") from exc
        if not isinstance(self.layout, TensorLayout):
            try:
                object.__setattr__(self, "layout", TensorLayout(self.layout))
            except (TypeError, ValueError) as exc:
                raise InvalidRequestError("unknown tensor layout") from exc
        if self.strides_elements is not None:
            if not isinstance(self.strides_elements, (tuple, list)):
                raise InvalidRequestError("explicit strides must be a sequence or None")
            strides = tuple(self.strides_elements)
            object.__setattr__(self, "strides_elements", strides)
            if len(strides) != len(self.shape.axes):
                raise InvalidRequestError("explicit strides must match shape rank")
            if any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in strides
            ):
                raise InvalidRequestError("explicit strides must be non-negative integers")

    @property
    def rank(self) -> int:
        """Return the tensor rank."""

        return len(self.shape.axes)

    @property
    def element_count(self) -> int:
        """Return the number of logical elements."""

        result = 1
        for axis in self.shape.axes:
            result *= axis.extent
        return result


class WorkloadKind(str, Enum):
    """Name one supported logical workload."""

    GEMM = "gemm"
    GQA_DECODE = "gqa_decode"
    FLASH_ATTENTION = "flash_attention"
    MLP = "mlp"


class EpilogueKind(str, Enum):
    """Name a GEMM epilogue."""

    NONE = "none"
    BIAS = "bias"
    RELU = "relu"
    GELU = "gelu"
    SILU = "silu"


class ActivationKind(str, Enum):
    """Name an MLP activation."""

    RELU = "relu"
    GELU = "gelu"
    SILU = "silu"
    SWIGLU = "swiglu"


@dataclass(frozen=True, slots=True)
class GemmSpec:
    """Describe one logical GEMM workload."""

    kind: WorkloadKind
    m: int
    n: int
    k: int
    dtype_a: DType
    dtype_b: DType
    dtype_accumulator: DType
    dtype_output: DType
    layout_a: TensorLayout
    layout_b: TensorLayout
    epilogue: EpilogueKind = EpilogueKind.NONE

    def __post_init__(self) -> None:
        _coerce_enum(self, "kind", WorkloadKind)
        _validate_workload_kind(self.kind, WorkloadKind.GEMM)
        _validate_positive_dimensions((self.m, self.n, self.k))
        _coerce_enum(self, "dtype_a", DType)
        _coerce_enum(self, "dtype_b", DType)
        _coerce_enum(self, "dtype_accumulator", DType)
        _coerce_enum(self, "dtype_output", DType)
        _coerce_enum(self, "layout_a", TensorLayout)
        _coerce_enum(self, "layout_b", TensorLayout)
        _coerce_enum(self, "epilogue", EpilogueKind)


@dataclass(frozen=True, slots=True)
class GqaDecodeSpec:
    """Describe one logical GQA decode workload."""

    kind: WorkloadKind
    batch_size: int
    query_heads: int
    kv_heads: int
    head_dim: int
    context_len: int
    dtype_query: DType
    dtype_kv: DType
    dtype_accumulator: DType
    dtype_output: DType

    def __post_init__(self) -> None:
        _coerce_enum(self, "kind", WorkloadKind)
        _validate_workload_kind(self.kind, WorkloadKind.GQA_DECODE)
        _validate_positive_dimensions(
            (self.batch_size, self.query_heads, self.kv_heads, self.head_dim, self.context_len)
        )
        if self.query_heads % self.kv_heads:
            raise WorkloadError("query_heads must be divisible by kv_heads")
        for name in ("dtype_query", "dtype_kv", "dtype_accumulator", "dtype_output"):
            _coerce_enum(self, name, DType)


@dataclass(frozen=True, slots=True)
class FlashAttentionSpec:
    """Describe one logical FlashAttention workload."""

    kind: WorkloadKind
    batch_size: int
    heads: int
    query_len: int
    key_len: int
    head_dim: int
    dtype_query: DType
    dtype_kv: DType
    dtype_accumulator: DType
    dtype_output: DType
    causal: bool

    def __post_init__(self) -> None:
        _coerce_enum(self, "kind", WorkloadKind)
        _validate_workload_kind(self.kind, WorkloadKind.FLASH_ATTENTION)
        _validate_positive_dimensions(
            (self.batch_size, self.heads, self.query_len, self.key_len, self.head_dim)
        )
        if not isinstance(self.causal, bool):
            raise WorkloadError("causal must be a boolean")
        for name in ("dtype_query", "dtype_kv", "dtype_accumulator", "dtype_output"):
            _coerce_enum(self, name, DType)


@dataclass(frozen=True, slots=True)
class MlpSpec:
    """Describe one logical two-layer MLP workload."""

    kind: WorkloadKind
    rows: int
    input_dim: int
    hidden_dim: int
    output_dim: int
    dtype_input: DType
    dtype_weight: DType
    dtype_accumulator: DType
    dtype_output: DType
    activation: ActivationKind

    def __post_init__(self) -> None:
        _coerce_enum(self, "kind", WorkloadKind)
        _validate_workload_kind(self.kind, WorkloadKind.MLP)
        _validate_positive_dimensions((self.rows, self.input_dim, self.hidden_dim, self.output_dim))
        for name in ("dtype_input", "dtype_weight", "dtype_accumulator", "dtype_output"):
            _coerce_enum(self, name, DType)
        _coerce_enum(self, "activation", ActivationKind)


WorkloadSpec: TypeAlias = GemmSpec | GqaDecodeSpec | FlashAttentionSpec | MlpSpec


def _coerce_enum(instance: object, field_name: str, enum_type: type[Enum]) -> None:
    value = getattr(instance, field_name)
    if isinstance(value, enum_type):
        return
    try:
        object.__setattr__(instance, field_name, enum_type(value))
    except (TypeError, ValueError) as exc:
        raise WorkloadError(f"invalid {field_name}") from exc


def _validate_workload_kind(value: object, expected: WorkloadKind) -> None:
    if not isinstance(value, WorkloadKind):
        try:
            value = WorkloadKind(value)
        except (TypeError, ValueError) as exc:
            raise WorkloadError("unknown workload kind") from exc
    if value is not expected:
        raise WorkloadError(f"workload kind must be {expected.value!r}")


def _validate_positive_dimensions(values: tuple[int, ...]) -> None:
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in values):
        raise WorkloadError("workload dimensions must be positive integers")


__all__ = [
    "ActivationKind",
    "AxisExtent",
    "BufferId",
    "ConfigurationId",
    "DType",
    "EpilogueKind",
    "FlashAttentionSpec",
    "GemmSpec",
    "GqaDecodeSpec",
    "HardwareSpecRef",
    "LoopId",
    "MeasurementId",
    "MlpSpec",
    "NamedShape",
    "OpId",
    "PhaseId",
    "ProfileKeyId",
    "ProgramId",
    "ResourceId",
    "TensorDescriptor",
    "TensorLayout",
    "TimingMetric",
    "ValueId",
    "WorkloadKind",
    "WorkloadSpec",
    "validate_identifier",
]


@dataclass(frozen=True, slots=True)
class HardwareSpecRef:
    """Identify an exact hardware document without implementing hardware facts."""

    hardware_id: str
    schema_version: int
    calibration_id: str

    def __post_init__(self) -> None:
        validate_identifier(self.hardware_id, label="hardware_id")
        validate_identifier(self.calibration_id, label="calibration_id")
        if (
            isinstance(self.schema_version, bool)
            or not isinstance(self.schema_version, int)
            or self.schema_version != HARDWARE_SCHEMA_VERSION
        ):
            raise InvalidRequestError(
                f"unsupported hardware schema version: {self.schema_version!r}"
            )
