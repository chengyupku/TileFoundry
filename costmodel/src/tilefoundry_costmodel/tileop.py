"""Typed operation-lowering and profile-identity records.

This module is the compiler-independent M2 composition boundary.  It carries
only immutable descriptions: no compiler IR, executable code, CUDA handles, or
measured timing values are stored here.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from typing import Protocol, TypeVar, cast

from .constants import PROFILE_SCHEMA_VERSION
from .errors import HardwareSpecError, ProfileStoreError, WorkloadError
from .hardware.model import HardwareSpec, HardwareSpecRef
from .model import (
    BufferId,
    ConfigurationId,
    LoopId,
    NamedShape,
    OpId,
    PhaseId,
    ProfileKeyId,
    ProgramId,
    ResourceId,
    TimingMetric,
    ValueId,
    WorkloadKind,
    validate_identifier,
)
from .program import (
    AlignedRelation,
    CopyOp,
    DependencyRelation,
    ElementwiseOp,
    EndpointRelation,
    GemmOp,
    LoopBarrier,
    MemorySpace,
    OpIterationDomain,
    ReduceOp,
    TileCandidate,
    TileOp,
    TileOpKind,
    TileProgram,
    TileValueType,
)
from .request import WarpConfig


def _identifier(value: object, label: str, error_type: type[Exception] = WorkloadError) -> str:
    if not isinstance(value, str):
        raise error_type(f"{label} must be a non-empty ASCII string")
    try:
        return validate_identifier(value, label=label)
    except Exception as exc:
        raise error_type(str(exc)) from exc


def _text(value: object, label: str, error_type: type[Exception] = WorkloadError) -> str:
    if not isinstance(value, str) or not value:
        raise error_type(f"{label} must be a non-empty string")
    return value


def _positive(value: object, label: str, error_type: type[Exception] = WorkloadError) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise error_type(f"{label} must be a positive integer")
    return value


def _non_negative(value: object, label: str, error_type: type[Exception] = WorkloadError) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise error_type(f"{label} must be a non-negative integer")
    return value


def _enum(instance: object, field_name: str, enum_type: type[Enum]) -> None:
    value = getattr(instance, field_name)
    if isinstance(value, enum_type):
        return
    try:
        object.__setattr__(instance, field_name, enum_type(value))
    except (TypeError, ValueError) as exc:
        raise WorkloadError(f"invalid {field_name}: {value!r}") from exc


def _tuple_of(
    value: object,
    item_type: type["_RecordT"],
    label: str,
    error_type: type[Exception] = WorkloadError,
) -> tuple["_RecordT", ...]:
    if not isinstance(value, (tuple, list)):
        raise error_type(f"{label} must be a sequence")
    values = tuple(value)
    if not all(type(item) is item_type for item in values):
        raise error_type(f"{label} must contain {item_type.__name__} records")
    return cast(tuple[_RecordT, ...], values)


_RecordT = TypeVar("_RecordT")


@dataclass(frozen=True, slots=True)
class CanonicalAttribute:
    """One canonical semantic or benchmark attribute."""

    name: str
    value: str

    def __post_init__(self) -> None:
        _identifier(self.name, "attribute name", ProfileStoreError)
        if not isinstance(self.value, str):
            raise ProfileStoreError("attribute value must be a string")


@dataclass(frozen=True, slots=True)
class TileOpSignature:
    """Identity-free typed operation semantics used by profile lookup."""

    op_kind: TileOpKind
    operands: tuple[TileValueType, ...]
    results: tuple[TileValueType, ...]
    semantic_attributes: tuple[CanonicalAttribute, ...] = ()

    def __post_init__(self) -> None:
        _enum(self, "op_kind", TileOpKind)
        operands = _tuple_of(self.operands, TileValueType, "signature operands", ProfileStoreError)
        results = _tuple_of(self.results, TileValueType, "signature results", ProfileStoreError)
        attrs = _tuple_of(
            self.semantic_attributes,
            CanonicalAttribute,
            "semantic attributes",
            ProfileStoreError,
        )
        object.__setattr__(self, "operands", operands)
        object.__setattr__(self, "results", results)
        object.__setattr__(
            self,
            "semantic_attributes",
            tuple(sorted(attrs, key=lambda item: (item.name, item.value))),
        )
        names = tuple(item.name for item in attrs)
        if len(names) != len(set(names)):
            raise ProfileStoreError("semantic attribute names must be unique")
        expected_operands = {
            TileOpKind.COPY: 1,
            TileOpKind.GEMM: 3,
            TileOpKind.REDUCE: 1,
        }
        if self.op_kind in expected_operands and len(operands) != expected_operands[self.op_kind]:
            raise ProfileStoreError("operation signature has the wrong operand count")
        if self.op_kind is TileOpKind.ELEMENTWISE and not operands:
            raise ProfileStoreError("elementwise signature requires at least one operand")
        if len(results) != 1:
            raise ProfileStoreError("operation signature must contain exactly one result")
        expected_attributes = {
            TileOpKind.COPY: set(),
            TileOpKind.GEMM: {"m_axis", "n_axis", "k_axis"},
            TileOpKind.REDUCE: {"axes", "reduction"},
            TileOpKind.ELEMENTWISE: {"function"},
        }[self.op_kind]
        if set(names) != expected_attributes:
            raise ProfileStoreError("operation signature semantic attributes are incomplete")


def _value_types(program: TileProgram) -> dict[ValueId, TileValueType]:
    return {value.value_id: value.value_type for value in program.values}


def _signature_attr(name: str, value: object) -> CanonicalAttribute:
    return CanonicalAttribute(name, str(value))


def tile_op_signature(op: TileOp, *, program: TileProgram) -> TileOpSignature:
    """Derive a canonical identity-free signature from typed program records."""

    if type(program) is not TileProgram:
        raise WorkloadError("program must be TileProgram")
    if type(op) not in (CopyOp, GemmOp, ReduceOp, ElementwiseOp):
        raise WorkloadError("operation must be a concrete TileOp")
    value_types = _value_types(program)
    if op not in program.operations:
        raise WorkloadError("operation does not belong to program")

    operands: tuple[TileValueType, ...]
    results: tuple[TileValueType, ...]
    attrs: tuple[CanonicalAttribute, ...]
    if isinstance(op, CopyOp):
        operands = (value_types[op.source],)
        results = (value_types[op.destination],)
        attrs = ()
    elif isinstance(op, GemmOp):
        operands = (value_types[op.lhs], value_types[op.rhs], value_types[op.accumulator])
        results = (value_types[op.result],)
        attrs = (
            _signature_attr("k_axis", op.k_axis),
            _signature_attr("m_axis", op.m_axis),
            _signature_attr("n_axis", op.n_axis),
        )
    elif isinstance(op, ReduceOp):
        operands = (value_types[op.source],)
        results = (value_types[op.result],)
        # Axis order is a caller-level presentation detail for reductions;
        # profile identity follows the semantic set while program JSON keeps
        # the original typed operation order unchanged.
        canonical_axes = tuple(sorted(op.axes))
        attrs = (
            _signature_attr(
                "axes",
                json.dumps(canonical_axes, ensure_ascii=False, separators=(",", ":")),
            ),
            _signature_attr("reduction", op.reduction.value),
        )
    else:
        operands = tuple(value_types[item] for item in op.inputs)
        results = (value_types[op.result],)
        attrs = (_signature_attr("function", op.function.value),)
    return TileOpSignature(op.kind, operands, results, attrs)


@dataclass(frozen=True, slots=True)
class TileOpProfileQuery:
    """One exact implementation benchmark query."""

    hardware: HardwareSpecRef
    operation: TileOpSignature
    implementation_id: str
    component_id: str
    tile_shape: NamedShape
    warp_config_id: str
    pipeline_depth: int
    layout_variant_id: str
    conditions: tuple[CanonicalAttribute, ...] = ()

    def __post_init__(self) -> None:
        if type(self.hardware) is not HardwareSpecRef:
            raise ProfileStoreError("profile query hardware must be HardwareSpecRef")
        if type(self.operation) is not TileOpSignature:
            raise ProfileStoreError("profile query operation must be TileOpSignature")
        for value, label in (
            (self.implementation_id, "implementation_id"),
            (self.component_id, "component_id"),
            (self.warp_config_id, "warp_config_id"),
            (self.layout_variant_id, "layout_variant_id"),
        ):
            _identifier(value, label, ProfileStoreError)
        if type(self.tile_shape) is not NamedShape:
            raise ProfileStoreError("tile_shape must be NamedShape")
        _positive(self.pipeline_depth, "pipeline_depth", ProfileStoreError)
        conditions = _tuple_of(
            self.conditions, CanonicalAttribute, "profile conditions", ProfileStoreError
        )
        conditions = tuple(sorted(conditions, key=lambda item: (item.name, item.value)))
        if len({item.name for item in conditions}) != len(conditions):
            raise ProfileStoreError("profile condition names must be unique")
        if any(
            value.memory_space is MemorySpace.GLOBAL
            for value in (*self.operation.operands, *self.operation.results)
        ) and not {item.name for item in conditions} >= {"memory_residency", "cache_policy"}:
            raise ProfileStoreError(
                "global-memory profile queries require memory_residency and cache_policy"
            )
        object.__setattr__(self, "conditions", conditions)

    def canonical_json(self) -> str:
        """Return the deterministic query payload used by providers."""

        return json.dumps(
            _canonical_value(self), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )


@dataclass(frozen=True, slots=True)
class ProfileRequirement:
    """Select one timing metric from an exact profile query."""

    query: TileOpProfileQuery
    timing_metric: TimingMetric

    def __post_init__(self) -> None:
        if type(self.query) is not TileOpProfileQuery:
            raise ProfileStoreError("profile requirement query must be TileOpProfileQuery")
        _enum(self, "timing_metric", TimingMetric)


@dataclass(frozen=True, slots=True)
class BenchmarkFingerprint:
    """Exact benchmark source/compiler identity."""

    provider_id: str
    provider_version: str
    benchmark_abi_version: int
    source_sha256: str
    compile_options_sha256: str

    def __post_init__(self) -> None:
        for value, label in (
            (self.provider_id, "provider_id"),
            (self.provider_version, "provider_version"),
        ):
            _identifier(value, label, ProfileStoreError)
        _positive(self.benchmark_abi_version, "benchmark_abi_version", ProfileStoreError)
        for value, label in (
            (self.source_sha256, "source_sha256"),
            (self.compile_options_sha256, "compile_options_sha256"),
        ):
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise ProfileStoreError(f"{label} must be a lowercase SHA-256 digest")


def _canonical_value(value: object) -> object:
    if isinstance(value, Enum):
        return _canonical_value(value.value)
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: _canonical_value(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, (tuple, list)):
        return [_canonical_value(item) for item in value]
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise ProfileStoreError("canonical object keys must be strings")
        return {
            str(key): _canonical_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ProfileStoreError("canonical profile identity cannot contain non-finite floats")
        raise ProfileStoreError("canonical profile identity cannot contain floats")
    if value is None or isinstance(value, (str, int, bool)):
        return value
    raise ProfileStoreError(f"unsupported profile identity value: {type(value).__name__}")


@dataclass(frozen=True, slots=True)
class TileOpProfileKey:
    """Exact profile key with deterministic JSON and SHA-256 identity."""

    schema_version: int
    query: TileOpProfileQuery
    fingerprint: BenchmarkFingerprint

    def __post_init__(self) -> None:
        if (
            isinstance(self.schema_version, bool)
            or not isinstance(self.schema_version, int)
            or self.schema_version != PROFILE_SCHEMA_VERSION
        ):
            raise ProfileStoreError(
                f"unsupported profile-key schema version: {self.schema_version!r}"
            )
        if type(self.query) is not TileOpProfileQuery:
            raise ProfileStoreError("profile key query must be TileOpProfileQuery")
        if type(self.fingerprint) is not BenchmarkFingerprint:
            raise ProfileStoreError("profile key fingerprint must be BenchmarkFingerprint")

    def canonical_json(self) -> str:
        payload = _canonical_value(self)
        return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    def key_id(self) -> ProfileKeyId:
        digest = hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()
        return ProfileKeyId(digest)


class ValueStoragePolicy(str, Enum):
    """Name one value allocation policy."""

    STATIC = "static"
    PIPELINE_RING = "pipeline_ring"


@dataclass(frozen=True, slots=True)
class ProducedValue:
    value_id: ValueId
    availability_id: str
    ready_phase_id: PhaseId

    def __post_init__(self) -> None:
        _identifier(self.value_id, "value_id")
        _identifier(self.availability_id, "availability_id")
        _identifier(self.ready_phase_id, "ready_phase_id")


@dataclass(frozen=True, slots=True)
class ConsumedValue:
    value_id: ValueId
    required_availability_id: str
    consume_phase_id: PhaseId
    release_phase_id: PhaseId

    def __post_init__(self) -> None:
        _identifier(self.value_id, "value_id")
        _identifier(self.required_availability_id, "required_availability_id")
        _identifier(self.consume_phase_id, "consume_phase_id")
        _identifier(self.release_phase_id, "release_phase_id")


@dataclass(frozen=True, slots=True)
class ValueStorage:
    value_id: ValueId
    allocation_phase_id: PhaseId
    storage_policy: ValueStoragePolicy

    def __post_init__(self) -> None:
        _identifier(self.value_id, "value_id")
        _identifier(self.allocation_phase_id, "allocation_phase_id")
        _enum(self, "storage_policy", ValueStoragePolicy)


@dataclass(frozen=True, slots=True)
class TemporalDemand:
    resource_id: ResourceId
    slots: int

    def __post_init__(self) -> None:
        _identifier(self.resource_id, "resource_id")
        _positive(self.slots, "slots")


@dataclass(frozen=True, slots=True)
class StaticDemand:
    resource_id: ResourceId
    units: int

    def __post_init__(self) -> None:
        _identifier(self.resource_id, "resource_id")
        _positive(self.units, "units")


@dataclass(frozen=True, slots=True)
class PhaseIterationDomain:
    loop_id: LoopId | None
    first_iteration: int
    iteration_count: int

    def __post_init__(self) -> None:
        if self.loop_id is not None:
            _identifier(self.loop_id, "loop_id")
        _non_negative(self.first_iteration, "first_iteration")
        _positive(self.iteration_count, "iteration_count")
        if self.loop_id is None and (self.first_iteration, self.iteration_count) != (0, 1):
            raise WorkloadError("a one-time phase domain must equal (None, 0, 1)")

    @classmethod
    def from_operation_domain(cls, domain: OpIterationDomain) -> "PhaseIterationDomain":
        if type(domain) is not OpIterationDomain:
            raise WorkloadError("operation domain must be OpIterationDomain")
        return cls(domain.loop_id, domain.first_iteration, domain.iteration_count)


@dataclass(frozen=True, slots=True)
class PhaseTemplate:
    phase_id: PhaseId
    source_op_id: OpId
    implementation_id: str
    phase_name: str
    component_id: str
    domain: PhaseIterationDomain
    profile: ProfileRequirement
    warp_ids: tuple[int, ...]
    temporal_demands: tuple[TemporalDemand, ...]

    def __post_init__(self) -> None:
        for value, label in (
            (self.phase_id, "phase_id"),
            (self.source_op_id, "source_op_id"),
            (self.implementation_id, "implementation_id"),
            (self.component_id, "component_id"),
        ):
            _identifier(value, label)
        _text(self.phase_name, "phase_name")
        if type(self.domain) is not PhaseIterationDomain:
            raise WorkloadError("phase domain must be PhaseIterationDomain")
        if type(self.profile) is not ProfileRequirement:
            raise WorkloadError("phase profile must be ProfileRequirement")
        if not isinstance(self.warp_ids, (tuple, list)):
            raise WorkloadError("phase warp_ids must be a sequence")
        warp_ids = tuple(self.warp_ids)
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in warp_ids
        ):
            raise WorkloadError("phase warp IDs must be non-negative integers")
        if len(warp_ids) != len(set(warp_ids)):
            raise WorkloadError("phase warp IDs must be unique")
        demands = _tuple_of(self.temporal_demands, TemporalDemand, "temporal demands")
        if len({item.resource_id for item in demands}) != len(demands):
            raise WorkloadError("phase temporal resource IDs must be unique")
        if any(str(item.resource_id) == "b200.warp_issue" for item in demands) and not warp_ids:
            raise WorkloadError("warp_issue phases must bind at least one warp")
        object.__setattr__(self, "warp_ids", tuple(sorted(warp_ids)))
        object.__setattr__(
            self,
            "temporal_demands",
            tuple(sorted(demands, key=lambda item: str(item.resource_id))),
        )


@dataclass(frozen=True, slots=True)
class PhaseDependency:
    src_phase_id: PhaseId
    dst_phase_id: PhaseId
    relation: DependencyRelation
    delay_ps: int = 0

    def __post_init__(self) -> None:
        _identifier(self.src_phase_id, "src_phase_id")
        _identifier(self.dst_phase_id, "dst_phase_id")
        if type(self.relation) not in (AlignedRelation, EndpointRelation):
            raise WorkloadError("phase dependency relation must be typed")
        _non_negative(self.delay_ps, "delay_ps")


@dataclass(frozen=True, slots=True)
class PhaseStartAlignment:
    src_phase_id: PhaseId
    dst_phase_id: PhaseId
    offset_ps: int = 0

    def __post_init__(self) -> None:
        _identifier(self.src_phase_id, "src_phase_id")
        _identifier(self.dst_phase_id, "dst_phase_id")
        _non_negative(self.offset_ps, "offset_ps")
        if self.src_phase_id == self.dst_phase_id:
            raise WorkloadError("phase start alignment endpoints must be distinct")


@dataclass(frozen=True, slots=True)
class BufferTemplate:
    buffer_id: BufferId
    value_id: ValueId
    storage_resource_id: ResourceId
    bytes_per_slot: int
    slot_count: int
    producer_phase_id: PhaseId
    release_phase_ids: tuple[PhaseId, ...]

    def __post_init__(self) -> None:
        for value, label in (
            (self.buffer_id, "buffer_id"),
            (self.value_id, "value_id"),
            (self.storage_resource_id, "storage_resource_id"),
            (self.producer_phase_id, "producer_phase_id"),
        ):
            _identifier(value, label)
        _positive(self.bytes_per_slot, "bytes_per_slot")
        _positive(self.slot_count, "slot_count")
        if not isinstance(self.release_phase_ids, (tuple, list)):
            raise WorkloadError("release_phase_ids must be a sequence")
        releases = tuple(self.release_phase_ids)
        if not releases:
            raise WorkloadError("release_phase_ids must be non-empty and unique")
        for phase_id in releases:
            _identifier(phase_id, "release_phase_id")
        if len(releases) != len(set(releases)):
            raise WorkloadError("release_phase_ids must be non-empty and unique")
        object.__setattr__(self, "release_phase_ids", tuple(sorted(releases)))


@dataclass(frozen=True, slots=True)
class LoopTemplate:
    loop_id: LoopId
    iterations: int

    def __post_init__(self) -> None:
        _identifier(self.loop_id, "loop_id")
        _positive(self.iterations, "iterations")


@dataclass(frozen=True, slots=True)
class OpImplementationSelection:
    op_id: OpId
    implementation_id: str

    def __post_init__(self) -> None:
        _identifier(self.op_id, "op_id")
        _identifier(self.implementation_id, "implementation_id")


@dataclass(frozen=True, slots=True)
class ConfigurationTemplate:
    configuration_id: ConfigurationId
    program_id: ProgramId
    workload_kind: WorkloadKind
    tile: TileCandidate
    implementations: tuple[OpImplementationSelection, ...]
    warps: WarpConfig
    pipeline_depth: int
    layout_variant_id: str
    loops: tuple[LoopTemplate, ...]
    phases: tuple[PhaseTemplate, ...]
    dependencies: tuple[PhaseDependency, ...]
    loop_barriers: tuple[LoopBarrier, ...]
    start_alignments: tuple[PhaseStartAlignment, ...]
    buffers: tuple[BufferTemplate, ...]
    static_demands: tuple[StaticDemand, ...]

    def __post_init__(self) -> None:
        _identifier(self.configuration_id, "configuration_id")
        if len(self.configuration_id) != 64 or any(
            character not in "0123456789abcdef" for character in self.configuration_id
        ):
            raise WorkloadError("configuration_id must be a lowercase SHA-256 digest")
        _identifier(self.program_id, "program_id")
        _enum(self, "workload_kind", WorkloadKind)
        if type(self.tile) is not TileCandidate:
            raise WorkloadError("configuration tile must be TileCandidate")
        if type(self.warps) is not WarpConfig:
            raise WorkloadError("configuration warps must be WarpConfig")
        _positive(self.pipeline_depth, "pipeline_depth")
        _identifier(self.layout_variant_id, "layout_variant_id")
        fields = (
            ("implementations", OpImplementationSelection),
            ("loops", LoopTemplate),
            ("phases", PhaseTemplate),
            ("dependencies", PhaseDependency),
            ("loop_barriers", LoopBarrier),
            ("start_alignments", PhaseStartAlignment),
            ("buffers", BufferTemplate),
            ("static_demands", StaticDemand),
        )
        for field_name, item_type in fields:
            values = _tuple_of(getattr(self, field_name), item_type, field_name)
            object.__setattr__(
                self,
                field_name,
                tuple(sorted(values, key=lambda item: template_canonical_json(item))),
            )
        _unique(self.implementations, "op_id", "implementation op IDs")
        _unique(self.loops, "loop_id", "loop IDs")
        _unique(self.phases, "phase_id", "phase IDs")
        _unique(self.buffers, "buffer_id", "buffer IDs")
        _unique(self.static_demands, "resource_id", "static resource IDs")
        _unique(self.loop_barriers, "barrier_id", "barrier IDs")
        alignment_keys = tuple(
            frozenset((item.src_phase_id, item.dst_phase_id)) for item in self.start_alignments
        )
        if len(alignment_keys) != len(set(alignment_keys)):
            raise WorkloadError("phase start alignments must be unique")
        phase_map = {item.phase_id: item for item in self.phases}
        loop_map = {item.loop_id: item for item in self.loops}
        implementation_map = {item.op_id: item.implementation_id for item in self.implementations}
        allowed_warps = {
            warp_id for assignment in self.warps.roles for warp_id in assignment.warp_ids
        }
        for phase in self.phases:
            if phase.domain.loop_id is not None:
                loop = loop_map.get(phase.domain.loop_id)
                if loop is None:
                    raise WorkloadError("phase references an unknown loop")
                if phase.domain.first_iteration + phase.domain.iteration_count > loop.iterations:
                    raise WorkloadError("phase domain exceeds configuration loop range")
            if implementation_map.get(phase.source_op_id) != phase.implementation_id:
                raise WorkloadError("phase does not match its operation implementation selection")
            if any(warp_id not in allowed_warps for warp_id in phase.warp_ids):
                raise WorkloadError("phase binds a warp outside the selected configuration")
            query = phase.profile.query
            if query.implementation_id != phase.implementation_id:
                raise WorkloadError("phase profile implementation must match the phase")
            if query.component_id != phase.component_id:
                raise WorkloadError("phase profile component must match the phase")
            if query.warp_config_id != self.warps.config_id:
                raise WorkloadError("phase profile warp configuration must match the candidate")
            if query.pipeline_depth != self.pipeline_depth:
                raise WorkloadError("phase profile depth must match the candidate")
            if query.layout_variant_id != self.layout_variant_id:
                raise WorkloadError("phase profile layout must match the candidate")
        for dep in self.dependencies:
            if dep.src_phase_id not in phase_map or dep.dst_phase_id not in phase_map:
                raise WorkloadError("phase dependency references an unknown phase")
            if dep.src_phase_id == dep.dst_phase_id and isinstance(dep.relation, EndpointRelation):
                raise WorkloadError("endpoint phase self-dependencies are invalid")
            if (
                dep.src_phase_id == dep.dst_phase_id
                and isinstance(dep.relation, AlignedRelation)
                and dep.relation.iteration_distance == 0
            ):
                raise WorkloadError("zero-distance aligned phase self-dependencies are invalid")
            src = phase_map[dep.src_phase_id]
            dst = phase_map[dep.dst_phase_id]
            if isinstance(dep.relation, AlignedRelation):
                if src.domain.loop_id is None or src.domain.loop_id != dst.domain.loop_id:
                    raise WorkloadError("aligned phase dependency requires one shared loop")
                destination_iterations = set(
                    range(
                        dst.domain.first_iteration,
                        dst.domain.first_iteration + dst.domain.iteration_count,
                    )
                )
                if not any(
                    iteration + dep.relation.iteration_distance in destination_iterations
                    for iteration in range(
                        src.domain.first_iteration,
                        src.domain.first_iteration + src.domain.iteration_count,
                    )
                ):
                    raise WorkloadError("aligned phase dependency has no valid instances")
        _validate_phase_graph(self.dependencies)
        for alignment in self.start_alignments:
            alignment_src = phase_map.get(alignment.src_phase_id)
            alignment_dst = phase_map.get(alignment.dst_phase_id)
            if alignment_src is None or alignment_dst is None:
                raise WorkloadError("phase alignment references an unknown phase")
            if alignment_src.domain != alignment_dst.domain:
                raise WorkloadError("phase alignment requires identical domains")
        _validate_async_phase_pairs(self.phases, self.start_alignments)
        for barrier in self.loop_barriers:
            if barrier.src_loop_id not in loop_map or barrier.dst_loop_id not in loop_map:
                raise WorkloadError("loop barrier references an unknown loop")
        _validate_template_barriers(self.loop_barriers)
        for buffer in self.buffers:
            if buffer.producer_phase_id not in phase_map or any(
                item not in phase_map for item in buffer.release_phase_ids
            ):
                raise WorkloadError("buffer references an unknown phase")
        if any(
            str(demand.resource_id) == "b200.warp_issue" and not phase.warp_ids
            for phase in self.phases
            for demand in phase.temporal_demands
        ):
            raise WorkloadError("warp_issue phases must bind at least one warp")
        if self.pipeline_depth > 1 and not any(buffer.slot_count > 1 for buffer in self.buffers):
            raise WorkloadError("pipeline depth greater than one requires ring storage")
        if any(
            buffer.slot_count != 1 and buffer.slot_count != self.pipeline_depth
            for buffer in self.buffers
        ):
            raise WorkloadError("buffer slot count must be one or the configuration pipeline depth")

    def canonical_payload(self) -> dict[str, object]:
        """Return a deterministic identity payload without relying on repr."""

        return {
            "program_id": self.program_id,
            "workload_kind": self.workload_kind.value,
            "tile": _canonical_value(self.tile),
            "implementations": [_canonical_value(item) for item in self.implementations],
            "warp_config_id": self.warps.config_id,
            "warp_shape": [
                [assignment.role.value, list(assignment.warp_ids)]
                for assignment in self.warps.roles
            ],
            "pipeline_depth": self.pipeline_depth,
            "layout_variant_id": self.layout_variant_id,
            "loops": [_canonical_value(item) for item in self.loops],
            "phases": [_canonical_value(item) for item in self.phases],
            "dependencies": [_canonical_value(item) for item in self.dependencies],
            "loop_barriers": [_canonical_value(item) for item in self.loop_barriers],
            "start_alignments": [_canonical_value(item) for item in self.start_alignments],
            "buffers": [_canonical_value(item) for item in self.buffers],
            "static_demands": [_canonical_value(item) for item in self.static_demands],
        }


@dataclass(frozen=True, slots=True)
class LoweringContext:
    program: TileProgram
    hardware: HardwareSpec
    warps: WarpConfig
    pipeline_depth: int
    layout_variant_id: str

    def __post_init__(self) -> None:
        if type(self.program) is not TileProgram:
            raise WorkloadError("lowering context program must be TileProgram")
        if type(self.hardware) is not HardwareSpec:
            raise HardwareSpecError("lowering context hardware must be HardwareSpec")
        if type(self.warps) is not WarpConfig:
            raise WorkloadError("lowering context warps must be WarpConfig")
        _positive(self.pipeline_depth, "pipeline_depth")
        _identifier(self.layout_variant_id, "layout_variant_id")


@dataclass(frozen=True, slots=True)
class LoweredTileOp:
    source_op_id: OpId
    implementation_id: str
    phases: tuple[PhaseTemplate, ...]
    internal_dependencies: tuple[PhaseDependency, ...]
    internal_start_alignments: tuple[PhaseStartAlignment, ...]
    produced_values: tuple[ProducedValue, ...]
    consumed_values: tuple[ConsumedValue, ...]
    value_storage: tuple[ValueStorage, ...]
    static_demands: tuple[StaticDemand, ...]

    def __post_init__(self) -> None:
        _identifier(self.source_op_id, "source_op_id")
        _identifier(self.implementation_id, "implementation_id")
        for field_name, item_type in (
            ("phases", PhaseTemplate),
            ("internal_dependencies", PhaseDependency),
            ("internal_start_alignments", PhaseStartAlignment),
            ("produced_values", ProducedValue),
            ("consumed_values", ConsumedValue),
            ("value_storage", ValueStorage),
            ("static_demands", StaticDemand),
        ):
            object.__setattr__(
                self,
                field_name,
                tuple(_tuple_of(getattr(self, field_name), item_type, field_name)),
            )
        phase_ids = {phase.phase_id for phase in self.phases}
        if not self.phases:
            raise WorkloadError("a lowered operation must contain at least one phase")
        if len(phase_ids) != len(self.phases):
            raise WorkloadError("lowered phase IDs must be unique")
        if any(
            phase.source_op_id != self.source_op_id
            or phase.implementation_id != self.implementation_id
            for phase in self.phases
        ):
            raise WorkloadError("lowered phases must retain source operation and implementation")
        for dep in self.internal_dependencies:
            if dep.src_phase_id not in phase_ids or dep.dst_phase_id not in phase_ids:
                raise WorkloadError("internal dependency references an unknown phase")
        for alignment in self.internal_start_alignments:
            if alignment.src_phase_id not in phase_ids or alignment.dst_phase_id not in phase_ids:
                raise WorkloadError("internal alignment references an unknown phase")
        for produced in self.produced_values:
            if produced.ready_phase_id not in phase_ids:
                raise WorkloadError("availability references an unknown phase")
        for consumed in self.consumed_values:
            if consumed.consume_phase_id not in phase_ids:
                raise WorkloadError("availability references an unknown phase")
            if consumed.release_phase_id not in phase_ids:
                raise WorkloadError("availability release references an unknown phase")
        if any(item.allocation_phase_id not in phase_ids for item in self.value_storage):
            raise WorkloadError("value storage references an unknown phase")
        produced_keys = tuple(
            (item.value_id, item.availability_id) for item in self.produced_values
        )
        if len(produced_keys) != len(set(produced_keys)):
            raise WorkloadError("produced availabilities must be unique")
        consumed_values = tuple(item.value_id for item in self.consumed_values)
        if len(consumed_values) != len(set(consumed_values)):
            raise WorkloadError("each lowered value may have one required availability")
        storage_values = tuple(item.value_id for item in self.value_storage)
        if len(storage_values) != len(set(storage_values)):
            raise WorkloadError("value storage records must be unique by value")
        static_ids = tuple(item.resource_id for item in self.static_demands)
        if len(static_ids) != len(set(static_ids)):
            raise WorkloadError("lowering static resources must be unique")


class TileOpLowering(Protocol):
    @property
    def op_kind(self) -> TileOpKind: ...

    @property
    def implementation_id(self) -> str: ...

    def supports(self, op: TileOp, *, context: LoweringContext) -> bool: ...

    def lower(self, op: TileOp, *, context: LoweringContext) -> LoweredTileOp: ...


def _unique(values: tuple[object, ...], attr: str, label: str) -> None:
    ids = tuple(str(getattr(item, attr)) for item in values)
    if len(ids) != len(set(ids)):
        raise WorkloadError(f"{label} must be unique")


def _validate_template_barriers(barriers: tuple[LoopBarrier, ...]) -> None:
    graph: dict[LoopId, list[LoopId]] = {}
    for barrier in barriers:
        graph.setdefault(barrier.src_loop_id, []).append(barrier.dst_loop_id)
    state: dict[LoopId, int] = {}

    def visit(loop_id: LoopId) -> None:
        state[loop_id] = 1
        for destination in graph.get(loop_id, ()):
            marker = state.get(destination, 0)
            if marker == 1:
                raise WorkloadError("configuration loop barrier graph must be acyclic")
            if marker == 0:
                visit(destination)
        state[loop_id] = 2

    for loop_id in tuple(graph):
        if state.get(loop_id, 0) == 0:
            visit(loop_id)


def _validate_phase_graph(dependencies: tuple[PhaseDependency, ...]) -> None:
    graph: dict[PhaseId, list[PhaseId]] = {}
    for dependency in dependencies:
        if dependency.src_phase_id != dependency.dst_phase_id:
            graph.setdefault(dependency.src_phase_id, []).append(dependency.dst_phase_id)
    state: dict[PhaseId, int] = {}

    def visit(phase_id: PhaseId) -> None:
        state[phase_id] = 1
        for destination in graph.get(phase_id, ()):
            marker = state.get(destination, 0)
            if marker == 1:
                raise WorkloadError("phase dependency graph must be acyclic")
            if marker == 0:
                visit(destination)
        state[phase_id] = 2

    for phase_id in tuple(graph):
        if state.get(phase_id, 0) == 0:
            visit(phase_id)


def _validate_async_phase_pairs(
    phases: tuple[PhaseTemplate, ...],
    alignments: tuple[PhaseStartAlignment, ...],
) -> None:
    groups: dict[tuple[OpId, str, str], list[PhaseTemplate]] = {}
    for phase in phases:
        groups.setdefault(
            (phase.source_op_id, phase.implementation_id, phase.component_id), []
        ).append(phase)
    zero_aligned_pairs = {
        frozenset((alignment.src_phase_id, alignment.dst_phase_id))
        for alignment in alignments
        if alignment.offset_ps == 0
    }
    for group in groups.values():
        issue = tuple(
            phase
            for phase in group
            if phase.profile.timing_metric is TimingMetric.INITIATION_INTERVAL
        )
        latency = tuple(
            phase for phase in group if phase.profile.timing_metric is TimingMetric.LATENCY
        )
        if not issue or not latency:
            continue
        if len(issue) != 1 or len(latency) != 1:
            raise WorkloadError("one asynchronous query must map to one issue and latency phase")
        if issue[0].profile.query != latency[0].profile.query:
            raise WorkloadError(
                "asynchronous issue and latency phases must share one profile query"
            )
        pair = frozenset((issue[0].phase_id, latency[0].phase_id))
        if pair not in zero_aligned_pairs:
            raise WorkloadError("asynchronous issue and latency phases must have equal starts")


# Expose a tiny helper for the builder without making it part of the public root.
def template_canonical_json(value: object) -> str:
    return json.dumps(
        _canonical_value(value), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


__all__ = [
    "AlignedRelation",
    "BenchmarkFingerprint",
    "BufferTemplate",
    "CanonicalAttribute",
    "ConfigurationTemplate",
    "ConsumedValue",
    "DependencyRelation",
    "EndpointRelation",
    "LoweredTileOp",
    "LoweringContext",
    "LoopTemplate",
    "OpImplementationSelection",
    "PhaseDependency",
    "PhaseIterationDomain",
    "PhaseStartAlignment",
    "PhaseTemplate",
    "ProducedValue",
    "ProfileRequirement",
    "StaticDemand",
    "TemporalDemand",
    "TimingMetric",
    "TileOpLowering",
    "TileOpProfileKey",
    "TileOpProfileQuery",
    "TileOpSignature",
    "ValueStorage",
    "ValueStoragePolicy",
    "profile_query_canonical_json",
    "profile_key_from_json",
    "profile_key_to_json",
    "tile_op_signature",
]


def profile_query_canonical_json(query: TileOpProfileQuery) -> str:
    """Functional spelling for query canonicalization."""

    if type(query) is not TileOpProfileQuery:
        raise ProfileStoreError("query must be TileOpProfileQuery")
    return query.canonical_json()


def profile_key_from_json(text: str) -> TileOpProfileKey:
    """Decode one strict profile key through the shared typed decoder."""

    from ._serialization import _decode_profile_key, _loads

    return _decode_profile_key(_loads(text, ProfileStoreError))


def profile_key_to_json(key: TileOpProfileKey) -> str:
    """Encode one exact profile key using its canonical bytes."""

    if type(key) is not TileOpProfileKey:
        raise ProfileStoreError("profile key must be TileOpProfileKey")
    text = key.canonical_json()
    return profile_key_from_json(text).canonical_json()
