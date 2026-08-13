"""Immutable result and selected-plan records for the public API."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum

from .constants import PLAN_SCHEMA_VERSION, RESULT_SCHEMA_VERSION
from .errors import InvalidRequestError
from .model import (
    BufferId,
    ConfigurationId,
    FlashAttentionSpec,
    GemmSpec,
    GqaDecodeSpec,
    HardwareSpecRef,
    LoopId,
    MeasurementId,
    MlpSpec,
    OpId,
    PhaseId,
    ProfileKeyId,
    ProgramId,
    ResourceId,
    TimingMetric,
    ValueId,
    WorkloadSpec,
    validate_identifier,
)
from .program import TileCandidate, TileProgram
from .request import ProfileSnapshotRef, TimingStatistic, WarpConfig


class TimelineRegion(str, Enum):
    """Name one exported schedule region."""

    FINITE = "finite"
    PROLOGUE = "prologue"
    STEADY = "steady"
    EPILOGUE = "epilogue"


class EvaluationStatus(str, Enum):
    """Name one public cost-model outcome."""

    OPTIMAL = "optimal"
    FEASIBLE = "feasible"
    TIMEOUT = "timeout"
    INFEASIBLE = "infeasible"
    UNSUPPORTED = "unsupported"
    MISSING_PROFILE = "missing_profile"
    PROFILE_FAILED = "profile_failed"


class DiagnosticCode(str, Enum):
    """Name one stable diagnostic category."""

    INVALID_CANDIDATE = "invalid_candidate"
    STATIC_CAPACITY = "static_capacity"
    MISSING_PROFILE = "missing_profile"
    PROFILE_FAILED = "profile_failed"
    UNSUPPORTED = "unsupported"
    INFEASIBLE = "infeasible"
    TIMEOUT = "timeout"


@dataclass(frozen=True, slots=True)
class Diagnostic:
    code: DiagnosticCode
    message: str
    subject_id: str | None = None

    def __post_init__(self) -> None:
        _coerce_enum(self, "code", DiagnosticCode)
        if not isinstance(self.message, str) or not self.message:
            raise InvalidRequestError("diagnostic message must be non-empty")
        if self.subject_id is not None:
            _result_identifier(self.subject_id, "diagnostic subject_id")


@dataclass(frozen=True, slots=True)
class RejectedCandidate:
    configuration_id: ConfigurationId | None
    code: DiagnosticCode
    message: str

    def __post_init__(self) -> None:
        _coerce_enum(self, "code", DiagnosticCode)
        if self.configuration_id is not None:
            _result_identifier(self.configuration_id, "configuration_id")
        if not isinstance(self.message, str) or not self.message:
            raise InvalidRequestError("candidate diagnostic message must be non-empty")


@dataclass(frozen=True, slots=True)
class ResourceReservation:
    resource_id: ResourceId
    slots: int

    def __post_init__(self) -> None:
        _result_identifier(self.resource_id, "resource_id")
        if isinstance(self.slots, bool) or not isinstance(self.slots, int) or self.slots <= 0:
            raise InvalidRequestError("resource slots must be positive")


@dataclass(frozen=True, slots=True)
class PhasePlacement:
    phase_id: PhaseId
    source_op_id: OpId
    loop_id: LoopId | None
    region: TimelineRegion
    iteration: int
    start_ps: int
    end_ps: int
    warp_ids: tuple[int, ...]
    resources: tuple[ResourceReservation, ...]

    def __post_init__(self) -> None:
        _result_identifier(self.phase_id, "phase_id")
        _result_identifier(self.source_op_id, "source_op_id")
        _coerce_enum(self, "region", TimelineRegion)
        if self.loop_id is not None:
            _result_identifier(self.loop_id, "loop_id")
        if isinstance(self.iteration, bool) or not isinstance(self.iteration, int):
            raise InvalidRequestError("placement iteration must be an integer")
        if (
            any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in (self.start_ps, self.end_ps)
            )
            or self.end_ps < self.start_ps
        ):
            raise InvalidRequestError("placement times must be non-negative and ordered")
        warp_ids = tuple(self.warp_ids)
        resources = tuple(self.resources)
        object.__setattr__(self, "warp_ids", warp_ids)
        object.__setattr__(self, "resources", resources)
        if len(warp_ids) != len(set(warp_ids)) or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in warp_ids
        ):
            raise InvalidRequestError("placement warp_ids must be unique non-negative integers")
        if not all(isinstance(item, ResourceReservation) for item in resources):
            raise InvalidRequestError(
                "placement resources must contain ResourceReservation records"
            )


@dataclass(frozen=True, slots=True)
class BufferAllocation:
    buffer_id: BufferId
    value_id: ValueId
    storage_resource_id: ResourceId
    bytes_per_slot: int
    slot_count: int
    total_bytes: int

    def __post_init__(self) -> None:
        for identifier_value, label in (
            (self.buffer_id, "buffer_id"),
            (self.value_id, "value_id"),
            (self.storage_resource_id, "storage_resource_id"),
        ):
            _result_identifier(identifier_value, label)
        for quantity, label in (
            (self.bytes_per_slot, "bytes_per_slot"),
            (self.slot_count, "slot_count"),
            (self.total_bytes, "total_bytes"),
        ):
            if isinstance(quantity, bool) or not isinstance(quantity, int) or quantity <= 0:
                raise InvalidRequestError(f"{label} must be positive")
        if self.total_bytes != self.bytes_per_slot * self.slot_count:
            raise InvalidRequestError("total_bytes must equal bytes_per_slot * slot_count")


@dataclass(frozen=True, slots=True)
class ResourceUtilization:
    resource_id: ResourceId
    capacity_slots: int
    busy_slot_ps: int
    horizon_ps: int

    def __post_init__(self) -> None:
        _result_identifier(self.resource_id, "resource_id")
        for quantity, label in (
            (self.capacity_slots, "capacity_slots"),
            (self.busy_slot_ps, "busy_slot_ps"),
            (self.horizon_ps, "horizon_ps"),
        ):
            if isinstance(quantity, bool) or not isinstance(quantity, int) or quantity < 0:
                raise InvalidRequestError(f"{label} must be a non-negative integer")
        if self.capacity_slots <= 0:
            raise InvalidRequestError("capacity_slots must be positive")
        if self.busy_slot_ps > self.capacity_slots * self.horizon_ps:
            raise InvalidRequestError("busy_slot_ps exceeds resource capacity over horizon")


@dataclass(frozen=True, slots=True)
class LoopTiming:
    loop_id: LoopId
    initiation_interval_ps: int | None
    prologue_ps: int
    epilogue_ps: int
    span_ps: int

    def __post_init__(self) -> None:
        _result_identifier(self.loop_id, "loop_id")
        for quantity, label in (
            (self.prologue_ps, "prologue_ps"),
            (self.epilogue_ps, "epilogue_ps"),
            (self.span_ps, "span_ps"),
        ):
            if isinstance(quantity, bool) or not isinstance(quantity, int) or quantity < 0:
                raise InvalidRequestError(f"{label} must be non-negative")
        if self.initiation_interval_ps is not None and (
            isinstance(self.initiation_interval_ps, bool)
            or not isinstance(self.initiation_interval_ps, int)
            or self.initiation_interval_ps <= 0
        ):
            raise InvalidRequestError("initiation_interval_ps must be positive or None")


@dataclass(frozen=True, slots=True)
class ProfileProvenance:
    phase_id: PhaseId
    source_op_id: OpId
    implementation_id: str
    phase_name: str
    component_id: str
    measurement_id: MeasurementId
    profile_key_id: ProfileKeyId
    environment_id: str
    timing_metric: TimingMetric
    statistic: TimingStatistic
    sensitivity_statistic: TimingStatistic

    def __post_init__(self) -> None:
        for text, label in (
            (self.phase_id, "phase_id"),
            (self.source_op_id, "source_op_id"),
            (self.implementation_id, "implementation_id"),
            (self.phase_name, "phase_name"),
            (self.component_id, "component_id"),
            (self.measurement_id, "measurement_id"),
            (self.profile_key_id, "profile_key_id"),
            (self.environment_id, "environment_id"),
        ):
            _result_identifier(text, label)
        _coerce_enum(self, "timing_metric", TimingMetric)
        _coerce_enum(self, "statistic", TimingStatistic)
        _coerce_enum(self, "sensitivity_statistic", TimingStatistic)


@dataclass(frozen=True, slots=True)
class SolveProof:
    status: EvaluationStatus
    objective_ps: int
    lower_bound_ps: int
    sensitivity_ps: int
    optimality_gap_ppm: int
    solver_name: str
    solver_version: str
    candidate_count: int
    solved_candidate_count: int
    rejected_candidate_count: int

    def __post_init__(self) -> None:
        _coerce_enum(self, "status", EvaluationStatus)
        for quantity, label in (
            (self.objective_ps, "objective_ps"),
            (self.lower_bound_ps, "lower_bound_ps"),
            (self.sensitivity_ps, "sensitivity_ps"),
            (self.optimality_gap_ppm, "optimality_gap_ppm"),
            (self.candidate_count, "candidate_count"),
            (self.solved_candidate_count, "solved_candidate_count"),
            (self.rejected_candidate_count, "rejected_candidate_count"),
        ):
            if isinstance(quantity, bool) or not isinstance(quantity, int) or quantity < 0:
                raise InvalidRequestError(f"{label} must be a non-negative integer")
        for text, label in (
            (self.solver_name, "solver_name"),
            (self.solver_version, "solver_version"),
        ):
            _result_identifier(text, label)


@dataclass(frozen=True, slots=True)
class SelectedImplementation:
    """Record one operation implementation in the selected configuration."""

    op_id: OpId
    implementation_id: str

    def __post_init__(self) -> None:
        _result_identifier(self.op_id, "op_id")
        _result_identifier(self.implementation_id, "implementation_id")


@dataclass(frozen=True, slots=True)
class SelectedStaticDemand:
    """Record one selected per-CTA static resource demand."""

    resource_id: ResourceId
    units: int

    def __post_init__(self) -> None:
        _result_identifier(self.resource_id, "resource_id")
        if isinstance(self.units, bool) or not isinstance(self.units, int) or self.units <= 0:
            raise InvalidRequestError("static demand units must be positive")


@dataclass(frozen=True, slots=True)
class SelectedConfiguration:
    configuration_id: ConfigurationId
    program_id: ProgramId
    tile: TileCandidate
    implementations: tuple[SelectedImplementation, ...]
    warps: WarpConfig
    pipeline_depth: int
    layout_variant_id: str
    static_demands: tuple[SelectedStaticDemand, ...]

    def __post_init__(self) -> None:
        _result_identifier(self.configuration_id, "configuration_id")
        _result_identifier(self.program_id, "program_id")
        if not isinstance(self.tile, TileCandidate):
            raise InvalidRequestError("tile must be TileCandidate")
        if not isinstance(self.warps, WarpConfig):
            raise InvalidRequestError("warps must be WarpConfig")
        if (
            isinstance(self.pipeline_depth, bool)
            or not isinstance(self.pipeline_depth, int)
            or self.pipeline_depth <= 0
        ):
            raise InvalidRequestError("pipeline_depth must be positive")
        _result_identifier(self.layout_variant_id, "layout_variant_id")
        implementations = tuple(
            _coerce_selected_implementation(item) for item in self.implementations
        )
        static_demands = tuple(_coerce_selected_static_demand(item) for item in self.static_demands)
        object.__setattr__(self, "implementations", implementations)
        object.__setattr__(self, "static_demands", static_demands)
        if not all(isinstance(item, SelectedImplementation) for item in implementations):
            raise InvalidRequestError("implementations must contain SelectedImplementation records")
        if not all(isinstance(item, SelectedStaticDemand) for item in static_demands):
            raise InvalidRequestError("static_demands must contain SelectedStaticDemand records")
        op_ids = tuple(item.op_id for item in implementations)
        if len(op_ids) != len(set(op_ids)):
            raise InvalidRequestError("selected implementation op IDs must be unique")
        resource_ids = tuple(item.resource_id for item in static_demands)
        if len(resource_ids) != len(set(resource_ids)):
            raise InvalidRequestError("selected static resource IDs must be unique")


@dataclass(frozen=True, slots=True)
class CostModelPlan:
    schema_version: int
    request_id: str
    hardware: HardwareSpecRef
    profile_snapshot: ProfileSnapshotRef
    workload: WorkloadSpec
    program: TileProgram
    selected: SelectedConfiguration
    end_to_end_ps: int
    loop_timings: tuple[LoopTiming, ...]
    placements: tuple[PhasePlacement, ...]
    buffers: tuple[BufferAllocation, ...]
    utilization: tuple[ResourceUtilization, ...]
    profiles: tuple[ProfileProvenance, ...]
    proof: SolveProof

    def __post_init__(self) -> None:
        if (
            isinstance(self.schema_version, bool)
            or not isinstance(self.schema_version, int)
            or self.schema_version != PLAN_SCHEMA_VERSION
        ):
            raise InvalidRequestError(f"unsupported plan schema version: {self.schema_version!r}")
        _result_identifier(self.request_id, "request_id")
        if not isinstance(self.hardware, HardwareSpecRef):
            raise InvalidRequestError("hardware must be HardwareSpecRef")
        if not isinstance(self.profile_snapshot, ProfileSnapshotRef):
            raise InvalidRequestError("profile_snapshot must be ProfileSnapshotRef")
        if not isinstance(self.workload, (GemmSpec, GqaDecodeSpec, FlashAttentionSpec, MlpSpec)):
            raise InvalidRequestError("workload must be a typed workload record")
        if not isinstance(self.program, TileProgram):
            raise InvalidRequestError("program must be TileProgram")
        if not isinstance(self.selected, SelectedConfiguration):
            raise InvalidRequestError("selected must be SelectedConfiguration")
        if self.selected.program_id != self.program.program_id:
            raise InvalidRequestError("selected program_id does not match plan program")
        if (
            isinstance(self.end_to_end_ps, bool)
            or not isinstance(self.end_to_end_ps, int)
            or self.end_to_end_ps < 0
        ):
            raise InvalidRequestError("end_to_end_ps must be a non-negative integer")
        _set_record_tuple(self, "loop_timings", LoopTiming)
        _set_record_tuple(self, "placements", PhasePlacement)
        _set_record_tuple(self, "buffers", BufferAllocation)
        _set_record_tuple(self, "utilization", ResourceUtilization)
        _set_record_tuple(self, "profiles", ProfileProvenance)
        if not isinstance(self.proof, SolveProof):
            raise InvalidRequestError("proof must be SolveProof")

    def to_json(self) -> str:
        from ._serialization import plan_to_json

        return plan_to_json(self)

    @classmethod
    def from_json(cls, text: str) -> "CostModelPlan":
        from ._serialization import plan_from_json

        return plan_from_json(text)


@dataclass(frozen=True, slots=True)
class CostModelResult:
    schema_version: int
    status: EvaluationStatus
    plan: CostModelPlan | None = None
    missing_profiles: tuple[ProfileKeyId, ...] = ()
    rejected_candidates: tuple[RejectedCandidate, ...] = ()
    diagnostics: tuple[Diagnostic, ...] = ()

    def __post_init__(self) -> None:
        if (
            isinstance(self.schema_version, bool)
            or not isinstance(self.schema_version, int)
            or self.schema_version != RESULT_SCHEMA_VERSION
        ):
            raise InvalidRequestError(f"unsupported result schema version: {self.schema_version!r}")
        _coerce_enum(self, "status", EvaluationStatus)
        missing_profiles = tuple(self.missing_profiles)
        object.__setattr__(self, "missing_profiles", missing_profiles)
        _set_record_tuple(self, "rejected_candidates", RejectedCandidate)
        _set_record_tuple(self, "diagnostics", Diagnostic)
        if self.plan is not None and not isinstance(self.plan, CostModelPlan):
            raise InvalidRequestError("plan must be CostModelPlan or None")
        for key_id in self.missing_profiles:
            _result_identifier(key_id, "profile key ID")
        if (
            self.status in (EvaluationStatus.OPTIMAL, EvaluationStatus.FEASIBLE)
            and self.plan is None
        ):
            raise InvalidRequestError(f"{self.status.value} result requires a plan")
        if (
            self.status
            in (
                EvaluationStatus.INFEASIBLE,
                EvaluationStatus.UNSUPPORTED,
                EvaluationStatus.MISSING_PROFILE,
                EvaluationStatus.PROFILE_FAILED,
            )
            and self.plan is not None
        ):
            raise InvalidRequestError(f"{self.status.value} result cannot contain a plan")

    def to_json(self) -> str:
        from ._serialization import result_to_json

        return result_to_json(self)

    @classmethod
    def from_json(cls, text: str) -> "CostModelResult":
        from ._serialization import result_from_json

        return result_from_json(text)


def _coerce_enum(instance: object, field_name: str, enum_type: type[Enum]) -> None:
    value = getattr(instance, field_name)
    if isinstance(value, enum_type):
        return
    try:
        object.__setattr__(instance, field_name, enum_type(value))
    except (TypeError, ValueError) as exc:
        raise InvalidRequestError(f"invalid {field_name}") from exc


def _result_identifier(value: object, label: str) -> None:
    if not isinstance(value, str):
        raise InvalidRequestError(f"{label} must be a non-empty ASCII string")
    validate_identifier(value, label=label)


def _set_record_tuple(instance: object, field_name: str, item_type: type[object]) -> None:
    values = tuple(getattr(instance, field_name))
    object.__setattr__(instance, field_name, values)
    if not all(isinstance(value, item_type) for value in values):
        raise InvalidRequestError(f"{field_name} must contain {item_type.__name__} records")


def _coerce_selected_implementation(value: object) -> SelectedImplementation:
    if isinstance(value, SelectedImplementation):
        return value
    if not isinstance(value, Mapping) or set(value) != {"op_id", "implementation_id"}:
        raise InvalidRequestError("invalid selected implementation record")
    op_id = value["op_id"]
    implementation_id = value["implementation_id"]
    if not isinstance(op_id, str) or not isinstance(implementation_id, str):
        raise InvalidRequestError("selected implementation fields must be strings")
    return SelectedImplementation(OpId(op_id), implementation_id)


def _coerce_selected_static_demand(value: object) -> SelectedStaticDemand:
    if isinstance(value, SelectedStaticDemand):
        return value
    if not isinstance(value, Mapping) or set(value) != {"resource_id", "units"}:
        raise InvalidRequestError("invalid selected static demand record")
    resource_id = value["resource_id"]
    units = value["units"]
    if not isinstance(resource_id, str) or isinstance(units, bool) or not isinstance(units, int):
        raise InvalidRequestError("selected static demand fields have invalid types")
    return SelectedStaticDemand(ResourceId(resource_id), units)


__all__ = [
    "BufferAllocation",
    "CostModelPlan",
    "CostModelResult",
    "Diagnostic",
    "DiagnosticCode",
    "EvaluationStatus",
    "LoopTiming",
    "PhasePlacement",
    "ProfileProvenance",
    "RejectedCandidate",
    "ResourceReservation",
    "ResourceUtilization",
    "SelectedConfiguration",
    "SelectedImplementation",
    "SelectedStaticDemand",
    "SolveProof",
    "TimelineRegion",
]
