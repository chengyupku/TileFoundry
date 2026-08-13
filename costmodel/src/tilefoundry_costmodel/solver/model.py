"""Immutable solver-ready records for the version 2 problem boundary."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from ..constants import SEARCH_PROBLEM_SCHEMA_VERSION
from ..errors import InvalidRequestError, SearchProblemError
from ..hardware.model import HardwareSpec
from ..model import (
    BufferId,
    ConfigurationId,
    FlashAttentionSpec,
    GemmSpec,
    GqaDecodeSpec,
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
    WorkloadKind,
    WorkloadSpec,
    validate_identifier,
)
from ..program import (
    AlignedRelation,
    DependencyRelation,
    EndpointRelation,
    LoopBarrier,
    TileCandidate,
    TileProgram,
)
from ..request import (
    ProfileSnapshotRef,
    SolverOptions,
    TimingStatistic,
    WarpConfig,
    _canonical_program_key,
)
from ..result import RejectedCandidate


@dataclass(frozen=True, slots=True)
class TemporalDemand:
    """Reserve temporal resource slots for one complete phase interval."""

    resource_id: ResourceId
    slots: int

    def __post_init__(self) -> None:
        _identifier(self.resource_id, "resource_id")
        _positive_int(self.slots, "slots")


@dataclass(frozen=True, slots=True)
class StaticDemand:
    """Consume one static per-CTA capacity."""

    resource_id: ResourceId
    units: int

    def __post_init__(self) -> None:
        _identifier(self.resource_id, "resource_id")
        _positive_int(self.units, "units")


@dataclass(frozen=True, slots=True)
class PhaseIterationDomain:
    """Place a lowered phase over one source operation domain."""

    loop_id: LoopId | None
    first_iteration: int
    iteration_count: int

    def __post_init__(self) -> None:
        if self.loop_id is not None:
            _identifier(self.loop_id, "loop_id")
        _non_negative_int(self.first_iteration, "first_iteration")
        _positive_int(self.iteration_count, "iteration_count")
        if self.loop_id is None and (self.first_iteration, self.iteration_count) != (0, 1):
            raise SearchProblemError("a one-time phase domain must equal (None, 0, 1)")


@dataclass(frozen=True, slots=True)
class OpImplementationSelection:
    """Record one selected implementation for one source operation."""

    op_id: OpId
    implementation_id: str

    def __post_init__(self) -> None:
        _identifier(self.op_id, "op_id")
        _identifier(self.implementation_id, "implementation_id")


@dataclass(frozen=True, slots=True)
class LoopTemplate:
    """Describe one solver-ready repeated region."""

    loop_id: LoopId
    iterations: int

    def __post_init__(self) -> None:
        _identifier(self.loop_id, "loop_id")
        _positive_int(self.iterations, "iterations")


@dataclass(frozen=True, slots=True)
class Phase:
    """Carry one fully timed implementation phase without backend state."""

    phase_id: PhaseId
    source_op_id: OpId
    implementation_id: str
    phase_name: str
    component_id: str
    domain: PhaseIterationDomain
    duration_ticks: int
    sensitivity_duration_ticks: int
    warp_ids: tuple[int, ...]
    temporal_demands: tuple[TemporalDemand, ...]
    measurement_id: MeasurementId
    profile_key_id: ProfileKeyId
    environment_id: str
    timing_metric: TimingMetric
    timing_statistic: TimingStatistic
    sensitivity_timing_statistic: TimingStatistic

    def __post_init__(self) -> None:
        for value, label in (
            (self.phase_id, "phase_id"),
            (self.source_op_id, "source_op_id"),
            (self.implementation_id, "implementation_id"),
            (self.phase_name, "phase_name"),
            (self.component_id, "component_id"),
            (self.measurement_id, "measurement_id"),
            (self.profile_key_id, "profile_key_id"),
            (self.environment_id, "environment_id"),
        ):
            _identifier(value, label)
        if not isinstance(self.domain, PhaseIterationDomain):
            raise SearchProblemError("phase domain must be PhaseIterationDomain")
        _positive_int(self.duration_ticks, "duration_ticks")
        _positive_int(self.sensitivity_duration_ticks, "sensitivity_duration_ticks")
        warp_ids = tuple(self.warp_ids)
        object.__setattr__(self, "warp_ids", warp_ids)
        if len(warp_ids) != len(set(warp_ids)):
            raise SearchProblemError("phase warp_ids must be unique")
        for warp_id in warp_ids:
            _non_negative_int(warp_id, "warp_id")
        temporal_demands = tuple(self.temporal_demands)
        object.__setattr__(self, "temporal_demands", temporal_demands)
        if not all(isinstance(item, TemporalDemand) for item in temporal_demands):
            raise SearchProblemError("temporal_demands must contain TemporalDemand records")
        _coerce_enum(self, "timing_metric", TimingMetric)
        _coerce_enum(self, "timing_statistic", TimingStatistic)
        _coerce_enum(self, "sensitivity_timing_statistic", TimingStatistic)


@dataclass(frozen=True, slots=True)
class PhaseDependency:
    """Constrain one phase completion before another phase starts."""

    src_phase_id: PhaseId
    dst_phase_id: PhaseId
    relation: DependencyRelation
    delay_ps: int = 0

    def __post_init__(self) -> None:
        _identifier(self.src_phase_id, "src_phase_id")
        _identifier(self.dst_phase_id, "dst_phase_id")
        if not isinstance(self.relation, (AlignedRelation, EndpointRelation)):
            raise SearchProblemError("phase dependency relation must be typed")
        _non_negative_int(self.delay_ps, "delay_ps")


@dataclass(frozen=True, slots=True)
class PhaseStartAlignment:
    """Constrain corresponding phase starts by an exact offset."""

    src_phase_id: PhaseId
    dst_phase_id: PhaseId
    offset_ps: int = 0

    def __post_init__(self) -> None:
        _identifier(self.src_phase_id, "src_phase_id")
        _identifier(self.dst_phase_id, "dst_phase_id")
        _non_negative_int(self.offset_ps, "offset_ps")


@dataclass(frozen=True, slots=True)
class BufferTemplate:
    """Describe one static or ring allocation in a numeric candidate."""

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
        _positive_int(self.bytes_per_slot, "bytes_per_slot")
        _positive_int(self.slot_count, "slot_count")
        releases = tuple(self.release_phase_ids)
        object.__setattr__(self, "release_phase_ids", releases)
        if not releases or len(releases) != len(set(releases)):
            raise SearchProblemError("release_phase_ids must be non-empty and unique")
        for phase_id in releases:
            _identifier(phase_id, "release_phase_id")


@dataclass(frozen=True, slots=True)
class Configuration:
    """Carry one complete solver-ready numeric candidate."""

    configuration_id: ConfigurationId
    program_id: ProgramId
    workload_kind: WorkloadKind
    tile: TileCandidate
    implementations: tuple[OpImplementationSelection, ...]
    warps: WarpConfig
    pipeline_depth: int
    layout_variant_id: str
    loops: tuple[LoopTemplate, ...]
    phases: tuple[Phase, ...]
    dependencies: tuple[PhaseDependency, ...]
    loop_barriers: tuple[LoopBarrier, ...]
    start_alignments: tuple[PhaseStartAlignment, ...]
    buffers: tuple[BufferTemplate, ...]
    static_demands: tuple[StaticDemand, ...]

    def __post_init__(self) -> None:
        _identifier(self.configuration_id, "configuration_id")
        _identifier(self.program_id, "program_id")
        _coerce_enum(self, "workload_kind", WorkloadKind)
        if not isinstance(self.tile, TileCandidate):
            raise SearchProblemError("configuration tile must be TileCandidate")
        if not isinstance(self.warps, WarpConfig):
            raise SearchProblemError("configuration warps must be WarpConfig")
        _positive_int(self.pipeline_depth, "pipeline_depth")
        _identifier(self.layout_variant_id, "layout_variant_id")

        collections: tuple[tuple[str, type[object]], ...] = (
            ("implementations", OpImplementationSelection),
            ("loops", LoopTemplate),
            ("phases", Phase),
            ("dependencies", PhaseDependency),
            ("loop_barriers", LoopBarrier),
            ("start_alignments", PhaseStartAlignment),
            ("buffers", BufferTemplate),
            ("static_demands", StaticDemand),
        )
        for field_name, item_type in collections:
            values = tuple(getattr(self, field_name))
            object.__setattr__(self, field_name, values)
            if not all(isinstance(item, item_type) for item in values):
                raise SearchProblemError(f"{field_name} contains an invalid record")

        _unique(self.implementations, "op_id", "implementation op IDs")
        _unique(self.loops, "loop_id", "configuration loop IDs")
        _unique(self.phases, "phase_id", "configuration phase IDs")
        _unique(self.buffers, "buffer_id", "configuration buffer IDs")
        _unique(self.static_demands, "resource_id", "static resource IDs")

        phase_ids = {phase.phase_id for phase in self.phases}
        for dependency in self.dependencies:
            if dependency.src_phase_id not in phase_ids or dependency.dst_phase_id not in phase_ids:
                raise SearchProblemError("phase dependency references an unknown phase")
        for alignment in self.start_alignments:
            if alignment.src_phase_id not in phase_ids or alignment.dst_phase_id not in phase_ids:
                raise SearchProblemError("phase alignment references an unknown phase")
        for buffer in self.buffers:
            referenced = (buffer.producer_phase_id, *buffer.release_phase_ids)
            if any(phase_id not in phase_ids for phase_id in referenced):
                raise SearchProblemError("buffer references an unknown phase")
        loop_ids = {loop.loop_id for loop in self.loops}
        if any(
            barrier.src_loop_id not in loop_ids or barrier.dst_loop_id not in loop_ids
            for barrier in self.loop_barriers
        ):
            raise SearchProblemError("loop barrier references an unknown configuration loop")


@dataclass(frozen=True, slots=True)
class SearchProblem:
    """A complete replayable numeric search problem."""

    schema_version: int
    request_id: str
    hardware: HardwareSpec
    workload: WorkloadSpec
    programs: tuple[TileProgram, ...]
    profile_snapshot: ProfileSnapshotRef
    solver_options: SolverOptions
    configurations: tuple[Configuration, ...]
    rejected_before_solve: tuple[RejectedCandidate, ...]

    def __post_init__(self) -> None:
        if (
            isinstance(self.schema_version, bool)
            or not isinstance(self.schema_version, int)
            or self.schema_version != SEARCH_PROBLEM_SCHEMA_VERSION
        ):
            raise SearchProblemError(
                f"unsupported search-problem schema version: {self.schema_version!r}"
            )
        _identifier(self.request_id, "request_id")
        if not isinstance(self.hardware, HardwareSpec):
            raise SearchProblemError("hardware must be HardwareSpec")
        if not isinstance(self.workload, (GemmSpec, GqaDecodeSpec, FlashAttentionSpec, MlpSpec)):
            raise SearchProblemError("workload must be a typed workload record")
        programs = tuple(self.programs)
        object.__setattr__(self, "programs", programs)
        if not programs or not all(isinstance(program, TileProgram) for program in programs):
            raise SearchProblemError("programs must contain at least one TileProgram")
        program_ids = tuple(program.program_id for program in programs)
        if len(program_ids) != len(set(program_ids)):
            raise SearchProblemError("problem program IDs must be unique")
        try:
            canonical_programs = tuple(_canonical_program_key(program) for program in programs)
        except (InvalidRequestError, TypeError, ValueError) as exc:
            raise SearchProblemError("problem programs must have a canonical JSON form") from exc
        if len(canonical_programs) != len(set(canonical_programs)):
            raise SearchProblemError("problem contains duplicate canonical programs")
        if any(program.workload_kind != self.workload.kind for program in programs):
            raise SearchProblemError("problem program workload kinds must match the workload")
        if not isinstance(self.profile_snapshot, ProfileSnapshotRef):
            raise SearchProblemError("profile_snapshot must be ProfileSnapshotRef")
        if not isinstance(self.solver_options, SolverOptions):
            raise SearchProblemError("solver_options must be SolverOptions")

        raw_configurations = tuple(self.configurations)
        if not all(isinstance(item, Configuration) for item in raw_configurations):
            raise SearchProblemError("configurations must contain Configuration records")
        configurations = tuple(
            sorted(raw_configurations, key=lambda item: str(item.configuration_id))
        )
        object.__setattr__(self, "configurations", configurations)
        configuration_ids = tuple(item.configuration_id for item in configurations)
        if len(configuration_ids) != len(set(configuration_ids)):
            raise SearchProblemError("configuration IDs must be unique")
        program_id_set = set(program_ids)
        if any(item.program_id not in program_id_set for item in configurations):
            raise SearchProblemError("configuration references an unknown program")
        if any(item.workload_kind != self.workload.kind for item in configurations):
            raise SearchProblemError("configuration workload kinds must match the workload")

        raw_rejected = tuple(self.rejected_before_solve)
        if not all(isinstance(item, RejectedCandidate) for item in raw_rejected):
            raise SearchProblemError("rejected_before_solve must contain RejectedCandidate records")
        rejected = tuple(
            sorted(
                raw_rejected,
                key=lambda item: (
                    str(item.configuration_id or ""),
                    str(item.code),
                    item.message,
                ),
            )
        )
        object.__setattr__(self, "rejected_before_solve", rejected)

    def to_json(self) -> str:
        from .._serialization import problem_to_json

        return problem_to_json(self)

    @classmethod
    def from_json(cls, text: str) -> SearchProblem:
        from .._serialization import problem_from_json

        return problem_from_json(text)


def _identifier(value: object, label: str) -> None:
    try:
        validate_identifier(value, label=label)  # type: ignore[arg-type]
    except InvalidRequestError as exc:
        raise SearchProblemError(str(exc)) from exc


def _positive_int(value: object, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise SearchProblemError(f"{label} must be a positive integer")


def _non_negative_int(value: object, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise SearchProblemError(f"{label} must be a non-negative integer")


def _coerce_enum(instance: object, field_name: str, enum_type: type[Enum]) -> None:
    value = getattr(instance, field_name)
    if isinstance(value, enum_type):
        return
    try:
        object.__setattr__(instance, field_name, enum_type(value))
    except (TypeError, ValueError) as exc:
        raise SearchProblemError(f"invalid {field_name}: {value!r}") from exc


def _unique(values: tuple[object, ...], field_name: str, label: str) -> None:
    identifiers = tuple(getattr(item, field_name) for item in values)
    if len(identifiers) != len(set(identifiers)):
        raise SearchProblemError(f"{label} must be unique")


__all__ = [
    "BufferTemplate",
    "Configuration",
    "LoopTemplate",
    "OpImplementationSelection",
    "Phase",
    "PhaseDependency",
    "PhaseIterationDomain",
    "PhaseStartAlignment",
    "SearchProblem",
    "StaticDemand",
    "TemporalDemand",
    "TimingMetric",
]
