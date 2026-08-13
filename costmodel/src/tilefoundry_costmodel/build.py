"""Pure M2 lowering composition and candidate enumeration.

``ConfigurationBuilder`` is deliberately narrower than the public
``tilefoundry_costmodel.api.build`` entry point.  It lowers typed operations to
immutable templates and derives candidates in memory; it does not resolve
measurements, open SQLite, compile CUDA, or create a solver problem.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from dataclasses import dataclass

from .constants import PROFILE_SCHEMA_VERSION
from .errors import (
    CostModelError,
    HardwareSpecError,
    InvalidRequestError,
    ProfileStoreError,
    UnsupportedError,
    WorkloadError,
)
from .hardware.b200 import (
    B200_REGISTERS_32BIT,
    B200_SMEM_BYTES,
    B200_TMEM_BYTES,
    B200_WARPS,
)
from .hardware.model import HardwareSpec
from .implementations.base import TileOpImplementation
from .implementations.registry import ImplementationCatalog
from .model import (
    BufferId,
    ConfigurationId,
    DType,
    OpId,
    PhaseId,
    ProgramId,
    ResourceId,
    TensorDescriptor,
    ValueId,
)
from .program import (
    CopyOp,
    DependencyRelation,
    ElementwiseOp,
    GemmOp,
    ReduceOp,
    TileOp,
    TileProgram,
)
from .request import SearchSpace, WarpConfig
from .tileop import (
    BenchmarkFingerprint,
    BufferTemplate,
    ConfigurationTemplate,
    LoopTemplate,
    LoweredTileOp,
    LoweringContext,
    OpImplementationSelection,
    PhaseDependency,
    PhaseTemplate,
    StaticDemand,
    TileOpProfileKey,
    TileOpProfileQuery,
    ValueStorage,
    ValueStoragePolicy,
    template_canonical_json,
    tile_op_signature,
)

_DTYPE_BYTES: dict[DType, int] = {
    DType.BF16: 2,
    DType.FP16: 2,
    DType.FP32: 4,
    DType.FP8_E4M3: 1,
    DType.FP8_E5M2: 1,
}


@dataclass(frozen=True, slots=True)
class _SingleProgramBuilder:
    """Implement one program's Cartesian enumeration behind the public API."""

    program: TileProgram
    hardware: HardwareSpec
    search_space: SearchSpace
    implementation_catalog: ImplementationCatalog

    def __post_init__(self) -> None:
        if type(self.program) is not TileProgram:
            raise WorkloadError("configuration builder program must be TileProgram")
        if type(self.hardware) is not HardwareSpec:
            raise HardwareSpecError("configuration builder hardware must be HardwareSpec")
        if type(self.search_space) is not SearchSpace:
            raise InvalidRequestError("configuration builder search_space must be SearchSpace")
        if type(self.implementation_catalog) is not ImplementationCatalog:
            raise InvalidRequestError(
                "configuration builder implementation_catalog must be ImplementationCatalog"
            )

    def enumerate(self) -> tuple[ConfigurationTemplate, ...]:
        """Return every legal candidate in canonical order."""

        operations = tuple(sorted(self.program.operations, key=lambda item: str(item.op_id)))
        # Compatibility decoding permits empty operation lists, but an
        # executable configuration must always contain phases. Treat such a
        # program as an empty legal set so it cannot reach zero-phase
        # composition below.
        if not operations:
            raise UnsupportedError(
                f"program {self.program.program_id!r} has no executable operations"
            )
        candidates: dict[str, ConfigurationTemplate] = {}
        for warp, depth, layout in itertools.product(
            self.search_space.warp_configs,
            self.search_space.pipeline_depths,
            self.search_space.layout_variant_ids,
        ):
            context = LoweringContext(self.program, self.hardware, warp, depth, layout)
            choices: list[tuple[TileOpImplementation, ...]] = []
            for op in operations:
                legal = self.implementation_catalog.choices_for(
                    op,
                    context=context,
                    allowed_implementation_ids=self.search_space.implementation_ids,
                )
                if not legal:
                    choices = []
                    break
                choices.append(legal)
            if not choices:
                continue
            for selected in itertools.product(*choices):
                try:
                    template = self._compose(warp, depth, layout, tuple(selected))
                except UnsupportedError:
                    continue
                identity = self.identity(template)
                candidates.setdefault(identity, template)
                if len(candidates) > self.search_space.max_candidates:
                    raise WorkloadError(
                        "canonical legal configuration count exceeds max_candidates"
                    )
        if not candidates:
            raise UnsupportedError(
                f"no legal configuration for program {self.program.program_id!r}"
            )
        return tuple(sorted(candidates.values(), key=self.identity))

    def _compose(
        self,
        warp: WarpConfig,
        requested_depth: int,
        layout_variant_id: str,
        selected: tuple[TileOpImplementation, ...],
    ) -> ConfigurationTemplate:
        context = LoweringContext(
            self.program,
            self.hardware,
            warp,
            requested_depth,
            layout_variant_id,
        )
        fragments = self._lower_all(context, selected)
        value_memory = {
            value.value_id: value.value_type.memory_space.value for value in self.program.values
        }
        effective_depth = (
            requested_depth
            if any(
                item.storage_policy is ValueStoragePolicy.PIPELINE_RING
                and _storage_resource(value_memory.get(item.value_id, "global")) is not None
                for fragment in fragments.values()
                for item in fragment.value_storage
            )
            else 1
        )
        if effective_depth != requested_depth:
            context = LoweringContext(
                self.program,
                self.hardware,
                warp,
                effective_depth,
                layout_variant_id,
            )
            fragments = self._lower_all(context, selected)
        phases = tuple(
            phase
            for op_id in sorted(fragments, key=str)
            for phase in sorted(fragments[op_id].phases, key=lambda item: str(item.phase_id))
        )
        self._validate_phase_ownership(phases, context)
        dependencies = self._compose_dependencies(fragments)
        alignments = tuple(
            sorted(
                (
                    alignment
                    for fragment in fragments.values()
                    for alignment in fragment.internal_start_alignments
                ),
                key=lambda item: (str(item.src_phase_id), str(item.dst_phase_id), item.offset_ps),
            )
        )
        loops = tuple(
            LoopTemplate(loop.loop_id, loop.iterations)
            for loop in sorted(self.program.loops, key=lambda item: str(item.loop_id))
        )
        buffers, buffer_demands = self._derive_buffers(fragments, effective_depth)
        static_demands = self._derive_static_demands(fragments, buffer_demands, warp.total_warps)
        selections = tuple(
            OpImplementationSelection(op.op_id, pair.lowering.implementation_id)
            for op, pair in sorted(
                zip(self.program.operations, selected), key=lambda item: str(item[0].op_id)
            )
        )
        provisional = ConfigurationTemplate(
            configuration_id=ConfigurationId("0" * 64),
            program_id=ProgramId(self.program.program_id),
            workload_kind=self.program.workload_kind,
            tile=self.program.tile,
            implementations=selections,
            warps=warp,
            pipeline_depth=effective_depth,
            layout_variant_id=layout_variant_id,
            loops=loops,
            phases=phases,
            dependencies=dependencies,
            loop_barriers=self.program.loop_barriers,
            start_alignments=alignments,
            buffers=buffers,
            static_demands=static_demands,
        )
        configuration_id = self.identity(provisional)
        return ConfigurationTemplate(
            configuration_id=ConfigurationId(configuration_id),
            program_id=provisional.program_id,
            workload_kind=provisional.workload_kind,
            tile=provisional.tile,
            implementations=provisional.implementations,
            warps=provisional.warps,
            pipeline_depth=provisional.pipeline_depth,
            layout_variant_id=provisional.layout_variant_id,
            loops=provisional.loops,
            phases=provisional.phases,
            dependencies=provisional.dependencies,
            loop_barriers=provisional.loop_barriers,
            start_alignments=provisional.start_alignments,
            buffers=provisional.buffers,
            static_demands=provisional.static_demands,
        )

    def _lower_all(
        self,
        context: LoweringContext,
        selected: tuple[TileOpImplementation, ...],
    ) -> dict[OpId, LoweredTileOp]:
        if any(
            value.value_type.tensor.dtype not in self.hardware.supported_dtypes
            for value in self.program.values
        ):
            raise UnsupportedError("program dtype is not supported by hardware")
        fragments: dict[OpId, LoweredTileOp] = {}
        for op, pair in zip(self.program.operations, selected):
            if (
                self.hardware.supported_implementation_ids
                and pair.lowering.implementation_id
                not in self.hardware.supported_implementation_ids
            ):
                raise UnsupportedError("implementation is not supported by hardware")
            try:
                supported = pair.lowering.supports(op, context=context)
            except CostModelError:
                raise
            except Exception as exc:
                raise WorkloadError("implementation support check failed") from exc
            if not supported:
                raise UnsupportedError(
                    f"implementation {pair.lowering.implementation_id!r} does not support {op.op_id!r}"
                )
            try:
                fragment = pair.lowering.lower(op, context=context)
            except CostModelError:
                raise
            except Exception as exc:
                raise WorkloadError("implementation lowering failed") from exc
            if type(fragment) is not LoweredTileOp:
                raise WorkloadError("lowering must return LoweredTileOp")
            if fragment.source_op_id != op.op_id:
                raise WorkloadError("lowering changed source operation identity")
            if fragment.implementation_id != pair.lowering.implementation_id:
                raise WorkloadError("lowering changed implementation identity")
            self._validate_fragment_values(op, fragment)
            validated_queries: set[TileOpProfileQuery] = set()
            for phase in fragment.phases:
                if phase.profile.query.implementation_id != pair.lowering.implementation_id:
                    raise WorkloadError("phase profile implementation does not match selected pair")
                if phase.profile.query not in validated_queries:
                    _provider_fingerprint(pair, phase.profile.query, self.hardware)
                    validated_queries.add(phase.profile.query)
            fragments[op.op_id] = fragment
        return fragments

    @staticmethod
    def _validate_fragment_values(op: TileOp, fragment: LoweredTileOp) -> None:
        """Keep availability and storage records inside the source op's value edges."""

        consumed: tuple[ValueId, ...]
        if isinstance(op, CopyOp):
            consumed = (op.source,)
            produced = op.destination
        elif isinstance(op, GemmOp):
            consumed = (op.lhs, op.rhs, op.accumulator)
            produced = op.result
        elif isinstance(op, ReduceOp):
            consumed = (op.source,)
            produced = op.result
        elif isinstance(op, ElementwiseOp):
            consumed = op.inputs
            produced = op.result
        else:  # pragma: no cover - TileOp is a closed union at the boundary
            raise WorkloadError("lowering received an unknown typed operation")
        consumed_ids = {item.value_id for item in fragment.consumed_values}
        if consumed_ids != set(consumed):
            raise WorkloadError("lowering consumed values do not match the source operation")
        produced_ids = {item.value_id for item in fragment.produced_values}
        if produced_ids != {produced}:
            raise WorkloadError("lowering produced values do not match the source operation")
        storage_ids = {item.value_id for item in fragment.value_storage}
        if storage_ids != {produced}:
            raise WorkloadError("lowering storage values do not match the source operation")

    def _validate_phase_ownership(
        self,
        phases: tuple[PhaseTemplate, ...],
        context: LoweringContext,
    ) -> None:
        operation_map = {op.op_id: op for op in self.program.operations}
        allowed_warps = {
            warp_id for assignment in context.warps.roles for warp_id in assignment.warp_ids
        }
        for phase in phases:
            op = operation_map.get(phase.source_op_id)
            if op is None:
                raise WorkloadError("phase references an unknown source operation")
            expected_domain = phase.domain
            source_domain = op.domain
            if (
                expected_domain.loop_id != source_domain.loop_id
                or expected_domain.first_iteration != source_domain.first_iteration
                or expected_domain.iteration_count != source_domain.iteration_count
            ):
                raise WorkloadError("phase domain must preserve source operation domain")
            if any(item not in allowed_warps for item in phase.warp_ids):
                raise WorkloadError("phase binds a warp outside the selected configuration")
            for demand in phase.temporal_demands:
                try:
                    capacity = self.hardware.temporal_capacity(demand.resource_id)
                except HardwareSpecError as exc:
                    raise WorkloadError(str(exc)) from exc
                if demand.slots > capacity:
                    raise UnsupportedError("phase temporal demand exceeds resource capacity")
            if phase.profile.query.pipeline_depth != context.pipeline_depth:
                raise WorkloadError("phase profile query depth must match configuration depth")
            if phase.component_id != phase.profile.query.component_id:
                raise WorkloadError("phase component and profile component must match")
            if phase.profile.query.hardware != self.hardware.ref:
                raise WorkloadError("phase profile hardware must match lowering hardware")
            if phase.profile.query.tile_shape != self.program.tile.shape:
                raise WorkloadError("phase profile tile shape must match the program tile")
            if phase.profile.query.operation != tile_op_signature(op, program=self.program):
                raise WorkloadError(
                    "phase profile operation must match the canonical typed operation signature"
                )

    def _compose_dependencies(
        self,
        fragments: dict[OpId, LoweredTileOp],
    ) -> tuple[PhaseDependency, ...]:
        dependencies: list[PhaseDependency] = []
        for fragment in fragments.values():
            dependencies.extend(fragment.internal_dependencies)
        for program_dependency in self.program.dependencies:
            source = fragments.get(program_dependency.src_op_id)
            destination = fragments.get(program_dependency.dst_op_id)
            if source is None or destination is None:
                raise WorkloadError("program dependency has no lowered source or destination")
            source_values = tuple(
                item
                for item in source.produced_values
                if item.value_id == program_dependency.value_id
            )
            destination_values = tuple(
                item
                for item in destination.consumed_values
                if item.value_id == program_dependency.value_id
            )
            if len(destination_values) != 1:
                raise WorkloadError(
                    f"availability for {program_dependency.value_id!r} is missing or ambiguous at destination"
                )
            required = destination_values[0].required_availability_id
            matching = tuple(item for item in source_values if item.availability_id == required)
            if len(matching) != 1:
                state = "missing" if not matching else "ambiguous"
                raise WorkloadError(
                    f"{state} availability {required!r} for value {program_dependency.value_id!r}"
                )
            dependencies.append(
                PhaseDependency(
                    matching[0].ready_phase_id,
                    destination_values[0].consume_phase_id,
                    program_dependency.relation,
                )
            )
        dedup: dict[tuple[PhaseId, PhaseId, DependencyRelation, int], PhaseDependency] = {}
        for phase_dependency in dependencies:
            key = (
                phase_dependency.src_phase_id,
                phase_dependency.dst_phase_id,
                phase_dependency.relation,
                phase_dependency.delay_ps,
            )
            if key in dedup:
                raise WorkloadError("duplicate phase dependency")
            dedup[key] = phase_dependency
        return tuple(
            sorted(
                dedup.values(),
                key=lambda item: (
                    str(item.src_phase_id),
                    str(item.dst_phase_id),
                    template_canonical_json(item.relation),
                ),
            )
        )

    def _derive_buffers(
        self,
        fragments: dict[OpId, LoweredTileOp],
        depth: int,
    ) -> tuple[tuple[BufferTemplate, ...], dict[ResourceId, int]]:
        values = {value.value_id: value for value in self.program.values}
        phases = {
            phase.phase_id: phase for fragment in fragments.values() for phase in fragment.phases
        }
        storage: dict[ValueId, ValueStorage] = {}
        releases: dict[ValueId, set[PhaseId]] = {}
        for fragment in fragments.values():
            for storage_item in fragment.value_storage:
                if (
                    storage_item.value_id in storage
                    and storage[storage_item.value_id] != storage_item
                ):
                    raise WorkloadError("value storage is ambiguous across lowerings")
                storage[storage_item.value_id] = storage_item
            for consumed_item in fragment.consumed_values:
                releases.setdefault(consumed_item.value_id, set()).add(
                    consumed_item.release_phase_id
                )
        buffers: list[BufferTemplate] = []
        demands: dict[ResourceId, int] = {}
        for value_id in sorted(storage, key=str):
            value = values.get(value_id)
            if value is None:
                raise WorkloadError("value storage references an unknown value")
            item = storage[value_id]
            storage_info = _storage_resource(value.value_type.memory_space.value)
            if storage_info is None:
                # Global memory is externally allocated and is not a static
                # per-CTA capacity.  The lifetime remains represented by the
                # phase availability records, but no local buffer is charged.
                continue
            resource, bytes_per_unit = storage_info
            bytes_per_slot = (
                _tensor_storage_elements(value.value_type.tensor)
                * _DTYPE_BYTES[value.value_type.tensor.dtype]
            )
            slot_count = depth if item.storage_policy is ValueStoragePolicy.PIPELINE_RING else 1
            release_ids: tuple[PhaseId, ...] = tuple(
                sorted(releases.get(value_id, {item.allocation_phase_id}), key=str)
            )
            if item.storage_policy is ValueStoragePolicy.PIPELINE_RING:
                producer_loop = phases[item.allocation_phase_id].domain.loop_id
                if producer_loop is None or any(
                    phases[phase_id].domain.loop_id != producer_loop for phase_id in release_ids
                ):
                    raise WorkloadError(
                        "pipeline-ring producer and release phases must use one loop"
                    )
            buffer = BufferTemplate(
                BufferId(f"buffer.{value_id}"),
                value_id,
                resource,
                bytes_per_slot,
                slot_count,
                item.allocation_phase_id,
                release_ids,
            )
            buffers.append(buffer)
            demand = ((bytes_per_slot + bytes_per_unit - 1) // bytes_per_unit) * slot_count
            demands[resource] = demands.get(resource, 0) + demand
        return tuple(buffers), demands

    def _derive_static_demands(
        self,
        fragments: dict[OpId, LoweredTileOp],
        buffer_demands: dict[ResourceId, int],
        warp_count: int,
    ) -> tuple[StaticDemand, ...]:
        totals: dict[ResourceId, int] = dict(buffer_demands)
        totals[B200_WARPS] = totals.get(B200_WARPS, 0) + warp_count
        for fragment in fragments.values():
            for demand in fragment.static_demands:
                totals[demand.resource_id] = totals.get(demand.resource_id, 0) + demand.units
        result: list[StaticDemand] = []
        for resource_id, units in sorted(totals.items(), key=lambda item: str(item[0])):
            try:
                capacity = self.hardware.static_capacity(resource_id)
            except HardwareSpecError as exc:
                raise WorkloadError(str(exc)) from exc
            if units > capacity:
                raise UnsupportedError(
                    f"static demand {resource_id!r}={units} exceeds capacity {capacity}"
                )
            result.append(StaticDemand(resource_id, units))
        return tuple(result)

    def identity(self, template: ConfigurationTemplate) -> str:
        """Return the canonical configuration identity digest."""

        payload = {
            "program": json.loads(_program_identity(self.program)),
            "hardware": {
                "hardware_id": self.hardware.ref.hardware_id,
                "schema_version": self.hardware.ref.schema_version,
                "calibration_id": self.hardware.ref.calibration_id,
            },
            "configuration": template.canonical_payload(),
        }
        digest = hashlib.sha256(template_canonical_json(payload).encode("utf-8")).hexdigest()
        return digest

    @staticmethod
    def profile_keys(
        template: ConfigurationTemplate,
        *,
        implementation_catalog: ImplementationCatalog,
        hardware: HardwareSpec,
    ) -> tuple[TileOpProfileKey, ...]:
        """Derive one exact key per distinct emitted query."""

        keys: dict[str, TileOpProfileKey] = {}
        query_keys: dict[TileOpProfileQuery, TileOpProfileKey] = {}
        for phase in template.phases:
            if phase.profile.query.hardware != hardware.ref:
                raise HardwareSpecError(
                    "template profile hardware does not match requested hardware"
                )
            key = query_keys.get(phase.profile.query)
            if key is None:
                pair = implementation_catalog.pair_for(
                    phase.profile.query.operation.op_kind,
                    phase.implementation_id,
                )
                fingerprint = _provider_fingerprint(pair, phase.profile.query, hardware)
                key = TileOpProfileKey(PROFILE_SCHEMA_VERSION, phase.profile.query, fingerprint)
                query_keys[phase.profile.query] = key
            keys[str(key.key_id())] = key
        return tuple(sorted(keys.values(), key=lambda item: str(item.key_id())))


@dataclass(frozen=True, slots=True, kw_only=True)
class ConfigurationBuilder:
    """Enumerate and compose untimed canonical configurations.

    The keyword-only constructor and ``enumerate_templates`` signature are the
    public M2 contract.  This object owns no request, store, solver, or runtime
    state; all input records are supplied explicitly for each enumeration.
    """

    implementations: ImplementationCatalog

    def __post_init__(self) -> None:
        if type(self.implementations) is not ImplementationCatalog:
            raise InvalidRequestError("implementations must be ImplementationCatalog")

    def enumerate_templates(
        self,
        programs: tuple[TileProgram, ...],
        *,
        search_space: SearchSpace,
        hardware: HardwareSpec,
    ) -> tuple[ConfigurationTemplate, ...]:
        if not isinstance(programs, (tuple, list)):
            raise WorkloadError("programs must be a sequence")
        typed_programs = tuple(programs)
        if not all(type(program) is TileProgram for program in typed_programs):
            raise WorkloadError("programs must contain TileProgram records")
        if type(search_space) is not SearchSpace:
            raise InvalidRequestError("search_space must be SearchSpace")
        if type(hardware) is not HardwareSpec:
            raise HardwareSpecError("hardware must be HardwareSpec")
        if not typed_programs:
            raise UnsupportedError("no typed programs were supplied")
        candidates: dict[str, ConfigurationTemplate] = {}
        # Program ordering is not observable: use the canonical program JSON
        # as the traversal key, while retaining the owning typed record.
        ordered_programs = tuple(
            sorted(typed_programs, key=lambda program: _program_identity(program))
        )
        for program in ordered_programs:
            one = _SingleProgramBuilder(program, hardware, search_space, self.implementations)
            try:
                program_templates = one.enumerate()
            except UnsupportedError:
                continue
            for template in program_templates:
                identity = str(template.configuration_id)
                candidates.setdefault(identity, template)
                if len(candidates) > search_space.max_candidates:
                    raise WorkloadError(
                        "canonical legal configuration count exceeds max_candidates"
                    )
        if not candidates:
            raise UnsupportedError("no legal configurations were generated")
        return tuple(sorted(candidates.values(), key=lambda item: str(item.configuration_id)))

    def profile_keys(
        self,
        template: ConfigurationTemplate,
        *,
        hardware: HardwareSpec,
    ) -> tuple[TileOpProfileKey, ...]:
        """Return the distinct exact keys emitted by one template."""

        if type(template) is not ConfigurationTemplate:
            raise WorkloadError("template must be ConfigurationTemplate")
        return _SingleProgramBuilder.profile_keys(
            template,
            implementation_catalog=self.implementations,
            hardware=hardware,
        )


def _program_identity(program: TileProgram) -> str:
    from ._serialization import program_to_json

    return program_to_json(program)


def _storage_resource(memory_space: str) -> tuple[ResourceId, int] | None:
    if memory_space == "global":
        return None
    if memory_space == "shared":
        return B200_SMEM_BYTES, 1
    if memory_space == "tensor":
        return B200_TMEM_BYTES, 1
    if memory_space == "register":
        return B200_REGISTERS_32BIT, 4
    raise UnsupportedError("global-memory values have no per-CTA static buffer in M2")


def _tensor_storage_elements(tensor: TensorDescriptor) -> int:
    if type(tensor) is not TensorDescriptor:
        raise WorkloadError("buffer tensor must be TensorDescriptor")
    if tensor.strides_elements is None:
        return tensor.element_count
    return 1 + sum(
        (axis.extent - 1) * stride
        for axis, stride in zip(tensor.shape.axes, tensor.strides_elements)
    )


def _provider_fingerprint(
    pair: TileOpImplementation,
    query: TileOpProfileQuery,
    hardware: HardwareSpec,
) -> BenchmarkFingerprint:
    try:
        supported = pair.benchmark_provider.supports(query)
    except CostModelError:
        raise
    except Exception as exc:
        raise ProfileStoreError("provider support check failed") from exc
    if not supported:
        raise UnsupportedError(
            f"provider {pair.benchmark_provider.provider_id!r} cannot serve profile query"
        )
    try:
        fingerprint = pair.benchmark_provider.fingerprint(query, hardware)
    except CostModelError:
        raise
    except Exception as exc:
        raise ProfileStoreError("provider fingerprint generation failed") from exc
    if type(fingerprint) is not BenchmarkFingerprint:
        raise ProfileStoreError("provider fingerprint must be BenchmarkFingerprint")
    if fingerprint.provider_id != pair.benchmark_provider.provider_id:
        raise ProfileStoreError("provider fingerprint ID does not match provider")
    if fingerprint.provider_version != pair.benchmark_provider.provider_version:
        raise ProfileStoreError("provider fingerprint version does not match provider")
    return fingerprint


__all__ = ["ConfigurationBuilder"]
