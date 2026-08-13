"""A deterministic in-memory lowering catalog for M2 tests and examples.

The provider emits identity metadata only.  It deliberately has no timing
numbers and never imports or executes CUDA; real materialization/profiling is
owned by M3.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from ..errors import ProfileRunError, UnsupportedError, WorkloadError
from ..hardware.b200 import (
    B200_CUDA_CORE,
    B200_TENSOR_CORE,
    B200_TENSOR_INFLIGHT,
    B200_TMA,
    B200_WARP_ISSUE,
)
from ..hardware.model import HardwareSpec
from ..model import OpId, PhaseId, ResourceId, TimingMetric, ValueId, validate_identifier
from ..profiler.base import (
    CudaBenchmark,
    CudaBenchmarkCase,
    CudaBufferArgument,
    CudaBufferInit,
    CudaBufferRole,
    CudaLaunchSpec,
    CudaScalarArgument,
    CudaScalarDType,
    ProfileRun,
)
from ..program import (
    AlignedRelation,
    CopyOp,
    ElementwiseOp,
    GemmOp,
    ReduceOp,
    TileOp,
    TileOpKind,
)
from ..request import WarpRole
from ..tileop import (
    BenchmarkFingerprint,
    CanonicalAttribute,
    ConsumedValue,
    LoweredTileOp,
    LoweringContext,
    PhaseIterationDomain,
    PhaseStartAlignment,
    PhaseTemplate,
    ProducedValue,
    ProfileRequirement,
    TemporalDemand,
    TileOpProfileKey,
    TileOpProfileQuery,
    TileOpSignature,
    ValueStorage,
    ValueStoragePolicy,
    profile_query_canonical_json,
    tile_op_signature,
)
from .base import TileOpImplementation
from .registry import ImplementationCatalog


def _role_warps(context: LoweringContext, role: WarpRole) -> tuple[int, ...]:
    for assignment in context.warps.roles:
        if assignment.role is role:
            return assignment.warp_ids
    # A caller may intentionally use a minimal configuration with no named
    # role.  Returning an empty set lets pure in-flight phases remain legal;
    # issue phases are rejected by PhaseTemplate's explicit warp contract.
    return ()


def _conditions(context: LoweringContext, op: TileOp) -> tuple[CanonicalAttribute, ...]:
    value_ids: tuple[ValueId, ...]
    if isinstance(op, CopyOp):
        value_ids = (op.source, op.destination)
    elif isinstance(op, GemmOp):
        value_ids = (op.lhs, op.rhs, op.accumulator, op.result)
    elif isinstance(op, ReduceOp):
        value_ids = (op.source, op.result)
    else:
        value_ids = (*op.inputs, op.result)
    values = {value.value_id: value for value in context.program.values}
    global_value = any(
        values[value_id].value_type.memory_space.value == "global" for value_id in value_ids
    )
    if global_value:
        return (
            CanonicalAttribute("cache_policy", "default"),
            CanonicalAttribute("memory_residency", "global"),
        )
    return ()


def _benchmark_source(query: TileOpProfileQuery) -> str:
    return (
        "// M2 synthetic benchmark metadata; execution intentionally deferred to M3\n"
        f"// provider=synthetic query={profile_query_canonical_json(query)}\n"
    )


def _compile_options(hardware: HardwareSpec) -> tuple[str, ...]:
    return tuple(sorted(("--synthetic-m2", f"--hardware={hardware.ref.hardware_id}")))


def _compile_options_digest(options: tuple[str, ...]) -> str:
    payload = json.dumps(options, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class SyntheticBenchmarkProvider:
    """Provider that owns identity metadata but no timing execution."""

    operation_kind: TileOpKind
    provider_id: str
    provider_version: str = "m2.synthetic.v1"

    def __post_init__(self) -> None:
        if not isinstance(self.operation_kind, TileOpKind):
            raise WorkloadError("synthetic provider operation_kind must be TileOpKind")
        for value, label in (
            (self.provider_id, "synthetic provider_id"),
            (self.provider_version, "synthetic provider_version"),
        ):
            try:
                validate_identifier(value, label=label)
            except Exception as exc:
                raise WorkloadError(str(exc)) from exc

    def supports(self, query: TileOpProfileQuery) -> bool:
        return (
            query.operation.op_kind is self.operation_kind
            and query.implementation_id == self.implementation_id
            and query.component_id == self.implementation_id
        )

    @property
    def implementation_id(self) -> str:
        return f"synthetic.{self.operation_kind.value}"

    def fingerprint(
        self, query: TileOpProfileQuery, hardware: HardwareSpec
    ) -> BenchmarkFingerprint:
        if not self.supports(query):
            raise UnsupportedError("synthetic provider does not support query")
        source = _benchmark_source(query)
        options = _compile_options(hardware)
        return BenchmarkFingerprint(
            self.provider_id,
            self.provider_version,
            1,
            hashlib.sha256(source.encode("utf-8")).hexdigest(),
            _compile_options_digest(options),
        )

    def materialize(self, key: TileOpProfileKey, hardware: HardwareSpec) -> CudaBenchmark:
        """Return an identity-only artifact; no timing or CUDA is performed."""

        if not self.supports(key.query):
            raise UnsupportedError("synthetic provider does not support profile key")
        if key.query.hardware != hardware.ref:
            raise UnsupportedError("profile key hardware does not match materialization hardware")
        if key.fingerprint != self.fingerprint(key.query, hardware):
            raise UnsupportedError("profile key fingerprint does not match synthetic provider")
        source = _benchmark_source(key.query)
        repetition = CudaScalarArgument("repetitions", CudaScalarDType.I32, 1)
        buffer = CudaBufferArgument("payload", 1, CudaBufferRole.INOUT, CudaBufferInit.ZERO)
        launch = CudaLaunchSpec((1, 1, 1), (32, 1, 1), 0)
        latency_case = CudaBenchmarkCase(
            TimingMetric.LATENCY,
            "synthetic_latency",
            launch,
            (buffer, repetition),
            "repetitions",
        )
        issue_case = CudaBenchmarkCase(
            TimingMetric.INITIATION_INTERVAL,
            "synthetic_issue",
            launch,
            (buffer, repetition),
            "repetitions",
        )
        return CudaBenchmark(key, source, _compile_options(hardware), latency_case, issue_case)

    def validate(self, benchmark: CudaBenchmark, run: ProfileRun) -> None:
        del benchmark, run
        raise ProfileRunError("synthetic provider has no executable timing workflow")


@dataclass(frozen=True, slots=True)
class SyntheticLowering:
    """Lower one typed op into explicit issue/latency metadata phases."""

    op_kind: TileOpKind
    implementation_id: str
    provider_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.op_kind, TileOpKind):
            raise WorkloadError("synthetic op_kind must be TileOpKind")
        for value, label in (
            (self.implementation_id, "synthetic implementation_id"),
            (self.provider_id, "synthetic provider_id"),
        ):
            try:
                validate_identifier(value, label=label)
            except Exception as exc:
                raise WorkloadError(str(exc)) from exc

    def supports(self, op: TileOp, *, context: LoweringContext) -> bool:
        operation_type = {
            TileOpKind.COPY: CopyOp,
            TileOpKind.GEMM: GemmOp,
            TileOpKind.REDUCE: ReduceOp,
            TileOpKind.ELEMENTWISE: ElementwiseOp,
        }[self.op_kind]
        required_role = {
            TileOpKind.COPY: WarpRole.TMA_PRODUCER,
            TileOpKind.GEMM: WarpRole.TENSOR_CONSUMER,
            TileOpKind.REDUCE: WarpRole.CUDA_EPILOGUE,
            TileOpKind.ELEMENTWISE: WarpRole.CUDA_EPILOGUE,
        }[self.op_kind]
        return type(op) is operation_type and bool(_role_warps(context, required_role))

    def lower(self, op: TileOp, *, context: LoweringContext) -> LoweredTileOp:
        if not self.supports(op, context=context):
            raise UnsupportedError("synthetic lowering does not support operation")
        signature = tile_op_signature(op, program=context.program)
        domain = PhaseIterationDomain.from_operation_domain(op.domain)
        if isinstance(op, GemmOp):
            return self._lower_gemm(op, context, signature, domain)
        if isinstance(op, CopyOp):
            return self._lower_copy(op, context, signature, domain)
        if isinstance(op, ElementwiseOp):
            return self._lower_elementwise(op, context, signature, domain)
        return self._lower_reduce(op, context, signature, domain)

    def _query(
        self,
        context: LoweringContext,
        signature: TileOpSignature,
        component_id: str,
        op: TileOp,
    ) -> TileOpProfileQuery:
        return TileOpProfileQuery(
            context.hardware.ref,
            signature,
            self.implementation_id,
            component_id,
            context.program.tile.shape,
            context.warps.config_id,
            context.pipeline_depth,
            context.layout_variant_id,
            _conditions(context, op),
        )

    def _phase(
        self,
        *,
        op: TileOp,
        context: LoweringContext,
        domain: PhaseIterationDomain,
        signature: TileOpSignature,
        phase_id: str,
        phase_name: str,
        metric: TimingMetric,
        resources: tuple[ResourceId, ...],
        warp_role: WarpRole,
        component_id: str,
    ) -> PhaseTemplate:
        query = self._query(context, signature, component_id, op)
        return PhaseTemplate(
            phase_id=PhaseId(phase_id),
            source_op_id=OpId(op.op_id),
            implementation_id=self.implementation_id,
            phase_name=phase_name,
            component_id=component_id,
            domain=domain,
            profile=ProfileRequirement(query, metric),
            warp_ids=_role_warps(context, warp_role),
            temporal_demands=tuple(TemporalDemand(resource, 1) for resource in resources),
        )

    def _lower_copy(
        self,
        op: CopyOp,
        context: LoweringContext,
        signature: TileOpSignature,
        domain: PhaseIterationDomain,
    ) -> LoweredTileOp:
        phase = self._phase(
            op=op,
            context=context,
            domain=domain,
            signature=signature,
            phase_id=f"{op.op_id}.copy",
            phase_name="copy",
            metric=TimingMetric.INITIATION_INTERVAL,
            resources=(B200_TMA, B200_WARP_ISSUE),
            warp_role=WarpRole.TMA_PRODUCER,
            component_id="synthetic.copy",
        )
        return LoweredTileOp(
            op.op_id,
            self.implementation_id,
            (phase,),
            (),
            (),
            (ProducedValue(op.destination, "complete", phase.phase_id),),
            (ConsumedValue(op.source, "complete", phase.phase_id, phase.phase_id),),
            (ValueStorage(op.destination, phase.phase_id, _storage_policy(op, context)),),
            (),
        )

    def _lower_gemm(
        self,
        op: GemmOp,
        context: LoweringContext,
        signature: TileOpSignature,
        domain: PhaseIterationDomain,
    ) -> LoweredTileOp:
        issue = self._phase(
            op=op,
            context=context,
            domain=domain,
            signature=signature,
            phase_id=f"{op.op_id}.issue",
            phase_name="issue",
            metric=TimingMetric.INITIATION_INTERVAL,
            resources=(B200_TENSOR_CORE, B200_WARP_ISSUE),
            warp_role=WarpRole.TENSOR_CONSUMER,
            component_id="synthetic.gemm",
        )
        latency = self._phase(
            op=op,
            context=context,
            domain=domain,
            signature=signature,
            phase_id=f"{op.op_id}.latency",
            phase_name="latency",
            metric=TimingMetric.LATENCY,
            resources=(B200_TENSOR_INFLIGHT,),
            warp_role=WarpRole.TENSOR_CONSUMER,
            component_id="synthetic.gemm",
        )
        return LoweredTileOp(
            op.op_id,
            self.implementation_id,
            (issue, latency),
            (),
            (PhaseStartAlignment(issue.phase_id, latency.phase_id, 0),),
            (
                ProducedValue(op.result, "ordered", issue.phase_id),
                ProducedValue(op.result, "complete", latency.phase_id),
            ),
            tuple(
                ConsumedValue(
                    value_id,
                    _required_availability(context, op, value_id),
                    issue.phase_id,
                    latency.phase_id,
                )
                for value_id in (op.lhs, op.rhs, op.accumulator)
            ),
            (ValueStorage(op.result, issue.phase_id, _storage_policy(op, context)),),
            (),
        )

    def _lower_elementwise(
        self,
        op: ElementwiseOp,
        context: LoweringContext,
        signature: TileOpSignature,
        domain: PhaseIterationDomain,
    ) -> LoweredTileOp:
        phase = self._phase(
            op=op,
            context=context,
            domain=domain,
            signature=signature,
            phase_id=f"{op.op_id}.elementwise",
            phase_name="elementwise",
            metric=TimingMetric.LATENCY,
            resources=(B200_CUDA_CORE, B200_WARP_ISSUE),
            warp_role=WarpRole.CUDA_EPILOGUE,
            component_id="synthetic.elementwise",
        )
        return LoweredTileOp(
            op.op_id,
            self.implementation_id,
            (phase,),
            (),
            (),
            (ProducedValue(op.result, "complete", phase.phase_id),),
            tuple(
                ConsumedValue(value_id, "complete", phase.phase_id, phase.phase_id)
                for value_id in _unique_value_ids(op.inputs)
            ),
            (ValueStorage(op.result, phase.phase_id, _storage_policy(op, context)),),
            (),
        )

    def _lower_reduce(
        self,
        op: ReduceOp,
        context: LoweringContext,
        signature: TileOpSignature,
        domain: PhaseIterationDomain,
    ) -> LoweredTileOp:
        phase = self._phase(
            op=op,
            context=context,
            domain=domain,
            signature=signature,
            phase_id=f"{op.op_id}.reduce",
            phase_name="reduce",
            metric=TimingMetric.LATENCY,
            resources=(B200_CUDA_CORE, B200_WARP_ISSUE),
            warp_role=WarpRole.CUDA_EPILOGUE,
            component_id="synthetic.reduce",
        )
        return LoweredTileOp(
            op.op_id,
            self.implementation_id,
            (phase,),
            (),
            (),
            (ProducedValue(op.result, "complete", phase.phase_id),),
            (ConsumedValue(op.source, "complete", phase.phase_id, phase.phase_id),),
            (ValueStorage(op.result, phase.phase_id, _storage_policy(op, context)),),
            (),
        )


def _storage_policy(op: TileOp, context: LoweringContext) -> ValueStoragePolicy:
    if op.domain.loop_id is None or context.pipeline_depth == 1:
        return ValueStoragePolicy.STATIC
    if isinstance(op, CopyOp):
        produced_value = op.destination
    elif isinstance(op, (GemmOp, ReduceOp, ElementwiseOp)):
        produced_value = op.result
    else:  # pragma: no cover - TileOp is a closed union at the boundary
        raise WorkloadError("synthetic storage received an unknown operation")
    consumers = tuple(
        dependency
        for dependency in context.program.dependencies
        if dependency.src_op_id == op.op_id and dependency.value_id == produced_value
    )
    if consumers and all(
        isinstance(dependency.relation, AlignedRelation) for dependency in consumers
    ):
        return ValueStoragePolicy.PIPELINE_RING
    return ValueStoragePolicy.STATIC


def _unique_value_ids(value_ids: tuple[ValueId, ...]) -> tuple[ValueId, ...]:
    """Keep one consumption/lifetime record per value in stable input order."""

    seen: set[ValueId] = set()
    unique: list[ValueId] = []
    for value_id in value_ids:
        if value_id not in seen:
            seen.add(value_id)
            unique.append(value_id)
    return tuple(unique)


def _required_availability(
    context: LoweringContext,
    op: GemmOp,
    value_id: ValueId,
) -> str:
    operations = {item.op_id: item for item in context.program.operations}
    for dependency in context.program.dependencies:
        if dependency.dst_op_id == op.op_id and dependency.value_id == value_id:
            return "ordered" if isinstance(operations[dependency.src_op_id], GemmOp) else "complete"
    return "complete"


def synthetic_implementation_catalog() -> ImplementationCatalog:
    """Return four paired, timing-free synthetic implementations."""

    pairs: list[TileOpImplementation] = []
    for kind in TileOpKind:
        implementation_id = f"synthetic.{kind.value}"
        lowering = SyntheticLowering(kind, implementation_id, f"synthetic-provider.{kind.value}")
        provider = SyntheticBenchmarkProvider(kind, f"synthetic-provider.{kind.value}")
        pairs.append(TileOpImplementation(lowering, provider))
    return ImplementationCatalog(tuple(pairs))


__all__ = [
    "SyntheticBenchmarkProvider",
    "SyntheticLowering",
    "synthetic_implementation_catalog",
]
