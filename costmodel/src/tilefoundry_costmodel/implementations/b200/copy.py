"""Minimal correctness-checked contiguous global-memory B200 copy provider."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ...errors import ProfileRunError, UnsupportedError, WorkloadError
from ...hardware.b200 import (
    B200_CALIBRATION_ID,
    B200_CUDA_CORE,
    B200_GMEM_READ,
    B200_GMEM_WRITE,
    B200_HARDWARE_ID,
    B200_SCHEMA_VERSION,
    B200_WARP_ISSUE,
)
from ...hardware.model import HardwareSpec
from ...model import (
    DType,
    OpId,
    PhaseId,
    TensorDescriptor,
    TensorLayout,
    TimingMetric,
    ValueId,
    validate_identifier,
)
from ...profiler.base import (
    CudaBenchmark,
    CudaBenchmarkCase,
    CudaBufferArgument,
    CudaBufferInit,
    CudaBufferRole,
    CudaLaunchSpec,
    CudaScalarArgument,
    CudaScalarDType,
    ProfileRun,
    compile_options_sha256,
    cuda_source_sha256,
)
from ...program import CopyOp, MemorySpace, TileOp, TileOpKind
from ...request import WarpRole
from ...tileop import (
    BenchmarkFingerprint,
    CanonicalAttribute,
    ConsumedValue,
    LoweredTileOp,
    LoweringContext,
    PhaseIterationDomain,
    PhaseTemplate,
    ProducedValue,
    ProfileRequirement,
    TemporalDemand,
    TileOpProfileKey,
    TileOpProfileQuery,
    TileOpSignature,
    ValueStorage,
    ValueStoragePolicy,
    tile_op_signature,
)
from ..base import TileOpImplementation

_SOURCE_PATH = Path(__file__).with_name("cuda") / "copy.cu"
_SOURCE_UTF8 = _SOURCE_PATH.read_text(encoding="utf-8")
_COMPILE_OPTIONS = ("--gpu-architecture=compute_100a", "--std=c++17")


def _warps(context: LoweringContext) -> tuple[int, ...]:
    for assignment in context.warps.roles:
        if assignment.role is WarpRole.CUDA_EPILOGUE:
            return assignment.warp_ids
    return ()


def _conditions() -> tuple[CanonicalAttribute, ...]:
    return (
        CanonicalAttribute("cache_policy", "default"),
        CanonicalAttribute("memory_residency", "global"),
    )


def _dtype_bytes(dtype: DType) -> int:
    return 4 if dtype is DType.FP32 else 2 if dtype in (DType.BF16, DType.FP16) else 1


def _copy_nbytes(query: TileOpProfileQuery) -> int:
    result = query.operation.results[0]
    count = 1
    for axis in result.tensor.shape.axes:
        count *= axis.extent
    return count * _dtype_bytes(result.tensor.dtype)


def _tensor_is_contiguous(tensor: TensorDescriptor) -> bool:
    strides = tensor.strides_elements
    if strides is None:
        return True
    axes_and_strides = tuple(zip(tensor.shape.axes, strides, strict=True))
    ordered = (
        tuple(reversed(axes_and_strides))
        if tensor.layout is TensorLayout.ROW_MAJOR
        else axes_and_strides
    )
    expected = 1
    for axis, stride in ordered:
        if stride != expected:
            return False
        expected *= axis.extent
    return True


def _is_supported_copy_query(query: TileOpProfileQuery) -> bool:
    if len(query.operation.operands) != 1 or len(query.operation.results) != 1:
        return False
    source = query.operation.operands[0]
    result = query.operation.results[0]
    return (
        source.memory_space is MemorySpace.GLOBAL
        and result.memory_space is MemorySpace.GLOBAL
        and source.tensor.dtype is result.tensor.dtype
        and source.tensor.element_count == result.tensor.element_count
        and _tensor_is_contiguous(source.tensor)
        and _tensor_is_contiguous(result.tensor)
    )


@dataclass(frozen=True, slots=True)
class B200CopyProvider:
    """Own the executable ``b200.copy`` benchmark artifact."""

    provider_id: str = "b200.copy"
    provider_version: str = "m3.1"

    def __post_init__(self) -> None:
        for value, label in (
            (self.provider_id, "provider_id"),
            (self.provider_version, "provider_version"),
        ):
            try:
                validate_identifier(value, label=label)
            except Exception as exc:
                raise WorkloadError(str(exc)) from exc

    def supports(self, query: TileOpProfileQuery) -> bool:
        return (
            type(query) is TileOpProfileQuery
            and query.operation.op_kind is TileOpKind.COPY
            and query.implementation_id == "b200.copy"
            and query.component_id == "b200.copy"
            and query.hardware.hardware_id == B200_HARDWARE_ID
            and query.hardware.schema_version == B200_SCHEMA_VERSION
            and query.hardware.calibration_id == B200_CALIBRATION_ID
            and query.conditions == _conditions()
            and _is_supported_copy_query(query)
        )

    def fingerprint(
        self, query: TileOpProfileQuery, hardware: HardwareSpec
    ) -> BenchmarkFingerprint:
        if not self.supports(query):
            raise UnsupportedError("B200 copy provider does not support this query")
        if query.hardware != hardware.ref:
            raise UnsupportedError("B200 copy query hardware does not match hardware document")
        return BenchmarkFingerprint(
            self.provider_id,
            self.provider_version,
            1,
            cuda_source_sha256(_SOURCE_UTF8),
            compile_options_sha256(_COMPILE_OPTIONS),
        )

    def materialize(self, key: TileOpProfileKey, hardware: HardwareSpec) -> CudaBenchmark:
        if not self.supports(key.query):
            raise UnsupportedError("B200 copy provider does not support profile key")
        if key.query.hardware != hardware.ref or key.fingerprint != self.fingerprint(
            key.query, hardware
        ):
            raise UnsupportedError("B200 copy profile key fingerprint or hardware is invalid")
        nbytes = _copy_nbytes(key.query)
        if nbytes <= 0 or nbytes > 16 * 1024 * 1024:
            raise UnsupportedError("B200 copy benchmark tile must fit the 16 MiB M3 bound")
        source = CudaBufferArgument(
            "src", nbytes, CudaBufferRole.INPUT, CudaBufferInit.SEQUENCE, 17
        )
        destination = CudaBufferArgument("dst", nbytes, CudaBufferRole.OUTPUT, CudaBufferInit.ZERO)
        nbytes_arg = CudaScalarArgument("nbytes", CudaScalarDType.I32, nbytes)
        repetitions = CudaScalarArgument("repetitions", CudaScalarDType.I32, 1)
        latency_case = CudaBenchmarkCase(
            TimingMetric.LATENCY,
            "tilefoundry_copy_latency",
            CudaLaunchSpec((1, 1, 1), (32, 1, 1), 0),
            (source, destination, nbytes_arg, repetitions),
            "repetitions",
        )
        chain_count = min(32, max(1, nbytes))
        interval_source = CudaBufferArgument(
            "src", nbytes * chain_count, CudaBufferRole.INPUT, CudaBufferInit.SEQUENCE, 17
        )
        interval_destination = CudaBufferArgument(
            "dst", nbytes * chain_count, CudaBufferRole.OUTPUT, CudaBufferInit.ZERO
        )
        chains = CudaScalarArgument("independent_chains", CudaScalarDType.I32, chain_count)
        interval_case = CudaBenchmarkCase(
            TimingMetric.INITIATION_INTERVAL,
            "tilefoundry_copy_ii",
            CudaLaunchSpec((1, 1, 1), (256, 1, 1), 0),
            (interval_source, interval_destination, nbytes_arg, repetitions, chains),
            "repetitions",
        )
        return CudaBenchmark(key, _SOURCE_UTF8, _COMPILE_OPTIONS, latency_case, interval_case)

    def validate(self, benchmark: CudaBenchmark, run: ProfileRun) -> None:
        if type(benchmark) is not CudaBenchmark or type(run) is not ProfileRun:
            raise ProfileRunError("B200 copy validation requires typed benchmark and run")
        nbytes = next(
            argument.nbytes
            for argument in benchmark.latency_case.arguments
            if isinstance(argument, CudaBufferArgument) and argument.name == "src"
        )
        expected = bytes((17 + index) % 251 for index in range(nbytes))
        interval_case = benchmark.initiation_interval_case
        if interval_case is None:
            raise ProfileRunError("B200 copy benchmark is missing its II case")
        interval_nbytes = next(
            argument.nbytes
            for argument in interval_case.arguments
            if isinstance(argument, CudaBufferArgument) and argument.name == "src"
        )
        expected_by_metric = {
            (TimingMetric.LATENCY, "dst"): expected,
            (TimingMetric.INITIATION_INTERVAL, "dst"): bytes(
                (17 + index) % 251 for index in range(interval_nbytes)
            ),
        }
        outputs = {(item.metric, item.name): item.data for item in run.outputs}
        if set(outputs) != set(expected_by_metric):
            raise ProfileRunError("B200 copy benchmark did not return all correctness outputs")
        for key, expected_output in expected_by_metric.items():
            if outputs[key] != expected_output:
                raise ProfileRunError("B200 copy correctness output does not match sequence input")


@dataclass(frozen=True, slots=True)
class B200CopyLowering:
    """Lower one contiguous global ``CopyOp`` to the M3 CUDA-core component."""

    op_kind: TileOpKind = TileOpKind.COPY
    implementation_id: str = "b200.copy"
    provider_id: str = "b200.copy"

    def supports(self, op: TileOp, *, context: LoweringContext) -> bool:
        if type(op) is not CopyOp or type(context) is not LoweringContext:
            return False
        values = {value.value_id: value for value in context.program.values}
        source = values.get(op.source)
        destination = values.get(op.destination)
        return bool(
            source is not None
            and destination is not None
            and source.value_type.memory_space is MemorySpace.GLOBAL
            and destination.value_type.memory_space is MemorySpace.GLOBAL
            and _tensor_is_contiguous(source.value_type.tensor)
            and _tensor_is_contiguous(destination.value_type.tensor)
            and _warps(context)
        )

    def lower(self, op: TileOp, *, context: LoweringContext) -> LoweredTileOp:
        if not isinstance(op, CopyOp) or not self.supports(op, context=context):
            raise UnsupportedError("B200 copy lowering requires contiguous global-memory CopyOp")
        signature: TileOpSignature = tile_op_signature(op, program=context.program)
        query = TileOpProfileQuery(
            context.hardware.ref,
            signature,
            self.implementation_id,
            self.component_id,
            context.program.tile.shape,
            context.warps.config_id,
            context.pipeline_depth,
            context.layout_variant_id,
            _conditions(),
        )
        phase = PhaseTemplate(
            PhaseId(f"{op.op_id}.copy"),
            OpId(op.op_id),
            self.implementation_id,
            "copy",
            self.component_id,
            PhaseIterationDomain.from_operation_domain(op.domain),
            ProfileRequirement(query, TimingMetric.INITIATION_INTERVAL),
            _warps(context),
            (
                TemporalDemand(B200_CUDA_CORE, 1),
                TemporalDemand(B200_GMEM_READ, 1),
                TemporalDemand(B200_GMEM_WRITE, 1),
                TemporalDemand(B200_WARP_ISSUE, 1),
            ),
        )
        policy = (
            ValueStoragePolicy.PIPELINE_RING
            if op.domain.loop_id is not None and context.pipeline_depth > 1
            else ValueStoragePolicy.STATIC
        )
        return LoweredTileOp(
            OpId(op.op_id),
            self.implementation_id,
            (phase,),
            (),
            (),
            (ProducedValue(ValueId(op.destination), "complete", phase.phase_id),),
            (ConsumedValue(ValueId(op.source), "complete", phase.phase_id, phase.phase_id),),
            (ValueStorage(ValueId(op.destination), phase.phase_id, policy),),
            (),
        )

    @property
    def component_id(self) -> str:
        return "b200.copy"


def b200_copy_implementation() -> TileOpImplementation:
    return TileOpImplementation(B200CopyLowering(), B200CopyProvider())


__all__ = ["B200CopyLowering", "B200CopyProvider", "b200_copy_implementation"]
