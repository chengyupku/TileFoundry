"""Observable M2 phase, lowering, and profile-identity contracts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace

import pytest

from tilefoundry_costmodel import ProfileStoreError, UnsupportedError, WorkloadError
from tilefoundry_costmodel.build import ConfigurationBuilder
from tilefoundry_costmodel.constants import PROFILE_SCHEMA_VERSION
from tilefoundry_costmodel.hardware import B200_SMEM_BYTES, b200_hardware_spec
from tilefoundry_costmodel.implementations import (
    ImplementationCatalog,
    SyntheticLowering,
    TileOpImplementation,
    synthetic_implementation_catalog,
)
from tilefoundry_costmodel.model import (
    AxisExtent,
    DType,
    NamedShape,
    TensorDescriptor,
    TensorLayout,
    TimingMetric,
    WorkloadKind,
)
from tilefoundry_costmodel.program import (
    AlignedRelation,
    DependencyRelationKind,
    ElementwiseKind,
    ElementwiseOp,
    EndpointRelation,
    GemmOp,
    InstanceEndpoint,
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
from tilefoundry_costmodel.request import (
    SearchSpace,
    WarpConfig,
    WarpRole,
    WarpRoleAssignment,
)
from tilefoundry_costmodel.tileop import (
    BenchmarkFingerprint,
    CanonicalAttribute,
    ConsumedValue,
    LoweredTileOp,
    LoweringContext,
    ProducedValue,
    TileOpProfileKey,
    TileOpProfileQuery,
    TileOpSignature,
    tile_op_signature,
)


def _shape(*axes: tuple[str, int]) -> NamedShape:
    return NamedShape(tuple(AxisExtent(name, extent) for name, extent in axes))


def _value(value_id: str, shape: NamedShape, memory: MemorySpace) -> TileValue:
    tensor = TensorDescriptor(shape, DType.BF16, TensorLayout.ROW_MAJOR)
    return TileValue(value_id, TileValueType(tensor, memory))


def _tile() -> TileCandidate:
    return TileCandidate("tile", _shape(("m", 16), ("n", 16), ("k", 8)))


def _warp_config(config_id: str = "warps") -> WarpConfig:
    return WarpConfig(
        config_id,
        4,
        (
            WarpRoleAssignment(WarpRole.TENSOR_CONSUMER, (0, 1)),
            WarpRoleAssignment(WarpRole.CUDA_EPILOGUE, (2,)),
            WarpRoleAssignment(WarpRole.TMA_PRODUCER, (3,)),
        ),
    )


def _one_time_gemm_program(*, program_id: str = "gemm-program", op_id: str = "gemm") -> TileProgram:
    mn = _shape(("m", 16), ("n", 16))
    lhs = _shape(("m", 16), ("k", 8))
    rhs = _shape(("k", 8), ("n", 16))
    values = (
        _value("lhs", lhs, MemorySpace.GLOBAL),
        _value("rhs", rhs, MemorySpace.GLOBAL),
        _value("acc", mn, MemorySpace.SHARED),
        _value("out", mn, MemorySpace.SHARED),
    )
    op = GemmOp(
        TileOpKind.GEMM,
        op_id,
        "lhs",
        "rhs",
        "acc",
        "out",
        "m",
        "n",
        "k",
        OpIterationDomain(None, 0, 1),
    )
    return TileProgram(
        2,
        program_id,
        WorkloadKind.GEMM,
        _tile(),
        values,
        (),
        (op,),
        (),
        (),
        ("lhs", "rhs", "acc"),
        ("out",),
    )


def _loop_gemm_program() -> TileProgram:
    mn = _shape(("m", 16), ("n", 16))
    lhs = _shape(("m", 16), ("k", 8))
    rhs = _shape(("k", 8), ("n", 16))
    values = (
        _value("lhs", lhs, MemorySpace.GLOBAL),
        _value("rhs", rhs, MemorySpace.GLOBAL),
        _value("state", mn, MemorySpace.SHARED),
    )
    op = GemmOp(
        TileOpKind.GEMM,
        "gemm",
        "lhs",
        "rhs",
        "state",
        "state",
        "m",
        "n",
        "k",
        OpIterationDomain("k_loop", 0, 4),
    )
    recurrence = TileDependency(
        "state",
        "gemm",
        "gemm",
        AlignedRelation(DependencyRelationKind.ALIGNED, 1),
    )
    return TileProgram(
        2,
        "loop-gemm",
        WorkloadKind.GEMM,
        _tile(),
        values,
        (TileLoop("k_loop", 4),),
        (op,),
        (recurrence,),
        (),
        ("lhs", "rhs"),
        ("state",),
    )


def _builder() -> ConfigurationBuilder:
    return ConfigurationBuilder(implementations=synthetic_implementation_catalog())


def _search(*implementation_ids: str, depths: tuple[int, ...] = (1,)) -> SearchSpace:
    return SearchSpace(implementation_ids, (_warp_config(),), depths)


def test_profile_identity_is_semantic_canonical_and_hashable() -> None:
    first = _one_time_gemm_program(program_id="first", op_id="first-op")
    second = _one_time_gemm_program(program_id="second", op_id="second-op")
    first_signature = tile_op_signature(first.operations[0], program=first)
    second_signature = tile_op_signature(second.operations[0], program=second)
    assert first_signature == second_signature

    conditions = (
        CanonicalAttribute("memory_residency", "global"),
        CanonicalAttribute("cache_policy", "default"),
    )
    query = TileOpProfileQuery(
        b200_hardware_spec().ref,
        first_signature,
        "synthetic.gemm",
        "synthetic.gemm",
        first.tile.shape,
        "warps",
        2,
        "default",
        conditions,
    )
    reordered = replace(query, conditions=tuple(reversed(conditions)))
    fingerprint = BenchmarkFingerprint("provider", "v1", 1, "0" * 64, "1" * 64)
    key = TileOpProfileKey(PROFILE_SCHEMA_VERSION, query, fingerprint)
    reordered_key = TileOpProfileKey(PROFILE_SCHEMA_VERSION, reordered, fingerprint)
    assert key.canonical_json() == reordered_key.canonical_json()
    assert key.key_id() == reordered_key.key_id()
    assert len(key.key_id()) == 64
    assert set(key.key_id()) <= set("0123456789abcdef")
    assert "first-op" not in key.canonical_json()
    assert '"program_id"' not in key.canonical_json()
    from tilefoundry_costmodel.tileop import profile_key_from_json, profile_key_to_json

    assert profile_key_to_json(profile_key_from_json(key.canonical_json())) == key.canonical_json()


def test_typed_profile_key_round_trips_through_profile_snapshot_codec() -> None:
    program = _one_time_gemm_program()
    signature = tile_op_signature(program.operations[0], program=program)
    hardware = b200_hardware_spec()
    query = TileOpProfileQuery(
        hardware.ref,
        signature,
        "synthetic.gemm",
        "synthetic.gemm",
        program.tile.shape,
        "warps",
        1,
        "default",
        (
            CanonicalAttribute("cache_policy", "default"),
            CanonicalAttribute("memory_residency", "global"),
        ),
    )
    key = TileOpProfileKey(
        PROFILE_SCHEMA_VERSION,
        query,
        BenchmarkFingerprint("provider", "v1", 1, "0" * 64, "1" * 64),
    )
    environment = {
        "environment_id": "env",
        "device_uuid": "uuid",
        "hardware": {
            "hardware_id": hardware.ref.hardware_id,
            "schema_version": hardware.ref.schema_version,
            "calibration_id": hardware.ref.calibration_id,
        },
        "cuda_arch": "sm_100a",
        "driver_version": "driver",
        "runtime_version": "runtime",
        "nvrtc_version": "nvrtc",
        "device_clock_khz": None,
        "memory_clock_khz": None,
        "power_limit_mw": None,
    }
    from tilefoundry_costmodel import (
        hardware_to_json,
        profile_snapshot_from_json,
        profile_snapshot_to_json,
    )

    snapshot = {
        "schema_version": 1,
        "snapshot_id": "snapshot",
        "revision": 1,
        "hardware": json.loads(hardware_to_json(hardware)),
        "measurements": [
            {
                "measurement_id": "measurement",
                "key": json.loads(key.canonical_json()),
                "environment": environment,
                "origin": "measured",
                "latency_p50_ps": 100,
                "latency_p90_ps": 120,
                "initiation_interval_p50_ps": 10,
                "initiation_interval_p90_ps": 12,
                "warmup_runs": 1,
                "sample_count": 1,
                "latency_repetitions_per_sample": 1,
                "initiation_interval_repetitions_per_sample": 1,
                "target_sample_ns": 100,
                "relative_iqr_ppm": 0,
                "raw_samples_retained": False,
                "raw_latency_samples_ps": [],
                "raw_initiation_interval_samples_ps": [],
                "measured_at_utc": "2026-08-10T00:00:00Z",
            }
        ],
    }
    encoded = profile_snapshot_to_json(snapshot)
    decoded = profile_snapshot_from_json(encoded)
    assert decoded.measurements[0].key == key
    assert profile_snapshot_to_json(decoded) == encoded


def test_synthetic_gemm_preserves_traceability_and_same_start_metrics() -> None:
    program = _one_time_gemm_program()
    builder = _builder()
    hardware = b200_hardware_spec()
    template = builder.enumerate_templates(
        (program,),
        search_space=_search("synthetic.gemm"),
        hardware=hardware,
    )[0]
    phases = {phase.phase_name: phase for phase in template.phases}
    issue = phases["issue"]
    latency = phases["latency"]
    assert issue.source_op_id == latency.source_op_id == "gemm"
    assert issue.implementation_id == latency.implementation_id == "synthetic.gemm"
    assert issue.component_id == latency.component_id == "synthetic.gemm"
    assert issue.profile.query == latency.profile.query
    assert issue.profile.timing_metric is TimingMetric.INITIATION_INTERVAL
    assert latency.profile.timing_metric is TimingMetric.LATENCY
    assert issue.warp_ids == latency.warp_ids == (0, 1)
    assert issue.temporal_demands and latency.temporal_demands
    assert issue.profile.query.implementation_id == issue.implementation_id
    assert latency.profile.query.component_id == latency.component_id
    assert template.start_alignments == (replace(template.start_alignments[0], offset_ps=0),)
    alignment = template.start_alignments[0]
    assert (alignment.src_phase_id, alignment.dst_phase_id, alignment.offset_ps) == (
        issue.phase_id,
        latency.phase_id,
        0,
    )
    assert not any(
        dependency.src_phase_id == issue.phase_id and dependency.dst_phase_id == latency.phase_id
        for dependency in template.dependencies
    )
    key = builder.profile_keys(template, hardware=hardware)[0]
    provider = builder.implementations.pair_for(
        TileOpKind.GEMM, "synthetic.gemm"
    ).benchmark_provider
    benchmark = provider.materialize(key, hardware)
    assert benchmark.key == key
    assert hashlib.sha256(benchmark.source_utf8.encode("utf-8")).hexdigest() == (
        key.fingerprint.source_sha256
    )
    options_payload = json.dumps(
        benchmark.compile_options, ensure_ascii=False, separators=(",", ":")
    )
    assert hashlib.sha256(options_payload.encode("utf-8")).hexdigest() == (
        key.fingerprint.compile_options_sha256
    )


def _elementwise_add_same_operand_program(*, program_id: str) -> TileProgram:
    shape = _shape(("m", 16), ("n", 16))
    operation = ElementwiseOp(
        TileOpKind.ELEMENTWISE,
        "add",
        ("x", "x"),
        "out",
        ElementwiseKind.ADD,
        OpIterationDomain(None, 0, 1),
    )
    return TileProgram(
        2,
        program_id,
        WorkloadKind.GEMM,
        _tile(),
        (_value("x", shape, MemorySpace.GLOBAL), _value("out", shape, MemorySpace.SHARED)),
        (),
        (operation,),
        (),
        (),
        ("x",),
        ("out",),
    )


def test_elementwise_duplicate_operand_keeps_signature_but_one_lifetime_record() -> None:
    typed = _elementwise_add_same_operand_program(program_id="duplicate-typed")
    decoded = TileProgram.from_json(typed.to_json())
    builder = _builder()
    search = _search("synthetic.elementwise")
    typed_template = builder.enumerate_templates(
        (typed,), search_space=search, hardware=b200_hardware_spec()
    )[0]
    decoded_template = builder.enumerate_templates(
        (decoded,), search_space=search, hardware=b200_hardware_spec()
    )[0]
    assert typed_template.configuration_id == decoded_template.configuration_id

    pair = builder.implementations.pair_for(TileOpKind.ELEMENTWISE, "synthetic.elementwise")
    context = LoweringContext(typed, b200_hardware_spec(), _warp_config(), 1, "default")
    lowered = pair.lowering.lower(typed.operations[0], context=context)
    signature = lowered.phases[0].profile.query.operation
    assert len(signature.operands) == 2
    assert signature.operands[0] == signature.operands[1]
    assert tuple(item.value_id for item in lowered.consumed_values) == ("x",)
    assert tuple(item.required_availability_id for item in lowered.consumed_values) == ("complete",)
    assert tuple(item.release_phase_id for item in lowered.consumed_values) == ("add.elementwise",)


def _reduction_program(*, program_id: str, axes: tuple[str, ...]) -> TileProgram:
    source_shape = _shape(("m", 16), ("n", 8))
    result_shape = _shape(("m", 16))
    operation = ReduceOp(
        TileOpKind.REDUCE,
        "reduce",
        "source",
        "result",
        axes,
        ReductionKind.SUM,
        OpIterationDomain(None, 0, 1),
    )
    return TileProgram(
        2,
        program_id,
        WorkloadKind.GEMM,
        _tile(),
        (
            _value("source", source_shape, MemorySpace.SHARED),
            _value("result", result_shape, MemorySpace.SHARED),
        ),
        (),
        (operation,),
        (),
        (),
        ("source",),
        ("result",),
    )


def test_reduce_axis_permutation_has_one_signature_query_and_profile_key() -> None:
    first = _reduction_program(program_id="reduce-first", axes=("n", "m"))
    second = _reduction_program(program_id="reduce-second", axes=("m", "n"))
    first_from_json = TileProgram.from_json(first.to_json())
    assert json.loads(first.to_json())["operations"][0]["axes"] == ["n", "m"]
    assert json.loads(second.to_json())["operations"][0]["axes"] == ["m", "n"]

    first_signature = tile_op_signature(first.operations[0], program=first)
    second_signature = tile_op_signature(second.operations[0], program=second)
    json_signature = tile_op_signature(first_from_json.operations[0], program=first_from_json)
    assert first_signature == second_signature == json_signature

    def query_for(signature: TileOpSignature) -> TileOpProfileQuery:
        return TileOpProfileQuery(
            b200_hardware_spec().ref,
            signature,
            "synthetic.reduce",
            "synthetic.reduce",
            first.tile.shape,
            "warps",
            1,
            "default",
        )

    first_query = query_for(first_signature)
    second_query = query_for(second_signature)
    assert first_query.canonical_json() == second_query.canonical_json()
    fingerprint = BenchmarkFingerprint("provider", "v1", 1, "0" * 64, "1" * 64)
    first_key = TileOpProfileKey(PROFILE_SCHEMA_VERSION, first_query, fingerprint)
    second_key = TileOpProfileKey(PROFILE_SCHEMA_VERSION, second_query, fingerprint)
    assert first_key.canonical_json() == second_key.canonical_json()
    assert first_key.key_id() == second_key.key_id()


@dataclass(frozen=True, slots=True)
class _RewrittenSignatureLowering:
    delegate: object
    wrong_signature: TileOpSignature
    op_kind: TileOpKind = TileOpKind.GEMM
    implementation_id: str = "synthetic.gemm"

    def supports(self, op: object, *, context: LoweringContext) -> bool:
        return self.delegate.supports(op, context=context)  # type: ignore[attr-defined,no-any-return]

    def lower(self, op: object, *, context: LoweringContext) -> LoweredTileOp:
        lowered = self.delegate.lower(op, context=context)  # type: ignore[attr-defined]
        phases = tuple(
            replace(
                phase,
                profile=replace(
                    phase.profile,
                    query=replace(phase.profile.query, operation=self.wrong_signature),
                ),
            )
            for phase in lowered.phases
        )
        return replace(lowered, phases=phases)


@dataclass(frozen=True, slots=True)
class _UnalignedGemmLowering:
    delegate: object
    op_kind: TileOpKind = TileOpKind.GEMM
    implementation_id: str = "synthetic.gemm"

    def supports(self, op: object, *, context: LoweringContext) -> bool:
        return self.delegate.supports(op, context=context)  # type: ignore[attr-defined,no-any-return]

    def lower(self, op: object, *, context: LoweringContext) -> LoweredTileOp:
        lowered = self.delegate.lower(op, context=context)  # type: ignore[attr-defined]
        return replace(lowered, internal_start_alignments=())


@dataclass(frozen=True, slots=True)
class _NoSharedWarpLowering:
    delegate: object
    op_kind: TileOpKind = TileOpKind.GEMM
    implementation_id: str = "synthetic.gemm"

    def supports(self, op: object, *, context: LoweringContext) -> bool:
        warp_ids = tuple(
            warp_id for assignment in context.warps.roles for warp_id in assignment.warp_ids
        )
        if len(warp_ids) != len(set(warp_ids)):
            return False
        return self.delegate.supports(op, context=context)  # type: ignore[attr-defined,no-any-return]

    def lower(self, op: object, *, context: LoweringContext) -> LoweredTileOp:
        return self.delegate.lower(op, context=context)  # type: ignore[attr-defined,no-any-return]


@dataclass(frozen=True, slots=True)
class _WrongVersionProvider:
    delegate: object

    @property
    def provider_id(self) -> str:
        return self.delegate.provider_id  # type: ignore[attr-defined,no-any-return]

    @property
    def provider_version(self) -> str:
        return "wrapper.version"

    def supports(self, query: TileOpProfileQuery) -> bool:
        return self.delegate.supports(query)  # type: ignore[attr-defined,no-any-return]

    def fingerprint(self, query: TileOpProfileQuery, hardware: object) -> BenchmarkFingerprint:
        fingerprint = self.delegate.fingerprint(query, hardware)  # type: ignore[attr-defined]
        return replace(fingerprint, provider_version="delegate.version")

    def materialize(self, key: TileOpProfileKey, hardware: object) -> object:
        return self.delegate.materialize(key, hardware)  # type: ignore[attr-defined]

    def validate(self, benchmark: object, run: object) -> None:
        self.delegate.validate(benchmark, run)  # type: ignore[attr-defined]


def test_phase_query_must_retain_full_typed_signature() -> None:
    program = _one_time_gemm_program()
    original = tile_op_signature(program.operations[0], program=program)
    changed_operand = replace(original.operands[0], memory_space=MemorySpace.REGISTER)
    wrong = replace(original, operands=(changed_operand, *original.operands[1:]))
    catalog = synthetic_implementation_catalog()
    pair = catalog.pair_for(TileOpKind.GEMM, "synthetic.gemm")
    rewritten = TileOpImplementation(
        _RewrittenSignatureLowering(pair.lowering, wrong), pair.benchmark_provider
    )
    builder = ConfigurationBuilder(implementations=ImplementationCatalog((rewritten,)))
    with pytest.raises(WorkloadError, match="canonical typed operation signature"):
        builder.enumerate_templates(
            (program,), search_space=_search("synthetic.gemm"), hardware=b200_hardware_spec()
        )


def test_async_issue_and_latency_require_zero_start_alignment() -> None:
    program = _one_time_gemm_program()
    catalog = synthetic_implementation_catalog()
    pair = catalog.pair_for(TileOpKind.GEMM, "synthetic.gemm")
    unaligned = TileOpImplementation(_UnalignedGemmLowering(pair.lowering), pair.benchmark_provider)
    builder = ConfigurationBuilder(implementations=ImplementationCatalog((unaligned,)))
    with pytest.raises(WorkloadError, match="equal starts"):
        builder.enumerate_templates(
            (program,), search_space=_search("synthetic.gemm"), hardware=b200_hardware_spec()
        )


def test_missing_required_warp_role_is_an_unsupported_candidate() -> None:
    program = _one_time_gemm_program()
    missing_tensor_warp = WarpConfig(
        "tma-only",
        4,
        (WarpRoleAssignment(WarpRole.TMA_PRODUCER, (0,)),),
    )
    search = SearchSpace(("synthetic.gemm",), (missing_tensor_warp,), (1,))
    with pytest.raises(UnsupportedError, match="no legal"):
        ConfigurationBuilder(
            implementations=synthetic_implementation_catalog()
        ).enumerate_templates((program,), search_space=search, hardware=b200_hardware_spec())


def test_implementation_can_reject_shared_warp_roles() -> None:
    program = _one_time_gemm_program()
    shared = WarpConfig(
        "shared",
        4,
        (
            WarpRoleAssignment(WarpRole.TENSOR_CONSUMER, (0, 1)),
            WarpRoleAssignment(WarpRole.CUDA_EPILOGUE, (0,)),
        ),
    )
    search = SearchSpace(("synthetic.gemm",), (shared,), (1,))
    catalog = synthetic_implementation_catalog()
    pair = catalog.pair_for(TileOpKind.GEMM, "synthetic.gemm")
    no_sharing = TileOpImplementation(_NoSharedWarpLowering(pair.lowering), pair.benchmark_provider)
    with pytest.raises(UnsupportedError, match="no legal"):
        ConfigurationBuilder(
            implementations=ImplementationCatalog((no_sharing,))
        ).enumerate_templates((program,), search_space=search, hardware=b200_hardware_spec())


def test_provider_fingerprint_version_must_match_pair() -> None:
    program = _one_time_gemm_program()
    catalog = synthetic_implementation_catalog()
    pair = catalog.pair_for(TileOpKind.GEMM, "synthetic.gemm")
    mismatched = TileOpImplementation(
        pair.lowering,
        _WrongVersionProvider(pair.benchmark_provider),
    )
    builder = ConfigurationBuilder(implementations=ImplementationCatalog((mismatched,)))
    with pytest.raises(ProfileStoreError, match="version"):
        builder.enumerate_templates(
            (program,), search_space=_search("synthetic.gemm"), hardware=b200_hardware_spec()
        )


def test_ring_depth_changes_slots_static_bytes_configuration_and_profile_key() -> None:
    program = _loop_gemm_program()
    builder = _builder()
    templates = builder.enumerate_templates(
        (program,),
        search_space=_search("synthetic.gemm", depths=(1, 2, 3)),
        hardware=b200_hardware_spec(),
    )
    by_depth = {template.pipeline_depth: template for template in templates}
    assert set(by_depth) == {1, 2, 3}
    expected_bytes = 16 * 16 * 2
    key_ids: set[str] = set()
    configuration_ids: set[str] = set()
    for depth, template in by_depth.items():
        assert len(template.buffers) == 1
        assert template.dependencies[0].relation == program.dependencies[0].relation
        assert template.buffers[0].slot_count == depth
        assert template.buffers[0].bytes_per_slot == expected_bytes
        smem = next(
            demand for demand in template.static_demands if demand.resource_id == B200_SMEM_BYTES
        )
        assert smem.units == expected_bytes * depth
        configuration_ids.add(template.configuration_id)
        keys = builder.profile_keys(template, hardware=b200_hardware_spec())
        assert len(keys) == 1
        assert keys[0].query.pipeline_depth == depth
        key_ids.add(keys[0].key_id())
    assert len(configuration_ids) == 3
    assert len(key_ids) == 3


@dataclass(frozen=True, slots=True)
class _MissingAvailabilityLowering:
    delegate: object
    op_kind: TileOpKind = TileOpKind.ELEMENTWISE
    implementation_id: str = "synthetic.elementwise"

    def supports(self, op: object, *, context: LoweringContext) -> bool:
        return self.delegate.supports(op, context=context)  # type: ignore[attr-defined,no-any-return]

    def lower(self, op: object, *, context: LoweringContext) -> LoweredTileOp:
        lowered = self.delegate.lower(op, context=context)  # type: ignore[attr-defined]
        consumed = tuple(
            ConsumedValue(
                item.value_id,
                "unavailable",
                item.consume_phase_id,
                item.release_phase_id,
            )
            for item in lowered.consumed_values
        )
        return replace(lowered, consumed_values=consumed)


def _availability_program() -> TileProgram:
    first_lhs = _shape(("m", 16), ("r", 4))
    first_rhs = _shape(("r", 4), ("k", 8))
    mid = _shape(("m", 16), ("k", 8))
    second_rhs = _shape(("k", 8), ("n", 16))
    output = _shape(("m", 16), ("n", 16))
    values = (
        _value("first_lhs", first_lhs, MemorySpace.GLOBAL),
        _value("first_rhs", first_rhs, MemorySpace.GLOBAL),
        _value("first_acc", mid, MemorySpace.SHARED),
        _value("mid", mid, MemorySpace.SHARED),
        _value("second_rhs", second_rhs, MemorySpace.GLOBAL),
        _value("second_acc", output, MemorySpace.SHARED),
        _value("gemm_out", output, MemorySpace.SHARED),
        _value("output", output, MemorySpace.SHARED),
    )
    first = GemmOp(
        TileOpKind.GEMM,
        "gemm0",
        "first_lhs",
        "first_rhs",
        "first_acc",
        "mid",
        "m",
        "k",
        "r",
        OpIterationDomain(None, 0, 1),
    )
    second = GemmOp(
        TileOpKind.GEMM,
        "gemm1",
        "mid",
        "second_rhs",
        "second_acc",
        "gemm_out",
        "m",
        "n",
        "k",
        OpIterationDomain(None, 0, 1),
    )
    epilogue = ElementwiseOp(
        TileOpKind.ELEMENTWISE,
        "epilogue",
        ("gemm_out",),
        "output",
        ElementwiseKind.RELU,
        OpIterationDomain(None, 0, 1),
    )
    endpoint = EndpointRelation(
        DependencyRelationKind.ENDPOINT,
        InstanceEndpoint.FIRST,
        InstanceEndpoint.FIRST,
    )
    return TileProgram(
        2,
        "availability",
        WorkloadKind.GEMM,
        _tile(),
        values,
        (),
        (first, second, epilogue),
        (
            TileDependency("mid", "gemm0", "gemm1", endpoint),
            TileDependency("gemm_out", "gemm1", "epilogue", endpoint),
        ),
        (),
        ("first_lhs", "first_rhs", "first_acc", "second_rhs", "second_acc"),
        ("output",),
    )


def test_ordered_and_complete_availability_are_selected_without_guessing() -> None:
    template = _builder().enumerate_templates(
        (_availability_program(),),
        search_space=_search("synthetic.gemm", "synthetic.elementwise"),
        hardware=b200_hardware_spec(),
    )[0]
    edges = {
        (dependency.src_phase_id, dependency.dst_phase_id) for dependency in template.dependencies
    }
    assert ("gemm0.issue", "gemm1.issue") in edges
    assert ("gemm1.latency", "epilogue.elementwise") in edges
    assert ("gemm0.latency", "gemm1.issue") not in edges
    assert ("gemm1.issue", "epilogue.elementwise") not in edges


def test_missing_and_ambiguous_availability_are_rejected_at_typed_boundaries() -> None:
    base = synthetic_implementation_catalog()
    gemm_pair = base.pair_for(TileOpKind.GEMM, "synthetic.gemm")
    elementwise_pair = base.pair_for(TileOpKind.ELEMENTWISE, "synthetic.elementwise")
    missing_pair = TileOpImplementation(
        _MissingAvailabilityLowering(elementwise_pair.lowering),
        elementwise_pair.benchmark_provider,
    )
    catalog = ImplementationCatalog((gemm_pair, missing_pair))
    builder = ConfigurationBuilder(implementations=catalog)
    with pytest.raises(WorkloadError, match="missing availability"):
        builder.enumerate_templates(
            (_availability_program(),),
            search_space=_search("synthetic.gemm", "synthetic.elementwise"),
            hardware=b200_hardware_spec(),
        )

    program = _one_time_gemm_program()
    context = LoweringContext(program, b200_hardware_spec(), _warp_config(), 1, "default")
    fragment = gemm_pair.lowering.lower(program.operations[0], context=context)
    duplicate = ProducedValue(
        fragment.produced_values[0].value_id,
        fragment.produced_values[0].availability_id,
        fragment.produced_values[0].ready_phase_id,
    )
    with pytest.raises(WorkloadError, match="produced availabilities"):
        replace(fragment, produced_values=(*fragment.produced_values, duplicate))


def test_catalog_rejects_duplicate_pairs_and_provider_ids() -> None:
    catalog = synthetic_implementation_catalog()
    pair = catalog.pair_for(TileOpKind.GEMM, "synthetic.gemm")
    with pytest.raises(WorkloadError, match="duplicate"):
        ImplementationCatalog((pair, pair))
    duplicate_provider = TileOpImplementation(
        SyntheticLowering(
            TileOpKind.ELEMENTWISE,
            "synthetic.elementwise",
            "synthetic-provider.gemm",
        ),
        pair.benchmark_provider,
    )
    with pytest.raises(WorkloadError, match="provider"):
        ImplementationCatalog((pair, duplicate_provider))
