from __future__ import annotations

import os
import sys
from dataclasses import replace
from pathlib import Path

import pytest
from _profile_fixtures import copy_key, copy_program, copy_query, profile_environment

from tilefoundry_costmodel import ProfileRunError, request_to_json
from tilefoundry_costmodel.cli import main
from tilefoundry_costmodel.hardware.b200 import b200_hardware_spec
from tilefoundry_costmodel.implementations.b200.copy import B200CopyProvider
from tilefoundry_costmodel.model import (
    DType,
    GemmSpec,
    HardwareSpecRef,
    TensorLayout,
    TimingMetric,
    WorkloadKind,
)
from tilefoundry_costmodel.profiler.base import (
    MeasurementPolicy,
    ProfileRun,
    summarize_profile_run,
)
from tilefoundry_costmodel.profiler.cuda import LocalCudaProfileRunner
from tilefoundry_costmodel.profiles.resolver import BenchmarkProviderCatalog, ProfileResolver
from tilefoundry_costmodel.profiles.store import open_profile_store
from tilefoundry_costmodel.program import MemorySpace
from tilefoundry_costmodel.request import (
    CostModelRequest,
    ProfileMode,
    ProfileSelection,
    ProfileSnapshotRef,
    SearchSpace,
    TimingStatistic,
    WarpConfig,
    WarpRole,
    WarpRoleAssignment,
)
from tilefoundry_costmodel.tileop import CanonicalAttribute, ProfileRequirement


def test_runner_rejects_artifact_hash_before_importing_cuda(tmp_path: Path) -> None:
    hardware = b200_hardware_spec()
    provider = B200CopyProvider()
    benchmark = provider.materialize(copy_key(), hardware)
    bad = replace(benchmark, source_utf8=benchmark.source_utf8 + "\n// modified")
    before = set(sys.modules)
    with pytest.raises(ProfileRunError, match="source hash"):
        LocalCudaProfileRunner(tmp_path).run(
            bad,
            hardware=hardware,
            policy=MeasurementPolicy(sample_count=1),
        )
    assert not any(name.startswith("cuda") for name in set(sys.modules) - before)


def test_b200_copy_provider_checks_complete_outputs() -> None:
    hardware = b200_hardware_spec()
    provider = B200CopyProvider()
    benchmark = provider.materialize(copy_key(), hardware)
    run = ProfileRun(profile_environment(), (100,), (10,), 1, 1, ())
    with pytest.raises(ProfileRunError, match="outputs"):
        provider.validate(benchmark, run)


def test_b200_copy_provider_rejects_queries_outside_its_exact_workflow() -> None:
    provider = B200CopyProvider()
    query = copy_key().query
    assert provider.supports(query)

    wrong_hardware = replace(
        query,
        hardware=HardwareSpecRef("b200", 1, "different-calibration"),
    )
    assert not provider.supports(wrong_hardware)

    result = query.operation.results[0]
    shared_result = replace(result, memory_space=MemorySpace.SHARED)
    assert not provider.supports(
        replace(query, operation=replace(query.operation, results=(shared_result,)))
    )

    wrong_dtype = replace(result, tensor=replace(result.tensor, dtype=DType.FP16))
    assert not provider.supports(
        replace(query, operation=replace(query.operation, results=(wrong_dtype,)))
    )

    non_contiguous = replace(
        result,
        tensor=replace(result.tensor, strides_elements=(8, 4, 1)),
    )
    assert not provider.supports(
        replace(query, operation=replace(query.operation, results=(non_contiguous,)))
    )

    extra_condition = replace(
        query,
        conditions=(*query.conditions, CanonicalAttribute("benchmark_variant", "other")),
    )
    assert not provider.supports(extra_condition)


@pytest.mark.skipif(
    os.environ.get("TILEFOUNDRY_RUN_B200") != "1",
    reason="requires a local B200 and the cuda extra",
)
def test_real_b200_copy_runner_excludes_setup_and_validates_correctness(tmp_path: Path) -> None:
    hardware = b200_hardware_spec()
    provider = B200CopyProvider()
    key = copy_key(extent=256)
    benchmark = provider.materialize(key, hardware)
    policy = MeasurementPolicy(
        warmup_runs=2,
        sample_count=5,
        target_sample_ns=20_000,
        max_repetitions_per_sample=100_000,
        max_relative_iqr_ppm=1_000_000,
        retain_raw_samples=True,
    )
    run = LocalCudaProfileRunner(tmp_path).run(
        benchmark,
        hardware=hardware,
        policy=policy,
    )
    provider.validate(benchmark, run)
    measurement = summarize_profile_run(
        key,
        run,
        policy=policy,
        measured_at_utc="2026-08-10T00:00:00Z",
    )
    assert measurement.environment.cuda_arch == "sm_100a"
    assert measurement.latency_p50_ps > 0
    assert measurement.initiation_interval_p50_ps is not None
    assert len(measurement.raw_latency_samples_ps) == policy.sample_count


@pytest.mark.skipif(
    os.environ.get("TILEFOUNDRY_RUN_B200") != "1",
    reason="requires a local B200 and the cuda extra",
)
def test_real_b200_jit_publish_freeze_and_exact_cache_hit(tmp_path: Path) -> None:
    hardware = b200_hardware_spec()
    provider = B200CopyProvider()
    policy = MeasurementPolicy(
        warmup_runs=2,
        sample_count=5,
        target_sample_ns=20_000,
        max_repetitions_per_sample=100_000,
        max_relative_iqr_ppm=1_000_000,
        retain_raw_samples=True,
    )
    query = copy_query(extent=256)
    latency = ProfileRequirement(query, TimingMetric.LATENCY)
    interval = ProfileRequirement(query, TimingMetric.INITIATION_INTERVAL)
    database = tmp_path / "profiles.db"

    with open_profile_store(database, writable=True) as store:
        ref = store.create_snapshot(
            snapshot_id="real-b200-copy",
            hardware=hardware,
            description="real B200 M3 copy workflow",
        )
        resolved = ProfileResolver(
            store=store,
            providers=BenchmarkProviderCatalog((provider,)),
            runner=LocalCudaProfileRunner(tmp_path / "nvrtc-cache"),
            measurement_policy=policy,
        ).resolve_many(
            (latency, interval, latency),
            hardware=hardware,
            selection=ProfileSelection(ref, ProfileMode.JIT_ON_MISS, TimingStatistic.P50),
        )
        assert len({item.measurement_id for item in resolved}) == 1
        assert resolved[0].selected_duration_ps == resolved[2].selected_duration_ps
        assert resolved[0].selected_duration_ps > 0
        assert resolved[1].selected_duration_ps > 0
        store.freeze(ref)

    with open_profile_store(database, writable=False) as store:
        cache_hit = ProfileResolver(
            store=store,
            providers=BenchmarkProviderCatalog((provider,)),
            runner=None,
            measurement_policy=policy,
        ).resolve_many(
            (latency, interval),
            hardware=hardware,
            selection=ProfileSelection(ref, ProfileMode.REQUIRE, TimingStatistic.P90),
        )
        assert cache_hit[0].measurement_id == cache_hit[1].measurement_id
        assert all(item.selected_duration_ps == item.sensitivity_duration_ps for item in cache_hit)


@pytest.mark.skipif(
    os.environ.get("TILEFOUNDRY_RUN_B200") != "1",
    reason="requires a local B200 and the cuda extra",
)
def test_real_b200_profile_cli_is_a_frozen_exact_cache_hit(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    hardware = b200_hardware_spec()
    ref = ProfileSnapshotRef("real-b200-cli", 1)
    database = tmp_path / "profiles.db"
    with open_profile_store(database, writable=True) as store:
        created = store.create_snapshot(
            snapshot_id=ref.snapshot_id,
            hardware=hardware,
            description="real B200 profile CLI",
        )
        assert created == ref

    extent = 4096
    request = CostModelRequest(
        2,
        "real-b200-copy-cli",
        GemmSpec(
            WorkloadKind.GEMM,
            1,
            1,
            extent,
            DType.BF16,
            DType.BF16,
            DType.FP32,
            DType.BF16,
            TensorLayout.ROW_MAJOR,
            TensorLayout.ROW_MAJOR,
        ),
        (copy_program(extent=extent),),
        hardware.ref,
        SearchSpace(
            ("b200.copy",),
            (
                WarpConfig(
                    "one-cuda-warp",
                    1,
                    (WarpRoleAssignment(WarpRole.CUDA_EPILOGUE, (0,)),),
                ),
            ),
            (1,),
        ),
        ProfileSelection(ref, ProfileMode.JIT_ON_MISS, TimingStatistic.P50),
    )
    request_path = tmp_path / "request.json"
    request_path.write_text(request_to_json(request), encoding="utf-8")
    profile_args = (
        "profile",
        "--request",
        str(request_path),
        "--profiles",
        str(database),
    )
    assert main(profile_args) == 0
    assert capsys.readouterr().out.strip() == "1"

    with open_profile_store(database, writable=True) as store:
        store.freeze(ref)
        before = store.logical_snapshot_bytes(ref)
    assert main(profile_args) == 0
    assert capsys.readouterr().out.strip() == "1"
    with open_profile_store(database, writable=False) as store:
        assert store.logical_snapshot_bytes(ref) == before
