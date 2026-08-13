from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import pytest
from _profile_fixtures import copy_key, copy_query, profile_environment

from tilefoundry_costmodel import MissingProfileError, ProfileRunError
from tilefoundry_costmodel.hardware.b200 import b200_hardware_spec
from tilefoundry_costmodel.hardware.model import HardwareSpec
from tilefoundry_costmodel.implementations.b200.copy import B200CopyProvider
from tilefoundry_costmodel.model import TimingMetric
from tilefoundry_costmodel.profiler.base import (
    CudaBenchmark,
    MeasurementPolicy,
    NamedBufferOutput,
    ProfileRun,
)
from tilefoundry_costmodel.profiles.resolver import BenchmarkProviderCatalog, ProfileResolver
from tilefoundry_costmodel.profiles.store import open_profile_store
from tilefoundry_costmodel.request import (
    ProfileMode,
    ProfileSelection,
    ProfileSnapshotRef,
    TimingStatistic,
)
from tilefoundry_costmodel.tileop import ProfileRequirement


@dataclass
class RecordingRunner:
    extents: list[int]

    def run(
        self,
        benchmark: CudaBenchmark,
        *,
        hardware: HardwareSpec,
        policy: MeasurementPolicy,
    ) -> ProfileRun:
        del hardware
        extent = benchmark.key.query.tile_shape.extent("k")
        self.extents.append(extent)
        return ProfileRun(
            profile_environment(),
            tuple(100 + index for index in range(policy.sample_count)),
            tuple(10 + index for index in range(policy.sample_count)),
            8,
            16,
            (),
        )


class AcceptingCopyProvider(B200CopyProvider):
    def validate(self, benchmark: CudaBenchmark, run: ProfileRun) -> None:
        del benchmark, run


def _selection(
    ref: ProfileSnapshotRef,
    mode: ProfileMode,
    statistic: TimingStatistic = TimingStatistic.P50,
) -> ProfileSelection:
    return ProfileSelection(ref, mode, statistic)


def test_require_only_reports_all_missing_without_logical_mutation(tmp_path: Path) -> None:
    hardware = b200_hardware_spec()
    with open_profile_store(tmp_path / "profiles.db", writable=True) as store:
        ref = store.create_snapshot(snapshot_id="copy", hardware=hardware, description="")
        before = store.logical_snapshot_bytes(ref)
        resolver = ProfileResolver(
            store=store,
            providers=BenchmarkProviderCatalog((B200CopyProvider(),)),
            runner=None,
            measurement_policy=MeasurementPolicy(sample_count=3),
        )
        requirements = tuple(
            ProfileRequirement(copy_query(extent=extent), TimingMetric.LATENCY) for extent in (4, 8)
        )
        with pytest.raises(MissingProfileError) as raised:
            resolver.resolve_many(
                requirements,
                hardware=hardware,
                selection=_selection(ref, ProfileMode.REQUIRE),
            )
        assert len(raised.value.key_ids) == 2
        assert store.logical_snapshot_bytes(ref) == before


def test_jit_deduplicates_first_use_then_becomes_exact_host_only_hit(tmp_path: Path) -> None:
    hardware = b200_hardware_spec()
    runner = RecordingRunner([])
    provider = AcceptingCopyProvider()
    policy = MeasurementPolicy(warmup_runs=0, sample_count=3, max_relative_iqr_ppm=1_000_000)
    database = tmp_path / "profiles.db"
    with open_profile_store(database, writable=True) as store:
        ref = store.create_snapshot(snapshot_id="copy", hardware=hardware, description="")
        resolver = ProfileResolver(
            store=store,
            providers=BenchmarkProviderCatalog((provider,)),
            runner=runner,
            measurement_policy=policy,
        )
        latency_8 = ProfileRequirement(copy_query(extent=8), TimingMetric.LATENCY)
        issue_4 = ProfileRequirement(copy_query(extent=4), TimingMetric.INITIATION_INTERVAL)
        latency_4 = ProfileRequirement(copy_query(extent=4), TimingMetric.LATENCY)
        resolved = resolver.resolve_many(
            (latency_8, issue_4, latency_4),
            hardware=hardware,
            selection=_selection(ref, ProfileMode.JIT_ON_MISS),
        )
        assert runner.extents == [8, 4]
        assert resolved[1].measurement_id == resolved[2].measurement_id
        assert resolved[1].selected_duration_ps != resolved[2].selected_duration_ps
        store.freeze(ref)

    before_modules = set(sys.modules)
    with open_profile_store(database, writable=False) as store:
        resolver = ProfileResolver(
            store=store,
            providers=BenchmarkProviderCatalog((provider,)),
            runner=None,
            measurement_policy=policy,
        )
        cache_hit = resolver.resolve_many(
            (latency_4,),
            hardware=hardware,
            selection=_selection(ref, ProfileMode.REQUIRE, TimingStatistic.P90),
        )[0]
        assert cache_hit.selected_duration_ps == cache_hit.sensitivity_duration_ps
    assert not any(name.startswith("cuda") for name in set(sys.modules) - before_modules)


def test_failed_stability_does_not_publish_measurement(tmp_path: Path) -> None:
    hardware = b200_hardware_spec()
    provider = AcceptingCopyProvider()

    @dataclass
    class UnstableRunner:
        def run(
            self,
            benchmark: CudaBenchmark,
            *,
            hardware: HardwareSpec,
            policy: MeasurementPolicy,
        ) -> ProfileRun:
            del benchmark, hardware, policy
            return ProfileRun(profile_environment(), (1, 100, 1_000), (1, 100, 1_000), 1, 1, ())

    with open_profile_store(tmp_path / "profiles.db", writable=True) as store:
        ref = store.create_snapshot(snapshot_id="copy", hardware=hardware, description="")
        resolver = ProfileResolver(
            store=store,
            providers=BenchmarkProviderCatalog((provider,)),
            runner=UnstableRunner(),
            measurement_policy=MeasurementPolicy(
                warmup_runs=0, sample_count=3, max_relative_iqr_ppm=1
            ),
        )
        requirement = ProfileRequirement(copy_query(), TimingMetric.LATENCY)
        with pytest.raises(ProfileRunError, match="stability"):
            resolver.resolve_many(
                (requirement,),
                hardware=hardware,
                selection=_selection(ref, ProfileMode.JIT_ON_MISS),
            )
        assert store.lookup(ref, copy_key()) is None


def test_failed_correctness_does_not_publish_measurement(tmp_path: Path) -> None:
    hardware = b200_hardware_spec()
    provider = B200CopyProvider()

    @dataclass
    class WrongOutputRunner:
        def run(
            self,
            benchmark: CudaBenchmark,
            *,
            hardware: HardwareSpec,
            policy: MeasurementPolicy,
        ) -> ProfileRun:
            del benchmark, hardware
            return ProfileRun(
                profile_environment(),
                tuple(100 for _ in range(policy.sample_count)),
                tuple(10 for _ in range(policy.sample_count)),
                1,
                1,
                (
                    NamedBufferOutput(TimingMetric.LATENCY, "dst", b"wrong"),
                    NamedBufferOutput(TimingMetric.INITIATION_INTERVAL, "dst", b"wrong"),
                ),
            )

    with open_profile_store(tmp_path / "profiles.db", writable=True) as store:
        ref = store.create_snapshot(snapshot_id="copy", hardware=hardware, description="")
        resolver = ProfileResolver(
            store=store,
            providers=BenchmarkProviderCatalog((provider,)),
            runner=WrongOutputRunner(),
            measurement_policy=MeasurementPolicy(warmup_runs=0, sample_count=3),
        )
        requirement = ProfileRequirement(copy_query(), TimingMetric.LATENCY)
        with pytest.raises(ProfileRunError, match="correctness"):
            resolver.resolve_many(
                (requirement,),
                hardware=hardware,
                selection=_selection(ref, ProfileMode.JIT_ON_MISS),
            )
        assert store.lookup(ref, copy_key()) is None
