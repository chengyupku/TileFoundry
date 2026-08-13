"""Exact cache/JIT timing resolution over canonical profile keys."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from ..constants import PROFILE_SCHEMA_VERSION
from ..errors import (
    MissingProfileError,
    ProfileConflictError,
    ProfileRunError,
    ProfileStoreError,
    UnsupportedError,
)
from ..hardware.model import HardwareSpec
from ..model import TimingMetric, validate_identifier
from ..profiler.base import (
    CudaBenchmarkProvider,
    MeasurementPolicy,
    ProfileRunner,
    summarize_profile_run,
)
from ..request import ProfileMode, ProfileSelection, TimingStatistic
from ..tileop import ProfileRequirement, TileOpProfileKey, TileOpProfileQuery
from .model import ProfileMeasurement, ResolvedTiming, SnapshotState
from .store import SqliteProfileStore


@dataclass(frozen=True, slots=True)
class BenchmarkProviderCatalog:
    """Resolve exactly one benchmark provider for each query."""

    providers: tuple[CudaBenchmarkProvider, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.providers, (tuple, list)):
            raise ProfileStoreError("benchmark providers must be a sequence")
        providers = tuple(self.providers)
        ids: list[str] = []
        for provider in providers:
            provider_id = getattr(provider, "provider_id", None)
            provider_version = getattr(provider, "provider_version", None)
            if not isinstance(provider_id, str) or not isinstance(provider_version, str):
                raise ProfileStoreError("benchmark providers must expose string IDs and versions")
            try:
                validate_identifier(provider_id, label="provider_id")
                validate_identifier(provider_version, label="provider_version")
            except Exception as exc:
                raise ProfileStoreError(str(exc)) from exc
            if any(
                not callable(getattr(provider, method, None))
                for method in ("supports", "fingerprint", "materialize", "validate")
            ):
                raise ProfileStoreError("benchmark provider does not implement its protocol")
            ids.append(provider_id)
        if len(ids) != len(set(ids)):
            raise ProfileStoreError("benchmark provider IDs must be unique")
        object.__setattr__(
            self,
            "providers",
            tuple(sorted(providers, key=lambda item: (item.provider_id, item.provider_version))),
        )

    def provider_for(self, query: TileOpProfileQuery) -> CudaBenchmarkProvider:
        if type(query) is not TileOpProfileQuery:
            raise ProfileStoreError("provider query must be TileOpProfileQuery")
        matches: list[CudaBenchmarkProvider] = []
        for provider in self.providers:
            try:
                if provider.supports(query):
                    matches.append(provider)
            except Exception as exc:
                raise ProfileStoreError(
                    f"benchmark provider support check failed: {provider.provider_id}"
                ) from exc
        if len(matches) != 1:
            detail = "no provider" if not matches else "ambiguous providers"
            raise UnsupportedError(
                f"{detail} for profile query {query.implementation_id}:{query.component_id}"
            )
        return matches[0]


class ProfileResolver:
    """Resolve exact timing requirements from a snapshot or explicit JIT."""

    def __init__(
        self,
        *,
        store: SqliteProfileStore,
        providers: BenchmarkProviderCatalog,
        runner: ProfileRunner | None,
        measurement_policy: MeasurementPolicy,
    ) -> None:
        if type(store) is not SqliteProfileStore:
            raise ProfileStoreError("resolver store must be SqliteProfileStore")
        if type(providers) is not BenchmarkProviderCatalog:
            raise ProfileStoreError("resolver providers must be BenchmarkProviderCatalog")
        if runner is not None and not callable(getattr(runner, "run", None)):
            raise ProfileStoreError("resolver runner must implement ProfileRunner")
        if type(measurement_policy) is not MeasurementPolicy:
            raise ProfileStoreError("measurement_policy must be MeasurementPolicy")
        self._store = store
        self._providers = providers
        self._runner = runner
        self._policy = measurement_policy

    def resolve_many(
        self,
        requirements: tuple[ProfileRequirement, ...],
        *,
        hardware: HardwareSpec,
        selection: ProfileSelection,
    ) -> tuple[ResolvedTiming, ...]:
        if not isinstance(requirements, (tuple, list)):
            raise ProfileStoreError("profile requirements must be a sequence")
        ordered_requirements = tuple(requirements)
        if not all(type(item) is ProfileRequirement for item in ordered_requirements):
            raise ProfileStoreError("profile requirements must contain ProfileRequirement records")
        if type(hardware) is not HardwareSpec:
            raise ProfileStoreError("profile resolution hardware must be HardwareSpec")
        if type(selection) is not ProfileSelection:
            raise ProfileStoreError("profile selection must be ProfileSelection")
        if not ordered_requirements:
            return ()

        # Construct each key once.  A canonical JSON string, rather than a
        # caller tuple position, owns deduplication and first-use order.
        key_order: list[str] = []
        keys: dict[str, TileOpProfileKey] = {}
        providers: dict[str, CudaBenchmarkProvider] = {}
        requirement_keys: list[str] = []
        for requirement in ordered_requirements:
            if requirement.query.hardware != hardware.ref:
                raise ProfileStoreError(
                    "profile requirement hardware does not match resolution hardware"
                )
            provider = self._providers.provider_for(requirement.query)
            try:
                fingerprint = provider.fingerprint(requirement.query, hardware)
                key = TileOpProfileKey(PROFILE_SCHEMA_VERSION, requirement.query, fingerprint)
            except Exception as exc:
                if isinstance(exc, (ProfileStoreError, UnsupportedError)):
                    raise
                raise ProfileStoreError("benchmark provider fingerprint failed") from exc
            identity = key.canonical_json()
            requirement_keys.append(identity)
            if identity not in keys:
                key_order.append(identity)
                keys[identity] = key
                providers[identity] = provider

        measurements: dict[str, ProfileMeasurement] = {}
        missing: list[str] = []
        unavailable_metric: set[str] = set()
        metrics_by_key: dict[str, set[TimingMetric]] = {}
        for identity, requirement in zip(requirement_keys, ordered_requirements, strict=True):
            metrics_by_key.setdefault(identity, set()).add(requirement.timing_metric)
        for identity in key_order:
            key = keys[identity]
            measurement = self._store.lookup(selection.snapshot, key)
            if measurement is None:
                missing.append(str(key.key_id()))
                continue
            measurements[identity] = measurement
            if (
                TimingMetric.INITIATION_INTERVAL in metrics_by_key[identity]
                and measurement.initiation_interval_p50_ps is None
            ):
                unavailable_metric.add(identity)
                missing.append(str(key.key_id()))

        if selection.mode is ProfileMode.REQUIRE:
            if missing:
                raise MissingProfileError(missing)
        elif selection.mode is ProfileMode.JIT_ON_MISS:
            if self._runner is None:
                raise UnsupportedError("JIT-on-miss requires an explicit profile runner")
            if missing:
                if self._store.snapshot_state(selection.snapshot) is SnapshotState.FROZEN:
                    raise ProfileConflictError("JIT-on-miss may write only to a draft snapshot")
                missing_ids = set(missing)
                for identity in key_order:
                    key = keys[identity]
                    if str(key.key_id()) not in missing_ids:
                        continue
                    if identity in unavailable_metric:
                        raise ProfileStoreError(
                            "snapshot contains a measurement missing a required metric; immutable keys cannot be replaced"
                        )
                    provider = providers[identity]
                    try:
                        benchmark = provider.materialize(key, hardware)
                        run = self._runner.run(
                            benchmark,
                            hardware=hardware,
                            policy=self._policy,
                        )
                        provider.validate(benchmark, run)
                        measurement = summarize_profile_run(
                            key,
                            run,
                            policy=self._policy,
                            measured_at_utc=_now_utc(),
                        )
                        # Store insertion is the publication point.  Compilation,
                        # execution, correctness, or stability failures above
                        # therefore leave no usable row.
                        self._store.insert(selection.snapshot, measurement)
                    except Exception as exc:
                        if isinstance(
                            exc,
                            (
                                ProfileRunError,
                                ProfileStoreError,
                                UnsupportedError,
                            ),
                        ):
                            raise
                        raise ProfileRunError("JIT profile workflow failed") from exc
                    measurements[identity] = measurement
        else:  # pragma: no cover - ProfileSelection closes this enum boundary
            raise ProfileStoreError(f"unknown profile mode: {selection.mode!r}")

        resolved: list[ResolvedTiming] = []
        for requirement, identity in zip(ordered_requirements, requirement_keys, strict=True):
            measurement = measurements.get(identity)
            if measurement is None:
                # The only normal path is the all-missing exception above.
                raise MissingProfileError((str(keys[identity].key_id()),))
            resolved.append(_select_timing(requirement, measurement, selection.timing_statistic))
        return tuple(resolved)


def _select_timing(
    requirement: ProfileRequirement,
    measurement: ProfileMeasurement,
    statistic: TimingStatistic,
) -> ResolvedTiming:
    if requirement.timing_metric is TimingMetric.LATENCY:
        p50 = measurement.latency_p50_ps
        p90 = measurement.latency_p90_ps
    else:
        interval_p50 = measurement.initiation_interval_p50_ps
        interval_p90 = measurement.initiation_interval_p90_ps
        if interval_p50 is None or interval_p90 is None:
            raise MissingProfileError(
                (str(measurement.key.key_id()),), "missing requested timing metric"
            )
        p50 = interval_p50
        p90 = interval_p90
    selected = p50 if statistic is TimingStatistic.P50 else p90
    return ResolvedTiming(
        requirement,
        measurement.measurement_id,
        measurement.key.key_id(),
        measurement.environment.environment_id,
        selected,
        statistic,
        p90,
        TimingStatistic.P90,
    )


def _now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


__all__ = ["BenchmarkProviderCatalog", "ProfileResolver"]
