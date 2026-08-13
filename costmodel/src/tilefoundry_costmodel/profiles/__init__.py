"""Canonical profile identity with lazy persistence/runner exports.

The strict JSON codec imports :mod:`profiles.model` while the package root is
being assembled.  Persistence, resolution, and CUDA runner modules therefore
remain lazy here to preserve the root import boundary and avoid a serialization
cycle.
"""

from __future__ import annotations

from .model import (
    MeasurementOrigin,
    ProfileEnvironment,
    ProfileMeasurement,
    ProfileSnapshot,
    ResolvedTiming,
    SnapshotState,
    TileOpMeasurement,
    measurement_aggregate_payload,
    measurement_id_for,
    profile_environment_canonical_json,
    profile_environment_id,
    profile_environment_payload,
)


def __getattr__(name: str) -> object:
    if name in {"MeasurementPolicy", "ProfileRun", "ProfileRunner"}:
        from ..profiler import base

        return getattr(base, name)
    if name == "LocalCudaProfileRunner":
        from ..profiler.cuda import LocalCudaProfileRunner

        return LocalCudaProfileRunner
    if name in {"STORE_SCHEMA_VERSION", "SqliteProfileStore", "open_profile_store"}:
        from . import store

        return getattr(store, name)
    if name in {"BenchmarkProviderCatalog", "ProfileResolver"}:
        from . import resolver

        return getattr(resolver, name)
    raise AttributeError(name)


__all__ = [
    "BenchmarkProviderCatalog",
    "LocalCudaProfileRunner",
    "MeasurementOrigin",
    "MeasurementPolicy",
    "ProfileEnvironment",
    "ProfileMeasurement",
    "ProfileResolver",
    "ProfileRun",
    "ProfileRunner",
    "ProfileSnapshot",
    "ResolvedTiming",
    "SnapshotState",
    "STORE_SCHEMA_VERSION",
    "SqliteProfileStore",
    "TileOpMeasurement",
    "measurement_aggregate_payload",
    "measurement_id_for",
    "open_profile_store",
    "profile_environment_canonical_json",
    "profile_environment_id",
    "profile_environment_payload",
]
