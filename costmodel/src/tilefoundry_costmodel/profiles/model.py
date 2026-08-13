"""Immutable timing-profile, environment, and snapshot records."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass
from datetime import datetime, timezone
from enum import Enum

from ..constants import PROFILE_SCHEMA_VERSION
from ..errors import InvalidRequestError, ProfileStoreError
from ..hardware.model import HardwareSpec
from ..model import MeasurementId, ProfileKeyId, validate_identifier
from ..request import TimingStatistic
from ..tileop import (
    BenchmarkFingerprint,
    CanonicalAttribute,
    ProfileRequirement,
    TileOpProfileKey,
    TileOpProfileQuery,
    TileOpSignature,
    profile_key_from_json,
    profile_key_to_json,
    profile_query_canonical_json,
    tile_op_signature,
)
from .types import ProfileEnvironment


class MeasurementOrigin(str, Enum):
    """Name an accepted timing origin."""

    MEASURED = "measured"


class SnapshotState(str, Enum):
    """Name profile snapshot mutability."""

    DRAFT = "draft"
    FROZEN = "frozen"


@dataclass(frozen=True, slots=True)
class ProfileMeasurement:
    """Carry validated timing aggregates for one exact profile key."""

    measurement_id: MeasurementId
    key: TileOpProfileKey
    environment: ProfileEnvironment
    origin: MeasurementOrigin
    latency_p50_ps: int
    latency_p90_ps: int
    initiation_interval_p50_ps: int | None
    initiation_interval_p90_ps: int | None
    warmup_runs: int
    sample_count: int
    latency_repetitions_per_sample: int
    initiation_interval_repetitions_per_sample: int | None
    target_sample_ns: int
    relative_iqr_ppm: int
    raw_samples_retained: bool
    raw_latency_samples_ps: tuple[int, ...]
    raw_initiation_interval_samples_ps: tuple[int, ...]
    measured_at_utc: str

    def __post_init__(self) -> None:
        _profile_identifier(self.measurement_id, "measurement_id")
        if type(self.key) is not TileOpProfileKey:
            raise ProfileStoreError("measurement key must be TileOpProfileKey")
        if type(self.environment) is not ProfileEnvironment:
            raise ProfileStoreError("measurement environment must be ProfileEnvironment")
        if self.key.query.hardware != self.environment.hardware:
            raise ProfileStoreError("measurement key and environment hardware must match")
        if len(self.environment.environment_id) == 64 and (
            self.environment.environment_id != profile_environment_id(self.environment)
        ):
            raise ProfileStoreError("environment_id does not match canonical environment data")
        _coerce_enum(self, "origin", MeasurementOrigin)

        _positive_int(self.latency_p50_ps, "latency_p50_ps")
        _positive_int(self.latency_p90_ps, "latency_p90_ps")
        if self.latency_p90_ps < self.latency_p50_ps:
            raise ProfileStoreError("latency_p90_ps must be at least latency_p50_ps")
        _optional_positive_int(self.initiation_interval_p50_ps, "initiation_interval_p50_ps")
        _optional_positive_int(self.initiation_interval_p90_ps, "initiation_interval_p90_ps")
        if (self.initiation_interval_p50_ps is None) != (self.initiation_interval_p90_ps is None):
            raise ProfileStoreError(
                "initiation-interval p50 and p90 must both be present or absent"
            )
        if (
            self.initiation_interval_p50_ps is not None
            and self.initiation_interval_p90_ps is not None
            and self.initiation_interval_p90_ps < self.initiation_interval_p50_ps
        ):
            raise ProfileStoreError(
                "initiation_interval_p90_ps must be at least initiation_interval_p50_ps"
            )

        _non_negative_int(self.warmup_runs, "warmup_runs")
        _positive_int(self.sample_count, "sample_count")
        _positive_int(self.latency_repetitions_per_sample, "latency_repetitions_per_sample")
        _optional_positive_int(
            self.initiation_interval_repetitions_per_sample,
            "initiation_interval_repetitions_per_sample",
        )
        if (self.initiation_interval_p50_ps is None) != (
            self.initiation_interval_repetitions_per_sample is None
        ):
            raise ProfileStoreError(
                "initiation-interval timing and repetitions must both be present or absent"
            )
        _positive_int(self.target_sample_ns, "target_sample_ns")
        _non_negative_int(self.relative_iqr_ppm, "relative_iqr_ppm")
        if not isinstance(self.raw_samples_retained, bool):
            raise ProfileStoreError("raw_samples_retained must be boolean")

        latency_samples = _positive_sample_tuple(
            self.raw_latency_samples_ps, "raw_latency_samples_ps"
        )
        interval_samples = _positive_sample_tuple(
            self.raw_initiation_interval_samples_ps,
            "raw_initiation_interval_samples_ps",
        )
        object.__setattr__(self, "raw_latency_samples_ps", latency_samples)
        object.__setattr__(self, "raw_initiation_interval_samples_ps", interval_samples)
        if not self.raw_samples_retained and (latency_samples or interval_samples):
            raise ProfileStoreError("raw samples must be empty when not retained")
        if self.raw_samples_retained:
            if len(latency_samples) != self.sample_count:
                raise ProfileStoreError("raw latency samples must match sample_count")
            expected_interval_count = (
                0 if self.initiation_interval_p50_ps is None else self.sample_count
            )
            if len(interval_samples) != expected_interval_count:
                raise ProfileStoreError(
                    "raw initiation-interval samples must match the measured metric"
                )
        _canonical_utc(self.measured_at_utc, "measured_at_utc")
        if len(str(self.measurement_id)) == 64 and str(self.measurement_id) != str(
            measurement_id_for(
                self.key,
                self.environment,
                latency_p50_ps=self.latency_p50_ps,
                latency_p90_ps=self.latency_p90_ps,
                initiation_interval_p50_ps=self.initiation_interval_p50_ps,
                initiation_interval_p90_ps=self.initiation_interval_p90_ps,
                warmup_runs=self.warmup_runs,
                sample_count=self.sample_count,
                latency_repetitions_per_sample=self.latency_repetitions_per_sample,
                initiation_interval_repetitions_per_sample=(
                    self.initiation_interval_repetitions_per_sample
                ),
                target_sample_ns=self.target_sample_ns,
                relative_iqr_ppm=self.relative_iqr_ppm,
                raw_samples_retained=self.raw_samples_retained,
                raw_latency_samples_ps=self.raw_latency_samples_ps,
                raw_initiation_interval_samples_ps=self.raw_initiation_interval_samples_ps,
                measured_at_utc=self.measured_at_utc,
            )
        ):
            raise ProfileStoreError("measurement_id does not match canonical measurement data")

    def aggregate_json(self) -> str:
        """Return the normalized aggregate row stored in SQLite."""

        return _canonical_json(measurement_aggregate_payload(self))

    def __eq__(self, other: object) -> bool:
        if isinstance(other, ProfileMeasurement):
            return _json_form(self) == _json_form(other)
        if isinstance(other, Mapping):
            return _json_form(self) == _json_form(other)
        return NotImplemented


@dataclass(frozen=True, slots=True)
class ResolvedTiming:
    """Bind one exact phase requirement to selected measured durations."""

    requirement: ProfileRequirement
    measurement_id: MeasurementId
    profile_key_id: ProfileKeyId
    environment_id: str
    selected_duration_ps: int
    statistic: TimingStatistic
    sensitivity_duration_ps: int
    sensitivity_statistic: TimingStatistic

    def __post_init__(self) -> None:
        if type(self.requirement) is not ProfileRequirement:
            raise ProfileStoreError("resolved timing requirement must be ProfileRequirement")
        _profile_identifier(self.measurement_id, "measurement_id")
        _profile_identifier(self.profile_key_id, "profile_key_id")
        _profile_identifier(self.environment_id, "environment_id")
        _positive_int(self.selected_duration_ps, "selected_duration_ps")
        _positive_int(self.sensitivity_duration_ps, "sensitivity_duration_ps")
        _coerce_enum(self, "statistic", TimingStatistic)
        _coerce_enum(self, "sensitivity_statistic", TimingStatistic)
        if self.sensitivity_statistic is not TimingStatistic.P90:
            raise ProfileStoreError("sensitivity timing statistic must be p90")
        if self.statistic is TimingStatistic.P90 and (
            self.selected_duration_ps != self.sensitivity_duration_ps
        ):
            raise ProfileStoreError("p90 primary and sensitivity durations must match")


@dataclass(frozen=True, slots=True)
class ProfileSnapshot:
    """Carry one strict immutable profile-snapshot-v1 document."""

    schema_version: int
    snapshot_id: str
    revision: int
    hardware: HardwareSpec
    measurements: tuple[ProfileMeasurement, ...]

    def __post_init__(self) -> None:
        if (
            isinstance(self.schema_version, bool)
            or not isinstance(self.schema_version, int)
            or self.schema_version != PROFILE_SCHEMA_VERSION
        ):
            raise ProfileStoreError(
                f"unsupported profile snapshot schema version: {self.schema_version!r}"
            )
        _profile_identifier(self.snapshot_id, "snapshot_id")
        _positive_int(self.revision, "revision")
        if type(self.hardware) is not HardwareSpec:
            raise ProfileStoreError("snapshot hardware must be HardwareSpec")
        if not isinstance(self.measurements, (tuple, list)):
            raise ProfileStoreError("measurements must be a sequence")
        measurements = tuple(self.measurements)
        if not all(type(item) is ProfileMeasurement for item in measurements):
            raise ProfileStoreError("measurements must contain ProfileMeasurement records")
        # Snapshot identity is independent of insertion order.  This is also
        # what makes export/import byte deterministic when JIT resolution has
        # discovered keys in a different order.
        measurements = tuple(sorted(measurements, key=lambda item: item.key.canonical_json()))
        object.__setattr__(self, "measurements", measurements)
        measurement_ids = tuple(item.measurement_id for item in measurements)
        key_ids = tuple(item.key.key_id() for item in measurements)
        if len(measurement_ids) != len(set(measurement_ids)):
            raise ProfileStoreError("measurement IDs must be unique")
        if len(key_ids) != len(set(key_ids)):
            raise ProfileStoreError("snapshot profile keys must be unique")
        environment_ids = {item.environment.environment_id for item in measurements}
        if len(environment_ids) > 1:
            raise ProfileStoreError("snapshot measurements must use one environment")
        for measurement in measurements:
            if measurement.environment.hardware != self.hardware.ref:
                raise ProfileStoreError("measurement environment hardware does not match snapshot")
            if measurement.key.query.hardware != self.hardware.ref:
                raise ProfileStoreError("profile key hardware does not match snapshot")

    def __eq__(self, other: object) -> bool:
        if isinstance(other, ProfileSnapshot):
            return _json_form(self) == _json_form(other)
        if isinstance(other, Mapping):
            return _json_form(self) == _json_form(other)
        return NotImplemented


# The older exported name remains an exact alias at API version (2, 0).
TileOpMeasurement = ProfileMeasurement


def measurement_aggregate_payload(measurement: ProfileMeasurement) -> dict[str, object]:
    """Return fields owned by the normalized measurement aggregate row."""

    if type(measurement) is not ProfileMeasurement:
        raise ProfileStoreError("measurement must be ProfileMeasurement")
    return {
        "measurement_id": str(measurement.measurement_id),
        "origin": measurement.origin.value,
        "latency_p50_ps": measurement.latency_p50_ps,
        "latency_p90_ps": measurement.latency_p90_ps,
        "initiation_interval_p50_ps": measurement.initiation_interval_p50_ps,
        "initiation_interval_p90_ps": measurement.initiation_interval_p90_ps,
        "warmup_runs": measurement.warmup_runs,
        "sample_count": measurement.sample_count,
        "latency_repetitions_per_sample": measurement.latency_repetitions_per_sample,
        "initiation_interval_repetitions_per_sample": (
            measurement.initiation_interval_repetitions_per_sample
        ),
        "target_sample_ns": measurement.target_sample_ns,
        "relative_iqr_ppm": measurement.relative_iqr_ppm,
        "raw_samples_retained": measurement.raw_samples_retained,
        "measured_at_utc": measurement.measured_at_utc,
    }


def profile_environment_payload(environment: ProfileEnvironment) -> dict[str, object]:
    """Return the canonical JSON-owned environment object."""

    if type(environment) is not ProfileEnvironment:
        raise ProfileStoreError("environment must be ProfileEnvironment")
    return {
        "environment_id": environment.environment_id,
        "device_uuid": environment.device_uuid,
        "hardware": {
            "hardware_id": environment.hardware.hardware_id,
            "schema_version": environment.hardware.schema_version,
            "calibration_id": environment.hardware.calibration_id,
        },
        "cuda_arch": environment.cuda_arch,
        "driver_version": environment.driver_version,
        "runtime_version": environment.runtime_version,
        "nvrtc_version": environment.nvrtc_version,
        "device_clock_khz": environment.device_clock_khz,
        "memory_clock_khz": environment.memory_clock_khz,
        "power_limit_mw": environment.power_limit_mw,
    }


def profile_environment_canonical_json(environment: ProfileEnvironment) -> str:
    """Serialize one typed environment deterministically."""

    return _canonical_json(profile_environment_payload(environment))


def profile_environment_id(environment: ProfileEnvironment) -> str:
    """Hash one environment excluding its self-identifying digest field."""

    payload = profile_environment_payload(environment)
    del payload["environment_id"]
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def measurement_id_for(
    key: TileOpProfileKey,
    environment: ProfileEnvironment,
    *,
    latency_p50_ps: int,
    latency_p90_ps: int,
    initiation_interval_p50_ps: int | None,
    initiation_interval_p90_ps: int | None,
    warmup_runs: int,
    sample_count: int,
    latency_repetitions_per_sample: int,
    initiation_interval_repetitions_per_sample: int | None,
    target_sample_ns: int,
    relative_iqr_ppm: int,
    raw_samples_retained: bool,
    raw_latency_samples_ps: tuple[int, ...],
    raw_initiation_interval_samples_ps: tuple[int, ...],
    measured_at_utc: str,
) -> MeasurementId:
    """Hash the canonical key, environment, aggregates, samples, and timestamp."""

    identity_payload = {
        "key": json.loads(key.canonical_json()),
        "environment": profile_environment_payload(environment),
        "aggregate": {
            "latency_p50_ps": latency_p50_ps,
            "latency_p90_ps": latency_p90_ps,
            "initiation_interval_p50_ps": initiation_interval_p50_ps,
            "initiation_interval_p90_ps": initiation_interval_p90_ps,
            "relative_iqr_ppm": relative_iqr_ppm,
            "warmup_runs": warmup_runs,
            "sample_count": sample_count,
            "latency_repetitions_per_sample": latency_repetitions_per_sample,
            "initiation_interval_repetitions_per_sample": (
                initiation_interval_repetitions_per_sample
            ),
            "target_sample_ns": target_sample_ns,
            "raw_samples_retained": raw_samples_retained,
        },
        "raw_latency_samples_ps": list(raw_latency_samples_ps),
        "raw_initiation_interval_samples_ps": list(raw_initiation_interval_samples_ps),
        "measured_at_utc": measured_at_utc,
    }
    return MeasurementId(
        hashlib.sha256(_canonical_json(identity_payload).encode("utf-8")).hexdigest()
    )


def _profile_identifier(value: object, label: str) -> None:
    try:
        validate_identifier(value, label=label)  # type: ignore[arg-type]
    except InvalidRequestError as exc:
        raise ProfileStoreError(str(exc)) from exc


def _coerce_enum(instance: object, field_name: str, enum_type: type[Enum]) -> None:
    value = getattr(instance, field_name)
    if isinstance(value, enum_type):
        return
    try:
        object.__setattr__(instance, field_name, enum_type(value))
    except (TypeError, ValueError) as exc:
        raise ProfileStoreError(f"invalid {field_name}: {value!r}") from exc


def _positive_int(value: object, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ProfileStoreError(f"{label} must be a positive integer")


def _optional_positive_int(value: object, label: str) -> None:
    if value is not None:
        _positive_int(value, label)


def _non_negative_int(value: object, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ProfileStoreError(f"{label} must be a non-negative integer")


def _positive_sample_tuple(value: object, label: str) -> tuple[int, ...]:
    if not isinstance(value, (tuple, list)):
        raise ProfileStoreError(f"{label} must be a sequence")
    samples = tuple(value)
    for sample in samples:
        _positive_int(sample, label)
    return samples


def _canonical_utc(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise ProfileStoreError(f"{label} must be a canonical UTC timestamp")
    try:
        parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except ValueError as exc:
        raise ProfileStoreError(f"{label} must be a canonical UTC timestamp") from exc
    if parsed.strftime("%Y-%m-%dT%H:%M:%SZ") != value:
        raise ProfileStoreError(f"{label} must be a canonical UTC timestamp")
    return value


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _json_form(value: object) -> object:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _json_form(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_form(item) for item in value]
    if isinstance(value, ProfileEnvironment):
        return profile_environment_payload(value)
    if isinstance(value, ProfileMeasurement):
        return {
            "measurement_id": str(value.measurement_id),
            "key": _json_form(value.key),
            "environment": _json_form(value.environment),
            "origin": value.origin.value,
            "latency_p50_ps": value.latency_p50_ps,
            "latency_p90_ps": value.latency_p90_ps,
            "initiation_interval_p50_ps": value.initiation_interval_p50_ps,
            "initiation_interval_p90_ps": value.initiation_interval_p90_ps,
            "warmup_runs": value.warmup_runs,
            "sample_count": value.sample_count,
            "latency_repetitions_per_sample": value.latency_repetitions_per_sample,
            "initiation_interval_repetitions_per_sample": (
                value.initiation_interval_repetitions_per_sample
            ),
            "target_sample_ns": value.target_sample_ns,
            "relative_iqr_ppm": value.relative_iqr_ppm,
            "raw_samples_retained": value.raw_samples_retained,
            "raw_latency_samples_ps": _json_form(value.raw_latency_samples_ps),
            "raw_initiation_interval_samples_ps": _json_form(
                value.raw_initiation_interval_samples_ps
            ),
            "measured_at_utc": value.measured_at_utc,
        }
    if isinstance(value, ProfileSnapshot):
        return {
            "schema_version": value.schema_version,
            "snapshot_id": value.snapshot_id,
            "revision": value.revision,
            "hardware": _json_form(value.hardware),
            "measurements": _json_form(value.measurements),
        }
    if isinstance(value, HardwareSpec):
        return {
            "schema_version": value.schema_version,
            "ref": _json_form(value.ref),
            "architecture": value.architecture,
            "temporal_resources": _json_form(value.temporal_resources),
            "static_resources": _json_form(value.static_resources),
            "supported_dtypes": _json_form(value.supported_dtypes),
            "supported_implementation_ids": _json_form(value.supported_implementation_ids),
        }
    if is_dataclass(value):
        return {field.name: _json_form(getattr(value, field.name)) for field in fields(value)}
    return value


__all__ = [
    "BenchmarkFingerprint",
    "CanonicalAttribute",
    "MeasurementOrigin",
    "ProfileEnvironment",
    "ProfileMeasurement",
    "ProfileRequirement",
    "ProfileSnapshot",
    "ResolvedTiming",
    "SnapshotState",
    "TileOpMeasurement",
    "TileOpProfileKey",
    "TileOpProfileQuery",
    "TileOpSignature",
    "measurement_aggregate_payload",
    "measurement_id_for",
    "profile_environment_canonical_json",
    "profile_environment_id",
    "profile_environment_payload",
    "profile_key_from_json",
    "profile_key_to_json",
    "profile_query_canonical_json",
    "tile_op_signature",
]
