"""Runner-neutral benchmark/provider protocols.

Importing this module never imports CUDA Python, NVRTC, a driver, or a
compiler.  The concrete M3 runner owns those optional dependencies lazily.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Protocol, TypeAlias

from ..errors import ProfileRunError, ProfileStoreError
from ..hardware.model import HardwareSpec
from ..model import TimingMetric, validate_identifier
from ..profiles.model import ProfileMeasurement, measurement_id_for
from ..profiles.types import ProfileEnvironment
from ..tileop import (
    BenchmarkFingerprint,
    TileOpProfileKey,
    TileOpProfileQuery,
)


def _id(value: object, label: str) -> None:
    if not isinstance(value, str):
        raise ProfileStoreError(f"{label} must be a non-empty ASCII string")
    try:
        validate_identifier(value, label=label)
    except Exception as exc:
        raise ProfileStoreError(str(exc)) from exc


def _positive(value: object, label: str, error: type[Exception] = ProfileStoreError) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise error(f"{label} must be a positive integer")


class CudaBufferRole(str, Enum):
    INPUT = "input"
    OUTPUT = "output"
    INOUT = "inout"


class CudaBufferInit(str, Enum):
    ZERO = "zero"
    RANDOM_UNIFORM = "random_uniform"
    SEQUENCE = "sequence"


class CudaScalarDType(str, Enum):
    I32 = "i32"
    I64 = "i64"
    U32 = "u32"
    U64 = "u64"
    F32 = "f32"
    F64 = "f64"


@dataclass(frozen=True, slots=True)
class CudaBufferArgument:
    name: str
    nbytes: int
    role: CudaBufferRole
    initialization: CudaBufferInit
    seed: int = 0

    def __post_init__(self) -> None:
        _id(self.name, "buffer argument name")
        _positive(self.nbytes, "buffer argument nbytes")
        _coerce_enum(self, "role", CudaBufferRole)
        _coerce_enum(self, "initialization", CudaBufferInit)
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ProfileStoreError("buffer argument seed must be non-negative")


@dataclass(frozen=True, slots=True)
class CudaScalarArgument:
    name: str
    dtype: CudaScalarDType
    value: int | float

    def __post_init__(self) -> None:
        _id(self.name, "scalar argument name")
        _coerce_enum(self, "dtype", CudaScalarDType)
        if isinstance(self.value, bool) or not isinstance(self.value, (int, float)):
            raise ProfileStoreError("scalar argument value must be numeric")
        if isinstance(self.value, float) and not math.isfinite(self.value):
            raise ProfileStoreError("scalar argument value must be finite")


CudaArgument: TypeAlias = CudaBufferArgument | CudaScalarArgument


def cuda_source_sha256(source_utf8: str) -> str:
    """Hash the exact UTF-8 CUDA source owned by a benchmark key."""

    if not isinstance(source_utf8, str) or not source_utf8:
        raise ProfileStoreError("benchmark source must be non-empty text")
    return hashlib.sha256(source_utf8.encode("utf-8")).hexdigest()


def compile_options_sha256(options: tuple[str, ...]) -> str:
    """Hash canonical sorted NVRTC options using the profile-key encoding."""

    if not isinstance(options, (tuple, list)):
        raise ProfileStoreError("compile options must be a sequence")
    values = tuple(options)
    if (
        not all(isinstance(item, str) and item for item in values)
        or len(values) != len(set(values))
        or values != tuple(sorted(values))
    ):
        raise ProfileStoreError("compile options must be unique and sorted")
    for option in values:
        try:
            option.encode("ascii")
        except UnicodeEncodeError as exc:
            raise ProfileStoreError("compile options must be ASCII") from exc
    payload = json.dumps(values, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def measurement_policy_canonical_json(policy: "MeasurementPolicy") -> str:
    """Serialize one immutable measurement policy deterministically."""

    if type(policy) is not MeasurementPolicy:
        raise ProfileStoreError("measurement policy must be MeasurementPolicy")
    payload = {
        "warmup_runs": policy.warmup_runs,
        "sample_count": policy.sample_count,
        "target_sample_ns": policy.target_sample_ns,
        "max_repetitions_per_sample": policy.max_repetitions_per_sample,
        "max_relative_iqr_ppm": policy.max_relative_iqr_ppm,
        "retain_raw_samples": policy.retain_raw_samples,
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True, slots=True)
class CudaLaunchSpec:
    grid: tuple[int, int, int]
    block: tuple[int, int, int]
    dynamic_smem_bytes: int

    def __post_init__(self) -> None:
        for value, label in ((self.grid, "grid"), (self.block, "block")):
            if not isinstance(value, (tuple, list)) or len(value) != 3:
                raise ProfileStoreError(f"{label} must contain three dimensions")
            if any(
                isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in value
            ):
                raise ProfileStoreError(f"{label} dimensions must be positive integers")
            object.__setattr__(self, label, tuple(value))
        if (
            isinstance(self.dynamic_smem_bytes, bool)
            or not isinstance(self.dynamic_smem_bytes, int)
            or self.dynamic_smem_bytes < 0
        ):
            raise ProfileStoreError("dynamic_smem_bytes must be non-negative")


@dataclass(frozen=True, slots=True)
class CudaBenchmarkCase:
    metric: TimingMetric
    kernel_name: str
    launch: CudaLaunchSpec
    arguments: tuple[CudaArgument, ...]
    repetition_argument_name: str

    def __post_init__(self) -> None:
        _coerce_enum(self, "metric", TimingMetric)
        _id(self.kernel_name, "kernel_name")
        if type(self.launch) is not CudaLaunchSpec:
            raise ProfileStoreError("benchmark launch must be CudaLaunchSpec")
        if not isinstance(self.arguments, (tuple, list)):
            raise ProfileStoreError("benchmark arguments must be a sequence")
        arguments = tuple(self.arguments)
        if not all(type(item) in (CudaBufferArgument, CudaScalarArgument) for item in arguments):
            raise ProfileStoreError("benchmark arguments must be typed")
        names = tuple(item.name for item in arguments)
        if len(names) != len(set(names)):
            raise ProfileStoreError("benchmark argument names must be unique")
        _id(self.repetition_argument_name, "repetition argument name")
        if self.repetition_argument_name not in names:
            raise ProfileStoreError("repetition argument must be present")
        repetition = next(item for item in arguments if item.name == self.repetition_argument_name)
        if not isinstance(repetition, CudaScalarArgument) or repetition.dtype not in (
            CudaScalarDType.I32,
            CudaScalarDType.I64,
            CudaScalarDType.U32,
            CudaScalarDType.U64,
        ):
            raise ProfileStoreError("repetition argument must be an integer scalar")
        object.__setattr__(self, "arguments", arguments)


@dataclass(frozen=True, slots=True)
class CudaBenchmark:
    key: TileOpProfileKey
    source_utf8: str
    compile_options: tuple[str, ...]
    latency_case: CudaBenchmarkCase
    initiation_interval_case: CudaBenchmarkCase | None

    def __post_init__(self) -> None:
        if type(self.key) is not TileOpProfileKey:
            raise ProfileStoreError("benchmark key must be TileOpProfileKey")
        if not isinstance(self.source_utf8, str) or not self.source_utf8:
            raise ProfileStoreError("benchmark source must be non-empty text")
        if not isinstance(self.compile_options, (tuple, list)):
            raise ProfileStoreError("compile_options must be a sequence")
        options = tuple(self.compile_options)
        if not all(isinstance(item, str) and item for item in options):
            raise ProfileStoreError("compile options must be non-empty strings")
        if len(options) != len(set(options)) or options != tuple(sorted(options)):
            raise ProfileStoreError("compile options must be unique and sorted")
        if type(self.latency_case) is not CudaBenchmarkCase:
            raise ProfileStoreError("latency_case must be CudaBenchmarkCase")
        if self.latency_case.metric is not TimingMetric.LATENCY:
            raise ProfileStoreError("latency_case must use latency metric")
        if self.latency_case.launch.grid != (1, 1, 1):
            raise ProfileStoreError("initial benchmark grid must equal (1, 1, 1)")
        if self.initiation_interval_case is not None:
            if type(self.initiation_interval_case) is not CudaBenchmarkCase:
                raise ProfileStoreError("initiation_interval_case must be CudaBenchmarkCase")
            if self.initiation_interval_case.metric is not TimingMetric.INITIATION_INTERVAL:
                raise ProfileStoreError("initiation case must use initiation_interval metric")
            if self.initiation_interval_case.launch.grid != (1, 1, 1):
                raise ProfileStoreError("initial benchmark grid must equal (1, 1, 1)")
        object.__setattr__(self, "compile_options", options)


@dataclass(frozen=True, slots=True)
class NamedBufferOutput:
    metric: TimingMetric
    name: str
    data: bytes

    def __post_init__(self) -> None:
        _coerce_enum(self, "metric", TimingMetric)
        _id(self.name, "output name")
        if not isinstance(self.data, bytes):
            raise ProfileRunError("benchmark output data must be bytes")


@dataclass(frozen=True, slots=True)
class ProfileRun:
    environment: ProfileEnvironment
    latency_samples_ps: tuple[int, ...]
    initiation_interval_samples_ps: tuple[int, ...]
    latency_repetitions_per_sample: int
    initiation_interval_repetitions_per_sample: int | None
    outputs: tuple[NamedBufferOutput, ...]

    def __post_init__(self) -> None:
        if type(self.environment) is not ProfileEnvironment:
            raise ProfileRunError("profile run environment must be ProfileEnvironment")
        for values, label in (
            (self.latency_samples_ps, "latency_samples_ps"),
            (self.initiation_interval_samples_ps, "initiation_interval_samples_ps"),
        ):
            if not isinstance(values, (tuple, list)):
                raise ProfileRunError(f"{label} must be a sequence")
            if any(
                isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in values
            ):
                raise ProfileRunError(f"{label} must contain positive integers")
            object.__setattr__(self, label, tuple(values))
        _positive(
            self.latency_repetitions_per_sample, "latency_repetitions_per_sample", ProfileRunError
        )
        if self.initiation_interval_repetitions_per_sample is not None:
            _positive(
                self.initiation_interval_repetitions_per_sample,
                "initiation_interval_repetitions_per_sample",
                ProfileRunError,
            )
        if not isinstance(self.outputs, (tuple, list)):
            raise ProfileRunError("profile outputs must be a sequence")
        outputs = tuple(self.outputs)
        if not all(type(item) is NamedBufferOutput for item in outputs):
            raise ProfileRunError("profile outputs must contain NamedBufferOutput records")
        output_keys = tuple((item.metric, item.name) for item in outputs)
        if len(output_keys) != len(set(output_keys)):
            raise ProfileRunError("profile output names must be unique per metric")
        object.__setattr__(self, "outputs", outputs)
        if not self.latency_samples_ps:
            raise ProfileRunError("profile run must contain latency samples")
        if bool(self.initiation_interval_samples_ps) != (
            self.initiation_interval_repetitions_per_sample is not None
        ):
            raise ProfileRunError(
                "initiation-interval samples and repetitions must both be present or absent"
            )


@dataclass(frozen=True, slots=True)
class MeasurementPolicy:
    """Runner-neutral controls for one stable CUDA measurement."""

    warmup_runs: int = 20
    sample_count: int = 100
    target_sample_ns: int = 100_000
    max_repetitions_per_sample: int = 1_000_000
    max_relative_iqr_ppm: int = 50_000
    retain_raw_samples: bool = False

    def __post_init__(self) -> None:
        for value, label in (
            (self.warmup_runs, "warmup_runs"),
            (self.sample_count, "sample_count"),
            (self.target_sample_ns, "target_sample_ns"),
            (self.max_repetitions_per_sample, "max_repetitions_per_sample"),
            (self.max_relative_iqr_ppm, "max_relative_iqr_ppm"),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ProfileStoreError(f"{label} must be non-negative")
        if (
            self.sample_count == 0
            or self.target_sample_ns == 0
            or self.max_repetitions_per_sample == 0
        ):
            raise ProfileStoreError(
                "sample_count, target_sample_ns and repetition cap must be positive"
            )
        if not isinstance(self.retain_raw_samples, bool):
            raise ProfileStoreError("retain_raw_samples must be boolean")


class ProfileRunner(Protocol):
    def run(
        self,
        benchmark: CudaBenchmark,
        *,
        hardware: HardwareSpec,
        policy: MeasurementPolicy,
    ) -> ProfileRun: ...


class CudaBenchmarkProvider(Protocol):
    """Materialize and validate one exact implementation benchmark."""

    @property
    def provider_id(self) -> str: ...

    @property
    def provider_version(self) -> str: ...

    def supports(self, query: TileOpProfileQuery) -> bool: ...

    def fingerprint(
        self, query: TileOpProfileQuery, hardware: HardwareSpec
    ) -> BenchmarkFingerprint: ...

    def materialize(self, key: TileOpProfileKey, hardware: HardwareSpec) -> CudaBenchmark: ...

    def validate(self, benchmark: CudaBenchmark, run: ProfileRun) -> None: ...


def summarize_profile_run(
    key: TileOpProfileKey,
    run: ProfileRun,
    *,
    policy: MeasurementPolicy,
    measured_at_utc: str,
) -> "ProfileMeasurement":
    """Validate samples and create one immutable measured aggregate."""

    if type(key) is not TileOpProfileKey:
        raise ProfileRunError("profile summary key must be TileOpProfileKey")
    if type(run) is not ProfileRun:
        raise ProfileRunError("profile summary run must be ProfileRun")
    if type(policy) is not MeasurementPolicy:
        raise ProfileRunError("profile summary policy must be MeasurementPolicy")
    _canonical_utc(measured_at_utc)
    if len(run.latency_samples_ps) != policy.sample_count:
        raise ProfileRunError("latency sample count does not match measurement policy")
    if (
        run.initiation_interval_samples_ps
        and len(run.initiation_interval_samples_ps) != policy.sample_count
    ):
        raise ProfileRunError("initiation-interval sample count does not match policy")

    latency = _summary_stats(run.latency_samples_ps)
    interval = (
        _summary_stats(run.initiation_interval_samples_ps)
        if run.initiation_interval_samples_ps
        else None
    )
    relative_iqr_ppm = latency[2] if interval is None else max(latency[2], interval[2])
    if relative_iqr_ppm > policy.max_relative_iqr_ppm:
        raise ProfileRunError("profile samples exceed the relative-IQR stability threshold")

    raw_latency = run.latency_samples_ps if policy.retain_raw_samples else ()
    raw_interval = run.initiation_interval_samples_ps if policy.retain_raw_samples else ()
    from ..profiles.model import MeasurementOrigin

    measurement_id = measurement_id_for(
        key,
        run.environment,
        latency_p50_ps=latency[0],
        latency_p90_ps=latency[1],
        initiation_interval_p50_ps=None if interval is None else interval[0],
        initiation_interval_p90_ps=None if interval is None else interval[1],
        warmup_runs=policy.warmup_runs,
        sample_count=policy.sample_count,
        latency_repetitions_per_sample=run.latency_repetitions_per_sample,
        initiation_interval_repetitions_per_sample=(run.initiation_interval_repetitions_per_sample),
        target_sample_ns=policy.target_sample_ns,
        relative_iqr_ppm=relative_iqr_ppm,
        raw_samples_retained=policy.retain_raw_samples,
        raw_latency_samples_ps=raw_latency,
        raw_initiation_interval_samples_ps=raw_interval,
        measured_at_utc=measured_at_utc,
    )
    try:
        return ProfileMeasurement(
            measurement_id,
            key,
            run.environment,
            MeasurementOrigin.MEASURED,
            latency[0],
            latency[1],
            None if interval is None else interval[0],
            None if interval is None else interval[1],
            policy.warmup_runs,
            policy.sample_count,
            run.latency_repetitions_per_sample,
            run.initiation_interval_repetitions_per_sample,
            policy.target_sample_ns,
            relative_iqr_ppm,
            policy.retain_raw_samples,
            raw_latency,
            raw_interval,
            measured_at_utc,
        )
    except Exception as exc:
        if isinstance(exc, ProfileRunError):
            raise
        raise ProfileRunError("profile aggregate failed typed validation") from exc


def _summary_stats(samples: tuple[int, ...]) -> tuple[int, int, int]:
    if not samples or any(not isinstance(item, int) or item <= 0 for item in samples):
        raise ProfileRunError("profile samples must be positive integers")
    ordered = tuple(sorted(samples))
    p50 = _nearest_rank(ordered, 50)
    p90 = _nearest_rank(ordered, 90)
    q25 = _nearest_rank(ordered, 25)
    q75 = _nearest_rank(ordered, 75)
    relative_iqr_ppm = ((q75 - q25) * 1_000_000) // p50
    return p50, p90, relative_iqr_ppm


def _nearest_rank(ordered: tuple[int, ...], percentile: int) -> int:
    index = max(1, math.ceil(len(ordered) * percentile / 100)) - 1
    return ordered[index]


def _canonical_utc(value: object) -> None:
    if not isinstance(value, str):
        raise ProfileRunError("measured_at_utc must be a canonical UTC timestamp")
    try:
        parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except ValueError as exc:
        raise ProfileRunError("measured_at_utc must be a canonical UTC timestamp") from exc
    if parsed.strftime("%Y-%m-%dT%H:%M:%SZ") != value:
        raise ProfileRunError("measured_at_utc must be a canonical UTC timestamp")


def _coerce_enum(instance: object, field_name: str, enum_type: type[Enum]) -> None:
    value = getattr(instance, field_name)
    if isinstance(value, enum_type):
        return
    try:
        object.__setattr__(instance, field_name, enum_type(value))
    except (TypeError, ValueError) as exc:
        raise ProfileStoreError(f"invalid {field_name}: {value!r}") from exc


__all__ = [
    "CudaArgument",
    "CudaBenchmark",
    "CudaBenchmarkCase",
    "CudaBufferArgument",
    "CudaBufferInit",
    "CudaBufferRole",
    "CudaBenchmarkProvider",
    "CudaLaunchSpec",
    "CudaScalarArgument",
    "CudaScalarDType",
    "compile_options_sha256",
    "cuda_source_sha256",
    "NamedBufferOutput",
    "MeasurementPolicy",
    "measurement_policy_canonical_json",
    "measurement_id_for",
    "ProfileEnvironment",
    "ProfileRun",
    "ProfileRunner",
    "summarize_profile_run",
]
