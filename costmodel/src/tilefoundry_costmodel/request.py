"""Immutable request-side records for the version 2 public API."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from enum import Enum

from .constants import PROGRAM_SCHEMA_VERSION, REQUEST_SCHEMA_VERSION
from .errors import InvalidRequestError
from .model import (
    FlashAttentionSpec,
    GemmSpec,
    GqaDecodeSpec,
    HardwareSpecRef,
    MlpSpec,
    WorkloadKind,
    WorkloadSpec,
    validate_identifier,
)
from .program import TileProgram


class WarpRole(str, Enum):
    """Name one initial B200 warp role."""

    TMA_PRODUCER = "tma_producer"
    TENSOR_CONSUMER = "tensor_consumer"
    CUDA_EPILOGUE = "cuda_epilogue"


@dataclass(frozen=True, slots=True)
class WarpRoleAssignment:
    role: WarpRole
    warp_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        _coerce_enum(self, "role", WarpRole)
        _require_sequence(self.warp_ids, "warp_ids")
        ids = tuple(self.warp_ids)
        if not ids:
            raise InvalidRequestError("warp role assignment must contain at least one warp")
        if len(ids) != len(set(ids)) or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in ids
        ):
            raise InvalidRequestError("warp IDs must be unique non-negative integers")
        object.__setattr__(self, "warp_ids", tuple(sorted(ids)))


@dataclass(frozen=True, slots=True)
class WarpConfig:
    config_id: str
    total_warps: int
    roles: tuple[WarpRoleAssignment, ...]

    def __post_init__(self) -> None:
        validate_identifier(self.config_id, label="warp config ID")
        if (
            isinstance(self.total_warps, bool)
            or not isinstance(self.total_warps, int)
            or self.total_warps <= 0
        ):
            raise InvalidRequestError("total_warps must be positive")
        _require_sequence(self.roles, "roles")
        roles = tuple(self.roles)
        if not all(type(assignment) is WarpRoleAssignment for assignment in roles):
            raise InvalidRequestError("roles must contain WarpRoleAssignment records")
        # One assignment per role is canonical.  The same physical warp may
        # participate in multiple different roles; only same-role overlap is
        # forbidden by the public contract.
        role_ids: set[WarpRole] = set()
        all_ids: list[int] = []
        for assignment in roles:
            if assignment.role in role_ids:
                raise InvalidRequestError("warp roles must be unique")
            role_ids.add(assignment.role)
            all_ids.extend(assignment.warp_ids)
        if any(warp_id >= self.total_warps for warp_id in all_ids):
            raise InvalidRequestError("warp IDs must be less than total_warps")
        object.__setattr__(
            self,
            "roles",
            tuple(sorted(roles, key=lambda item: item.role.value)),
        )


@dataclass(frozen=True, slots=True)
class SearchSpace:
    implementation_ids: tuple[str, ...]
    warp_configs: tuple[WarpConfig, ...]
    pipeline_depths: tuple[int, ...]
    layout_variant_ids: tuple[str, ...] = ()
    max_candidates: int = 10_000

    def __post_init__(self) -> None:
        _require_sequence(self.implementation_ids, "implementation_ids")
        _require_sequence(self.warp_configs, "warp_configs")
        _require_sequence(self.pipeline_depths, "pipeline_depths")
        _require_sequence(self.layout_variant_ids, "layout_variant_ids")
        implementations = tuple(self.implementation_ids)
        warps = tuple(self.warp_configs)
        depths = tuple(self.pipeline_depths)
        layouts = tuple(self.layout_variant_ids)
        if not layouts:
            layouts = ("default",)
        if not implementations or any(
            not isinstance(value, str) or not value for value in implementations
        ):
            raise InvalidRequestError("implementation_ids must be non-empty strings")
        for value in implementations:
            validate_identifier(value, label="implementation_id")
        if len(implementations) != len(set(implementations)):
            raise InvalidRequestError("implementation_ids must be unique")
        if not warps:
            raise InvalidRequestError("warp_configs must not be empty")
        if not all(type(value) is WarpConfig for value in warps):
            raise InvalidRequestError("warp_configs must contain WarpConfig records")
        warp_ids = tuple(config.config_id for config in warps)
        if len(warp_ids) != len(set(warp_ids)):
            raise InvalidRequestError("warp config IDs must be unique")
        warp_shapes = tuple(
            (
                config.total_warps,
                tuple((assignment.role.value, assignment.warp_ids) for assignment in config.roles),
            )
            for config in warps
        )
        if len(warp_shapes) != len(set(warp_shapes)):
            raise InvalidRequestError("warp_configs must not contain duplicate canonical choices")
        if (
            not depths
            or len(depths) != len(set(depths))
            or any(
                isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 8
                for value in depths
            )
        ):
            raise InvalidRequestError("pipeline_depths must be unique integers in 1..8")
        if any(not isinstance(value, str) or not value for value in layouts):
            raise InvalidRequestError("layout_variant_ids must contain non-empty strings")
        for value in layouts:
            validate_identifier(value, label="layout_variant_id")
        if len(layouts) != len(set(layouts)):
            raise InvalidRequestError("layout_variant_ids must be unique")
        if (
            isinstance(self.max_candidates, bool)
            or not isinstance(self.max_candidates, int)
            or self.max_candidates <= 0
        ):
            raise InvalidRequestError("max_candidates must be positive")
        object.__setattr__(self, "implementation_ids", tuple(sorted(implementations)))
        object.__setattr__(
            self, "warp_configs", tuple(sorted(warps, key=lambda item: item.config_id))
        )
        object.__setattr__(self, "pipeline_depths", tuple(sorted(depths)))
        object.__setattr__(self, "layout_variant_ids", tuple(sorted(layouts)))


class ProfileMode(str, Enum):
    """Control behavior on an exact profile miss."""

    REQUIRE = "require"
    JIT_ON_MISS = "jit_on_miss"


class TimingStatistic(str, Enum):
    """Select a measured timing aggregate."""

    P50 = "p50"
    P90 = "p90"


@dataclass(frozen=True, slots=True)
class ProfileSnapshotRef:
    snapshot_id: str
    revision: int

    def __post_init__(self) -> None:
        validate_identifier(self.snapshot_id, label="snapshot_id")
        if (
            isinstance(self.revision, bool)
            or not isinstance(self.revision, int)
            or self.revision <= 0
        ):
            raise InvalidRequestError("snapshot revision must be positive")


@dataclass(frozen=True, slots=True)
class ProfileSelection:
    snapshot: ProfileSnapshotRef
    mode: ProfileMode = ProfileMode.REQUIRE
    timing_statistic: TimingStatistic = TimingStatistic.P50

    def __post_init__(self) -> None:
        if type(self.snapshot) is not ProfileSnapshotRef:
            raise InvalidRequestError("snapshot must be ProfileSnapshotRef")
        _coerce_enum(self, "mode", ProfileMode)
        _coerce_enum(self, "timing_statistic", TimingStatistic)


@dataclass(frozen=True, slots=True)
class SolverOptions:
    """Configure candidate search and numeric solving.

    The fields are intentionally solver-neutral.  The request boundary does
    not import or configure OR-Tools; a later solver can consume this frozen
    replayable value without sharing mutable backend state.
    """

    candidate_timeout_s: float | None = 30.0
    search_timeout_s: float | None = None
    time_resolution_ps: int = 10
    ortools_workers: int = 1
    candidate_workers: int = 1
    random_seed: int = 0
    finite_unroll_limit: int = 64
    stop_after_first_solution: bool = False
    deterministic: bool = True

    def __post_init__(self) -> None:
        for name in ("candidate_timeout_s", "search_timeout_s"):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or value <= 0
            ):
                raise InvalidRequestError(f"{name} must be positive or None")
        for name in (
            "time_resolution_ps",
            "ortools_workers",
            "candidate_workers",
            "finite_unroll_limit",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise InvalidRequestError(f"{name} must be a positive integer")
        if isinstance(self.random_seed, bool) or not isinstance(self.random_seed, int):
            raise InvalidRequestError("random_seed must be an integer")
        if not isinstance(self.stop_after_first_solution, bool) or not isinstance(
            self.deterministic, bool
        ):
            raise InvalidRequestError("solver boolean options must be boolean")
        if self.deterministic and (
            self.ortools_workers != 1
            or self.candidate_workers != 1
            or self.stop_after_first_solution
        ):
            raise InvalidRequestError("deterministic mode requires one worker and no early stop")


@dataclass(frozen=True, slots=True)
class CostModelRequest:
    """Describe one finite version 2 cost-model search."""

    schema_version: int
    request_id: str
    workload: WorkloadSpec
    programs: tuple[TileProgram, ...]
    hardware: HardwareSpecRef
    search_space: SearchSpace
    profiles: ProfileSelection
    solver: SolverOptions = field(default_factory=SolverOptions)

    def __post_init__(self) -> None:
        if (
            isinstance(self.schema_version, bool)
            or not isinstance(self.schema_version, int)
            or self.schema_version != REQUEST_SCHEMA_VERSION
        ):
            raise InvalidRequestError(
                f"unsupported request schema version: {self.schema_version!r}"
            )
        validate_identifier(self.request_id, label="request_id")
        if not isinstance(self.programs, (tuple, list)):
            raise InvalidRequestError("programs must be a sequence")
        programs = tuple(self.programs)
        if not programs:
            raise InvalidRequestError("request must contain at least one program")
        if type(self.workload) not in (GemmSpec, GqaDecodeSpec, FlashAttentionSpec, MlpSpec):
            raise InvalidRequestError("workload must be a typed workload record")
        if type(self.hardware) is not HardwareSpecRef:
            raise InvalidRequestError("hardware must be HardwareSpecRef")
        if type(self.search_space) is not SearchSpace:
            raise InvalidRequestError("search_space must be SearchSpace")
        if type(self.profiles) is not ProfileSelection:
            raise InvalidRequestError("profiles must be ProfileSelection")
        if not all(type(program) is TileProgram for program in programs):
            raise InvalidRequestError("programs must contain TileProgram records")
        programs = tuple(sorted(programs, key=lambda program: str(program.program_id)))
        object.__setattr__(self, "programs", programs)
        ids = tuple(str(program.program_id) for program in programs)
        if len(ids) != len(set(ids)):
            raise InvalidRequestError("program IDs must be unique")
        if any(program.schema_version != PROGRAM_SCHEMA_VERSION for program in programs):
            raise InvalidRequestError("request contains an unsupported program schema version")
        try:
            canonical_programs = tuple(_canonical_program_key(program) for program in programs)
        except (InvalidRequestError, TypeError, ValueError) as exc:
            raise InvalidRequestError("request programs must have a canonical JSON form") from exc
        if len(canonical_programs) != len(set(canonical_programs)):
            raise InvalidRequestError("request contains duplicate canonical programs")
        if hasattr(self.workload, "kind") and any(
            program.workload_kind != self.workload.kind for program in programs
        ):
            raise InvalidRequestError("program workload kinds must match request workload")
        for program in programs:
            _validate_frontend_tile_axes(program, self.workload.kind)
        if type(self.solver) is not SolverOptions:
            raise InvalidRequestError("solver must be SolverOptions")

    def to_json(self) -> str:
        from ._serialization import request_to_json

        return request_to_json(self)

    @classmethod
    def from_json(cls, text: str) -> "CostModelRequest":
        from ._serialization import request_from_json

        return request_from_json(text)


def _coerce_enum(instance: object, field_name: str, enum_type: type[Enum]) -> None:
    value = getattr(instance, field_name)
    if isinstance(value, enum_type):
        return
    try:
        object.__setattr__(instance, field_name, enum_type(value))
    except (TypeError, ValueError) as exc:
        raise InvalidRequestError(f"invalid {field_name}") from exc


def _require_sequence(value: object, label: str) -> None:
    if not isinstance(value, (tuple, list)):
        raise InvalidRequestError(f"{label} must be a sequence")


def _canonical_program_key(program: TileProgram) -> str:
    """Return a variant identity that excludes only the program-local ID."""

    from ._serialization import canonical_json, program_to_json

    payload = json.loads(program_to_json(program))
    if not isinstance(payload, dict):  # pragma: no cover - guarded by the codec
        raise TypeError("program JSON must be an object")
    payload.pop("program_id", None)
    return canonical_json(payload)


def _validate_frontend_tile_axes(program: TileProgram, kind: object) -> None:
    """Enforce the named tile-axis contract for executable typed programs."""

    if not program.operations:
        return
    required_by_kind = {
        "gemm": {"m", "n", "k"},
        "gqa_decode": {"q_heads", "kv_tokens", "head_dim"},
        "flash_attention": {"q_tokens", "kv_tokens", "head_dim"},
        "mlp": {"up_m", "up_n", "up_k", "down_m", "down_n", "down_k"},
    }
    key = kind.value if isinstance(kind, WorkloadKind) else str(kind)
    required = required_by_kind.get(key)
    if required is None:
        raise InvalidRequestError("unknown workload kind")
    actual = {axis.name for axis in program.tile.shape.axes}
    if actual != required:
        raise InvalidRequestError(f"tile shape for {key} must contain exactly {sorted(required)}")


__all__ = [
    "CostModelRequest",
    "ProfileMode",
    "ProfileSelection",
    "ProfileSnapshotRef",
    "SearchSpace",
    "SolverOptions",
    "TimingStatistic",
    "WarpConfig",
    "WarpRole",
    "WarpRoleAssignment",
]
