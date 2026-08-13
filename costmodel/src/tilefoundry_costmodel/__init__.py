"""Standalone TileFoundry cost-model API ``(2, 0)``.

The root imports only dependency-free records and orchestration boundaries.  The
pre-version-2 fixed-stage API is available explicitly from
``tilefoundry_costmodel.legacy`` and is never imported here.
"""

from __future__ import annotations

from . import language as T
from .api import (
    build,
    evaluate,
    hardware_from_json,
    hardware_to_json,
    plan_from_json,
    plan_to_json,
    problem_from_json,
    problem_to_json,
    profile_snapshot_from_json,
    profile_snapshot_to_json,
    program_from_json,
    program_to_json,
    render_timeline,
    request_from_json,
    request_to_json,
    result_from_json,
    result_to_json,
    solve,
)
from .constants import (
    COST_MODEL_API_VERSION,
    HARDWARE_SCHEMA_VERSION,
    PLAN_SCHEMA_VERSION,
    PROFILE_SCHEMA_VERSION,
    PROGRAM_SCHEMA_VERSION,
    REQUEST_SCHEMA_VERSION,
    RESULT_SCHEMA_VERSION,
    SEARCH_PROBLEM_SCHEMA_VERSION,
)
from .errors import (
    CostModelError,
    HardwareSpecError,
    InvalidRequestError,
    MissingProfileError,
    ProfileConflictError,
    ProfileError,
    ProfileRunError,
    ProfileStoreError,
    SearchProblemError,
    SolverError,
    UnsupportedError,
    WorkloadError,
)
from .hardware import HardwareCatalog, b200_hardware_catalog
from .implementations import b200_implementation_catalog
from .profiler.base import MeasurementPolicy, ProfileRunner
from .profiler.cuda import LocalCudaProfileRunner
from .profiles.store import SqliteProfileStore, open_profile_store
from .request import (
    CostModelRequest,
    ProfileMode,
    ProfileSelection,
    ProfileSnapshotRef,
    SearchSpace,
    SolverOptions,
    TimingStatistic,
    WarpConfig,
    WarpRole,
    WarpRoleAssignment,
)
from .result import (
    CostModelPlan,
    CostModelResult,
    Diagnostic,
    DiagnosticCode,
    EvaluationStatus,
    RejectedCandidate,
)
from .solver import SearchProblem

try:
    from importlib.metadata import PackageNotFoundError, version

    try:
        _metadata_version = version("tilefoundry-costmodel")
        __version__ = _metadata_version if _metadata_version.startswith("2.") else "2.0.0"
    except PackageNotFoundError:
        __version__ = "2.0.0"
except ImportError:  # pragma: no cover - importlib.metadata is in supported Python
    __version__ = "2.0.0"

__all__ = [
    "COST_MODEL_API_VERSION",
    "CostModelError",
    "CostModelPlan",
    "CostModelRequest",
    "CostModelResult",
    "Diagnostic",
    "DiagnosticCode",
    "EvaluationStatus",
    "HARDWARE_SCHEMA_VERSION",
    "HardwareCatalog",
    "HardwareSpecError",
    "InvalidRequestError",
    "LocalCudaProfileRunner",
    "MeasurementPolicy",
    "MissingProfileError",
    "PLAN_SCHEMA_VERSION",
    "PROFILE_SCHEMA_VERSION",
    "ProfileConflictError",
    "ProfileError",
    "ProfileMode",
    "ProfileRunError",
    "ProfileRunner",
    "ProfileSelection",
    "ProfileSnapshotRef",
    "ProfileStoreError",
    "PROGRAM_SCHEMA_VERSION",
    "REQUEST_SCHEMA_VERSION",
    "RejectedCandidate",
    "RESULT_SCHEMA_VERSION",
    "SEARCH_PROBLEM_SCHEMA_VERSION",
    "SearchProblem",
    "SearchProblemError",
    "SearchSpace",
    "SolverError",
    "SolverOptions",
    "SqliteProfileStore",
    "T",
    "TimingStatistic",
    "UnsupportedError",
    "WarpConfig",
    "WarpRole",
    "WarpRoleAssignment",
    "WorkloadError",
    "__version__",
    "b200_hardware_catalog",
    "b200_implementation_catalog",
    "build",
    "evaluate",
    "hardware_from_json",
    "hardware_to_json",
    "open_profile_store",
    "plan_from_json",
    "plan_to_json",
    "problem_from_json",
    "problem_to_json",
    "program_from_json",
    "program_to_json",
    "profile_snapshot_from_json",
    "profile_snapshot_to_json",
    "render_timeline",
    "request_from_json",
    "request_to_json",
    "result_from_json",
    "result_to_json",
    "solve",
]
