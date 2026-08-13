"""Public version-2 orchestration and serialization entry points."""

from __future__ import annotations

from ._serialization import (
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
)
from .constants import RESULT_SCHEMA_VERSION
from .errors import (
    CostModelError,
    InvalidRequestError,
    MissingProfileError,
    ProfileRunError,
    SearchProblemError,
    UnsupportedError,
)
from .hardware import HardwareCatalog
from .implementations import ImplementationCatalog
from .model import ProfileKeyId
from .profiler.base import MeasurementPolicy, ProfileRunner
from .profiles.store import SqliteProfileStore
from .request import CostModelRequest
from .result import CostModelResult, Diagnostic, DiagnosticCode, EvaluationStatus
from .solver.model import SearchProblem


def build(
    request: CostModelRequest,
    *,
    hardware_catalog: HardwareCatalog,
    implementation_catalog: ImplementationCatalog,
    profile_store: SqliteProfileStore,
    profile_runner: ProfileRunner | None = None,
    measurement_policy: MeasurementPolicy = MeasurementPolicy(),
) -> SearchProblem:
    """Build a replayable problem once later milestones are installed."""

    if type(request) is not CostModelRequest:
        raise InvalidRequestError("request must be CostModelRequest")
    del (
        request,
        hardware_catalog,
        implementation_catalog,
        profile_store,
        profile_runner,
        measurement_policy,
    )
    raise UnsupportedError("public profile-resolving build is scheduled for M4")


def solve(problem: SearchProblem) -> CostModelResult:
    """Solve one replayable problem once a solver backend is installed."""

    if type(problem) is not SearchProblem:
        raise SearchProblemError("problem must be SearchProblem")
    del problem
    raise UnsupportedError("cost-model solve is scheduled for M4")


def evaluate(
    request: CostModelRequest,
    *,
    hardware_catalog: HardwareCatalog,
    implementation_catalog: ImplementationCatalog,
    profile_store: SqliteProfileStore,
    profile_runner: ProfileRunner | None = None,
    measurement_policy: MeasurementPolicy = MeasurementPolicy(),
) -> CostModelResult:
    """Build and solve one request through the typed orchestration boundary."""

    try:
        problem = build(
            request,
            hardware_catalog=hardware_catalog,
            implementation_catalog=implementation_catalog,
            profile_store=profile_store,
            profile_runner=profile_runner,
            measurement_policy=measurement_policy,
        )
        return solve(problem)
    except UnsupportedError as exc:
        return CostModelResult(
            schema_version=RESULT_SCHEMA_VERSION,
            status=EvaluationStatus.UNSUPPORTED,
            diagnostics=(Diagnostic(DiagnosticCode.UNSUPPORTED, str(exc)),),
        )
    except MissingProfileError as exc:
        return CostModelResult(
            schema_version=RESULT_SCHEMA_VERSION,
            status=EvaluationStatus.MISSING_PROFILE,
            missing_profiles=tuple(ProfileKeyId(key_id) for key_id in exc.key_ids),
            diagnostics=(Diagnostic(DiagnosticCode.MISSING_PROFILE, str(exc)),),
        )
    except ProfileRunError as exc:
        return CostModelResult(
            schema_version=RESULT_SCHEMA_VERSION,
            status=EvaluationStatus.PROFILE_FAILED,
            diagnostics=(Diagnostic(DiagnosticCode.PROFILE_FAILED, str(exc)),),
        )
    except CostModelError:
        raise


__all__ = [
    "build",
    "evaluate",
    "hardware_from_json",
    "hardware_to_json",
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
