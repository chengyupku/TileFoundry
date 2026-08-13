"""Public exception hierarchy for the standalone cost model."""

from __future__ import annotations

from collections.abc import Iterable


class CostModelError(Exception):
    """Base cost-model failure."""


class InvalidRequestError(CostModelError, ValueError):
    """Report malformed caller input."""


class HardwareSpecError(CostModelError, ValueError):
    """Report an invalid hardware document."""


class WorkloadError(CostModelError, ValueError):
    """Report an invalid logical workload or typed program."""


class UnsupportedError(CostModelError):
    """Report a valid request outside installed capability."""


class ProfileError(CostModelError):
    """Base timing-profile failure."""


class MissingProfileError(ProfileError):
    """Report every required profile key absent from a snapshot."""

    key_ids: tuple[str, ...]

    def __init__(self, key_ids: Iterable[str], message: str | None = None) -> None:
        self.key_ids = tuple(sorted(set(key_ids)))
        detail = message or "missing required timing profiles"
        if self.key_ids:
            detail = f"{detail}: {', '.join(self.key_ids)}"
        super().__init__(detail)


class ProfileConflictError(ProfileError):
    """Report conflicting immutable profile data."""


class ProfileStoreError(ProfileError):
    """Report profile-store corruption or schema failure."""


class ProfileRunError(ProfileError):
    """Report CUDA compilation, execution, or validation failure."""


class SearchProblemError(CostModelError, ValueError):
    """Report an invalid solver-ready problem."""


class SolverError(CostModelError):
    """Report an internal solver failure."""


__all__ = [
    "CostModelError",
    "InvalidRequestError",
    "HardwareSpecError",
    "WorkloadError",
    "UnsupportedError",
    "ProfileError",
    "MissingProfileError",
    "ProfileConflictError",
    "ProfileStoreError",
    "ProfileRunError",
    "SearchProblemError",
    "SolverError",
]
