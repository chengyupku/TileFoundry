"""Errors raised by the strict warpgroup scheduling boundary."""

from __future__ import annotations


class WarpgroupValidationError(ValueError):
    """A typed warpgroup document violates the scheduling contract."""


class WarpgroupSerializationError(WarpgroupValidationError):
    """A warpgroup JSON document is malformed or has the wrong format."""


class WarpgroupSolveError(RuntimeError):
    """The finite numeric scheduling model cannot produce a solve result."""


class WarpgroupInfeasibleError(WarpgroupSolveError):
    """The closed problem has no feasible finite schedule."""


class WarpgroupNoSolutionError(WarpgroupSolveError):
    """The solver stopped without an incumbent result."""


class WarpgroupModelError(WarpgroupSolveError):
    """The backend rejected the constructed model or returned an unknown state."""


class WarpgroupVerificationError(WarpgroupValidationError):
    """A schedule does not satisfy its closed warpgroup problem."""


__all__ = [
    "WarpgroupInfeasibleError",
    "WarpgroupModelError",
    "WarpgroupNoSolutionError",
    "WarpgroupSerializationError",
    "WarpgroupSolveError",
    "WarpgroupValidationError",
    "WarpgroupVerificationError",
]
