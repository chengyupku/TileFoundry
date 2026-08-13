"""Typed lowering/provider pair boundary for M2.

Phase fragments are owned by ``tileop`` and benchmark artifacts by
``profiler.base``; this module owns the immutable pair joining those protocols.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..errors import WorkloadError
from ..model import validate_identifier
from ..profiler.base import CudaBenchmarkProvider
from ..program import TileOpKind
from ..tileop import (
    LoweredTileOp,
    LoweringContext,
    TileOpLowering,
)


def _identifier(value: object, label: str) -> None:
    if not isinstance(value, str):
        raise WorkloadError(f"{label} must be a non-empty ASCII string")
    try:
        validate_identifier(value, label=label)
    except Exception as exc:
        raise WorkloadError(str(exc)) from exc


@dataclass(frozen=True, slots=True)
class TileOpImplementation:
    """Pair one typed lowering with its exact benchmark provider."""

    lowering: TileOpLowering
    benchmark_provider: CudaBenchmarkProvider

    def __post_init__(self) -> None:
        for item, label in (
            (self.lowering, "lowering"),
            (self.benchmark_provider, "benchmark_provider"),
        ):
            if item is None:
                raise WorkloadError(f"{label} must be provided")
        op_kind = getattr(self.lowering, "op_kind", None)
        implementation_id = getattr(self.lowering, "implementation_id", None)
        if not isinstance(op_kind, TileOpKind):
            raise WorkloadError("lowering op_kind must be TileOpKind")
        _identifier(implementation_id, "implementation_id")
        provider_id = getattr(self.benchmark_provider, "provider_id", None)
        _identifier(provider_id, "provider_id")
        _identifier(getattr(self.benchmark_provider, "provider_version", None), "provider_version")
        for item, method_names, label in (
            (self.lowering, ("supports", "lower"), "lowering"),
            (
                self.benchmark_provider,
                ("supports", "fingerprint", "materialize", "validate"),
                "benchmark provider",
            ),
        ):
            if any(not callable(getattr(item, name, None)) for name in method_names):
                raise WorkloadError(f"{label} does not implement its protocol")
        declared_provider_id = getattr(self.lowering, "provider_id", None)
        if declared_provider_id is not None and declared_provider_id != provider_id:
            raise WorkloadError("lowering and benchmark provider IDs must match")


__all__ = [
    "CudaBenchmarkProvider",
    "LoweredTileOp",
    "LoweringContext",
    "TileOpImplementation",
    "TileOpLowering",
]
