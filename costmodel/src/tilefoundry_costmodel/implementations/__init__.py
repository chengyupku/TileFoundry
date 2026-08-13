"""Operation implementation catalogs and lowering protocols."""

from __future__ import annotations

from .b200 import B200CopyLowering, B200CopyProvider, b200_copy_implementation
from .base import (
    CudaBenchmarkProvider,
    LoweredTileOp,
    LoweringContext,
    TileOpImplementation,
    TileOpLowering,
)
from .registry import ImplementationCatalog
from .synthetic import (
    SyntheticBenchmarkProvider,
    SyntheticLowering,
    synthetic_implementation_catalog,
)


def b200_implementation_catalog() -> ImplementationCatalog:
    """Return the installed real B200 M3 implementation catalog."""

    return ImplementationCatalog((b200_copy_implementation(),))


__all__ = [
    "CudaBenchmarkProvider",
    "B200CopyLowering",
    "B200CopyProvider",
    "ImplementationCatalog",
    "LoweredTileOp",
    "LoweringContext",
    "SyntheticBenchmarkProvider",
    "SyntheticLowering",
    "TileOpImplementation",
    "TileOpLowering",
    "b200_implementation_catalog",
    "b200_copy_implementation",
    "synthetic_implementation_catalog",
]
