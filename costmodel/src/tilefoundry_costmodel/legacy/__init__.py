"""Compatibility namespace for the pre-version-2 fixed-stage API.

The records and schedulers in this namespace are retained verbatim so existing
``(0, 2)`` callers can migrate independently.  The version-2 package root does
not import this module.
"""

from .cpsat_solver import CpSatPipelineSolver
from .list_solver import ListPipelineSolver
from .model import (
    PipelineHardware,
    PipelineProblem,
    PipelineSolution,
    Placement,
    Precedence,
    ResourceDemand,
    ResourceSpec,
    SolveStatus,
    StageSpec,
    StageTiming,
    TimingOracle,
)

COST_MODEL_API_VERSION: tuple[int, int] = (0, 2)
__version__ = "0.2.0"

__all__ = [
    "COST_MODEL_API_VERSION",
    "CpSatPipelineSolver",
    "ListPipelineSolver",
    "PipelineHardware",
    "PipelineProblem",
    "PipelineSolution",
    "Placement",
    "Precedence",
    "ResourceDemand",
    "ResourceSpec",
    "SolveStatus",
    "StageSpec",
    "StageTiming",
    "TimingOracle",
    "__version__",
]
