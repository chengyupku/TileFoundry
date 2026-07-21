"""Backend-neutral software-pipeline scheduling API owned by TileFoundry."""

from importlib.metadata import PackageNotFoundError, version

from .cpsat_solver import CpSatPipelineSolver
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
from .solver import ListPipelineSolver

COST_MODEL_API_VERSION = (0, 2)

try:
    __version__ = version("tilefoundry-costmodel")
except PackageNotFoundError:
    __version__ = "0+unknown"

__all__ = [
    "CpSatPipelineSolver",
    "COST_MODEL_API_VERSION",
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
