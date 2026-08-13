"""Version constants owned by the public cost-model contract.

The values in this module mirror the constants in ``docs/spec/cost-model.md``.
Keeping them in one dependency-free module lets the package root be imported in
environments that do not have CUDA or OR-Tools installed.
"""

from __future__ import annotations

COST_MODEL_API_VERSION: tuple[int, int] = (2, 0)
HARDWARE_SCHEMA_VERSION: int = 1
PLAN_SCHEMA_VERSION: int = 2
PROFILE_SCHEMA_VERSION: int = 1
PROGRAM_SCHEMA_VERSION: int = 2
REQUEST_SCHEMA_VERSION: int = 2
SEARCH_PROBLEM_SCHEMA_VERSION: int = 2
RESULT_SCHEMA_VERSION: int = 2

__all__ = [
    "COST_MODEL_API_VERSION",
    "HARDWARE_SCHEMA_VERSION",
    "PLAN_SCHEMA_VERSION",
    "PROFILE_SCHEMA_VERSION",
    "PROGRAM_SCHEMA_VERSION",
    "REQUEST_SCHEMA_VERSION",
    "SEARCH_PROBLEM_SCHEMA_VERSION",
    "RESULT_SCHEMA_VERSION",
]
