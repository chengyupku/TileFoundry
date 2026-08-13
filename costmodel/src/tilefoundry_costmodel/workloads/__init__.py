"""Workload frontend protocols and deterministic catalog resolution.

M1 defines the standalone protocol only.  Workload-specific GEMM/GQA/
FlashAttention/MLP program builders remain later milestones.
"""

from .base import WorkloadFrontend, WorkloadFrontendCatalog, builtin_workload_frontends

__all__ = ["WorkloadFrontend", "WorkloadFrontendCatalog", "builtin_workload_frontends"]
