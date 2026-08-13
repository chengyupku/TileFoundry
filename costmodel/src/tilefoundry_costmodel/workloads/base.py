"""Typed workload frontend protocol and catalog."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, cast

from ..errors import UnsupportedError, WorkloadError
from ..model import WorkloadKind, WorkloadSpec
from ..program import TileCandidate, TileProgram


class WorkloadFrontend(Protocol):
    """Construct typed programs for exactly one logical workload kind."""

    @property
    def workload_kind(self) -> WorkloadKind:
        """Exact workload discriminator served by the frontend."""
        ...

    def build_programs(
        self,
        workload: WorkloadSpec,
        *,
        tiles: tuple[TileCandidate, ...],
    ) -> tuple[TileProgram, ...]:
        """Return deterministic concrete program variants."""


@dataclass(frozen=True, slots=True)
class _UnavailableFrontend:
    workload_kind: WorkloadKind

    def build_programs(
        self,
        workload: WorkloadSpec,
        *,
        tiles: tuple[TileCandidate, ...],
    ) -> tuple[TileProgram, ...]:
        del workload, tiles
        raise UnsupportedError(
            f"workload frontend for {self.workload_kind.value} is not installed in M1"
        )


@dataclass(frozen=True, slots=True)
class WorkloadFrontendCatalog:
    """Index at most one frontend per supported workload kind."""

    frontends: tuple[WorkloadFrontend, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.frontends, (tuple, list)):
            raise WorkloadError("frontends must be a sequence")
        frontends = tuple(self.frontends)
        if not all(_looks_like_frontend(frontend) for frontend in frontends):
            raise WorkloadError("frontends must implement WorkloadFrontend")
        kinds: list[WorkloadKind] = []
        for frontend in frontends:
            kind = _coerce_kind(getattr(frontend, "workload_kind"))
            kinds.append(kind)
        if len(kinds) != len(set(kinds)):
            raise WorkloadError("workload frontend kinds must be unique")
        object.__setattr__(
            self,
            "frontends",
            tuple(sorted(frontends, key=lambda frontend: getattr(frontend, "workload_kind").value)),
        )

    def frontend_for(self, kind: WorkloadKind) -> WorkloadFrontend:
        """Resolve one exact workload frontend."""

        requested = _coerce_kind(kind)
        for frontend in self.frontends:
            if getattr(frontend, "workload_kind") is requested:
                return frontend
        raise UnsupportedError(f"no frontend is installed for {requested.value}")


def builtin_workload_frontends() -> WorkloadFrontendCatalog:
    """Return the deterministic catalog of the four supported workload kinds.

    The entries deliberately raise ``UnsupportedError`` when called: M1 owns
    the protocol and exact dispatch, while concrete workload programs begin in
    later milestones.
    """

    return WorkloadFrontendCatalog(
        cast(
            tuple[WorkloadFrontend, ...],
            tuple(_UnavailableFrontend(kind) for kind in WorkloadKind),
        )
    )


def _looks_like_frontend(value: object) -> bool:
    return isinstance(getattr(value, "workload_kind", None), WorkloadKind) and callable(
        getattr(value, "build_programs", None)
    )


def _coerce_kind(value: object) -> WorkloadKind:
    if isinstance(value, WorkloadKind):
        return value
    try:
        return WorkloadKind(value)
    except (TypeError, ValueError) as exc:
        raise WorkloadError(f"invalid workload kind: {value!r}") from exc


__all__ = ["WorkloadFrontend", "WorkloadFrontendCatalog", "builtin_workload_frontends"]
