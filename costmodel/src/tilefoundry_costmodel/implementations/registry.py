"""Deterministic operation implementation catalog."""

from __future__ import annotations

from dataclasses import dataclass

from ..errors import UnsupportedError, WorkloadError
from ..model import validate_identifier
from ..program import CopyOp, ElementwiseOp, GemmOp, ReduceOp, TileOp, TileOpKind
from ..tileop import LoweringContext
from .base import TileOpImplementation


def _id(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise WorkloadError(f"{label} must be a non-empty ASCII string")
    try:
        return validate_identifier(value, label=label)
    except Exception as exc:
        raise WorkloadError(str(exc)) from exc


@dataclass(frozen=True, slots=True)
class ImplementationCatalog:
    """Own one canonical set of lowering/provider pairs."""

    implementations: tuple[TileOpImplementation, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.implementations, (tuple, list)):
            raise WorkloadError("implementations must be a sequence")
        values = tuple(self.implementations)
        if not all(type(item) is TileOpImplementation for item in values):
            raise WorkloadError("implementations must contain TileOpImplementation records")
        pairs: list[tuple[str, str]] = []
        provider_ids: list[str] = []
        for pair in values:
            lowering = pair.lowering
            op_kind = getattr(lowering, "op_kind", None)
            implementation_id = getattr(lowering, "implementation_id", None)
            if not isinstance(op_kind, str) and not hasattr(op_kind, "value"):
                raise WorkloadError("lowering op_kind is invalid")
            op_name = str(getattr(op_kind, "value", op_kind))
            normalized_implementation_id = _id(implementation_id, "implementation_id")
            provider_id = getattr(pair.benchmark_provider, "provider_id", None)
            provider_version = getattr(pair.benchmark_provider, "provider_version", None)
            provider_ids.append(_id(provider_id, "provider_id"))
            _id(provider_version, "provider_version")
            pairs.append((op_name, normalized_implementation_id))
        if len(pairs) != len(set(pairs)):
            raise WorkloadError("duplicate operation implementation pair")
        if len(provider_ids) != len(set(provider_ids)):
            raise WorkloadError("duplicate benchmark provider ID")
        object.__setattr__(
            self,
            "implementations",
            tuple(
                sorted(
                    values,
                    key=lambda item: (
                        str(getattr(item.lowering.op_kind, "value", item.lowering.op_kind)),
                        str(item.lowering.implementation_id),
                    ),
                )
            ),
        )

    def choices_for(
        self,
        op: TileOp,
        *,
        context: LoweringContext,
        allowed_implementation_ids: tuple[str, ...],
    ) -> tuple[TileOpImplementation, ...]:
        """Return all legal allow-listed pairs in canonical order.

        Lowering legality is checked here. The builder checks the paired
        provider against every concrete query emitted by that lowering. Neither
        step measures or invokes a CUDA runtime.
        """

        if type(op) not in (CopyOp, GemmOp, ReduceOp, ElementwiseOp):
            raise WorkloadError("op must be a concrete TileOp")
        if type(context) is not LoweringContext:
            raise WorkloadError("context must be LoweringContext")
        if not isinstance(allowed_implementation_ids, (tuple, list)):
            raise WorkloadError("allowed_implementation_ids must be a sequence")
        allowed = tuple(allowed_implementation_ids)
        if any(not isinstance(item, str) for item in allowed):
            raise WorkloadError("allowed implementation IDs must be strings")
        for item in allowed:
            _id(item, "allowed implementation ID")
        if len(allowed) != len(set(allowed)):
            raise WorkloadError("allowed implementation IDs must be unique")
        choices: list[TileOpImplementation] = []
        for pair in self.implementations:
            lowering = pair.lowering
            if str(getattr(lowering.op_kind, "value", lowering.op_kind)) != str(op.kind.value):
                continue
            if lowering.implementation_id not in allowed:
                continue
            try:
                legal = lowering.supports(op, context=context)
            except Exception as exc:
                raise WorkloadError("implementation support check failed") from exc
            if not legal:
                continue
            choices.append(pair)
        return tuple(choices)

    def pair_for(self, op_kind: TileOpKind, implementation_id: str) -> TileOpImplementation:
        """Resolve one exact pair, rejecting nearby or ambiguous IDs."""

        if not isinstance(op_kind, TileOpKind):
            raise WorkloadError("op_kind must be TileOpKind")
        name = op_kind.value
        _id(implementation_id, "implementation_id")
        matches = tuple(
            item
            for item in self.implementations
            if str(getattr(item.lowering.op_kind, "value", item.lowering.op_kind)) == name
            and item.lowering.implementation_id == implementation_id
        )
        if len(matches) != 1:
            raise UnsupportedError(f"no unique implementation pair for {name}:{implementation_id}")
        return matches[0]


__all__ = ["ImplementationCatalog"]
