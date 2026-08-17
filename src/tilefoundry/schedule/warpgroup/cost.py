"""Identity-free operation signatures and exact immutable cost libraries."""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Protocol

from ._identifiers import validate_id
from .errors import WarpgroupValidationError
from .model import DType, MemorySpace, ResourceCapacity, ResourceWindow

HARDWARE_FORMAT = "tilefoundry.warpgroup_hardware"


class WarpgroupCostError(WarpgroupValidationError):
    """A cost library cannot provide an exact numeric closure."""


class WarpgroupCostMissingError(WarpgroupCostError):
    """No exact cost entry exists for one operation signature."""

    def __init__(self, signature: OperationSignature) -> None:
        self.signature = signature
        super().__init__(f"missing cost for operation signature {signature.canonical_key}")


class WarpgroupCostAmbiguityError(WarpgroupCostError):
    """More than one entry claims the same canonical operation signature."""


class WarpgroupMissingSignaturesError(WarpgroupCostError):
    """A build could not resolve all signatures and returned no problem."""

    def __init__(self, signatures: Sequence[OperationSignature]) -> None:
        self.signatures = tuple(signatures)
        rendered = ", ".join(signature.canonical_key for signature in self.signatures)
        super().__init__(f"missing cost signatures ({len(self.signatures)}): {rendered}")


class OperationKind(str, Enum):
    """The coarse work class used by a cost catalog."""

    COMPUTE = "compute"
    VIEW = "view"
    COPY = "copy"


@dataclass(frozen=True, slots=True)
class SignatureValueType:
    """A value type with aliases and SSA spelling erased."""

    shape: tuple[int, ...]
    dtype: DType
    space: MemorySpace

    def __post_init__(self) -> None:
        if type(self.shape) is not tuple or not self.shape:
            raise WarpgroupValidationError("signature value shape must be a non-empty tuple")
        if not all(type(extent) is int and extent > 0 for extent in self.shape):
            raise WarpgroupValidationError(
                "signature value shape extents must be positive integers"
            )
        if type(self.dtype) is not DType or type(self.space) is not MemorySpace:
            raise WarpgroupValidationError("signature value type has invalid dtype or space")


type CanonicalAtom = str | int | float
type CanonicalExpression = tuple[CanonicalAtom | CanonicalExpression, ...]


@dataclass(frozen=True, slots=True)
class SignatureOutput:
    """One output type and its identity-free expression tree."""

    type: SignatureValueType
    expression: CanonicalExpression

    def __post_init__(self) -> None:
        if type(self.type) is not SignatureValueType or type(self.expression) is not tuple:
            raise WarpgroupValidationError("signature output must contain exact typed values")
        _validate_expression(self.expression)


@dataclass(frozen=True, slots=True)
class OperationSignature:
    """Canonical operation semantics, independent of all caller naming."""

    kind: OperationKind
    operands: tuple[SignatureValueType, ...]
    outputs: tuple[SignatureOutput, ...]

    def __post_init__(self) -> None:
        if type(self.kind) is not OperationKind:
            raise WarpgroupValidationError("operation signature kind must be OperationKind")
        if type(self.operands) is not tuple or type(self.outputs) is not tuple:
            raise WarpgroupValidationError("operation signature collections must be tuples")
        if not self.outputs:
            raise WarpgroupValidationError("operation signature requires an output")
        if not all(type(item) is SignatureValueType for item in self.operands):
            raise WarpgroupValidationError("operation signature operands must be typed records")
        if not all(type(item) is SignatureOutput for item in self.outputs):
            raise WarpgroupValidationError("operation signature outputs must be typed records")

    @property
    def canonical_key(self) -> str:
        """Return one deterministic JSON key for exact lookup and diagnostics."""
        return json.dumps(
            {
                "kind": self.kind.value,
                "operands": [_encode_value_type(item) for item in self.operands],
                "outputs": [
                    {
                        "type": _encode_value_type(item.type),
                        "expr": _encode_expression(item.expression),
                    }
                    for item in self.outputs
                ],
            },
            sort_keys=True,
            separators=(",", ":"),
        )


@dataclass(frozen=True, slots=True)
class OperationCost:
    """Exact positive integer cost for one operation signature."""

    issue_duration: int
    completion_latency: int
    resource_windows: tuple[ResourceWindow, ...] = ()

    def __post_init__(self) -> None:
        if type(self.issue_duration) is not int or self.issue_duration <= 0:
            raise WarpgroupValidationError("issue duration must be a positive integer")
        if type(self.completion_latency) is not int or self.completion_latency <= 0:
            raise WarpgroupValidationError("completion latency must be a positive integer")
        if self.completion_latency < self.issue_duration:
            raise WarpgroupValidationError("completion latency must be >= issue duration")
        windows = self.resource_windows
        if type(windows) is not tuple or not all(type(item) is ResourceWindow for item in windows):
            raise WarpgroupValidationError(
                "operation resource windows must contain exact ResourceWindow records"
            )
        for window in windows:
            if window.duration > self.completion_latency:
                raise WarpgroupValidationError("resource window exceeds completion latency")
        object.__setattr__(self, "resource_windows", tuple(sorted(windows)))


@dataclass(frozen=True, slots=True)
class OperationCostEntry:
    """One signature-to-cost mapping in an immutable catalog."""

    signature: OperationSignature
    cost: OperationCost

    def __post_init__(self) -> None:
        if type(self.signature) is not OperationSignature or type(self.cost) is not OperationCost:
            raise WarpgroupValidationError("cost entry must contain exact typed records")


class CostLibrary(Protocol):
    """The dependency-free provider boundary consumed by problem construction."""

    @property
    def time_unit(self) -> str: ...

    @property
    def resources(self) -> tuple[ResourceCapacity, ...]: ...

    def lookup(self, signature: OperationSignature) -> OperationCost: ...


@dataclass(frozen=True, slots=True)
class WarpgroupHardware:
    """A serializable target cost catalog matched by operation signature."""

    format: str
    time_unit: str
    resources: tuple[ResourceCapacity, ...]
    entries: tuple[OperationCostEntry, ...]

    def __post_init__(self) -> None:
        if self.format != HARDWARE_FORMAT:
            raise WarpgroupValidationError(f"hardware format must be {HARDWARE_FORMAT!r}")
        validate_id(self.time_unit, "time_unit")
        if type(self.resources) is not tuple or not all(
            type(item) is ResourceCapacity for item in self.resources
        ):
            raise WarpgroupValidationError("hardware resources must be exact typed records")
        if type(self.entries) is not tuple or not all(
            type(item) is OperationCostEntry for item in self.entries
        ):
            raise WarpgroupValidationError("hardware entries must be exact typed records")
        resource_ids = tuple(item.id for item in self.resources)
        if len(resource_ids) != len(set(resource_ids)):
            raise WarpgroupValidationError("hardware contains duplicate resources")
        capacities = {item.id: item.capacity for item in self.resources}
        for entry in self.entries:
            for window in entry.cost.resource_windows:
                if window.resource_id not in capacities:
                    raise WarpgroupValidationError(
                        f"cost entry uses undefined resource {window.resource_id!r}"
                    )
                if window.amount > capacities[window.resource_id]:
                    raise WarpgroupValidationError(
                        f"cost entry demand for {window.resource_id!r} exceeds capacity"
                    )
        signatures = tuple(item.signature for item in self.entries)
        if len(signatures) != len(set(signatures)):
            raise WarpgroupCostAmbiguityError("hardware contains duplicate signatures")
        object.__setattr__(
            self, "resources", tuple(sorted(self.resources, key=lambda item: item.id))
        )
        object.__setattr__(
            self,
            "entries",
            tuple(sorted(self.entries, key=lambda item: item.signature.canonical_key)),
        )

    def lookup(self, signature: OperationSignature) -> OperationCost:
        """Resolve exactly one signature, with no default timing."""
        if type(signature) is not OperationSignature:
            raise WarpgroupValidationError("cost lookup requires an exact OperationSignature")
        matches = tuple(item.cost for item in self.entries if item.signature == signature)
        if not matches:
            raise WarpgroupCostMissingError(signature)
        if len(matches) != 1:
            raise WarpgroupCostAmbiguityError(f"ambiguous cost for {signature.canonical_key}")
        return matches[0]


def _encode_value_type(value: SignatureValueType) -> dict[str, object]:
    return {"shape": list(value.shape), "dtype": value.dtype.value, "space": value.space.value}


def _encode_expression(value: CanonicalExpression) -> object:
    return [item if type(item) is not tuple else _encode_expression(item) for item in value]


def _validate_expression(value: CanonicalExpression) -> None:
    for item in value:
        if type(item) is tuple:
            _validate_expression(item)
        elif type(item) not in (str, int, float) or (
            type(item) is float and not math.isfinite(item)
        ):
            raise WarpgroupValidationError(
                "canonical expression must contain only tuples and finite scalar atoms"
            )


__all__ = [
    "CanonicalExpression",
    "CostLibrary",
    "HARDWARE_FORMAT",
    "OperationCost",
    "OperationCostEntry",
    "OperationKind",
    "OperationSignature",
    "SignatureOutput",
    "SignatureValueType",
    "WarpgroupHardware",
    "ResourceWindow",
    "WarpgroupCostAmbiguityError",
    "WarpgroupCostError",
    "WarpgroupCostMissingError",
    "WarpgroupMissingSignaturesError",
]
