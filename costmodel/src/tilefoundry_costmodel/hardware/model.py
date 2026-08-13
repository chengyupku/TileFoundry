"""Immutable hardware facts and strict temporal/static resource boundaries."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import TypeVar, cast

from ..constants import HARDWARE_SCHEMA_VERSION
from ..errors import HardwareSpecError, InvalidRequestError
from ..model import DType, HardwareSpecRef, ResourceId, validate_identifier


class FactOrigin(str, Enum):
    """Name the provenance class of one capacity fact."""

    VENDOR = "vendor"
    MEASURED = "measured"
    DERIVED = "derived"
    CONSERVATIVE = "conservative"
    UNAVAILABLE = "unavailable"


class StaticUnit(str, Enum):
    """Name a static per-CTA capacity unit."""

    BYTES = "bytes"
    REGISTERS_32BIT = "registers_32bit"
    WARPS = "warps"
    SLOTS = "slots"


@dataclass(frozen=True, slots=True)
class FactProvenance:
    """Explain the origin and conditions of one schedulable fact."""

    origin: FactOrigin
    source: str
    conditions: str

    def __post_init__(self) -> None:
        _coerce_enum(self, "origin", FactOrigin)
        _non_empty_string(self.source, "provenance source")
        _non_empty_string(self.conditions, "provenance conditions")


@dataclass(frozen=True, slots=True)
class TemporalResourceSpec:
    """Describe one schedulable time-varying resource."""

    resource_id: ResourceId
    capacity_slots: int
    description: str
    provenance: FactProvenance

    def __post_init__(self) -> None:
        _identifier(self.resource_id, "resource_id")
        _positive_int(self.capacity_slots, "capacity_slots")
        _non_empty_string(self.description, "resource description")
        if type(self.provenance) is not FactProvenance:
            raise HardwareSpecError("resource provenance must be FactProvenance")
        if self.provenance.origin is FactOrigin.UNAVAILABLE:
            raise HardwareSpecError("unavailable temporal facts are not schedulable")


@dataclass(frozen=True, slots=True)
class StaticResourceSpec:
    """Describe one positive per-CTA static capacity."""

    resource_id: ResourceId
    capacity_units: int
    unit: StaticUnit
    description: str
    provenance: FactProvenance

    def __post_init__(self) -> None:
        _identifier(self.resource_id, "resource_id")
        _positive_int(self.capacity_units, "capacity_units")
        _coerce_enum(self, "unit", StaticUnit)
        _non_empty_string(self.description, "resource description")
        if type(self.provenance) is not FactProvenance:
            raise HardwareSpecError("resource provenance must be FactProvenance")
        if self.provenance.origin is FactOrigin.UNAVAILABLE:
            raise HardwareSpecError("unavailable static facts are not schedulable")


@dataclass(frozen=True, slots=True)
class HardwareSpec:
    """Complete schedulable hardware document."""

    ref: HardwareSpecRef
    architecture: str
    temporal_resources: tuple[TemporalResourceSpec, ...]
    static_resources: tuple[StaticResourceSpec, ...]
    supported_dtypes: tuple[DType | str, ...]
    supported_implementation_ids: tuple[str, ...]
    schema_version: int = HARDWARE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if type(self.ref) is not HardwareSpecRef:
            raise HardwareSpecError("hardware ref must be HardwareSpecRef")
        _non_empty_string(self.architecture, "architecture")
        if (
            isinstance(self.schema_version, bool)
            or not isinstance(self.schema_version, int)
            or self.schema_version != HARDWARE_SCHEMA_VERSION
        ):
            raise HardwareSpecError(f"unsupported hardware schema version: {self.schema_version!r}")
        if self.ref.schema_version != self.schema_version:
            raise HardwareSpecError("hardware ref and document schema versions must match")

        temporal = _typed_tuple(self.temporal_resources, TemporalResourceSpec, "temporal resources")
        static = _typed_tuple(self.static_resources, StaticResourceSpec, "static resources")
        object.__setattr__(self, "temporal_resources", temporal)
        object.__setattr__(self, "static_resources", static)
        _unique_ids(temporal, "temporal resource IDs")
        _unique_ids(static, "static resource IDs")
        if {str(item.resource_id) for item in temporal}.intersection(
            str(item.resource_id) for item in static
        ):
            raise HardwareSpecError("resource IDs must not overlap temporal and static namespaces")

        if not isinstance(self.supported_dtypes, (tuple, list)):
            raise HardwareSpecError("supported dtypes must be a sequence")
        dtypes: list[DType] = []
        for value in tuple(self.supported_dtypes):
            try:
                dtype = value if isinstance(value, DType) else DType(value)
            except (TypeError, ValueError) as exc:
                raise HardwareSpecError(f"invalid supported dtype: {value!r}") from exc
            dtypes.append(dtype)
        if len(dtypes) != len(set(dtypes)):
            raise HardwareSpecError("supported dtypes must be unique")
        object.__setattr__(self, "supported_dtypes", tuple(dtypes))

        if not isinstance(self.supported_implementation_ids, (tuple, list)):
            raise HardwareSpecError("supported implementation IDs must be a sequence")
        implementations = tuple(self.supported_implementation_ids)
        if not all(isinstance(value, str) for value in implementations):
            raise HardwareSpecError("supported implementation IDs must be strings")
        for value in implementations:
            _identifier(value, "supported implementation ID")
        if len(implementations) != len(set(implementations)):
            raise HardwareSpecError("supported implementation IDs must be unique")
        object.__setattr__(self, "supported_implementation_ids", tuple(sorted(implementations)))

        # Stable canonical resource order makes Python and JSON construction
        # independent of caller enumeration order.
        object.__setattr__(
            self,
            "temporal_resources",
            tuple(sorted(temporal, key=lambda item: str(item.resource_id))),
        )
        object.__setattr__(
            self,
            "static_resources",
            tuple(sorted(static, key=lambda item: str(item.resource_id))),
        )

    def __eq__(self, other: object) -> bool:
        if isinstance(other, HardwareSpec):
            return (
                self.ref == other.ref
                and self.architecture == other.architecture
                and self.temporal_resources == other.temporal_resources
                and self.static_resources == other.static_resources
                and self.supported_dtypes == other.supported_dtypes
                and self.supported_implementation_ids == other.supported_implementation_ids
                and self.schema_version == other.schema_version
            )
        if isinstance(other, Mapping):
            return _json_form(self) == _json_form(other)
        return NotImplemented

    def temporal_capacity(self, resource_id: ResourceId) -> int:
        """Return a temporal capacity and reject static IDs."""

        _identifier(resource_id, "resource_id")
        for resource in self.temporal_resources:
            if resource.resource_id == resource_id:
                return resource.capacity_slots
        if any(resource.resource_id == resource_id for resource in self.static_resources):
            raise HardwareSpecError(f"resource is static, not temporal: {resource_id!r}")
        raise HardwareSpecError(f"unknown temporal resource: {resource_id!r}")

    def static_capacity(self, resource_id: ResourceId) -> int:
        """Return a static capacity and reject temporal IDs."""

        _identifier(resource_id, "resource_id")
        for resource in self.static_resources:
            if resource.resource_id == resource_id:
                return resource.capacity_units
        if any(resource.resource_id == resource_id for resource in self.temporal_resources):
            raise HardwareSpecError(f"resource is temporal, not static: {resource_id!r}")
        raise HardwareSpecError(f"unknown static resource: {resource_id!r}")

    def to_json(self) -> str:
        """Serialize this owned hardware document through the strict codec."""

        from .._serialization import hardware_to_json

        return hardware_to_json(self)

    @classmethod
    def from_json(cls, text: str) -> "HardwareSpec":
        """Decode one strict hardware document."""

        from .._serialization import hardware_from_json

        return hardware_from_json(text)


_EnumT = TypeVar("_EnumT", bound=Enum)
_HardwareRecordT = TypeVar("_HardwareRecordT")


def _typed_tuple(
    value: object, item_type: type[_HardwareRecordT], label: str
) -> tuple[_HardwareRecordT, ...]:
    if not isinstance(value, (tuple, list)):
        raise HardwareSpecError(f"{label} must be a sequence")
    items: tuple[object, ...] = tuple(value)
    if not all(type(item) is item_type for item in items):
        raise HardwareSpecError(f"{label} must contain typed records")
    return cast(tuple[_HardwareRecordT, ...], items)


def _coerce_enum(instance: object, field_name: str, enum_type: type[Enum]) -> None:
    value = getattr(instance, field_name)
    if isinstance(value, enum_type):
        return
    try:
        object.__setattr__(instance, field_name, enum_type(value))
    except (TypeError, ValueError) as exc:
        raise HardwareSpecError(f"invalid {field_name}: {value!r}") from exc


def _coerce_enum_value(value: object, enum_type: type[_EnumT], label: str) -> _EnumT:
    try:
        return value if isinstance(value, enum_type) else enum_type(value)
    except (TypeError, ValueError) as exc:
        raise HardwareSpecError(f"invalid {label}: {value!r}") from exc


def _identifier(value: object, label: str) -> None:
    if not isinstance(value, str):
        raise HardwareSpecError(f"{label} must be a non-empty ASCII string")
    try:
        validate_identifier(value, label=label)
    except InvalidRequestError as exc:
        raise HardwareSpecError(str(exc)) from exc


def _non_empty_string(value: object, label: str) -> None:
    if not isinstance(value, str) or not value:
        raise HardwareSpecError(f"{label} must be a non-empty string")


def _positive_int(value: object, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise HardwareSpecError(f"{label} must be a positive integer")


def _unique_ids(values: tuple[object, ...], label: str) -> None:
    ids = tuple(str(getattr(item, "resource_id")) for item in values)
    if len(ids) != len(set(ids)):
        raise HardwareSpecError(f"{label} must be unique")


def _json_form(value: object) -> object:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _json_form(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_form(item) for item in value]
    if isinstance(value, FactProvenance):
        return {
            "origin": value.origin.value,
            "source": value.source,
            "conditions": value.conditions,
        }
    if isinstance(value, TemporalResourceSpec):
        return {
            "resource_id": str(value.resource_id),
            "capacity_slots": value.capacity_slots,
            "description": value.description,
            "provenance": _json_form(value.provenance),
        }
    if isinstance(value, StaticResourceSpec):
        return {
            "resource_id": str(value.resource_id),
            "capacity_units": value.capacity_units,
            "unit": value.unit.value,
            "description": value.description,
            "provenance": _json_form(value.provenance),
        }
    if isinstance(value, HardwareSpecRef):
        return {
            "hardware_id": value.hardware_id,
            "schema_version": value.schema_version,
            "calibration_id": value.calibration_id,
        }
    if isinstance(value, HardwareSpec):
        return {
            "schema_version": value.schema_version,
            "ref": _json_form(value.ref),
            "architecture": value.architecture,
            "temporal_resources": _json_form(value.temporal_resources),
            "static_resources": _json_form(value.static_resources),
            "supported_dtypes": _json_form(value.supported_dtypes),
            "supported_implementation_ids": _json_form(value.supported_implementation_ids),
        }
    return value


__all__ = [
    "FactOrigin",
    "FactProvenance",
    "HardwareSpec",
    "HardwareSpecRef",
    "StaticResourceSpec",
    "StaticUnit",
    "TemporalResourceSpec",
]
