"""Small immutable JSON value helpers used by deferred schema documents."""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping
from types import MappingProxyType
from typing import TypeAlias, Union

FrozenJson: TypeAlias = Union[
    None,
    bool,
    int,
    float,
    str,
    tuple["FrozenJson", ...],
    "FrozenJsonObject",
]


class FrozenJsonObject(Mapping[str, FrozenJson]):
    """Recursively immutable JSON object.

    This is deliberately a mapping rather than a mutable ``dict`` so deferred
    M0 documents can retain their JSON shape without leaking caller-owned
    containers into a replayable problem or profile snapshot.
    """

    __slots__ = ("_values",)
    _values: Mapping[str, FrozenJson]

    def __init__(self, values: Mapping[str, object]) -> None:
        converted: dict[str, FrozenJson] = {}
        for key, value in values.items():
            if not isinstance(key, str):
                raise TypeError("JSON object keys must be strings")
            converted[key] = freeze_json(value)
        object.__setattr__(self, "_values", MappingProxyType(converted))

    def __setattr__(self, name: str, value: object) -> None:
        del value
        raise AttributeError(f"{type(self).__name__} is immutable")

    def __getitem__(self, key: str) -> FrozenJson:
        return self._values[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def __repr__(self) -> str:
        return f"FrozenJsonObject({dict(self._values)!r})"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Mapping):
            return _normal_form(self) == _normal_form(other)
        return NotImplemented


def freeze_json(value: object) -> FrozenJson:
    """Copy one JSON-compatible value into recursively immutable containers."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite floating point value")
        return value
    if isinstance(value, FrozenJsonObject):
        return value
    if isinstance(value, Mapping):
        return FrozenJsonObject(value)
    if isinstance(value, (list, tuple)):
        return tuple(freeze_json(item) for item in value)
    raise TypeError(f"unsupported JSON value type {type(value).__name__}")


def _normal_form(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _normal_form(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_normal_form(item) for item in value]
    return value


__all__ = ["FrozenJson", "FrozenJsonObject", "freeze_json"]
