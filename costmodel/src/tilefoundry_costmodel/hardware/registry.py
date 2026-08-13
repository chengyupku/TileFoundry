"""Exact immutable hardware catalog resolution."""

from __future__ import annotations

from dataclasses import dataclass

from ..errors import HardwareSpecError, UnsupportedError
from ..model import HardwareSpecRef
from .b200 import b200_hardware_spec
from .model import HardwareSpec


@dataclass(frozen=True, slots=True)
class HardwareCatalog:
    """Resolve only exact ``(hardware_id, schema_version, calibration_id)`` refs."""

    specs: tuple[HardwareSpec, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.specs, (tuple, list)):
            raise HardwareSpecError("hardware catalog specs must be a sequence")
        specs = tuple(self.specs)
        if not all(type(spec) is HardwareSpec for spec in specs):
            raise HardwareSpecError("hardware catalog must contain HardwareSpec records")
        refs = tuple(spec.ref for spec in specs)
        if len(refs) != len(set(refs)):
            raise HardwareSpecError("hardware catalog references must be unique")
        object.__setattr__(self, "specs", tuple(sorted(specs, key=lambda spec: _ref_key(spec.ref))))

    def resolve(self, ref: HardwareSpecRef) -> HardwareSpec:
        """Return an exact match; nearby names/calibrations are not aliases."""

        if type(ref) is not HardwareSpecRef:
            raise HardwareSpecError("hardware lookup requires HardwareSpecRef")
        for spec in self.specs:
            if spec.ref == ref:
                return spec
        raise UnsupportedError(f"hardware reference is not installed: {ref!r}")

    @classmethod
    def from_json(cls, text: str) -> "HardwareCatalog":
        """Build a one-document catalog from strict hardware JSON."""

        from .._serialization import hardware_from_json

        return cls((hardware_from_json(text),))


def _ref_key(ref: HardwareSpecRef) -> tuple[str, int, str]:
    return (ref.hardware_id, ref.schema_version, ref.calibration_id)


def b200_hardware_catalog() -> HardwareCatalog:
    """Return the immutable M1 B200 catalog."""

    return HardwareCatalog((b200_hardware_spec(),))


__all__ = ["HardwareCatalog", "b200_hardware_catalog"]
