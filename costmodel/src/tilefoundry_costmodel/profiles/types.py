"""Shared immutable profile-environment records."""

from __future__ import annotations

from dataclasses import dataclass

from ..errors import InvalidRequestError, ProfileStoreError
from ..hardware.model import HardwareSpecRef
from ..model import validate_identifier


@dataclass(frozen=True, slots=True)
class ProfileEnvironment:
    """Describe the exact CUDA measurement environment."""

    environment_id: str
    device_uuid: str
    hardware: HardwareSpecRef
    cuda_arch: str
    driver_version: str
    runtime_version: str
    nvrtc_version: str
    device_clock_khz: int | None
    memory_clock_khz: int | None
    power_limit_mw: int | None

    def __post_init__(self) -> None:
        for value, label in (
            (self.environment_id, "environment_id"),
            (self.device_uuid, "device_uuid"),
            (self.cuda_arch, "cuda_arch"),
            (self.driver_version, "driver_version"),
            (self.runtime_version, "runtime_version"),
            (self.nvrtc_version, "nvrtc_version"),
        ):
            _identifier(value, label)
        if type(self.hardware) is not HardwareSpecRef:
            raise ProfileStoreError("profile environment hardware must be HardwareSpecRef")
        for quantity, label in (
            (self.device_clock_khz, "device_clock_khz"),
            (self.memory_clock_khz, "memory_clock_khz"),
            (self.power_limit_mw, "power_limit_mw"),
        ):
            if quantity is not None and (
                isinstance(quantity, bool) or not isinstance(quantity, int) or quantity <= 0
            ):
                raise ProfileStoreError(f"{label} must be positive or null")


def _identifier(value: object, label: str) -> None:
    try:
        validate_identifier(value, label=label)  # type: ignore[arg-type]
    except InvalidRequestError as exc:
        raise ProfileStoreError(str(exc)) from exc


__all__ = ["ProfileEnvironment"]
