"""Identifier validation shared by warpgroup records and expressions."""

from __future__ import annotations

import re

from .errors import WarpgroupValidationError

_ID = re.compile(r"[A-Za-z_][A-Za-z0-9_.-]*\Z", re.ASCII)
_SSA_ID = re.compile(r"%[A-Za-z_][A-Za-z0-9_.-]*\Z", re.ASCII)


def validate_id(value: str, label: str) -> None:
    """Require one non-empty portable caller-defined identifier."""
    if type(value) is not str or _ID.fullmatch(value) is None:
        raise WarpgroupValidationError(
            f"{label} must be a non-empty ASCII identifier, got {value!r}"
        )


def validate_ssa_id(value: str, label: str) -> None:
    """Require one portable SSA identifier beginning with percent."""
    if type(value) is not str or _SSA_ID.fullmatch(value) is None:
        raise WarpgroupValidationError(
            f"{label} must be an ASCII SSA identifier beginning with '%', got {value!r}"
        )


def positive_int(value: int, label: str) -> None:
    """Require a positive integer without accepting booleans."""
    if type(value) is not int or value <= 0:
        raise WarpgroupValidationError(f"{label} must be a positive integer, got {value!r}")


def non_negative_int(value: int, label: str) -> None:
    """Require a non-negative integer without accepting booleans."""
    if type(value) is not int or value < 0:
        raise WarpgroupValidationError(f"{label} must be a non-negative integer, got {value!r}")
