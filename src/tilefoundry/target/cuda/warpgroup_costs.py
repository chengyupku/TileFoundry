"""Exact B200 warpgroup calibration coverage without fallback timings."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from tilefoundry.schedule.warpgroup.cost import (
    CanonicalExpression,
    OperationKind,
    OperationSignature,
    SignatureOutput,
    SignatureValueType,
    WarpgroupCostError,
)
from tilefoundry.schedule.warpgroup.model import (
    DType,
    MemorySpace,
    ResourceDemand,
)


class B200OperationFamily(str, Enum):
    """Name one implementation-independent calibration family."""

    GLOBAL_TO_SHARED_COPY = "global_to_shared_copy"
    SHARED_TO_SHARED_COMPUTE = "shared_to_shared_compute"
    REGISTER_SHARED_LOCAL_COMPUTE = "register_shared_local_compute"
    SHARED_SHARED_REMOTE_COMPUTE = "shared_shared_remote_compute"
    FUSED_REDUCTION_ELEMENTWISE_COMPUTE = "fused_reduction_elementwise_compute"
    REGISTER_RESCALE = "register_rescale"
    REGISTER_TO_SHARED_PUBLICATION = "register_to_shared_publication"


class CalibrationStatus(str, Enum):
    """State whether an exact signature has executable and measured coverage."""

    MISSING = "missing"
    PROVIDER_READY = "provider_ready"
    MEASURED = "measured"


class B200CalibrationMissingError(WarpgroupCostError):
    """An exact B200 signature has no correctness-checked measurement."""

    def __init__(self, signature: OperationSignature) -> None:
        self.signature = signature
        super().__init__(
            "missing correctness-checked B200 calibration for operation signature "
            f"{signature.canonical_key}"
        )


@dataclass(frozen=True, slots=True)
class B200CalibrationCondition:
    """One exact implementation condition required by a coverage row."""

    name: str
    value: str

    def __post_init__(self) -> None:
        if not all(
            type(item) is str and item and item.isascii() for item in (self.name, self.value)
        ):
            raise TypeError("calibration condition must contain non-empty ASCII text")


@dataclass(frozen=True, slots=True)
class B200CoverageEntry:
    """One exact signature and its resource/implementation measurement state."""

    family: B200OperationFamily
    signature: OperationSignature
    implementation_id: str
    conditions: tuple[B200CalibrationCondition, ...]
    resources: tuple[ResourceDemand, ...]
    status: CalibrationStatus

    def __post_init__(self) -> None:
        if type(self.family) is not B200OperationFamily:
            raise TypeError("coverage family must be B200OperationFamily")
        if type(self.signature) is not OperationSignature:
            raise TypeError("coverage signature must be OperationSignature")
        if type(self.implementation_id) is not str or not self.implementation_id.isascii():
            raise TypeError("coverage implementation ID must be ASCII text")
        if not self.implementation_id:
            raise ValueError("coverage implementation ID must be non-empty")
        if type(self.conditions) is not tuple or not all(
            type(item) is B200CalibrationCondition for item in self.conditions
        ):
            raise TypeError("coverage conditions must be exact B200CalibrationCondition records")
        condition_names = tuple(item.name for item in self.conditions)
        if len(condition_names) != len(set(condition_names)):
            raise ValueError("coverage entry contains duplicate calibration conditions")
        if type(self.resources) is not tuple or not all(
            type(item) is ResourceDemand for item in self.resources
        ):
            raise TypeError("coverage resources must be exact ResourceDemand records")
        if type(self.status) is not CalibrationStatus:
            raise TypeError("coverage status must be CalibrationStatus")
        ids = tuple(item.resource_id for item in self.resources)
        if len(ids) != len(set(ids)):
            raise ValueError("coverage entry contains duplicate resource demands")
        object.__setattr__(
            self, "resources", tuple(sorted(self.resources, key=lambda x: x.resource_id))
        )
        object.__setattr__(self, "conditions", tuple(sorted(self.conditions, key=lambda x: x.name)))


@dataclass(frozen=True, slots=True)
class B200CoverageMatrix:
    """Canonical exact-signature coverage for all required operation families."""

    entries: tuple[B200CoverageEntry, ...]

    def __post_init__(self) -> None:
        if type(self.entries) is not tuple or not all(
            type(item) is B200CoverageEntry for item in self.entries
        ):
            raise TypeError("coverage matrix must contain exact B200CoverageEntry records")
        signatures = tuple(item.signature for item in self.entries)
        if len(signatures) != len(set(signatures)):
            raise ValueError("coverage matrix contains duplicate operation signatures")
        object.__setattr__(
            self,
            "entries",
            tuple(sorted(self.entries, key=lambda item: item.signature.canonical_key)),
        )

    def lookup(self, signature: OperationSignature) -> B200CoverageEntry:
        """Return the one exact coverage row, with no nearby-signature match."""
        if type(signature) is not OperationSignature:
            raise TypeError("coverage lookup requires an exact OperationSignature")
        for entry in self.entries:
            if entry.signature == signature:
                return entry
        raise B200CalibrationMissingError(signature)

    def require_measured(self, signature: OperationSignature) -> B200CoverageEntry:
        """Return an exact measured row or fail without a synthetic duration."""
        entry = self.lookup(signature)
        if entry.status is not CalibrationStatus.MEASURED:
            raise B200CalibrationMissingError(signature)
        return entry


def classify_b200_operation_signature(signature: OperationSignature) -> B200OperationFamily:
    """Classify one exact signature using only expression and value semantics."""
    if type(signature) is not OperationSignature:
        raise TypeError("B200 coverage classification requires an OperationSignature")
    operand_spaces = tuple(item.space for item in signature.operands)
    output_spaces = tuple(item.type.space for item in signature.outputs)
    if signature.kind is OperationKind.COPY:
        if operand_spaces == (MemorySpace.GLOBAL,) and output_spaces == (MemorySpace.SHARED,):
            return B200OperationFamily.GLOBAL_TO_SHARED_COPY
        if operand_spaces == (MemorySpace.REGISTER,) and output_spaces == (MemorySpace.SHARED,):
            return B200OperationFamily.REGISTER_TO_SHARED_PUBLICATION
        raise B200CalibrationMissingError(signature)
    if signature.kind is not OperationKind.COMPUTE:
        raise B200CalibrationMissingError(signature)

    expressions = tuple(item.expression for item in signature.outputs)
    if any(
        _contains_operator(expression, {"exp", "reduce", "select"}) for expression in expressions
    ):
        return B200OperationFamily.FUSED_REDUCTION_ELEMENTWISE_COMPUTE
    matmuls = tuple(
        node for expression in expressions for node in _operator_nodes(expression, "matmul")
    )
    if matmuls:
        if all(expression and expression[0] == "matmul" for expression in expressions) and set(
            operand_spaces
        ) == {MemorySpace.SHARED}:
            return B200OperationFamily.SHARED_TO_SHARED_COMPUTE
        matmul_spaces = {
            signature.operands[index].space
            for matmul in matmuls
            for index in _reference_indices(matmul)
            if index < len(signature.operands)
        }
        if MemorySpace.REGISTER in matmul_spaces and MemorySpace.SHARED in matmul_spaces:
            return B200OperationFamily.REGISTER_SHARED_LOCAL_COMPUTE
        if matmul_spaces == {MemorySpace.SHARED}:
            return B200OperationFamily.SHARED_SHARED_REMOTE_COMPUTE
        raise B200CalibrationMissingError(signature)
    if (
        operand_spaces
        and set(operand_spaces) == {MemorySpace.REGISTER}
        and any(_contains_operator(expression, {"mul"}) for expression in expressions)
    ):
        return B200OperationFamily.REGISTER_RESCALE
    return B200OperationFamily.FUSED_REDUCTION_ELEMENTWISE_COMPUTE


def _operator_nodes(
    expression: CanonicalExpression, operator: str
) -> tuple[CanonicalExpression, ...]:
    nested = tuple(
        found
        for item in expression
        if type(item) is tuple
        for found in _operator_nodes(item, operator)
    )
    return ((expression,) if expression and expression[0] == operator else ()) + nested


def _contains_operator(expression: CanonicalExpression, operators: set[str]) -> bool:
    return bool(expression and expression[0] in operators) or any(
        _contains_operator(item, operators) for item in expression if type(item) is tuple
    )


def _reference_indices(expression: CanonicalExpression) -> tuple[int, ...]:
    if len(expression) == 2 and expression[0] == "ref" and type(expression[1]) is int:
        return (expression[1],)
    return tuple(
        index for item in expression if type(item) is tuple for index in _reference_indices(item)
    )


def _family_policy(
    family: B200OperationFamily,
) -> tuple[
    str,
    tuple[B200CalibrationCondition, ...],
    tuple[ResourceDemand, ...],
    CalibrationStatus,
]:
    policies = {
        B200OperationFamily.GLOBAL_TO_SHARED_COPY: (
            "b200.cuda.global_to_shared_copy",
            ("b200.cuda_core", "b200.gmem_read", "b200.smem_write", "b200.warp_issue"),
            CalibrationStatus.PROVIDER_READY,
        ),
        B200OperationFamily.SHARED_TO_SHARED_COMPUTE: (
            "b200.tensor.shared_compute",
            ("b200.smem_read", "b200.tensor_core", "b200.warp_issue"),
            CalibrationStatus.MISSING,
        ),
        B200OperationFamily.REGISTER_SHARED_LOCAL_COMPUTE: (
            "b200.tensor.register_shared_local_compute",
            ("b200.rf_read", "b200.smem_read", "b200.tensor_core", "b200.warp_issue"),
            CalibrationStatus.MISSING,
        ),
        B200OperationFamily.SHARED_SHARED_REMOTE_COMPUTE: (
            "b200.tensor.shared_shared_remote_compute",
            ("b200.rf_read", "b200.smem_read", "b200.tensor_core", "b200.warp_issue"),
            CalibrationStatus.MISSING,
        ),
        B200OperationFamily.FUSED_REDUCTION_ELEMENTWISE_COMPUTE: (
            "b200.cuda.fused_reduction_elementwise",
            ("b200.cuda_core", "b200.rf_read", "b200.rf_write", "b200.warp_issue"),
            CalibrationStatus.MISSING,
        ),
        B200OperationFamily.REGISTER_RESCALE: (
            "b200.cuda.register_rescale",
            ("b200.cuda_core", "b200.rf_read", "b200.rf_write", "b200.warp_issue"),
            CalibrationStatus.MISSING,
        ),
        B200OperationFamily.REGISTER_TO_SHARED_PUBLICATION: (
            "b200.cuda.register_to_shared_publication",
            ("b200.rf_read", "b200.smem_write", "b200.warp_issue"),
            CalibrationStatus.MISSING,
        ),
    }
    implementation_id, resources, status = policies[family]
    conditions = (
        B200CalibrationCondition("cuda_arch", "sm_100a"),
        B200CalibrationCondition("hardware", "b200"),
        B200CalibrationCondition("pipeline_depth", "1"),
    )
    return (
        implementation_id,
        conditions,
        tuple(ResourceDemand(item, 1) for item in resources),
        status,
    )


def _coverage_entry(signature: OperationSignature) -> B200CoverageEntry:
    family = classify_b200_operation_signature(signature)
    implementation_id, conditions, resources, status = _family_policy(family)
    if family is B200OperationFamily.GLOBAL_TO_SHARED_COPY and not _supports_copy_provider(
        signature
    ):
        status = CalibrationStatus.MISSING
    return B200CoverageEntry(family, signature, implementation_id, conditions, resources, status)


def _supports_copy_provider(signature: OperationSignature) -> bool:
    if len(signature.operands) != 1 or len(signature.outputs) != 1:
        return False
    source = signature.operands[0]
    output = signature.outputs[0]
    return (
        source.shape == (64, 64)
        and source.dtype is DType.BF16
        and source.space is MemorySpace.GLOBAL
        and output.type.shape == source.shape
        and output.type.dtype is source.dtype
        and output.type.space is MemorySpace.SHARED
        and output.expression == ("copy", ("ref", 0))
    )


def b200_global_to_shared_copy_signature() -> OperationSignature:
    """Return the one exact M5.2 signature supported by the initial provider."""
    source = SignatureValueType((64, 64), DType.BF16, MemorySpace.GLOBAL)
    result = SignatureValueType((64, 64), DType.BF16, MemorySpace.SHARED)
    return OperationSignature(
        OperationKind.COPY,
        (source,),
        (SignatureOutput(result, ("copy", ("ref", 0))),),
    )


def b200_warpgroup_coverage_matrix(
    signatures: tuple[OperationSignature, ...],
) -> B200CoverageMatrix:
    """Map exact signatures to all generic M5 families and measurement states."""
    if type(signatures) is not tuple or not all(
        type(item) is OperationSignature for item in signatures
    ):
        raise TypeError("B200 coverage signatures must be exact OperationSignature records")
    return B200CoverageMatrix(tuple(_coverage_entry(item) for item in signatures))


__all__ = [
    "B200CalibrationMissingError",
    "B200CalibrationCondition",
    "B200CoverageEntry",
    "B200CoverageMatrix",
    "B200OperationFamily",
    "CalibrationStatus",
    "b200_global_to_shared_copy_signature",
    "b200_warpgroup_coverage_matrix",
    "classify_b200_operation_signature",
]
