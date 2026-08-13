"""Pure ``T.*`` constructors for the standalone typed-program boundary.

The module is intentionally data-only.  Every helper fixes its discriminator
and delegates all semantic checks to the immutable records in ``program.py``.
"""

from __future__ import annotations

from .model import LoopId, OpId, ProgramId, ValueId, WorkloadKind
from .program import (
    AlignedRelation,
    CopyOp,
    DependencyRelation,
    DependencyRelationKind,
    ElementwiseKind,
    ElementwiseOp,
    EndpointRelation,
    GemmOp,
    InstanceEndpoint,
    LoopBarrier,
    OpIterationDomain,
    ReduceOp,
    ReductionKind,
    TileCandidate,
    TileDependency,
    TileLoop,
    TileOp,
    TileOpKind,
    TileProgram,
    TileValue,
    TileValueType,
)


def value(*, value_id: ValueId, value_type: TileValueType) -> TileValue:
    """Construct one typed program value."""

    return TileValue(value_id, value_type)


def pipeline(*, loop_id: LoopId, iterations: int) -> TileLoop:
    """Construct one repeated pipeline region."""

    return TileLoop(loop_id, iterations)


def copy(
    *,
    op_id: OpId,
    source: ValueId,
    destination: ValueId,
    domain: OpIterationDomain,
) -> CopyOp:
    """Construct one copy operation with a fixed ``copy`` discriminator."""

    return CopyOp(TileOpKind.COPY, op_id, source, destination, domain)


def gemm(
    *,
    op_id: OpId,
    lhs: ValueId,
    rhs: ValueId,
    accumulator: ValueId,
    result: ValueId,
    m_axis: str,
    n_axis: str,
    k_axis: str,
    domain: OpIterationDomain,
) -> GemmOp:
    """Construct one GEMM operation with a fixed ``gemm`` discriminator."""

    return GemmOp(
        TileOpKind.GEMM,
        op_id,
        lhs,
        rhs,
        accumulator,
        result,
        m_axis,
        n_axis,
        k_axis,
        domain,
    )


def reduce(
    *,
    op_id: OpId,
    source: ValueId,
    result: ValueId,
    axes: tuple[str, ...],
    reduction: ReductionKind,
    domain: OpIterationDomain,
) -> ReduceOp:
    """Construct one reduction operation."""

    return ReduceOp(TileOpKind.REDUCE, op_id, source, result, axes, reduction, domain)


def elementwise(
    *,
    op_id: OpId,
    inputs: tuple[ValueId, ...],
    result: ValueId,
    function: ElementwiseKind,
    domain: OpIterationDomain,
) -> ElementwiseOp:
    """Construct one elementwise operation."""

    return ElementwiseOp(TileOpKind.ELEMENTWISE, op_id, inputs, result, function, domain)


def aligned(*, iteration_distance: int = 0) -> AlignedRelation:
    """Construct a corresponding-instance relation."""

    return AlignedRelation(DependencyRelationKind.ALIGNED, iteration_distance)


def endpoint(
    *,
    src_endpoint: InstanceEndpoint,
    dst_endpoint: InstanceEndpoint,
) -> EndpointRelation:
    """Construct an exact single-instance endpoint relation."""

    return EndpointRelation(
        DependencyRelationKind.ENDPOINT,
        src_endpoint,
        dst_endpoint,
    )


def depends(
    *,
    value_id: ValueId,
    src_op_id: OpId,
    dst_op_id: OpId,
    relation: DependencyRelation,
) -> TileDependency:
    """Construct one explicit value dependency."""

    return TileDependency(value_id, src_op_id, dst_op_id, relation)


def program(
    *,
    schema_version: int,
    program_id: ProgramId,
    workload_kind: WorkloadKind,
    tile: TileCandidate,
    values: tuple[TileValue, ...],
    loops: tuple[TileLoop, ...],
    operations: tuple[TileOp, ...],
    dependencies: tuple[TileDependency, ...],
    loop_barriers: tuple[LoopBarrier, ...],
    inputs: tuple[ValueId, ...],
    outputs: tuple[ValueId, ...],
) -> TileProgram:
    """Construct and fully validate one concrete typed program."""

    return TileProgram(
        schema_version,
        program_id,
        workload_kind,
        tile,
        values,
        loops,
        operations,
        dependencies,
        loop_barriers,
        inputs,
        outputs,
    )


__all__ = [
    "aligned",
    "copy",
    "depends",
    "elementwise",
    "endpoint",
    "gemm",
    "pipeline",
    "program",
    "reduce",
    "value",
]
