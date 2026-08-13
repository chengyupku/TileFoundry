"""Reachable M1 typed-program workflows and semantic rejection cases."""

from __future__ import annotations

import json

import pytest

import tilefoundry_costmodel as cm
from tilefoundry_costmodel import T
from tilefoundry_costmodel.model import (
    AxisExtent,
    DType,
    NamedShape,
    TensorDescriptor,
    TensorLayout,
    WorkloadKind,
)
from tilefoundry_costmodel.program import (
    CopyOp,
    DependencyRelationKind,
    ElementwiseKind,
    ElementwiseOp,
    EndpointRelation,
    InstanceEndpoint,
    LoopBarrier,
    MemorySpace,
    OpIterationDomain,
    ReductionKind,
    TileCandidate,
    TileDependency,
    TileLoop,
    TileOpKind,
    TileProgram,
    TileValue,
    TileValueType,
)
from tilefoundry_costmodel.workloads import (
    WorkloadFrontendCatalog,
    builtin_workload_frontends,
)


def _shape(*axes: tuple[str, int]) -> NamedShape:
    return NamedShape(tuple(AxisExtent(name, extent) for name, extent in axes))


def _value(value_id: str, shape: NamedShape, dtype: DType = DType.BF16) -> TileValue:
    return TileValue(
        value_id,
        TileValueType(
            TensorDescriptor(shape, dtype, TensorLayout.ROW_MAJOR),
            MemorySpace.REGISTER,
        ),
    )


def _base_program(
    *,
    values: tuple[TileValue, ...],
    loops: tuple[TileLoop, ...] = (),
    operations: tuple[CopyOp, ...] = (),
    dependencies: tuple[TileDependency, ...] = (),
    barriers: tuple[LoopBarrier, ...] = (),
    inputs: tuple[str, ...] = (),
    outputs: tuple[str, ...] = (),
) -> TileProgram:
    return TileProgram(
        2,
        "program",
        WorkloadKind.GEMM,
        TileCandidate("tile", _shape(("m", 2), ("n", 2), ("k", 2))),
        values,
        loops,
        operations,
        dependencies,
        barriers,
        inputs,
        outputs,
    )


def test_python_and_json_golden_reference_are_byte_identical() -> None:
    shape = _shape(("m", 2), ("n", 2))
    program = _base_program(
        values=(_value("input", shape), _value("output", shape)),
        operations=(
            T.copy(
                op_id="copy",
                source="input",
                destination="output",
                domain=OpIterationDomain(None, 0, 1),
            ),
        ),
        inputs=("input",),
        outputs=("output",),
    )
    encoded = cm.program_to_json(program)
    decoded = cm.program_from_json(encoded)
    assert encoded == cm.program_to_json(decoded)
    assert json.loads(encoded)["operations"][0]["kind"] == "copy"
    wrong_discriminator = json.loads(encoded)
    wrong_discriminator["operations"][0]["kind"] = "unknown"
    with pytest.raises(cm.WorkloadError, match="operation kind"):
        cm.program_from_json(json.dumps(wrong_discriminator))


def test_aligned_recurrence_expands_corresponding_instances() -> None:
    shape = _shape(("m", 2), ("n", 2))
    loop = T.pipeline(loop_id="loop", iterations=4)
    load = T.copy(
        op_id="load",
        source="input",
        destination="buffer",
        domain=OpIterationDomain("loop", 0, 4),
    )
    use = T.copy(
        op_id="use",
        source="buffer",
        destination="output",
        domain=OpIterationDomain("loop", 0, 4),
    )
    program = _base_program(
        values=(_value("input", shape), _value("buffer", shape), _value("output", shape)),
        loops=(loop,),
        operations=(load, use),
        dependencies=(
            T.depends(value_id="buffer", src_op_id="load", dst_op_id="use", relation=T.aligned()),
        ),
        inputs=("input",),
        outputs=("output",),
    )
    assert len(program.expanded_dependencies) == 4
    assert all(edge.src_iteration == edge.dst_iteration for edge in program.expanded_dependencies)


def test_positive_distance_aligned_self_recurrence_is_finite() -> None:
    state_shape = _shape(("m", 2), ("n", 2))
    step = T.elementwise(
        op_id="step",
        inputs=("state",),
        result="state",
        function=ElementwiseKind.RELU,
        domain=OpIterationDomain("loop", 0, 4),
    )
    program = TileProgram(
        2,
        "recurrence_program",
        WorkloadKind.GEMM,
        TileCandidate("tile", _shape(("m", 2), ("n", 2), ("k", 1))),
        (_value("state", state_shape),),
        (TileLoop("loop", 4),),
        (step,),
        (
            T.depends(
                value_id="state",
                src_op_id="step",
                dst_op_id="step",
                relation=T.aligned(iteration_distance=1),
            ),
        ),
        (),
        (),
        ("state",),
    )
    assert [(edge.src_iteration, edge.dst_iteration) for edge in program.expanded_dependencies] == [
        (0, 1),
        (1, 2),
        (2, 3),
    ]


def test_endpoint_relation_crosses_loop_boundaries_with_one_edge() -> None:
    shape = _shape(("m", 2), ("n", 2))
    first = T.copy(
        op_id="first_op",
        source="input",
        destination="middle",
        domain=OpIterationDomain("first_loop", 0, 3),
    )
    last = T.copy(
        op_id="last_op",
        source="middle",
        destination="output",
        domain=OpIterationDomain("second_loop", 0, 2),
    )
    program = _base_program(
        values=(_value("input", shape), _value("middle", shape), _value("output", shape)),
        loops=(TileLoop("first_loop", 3), TileLoop("second_loop", 2)),
        operations=(first, last),
        dependencies=(
            TileDependency(
                "middle",
                "first_op",
                "last_op",
                EndpointRelation(
                    DependencyRelationKind.ENDPOINT,
                    InstanceEndpoint.LAST,
                    InstanceEndpoint.FIRST,
                ),
            ),
        ),
        inputs=("input",),
        outputs=("output",),
    )
    assert len(program.expanded_dependencies) == 1
    edge = program.expanded_dependencies[0]
    assert (edge.src_iteration, edge.dst_iteration) == (2, 0)
    encoded = cm.program_to_json(program)
    decoded = cm.program_from_json(encoded)
    assert cm.program_to_json(decoded) == encoded
    assert isinstance(decoded.dependencies[0].relation, EndpointRelation)


def test_relation_level_cycle_is_rejected_even_when_instance_edges_are_acyclic() -> None:
    shape = _shape(("m", 2), ("n", 2))
    left = T.copy(
        op_id="left",
        source="right_value",
        destination="left_value",
        domain=OpIterationDomain("left_loop", 0, 2),
    )
    right = T.copy(
        op_id="right",
        source="left_value",
        destination="right_value",
        domain=OpIterationDomain("right_loop", 0, 2),
    )
    with pytest.raises(cm.WorkloadError, match="relation graph"):
        _base_program(
            values=(_value("left_value", shape), _value("right_value", shape)),
            loops=(TileLoop("left_loop", 2), TileLoop("right_loop", 2)),
            operations=(left, right),
            dependencies=(
                TileDependency(
                    "left_value",
                    "left",
                    "right",
                    T.endpoint(
                        src_endpoint=InstanceEndpoint.LAST,
                        dst_endpoint=InstanceEndpoint.FIRST,
                    ),
                ),
                TileDependency(
                    "right_value",
                    "right",
                    "left",
                    T.endpoint(
                        src_endpoint=InstanceEndpoint.LAST,
                        dst_endpoint=InstanceEndpoint.FIRST,
                    ),
                ),
            ),
            outputs=("left_value", "right_value"),
        )


def test_explicit_loop_barrier_is_not_replaced_by_operation_order() -> None:
    shape = _shape(("m", 2), ("n", 2))
    left = T.copy(
        op_id="left_op",
        source="left_input",
        destination="left_output",
        domain=OpIterationDomain("left_loop", 0, 2),
    )
    right = T.copy(
        op_id="right_op",
        source="right_input",
        destination="right_output",
        domain=OpIterationDomain("right_loop", 0, 2),
    )
    program = _base_program(
        values=tuple(
            _value(name, shape)
            for name in ("left_input", "left_output", "right_input", "right_output")
        ),
        loops=(TileLoop("left_loop", 2), TileLoop("right_loop", 2)),
        operations=(left, right),
        barriers=(LoopBarrier("barrier", "left_loop", "right_loop"),),
        inputs=("left_input", "right_input"),
        outputs=("left_output", "right_output"),
    )
    assert program.loop_barriers[0].src_loop_id == "left_loop"
    assert program.expanded_dependencies == ()
    with pytest.raises(cm.WorkloadError, match="barrier graph"):
        _base_program(
            values=program.values,
            loops=program.loops,
            operations=program.operations,
            barriers=(
                LoopBarrier("barrier_a", "left_loop", "right_loop"),
                LoopBarrier("barrier_b", "right_loop", "left_loop"),
            ),
            inputs=program.inputs,
            outputs=program.outputs,
        )


def test_program_rejects_missing_dependency_and_external_producer() -> None:
    shape = _shape(("m", 2), ("n", 2))
    producer = T.copy(
        op_id="producer",
        source="input",
        destination="middle",
        domain=OpIterationDomain(None, 0, 1),
    )
    consumer = T.copy(
        op_id="consumer",
        source="middle",
        destination="output",
        domain=OpIterationDomain(None, 0, 1),
    )
    kwargs = {
        "values": tuple(_value(name, shape) for name in ("input", "middle", "output")),
        "operations": (producer, consumer),
        "inputs": ("input",),
        "outputs": ("output",),
    }
    with pytest.raises(cm.WorkloadError, match="explicit dependency"):
        _base_program(**kwargs)
    with pytest.raises(cm.WorkloadError, match="external input"):
        _base_program(
            **{**kwargs, "inputs": ("input", "middle")},
            dependencies=(
                TileDependency(
                    "middle",
                    "producer",
                    "consumer",
                    T.endpoint(
                        src_endpoint=InstanceEndpoint.FIRST, dst_endpoint=InstanceEndpoint.FIRST
                    ),
                ),
            ),
        )


def test_program_rejects_invalid_domain_and_broadcasting() -> None:
    shape = _shape(("m", 2), ("n", 2))
    with pytest.raises(cm.WorkloadError, match="exceeds"):
        _base_program(
            values=(_value("input", shape), _value("output", shape)),
            loops=(TileLoop("loop", 2),),
            operations=(
                T.copy(
                    op_id="copy",
                    source="input",
                    destination="output",
                    domain=OpIterationDomain("loop", 1, 2),
                ),
            ),
            inputs=("input",),
            outputs=("output",),
        )

    scalar = _value("scalar", _shape(("m", 1), ("n", 1)))
    matrix = _value("matrix", shape)
    bad_result = _value("bad_result", _shape(("m", 3), ("n", 2)))
    with pytest.raises(cm.WorkloadError, match="broadcasting"):
        _base_program(
            values=(scalar, matrix, bad_result),
            operations=(
                ElementwiseOp(
                    TileOpKind.ELEMENTWISE,
                    "ew",
                    ("scalar", "matrix"),
                    "bad_result",
                    ElementwiseKind.ADD,
                    OpIterationDomain(None, 0, 1),
                ),
            ),
            inputs=("scalar", "matrix"),
            outputs=("bad_result",),
        )


def test_gemm_and_reduction_axes_are_validated_before_lowering() -> None:
    lhs = _value("lhs", _shape(("m", 2), ("k", 4)))
    rhs = _value("rhs", _shape(("k", 4), ("n", 3)))
    accumulator = _value("accumulator", _shape(("m", 2), ("n", 3)), DType.FP32)
    result = _value("result", _shape(("m", 2), ("n", 3)), DType.FP32)
    operation = T.gemm(
        op_id="gemm",
        lhs="lhs",
        rhs="rhs",
        accumulator="accumulator",
        result="result",
        m_axis="m",
        n_axis="n",
        k_axis="k",
        domain=OpIterationDomain(None, 0, 1),
    )
    program = _base_program(
        values=(lhs, rhs, accumulator, result),
        operations=(operation,),  # type: ignore[arg-type]
        inputs=("lhs", "rhs", "accumulator"),
        outputs=("result",),
    )
    assert json.loads(cm.program_to_json(program))["operations"][0]["kind"] == "gemm"

    bad_rhs = _value("rhs", _shape(("k", 5), ("n", 3)))
    with pytest.raises(cm.WorkloadError, match="GEMM k axis extents"):
        _base_program(
            values=(lhs, bad_rhs, accumulator, result),
            operations=(operation,),  # type: ignore[arg-type]
            inputs=("lhs", "rhs", "accumulator"),
            outputs=("result",),
        )

    reduction_result = _value("reduction_result", _shape(("m", 2)))
    with pytest.raises(cm.WorkloadError, match="reduction axis"):
        _base_program(
            values=(lhs, reduction_result),
            operations=(
                T.reduce(
                    op_id="reduce",
                    source="lhs",
                    result="reduction_result",
                    axes=("absent",),
                    reduction=ReductionKind.SUM,
                    domain=OpIterationDomain(None, 0, 1),
                ),
            ),  # type: ignore[arg-type]
            inputs=("lhs",),
            outputs=("reduction_result",),
        )
    with pytest.raises(cm.WorkloadError, match="axes must be a sequence"):
        T.reduce(
            op_id="reduce",
            source="lhs",
            result="reduction_result",
            axes=None,  # type: ignore[arg-type]
            reduction=ReductionKind.SUM,
            domain=OpIterationDomain(None, 0, 1),
        )


def test_workload_frontend_catalog_is_exact_and_deterministic() -> None:
    catalog = builtin_workload_frontends()
    assert tuple(frontend.workload_kind for frontend in catalog.frontends) == tuple(
        sorted(WorkloadKind, key=lambda kind: kind.value)
    )
    for kind in WorkloadKind:
        frontend = catalog.frontend_for(kind)
        assert frontend.workload_kind is kind
        with pytest.raises(cm.UnsupportedError):
            frontend.build_programs(object(), tiles=())  # type: ignore[arg-type]
    with pytest.raises(cm.WorkloadError, match="frontends must be a sequence"):
        WorkloadFrontendCatalog(None)  # type: ignore[arg-type]
