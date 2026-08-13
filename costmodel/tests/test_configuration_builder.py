"""M2 configuration enumeration, identity, and relation preservation."""

from __future__ import annotations

from dataclasses import replace

import pytest

from tilefoundry_costmodel import UnsupportedError
from tilefoundry_costmodel.build import ConfigurationBuilder
from tilefoundry_costmodel.hardware import b200_hardware_spec
from tilefoundry_costmodel.implementations import synthetic_implementation_catalog
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
    EndpointRelation,
    InstanceEndpoint,
    LoopBarrier,
    MemorySpace,
    OpIterationDomain,
    TileCandidate,
    TileDependency,
    TileLoop,
    TileOpKind,
    TileProgram,
    TileValue,
    TileValueType,
)
from tilefoundry_costmodel.request import SearchSpace, WarpConfig, WarpRole, WarpRoleAssignment


def _shape(*axes: tuple[str, int]) -> NamedShape:
    return NamedShape(tuple(AxisExtent(name, extent) for name, extent in axes))


def _value(value_id: str, memory: MemorySpace) -> TileValue:
    tensor = TensorDescriptor(_shape(("x", 16)), DType.BF16, TensorLayout.ROW_MAJOR)
    return TileValue(value_id, TileValueType(tensor, memory))


def _endpoint_barrier_program(program_id: str = "endpoint-program") -> TileProgram:
    first = CopyOp(
        TileOpKind.COPY,
        "first",
        "input",
        "middle",
        OpIterationDomain("first_loop", 0, 2),
    )
    second = CopyOp(
        TileOpKind.COPY,
        "second",
        "middle",
        "output",
        OpIterationDomain("second_loop", 0, 3),
    )
    relation = EndpointRelation(
        DependencyRelationKind.ENDPOINT,
        InstanceEndpoint.LAST,
        InstanceEndpoint.FIRST,
    )
    barrier = LoopBarrier("barrier", "first_loop", "second_loop")
    return TileProgram(
        2,
        program_id,
        WorkloadKind.GEMM,
        TileCandidate("tile", _shape(("m", 1), ("n", 1), ("k", 1))),
        (
            _value("input", MemorySpace.GLOBAL),
            _value("middle", MemorySpace.SHARED),
            _value("output", MemorySpace.GLOBAL),
        ),
        (TileLoop("first_loop", 2), TileLoop("second_loop", 3)),
        (first, second),
        (TileDependency("middle", "first", "second", relation),),
        (barrier,),
        ("input",),
        ("output",),
    )


def _warps(config_id: str, warp_id: int) -> WarpConfig:
    return WarpConfig(
        config_id,
        4,
        (WarpRoleAssignment(WarpRole.TMA_PRODUCER, (warp_id,)),),
    )


def test_endpoint_relation_and_explicit_barrier_are_preserved_exactly() -> None:
    program = _endpoint_barrier_program()
    search = SearchSpace(("synthetic.copy",), (_warps("w", 0),), (2,))
    template = ConfigurationBuilder(
        implementations=synthetic_implementation_catalog()
    ).enumerate_templates((program,), search_space=search, hardware=b200_hardware_spec())[0]
    assert len(template.dependencies) == 1
    assert template.dependencies[0].relation == program.dependencies[0].relation
    assert template.loop_barriers == program.loop_barriers
    assert template.dependencies[0].src_phase_id == "first.copy"
    assert template.dependencies[0].dst_phase_id == "second.copy"


def test_candidate_identity_and_enumeration_are_input_order_invariant() -> None:
    first = _endpoint_barrier_program("first-program")
    second = replace(first, program_id="second-program")
    forward = SearchSpace(
        ("synthetic.copy",),
        (_warps("warp-a", 0), _warps("warp-b", 1)),
        (1, 2),
        ("layout-a", "layout-b"),
    )
    reverse = SearchSpace(
        ("synthetic.copy",),
        tuple(reversed(forward.warp_configs)),
        tuple(reversed(forward.pipeline_depths)),
        tuple(reversed(forward.layout_variant_ids)),
    )
    builder = ConfigurationBuilder(implementations=synthetic_implementation_catalog())
    forward_templates = builder.enumerate_templates(
        (first, second), search_space=forward, hardware=b200_hardware_spec()
    )
    reverse_templates = builder.enumerate_templates(
        (second, first), search_space=reverse, hardware=b200_hardware_spec()
    )
    assert tuple(item.configuration_id for item in forward_templates) == tuple(
        item.configuration_id for item in reverse_templates
    )
    assert all(len(item.configuration_id) == 64 for item in forward_templates)


def test_no_ring_storage_canonicalizes_requested_depth_to_one() -> None:
    program = _endpoint_barrier_program()
    # Replacing both operation domains with one-time work removes all ring
    # storage while keeping the same typed copy workflow.
    operations = tuple(
        replace(operation, domain=OpIterationDomain(None, 0, 1)) for operation in program.operations
    )
    relation = EndpointRelation(
        DependencyRelationKind.ENDPOINT,
        InstanceEndpoint.FIRST,
        InstanceEndpoint.FIRST,
    )
    one_time = TileProgram(
        2,
        "one-time",
        WorkloadKind.GEMM,
        program.tile,
        program.values,
        (),
        operations,
        (TileDependency("middle", "first", "second", relation),),
        (),
        program.inputs,
        program.outputs,
    )
    search = SearchSpace(("synthetic.copy",), (_warps("w", 0),), (1, 2, 3))
    templates = ConfigurationBuilder(
        implementations=synthetic_implementation_catalog()
    ).enumerate_templates((one_time,), search_space=search, hardware=b200_hardware_spec())
    assert len(templates) == 1
    assert templates[0].pipeline_depth == 1


def test_empty_compatibility_program_is_not_an_executable_candidate() -> None:
    source = _endpoint_barrier_program("empty-compatibility")
    empty = TileProgram(
        2,
        source.program_id,
        source.workload_kind,
        source.tile,
        source.values,
        (),
        (),
        (),
        (),
        source.inputs,
        source.outputs,
    )
    encoded = empty.to_json()
    decoded = TileProgram.from_json(encoded)
    assert decoded.to_json() == encoded

    search = SearchSpace(("synthetic.copy",), (_warps("w", 0),), (1,))
    builder = ConfigurationBuilder(implementations=synthetic_implementation_catalog())
    for program in (empty, decoded):
        with pytest.raises(UnsupportedError, match="no legal"):
            builder.enumerate_templates(
                (program,), search_space=search, hardware=b200_hardware_spec()
            )
