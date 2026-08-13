from __future__ import annotations

import hashlib
import json
import sys

import tilefoundry_costmodel as cm
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
    AlignedRelation,
    DependencyRelationKind,
    GemmOp,
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


def shape(*axes: tuple[str, int]) -> NamedShape:
    return NamedShape(tuple(AxisExtent(name, extent) for name, extent in axes))


def value(value_id: str, tensor_shape: NamedShape, space: MemorySpace) -> TileValue:
    tensor = TensorDescriptor(tensor_shape, DType.BF16, TensorLayout.ROW_MAJOR)
    return TileValue(value_id, TileValueType(tensor, space))


def main() -> None:
    assert cm.COST_MODEL_API_VERSION == (2, 0)
    print("module=" + str(cm.__file__))
    assert "ortools" not in sys.modules
    assert "cuda" not in sys.modules
    assert "tilefoundry_costmodel.legacy" not in sys.modules
    mn = shape(("m", 16), ("n", 16))
    lhs = shape(("m", 16), ("k", 8))
    rhs = shape(("k", 8), ("n", 16))
    op = GemmOp(
        TileOpKind.GEMM,
        "gemm",
        "lhs",
        "rhs",
        "state",
        "state",
        "m",
        "n",
        "k",
        OpIterationDomain("k_loop", 0, 4),
    )
    program = TileProgram(
        2,
        "golden",
        WorkloadKind.GEMM,
        TileCandidate("tile", shape(("m", 16), ("n", 16), ("k", 8))),
        (
            value("lhs", lhs, MemorySpace.GLOBAL),
            value("rhs", rhs, MemorySpace.GLOBAL),
            value("state", mn, MemorySpace.SHARED),
        ),
        (TileLoop("k_loop", 4),),
        (op,),
        (
            TileDependency(
                "state",
                "gemm",
                "gemm",
                AlignedRelation(DependencyRelationKind.ALIGNED, 1),
            ),
        ),
        (),
        ("lhs", "rhs"),
        ("state",),
    )
    warps = WarpConfig(
        "warps",
        4,
        (WarpRoleAssignment(WarpRole.TENSOR_CONSUMER, (0, 1)),),
    )
    search = SearchSpace(("synthetic.gemm",), (warps,), (1, 2, 3))
    builder = ConfigurationBuilder(implementations=synthetic_implementation_catalog())
    templates = builder.enumerate_templates(
        (program,), search_space=search, hardware=b200_hardware_spec()
    )
    payload = [
        {
            "configuration_id": item.configuration_id,
            "pipeline_depth": item.pipeline_depth,
            "slot_count": item.buffers[0].slot_count,
            "static_units": item.static_demands[0].units,
            "profile_keys": [
                key.key_id() for key in builder.profile_keys(item, hardware=b200_hardware_spec())
            ],
        }
        for item in templates
    ]
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    print(text)
    print("golden_sha256=" + hashlib.sha256(text.encode("utf-8")).hexdigest())
    print("import_boundary=PASS")


if __name__ == "__main__":
    main()
