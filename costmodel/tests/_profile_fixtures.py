from __future__ import annotations

from tilefoundry_costmodel.hardware.b200 import b200_hardware_spec
from tilefoundry_costmodel.implementations.b200.copy import B200CopyProvider
from tilefoundry_costmodel.model import (
    AxisExtent,
    DType,
    MeasurementId,
    NamedShape,
    TensorDescriptor,
    TensorLayout,
    WorkloadKind,
)
from tilefoundry_costmodel.profiles.model import (
    MeasurementOrigin,
    ProfileEnvironment,
    ProfileMeasurement,
)
from tilefoundry_costmodel.program import (
    CopyOp,
    MemorySpace,
    OpIterationDomain,
    TileCandidate,
    TileOpKind,
    TileProgram,
    TileValue,
    TileValueType,
)
from tilefoundry_costmodel.tileop import (
    CanonicalAttribute,
    TileOpProfileKey,
    TileOpProfileQuery,
    tile_op_signature,
)


def copy_program(*, extent: int = 4) -> TileProgram:
    shape = NamedShape(
        (
            AxisExtent("m", 1),
            AxisExtent("n", 1),
            AxisExtent("k", extent),
        )
    )
    tensor = TensorDescriptor(shape, DType.BF16, TensorLayout.ROW_MAJOR)
    values = (
        TileValue("src", TileValueType(tensor, MemorySpace.GLOBAL)),
        TileValue("dst", TileValueType(tensor, MemorySpace.GLOBAL)),
    )
    operation = CopyOp(
        TileOpKind.COPY,
        "copy",
        "src",
        "dst",
        OpIterationDomain(None, 0, 1),
    )
    return TileProgram(
        2,
        f"copy-program-{extent}",
        WorkloadKind.GEMM,
        TileCandidate(f"tile-{extent}", shape),
        values,
        (),
        (operation,),
        (),
        (),
        ("src",),
        ("dst",),
    )


def copy_query(*, extent: int = 4, pipeline_depth: int = 1) -> TileOpProfileQuery:
    program = copy_program(extent=extent)
    hardware = b200_hardware_spec()
    return TileOpProfileQuery(
        hardware.ref,
        tile_op_signature(program.operations[0], program=program),
        "b200.copy",
        "b200.copy",
        program.tile.shape,
        "one-tma-warp",
        pipeline_depth,
        "default",
        (
            CanonicalAttribute("cache_policy", "default"),
            CanonicalAttribute("memory_residency", "global"),
        ),
    )


def copy_key(*, extent: int = 4, pipeline_depth: int = 1) -> TileOpProfileKey:
    provider = B200CopyProvider()
    hardware = b200_hardware_spec()
    query = copy_query(extent=extent, pipeline_depth=pipeline_depth)
    return TileOpProfileKey(1, query, provider.fingerprint(query, hardware))


def profile_environment(*, environment_id: str = "environment") -> ProfileEnvironment:
    hardware = b200_hardware_spec()
    return ProfileEnvironment(
        environment_id,
        "00112233445566778899aabbccddeeff",
        hardware.ref,
        "sm_100a",
        "13.1",
        "12.9",
        "13.1",
        1_800_000,
        4_000_000,
        None,
    )


def measurement(
    *,
    extent: int = 4,
    pipeline_depth: int = 1,
    measurement_id: str = "measurement",
    environment_id: str = "environment",
    retain_raw: bool = True,
) -> ProfileMeasurement:
    latency = (100, 100, 101) if retain_raw else ()
    interval = (10, 10, 10) if retain_raw else ()
    return ProfileMeasurement(
        MeasurementId(measurement_id),
        copy_key(extent=extent, pipeline_depth=pipeline_depth),
        profile_environment(environment_id=environment_id),
        MeasurementOrigin.MEASURED,
        100,
        101,
        10,
        10,
        2,
        3,
        8,
        16,
        100_000,
        10_000,
        retain_raw,
        latency,
        interval,
        "2026-08-10T00:00:00Z",
    )
