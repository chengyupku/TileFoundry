"""Stable B200 resource identifiers and the M1 hardware fact catalog."""

from __future__ import annotations

from ..constants import HARDWARE_SCHEMA_VERSION
from ..model import DType, HardwareSpecRef, ResourceId
from .model import (
    FactOrigin,
    FactProvenance,
    HardwareSpec,
    StaticResourceSpec,
    StaticUnit,
    TemporalResourceSpec,
)

B200_CALIBRATION_ID = "b200-m1-baseline"
B200_HARDWARE_ID = "b200"
B200_SCHEMA_VERSION = HARDWARE_SCHEMA_VERSION

B200_TMA = ResourceId("b200.tma")
B200_TENSOR_CORE = ResourceId("b200.tensor_core")
B200_CUDA_CORE = ResourceId("b200.cuda_core")
B200_WARP_ISSUE = ResourceId("b200.warp_issue")
B200_GMEM_READ = ResourceId("b200.gmem_read")
B200_GMEM_WRITE = ResourceId("b200.gmem_write")
B200_SMEM_READ = ResourceId("b200.smem_read")
B200_SMEM_WRITE = ResourceId("b200.smem_write")
B200_TMEM_READ = ResourceId("b200.tmem_read")
B200_TMEM_WRITE = ResourceId("b200.tmem_write")
B200_RF_READ = ResourceId("b200.rf_read")
B200_RF_WRITE = ResourceId("b200.rf_write")
B200_TMA_INFLIGHT = ResourceId("b200.tma_inflight")
B200_TENSOR_INFLIGHT = ResourceId("b200.tensor_inflight")
B200_SMEM_BYTES = ResourceId("b200.smem_bytes")
B200_TMEM_BYTES = ResourceId("b200.tmem_bytes")
B200_REGISTERS_32BIT = ResourceId("b200.registers_32bit")
B200_WARPS = ResourceId("b200.warps")
B200_MBARRIER_SLOTS = ResourceId("b200.mbarrier_slots")

B200_TEMPORAL_RESOURCE_IDS: tuple[ResourceId, ...] = (
    B200_TMA,
    B200_TENSOR_CORE,
    B200_CUDA_CORE,
    B200_WARP_ISSUE,
    B200_GMEM_READ,
    B200_GMEM_WRITE,
    B200_SMEM_READ,
    B200_SMEM_WRITE,
    B200_TMEM_READ,
    B200_TMEM_WRITE,
    B200_RF_READ,
    B200_RF_WRITE,
    B200_TMA_INFLIGHT,
    B200_TENSOR_INFLIGHT,
)
B200_STATIC_RESOURCE_IDS: tuple[ResourceId, ...] = (
    B200_SMEM_BYTES,
    B200_TMEM_BYTES,
    B200_REGISTERS_32BIT,
    B200_WARPS,
    B200_MBARRIER_SLOTS,
)

_CUDA_TECHNICAL_SPECIFICATIONS = (
    "https://docs.nvidia.com/cuda/archive/12.8.1/cuda-c-programming-guide/"
    "index.html#features-and-technical-specifications-technical-specifications-per-compute-capability"
)
_BLACKWELL_TUNING_GUIDE = (
    "https://docs.nvidia.com/cuda/blackwell-tuning-guide/"
    "index.html#unified-shared-memory-l1-texture-cache"
)
_PTX_TMA = (
    "https://docs.nvidia.com/cuda/parallel-thread-execution/"
    "index.html#data-movement-and-conversion-instructions-bulk-copy"
)
_PTX_TENSOR_CORE = (
    "https://docs.nvidia.com/cuda/parallel-thread-execution/"
    "index.html#tensorcore-5th-generation-instructions"
)
_PTX_TENSOR_MEMORY = (
    "https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tensor-memory"
)
_PTX_MBARRIER = (
    "https://docs.nvidia.com/cuda/parallel-thread-execution/"
    "index.html#parallel-synchronization-and-communication-instructions-mbarrier"
)
_TENSORCORE_ARCHITECTURES = "sm_100a/sm_100f"


def _provenance(origin: FactOrigin, source: str, conditions: str) -> FactProvenance:
    return FactProvenance(origin, source, conditions)


def _conservative(source: str, subject: str, *, architectures: str = "sm_100") -> FactProvenance:
    return _provenance(
        FactOrigin.CONSERVATIVE,
        source,
        f"{architectures}, one resident CTA; {subject}; capacity is an M1 scheduling bound",
    )


def b200_hardware_spec() -> HardwareSpec:
    """Return the immutable schedulable B200 M1 facts."""

    temporal = (
        TemporalResourceSpec(
            B200_TMA, 1, "TMA issue engine", _conservative(_PTX_TMA, "TMA issue serialization")
        ),
        TemporalResourceSpec(
            B200_TENSOR_CORE,
            1,
            "tensor-core issue engine",
            _conservative(
                _PTX_TENSOR_CORE,
                "tcgen05 issue serialization",
                architectures=_TENSORCORE_ARCHITECTURES,
            ),
        ),
        TemporalResourceSpec(
            B200_CUDA_CORE,
            1,
            "CUDA-core issue engine",
            _conservative(_CUDA_TECHNICAL_SPECIFICATIONS, "CUDA-core issue serialization"),
        ),
        TemporalResourceSpec(
            B200_WARP_ISSUE,
            1,
            "warp issue pipe",
            _conservative(_CUDA_TECHNICAL_SPECIFICATIONS, "warp issue serialization"),
        ),
        TemporalResourceSpec(
            B200_GMEM_READ,
            1,
            "global-memory read path",
            _conservative(_CUDA_TECHNICAL_SPECIFICATIONS, "global-read path serialization"),
        ),
        TemporalResourceSpec(
            B200_GMEM_WRITE,
            1,
            "global-memory write path",
            _conservative(_CUDA_TECHNICAL_SPECIFICATIONS, "global-write path serialization"),
        ),
        TemporalResourceSpec(
            B200_SMEM_READ,
            1,
            "shared-memory read path",
            _conservative(_BLACKWELL_TUNING_GUIDE, "shared-read path serialization"),
        ),
        TemporalResourceSpec(
            B200_SMEM_WRITE,
            1,
            "shared-memory write path",
            _conservative(_BLACKWELL_TUNING_GUIDE, "shared-write path serialization"),
        ),
        TemporalResourceSpec(
            B200_TMEM_READ,
            1,
            "tensor-memory read path",
            _conservative(
                _PTX_TENSOR_MEMORY,
                "tensor-memory read serialization",
                architectures=_TENSORCORE_ARCHITECTURES,
            ),
        ),
        TemporalResourceSpec(
            B200_TMEM_WRITE,
            1,
            "tensor-memory write path",
            _conservative(
                _PTX_TENSOR_MEMORY,
                "tensor-memory write serialization",
                architectures=_TENSORCORE_ARCHITECTURES,
            ),
        ),
        TemporalResourceSpec(
            B200_RF_READ,
            1,
            "register-file read path",
            _conservative(_CUDA_TECHNICAL_SPECIFICATIONS, "register-read path serialization"),
        ),
        TemporalResourceSpec(
            B200_RF_WRITE,
            1,
            "register-file write path",
            _conservative(_CUDA_TECHNICAL_SPECIFICATIONS, "register-write path serialization"),
        ),
        TemporalResourceSpec(
            B200_TMA_INFLIGHT,
            1,
            "in-flight TMA transaction slots",
            _conservative(_PTX_TMA, "catalog restricts TMA concurrency to one transaction"),
        ),
        TemporalResourceSpec(
            B200_TENSOR_INFLIGHT,
            1,
            "in-flight tensor-core operation slots",
            _conservative(
                _PTX_TENSOR_CORE,
                "catalog restricts tensor-core concurrency to one tcgen05 operation",
                architectures=_TENSORCORE_ARCHITECTURES,
            ),
        ),
    )
    static = (
        StaticResourceSpec(
            B200_SMEM_BYTES,
            232_448,
            StaticUnit.BYTES,
            "shared memory bytes",
            _provenance(
                FactOrigin.VENDOR,
                _BLACKWELL_TUNING_GUIDE,
                "sm_100; 227 KiB opt-in maximum shared memory addressable by one CTA",
            ),
        ),
        StaticResourceSpec(
            B200_TMEM_BYTES,
            262_144,
            StaticUnit.BYTES,
            "tensor memory bytes",
            _provenance(
                FactOrigin.DERIVED,
                _PTX_TENSOR_MEMORY,
                "sm_100a/sm_100f; 512 columns * 128 lanes * 32 bits per cell for one CTA",
            ),
        ),
        StaticResourceSpec(
            B200_REGISTERS_32BIT,
            65_536,
            StaticUnit.REGISTERS_32BIT,
            "32-bit register capacity",
            _provenance(
                FactOrigin.VENDOR,
                _CUDA_TECHNICAL_SPECIFICATIONS,
                "compute capability 10.x; 64K 32-bit registers per thread block",
            ),
        ),
        StaticResourceSpec(
            B200_WARPS,
            32,
            StaticUnit.WARPS,
            "CTA warp capacity",
            _provenance(
                FactOrigin.DERIVED,
                _CUDA_TECHNICAL_SPECIFICATIONS,
                "compute capability 10.x; 1024 threads per block / 32 threads per warp",
            ),
        ),
        StaticResourceSpec(
            B200_MBARRIER_SLOTS,
            16,
            StaticUnit.SLOTS,
            "mbarrier slot capacity",
            _conservative(
                _PTX_MBARRIER,
                "catalog policy cap of 16 shared-memory mbarrier objects, not a hardware maximum",
            ),
        ),
    )
    return HardwareSpec(
        HardwareSpecRef(B200_HARDWARE_ID, HARDWARE_SCHEMA_VERSION, B200_CALIBRATION_ID),
        "B200",
        temporal,
        static,
        tuple(DType),
        (),
    )


__all__ = [
    "B200_CALIBRATION_ID",
    "B200_HARDWARE_ID",
    "B200_SCHEMA_VERSION",
    "B200_CUDA_CORE",
    "B200_GMEM_READ",
    "B200_GMEM_WRITE",
    "B200_MBARRIER_SLOTS",
    "B200_REGISTERS_32BIT",
    "B200_RF_READ",
    "B200_RF_WRITE",
    "B200_SMEM_BYTES",
    "B200_SMEM_READ",
    "B200_SMEM_WRITE",
    "B200_TENSOR_CORE",
    "B200_TENSOR_INFLIGHT",
    "B200_TMA",
    "B200_TMA_INFLIGHT",
    "B200_TMEM_BYTES",
    "B200_TMEM_READ",
    "B200_TMEM_WRITE",
    "B200_WARPS",
    "B200_WARP_ISSUE",
    "B200_STATIC_RESOURCE_IDS",
    "B200_TEMPORAL_RESOURCE_IDS",
    "b200_hardware_spec",
]
