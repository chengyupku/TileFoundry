"""M1 exact B200 catalog and provenance workflows."""

from __future__ import annotations

from pathlib import Path

import pytest

import tilefoundry_costmodel as cm
from tilefoundry_costmodel.errors import HardwareSpecError, UnsupportedError
from tilefoundry_costmodel.hardware.b200 import (
    B200_CALIBRATION_ID,
    B200_MBARRIER_SLOTS,
    B200_REGISTERS_32BIT,
    B200_SMEM_BYTES,
    B200_STATIC_RESOURCE_IDS,
    B200_TEMPORAL_RESOURCE_IDS,
    B200_TENSOR_INFLIGHT,
    B200_TMA,
    B200_TMA_INFLIGHT,
    B200_TMEM_BYTES,
    B200_WARPS,
)
from tilefoundry_costmodel.hardware.model import (
    FactOrigin,
    FactProvenance,
    StaticUnit,
    TemporalResourceSpec,
)
from tilefoundry_costmodel.hardware.registry import HardwareCatalog
from tilefoundry_costmodel.model import HardwareSpecRef


def test_b200_catalog_exposes_stable_temporal_and_static_facts() -> None:
    catalog = cm.b200_hardware_catalog()
    spec = catalog.resolve(HardwareSpecRef("b200", 1, B200_CALIBRATION_ID))
    assert {resource.resource_id for resource in spec.temporal_resources} == set(
        B200_TEMPORAL_RESOURCE_IDS
    )
    assert {resource.resource_id for resource in spec.static_resources} == set(
        B200_STATIC_RESOURCE_IDS
    )
    assert all(resource.capacity_slots > 0 for resource in spec.temporal_resources)
    assert all(resource.capacity_units > 0 for resource in spec.static_resources)
    assert all(
        resource.provenance.origin is not FactOrigin.UNAVAILABLE
        for resource in spec.temporal_resources
    )
    assert spec.static_capacity(B200_SMEM_BYTES) == 232_448
    assert spec.static_capacity(B200_TMEM_BYTES) == 262_144
    assert spec.static_capacity(B200_REGISTERS_32BIT) == 65_536
    assert spec.static_capacity(B200_WARPS) == 32
    assert spec.static_capacity(B200_MBARRIER_SLOTS) == 16
    assert spec.temporal_capacity(B200_TMA_INFLIGHT) == 1
    assert spec.temporal_capacity(B200_TENSOR_INFLIGHT) == 1
    assert all(
        resource.provenance.source.startswith("https://docs.nvidia.com/")
        for resource in (*spec.temporal_resources, *spec.static_resources)
    )
    tensor_core_sources = {
        resource.provenance.source
        for resource in spec.temporal_resources
        if "tensor" in resource.description
    }
    assert all(
        source.endswith("#tensorcore-5th-generation-instructions")
        for source in tensor_core_sources
        if "tensorcore" in source
    )
    assert all(
        "sm_100a/sm_100f" in resource.provenance.conditions
        for resource in spec.static_resources
        if resource.resource_id == B200_TMEM_BYTES
    )
    assert all(resource.provenance.conditions for resource in spec.temporal_resources)
    static_units = {resource.resource_id: resource.unit for resource in spec.static_resources}
    assert static_units == {
        B200_MBARRIER_SLOTS: StaticUnit.SLOTS,
        B200_REGISTERS_32BIT: StaticUnit.REGISTERS_32BIT,
        B200_SMEM_BYTES: StaticUnit.BYTES,
        B200_TMEM_BYTES: StaticUnit.BYTES,
        B200_WARPS: StaticUnit.WARPS,
    }
    assert all(
        resource.provenance.source and resource.provenance.conditions
        for resource in spec.static_resources
    )


def test_calibration_document_matches_the_installed_b200_catalog() -> None:
    path = Path(__file__).parents[1] / "calibration" / "b200-hardware.json"
    from_document = cm.hardware_from_json(path.read_text(encoding="utf-8"))
    from_catalog = cm.b200_hardware_catalog().specs[0]
    assert cm.hardware_to_json(from_document) == cm.hardware_to_json(from_catalog)


def test_b200_lookup_requires_exact_identity_and_resource_class() -> None:
    catalog = cm.b200_hardware_catalog()
    with pytest.raises(UnsupportedError):
        catalog.resolve(HardwareSpecRef("b200-nearby", 1, B200_CALIBRATION_ID))
    with pytest.raises(UnsupportedError):
        catalog.resolve(HardwareSpecRef("b200", 1, "different-calibration"))
    spec = catalog.specs[0]
    assert spec.temporal_capacity(B200_TMA) > 0
    assert spec.static_capacity(B200_SMEM_BYTES) > 0
    with pytest.raises(HardwareSpecError, match="static"):
        spec.temporal_capacity(B200_SMEM_BYTES)
    with pytest.raises(HardwareSpecError, match="temporal"):
        spec.static_capacity(B200_TMA)


def test_unavailable_fact_cannot_become_schedulable_capacity() -> None:
    provenance = FactProvenance(FactOrigin.UNAVAILABLE, "unknown", "not measured")
    with pytest.raises(HardwareSpecError, match="unavailable"):
        TemporalResourceSpec("b200.unavailable", 1, "unknown", provenance)
    with pytest.raises(HardwareSpecError, match="specs must be a sequence"):
        HardwareCatalog(None)  # type: ignore[arg-type]
