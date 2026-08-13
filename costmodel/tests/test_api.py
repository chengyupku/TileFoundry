from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import pytest

import tilefoundry_costmodel as costmodel
from tilefoundry_costmodel.model import (
    AxisExtent,
    DType,
    GemmSpec,
    HardwareSpecRef,
    NamedShape,
    TensorDescriptor,
    TensorLayout,
    WorkloadKind,
)
from tilefoundry_costmodel.program import (
    MemorySpace,
    TileCandidate,
    TileProgram,
    TileValue,
    TileValueType,
)
from tilefoundry_costmodel.request import (
    CostModelRequest,
    ProfileSelection,
    ProfileSnapshotRef,
    SearchSpace,
    WarpConfig,
    WarpRole,
    WarpRoleAssignment,
)
from tilefoundry_costmodel.result import ProfileProvenance
from tilefoundry_costmodel.solver import SearchProblem

_SCHEMA_DIR = Path(__file__).parents[1] / "schemas"


def _schema(name: str) -> dict[str, object]:
    value = json.loads((_SCHEMA_DIR / name).read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _request() -> CostModelRequest:
    shape = NamedShape((AxisExtent("m", 1),))
    tensor = TensorDescriptor(shape, DType.BF16, TensorLayout.ROW_MAJOR)
    value = TileValue("x", TileValueType(tensor, MemorySpace.GLOBAL))
    program = TileProgram(
        2,
        "program",
        WorkloadKind.GEMM,
        TileCandidate("tile", shape),
        (value,),
        (),
        (),
        (),
        (),
        ("x",),
        ("x",),
    )
    workload = GemmSpec(
        WorkloadKind.GEMM,
        1,
        1,
        1,
        DType.BF16,
        DType.BF16,
        DType.FP32,
        DType.BF16,
        TensorLayout.ROW_MAJOR,
        TensorLayout.ROW_MAJOR,
    )
    search_space = SearchSpace(
        ("implementation",),
        (WarpConfig("warp", 1, (WarpRoleAssignment(WarpRole.TMA_PRODUCER, (0,)),)),),
        (1,),
    )
    return CostModelRequest(
        2,
        "request",
        workload,
        (program,),
        HardwareSpecRef("b200", 1, "calibration"),
        search_space,
        ProfileSelection(ProfileSnapshotRef("snapshot", 1)),
    )


def test_root_is_version_2_and_does_not_load_optional_backends() -> None:
    assert costmodel.COST_MODEL_API_VERSION == (2, 0)
    assert "StageSpec" not in costmodel.__all__
    assert "ImplementationCatalog" not in costmodel.__all__
    assert "TileProgram" not in costmodel.__all__
    assert "FactOrigin" not in costmodel.__all__
    assert "B200_SMEM_BYTES" not in costmodel.__all__
    probe = subprocess.run(
        (
            sys.executable,
            "-c",
            "import sys\n"
            "before = {name for name in sys.modules if name == 'cuda' or name.startswith('cuda.')}\n"
            "import tilefoundry_costmodel as costmodel\n"
            "after = {name for name in sys.modules if name == 'cuda' or name.startswith('cuda.')}\n"
            "assert costmodel.COST_MODEL_API_VERSION == (2, 0)\n"
            "assert after == before, (before, after)\n"
            "assert 'ortools' not in sys.modules\n"
            "assert 'tilefoundry_costmodel.legacy' not in sys.modules\n",
        ),
        check=False,
        capture_output=True,
        text=True,
    )
    assert probe.returncode == 0, probe.stdout + probe.stderr


def test_request_rejects_untyped_program_before_canonical_sorting() -> None:
    request = _request()
    with pytest.raises(costmodel.InvalidRequestError, match="TileProgram"):
        CostModelRequest(
            request.schema_version,
            request.request_id,
            request.workload,
            ({"program_id": "not-typed"},),  # type: ignore[arg-type]
            request.hardware,
            request.search_space,
            request.profiles,
            request.solver,
        )
    with pytest.raises(costmodel.InvalidRequestError, match="programs must be a sequence"):
        CostModelRequest(
            request.schema_version,
            request.request_id,
            request.workload,
            None,  # type: ignore[arg-type]
            request.hardware,
            request.search_space,
            request.profiles,
            request.solver,
        )
    with pytest.raises(costmodel.InvalidRequestError, match="warp_ids must be a sequence"):
        WarpRoleAssignment(WarpRole.TMA_PRODUCER, None)  # type: ignore[arg-type]


def test_request_round_trip_is_canonical_and_strict() -> None:
    request = _request()
    text = costmodel.request_to_json(request)
    assert text == costmodel.request_to_json(costmodel.request_from_json(text))

    unknown = json.loads(text)
    unknown["unexpected"] = True
    with pytest.raises(costmodel.InvalidRequestError):
        costmodel.request_from_json(json.dumps(unknown))

    unknown["schema_version"] = 999
    del unknown["unexpected"]
    with pytest.raises(costmodel.InvalidRequestError):
        costmodel.request_from_json(json.dumps(unknown))


def test_request_normalizes_default_layout_and_rejects_duplicate_programs() -> None:
    request = _request()
    assert request.search_space.layout_variant_ids == ("default",)
    duplicate = replace(request.programs[0], program_id="program-copy")
    with pytest.raises(costmodel.InvalidRequestError, match="duplicate canonical programs"):
        CostModelRequest(
            request.schema_version,
            request.request_id,
            request.workload,
            (request.programs[0], duplicate),
            request.hardware,
            request.search_space,
            request.profiles,
            request.solver,
        )


def test_request_schema_matches_optional_solver_and_typed_copy_decoder() -> None:
    request_payload = json.loads(costmodel.request_to_json(_request()))
    del request_payload["solver"]
    parsed = costmodel.request_from_json(json.dumps(request_payload))
    assert parsed.solver == _request().solver

    request_schema = _schema("request-v2.schema.json")
    request_required = request_schema["required"]
    assert isinstance(request_required, list)
    assert "solver" not in request_required

    solver_definition = request_schema["$defs"]["solver"]
    assert isinstance(solver_definition, dict)
    solver_constraints = solver_definition["allOf"]
    assert isinstance(solver_constraints, list)
    assert solver_constraints
    invalid_solver = json.loads(costmodel.request_to_json(_request()))
    invalid_solver["solver"]["ortools_workers"] = 2
    with pytest.raises(costmodel.InvalidRequestError):
        costmodel.request_from_json(json.dumps(invalid_solver))

    program_payload = json.loads(costmodel.program_to_json(_request().programs[0]))
    copy_operation = {
        "kind": "copy",
        "op_id": "copy",
        "source": "x",
        "destination": "x",
        "domain": {"loop_id": None, "first_iteration": 0, "iteration_count": 1},
    }
    for missing_field in ("source", "destination"):
        invalid_program = json.loads(json.dumps(program_payload))
        invalid_program["operations"] = [dict(copy_operation)]
        del invalid_program["operations"][0][missing_field]
        with pytest.raises(costmodel.WorkloadError):
            costmodel.program_from_json(json.dumps(invalid_program))

    copy_definitions = (
        ("program-v2.schema.json", "copy"),
        ("request-v2.schema.json", "copyOperation"),
        ("search-problem-v2.schema.json", "copyOperation"),
        ("plan-v2.schema.json", "copyOperation"),
        ("result-v2.schema.json", "copyOperation"),
    )
    for schema_name, definition_name in copy_definitions:
        schema = _schema(schema_name)
        definitions = schema["$defs"]
        assert isinstance(definitions, dict)
        definition = definitions[definition_name]
        assert isinstance(definition, dict)
        required = definition["required"]
        assert isinstance(required, list)
        assert {"source", "destination"} <= set(required)


@dataclass(frozen=True)
class _WrongSchemaDocument:
    schema_version: int = 2
    unexpected: str = "accepted only by a generic encoder"


def test_strict_serializers_require_their_owned_typed_records() -> None:
    document = _WrongSchemaDocument()
    with pytest.raises(costmodel.InvalidRequestError):
        costmodel.request_to_json(document)  # type: ignore[arg-type]
    with pytest.raises(costmodel.WorkloadError):
        costmodel.program_to_json(document)  # type: ignore[arg-type]
    with pytest.raises(costmodel.InvalidRequestError):
        costmodel.plan_to_json(document)  # type: ignore[arg-type]
    with pytest.raises(costmodel.InvalidRequestError):
        costmodel.result_to_json(document)  # type: ignore[arg-type]
    with pytest.raises(costmodel.SearchProblemError):
        costmodel.problem_to_json(document)  # type: ignore[arg-type]


def test_strict_serializers_reject_extended_record_subclasses() -> None:
    @dataclass(frozen=True, slots=True)
    class _ExtendedRequest(CostModelRequest):
        extra: str = "out-of-schema"

    request = _request()
    extended = _ExtendedRequest(
        request.schema_version,
        request.request_id,
        request.workload,
        request.programs,
        request.hardware,
        request.search_space,
        request.profiles,
        request.solver,
    )
    with pytest.raises(costmodel.InvalidRequestError):
        costmodel.request_to_json(extended)


def test_program_round_trip_rejects_unknown_nested_fields() -> None:
    text = costmodel.program_to_json(_request().programs[0])
    payload = json.loads(text)
    payload["tile"]["extra"] = 1
    with pytest.raises(costmodel.WorkloadError):
        costmodel.program_from_json(json.dumps(payload))


def test_hardware_and_profile_documents_use_strict_schema_versions() -> None:
    hardware = {
        "schema_version": 1,
        "ref": {
            "hardware_id": "b200",
            "schema_version": 1,
            "calibration_id": "calibration",
        },
        "architecture": "B200",
        "temporal_resources": [],
        "static_resources": [],
        "supported_dtypes": ["bf16"],
        "supported_implementation_ids": [],
    }
    assert costmodel.hardware_from_json(costmodel.hardware_to_json(hardware)) == hardware
    broken = dict(hardware)
    broken["unexpected"] = True
    with pytest.raises(costmodel.HardwareSpecError):
        costmodel.hardware_to_json(broken)

    snapshot = {
        "schema_version": 1,
        "snapshot_id": "snapshot",
        "revision": 1,
        "hardware": hardware,
        "measurements": [],
    }
    assert (
        costmodel.profile_snapshot_from_json(costmodel.profile_snapshot_to_json(snapshot))
        == snapshot
    )


def test_hardware_and_profile_serializers_reject_non_owned_dataclasses() -> None:
    @dataclass(frozen=True)
    class _HardwareLike:
        schema_version: int
        ref: object
        architecture: str
        temporal_resources: tuple[object, ...]
        static_resources: tuple[object, ...]
        supported_dtypes: tuple[str, ...]
        supported_implementation_ids: tuple[str, ...]

    hardware_like = _HardwareLike(
        1,
        {"hardware_id": "b200", "schema_version": 1, "calibration_id": "calibration"},
        "B200",
        (),
        (),
        ("bf16",),
        (),
    )
    with pytest.raises(costmodel.HardwareSpecError):
        costmodel.hardware_to_json(hardware_like)  # type: ignore[arg-type]

    @dataclass(frozen=True)
    class _SnapshotLike:
        schema_version: int
        snapshot_id: str
        revision: int
        hardware: object
        measurements: tuple[object, ...]

    snapshot_like = _SnapshotLike(1, "snapshot", 1, hardware_like, ())
    with pytest.raises(costmodel.ProfileStoreError):
        costmodel.profile_snapshot_to_json(snapshot_like)  # type: ignore[arg-type]


def test_legacy_api_is_explicitly_versioned() -> None:
    from tilefoundry_costmodel import legacy

    assert legacy.COST_MODEL_API_VERSION == (0, 2)


def _problem_payload() -> dict[str, object]:
    request = _request()
    request_payload = json.loads(costmodel.request_to_json(request))
    hardware = {
        "schema_version": 1,
        "ref": {
            "hardware_id": "b200",
            "schema_version": 1,
            "calibration_id": "calibration",
        },
        "architecture": "B200",
        "temporal_resources": [],
        "static_resources": [],
        "supported_dtypes": ["bf16"],
        "supported_implementation_ids": [],
    }
    program = request_payload["programs"][0]
    configuration = {
        "configuration_id": "configuration",
        "program_id": "program",
        "workload_kind": "gemm",
        "tile": program["tile"],
        "implementations": [],
        "warps": {
            "config_id": "warp",
            "total_warps": 1,
            "roles": [{"role": "tma_producer", "warp_ids": [0]}],
        },
        "pipeline_depth": 1,
        "layout_variant_id": "default",
        "loops": [],
        "phases": [],
        "dependencies": [],
        "loop_barriers": [],
        "start_alignments": [],
        "buffers": [],
        "static_demands": [],
    }
    return {
        "schema_version": 2,
        "request_id": "request",
        "hardware": hardware,
        "workload": request_payload["workload"],
        "programs": [program],
        "profile_snapshot": {"snapshot_id": "snapshot", "revision": 1},
        "solver_options": {},
        "configurations": [configuration],
        "rejected_before_solve": [],
    }


def test_shared_warp_roles_are_legal_at_request_boundary() -> None:
    config = WarpConfig(
        "shared",
        1,
        (
            WarpRoleAssignment(WarpRole.TMA_PRODUCER, (0,)),
            WarpRoleAssignment(WarpRole.CUDA_EPILOGUE, (0,)),
        ),
    )
    assert config.roles[0].warp_ids == config.roles[1].warp_ids


def test_duplicate_warp_role_assignments_are_rejected_by_boundary_schema() -> None:
    with pytest.raises(costmodel.InvalidRequestError, match="roles must be unique"):
        WarpConfig(
            "duplicate",
            2,
            (
                WarpRoleAssignment(WarpRole.TMA_PRODUCER, (0,)),
                WarpRoleAssignment(WarpRole.TMA_PRODUCER, (1,)),
            ),
        )
    schema = _schema("request-v2.schema.json")
    definitions = schema["$defs"]
    assert isinstance(definitions, dict)
    warp_config = definitions["warpConfig"]
    assert isinstance(warp_config, dict)
    properties = warp_config["properties"]
    assert isinstance(properties, dict)
    roles = properties["roles"]
    assert isinstance(roles, dict)
    constraints = roles["allOf"]
    assert isinstance(constraints, list)
    assert {item["maxContains"] for item in constraints if isinstance(item, dict)} == {1}


def test_warp_ids_are_canonicalized_as_sorted_sets() -> None:
    unsorted = WarpRoleAssignment(WarpRole.TMA_PRODUCER, (1, 0))
    assert unsorted.warp_ids == (0, 1)
    with pytest.raises(costmodel.InvalidRequestError, match="duplicate canonical choices"):
        SearchSpace(
            ("implementation",),
            (
                WarpConfig("first", 2, (unsorted,)),
                WarpConfig(
                    "second",
                    2,
                    (WarpRoleAssignment(WarpRole.TMA_PRODUCER, (0, 1)),),
                ),
            ),
            (1,),
        )


def test_identifier_schema_and_decoder_agree_on_ascii() -> None:
    payload = json.loads(costmodel.request_to_json(_request()))
    payload["request_id"] = "\u8bf7\u6c42"
    with pytest.raises(costmodel.InvalidRequestError):
        costmodel.request_from_json(json.dumps(payload))
    schema = _schema("request-v2.schema.json")
    definitions = schema["$defs"]
    assert isinstance(definitions, dict)
    identifier = definitions["id"]
    assert isinstance(identifier, dict)
    assert identifier["pattern"] == "^[\\u0000-\\u007F]+$"

    domain = definitions["domain"]
    assert isinstance(domain, dict)
    domain_properties = domain["properties"]
    assert isinstance(domain_properties, dict)
    loop_id = domain_properties["loop_id"]
    assert isinstance(loop_id, dict)
    assert loop_id["minLength"] == 1


def test_program_and_profile_identifier_schema_boundaries_match_decoders() -> None:
    program_payload = json.loads(costmodel.program_to_json(_request().programs[0]))
    program_payload["program_id"] = "\u7a0b\u5e8f"
    with pytest.raises(costmodel.WorkloadError):
        costmodel.program_from_json(json.dumps(program_payload))

    snapshot = {
        "schema_version": 1,
        "snapshot_id": "snapshot",
        "revision": 1,
        "hardware": _problem_payload()["hardware"],
        "measurements": [],
    }
    snapshot["snapshot_id"] = "\u5feb\u7167"
    with pytest.raises(costmodel.ProfileStoreError):
        costmodel.profile_snapshot_from_json(json.dumps(snapshot))

    profile_schema = _schema("profile-snapshot-v1.schema.json")
    properties = profile_schema["properties"]
    assert isinstance(properties, dict)
    snapshot_id = properties["snapshot_id"]
    assert isinstance(snapshot_id, dict)
    assert snapshot_id["pattern"] == "^[\\u0000-\\u007F]+$"


def test_request_records_reject_untyped_mutable_children() -> None:
    with pytest.raises(costmodel.InvalidRequestError):
        SearchSpace(("implementation",), ({"config_id": "mutable"},), (1,))  # type: ignore[arg-type]
    with pytest.raises(costmodel.InvalidRequestError):
        ProfileSelection({"snapshot_id": "mutable", "revision": 1})  # type: ignore[arg-type]


def test_program_and_result_records_do_not_retain_mutable_children() -> None:
    with pytest.raises(costmodel.WorkloadError):
        TileCandidate("tile", {"axes": []})  # type: ignore[arg-type]
    with pytest.raises(costmodel.InvalidRequestError):
        costmodel.CostModelResult(
            2,
            costmodel.EvaluationStatus.UNSUPPORTED,
            rejected_candidates=({"code": "unsupported", "message": "mutable"},),  # type: ignore[arg-type]
        )
    with pytest.raises(costmodel.InvalidRequestError):
        costmodel.CostModelResult(
            2,
            costmodel.EvaluationStatus.UNSUPPORTED,
            diagnostics=({"code": "unsupported", "message": "mutable"},),  # type: ignore[arg-type]
        )

    diagnostics = [costmodel.Diagnostic(costmodel.DiagnosticCode.UNSUPPORTED, "unsupported")]
    rejected = [
        costmodel.RejectedCandidate(None, costmodel.DiagnosticCode.UNSUPPORTED, "unsupported")
    ]
    result = costmodel.CostModelResult(
        2,
        costmodel.EvaluationStatus.UNSUPPORTED,
        rejected_candidates=rejected,  # type: ignore[arg-type]
        diagnostics=diagnostics,  # type: ignore[arg-type]
    )
    before = costmodel.result_to_json(result)
    diagnostics.clear()
    rejected.clear()
    assert costmodel.result_to_json(result) == before


def test_profile_provenance_rejects_values_outside_schema_enums() -> None:
    fields = ("phase", "op", "implementation", "phase_name", "component", "measurement", "key")
    with pytest.raises(costmodel.InvalidRequestError):
        ProfileProvenance(*fields, "environment", "unknown", "p50", "p90")


def test_search_problem_configuration_decoder_is_strict_and_immutable() -> None:
    payload = _problem_payload()
    parsed = costmodel.problem_from_json(json.dumps(payload))
    before = costmodel.problem_to_json(parsed)
    assert isinstance(parsed, SearchProblem)
    with pytest.raises((AttributeError, TypeError)):
        parsed.configurations[0].layout_variant_id = "changed"  # type: ignore[misc]
    with pytest.raises((AttributeError, TypeError)):
        parsed.configurations[0].tile.shape.axes[0].extent = 2  # type: ignore[misc]
    assert costmodel.problem_to_json(parsed) == before

    unknown = json.loads(json.dumps(payload))
    unknown["configurations"][0]["unexpected"] = True
    with pytest.raises(costmodel.SearchProblemError):
        costmodel.problem_from_json(json.dumps(unknown))


def test_hardware_and_profile_codecs_reject_incomplete_nested_records() -> None:
    hardware = _problem_payload()["hardware"]
    broken_hardware = json.loads(json.dumps(hardware))
    broken_hardware["temporal_resources"] = [{}]
    with pytest.raises(costmodel.HardwareSpecError):
        costmodel.hardware_from_json(json.dumps(broken_hardware))

    snapshot = {
        "schema_version": 1,
        "snapshot_id": "snapshot",
        "revision": 1,
        "hardware": hardware,
        "measurements": [],
    }
    invalid_snapshot = json.loads(json.dumps(snapshot))
    invalid_snapshot["hardware"]["ref"] = {"hardware_id": "b200"}
    with pytest.raises(costmodel.ProfileStoreError):
        costmodel.profile_snapshot_from_json(json.dumps(invalid_snapshot))

    empty_measurement = json.loads(json.dumps(snapshot))
    empty_measurement["measurements"] = [{}]
    with pytest.raises(costmodel.ProfileStoreError):
        costmodel.profile_snapshot_from_json(json.dumps(empty_measurement))
