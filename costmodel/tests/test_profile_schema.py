from __future__ import annotations

import copy
import json
from pathlib import Path

from _profile_fixtures import measurement
from jsonschema import Draft202012Validator, FormatChecker

from tilefoundry_costmodel import profile_snapshot_to_json
from tilefoundry_costmodel.constants import PROFILE_SCHEMA_VERSION
from tilefoundry_costmodel.hardware.b200 import b200_hardware_spec
from tilefoundry_costmodel.profiles.model import ProfileSnapshot

_SCHEMA_DIR = Path(__file__).parents[1] / "schemas"


def test_all_published_json_schemas_are_valid_draft_2020_12() -> None:
    schemas = sorted(_SCHEMA_DIR.glob("*.json"))
    assert len(schemas) == 7
    for path in schemas:
        Draft202012Validator.check_schema(json.loads(path.read_text(encoding="utf-8")))


def test_profile_snapshot_schema_matches_runtime_measurement_conditions() -> None:
    schema = json.loads((_SCHEMA_DIR / "profile-snapshot-v1.schema.json").read_text())
    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    snapshot = ProfileSnapshot(
        PROFILE_SCHEMA_VERSION,
        "schema-check",
        1,
        b200_hardware_spec(),
        (measurement(),),
    )
    document = json.loads(profile_snapshot_to_json(snapshot))
    validator.validate(document)

    raw_without_retention = copy.deepcopy(document)
    raw_without_retention["measurements"][0]["raw_samples_retained"] = False
    assert list(validator.iter_errors(raw_without_retention))

    unpaired_interval = copy.deepcopy(document)
    unpaired_interval["measurements"][0]["initiation_interval_p50_ps"] = None
    assert list(validator.iter_errors(unpaired_interval))

    noncanonical_timestamp = copy.deepcopy(document)
    noncanonical_timestamp["measurements"][0]["measured_at_utc"] = "2026-08-10T00:00:00+00:00"
    assert list(validator.iter_errors(noncanonical_timestamp))
