from __future__ import annotations

import json
from pathlib import Path

import pytest
from _profile_fixtures import measurement

from tilefoundry_costmodel import hardware_to_json
from tilefoundry_costmodel.cli import main
from tilefoundry_costmodel.hardware.b200 import b200_hardware_spec
from tilefoundry_costmodel.profiles.store import open_profile_store
from tilefoundry_costmodel.request import ProfileSnapshotRef


def test_snapshot_cli_create_base_inspect_freeze_export_import(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    hardware = b200_hardware_spec()
    hardware_path = tmp_path / "hardware.json"
    hardware_path.write_text(hardware_to_json(hardware), encoding="utf-8")
    database = tmp_path / "profiles.db"

    assert (
        main(
            (
                "profiles",
                "create",
                "--profiles",
                str(database),
                "--snapshot",
                "base",
                "--hardware",
                str(hardware_path),
                "--description",
                "base snapshot",
            )
        )
        == 0
    )
    captured = capsys.readouterr()
    assert captured.out.strip() == "base@1"

    base = ProfileSnapshotRef("base", 1)
    with open_profile_store(database, writable=True) as store:
        store.insert(base, measurement())

    assert (
        main(
            (
                "profiles",
                "freeze",
                "--profiles",
                str(database),
                "--snapshot",
                "base@1",
            )
        )
        == 0
    )
    capsys.readouterr()

    assert (
        main(
            (
                "profiles",
                "create",
                "--profiles",
                str(database),
                "--snapshot",
                "derived",
                "--hardware",
                str(hardware_path),
                "--description",
                "derived snapshot",
                "--base",
                "base@1",
            )
        )
        == 0
    )
    captured = capsys.readouterr()
    assert captured.out.strip() == "derived@1"

    assert (
        main(
            (
                "profiles",
                "inspect",
                "--profiles",
                str(database),
                "--snapshot",
                "derived@1",
            )
        )
        == 0
    )
    captured = capsys.readouterr()
    inspected = json.loads(captured.out)
    assert inspected["state"] == "draft"
    assert len(inspected["members"]) == 1

    assert (
        main(
            (
                "profiles",
                "freeze",
                "--profiles",
                str(database),
                "--snapshot",
                "derived@1",
            )
        )
        == 0
    )
    capsys.readouterr()

    exported = tmp_path / "derived.json"
    assert (
        main(
            (
                "profiles",
                "export",
                "--profiles",
                str(database),
                "--snapshot",
                "derived@1",
                "--output",
                str(exported),
            )
        )
        == 0
    )

    imported_database = tmp_path / "imported.db"
    assert (
        main(
            (
                "profiles",
                "import",
                "--profiles",
                str(imported_database),
                str(exported),
            )
        )
        == 0
    )
    captured = capsys.readouterr()
    assert captured.out.strip() == "derived@1"

    round_trip = tmp_path / "round-trip.json"
    assert (
        main(
            (
                "profiles",
                "export",
                "--profiles",
                str(imported_database),
                "--snapshot",
                "derived@1",
                "--output",
                str(round_trip),
            )
        )
        == 0
    )
    assert round_trip.read_bytes() == exported.read_bytes()
