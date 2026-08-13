from __future__ import annotations

import json
import sqlite3
from dataclasses import replace
from pathlib import Path

import pytest
from _profile_fixtures import copy_key, measurement

from tilefoundry_costmodel import ProfileConflictError, ProfileStoreError
from tilefoundry_costmodel.hardware.b200 import b200_hardware_spec
from tilefoundry_costmodel.profiles import store as profile_store
from tilefoundry_costmodel.profiles.model import SnapshotState
from tilefoundry_costmodel.profiles.store import open_profile_store


def test_measurement_rejects_mismatched_canonical_environment_digest() -> None:
    timing = measurement()
    wrong_environment = replace(timing.environment, environment_id="0" * 64)
    with pytest.raises(ProfileStoreError, match="canonical environment"):
        replace(timing, environment=wrong_environment)


def test_sqlite_schema_durability_and_frozen_snapshot_round_trip(tmp_path: Path) -> None:
    database = tmp_path / "profiles.db"
    exported = tmp_path / "snapshot.json"
    imported_database = tmp_path / "imported.db"
    hardware = b200_hardware_spec()
    timing = measurement()

    with open_profile_store(database, writable=True) as store:
        ref = store.create_snapshot(
            snapshot_id="copy",
            hardware=hardware,
            description="real B200 copy",
        )
        store.insert(ref, timing)
        assert store.lookup(ref, timing.key) == timing
        store.freeze(ref)
        assert store.snapshot_state(ref) is SnapshotState.FROZEN
        with pytest.raises(ProfileConflictError, match="immutable"):
            store.insert(ref, timing)
        store.export_snapshot(ref, exported)

    connection = sqlite3.connect(database)
    try:
        tables = {
            row[0]
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        assert {
            "metadata",
            "profile_keys",
            "environments",
            "measurements",
            "samples",
            "snapshots",
            "snapshot_measurements",
        } <= tables
        assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        assert connection.execute("PRAGMA synchronous").fetchone()[0] == 2
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
    finally:
        connection.close()

    with open_profile_store(imported_database, writable=True) as store:
        imported_ref = store.import_snapshot(exported)
        assert store.snapshot_state(imported_ref) is SnapshotState.FROZEN
        round_trip = tmp_path / "round-trip.json"
        store.export_snapshot(imported_ref, round_trip)
    assert round_trip.read_bytes() == exported.read_bytes()


def test_failed_insert_rolls_back_key_environment_and_membership(tmp_path: Path) -> None:
    database = tmp_path / "profiles.db"
    hardware = b200_hardware_spec()
    first = measurement()
    conflicting_environment = replace(
        first.environment,
        driver_version="13.2",
    )
    conflict = replace(first, environment=conflicting_environment)

    with open_profile_store(database, writable=True) as store:
        ref = store.create_snapshot(snapshot_id="copy", hardware=hardware, description="")
        store.insert(ref, first)
        before = store.logical_snapshot_bytes(ref)
        with pytest.raises(ProfileStoreError, match="environment ID"):
            store.insert(ref, conflict)
        assert store.logical_snapshot_bytes(ref) == before
        assert store.lookup(ref, copy_key()) == first


def test_freeze_rejects_empty_and_read_only_rejects_writes(tmp_path: Path) -> None:
    database = tmp_path / "profiles.db"
    hardware = b200_hardware_spec()
    with open_profile_store(database, writable=True) as store:
        ref = store.create_snapshot(snapshot_id="copy", hardware=hardware, description="")
        with pytest.raises(ProfileConflictError, match="empty"):
            store.freeze(ref)
    with open_profile_store(database, writable=False) as store:
        with pytest.raises(ProfileStoreError, match="read-only"):
            store.insert(ref, measurement())


def test_export_failure_preserves_existing_destination_atomically(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = tmp_path / "profiles.db"
    output = tmp_path / "snapshot.json"
    output.write_text("existing", encoding="utf-8")
    hardware = b200_hardware_spec()
    with open_profile_store(database, writable=True) as store:
        ref = store.create_snapshot(snapshot_id="copy", hardware=hardware, description="")
        store.insert(ref, measurement())
        store.freeze(ref)

        def fail_replace(_source: object, _destination: object) -> None:
            raise OSError("publish failed")

        monkeypatch.setattr(profile_store.os, "replace", fail_replace)
        with pytest.raises(ProfileStoreError, match="publish failed"):
            store.export_snapshot(ref, output)

    assert output.read_text(encoding="utf-8") == "existing"
    assert list(tmp_path.glob(f".{output.name}.*")) == []


def test_open_rejects_schema_and_measurement_corruption(tmp_path: Path) -> None:
    database = tmp_path / "profiles.db"
    hardware = b200_hardware_spec()
    timing = measurement()
    with open_profile_store(database, writable=True) as store:
        ref = store.create_snapshot(snapshot_id="copy", hardware=hardware, description="")
        store.insert(ref, timing)

    connection = sqlite3.connect(database)
    try:
        aggregate_text = connection.execute(
            "SELECT aggregate_json FROM measurements WHERE measurement_id = ?",
            (str(timing.measurement_id),),
        ).fetchone()[0]
        aggregate = json.loads(aggregate_text)
        aggregate["raw_samples_retained"] = 1
        connection.execute(
            "UPDATE measurements SET aggregate_json = ? WHERE measurement_id = ?",
            (json.dumps(aggregate, sort_keys=True, separators=(",", ":")), timing.measurement_id),
        )
        connection.commit()
    finally:
        connection.close()

    with open_profile_store(database, writable=False) as store:
        with pytest.raises(ProfileStoreError, match="must be boolean"):
            store.lookup(ref, timing.key)

    connection = sqlite3.connect(database)
    try:
        aggregate["raw_samples_retained"] = True
        aggregate["unexpected"] = "corrupt"
        connection.execute(
            "UPDATE measurements SET aggregate_json = ? WHERE measurement_id = ?",
            (json.dumps(aggregate, sort_keys=True, separators=(",", ":")), timing.measurement_id),
        )
        connection.commit()
    finally:
        connection.close()

    with open_profile_store(database, writable=False) as store:
        with pytest.raises(ProfileStoreError, match="unknown or missing"):
            store.lookup(ref, timing.key)

    connection = sqlite3.connect(database)
    try:
        connection.execute("ALTER TABLE metadata ADD COLUMN unexpected TEXT")
        connection.commit()
    finally:
        connection.close()
    with pytest.raises(ProfileStoreError, match="exact SQL schema"):
        open_profile_store(database, writable=False)


def test_read_only_uri_handles_special_path_and_frozen_base_copy(tmp_path: Path) -> None:
    database = tmp_path / "profiles ? #.db"
    hardware = b200_hardware_spec()
    timing = measurement()
    with open_profile_store(database, writable=True) as store:
        base = store.create_snapshot(snapshot_id="base", hardware=hardware, description="base")
        store.insert(base, timing)
        store.freeze(base)
        derived = store.create_snapshot(
            snapshot_id="derived",
            hardware=hardware,
            description="derived",
            base=base,
        )
        assert store.snapshot_state(derived) is SnapshotState.DRAFT
        assert store.lookup(derived, timing.key) == timing

    with open_profile_store(database, writable=False) as store:
        assert store.lookup(base, timing.key) == timing
        assert store.lookup(derived, timing.key) == timing


def test_import_does_not_treat_matching_draft_as_frozen_idempotence(tmp_path: Path) -> None:
    database = tmp_path / "profiles.db"
    exported = tmp_path / "draft.json"
    hardware = b200_hardware_spec()
    with open_profile_store(database, writable=True) as store:
        ref = store.create_snapshot(snapshot_id="draft", hardware=hardware, description="")
        store.insert(ref, measurement())
        store.export_snapshot(ref, exported)
        with pytest.raises(ProfileConflictError, match="not frozen"):
            store.import_snapshot(exported)
        assert store.snapshot_state(ref) is SnapshotState.DRAFT
