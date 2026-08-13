"""Transactional SQLite storage for exact timing snapshots.

The store deliberately uses only :mod:`sqlite3` and the dependency-free typed
records.  CUDA and provider code never enters this module, which means a
frozen snapshot can be replayed on a host that has no GPU or CUDA installation.
"""

from __future__ import annotations

import json
import os
import sqlite3
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import cast

from .._serialization import (
    _decode_profile_environment,
    _decode_profile_key,
    _loads,
    hardware_from_json,
    profile_snapshot_from_json,
    profile_snapshot_to_json,
)
from ..constants import PROFILE_SCHEMA_VERSION
from ..errors import (
    HardwareSpecError,
    InvalidRequestError,
    ProfileConflictError,
    ProfileStoreError,
)
from ..hardware.model import HardwareSpec
from ..model import HardwareSpecRef, MeasurementId, TimingMetric, validate_identifier
from ..request import ProfileSnapshotRef
from ..tileop import TileOpProfileKey
from .model import (
    MeasurementOrigin,
    ProfileMeasurement,
    ProfileSnapshot,
    SnapshotState,
    profile_environment_canonical_json,
    profile_environment_id,
)

STORE_SCHEMA_VERSION = 1

_TABLES = (
    "metadata",
    "profile_keys",
    "environments",
    "measurements",
    "samples",
    "snapshots",
    "snapshot_measurements",
)

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS profile_keys (
    key_id TEXT PRIMARY KEY,
    canonical_json TEXT NOT NULL UNIQUE
);
CREATE TABLE IF NOT EXISTS environments (
    environment_id TEXT PRIMARY KEY,
    canonical_json TEXT NOT NULL UNIQUE
);
CREATE TABLE IF NOT EXISTS measurements (
    measurement_id TEXT PRIMARY KEY,
    key_id TEXT NOT NULL,
    environment_id TEXT NOT NULL,
    aggregate_json TEXT NOT NULL,
    created_at_utc TEXT NOT NULL,
    UNIQUE (measurement_id, key_id),
    FOREIGN KEY (key_id) REFERENCES profile_keys(key_id),
    FOREIGN KEY (environment_id) REFERENCES environments(environment_id)
);
CREATE TABLE IF NOT EXISTS samples (
    measurement_id TEXT NOT NULL,
    metric TEXT NOT NULL CHECK (metric IN ('latency', 'initiation_interval')),
    sample_index INTEGER NOT NULL,
    elapsed_ps INTEGER NOT NULL,
    PRIMARY KEY (measurement_id, metric, sample_index),
    FOREIGN KEY (measurement_id) REFERENCES measurements(measurement_id)
);
CREATE TABLE IF NOT EXISTS snapshots (
    snapshot_id TEXT NOT NULL,
    revision INTEGER NOT NULL,
    state TEXT NOT NULL CHECK (state IN ('draft', 'frozen')),
    hardware_ref_json TEXT NOT NULL,
    environment_id TEXT,
    description TEXT NOT NULL,
    created_at_utc TEXT NOT NULL,
    PRIMARY KEY (snapshot_id, revision),
    FOREIGN KEY (environment_id) REFERENCES environments(environment_id)
);
CREATE TABLE IF NOT EXISTS snapshot_measurements (
    snapshot_id TEXT NOT NULL,
    snapshot_revision INTEGER NOT NULL,
    key_id TEXT NOT NULL,
    measurement_id TEXT NOT NULL,
    PRIMARY KEY (snapshot_id, snapshot_revision, key_id),
    FOREIGN KEY (snapshot_id, snapshot_revision)
        REFERENCES snapshots(snapshot_id, revision),
    FOREIGN KEY (key_id) REFERENCES profile_keys(key_id),
    FOREIGN KEY (measurement_id, key_id)
        REFERENCES measurements(measurement_id, key_id)
);
"""

_EXPECTED_COLUMNS: dict[str, tuple[tuple[str, str, int, int], ...]] = {
    "metadata": (("key", "TEXT", 0, 1), ("value", "TEXT", 1, 0)),
    "profile_keys": (
        ("key_id", "TEXT", 0, 1),
        ("canonical_json", "TEXT", 1, 0),
    ),
    "environments": (
        ("environment_id", "TEXT", 0, 1),
        ("canonical_json", "TEXT", 1, 0),
    ),
    "measurements": (
        ("measurement_id", "TEXT", 0, 1),
        ("key_id", "TEXT", 1, 0),
        ("environment_id", "TEXT", 1, 0),
        ("aggregate_json", "TEXT", 1, 0),
        ("created_at_utc", "TEXT", 1, 0),
    ),
    "samples": (
        ("measurement_id", "TEXT", 1, 1),
        ("metric", "TEXT", 1, 2),
        ("sample_index", "INTEGER", 1, 3),
        ("elapsed_ps", "INTEGER", 1, 0),
    ),
    "snapshots": (
        ("snapshot_id", "TEXT", 1, 1),
        ("revision", "INTEGER", 1, 2),
        ("state", "TEXT", 1, 0),
        ("hardware_ref_json", "TEXT", 1, 0),
        ("environment_id", "TEXT", 0, 0),
        ("description", "TEXT", 1, 0),
        ("created_at_utc", "TEXT", 1, 0),
    ),
    "snapshot_measurements": (
        ("snapshot_id", "TEXT", 1, 1),
        ("snapshot_revision", "INTEGER", 1, 2),
        ("key_id", "TEXT", 1, 3),
        ("measurement_id", "TEXT", 1, 0),
    ),
}

_EXPECTED_UNIQUE_INDEXES: dict[str, set[tuple[str, ...]]] = {
    "metadata": {("key",)},
    "profile_keys": {("key_id",), ("canonical_json",)},
    "environments": {("environment_id",), ("canonical_json",)},
    "measurements": {("measurement_id",), ("measurement_id", "key_id")},
    "samples": {("measurement_id", "metric", "sample_index")},
    "snapshots": {("snapshot_id", "revision")},
    "snapshot_measurements": {("snapshot_id", "snapshot_revision", "key_id")},
}

_ForeignKeyRow = tuple[str, str, str, str, str, str]
_EXPECTED_FOREIGN_KEYS: dict[str, set[tuple[_ForeignKeyRow, ...]]] = {
    "metadata": set(),
    "profile_keys": set(),
    "environments": set(),
    "measurements": {
        (("profile_keys", "key_id", "key_id", "NO ACTION", "NO ACTION", "NONE"),),
        (
            (
                "environments",
                "environment_id",
                "environment_id",
                "NO ACTION",
                "NO ACTION",
                "NONE",
            ),
        ),
    },
    "samples": {
        (
            (
                "measurements",
                "measurement_id",
                "measurement_id",
                "NO ACTION",
                "NO ACTION",
                "NONE",
            ),
        )
    },
    "snapshots": {
        (
            (
                "environments",
                "environment_id",
                "environment_id",
                "NO ACTION",
                "NO ACTION",
                "NONE",
            ),
        )
    },
    "snapshot_measurements": {
        (
            ("snapshots", "snapshot_id", "snapshot_id", "NO ACTION", "NO ACTION", "NONE"),
            (
                "snapshots",
                "snapshot_revision",
                "revision",
                "NO ACTION",
                "NO ACTION",
                "NONE",
            ),
        ),
        (("profile_keys", "key_id", "key_id", "NO ACTION", "NO ACTION", "NONE"),),
        (
            (
                "measurements",
                "measurement_id",
                "measurement_id",
                "NO ACTION",
                "NO ACTION",
                "NONE",
            ),
            (
                "measurements",
                "key_id",
                "key_id",
                "NO ACTION",
                "NO ACTION",
                "NONE",
            ),
        ),
    },
}

_EXPECTED_CHECKS = {
    "samples": "check(metricin('latency','initiation_interval'))",
    "snapshots": "check(statein('draft','frozen'))",
}

_AGGREGATE_FIELDS = {
    "measurement_id",
    "origin",
    "latency_p50_ps",
    "latency_p90_ps",
    "initiation_interval_p50_ps",
    "initiation_interval_p90_ps",
    "warmup_runs",
    "sample_count",
    "latency_repetitions_per_sample",
    "initiation_interval_repetitions_per_sample",
    "target_sample_ns",
    "relative_iqr_ppm",
    "raw_samples_retained",
    "measured_at_utc",
}


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _ref_payload(ref: HardwareSpecRef) -> dict[str, object]:
    return {
        "hardware_id": ref.hardware_id,
        "schema_version": ref.schema_version,
        "calibration_id": ref.calibration_id,
    }


def _ref_from_json(text: str) -> HardwareSpecRef:
    try:
        value = _loads(text, ProfileStoreError)
        if not isinstance(value, dict):
            raise ProfileStoreError("hardware reference JSON must be an object")
        expected = {"hardware_id", "schema_version", "calibration_id"}
        if set(value) != expected:
            raise ProfileStoreError("hardware reference JSON has unknown or missing fields")
        return HardwareSpecRef(
            value["hardware_id"],
            value["schema_version"],
            value["calibration_id"],
        )
    except ProfileStoreError:
        raise
    except (TypeError, ValueError, InvalidRequestError) as exc:
        raise ProfileStoreError("invalid hardware reference JSON") from exc


def _ref_from_object(value: object) -> HardwareSpecRef:
    if type(value) is HardwareSpecRef:
        return value
    if type(value) is HardwareSpec:
        return value.ref
    raise HardwareSpecError("snapshot hardware must be HardwareSpecRef or HardwareSpec")


def _now_utc() -> str:
    # SQLite metadata timestamps are only audit information.  Keep the format
    # accepted by the public decoder and avoid floating-point time values.
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _validate_ref(value: object, label: str = "snapshot") -> ProfileSnapshotRef:
    if type(value) is not ProfileSnapshotRef:
        raise ProfileStoreError(f"{label} must be ProfileSnapshotRef")
    return value


class SqliteProfileStore:
    """Own one SQLite profile-store connection.

    All mutating methods use an explicit transaction.  The connection is never
    exposed, so a failed multi-table insert cannot leave a usable partial
    measurement behind.
    """

    def __init__(self, connection: sqlite3.Connection, *, writable: bool) -> None:
        self._connection = connection
        self._writable = writable
        self._closed = False

    @property
    def path(self) -> Path | None:
        """Return the path when sqlite exposes one (diagnostic convenience)."""

        row = self._connection.execute("PRAGMA database_list").fetchone()
        if row is None or not row[2]:
            return None
        return Path(str(row[2]))

    def close(self) -> None:
        if not self._closed:
            self._connection.close()
            self._closed = True

    def __enter__(self) -> "SqliteProfileStore":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def create_snapshot(
        self,
        *,
        snapshot_id: str,
        hardware: HardwareSpecRef | HardwareSpec,
        description: str,
        base: ProfileSnapshotRef | None = None,
    ) -> ProfileSnapshotRef:
        """Create a draft revision, optionally copying a frozen base."""

        self._ensure_writable()
        try:
            validate_identifier(snapshot_id, label="snapshot_id")
        except InvalidRequestError as exc:
            raise ProfileStoreError(str(exc)) from exc
        if not isinstance(description, str):
            raise ProfileStoreError("snapshot description must be text")
        hardware_ref = _ref_from_object(hardware)
        base_ref = None if base is None else _validate_ref(base, "base snapshot")
        try:
            self._begin()
            if type(hardware) is HardwareSpec:
                self._remember_hardware_in_transaction(hardware)
            if base_ref is not None:
                base_row = self._snapshot_row(base_ref)
                if base_row["state"] != SnapshotState.FROZEN.value:
                    raise ProfileConflictError("base snapshot must be frozen")
                if _ref_from_json(str(base_row["hardware_ref_json"])) != hardware_ref:
                    raise ProfileConflictError("base and new snapshot hardware must match")
            row = self._connection.execute(
                "SELECT COALESCE(MAX(revision), 0) + 1 AS next_revision "
                "FROM snapshots WHERE snapshot_id = ?",
                (snapshot_id,),
            ).fetchone()
            assert row is not None
            revision = int(row["next_revision"])
            created = _now_utc()
            self._connection.execute(
                "INSERT INTO snapshots(snapshot_id, revision, state, hardware_ref_json, "
                "environment_id, description, created_at_utc) VALUES (?, ?, 'draft', ?, NULL, ?, ?)",
                (
                    snapshot_id,
                    revision,
                    _canonical_json(_ref_payload(hardware_ref)),
                    description,
                    created,
                ),
            )
            if base_ref is not None:
                self._connection.execute(
                    "UPDATE snapshots SET environment_id = (SELECT environment_id FROM snapshots "
                    "WHERE snapshot_id = ? AND revision = ?) WHERE snapshot_id = ? AND revision = ?",
                    (base_ref.snapshot_id, base_ref.revision, snapshot_id, revision),
                )
                self._connection.execute(
                    "INSERT INTO snapshot_measurements(snapshot_id, snapshot_revision, key_id, measurement_id) "
                    "SELECT ?, ?, key_id, measurement_id FROM snapshot_measurements "
                    "WHERE snapshot_id = ? AND snapshot_revision = ?",
                    (snapshot_id, revision, base_ref.snapshot_id, base_ref.revision),
                )
            self._commit()
            return ProfileSnapshotRef(snapshot_id, revision)
        except Exception:
            self._rollback()
            raise

    def snapshot_state(self, ref: ProfileSnapshotRef) -> SnapshotState:
        row = self._snapshot_row(_validate_ref(ref))
        try:
            return SnapshotState(str(row["state"]))
        except ValueError as exc:
            raise ProfileStoreError("snapshot contains an invalid state") from exc

    def lookup(
        self,
        ref: ProfileSnapshotRef,
        key: TileOpProfileKey,
    ) -> ProfileMeasurement | None:
        snapshot_ref = _validate_ref(ref)
        if type(key) is not TileOpProfileKey:
            raise ProfileStoreError("profile lookup key must be TileOpProfileKey")
        snapshot = self._snapshot_row(snapshot_ref)
        snapshot_hardware = _ref_from_json(str(snapshot["hardware_ref_json"]))
        if key.query.hardware != snapshot_hardware:
            raise ProfileConflictError("profile key hardware does not match snapshot")
        key_id = str(key.key_id())
        key_row = self._connection.execute(
            "SELECT canonical_json FROM profile_keys WHERE key_id = ?", (key_id,)
        ).fetchone()
        if key_row is not None and str(key_row["canonical_json"]) != key.canonical_json():
            raise ProfileStoreError("profile-key hash collision or corrupted canonical JSON")
        membership = self._connection.execute(
            "SELECT measurement_id FROM snapshot_measurements WHERE snapshot_id = ? "
            "AND snapshot_revision = ? AND key_id = ?",
            (snapshot_ref.snapshot_id, snapshot_ref.revision, key_id),
        ).fetchone()
        if membership is None:
            return None
        return self._measurement_by_id(str(membership["measurement_id"]), key)

    def insert(self, ref: ProfileSnapshotRef, measurement: ProfileMeasurement) -> None:
        """Insert a complete measurement and membership atomically."""

        self._ensure_writable()
        snapshot_ref = _validate_ref(ref)
        if type(measurement) is not ProfileMeasurement:
            raise ProfileStoreError("measurement must be ProfileMeasurement")
        try:
            self._begin()
            snapshot = self._snapshot_row(snapshot_ref)
            if snapshot["state"] != SnapshotState.DRAFT.value:
                raise ProfileConflictError("frozen snapshots are immutable")
            snapshot_hardware = _ref_from_json(str(snapshot["hardware_ref_json"]))
            if measurement.key.query.hardware != snapshot_hardware:
                raise ProfileConflictError("measurement key hardware does not match snapshot")
            if measurement.environment.hardware != snapshot_hardware:
                raise ProfileConflictError(
                    "measurement environment hardware does not match snapshot"
                )

            environment_id = measurement.environment.environment_id
            environment_json = profile_environment_canonical_json(measurement.environment)
            self._insert_environment(environment_id, environment_json)
            pinned = snapshot["environment_id"]
            if pinned is not None and str(pinned) != environment_id:
                raise ProfileConflictError("snapshot measurements must use one environment")
            if pinned is None:
                self._connection.execute(
                    "UPDATE snapshots SET environment_id = ? WHERE snapshot_id = ? AND revision = ?",
                    (environment_id, snapshot_ref.snapshot_id, snapshot_ref.revision),
                )

            key_id = str(measurement.key.key_id())
            key_json = measurement.key.canonical_json()
            self._insert_key(key_id, key_json)
            self._insert_measurement(measurement, key_id, environment_id)
            existing = self._connection.execute(
                "SELECT measurement_id FROM snapshot_measurements WHERE snapshot_id = ? "
                "AND snapshot_revision = ? AND key_id = ?",
                (snapshot_ref.snapshot_id, snapshot_ref.revision, key_id),
            ).fetchone()
            if existing is not None:
                if str(existing["measurement_id"]) != str(measurement.measurement_id):
                    raise ProfileConflictError("snapshot already contains this profile key")
            else:
                self._connection.execute(
                    "INSERT INTO snapshot_measurements(snapshot_id, snapshot_revision, key_id, measurement_id) "
                    "VALUES (?, ?, ?, ?)",
                    (
                        snapshot_ref.snapshot_id,
                        snapshot_ref.revision,
                        key_id,
                        str(measurement.measurement_id),
                    ),
                )
            self._commit()
        except Exception:
            self._rollback()
            raise

    def freeze(self, ref: ProfileSnapshotRef) -> ProfileSnapshotRef:
        self._ensure_writable()
        snapshot_ref = _validate_ref(ref)
        try:
            self._begin()
            snapshot = self._snapshot_row(snapshot_ref)
            if snapshot["state"] != SnapshotState.DRAFT.value:
                raise ProfileConflictError("snapshot is already frozen")
            count_row = self._connection.execute(
                "SELECT COUNT(*) AS count FROM snapshot_measurements WHERE snapshot_id = ? "
                "AND snapshot_revision = ?",
                (snapshot_ref.snapshot_id, snapshot_ref.revision),
            ).fetchone()
            assert count_row is not None
            if int(count_row["count"]) == 0:
                raise ProfileConflictError("cannot freeze an empty snapshot")
            self._connection.execute(
                "UPDATE snapshots SET state = 'frozen' WHERE snapshot_id = ? AND revision = ?",
                (snapshot_ref.snapshot_id, snapshot_ref.revision),
            )
            self._commit()
            return snapshot_ref
        except Exception:
            self._rollback()
            raise

    def export_snapshot(self, ref: ProfileSnapshotRef, output: Path) -> None:
        snapshot = self.snapshot(_validate_ref(ref))
        output_path = Path(output)
        temporary_path: Path | None = None
        try:
            text = profile_snapshot_to_json(snapshot)
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=output_path.parent,
                prefix=f".{output_path.name}.",
                delete=False,
            ) as temporary:
                temporary.write(text)
                temporary.flush()
                os.fsync(temporary.fileno())
                temporary_path = Path(temporary.name)
            os.replace(temporary_path, output_path)
            temporary_path = None
        except OSError as exc:
            raise ProfileStoreError(f"cannot export profile snapshot: {exc}") from exc
        finally:
            if temporary_path is not None:
                try:
                    temporary_path.unlink()
                except FileNotFoundError:
                    pass

    def import_snapshot(self, source: Path) -> ProfileSnapshotRef:
        try:
            snapshot = profile_snapshot_from_json(Path(source).read_text(encoding="utf-8"))
        except OSError as exc:
            raise ProfileStoreError(f"cannot read profile snapshot: {exc}") from exc
        # Import is a complete operation: create the exact revision as a draft,
        # insert every row, then freeze it in one transaction.
        self._ensure_writable()
        ref = ProfileSnapshotRef(snapshot.snapshot_id, snapshot.revision)
        try:
            self._begin()
            self._remember_hardware_in_transaction(snapshot.hardware)
            existing = self._connection.execute(
                "SELECT state FROM snapshots WHERE snapshot_id = ? AND revision = ?",
                (ref.snapshot_id, ref.revision),
            ).fetchone()
            if existing is not None:
                # Idempotent import is allowed only when the logical bytes are
                # identical; a differing document is an immutable conflict.
                if str(existing["state"]) != SnapshotState.FROZEN.value:
                    raise ProfileConflictError("snapshot revision already exists but is not frozen")
                try:
                    existing_document = profile_snapshot_to_json(self.snapshot(ref))
                except ProfileStoreError:
                    raise
                if existing_document != profile_snapshot_to_json(snapshot):
                    raise ProfileConflictError("snapshot revision already contains different data")
                self._rollback()
                return ref
            self._connection.execute(
                "INSERT INTO snapshots(snapshot_id, revision, state, hardware_ref_json, environment_id, description, created_at_utc) "
                "VALUES (?, ?, 'draft', ?, NULL, ?, ?)",
                (
                    ref.snapshot_id,
                    ref.revision,
                    _canonical_json(_ref_payload(snapshot.hardware.ref)),
                    "imported",
                    _now_utc(),
                ),
            )
            for measurement in snapshot.measurements:
                self._insert_measurement_for_snapshot(ref, measurement)
            if not snapshot.measurements:
                raise ProfileConflictError("cannot import an empty snapshot")
            self._connection.execute(
                "UPDATE snapshots SET state = 'frozen' WHERE snapshot_id = ? AND revision = ?",
                (ref.snapshot_id, ref.revision),
            )
            self._commit()
            return ref
        except Exception:
            self._rollback()
            raise

    def snapshot(self, ref: ProfileSnapshotRef) -> ProfileSnapshot:
        """Materialize a typed snapshot document from normalized rows."""

        snapshot_ref = _validate_ref(ref)
        row = self._snapshot_row(snapshot_ref)
        hardware = self._hardware_for_ref(_ref_from_json(str(row["hardware_ref_json"])))
        rows = self._connection.execute(
            "SELECT sm.measurement_id, sm.key_id FROM snapshot_measurements sm "
            "WHERE sm.snapshot_id = ? AND sm.snapshot_revision = ? ORDER BY sm.key_id",
            (snapshot_ref.snapshot_id, snapshot_ref.revision),
        ).fetchall()
        measurements: list[ProfileMeasurement] = []
        for item in rows:
            measurement = self._measurement_by_id(str(item["measurement_id"]), key=None)
            if str(measurement.key.key_id()) != str(item["key_id"]):
                raise ProfileStoreError("snapshot membership key does not match measurement key")
            measurements.append(measurement)
        return ProfileSnapshot(
            PROFILE_SCHEMA_VERSION,
            snapshot_ref.snapshot_id,
            snapshot_ref.revision,
            hardware,
            tuple(measurements),
        )

    def logical_snapshot_json(self, ref: ProfileSnapshotRef) -> str:
        """Return canonical logical membership/aggregate bytes for AC-3-1."""

        return self._logical_snapshot_json(_validate_ref(ref))

    def logical_snapshot_bytes(self, ref: ProfileSnapshotRef) -> bytes:
        return self.logical_snapshot_json(ref).encode("utf-8")

    def description(self, ref: ProfileSnapshotRef) -> str:
        row = self._snapshot_row(_validate_ref(ref))
        return str(row["description"])

    def _insert_measurement_for_snapshot(
        self, ref: ProfileSnapshotRef, measurement: ProfileMeasurement
    ) -> None:
        snapshot = self._snapshot_row(ref)
        snapshot_hardware = _ref_from_json(str(snapshot["hardware_ref_json"]))
        if (
            measurement.key.query.hardware != snapshot_hardware
            or measurement.environment.hardware != snapshot_hardware
        ):
            raise ProfileConflictError("imported measurement hardware does not match snapshot")
        self._insert_environment(
            measurement.environment.environment_id,
            profile_environment_canonical_json(measurement.environment),
        )
        pinned = snapshot["environment_id"]
        if pinned is not None and str(pinned) != measurement.environment.environment_id:
            raise ProfileConflictError("snapshot measurements must use one environment")
        if pinned is None:
            self._connection.execute(
                "UPDATE snapshots SET environment_id = ? WHERE snapshot_id = ? AND revision = ?",
                (measurement.environment.environment_id, ref.snapshot_id, ref.revision),
            )
        key_id = str(measurement.key.key_id())
        self._insert_key(key_id, measurement.key.canonical_json())
        self._insert_measurement(measurement, key_id, measurement.environment.environment_id)
        self._connection.execute(
            "INSERT INTO snapshot_measurements(snapshot_id, snapshot_revision, key_id, measurement_id) VALUES (?, ?, ?, ?)",
            (ref.snapshot_id, ref.revision, key_id, str(measurement.measurement_id)),
        )

    def _insert_key(self, key_id: str, canonical: str) -> None:
        row = self._connection.execute(
            "SELECT canonical_json FROM profile_keys WHERE key_id = ?", (key_id,)
        ).fetchone()
        if row is not None:
            if str(row["canonical_json"]) != canonical:
                raise ProfileStoreError("profile-key hash collision or corrupted canonical JSON")
            return
        try:
            self._connection.execute(
                "INSERT INTO profile_keys(key_id, canonical_json) VALUES (?, ?)",
                (key_id, canonical),
            )
        except sqlite3.IntegrityError as exc:
            raise ProfileConflictError(
                "profile key canonical JSON conflicts with an existing row"
            ) from exc

    def _insert_environment(self, environment_id: str, canonical: str) -> None:
        row = self._connection.execute(
            "SELECT canonical_json FROM environments WHERE environment_id = ?",
            (environment_id,),
        ).fetchone()
        if row is not None:
            if str(row["canonical_json"]) != canonical:
                raise ProfileStoreError("environment ID conflicts with canonical environment JSON")
            return
        try:
            self._connection.execute(
                "INSERT INTO environments(environment_id, canonical_json) VALUES (?, ?)",
                (environment_id, canonical),
            )
        except sqlite3.IntegrityError as exc:
            raise ProfileConflictError(
                "environment canonical JSON conflicts with an existing row"
            ) from exc

    def _insert_measurement(
        self, measurement: ProfileMeasurement, key_id: str, environment_id: str
    ) -> None:
        measurement_id = str(measurement.measurement_id)
        aggregate = measurement.aggregate_json()
        row = self._connection.execute(
            "SELECT key_id, environment_id, aggregate_json FROM measurements WHERE measurement_id = ?",
            (measurement_id,),
        ).fetchone()
        if row is not None:
            if (
                str(row["key_id"]) != key_id
                or str(row["environment_id"]) != environment_id
                or str(row["aggregate_json"]) != aggregate
            ):
                raise ProfileConflictError("measurement ID conflicts with existing measurement")
            self._verify_samples(measurement)
            return
        try:
            self._connection.execute(
                "INSERT INTO measurements(measurement_id, key_id, environment_id, aggregate_json, created_at_utc) "
                "VALUES (?, ?, ?, ?, ?)",
                (measurement_id, key_id, environment_id, aggregate, measurement.measured_at_utc),
            )
            if measurement.raw_samples_retained:
                self._connection.executemany(
                    "INSERT INTO samples(measurement_id, metric, sample_index, elapsed_ps) VALUES (?, ?, ?, ?)",
                    (
                        (measurement_id, TimingMetric.LATENCY.value, index, elapsed)
                        for index, elapsed in enumerate(measurement.raw_latency_samples_ps)
                    ),
                )
                if measurement.raw_initiation_interval_samples_ps:
                    self._connection.executemany(
                        "INSERT INTO samples(measurement_id, metric, sample_index, elapsed_ps) VALUES (?, ?, ?, ?)",
                        (
                            (measurement_id, TimingMetric.INITIATION_INTERVAL.value, index, elapsed)
                            for index, elapsed in enumerate(
                                measurement.raw_initiation_interval_samples_ps
                            )
                        ),
                    )
        except sqlite3.IntegrityError as exc:
            raise ProfileConflictError("measurement rows conflict with existing data") from exc

    def _verify_samples(self, measurement: ProfileMeasurement) -> None:
        measurement_id = str(measurement.measurement_id)
        rows = self._connection.execute(
            "SELECT metric, sample_index, elapsed_ps FROM samples WHERE measurement_id = ? "
            "ORDER BY metric, sample_index",
            (measurement_id,),
        ).fetchall()
        expected = [
            (TimingMetric.LATENCY.value, index, elapsed)
            for index, elapsed in enumerate(measurement.raw_latency_samples_ps)
        ] + [
            (TimingMetric.INITIATION_INTERVAL.value, index, elapsed)
            for index, elapsed in enumerate(measurement.raw_initiation_interval_samples_ps)
        ]
        if [
            (str(row["metric"]), int(row["sample_index"]), int(row["elapsed_ps"])) for row in rows
        ] != sorted(expected):
            raise ProfileStoreError("measurement sample rows are corrupted")

    def _measurement_by_id(
        self, measurement_id: str, key: TileOpProfileKey | None
    ) -> ProfileMeasurement:
        row = self._connection.execute(
            "SELECT measurement_id, key_id, environment_id, aggregate_json, created_at_utc "
            "FROM measurements WHERE measurement_id = ?",
            (measurement_id,),
        ).fetchone()
        if row is None:
            raise ProfileStoreError("snapshot references a missing measurement")
        key_row = self._connection.execute(
            "SELECT canonical_json FROM profile_keys WHERE key_id = ?", (str(row["key_id"]),)
        ).fetchone()
        environment_row = self._connection.execute(
            "SELECT canonical_json FROM environments WHERE environment_id = ?",
            (str(row["environment_id"]),),
        ).fetchone()
        if key_row is None or environment_row is None:
            raise ProfileStoreError("measurement references missing key or environment")
        key_json = str(key_row["canonical_json"])
        decoded_key = _decode_profile_key(_loads(key_json, ProfileStoreError))
        if decoded_key.canonical_json() != key_json:
            raise ProfileStoreError("stored profile key JSON is not canonical")
        if str(decoded_key.key_id()) != str(row["key_id"]):
            raise ProfileStoreError("stored profile key ID does not match canonical JSON")
        if key is not None and decoded_key.canonical_json() != key.canonical_json():
            raise ProfileStoreError("snapshot membership key does not match measurement key")
        environment_json = str(environment_row["canonical_json"])
        environment = _decode_profile_environment(_loads(environment_json, ProfileStoreError))
        if profile_environment_canonical_json(environment) != environment_json:
            raise ProfileStoreError("stored environment JSON is not canonical")
        if environment.environment_id != str(row["environment_id"]):
            raise ProfileStoreError("stored environment ID does not match canonical JSON")
        if len(environment.environment_id) == 64 and (
            environment.environment_id != profile_environment_id(environment)
        ):
            raise ProfileStoreError("stored environment digest does not match canonical data")
        aggregate_json = str(row["aggregate_json"])
        aggregate_value = _loads(aggregate_json, ProfileStoreError)
        if not isinstance(aggregate_value, dict):
            raise ProfileStoreError("measurement aggregate JSON must be an object")
        if set(aggregate_value) != _AGGREGATE_FIELDS:
            raise ProfileStoreError("measurement aggregate JSON has unknown or missing fields")
        if _canonical_json(aggregate_value) != aggregate_json:
            raise ProfileStoreError("measurement aggregate JSON is not canonical")
        samples = self._connection.execute(
            "SELECT metric, sample_index, elapsed_ps FROM samples WHERE measurement_id = ? "
            "ORDER BY metric, sample_index",
            (measurement_id,),
        ).fetchall()
        samples_by_metric: dict[str, list[tuple[int, int]]] = {
            TimingMetric.LATENCY.value: [],
            TimingMetric.INITIATION_INTERVAL.value: [],
        }
        for item in samples:
            metric = str(item["metric"])
            if metric not in samples_by_metric:
                raise ProfileStoreError("measurement contains an unknown sample metric")
            sample_index = item["sample_index"]
            elapsed_ps = item["elapsed_ps"]
            if (
                isinstance(sample_index, bool)
                or not isinstance(sample_index, int)
                or sample_index < 0
                or isinstance(elapsed_ps, bool)
                or not isinstance(elapsed_ps, int)
                or elapsed_ps <= 0
            ):
                raise ProfileStoreError("measurement sample row is invalid")
            samples_by_metric[metric].append((sample_index, elapsed_ps))
        for values in samples_by_metric.values():
            if [index for index, _elapsed in values] != list(range(len(values))):
                raise ProfileStoreError("measurement sample indices are not contiguous")
        latency = [elapsed for _index, elapsed in samples_by_metric[TimingMetric.LATENCY.value]]
        interval = [
            elapsed for _index, elapsed in samples_by_metric[TimingMetric.INITIATION_INTERVAL.value]
        ]
        retained_value = aggregate_value["raw_samples_retained"]
        if type(retained_value) is not bool:
            raise ProfileStoreError("raw_samples_retained must be boolean")
        retained = retained_value
        if not retained and (latency or interval):
            raise ProfileStoreError("non-retained measurement has sample rows")
        try:
            aggregate_measurement_id = aggregate_value["measurement_id"]
            if not isinstance(aggregate_measurement_id, str):
                raise ProfileStoreError("aggregate measurement ID must be text")
            if aggregate_measurement_id != measurement_id:
                raise ProfileStoreError("aggregate measurement ID does not match row identity")
            measured_at_utc = aggregate_value["measured_at_utc"]
            if not isinstance(measured_at_utc, str):
                raise ProfileStoreError("aggregate measured_at_utc must be text")
            if str(row["created_at_utc"]) != measured_at_utc:
                raise ProfileStoreError("measurement row timestamp does not match aggregate")
            return ProfileMeasurement(
                MeasurementId(aggregate_measurement_id),
                decoded_key,
                environment,
                MeasurementOrigin(cast(str, aggregate_value["origin"])),
                cast(int, aggregate_value["latency_p50_ps"]),
                cast(int, aggregate_value["latency_p90_ps"]),
                cast(int | None, aggregate_value.get("initiation_interval_p50_ps")),
                cast(int | None, aggregate_value.get("initiation_interval_p90_ps")),
                cast(int, aggregate_value["warmup_runs"]),
                cast(int, aggregate_value["sample_count"]),
                cast(int, aggregate_value["latency_repetitions_per_sample"]),
                cast(int | None, aggregate_value.get("initiation_interval_repetitions_per_sample")),
                cast(int, aggregate_value["target_sample_ns"]),
                cast(int, aggregate_value["relative_iqr_ppm"]),
                retained,
                tuple(latency),
                tuple(interval),
                measured_at_utc,
            )
        except (KeyError, TypeError, ValueError, ProfileStoreError) as exc:
            if isinstance(exc, ProfileStoreError):
                raise
            raise ProfileStoreError("invalid measurement aggregate row") from exc

    def _snapshot_row(self, ref: ProfileSnapshotRef) -> sqlite3.Row:
        row = self._connection.execute(
            "SELECT snapshot_id, revision, state, hardware_ref_json, environment_id, description, created_at_utc "
            "FROM snapshots WHERE snapshot_id = ? AND revision = ?",
            (ref.snapshot_id, ref.revision),
        ).fetchone()
        if row is None:
            raise ProfileStoreError(
                f"snapshot revision does not exist: {ref.snapshot_id}@{ref.revision}"
            )
        return cast(sqlite3.Row, row)

    def _hardware_for_ref(self, ref: HardwareSpecRef) -> HardwareSpec:
        row = self._connection.execute(
            "SELECT value FROM metadata WHERE key = ?", (_hardware_metadata_key(ref),)
        ).fetchone()
        if row is not None:
            try:
                return hardware_from_json(str(row["value"]))
            except (HardwareSpecError, ValueError) as exc:
                raise ProfileStoreError("stored hardware document is invalid") from exc
        # The M3 standalone catalog is B200-only.  This fallback keeps stores
        # created from a bare ref exportable for the calibrated device.
        try:
            from ..hardware.registry import b200_hardware_catalog

            return b200_hardware_catalog().resolve(ref)
        except Exception as exc:
            raise ProfileStoreError("hardware document is unavailable for snapshot export") from exc

    def _remember_hardware_in_transaction(self, hardware: HardwareSpec) -> None:
        canonical = hardware.to_json()
        row = self._connection.execute(
            "SELECT value FROM metadata WHERE key = ?",
            (_hardware_metadata_key(hardware.ref),),
        ).fetchone()
        if row is not None and str(row["value"]) != canonical:
            raise ProfileConflictError("hardware reference is already bound to different facts")
        if row is None:
            self._connection.execute(
                "INSERT INTO metadata(key, value) VALUES (?, ?)",
                (_hardware_metadata_key(hardware.ref), canonical),
            )

    def _logical_snapshot_json(self, ref: ProfileSnapshotRef) -> str:
        snapshot_ref = _validate_ref(ref)
        row = self._snapshot_row(snapshot_ref)
        members = self._connection.execute(
            "SELECT sm.key_id, sm.measurement_id, m.environment_id, m.aggregate_json "
            "FROM snapshot_measurements sm JOIN measurements m ON m.measurement_id = sm.measurement_id "
            "WHERE sm.snapshot_id = ? AND sm.snapshot_revision = ? ORDER BY sm.key_id",
            (snapshot_ref.snapshot_id, snapshot_ref.revision),
        ).fetchall()
        payload = {
            "snapshot_id": snapshot_ref.snapshot_id,
            "revision": snapshot_ref.revision,
            "state": str(row["state"]),
            "hardware_ref": _loads(str(row["hardware_ref_json"]), ProfileStoreError),
            "environment_id": row["environment_id"],
            "members": [
                {
                    "key_id": str(item["key_id"]),
                    "measurement_id": str(item["measurement_id"]),
                    "environment_id": str(item["environment_id"]),
                    "aggregate": _loads(str(item["aggregate_json"]), ProfileStoreError),
                }
                for item in members
            ],
        }
        return _canonical_json(payload)

    def _begin(self) -> None:
        try:
            self._connection.execute("BEGIN IMMEDIATE")
        except sqlite3.Error as exc:
            raise ProfileStoreError(f"cannot begin profile-store transaction: {exc}") from exc

    def _commit(self) -> None:
        try:
            self._connection.commit()
        except sqlite3.Error as exc:
            raise ProfileStoreError(f"cannot commit profile-store transaction: {exc}") from exc

    def _rollback(self) -> None:
        try:
            self._connection.rollback()
        except sqlite3.Error:
            pass

    def _ensure_writable(self) -> None:
        if not self._writable:
            raise ProfileStoreError("profile store is read-only")


def _hardware_metadata_key(ref: HardwareSpecRef) -> str:
    return "hardware:" + _canonical_json(_ref_payload(ref))


def _connect(path: Path, *, writable: bool) -> sqlite3.Connection:
    if writable:
        try:
            connection = sqlite3.connect(str(path), isolation_level=None)
        except sqlite3.Error as exc:
            raise ProfileStoreError(f"cannot open profile database: {exc}") from exc
    else:
        if not path.exists():
            raise ProfileStoreError(f"profile database does not exist: {path}")
        uri = path.resolve().as_uri() + "?mode=ro"
        try:
            connection = sqlite3.connect(uri, uri=True, isolation_level=None)
        except sqlite3.Error as exc:
            raise ProfileStoreError(f"cannot open read-only profile database: {exc}") from exc
    connection.row_factory = sqlite3.Row
    try:
        connection.execute("PRAGMA foreign_keys = ON")
        if writable:
            journal = connection.execute("PRAGMA journal_mode = WAL").fetchone()
            if journal is None or str(journal[0]).lower() != "wal":
                raise ProfileStoreError("profile store could not enable WAL mode")
            connection.execute("PRAGMA synchronous = FULL")
            synchronous = connection.execute("PRAGMA synchronous").fetchone()
            if synchronous is None or int(synchronous[0]) != 2:
                raise ProfileStoreError("profile store could not enable FULL synchronous mode")
            _migrate(connection)
        else:
            _validate_existing_schema(connection)
    except Exception:
        connection.close()
        raise
    return connection


def _migrate(connection: sqlite3.Connection) -> None:
    try:
        connection.execute("BEGIN IMMEDIATE")
        connection.execute(
            "CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        row = connection.execute(
            "SELECT value FROM metadata WHERE key = 'store_schema_version'"
        ).fetchone()
        if row is not None and str(row[0]) != str(STORE_SCHEMA_VERSION):
            raise ProfileStoreError(f"unsupported profile-store schema version: {row[0]!r}")
        for statement in _SCHEMA_SQL.split(";\n"):
            statement = statement.strip()
            if statement:
                connection.execute(statement)
        connection.execute(
            "INSERT INTO metadata(key, value) VALUES ('store_schema_version', ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (str(STORE_SCHEMA_VERSION),),
        )
        _validate_existing_schema(connection)
        connection.commit()
    except Exception as exc:
        connection.rollback()
        if isinstance(exc, ProfileStoreError):
            raise
        if isinstance(exc, sqlite3.Error):
            raise ProfileStoreError(f"cannot migrate profile database: {exc}") from exc
        raise


def _validate_existing_schema(connection: sqlite3.Connection) -> None:
    try:
        rows = connection.execute(
            "SELECT name, sql FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
        table_sql = {
            str(row["name"]): str(row["sql"])
            for row in rows
            if not str(row["name"]).startswith("sqlite_")
        }
        if set(table_sql) != set(_TABLES):
            raise ProfileStoreError("profile database table set does not match the exact schema")
        for table in _TABLES:
            _validate_table_schema(connection, table, table_sql[table])

        row = connection.execute(
            "SELECT value FROM metadata WHERE key = 'store_schema_version'"
        ).fetchone()
        if row is None or str(row[0]) != str(STORE_SCHEMA_VERSION):
            raise ProfileStoreError("unsupported or missing profile-store schema version")
        foreign_keys = connection.execute("PRAGMA foreign_keys").fetchone()
        if foreign_keys is None or int(foreign_keys[0]) != 1:
            raise ProfileStoreError("profile store must enforce foreign keys")
        integrity = connection.execute("PRAGMA integrity_check").fetchall()
        if [str(item[0]) for item in integrity] != ["ok"]:
            raise ProfileStoreError("profile database integrity check failed")
        if connection.execute("PRAGMA foreign_key_check").fetchall():
            raise ProfileStoreError("profile database contains foreign-key violations")
    except sqlite3.Error as exc:
        raise ProfileStoreError(f"cannot validate profile database schema: {exc}") from exc


def _validate_table_schema(connection: sqlite3.Connection, table: str, create_sql: str) -> None:
    if _normalize_schema_sql(create_sql) != _expected_table_sql(table):
        raise ProfileStoreError(f"profile table {table!r} does not match the exact SQL schema")
    columns = tuple(
        (
            str(row["name"]),
            str(row["type"]).upper(),
            int(row["notnull"]),
            int(row["pk"]),
        )
        for row in connection.execute(f"PRAGMA table_info({table})").fetchall()
    )
    if columns != _EXPECTED_COLUMNS[table]:
        raise ProfileStoreError(f"profile table {table!r} has an invalid column layout")
    defaults = connection.execute(f"PRAGMA table_info({table})").fetchall()
    if any(row["dflt_value"] is not None for row in defaults):
        raise ProfileStoreError(f"profile table {table!r} has unexpected column defaults")

    unique_indexes: set[tuple[str, ...]] = set()
    for row in connection.execute(f"PRAGMA index_list({table})").fetchall():
        if int(row["unique"]) != 1:
            continue
        index_name = str(row["name"]).replace("'", "''")
        index_columns = tuple(
            str(item["name"])
            for item in connection.execute(f"PRAGMA index_info('{index_name}')").fetchall()
        )
        unique_indexes.add(index_columns)
    if unique_indexes != _EXPECTED_UNIQUE_INDEXES[table]:
        raise ProfileStoreError(f"profile table {table!r} has invalid unique constraints")

    grouped_foreign_keys: dict[int, list[tuple[int, _ForeignKeyRow]]] = defaultdict(list)
    for row in connection.execute(f"PRAGMA foreign_key_list({table})").fetchall():
        grouped_foreign_keys[int(row["id"])].append(
            (
                int(row["seq"]),
                (
                    str(row["table"]),
                    str(row["from"]),
                    str(row["to"]),
                    str(row["on_update"]),
                    str(row["on_delete"]),
                    str(row["match"]),
                ),
            )
        )
    foreign_keys = {
        tuple(item for _sequence, item in sorted(group)) for group in grouped_foreign_keys.values()
    }
    if foreign_keys != _EXPECTED_FOREIGN_KEYS[table]:
        raise ProfileStoreError(f"profile table {table!r} has invalid foreign keys")

    expected_check = _EXPECTED_CHECKS.get(table)
    normalized_sql = _normalize_schema_sql(create_sql)
    if expected_check is not None and expected_check not in normalized_sql:
        raise ProfileStoreError(f"profile table {table!r} is missing its CHECK constraint")


def _normalize_schema_sql(value: str) -> str:
    return "".join(value.lower().split()).replace('"', "").replace("`", "")


def _expected_table_sql(table: str) -> str:
    for statement in _SCHEMA_SQL.split(";\n"):
        normalized = _normalize_schema_sql(statement.strip()).replace(
            "createtableifnotexists", "createtable", 1
        )
        if normalized.startswith(f"createtable{table}("):
            return normalized
    raise ProfileStoreError(f"internal profile schema is missing table {table!r}")


def open_profile_store(path: Path, *, writable: bool) -> SqliteProfileStore:
    """Open and validate one profile database."""

    if not isinstance(path, Path):
        path = Path(path)
    if writable:
        path.parent.mkdir(parents=True, exist_ok=True)
    return SqliteProfileStore(_connect(path, writable=writable), writable=writable)


__all__ = ["STORE_SCHEMA_VERSION", "SqliteProfileStore", "open_profile_store"]
