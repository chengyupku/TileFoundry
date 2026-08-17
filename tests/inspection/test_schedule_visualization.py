"""Small generic contract tests for warpgroup schedule visualization."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import cast

import pytest

from tilefoundry.cli import main
from tilefoundry.inspection.schedule import render_warpgroup_schedule_html
from tilefoundry.schedule.warpgroup import warpgroup_schedule_from_json


def _schedule_text() -> str:
    return json.dumps(
        {
            "format": "tilefoundry.warpgroup_schedule",
            "lanes": [["load_a", "compute_b"], ["publish_c"]],
            "sync": [
                {"after": "load_a", "before": "compute_b", "distance": 0},
                {"after": "publish_c", "before": "load_a", "distance": 1},
            ],
            "times": [
                [0, "load_a", 100, 102, 110],
                [0, "compute_b", 103, 105, 105],
                [0, "publish_c", 106, 107, 115],
                [1, "load_a", 200, 202, 210],
                [1, "compute_b", 203, 205, 205],
                [1, "publish_c", 206, 207, 215],
            ],
        }
    )


def _embedded_data(document: str) -> dict[str, object]:
    prefix = "    const DATA = "
    start = document.index(prefix) + len(prefix)
    end = document.index(";\n    const svg", start)
    return cast(dict[str, object], json.loads(document[start:end]))


def test_render_generic_timing() -> None:
    schedule = warpgroup_schedule_from_json(_schedule_text())
    document = render_warpgroup_schedule_html(schedule, title="Example schedule")

    rows = cast(list[list[object]], _embedded_data(document)["times"])
    assert next(row for row in rows if row[0] == 0 and row[1] == "load_a") == [
        0,
        "load_a",
        100,
        102,
        110,
    ]
    assert "issue interval" in document
    assert "completion tail" in document
    assert "distance-0 sync" in document
    assert "distance-1 sync" in document
    assert "operation family" not in document


def test_html_is_deterministic_escapes_title_and_has_no_invalid_text() -> None:
    schedule = warpgroup_schedule_from_json(_schedule_text())
    title = 'A <schedule> & "timing"'
    first = render_warpgroup_schedule_html(schedule, title=title)
    second = render_warpgroup_schedule_html(schedule, title=title)

    assert first == second
    assert "A &lt;schedule&gt; &amp; &quot;timing&quot;" in first
    assert "A <schedule>" not in first
    assert "NaN" not in first
    assert "undefined" not in first


def test_visualize_cli_writes_file_and_stdout(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = tmp_path / "schedule.json"
    output = tmp_path / "schedule.html"
    source.write_text(_schedule_text(), encoding="utf-8")

    assert main(["visualize", str(source), "--out", str(output)]) == 0
    assert output.is_file()
    assert '<svg id="chart"' in output.read_text(encoding="utf-8")
    assert f"wrote {output}" in capsys.readouterr().out

    assert main(["visualize", str(source), "--out", "-"]) == 0
    assert '<svg id="chart"' in capsys.readouterr().out


def test_visualize_cli_rejects_stdout_and_open_together(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = tmp_path / "schedule.json"
    source.write_text(_schedule_text(), encoding="utf-8")

    assert main(["visualize", str(source), "--out", "-", "--open"]) == 1
    assert "--out - cannot be combined with --open" in capsys.readouterr().err


def test_visualize_cli_reports_malformed_schedule(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = tmp_path / "malformed.json"
    source.write_text("{not json}", encoding="utf-8")

    assert main(["visualize", str(source)]) == 1
    assert "tilefoundry visualize: error:" in capsys.readouterr().err


def test_cli_import_isolation() -> None:
    root = Path(__file__).parents[2]
    script = """
import sys
import tilefoundry.cli
assert 'tilefoundry.inspection.schedule' not in sys.modules
assert 'tilefoundry.schedule.warpgroup.serialization' not in sys.modules
assert 'ortools' not in sys.modules
print('isolated')
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "isolated"
