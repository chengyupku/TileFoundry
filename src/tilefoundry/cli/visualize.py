"""The ``tilefoundry visualize`` command."""

from __future__ import annotations

import sys
import webbrowser
from pathlib import Path


def run_visualize(
    path: str,
    *,
    output: str | None = None,
    title: str | None = None,
    open_browser: bool = False,
) -> int:
    """Render a warpgroup schedule JSON document to a standalone HTML file."""
    if output == "-" and open_browser:
        raise ValueError("--out - cannot be combined with --open")
    source = Path(path)
    from tilefoundry.inspection.schedule import (  # noqa: PLC0415
        render_warpgroup_schedule_html,
    )
    from tilefoundry.schedule.warpgroup import (  # noqa: PLC0415
        warpgroup_schedule_from_json,
    )

    schedule = warpgroup_schedule_from_json(source.read_text(encoding="utf-8"))
    page_title = title or source.stem.replace("_", " ")
    document = render_warpgroup_schedule_html(schedule, title=page_title)
    if output == "-":
        sys.stdout.write(document)
        return 0
    destination = Path(output) if output else source.with_suffix(".html")
    destination.write_text(document, encoding="utf-8")
    if open_browser:
        webbrowser.open(destination.resolve().as_uri())
    sys.stdout.write(f"wrote {destination}\n")
    return 0


__all__ = ["run_visualize"]
