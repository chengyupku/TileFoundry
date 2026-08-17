"""The warpgroup `schedule` command."""

from __future__ import annotations

import sys
from pathlib import Path

from tilefoundry.schedule.warpgroup import (
    WarpgroupScheduleResult,
    schedule_warpgroups,
    warpgroup_hardware_from_json,
    warpgroup_program_from_json,
    warpgroup_schedule_to_json,
)


def _render_warpgroup_result(result: WarpgroupScheduleResult) -> str:
    schedule = result.schedule
    times = {(timed.iteration, timed.operation_id): timed for timed in schedule.times}
    lines = [f"warpgroup schedule: {result.status} makespan={result.makespan}"]
    iterations = sorted({timed.iteration for timed in schedule.times})
    for lane_index, lane in enumerate(schedule.lanes):
        lines.append(f"lane {lane_index}: {' -> '.join(lane.operations) or '(empty)'}")
        for iteration in iterations:
            intervals = " ".join(
                (
                    f"{operation_id}[{times[(iteration, operation_id)].start},"
                    f"{times[(iteration, operation_id)].completion})"
                    if times[(iteration, operation_id)].issue_end
                    == times[(iteration, operation_id)].completion
                    else f"{operation_id}[{times[(iteration, operation_id)].start},"
                    f"{times[(iteration, operation_id)].issue_end}|"
                    f"{times[(iteration, operation_id)].completion})"
                )
                for operation_id in lane.operations
            )
            lines.append(f"  iteration {iteration}: {intervals or '(empty)'}")
    lines.append("sync:")
    lines.extend(
        f"  {edge.after} -> {edge.before} distance={edge.distance}" for edge in schedule.sync
    )
    if not schedule.sync:
        lines.append("  (none)")
    return "\n".join(lines)


def run_warpgroup_schedule(
    program_path: str,
    hardware_path: str,
    *,
    as_json: bool = False,
    solver_timeout: float | None = None,
) -> int:
    """Run program and hardware JSON through the shared typed workflow."""
    timeout = 60.0 if solver_timeout is None else solver_timeout
    program = warpgroup_program_from_json(Path(program_path).read_text(encoding="utf-8"))
    hardware = warpgroup_hardware_from_json(Path(hardware_path).read_text(encoding="utf-8"))
    result = schedule_warpgroups(program, hardware, timeout_seconds=timeout)
    output = (
        warpgroup_schedule_to_json(result.schedule) if as_json else _render_warpgroup_result(result)
    )
    sys.stdout.write(output + "\n")
    return 0


__all__ = ["run_warpgroup_schedule"]
