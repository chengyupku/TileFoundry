"""The warpgroup `schedule` command."""

from __future__ import annotations

import sys
from pathlib import Path

from tilefoundry.schedule.warpgroup import (
    OperationCost,
    OperationCostEntry,
    OperationCostLibrary,
    WarpgroupProgram,
    WarpgroupScheduleResult,
    operation_signature,
    schedule_warpgroups,
    warpgroup_problem_from_json,
    warpgroup_program_from_json,
    warpgroup_schedule_to_json,
)


def _fixture_cost_library(program: WarpgroupProgram) -> OperationCostLibrary:
    """Build illustrative unit costs; these are design fixtures, not calibration."""
    signatures = tuple(
        sorted(
            {operation_signature(program, operation) for operation in program.loop.ops},
            key=lambda signature: signature.canonical_key,
        )
    )
    return OperationCostLibrary(
        "fixture_tick",
        (),
        tuple(OperationCostEntry(signature, OperationCost(1, ())) for signature in signatures),
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
                    f"{times[(iteration, operation_id)].end})"
                    if times[(iteration, operation_id)].issue_end
                    == times[(iteration, operation_id)].end
                    else f"{operation_id}[{times[(iteration, operation_id)].start},"
                    f"{times[(iteration, operation_id)].issue_end}|"
                    f"{times[(iteration, operation_id)].end})"
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
    path: str,
    *,
    is_program: bool,
    fixture_costs: bool,
    as_json: bool = False,
    solver_timeout: float | None = None,
) -> int:
    """Run one strict warpgroup JSON document through the shared typed workflow."""
    text = Path(path).read_text(encoding="utf-8")
    timeout = 60.0 if solver_timeout is None else solver_timeout
    if is_program:
        if not fixture_costs:
            raise ValueError(
                "--program requires --fixture-costs "
                "(illustrative fixture costs, not B200 calibration)"
            )
        program = warpgroup_program_from_json(text)
        result = schedule_warpgroups(
            program,
            _fixture_cost_library(program),
            timeout_seconds=timeout,
        )
    else:
        if fixture_costs:
            raise ValueError("--problem rejects --fixture-costs")
        result = schedule_warpgroups(
            warpgroup_problem_from_json(text),
            timeout_seconds=timeout,
        )
    output = (
        warpgroup_schedule_to_json(result.schedule) if as_json else _render_warpgroup_result(result)
    )
    sys.stdout.write(output + "\n")
    return 0


__all__ = ["run_warpgroup_schedule"]
