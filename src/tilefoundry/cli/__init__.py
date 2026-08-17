"""Command-line interface for warpgroup scheduling."""

from __future__ import annotations

import argparse
import sys
from typing import Never, Sequence


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> Never:
        self._print_message(f"{self.prog}: error: {message}\n\n", sys.stderr)
        self.print_help(sys.stderr)
        self.exit(2)


def build_parser() -> argparse.ArgumentParser:
    """Build the complete command parser without importing solver internals."""
    parser = _Parser(prog="tilefoundry")
    commands = parser.add_subparsers(dest="command", parser_class=_Parser)

    schedule = commands.add_parser("schedule", help="search a warpgroup schedule")
    source = schedule.add_mutually_exclusive_group(required=True)
    source.add_argument("--program", metavar="PROGRAM.json", help="typed warpgroup program")
    source.add_argument("--problem", metavar="PROBLEM.json", help="closed numeric problem")
    schedule.add_argument(
        "--fixture-costs",
        action="store_true",
        help="use illustrative unit costs for a program",
    )
    schedule.add_argument("--json", action="store_true", help="write schedule JSON")
    schedule.add_argument(
        "--solver-timeout",
        type=float,
        default=60.0,
        metavar="SECONDS",
        help="CP-SAT search budget (default: 60)",
    )

    visualize = commands.add_parser("visualize", help="render a schedule as HTML")
    visualize.add_argument("schedule", metavar="SCHEDULE.json", help="warpgroup schedule JSON")
    visualize.add_argument("--out", metavar="OUTPUT.html", help="output path, or '-' for stdout")
    visualize.add_argument("--title", help="diagram title")
    visualize.add_argument("--open", action="store_true", dest="open_browser")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the selected command."""
    parser = build_parser()
    args = parser.parse_args(tuple(sys.argv[1:] if argv is None else argv))
    if args.command is None:
        parser.print_help()
        return 0
    if args.command == "schedule":
        try:
            from tilefoundry.cli.schedule import run_warpgroup_schedule  # noqa: PLC0415
            from tilefoundry.schedule.warpgroup.errors import (  # noqa: PLC0415
                WarpgroupSolveError,
            )

            return run_warpgroup_schedule(
                args.program or args.problem,
                is_program=args.program is not None,
                fixture_costs=args.fixture_costs,
                as_json=args.json,
                solver_timeout=args.solver_timeout,
            )
        except (WarpgroupSolveError, OSError, TypeError, ValueError) as error:
            print(f"tilefoundry schedule: error: {error}", file=sys.stderr)
            return 1
    try:
        from tilefoundry.cli.visualize import run_visualize  # noqa: PLC0415

        return run_visualize(
            args.schedule,
            output=args.out,
            title=args.title,
            open_browser=args.open_browser,
        )
    except (OSError, TypeError, ValueError) as error:
        print(f"tilefoundry visualize: error: {error}", file=sys.stderr)
        return 1


__all__ = ["build_parser", "main"]
