"""Command-line interface for authored TileFoundry HIR analysis."""

from __future__ import annotations

import argparse
import sys
from importlib import import_module
from typing import Never, Sequence

_ANALYSES = ("compute-cost", "memory", "roofline", "timeline")

_PUBLIC = {
    "load_authored_ir": ("tilefoundry.cli.source", "load_authored_ir"),
    "one_extent_per_dim": ("tilefoundry.cli.source", "one_extent_per_dim"),
    "parse_dims": ("tilefoundry.cli.source", "parse_dims"),
    "read_spec": ("tilefoundry.cli.spec", "read_spec"),
    "spec_path": ("tilefoundry.cli.spec", "spec_path"),
}


def __getattr__(name: str) -> object:
    """Resolve established CLI helpers without importing command implementations."""
    entry = _PUBLIC.get(name)
    if entry is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = entry
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value


#: Every command and its one-line description, in the order an agent meets them
#: rather than alphabetically -- the order itself is meant to read as the workflow.
#: One table, so the parser and the overview cannot describe different surfaces.
_COMMANDS = {
    "models": "list the described models, or show one of them",
    "spec": "read one specification: its sections, or one of them",
    "tutorial": "learn the two-step workflow: its pages, or one of them",
    "check": "compare an implementation against its reference, output by output",
    "analyze": "type-check and analyze authored HIR",
    "schedule": "schedule authored HIR or an explicit warpgroup JSON document",
    "inspect": "inspect installed target facts",
}

_INSPECT_COMMANDS = {
    "capabilities": (
        "the facts a selection's target was composed from, or the installed "
        "hardware documents there are"
    ),
}


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> Never:
        self._print_message(f"{self.prog}: error: {message}\n\n", sys.stderr)
        self.print_help(sys.stderr)
        self.exit(2)


def _project_summary() -> str:
    """The packaged one-line description of the project.

    Read from installed metadata rather than restated here, so there is one copy
    of the sentence and no second one to drift.
    """
    from importlib.metadata import metadata  # noqa: PLC0415

    return metadata("tilefoundry")["Summary"].rstrip(".")


def overview() -> str:
    """What a bare invocation prints: what this is, and how to ask it something."""
    width = max(len(name) for name in _COMMANDS)
    commands = "\n".join(
        f"  {name:<{width}}  {description}" for name, description in _COMMANDS.items()
    )
    return (
        f"TileFoundry — {_project_summary()}\n"
        f"\n"
        f"Usage:\n"
        f"  tilefoundry <command> [options]\n"
        f"\n"
        f"Common commands:\n"
        f"{commands}\n"
        f"\n"
        f"Options:\n"
        f"  -h, --help  print this, or a command's own help after the command\n"
    )


def _inspect_overview() -> str:
    """What ``tilefoundry inspect`` prints without a subcommand."""
    width = max(len(name) for name in _INSPECT_COMMANDS)
    commands = "\n".join(
        f"  {name:<{width}}  {description}" for name, description in _INSPECT_COMMANDS.items()
    )
    return (
        f"tilefoundry inspect — {_COMMANDS['inspect']}\n"
        f"\n"
        f"Usage:\n"
        f"  tilefoundry inspect <command> [options]\n"
        f"\n"
        f"Commands:\n"
        f"{commands}\n"
        f"\n"
        f"Options:\n"
        f"  -h, --help  print this, or a command's own help after the command\n"
    )


def _add_source_argument(parser: argparse.ArgumentParser, *, optional: bool = False) -> None:
    help_text = "model.py[:Module[.child_module...][.function]]"
    if optional:
        parser.add_argument("source", nargs="?", metavar="SOURCE", help=help_text)
    else:
        parser.add_argument("source", metavar="SOURCE", help=help_text)


def build_parser(*, _selected_command: str | None = None) -> argparse.ArgumentParser:
    from tilefoundry.cli.tutorial import PAGES  # noqa: PLC0415

    parser = _Parser(prog="tilefoundry")
    # Not required: naming no command is how the overview is asked for.
    commands = parser.add_subparsers(dest="command", parser_class=_Parser)

    models = commands.add_parser("models", help=_COMMANDS["models"])
    models.add_argument(
        "name",
        nargs="?",
        metavar="NAME",
        help="which model; with none, list the models there are",
    )
    models.add_argument(
        "--source", action="store_true", help="print the model's authored source instead"
    )

    tutorial = commands.add_parser("tutorial", help=_COMMANDS["tutorial"])
    tutorial.add_argument(
        "page",
        nargs="?",
        choices=(*PAGES[1:], "orchestrator"),
        metavar="PAGE",
        help="which page; with none, the overview and the pages there are",
    )
    tutorial.add_argument(
        "family",
        nargs="?",
        metavar="FAMILY",
        help="which orchestrator family to show",
    )

    spec = commands.add_parser("spec", help=_COMMANDS["spec"])
    spec.add_argument(
        "topic",
        nargs="?",
        metavar="TOPIC",
        help="which document; with none, list the documents there are",
    )
    spec.add_argument(
        "section",
        nargs="?",
        metavar="SECTION",
        help="one section's key, as the outline prints it; with none, print the outline",
    )

    if _selected_command in (None, "check"):
        from tilefoundry.cli.check import add_arguments as add_check_arguments  # noqa: PLC0415
        from tilefoundry.cli.check import guidance  # noqa: PLC0415

        check = commands.add_parser(
            "check",
            help=_COMMANDS["check"],
            epilog=guidance(),
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )
        add_check_arguments(check)
    else:
        commands.add_parser("check", help=_COMMANDS["check"])

    analyze = commands.add_parser("analyze", help=_COMMANDS["analyze"])
    _add_source_argument(analyze)
    for analysis in _ANALYSES:
        analyze.add_argument(
            f"--{analysis}", action="store_true", help=f"run the {analysis} analysis"
        )
    analyze.add_argument(
        "--dim",
        action="append",
        metavar="NAME=EXTENT",
        help="bind one dimension the model left open, for example ctx_len=1024",
    )
    analyze.add_argument(
        "--json", action="store_true", help="print the report as JSON instead of text"
    )

    schedule = commands.add_parser("schedule", help=_COMMANDS["schedule"])
    _add_source_argument(schedule, optional=True)
    schedule.add_argument(
        "--topology",
        metavar="LEVEL",
        help="declared topology level to schedule (for example cta)",
    )
    warpgroup = schedule.add_mutually_exclusive_group()
    warpgroup.add_argument(
        "--warpgroup-program",
        metavar="PROGRAM.json",
        help="schedule strict authored warpgroup JSON with explicitly selected fixture costs",
    )
    warpgroup.add_argument(
        "--warpgroup-problem",
        metavar="PROBLEM.json",
        help="schedule one strict closed warpgroup problem JSON",
    )
    schedule.add_argument(
        "--fixture-costs",
        action="store_true",
        help="use illustrative fixture costs, not B200 calibration, for a warpgroup program",
    )
    schedule.add_argument(
        "--dim",
        action="append",
        metavar="NAME=EXTENT",
        help="bind one dimension the model left open, for example ctx_len=1024",
    )
    schedule.add_argument("--json", action="store_true", help="print the selected plan as JSON")
    schedule.add_argument(
        "--solver-timeout",
        type=float,
        metavar="SECONDS",
        help="how long the solver may search before it reports no answer",
    )
    schedule.add_argument(
        "--solver-workers",
        type=int,
        metavar="COUNT",
        help=(
            "how many search workers the solver may use; the default lets it "
            "size itself to the machine, which oversubscribes when several "
            "schedules run at once"
        ),
    )
    schedule.add_argument(
        "--first-plan",
        action="store_true",
        help="stop at the first plan that satisfies the constraints instead of "
        "searching the whole budget for the best one",
    )

    inspect = commands.add_parser("inspect", help=_COMMANDS["inspect"])
    inspect_commands = inspect.add_subparsers(dest="inspect_command", parser_class=_Parser)
    capabilities = inspect_commands.add_parser(
        "capabilities", help=_INSPECT_COMMANDS["capabilities"]
    )
    _add_source_argument(capabilities, optional=True)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = tuple(sys.argv[1:] if argv is None else argv)
    selected_command = arguments[0] if arguments else None
    parser = build_parser(_selected_command=selected_command)
    args = parser.parse_args(arguments)
    if args.command is None:
        sys.stdout.write(overview())
        return 0
    if args.command == "models":
        try:
            from tilefoundry.cli.models import run_models  # noqa: PLC0415

            return run_models(args.name, source=args.source)
        except (OSError, ValueError) as error:
            print(f"tilefoundry: error: {error}", file=sys.stderr)
            return 1
    if args.command == "spec":
        try:
            from tilefoundry.cli.spec import run_spec  # noqa: PLC0415

            return run_spec(args.topic, args.section)
        except (OSError, ValueError) as error:
            print(f"tilefoundry: error: {error}", file=sys.stderr)
            return 1
    if args.command == "tutorial":
        try:
            from tilefoundry.cli.tutorial import run_tutorial  # noqa: PLC0415

            return run_tutorial(args.page, args.family)
        except (OSError, ValueError) as error:
            print(f"tilefoundry: error: {error}", file=sys.stderr)
            return 1
    if args.command == "check":
        try:
            from tilefoundry.cli.check import run_check  # noqa: PLC0415

            return run_check(args)
        except Exception as error:
            print(f"tilefoundry check: error: {error}", file=sys.stderr)
            return 1
    if args.command == "inspect":
        if args.inspect_command is None:
            sys.stdout.write(_inspect_overview())
            return 0
        try:
            from tilefoundry.cli.inspect import run_capabilities  # noqa: PLC0415

            return run_capabilities(args.source)
        except Exception as error:
            print(f"tilefoundry: error: {error}", file=sys.stderr)
            return 1
    if args.command == "schedule":
        warpgroup_path = args.warpgroup_program or args.warpgroup_problem
        if args.source is not None and warpgroup_path is not None:
            print(
                "tilefoundry: error: HIR SOURCE and warpgroup JSON paths are mutually exclusive",
                file=sys.stderr,
            )
            return 1
        if warpgroup_path is not None:
            try:
                from tilefoundry.cli.schedule import run_warpgroup_schedule  # noqa: PLC0415
                from tilefoundry.schedule.warpgroup.errors import (  # noqa: PLC0415
                    WarpgroupSolveError,
                )

                if args.topology is not None:
                    raise ValueError("--topology is only valid with an authored HIR SOURCE")
                if args.dim or args.solver_workers is not None or args.first_plan:
                    raise ValueError(
                        "--dim, --solver-workers, and --first-plan are only valid with HIR SOURCE"
                    )
                return run_warpgroup_schedule(
                    warpgroup_path,
                    is_program=args.warpgroup_program is not None,
                    fixture_costs=args.fixture_costs,
                    as_json=args.json,
                    solver_timeout=args.solver_timeout,
                )
            except (WarpgroupSolveError, OSError, TypeError, ValueError) as error:
                print(f"tilefoundry: error: {error}", file=sys.stderr)
                return 1
        if args.source is None:
            parser.error("schedule requires SOURCE, --warpgroup-program, or --warpgroup-problem")
        if args.topology is None:
            parser.error("the following arguments are required: --topology")
        if args.fixture_costs:
            print(
                "tilefoundry: error: --fixture-costs is only valid with --warpgroup-program",
                file=sys.stderr,
            )
            return 1
        try:
            from tilefoundry.analysis.poly import ExtractError  # noqa: PLC0415
            from tilefoundry.cli.schedule import run_schedule  # noqa: PLC0415
            from tilefoundry.cli.source import (  # noqa: PLC0415
                one_extent_per_dim,
                parse_dims,
            )
            from tilefoundry.schedule.errors import ScheduleError  # noqa: PLC0415
        except Exception as error:
            print(f"tilefoundry: error: {error}", file=sys.stderr)
            return 1
        try:
            return run_schedule(
                args.source,
                args.topology,
                as_json=args.json,
                dims=one_extent_per_dim(parse_dims(args.dim)),
                solver_timeout=args.solver_timeout,
                solver_workers=args.solver_workers,
                first_plan=args.first_plan,
            )
        except (ExtractError, ScheduleError, OSError, TypeError, ValueError) as error:
            print(f"tilefoundry: error: {error}", file=sys.stderr)
            return 1

    analyses = tuple(name for name in _ANALYSES if getattr(args, name.replace("-", "_")))
    if not analyses:
        analyses = _ANALYSES
    try:
        from tilefoundry.analysis.errors import AnalysisError  # noqa: PLC0415
        from tilefoundry.cli.analyze import run_authored_analysis  # noqa: PLC0415
        from tilefoundry.cli.source import one_extent_per_dim, parse_dims  # noqa: PLC0415
        from tilefoundry.ir.core.errors import VerifyError  # noqa: PLC0415

        return run_authored_analysis(
            args.source, analyses, as_json=args.json, dims=one_extent_per_dim(parse_dims(args.dim))
        )
    except (AnalysisError, VerifyError, OSError, TypeError, ValueError) as error:
        print(f"tilefoundry: error: {error}", file=sys.stderr)
        return 1


__all__ = [
    "build_parser",
    "load_authored_ir",
    "main",
    "overview",
    "parse_dims",
    "read_spec",
    "spec_path",
]


if __name__ == "__main__":
    raise SystemExit(main())
