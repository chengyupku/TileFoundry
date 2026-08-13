"""Command-line boundary for ``tilefoundry-costmodel``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from . import (
    CostModelError,
    EvaluationStatus,
    InvalidRequestError,
    MissingProfileError,
    ProfileRunError,
    UnsupportedError,
    __version__,
    hardware_from_json,
    problem_from_json,
    request_from_json,
    result_to_json,
    solve,
)
from .build import ConfigurationBuilder
from .hardware.registry import b200_hardware_catalog
from .implementations import b200_implementation_catalog
from .profiler.base import MeasurementPolicy
from .profiles.resolver import BenchmarkProviderCatalog, ProfileResolver
from .profiles.store import open_profile_store
from .request import ProfileMode, ProfileSnapshotRef


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tilefoundry-costmodel", description="TileFoundry B200 cost model"
    )
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command")

    build_parser = sub.add_parser("build", help="build a replayable search problem")
    _request_profiles_output(build_parser, require_output=True)

    solve_parser = sub.add_parser("solve", help="solve a replayable search problem")
    solve_parser.add_argument("--problem", required=True, type=Path)
    solve_parser.add_argument("--output", required=True, type=Path)

    search_parser = sub.add_parser("search", help="build and solve a request")
    _request_profiles_output(search_parser, require_output=True)

    profile_parser = sub.add_parser("profile", help="measure missing profiles on a local B200")
    _request_profiles_output(profile_parser, require_output=False)

    profiles_parser = sub.add_parser("profiles", help="inspect profile snapshots")
    profiles_sub = profiles_parser.add_subparsers(dest="profiles_command")
    create_parser = profiles_sub.add_parser("create")
    create_parser.add_argument("--profiles", required=True, type=Path)
    create_parser.add_argument("--snapshot", required=True)
    create_parser.add_argument(
        "--hardware", "--hardware-json", dest="hardware", required=True, type=Path
    )
    create_parser.add_argument("--description", default="")
    create_parser.add_argument("--base")
    freeze_parser = profiles_sub.add_parser("freeze")
    freeze_parser.add_argument("--profiles", required=True, type=Path)
    freeze_parser.add_argument("--snapshot", required=True)
    export_parser = profiles_sub.add_parser("export")
    export_parser.add_argument("--profiles", required=True, type=Path)
    export_parser.add_argument("--snapshot", required=True)
    export_parser.add_argument("--output", required=True, type=Path)
    import_parser = profiles_sub.add_parser("import")
    import_parser.add_argument("--profiles", required=True, type=Path)
    import_parser.add_argument("snapshot", type=Path)
    inspect_parser = profiles_sub.add_parser("inspect")
    inspect_parser.add_argument("--profiles", required=True, type=Path)
    inspect_parser.add_argument("--snapshot", required=True)
    return parser


def _request_profiles_output(parser: argparse.ArgumentParser, *, require_output: bool) -> None:
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--profiles", required=True, type=Path)
    parser.add_argument("--output", required=require_output, type=Path)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command is None:
        _parser().print_help()
        return 0
    try:
        if args.command == "solve":
            return _run_solve(args)
        if args.command in {"build", "search", "profile"}:
            return _run_request_command(args)
        if args.command == "profiles":
            return _run_profiles(args)
        raise InvalidRequestError(f"unknown command: {args.command}")
    except MissingProfileError as exc:
        print(str(exc), file=sys.stderr)
        return 3
    except UnsupportedError as exc:
        print(str(exc), file=sys.stderr)
        return 3
    except ProfileRunError as exc:
        print(str(exc), file=sys.stderr)
        return 4
    except (InvalidRequestError, CostModelError, OSError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2


def _run_solve(args: argparse.Namespace) -> int:
    problem = problem_from_json(args.problem.read_text(encoding="utf-8"))
    result = solve(problem)
    args.output.write_text(result_to_json(result), encoding="utf-8")
    if result.status in (EvaluationStatus.OPTIMAL, EvaluationStatus.FEASIBLE):
        return 0
    if result.status in (EvaluationStatus.UNSUPPORTED, EvaluationStatus.MISSING_PROFILE):
        return 3
    if result.status is EvaluationStatus.PROFILE_FAILED:
        return 4
    if result.status is EvaluationStatus.INFEASIBLE:
        return 5
    if result.status is EvaluationStatus.TIMEOUT:
        return 0 if result.plan is not None else 6
    raise CostModelError(f"unhandled result status: {result.status.value}")


def _run_request_command(args: argparse.Namespace) -> int:
    request = request_from_json(args.request.read_text(encoding="utf-8"))
    if args.command == "profile":
        return _run_profile_request(args, request)
    with open_profile_store(args.profiles, writable=False) as store:
        del store
        raise UnsupportedError("build/search require M3 profile resolution and M4 orchestration")


def _run_profiles(args: argparse.Namespace) -> int:
    command = args.profiles_command
    if command is None:
        raise InvalidRequestError("profiles requires create, inspect, freeze, export, or import")
    if command == "import":
        with open_profile_store(args.profiles, writable=True) as store:
            ref = store.import_snapshot(args.snapshot)
        print(f"{ref.snapshot_id}@{ref.revision}")
        return 0
    if command == "create":
        hardware = hardware_from_json(args.hardware.read_text(encoding="utf-8"))
        snapshot_id, revision = _parse_snapshot(args.snapshot)
        if revision is not None:
            raise InvalidRequestError("create snapshot must not include a revision")
        base_ref: ProfileSnapshotRef | None = None
        if args.base is not None:
            base_id, base_revision = _parse_snapshot(args.base)
            if base_revision is None:
                raise InvalidRequestError("base snapshot reference must be ID@REV")
            base_ref = ProfileSnapshotRef(base_id, base_revision)
        with open_profile_store(args.profiles, writable=True) as store:
            ref = store.create_snapshot(
                snapshot_id=snapshot_id,
                hardware=hardware,
                description=args.description,
                base=base_ref,
            )
        print(f"{ref.snapshot_id}@{ref.revision}")
        return 0
    snapshot_id, revision = _parse_snapshot(args.snapshot)
    if revision is None:
        raise InvalidRequestError("snapshot reference must be ID@REV")
    ref = ProfileSnapshotRef(snapshot_id, revision)
    if command == "freeze":
        with open_profile_store(args.profiles, writable=True) as store:
            store.freeze(ref)
        print(f"{ref.snapshot_id}@{ref.revision}")
        return 0
    if command == "inspect":
        with open_profile_store(args.profiles, writable=False) as store:
            print(store.logical_snapshot_json(ref))
        return 0
    if command == "export":
        with open_profile_store(args.profiles, writable=False) as store:
            store.export_snapshot(ref, args.output)
        return 0
    raise InvalidRequestError(f"unknown profiles command: {command}")


def _run_profile_request(args: argparse.Namespace, request: object) -> int:
    from .request import CostModelRequest

    if type(request) is not CostModelRequest:
        raise InvalidRequestError("profile request must be CostModelRequest")
    if request.profiles.mode is not ProfileMode.JIT_ON_MISS:
        raise InvalidRequestError("profile requires JIT-on-miss mode")
    from .profiler.cuda import LocalCudaProfileRunner

    hardware = b200_hardware_catalog().resolve(request.hardware)
    implementation_catalog = b200_implementation_catalog()
    templates = ConfigurationBuilder(implementations=implementation_catalog).enumerate_templates(
        request.programs,
        search_space=request.search_space,
        hardware=hardware,
    )
    requirements = tuple(phase.profile for template in templates for phase in template.phases)
    providers = BenchmarkProviderCatalog(
        tuple(pair.benchmark_provider for pair in implementation_catalog.implementations)
    )
    with open_profile_store(args.profiles, writable=True) as store:
        resolver = ProfileResolver(
            store=store,
            providers=providers,
            runner=LocalCudaProfileRunner(),
            measurement_policy=MeasurementPolicy(),
        )
        timings = resolver.resolve_many(
            requirements,
            hardware=hardware,
            selection=request.profiles,
        )
    print(len(timings))
    return 0


def _parse_snapshot(value: str) -> tuple[str, int | None]:
    if not isinstance(value, str) or not value:
        raise InvalidRequestError("snapshot must be non-empty ID or ID@REV")
    if "@" not in value:
        return value, None
    snapshot_id, revision_text = value.rsplit("@", 1)
    try:
        revision = int(revision_text)
    except ValueError as exc:
        raise InvalidRequestError("snapshot revision must be an integer") from exc
    return snapshot_id, revision


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["main"]
