# TileFoundry

TileFoundry searches warpgroup pipeline schedules from typed tile operations.
It models SSA dependencies, fixed warpgroup ownership, asynchronous
issue/completion timing, shared-memory handoffs and reuse, finite resources,
and compact periodic loop bodies. CP-SAT produces a timed schedule; an
independent verifier checks the exported synchronization contract.

## Install

```bash
python -m pip install -e '.[test]'
```

Python 3.12 or newer and OR-Tools 9.15 or newer are required.

## Search

The retained MLA example contains one semantic program, one hardware cost
description, and one independently verified schedule:

```bash
tilefoundry schedule \
  --program examples/mla-schedule/program.json \
  --hardware examples/mla-schedule/hardware.json \
  --solver-timeout 60 \
  --json > /tmp/mla-schedule.json
```

## Visualize

```bash
tilefoundry visualize \
  /tmp/mla-schedule.json \
  --out /tmp/mla-schedule.html \
  --open
```

The generated HTML is standalone and contains iteration filters, operation
search, dependency overlays, zoom controls, and issue/completion timing.

## Documents

- [Scheduling contract](docs/spec/schedule.md)
- [CLI contract](docs/spec/cli.md)
- [TileProf integration boundary](docs/tileprof-integration.md)
- [MLA example](examples/mla-schedule/README.md)
