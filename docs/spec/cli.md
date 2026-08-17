# Command-Line Interface

TileFoundry exposes two commands. Importing `tilefoundry.cli` must not load
OR-Tools, the schedule serializer, or the HTML renderer.

## Schedule

```text
tilefoundry schedule (--program PROGRAM.json | --problem PROBLEM.json)
                     [--fixture-costs] [--solver-timeout SECONDS] [--json]
```

`--program` parses a typed semantic program. It requires `--fixture-costs`
until an external measured-cost adapter is selected by a future interface.
`--problem` parses a closed numeric problem and rejects `--fixture-costs`.

`--solver-timeout` is the shared CP-SAT deadline for makespan optimization, II
optimization, and deterministic tie-breaking. Model construction,
materialization, verification, and serialization are outside that deadline.

Without `--json`, the command prints status, makespan, lane order, timing rows,
and synchronization edges. With `--json`, stdout contains only the strict
schedule document.

## Visualize

```text
tilefoundry visualize SCHEDULE.json [--out OUTPUT.html] [--title TITLE] [--open]
```

The default output replaces the input suffix with `.html`. `--out -` writes
HTML to stdout and cannot be combined with `--open`.

## Exit Status

Both commands return zero on success and one for invalid documents, invalid
option combinations, I/O errors, or solve failures. Argument-parser errors use
exit status two.
