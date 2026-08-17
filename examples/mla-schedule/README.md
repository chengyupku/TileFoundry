# H200 MLA Schedule Search

This directory contains the complete three-document workflow for the central
loop of an MLA kernel. The program has 51 static operations, 16 requested
iterations, and fixed ownership across three warpgroups. Hardware costs are
exploratory H200 timing values rather than a production calibration.

Files:

- `program.json`: typed SSA semantics and fixed warpgroup ownership.
- `hardware.json`: 37 canonical operation signatures with H200 timing facts.
- `schedule.json`: independently verified witness with 816 timed instances,
  50 synchronization edges, periodic II 3674 cycles, and makespan 59810 cycles.

The solver searches periodic lane order, operation timing, and synchronization
relations while preserving authored ownership. This hardware document has no
resource windows, so the result does not model hardware-engine contention.

Search from the repository root:

```bash
.venv/bin/python -m tilefoundry.cli schedule \
  --program examples/mla-schedule/program.json \
  --hardware examples/mla-schedule/hardware.json \
  --solver-timeout 60 \
  --json > /tmp/mla-h200-schedule.json
```

The committed schedule is an independently verified witness. It is not a
promise that every time-limited solve returns the same witness: when a
lexicographic optimization stage reaches its timeout, the workflow may return
a different `FEASIBLE_NOT_PROVEN` schedule.

Generate an interactive visualization without committing the derived HTML:

```bash
.venv/bin/python -m tilefoundry.cli visualize \
  examples/mla-schedule/schedule.json \
  --out /tmp/mla-h200-schedule.html \
  --title "H200 MLA schedule, 16 iterations" \
  --open
```
