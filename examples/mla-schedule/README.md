# H200 MLA Schedule Search

This directory retains replayable CP-SAT inputs and verified outputs for the
central loop of an MLA kernel. The two closed problems contain the same 51
static operations, fixed ownership across three warpgroups, and exploratory
H200 timing values. They differ only in the requested finite loop extent.

Files:

- `problem-2.json`: two-iteration `WarpgroupProblem` v3.
- `schedule-2.json`: verified two-iteration `WarpgroupSchedule` v3, with 102
  timed instances, 48 synchronization edges, and makespan 8291 cycles.
- `problem-16.json`: sixteen-iteration `WarpgroupProblem` v3.
- `schedule-16.json`: verified sixteen-iteration `WarpgroupSchedule` v3, with
  816 timed instances, 50 synchronization edges, periodic II 3570 cycles, and
  makespan 58271 cycles.

The solver preserves authored warpgroup ownership and searches the periodic
lane order, operation timing, and synchronization relations. The input has no
resource windows, so these results do not model hardware-engine contention.
The timing values are exploratory rather than a production H200 calibration:
they were assembled from TileProf records and documented timing values before
the immutable TileProf identity adapter was available.

Replay both schedules from the repository root:

```bash
.venv/bin/python -m tilefoundry.cli schedule \
  --warpgroup-problem examples/mla-schedule/problem-2.json \
  --solver-timeout 60 \
  --json > /tmp/mla-h200-schedule-2.json

.venv/bin/python -m tilefoundry.cli schedule \
  --warpgroup-problem examples/mla-schedule/problem-16.json \
  --solver-timeout 60 \
  --json > /tmp/mla-h200-schedule-16.json
```

The committed schedules are independently verified witnesses. They are not a
promise that every time-limited solve returns the same witness: when a
lexicographic optimization stage reaches its timeout, the public workflow may
return a different `FEASIBLE_NOT_PROVEN` schedule. The retained artifacts are
parsed and checked by the schedule example smoke test.

Generate an interactive visualization without committing the derived HTML:

```bash
.venv/bin/python -m tilefoundry.cli visualize \
  examples/mla-schedule/schedule-16.json \
  --out /tmp/mla-h200-schedule-16.html \
  --title "H200 MLA schedule, 16 iterations" \
  --open
```
