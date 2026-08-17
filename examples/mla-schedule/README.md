# H200 MLA Schedule Search

This directory contains the complete three-document workflow for the central
loop of an MLA kernel. The program has 51 static operations, 16 requested
iterations, and fixed ownership across three warpgroups.

Files:

- `program.json`: typed SSA semantics and fixed warpgroup ownership.
- `hardware.json`: 37 canonical operation signatures with measured H200 timing.
- `schedule.json`: independently verified witness with 816 timed instances,
  48 synchronization edges, periodic II 3806 cycles, and makespan 62235 cycles.

The timing was measured by TileProf commit
`48ac28281396038c9a18e9acc141e8db6a4f8cc7` on an idle NVIDIA H200
(`sm_90a`) with CUDA 13.2. All 12 normalized TileProf specs passed correctness
and provenance gates with `valid=true` and `grounding_status=measured`; the raw
JSONL SHA-256 is
`0280ad73259c0ab54189fdcd14ca5963e674cfdcab4ac1d22921e4a811eb59b0`.
Each TileFoundry cost independently rounds `warp_busy_cycles` up to
`issue_duration` and `total_latency_from_start_cycles` up to
`completion_latency`. The 51 operations were verified to have identical SSA
semantics in the TileProf fixture before its 12 specs were expanded to 37 full
TileFoundry signatures.

The solver searches periodic lane order, operation timing, and synchronization
relations while preserving authored ownership. TileProf does not yet provide a
validated resource-capacity contract for these measurements, so this hardware
document has no resource windows and the result does not model hardware-engine
contention.

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
