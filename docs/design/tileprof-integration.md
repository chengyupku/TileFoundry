# TileProf Integration Boundary

## Ownership

TileProf owns profiling, CUDA benchmarks, offline/online cross-checks,
provenance, record storage, and B200 calibration. TileFoundry MUST NOT import
the TileProf runtime, parse TileProf JSONL or manifests, open its database, or
copy its profiler and record-store implementation.

TileFoundry's public cost boundary is limited to the existing generic
interfaces:

- `OperationSignature`;
- `OperationCost`;
- `CostLibrary`;
- `build_warpgroup_problem()`;
- the `WarpgroupProblem` v2 codec.

TileProf may implement `CostLibrary` directly, or may generate a validated
`WarpgroupProblem` v2 document through an external integration. TileFoundry
receives only that typed library or typed problem. It does not know whether the
values came from JSONL, a manifest, provenance storage, or a database.

The M5.2 measured B200 corpus and real B200 validation remain an external
integration gate. They do not block development of the generic TileFoundry
scheduler, and no TileFoundry acceptance criterion may be marked complete from
fixture timing or an unmeasured artifact.

## Semantic Handoff

TileProf resolves its complete `TileOpSpec.identity()` internally before it
constructs a cost entry or a v2 problem. A TileProf spec may serve multiple
SSA-renamed operations only when TileProf has proved the exact semantic
identity and cost equivalence. Operation IDs, SSA names, type aliases,
workload roles, lane ownership, and instruction spelling are not matching
inputs at the TileFoundry boundary.

The current TileProf schema-v1 record does not carry the complete canonical
`TileOpSpec.identity()` payload. Its `key` and `id` are not sufficient to
construct an M1 signature. TileProf therefore cannot hand a schema-v1 JSONL
line to TileFoundry as a cost entry without first resolving the complete
identity in TileProf and emitting the generic `CostLibrary` or v2 problem
boundary. TileFoundry performs no nearest matching, fallback, or field-based
identity inference.

Missing, invalid, unverified, wrong-architecture, wrong-CUDA-version, or
ambiguous measurements fail in the external TileProf integration. The
TileFoundry scheduler only receives complete typed entries and keeps its exact
lookup and missing-signature failures.

## Timing Contract

TileProf records expose three values:

- `warp_busy_cycles`: issue-side occupancy;
- `ready_latency_after_issue_cycles`: additional wait after issue;
- `total_latency_from_start_cycles`: result completion from start.

TileFoundry v2 represents `issue_duration` and `completion_latency` separately.
An external producer must explicitly bind the selected measurements to those
generic fields; TileFoundry does not copy a field by name or infer an async
operation class. Both values are positive integers and completion is no earlier
than issue end.

TileProf cycle values must be finite and positive. Each original absolute
timing field is independently converted with `ceil`; truncation, nearest
rounding, and reconstructing a field by splitting or adding other fields are
forbidden. The producer explicitly selects the primary statistic, with no
implicit p50 or p90, and runs sensitivity statistics as separate closed
problem/solve inputs.

## Resource Contract

TileProf timing records do not directly define TileFoundry resource demands or
capacities. The external producer must supply an explicit, versioned, auditable
target contract that creates v2 resource windows. TileFoundry MUST NOT infer a
resource from `op_kind`, instruction name, residency, workload name, or
provenance. A missing mapping fails the external handoff.

## B200 Status

TileProf documentation and corpus remain centered on H200 `sm_90a`. A measured
B200 `sm_100a` catalog and its real validation gate remain TileProf work.
TileFoundry does not generate a measured catalog, does not parse a TileProf
artifact, and does not mark AC-5 from this design boundary.
