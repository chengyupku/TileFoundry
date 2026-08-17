# Warpgroup Scheduling Contract

TileFoundry schedules one typed SSA loop across a fixed number of warpgroup
lanes. The core API has three immutable documents:

```text
WarpgroupProgram + CostLibrary -> WarpgroupProblem
WarpgroupProblem               -> WarpgroupSolveResult
WarpgroupProblem + SolveResult -> WarpgroupSchedule -> independent verify
```

The solver never calls a profiler, target object, device runtime, or record
store. Every cost and resource fact is numeric before solving begins.

## Documents

JSON decoders reject unknown fields, malformed identifiers, invalid expression
arity, ill-typed operations, duplicate SSA definitions, undefined values, and
invalid loop-carried state. The schemas under `schemas/` define structural
syntax; typed decoders own cross-record semantics.

Supported formats are:

| Document | Formats | Purpose |
| --- | --- | --- |
| Program | v1, v2 | Typed semantic operations; v2 fixes operation ownership |
| Problem | v1, v2, v3 | Closed costs/resources; v2 adds asynchronous timing, v3 fixes ownership |
| Schedule | v1, v2, v3 | Lane order, synchronization edges, and timing witness |

Problem and schedule versions pair exactly. No decoder guesses or silently
migrates a version. v1 remains the synchronous compatibility form; v2/v3 use
explicit issue and completion timing.

## Program

A program contains `format`, `warp_groups`, `types`, `inputs`, and `loop`.
Each operation contains an ID, typed SSA outputs, expression trees, and in v2 a
fixed `warp_group`. Input order and operation order have no scheduling meaning.

Expressions cover constants, references, indexing, copies, casts, transpose,
concatenation, selection, elementwise arithmetic, reductions, exponentials,
and matrix multiplication. Type checking includes shape, dtype, memory space,
axis validity, and index bounds.

Dependencies are derived from SSA definitions and uses. They are not a second
authored edge list. Loop `iter_args` give the initial value and the SSA value
yielded to the next iteration.

## Cost Closure

`OperationSignature` is derived from the complete expression forest and typed
operands/results. It preserves cost-relevant operators, constants, axes,
aliasing, shape, dtype, and memory space while erasing operation IDs, SSA
spelling, loop-index spelling, and type aliases.

`build_warpgroup_problem` performs one exact lookup per distinct signature.
Missing signatures are reported together; ambiguous entries are rejected. A
failed build never returns a partial problem.

## Timing and Resources

For asynchronous problems every operation has:

- `issue_duration`: lane occupancy after start;
- `completion_latency`: time until outputs are ready;
- `resource_windows`: resource, amount, start offset, and duration.

Lane `NoOverlap` applies to issue intervals. SSA visibility and shared-memory
lifetime use completion. Resources use cumulative capacity over their declared
windows. The objective is the maximum completion time of the finite requested
prefix.

v1 `duration` means equal issue and completion timing with full-duration
resource windows. Encoding a genuinely asynchronous document as v1 is an
error.

## Memory Semantics

Register values are lane-local: a definition and every use belong to one
warpgroup. Shared values may cross lanes after a completion-to-start
synchronization path. A computed register value reaches shared memory only
through an explicit copy operation.

Each shared SSA definition is one logical allocation reused at the same body
position. Before iteration `i` overwrites it, every use from iteration `i - 1`
must complete. External shared initialization is ready at the iteration-zero
boundary, so the first consumer may overlap the first publication.

## Periodic Fixed-Owner Search

Problem v3 uses a compact periodic model:

- one prologue start per operation;
- one body start offset per operation;
- one global positive initiation interval;
- one cyclic issue order per fixed warpgroup.

For body iterations `i >= 1`, `start(i, op) = offset(op) + i * II`. Periodic
resource conflicts are checked with the finite set of neighboring copies that
can overlap a static resource window, so CP-SAT model size does not grow with
the requested iteration count. Materialized output still contains one timing
row per operation per requested iteration.

The last requested iteration uses omitted-final-successor semantics. It is a
finite body row, not an independently searched epilogue.

Optimization is lexicographic: finite makespan, then II, then stable starts,
offsets, and lane-order encoding. If a later optimization stage reaches the
deadline, the last proven witness is returned as `FEASIBLE_NOT_PROVEN`.

## Schedule and Verification

A schedule contains exactly `format`, `lanes`, `sync`, and `times`.
Synchronization edges identify producer, consumer, and iteration distance.
Times contain iteration, operation ID, start, issue end, and completion for
v2/v3; v1 contains start/end.

Export removes completion relations already implied by lane and sync paths.
The independent verifier reconstructs the finite event graph and checks:

- complete and unique timing coverage;
- declared durations and completion latency;
- fixed ownership and lane order;
- SSA readiness and register locality;
- shared visibility and overwrite lifetime;
- cumulative resource capacity;
- synchronization inequalities and acyclicity;
- one periodic II across all body rows in v3.

Deleting a required synchronization edge or mutating a timing witness must
cause verification to fail.
