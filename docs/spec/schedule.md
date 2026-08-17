# Warpgroup Scheduling Contract

TileFoundry schedules one typed SSA loop across a fixed number of warpgroup
lanes. The public interface has exactly three immutable JSON documents:

```text
program.json + hardware.json -> internal WarpgroupProblem
internal WarpgroupProblem     -> CP-SAT -> schedule.json -> independent verify
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
| Program | `tilefoundry.warpgroup_program` | Typed semantic operations with fixed ownership |
| Hardware | `tilefoundry.warpgroup_hardware` | Signature-indexed timing and resources |
| Schedule | `tilefoundry.warpgroup_schedule` | Lane order, synchronization, and timing witness |

There are no legacy variants or serialized problem documents. The internal
problem is the validated closure of one program and one hardware description.

## Program

A program contains `format`, `warp_groups`, `types`, `inputs`, and `loop`.
Each operation contains an ID, fixed `warp_group`, typed SSA outputs, and
expression trees. Input order and operation order have no scheduling meaning.

Expressions cover constants, references, indexing, copies, casts, transpose,
concatenation, selection, elementwise arithmetic, reductions, exponentials,
and matrix multiplication. Type checking includes shape, dtype, memory space,
axis validity, and index bounds.

Dependencies are derived from SSA definitions and uses. They are not a second
authored edge list. Loop `iter_args` give the initial value and the SSA value
yielded to the next iteration.

## Hardware And Cost Closure

`OperationSignature` is derived from the complete expression forest and typed
operands/results. It preserves cost-relevant operators, constants, axes,
aliasing, shape, dtype, and memory space while erasing operation IDs, SSA
spelling, loop-index spelling, and type aliases.

Hardware contains `format`, `time_unit`, resource capacities, and cost entries.
Each cost entry maps a complete `OperationSignature` to `issue_duration`,
`completion_latency`, and `resource_windows`. It contains no operation IDs,
SSA names, loop, or copied program semantics.

`build_warpgroup_problem` performs one exact lookup per distinct signature.
Missing signatures are reported together; ambiguous entries are rejected. A
failed build never returns a partial problem.

## Timing and Resources

Every closed operation has:

- `issue_duration`: lane occupancy after start;
- `completion_latency`: time until outputs are ready;
- `resource_windows`: resource, amount, start offset, and duration.

Lane `NoOverlap` applies to issue intervals. SSA visibility and shared-memory
lifetime use completion. Resources use cumulative capacity over their declared
windows. The objective is the maximum completion time of the finite requested
prefix.

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

The internal problem uses a compact periodic model:

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
Each time row contains iteration, operation ID, start, issue end, and
completion.

Export removes completion relations already implied by lane and sync paths.
The independent verifier reconstructs the finite event graph and checks:

- complete and unique timing coverage;
- declared durations and completion latency;
- fixed ownership and lane order;
- SSA readiness and register locality;
- shared visibility and overwrite lifetime;
- cumulative resource capacity;
- synchronization inequalities and acyclicity;
- one periodic II across all body rows.

Deleting a required synchronization edge or mutating a timing witness must
cause verification to fail.
