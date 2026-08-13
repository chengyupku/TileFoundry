# Minimal Tile-Operation Scheduling Contract

This document illustrates the kernel-independent warpgroup boundary specified
in [schedule §6](../spec/schedule.md#6-warpgroup-scheduling-documents). The
three concrete documents are the authored
[program](./lsy-schedule-input.json), its independently serializable
[closed problem](./warpgroup-closed-problem.json), and an
[LSY-style result](./lsy-schedule-output.json). They are design artifacts, not
public `SchedulePlan` types.

The example models the central loop of FlashMLA sparse prefill
`csrc/sm90/prefill/sparse/phase1.cuh`: two consumer warpgroups and one producer
warpgroup process two KV blocks per loop iteration. `D_QK = 576` gives nine
`64 x 64` QK microtiles. Each consumer accumulates one `64 x 256` half of the
`D_V = 512` output. The numbers and operation names belong to this example; the
format does not give them special meaning.

FlashMLA's `kv` tensor stores one 576-element latent vector per token. The
producer copies each `64 x 64` region once into shared memory. QK consumes that
same SSA tile through a transposed view, while PV concatenates four tiles and
consumes the resulting `64 x 256` view. Thus the input uses one `kv_tile` type
and one output per copy. It does not invent separate K/V buffers or encode a
packed physical offset scheme; those views are already explicit in the
consumer expressions.

The problem boundary is the repeated block loop. The loop-invariant Q TMA load
before the loop and the final output/LSE stores after it are fixed prelude and
epilogue work; they are intentionally outside this minimal pipeline search.

In particular, type names, resource names, SSA names, and operation IDs are
caller-defined identifiers. A different kernel can replace every such string
without changing the grammar. Only JSON property names and expression operator
names belong to the format.

## Contract

The authored program describes semantic work only. The closed problem adds the
integer facts consumed by a solver. The output describes only the selected
partial order and one timed realization.

Input array order has no scheduling meaning. Dependencies are derived from SSA
definitions and uses. A register value's definition and all of its uses must be
assigned to one warpgroup. A loop-body shared value may cross warpgroups; such a
use must be ordered by a path containing a synchronization relation in the
result. The loop's `yield` values become the corresponding `iter_args` in the
next iteration. External `inputs` are assumed ready at iteration zero; an
external shared input is therefore already visible to every lane at the problem
boundary, and its prelude handoff is outside this result. Work that produces
external inputs belongs to the fixed prelude or to an enclosing schedule.

Every loop-body shared SSA definition denotes one logical allocation reused at
the same body position in the next iteration. Distinct definitions are distinct
logical allocations; a later lowering may pack them into a larger shared array.
Iteration `i + 1` may overwrite one such allocation only after every use in
iteration `i` has finished. Therefore the expanded graph (lane order, SSA
edges, and synchronization edges) must contain a path from every old use to the
next definition. A distance-one edge is required only when that path is not
already implied. This single rule expresses lifetime without input buffer
names, slots, ring depths, or barrier generations.

The default and only objective of this format is the finite loop makespan. It is
not a field because the problem size is fixed by `iterations`. A future API that
offers another objective would need to version the format.

## Authored Program Fields

The authored program contains exactly `format`, `warp_groups`, `types`,
`inputs`, and `loop`. Its operations contain only `id` and `outputs`; they do
not contain durations or resource demands. This keeps user-owned semantics
separate from costs supplied by a cost-closing boundary.

## Closed Problem Fields

The closed problem contains exactly these fields:

| Field | Why it cannot be derived |
| --- | --- |
| `format` | Selects the grammar and semantics. |
| `time_unit` | Gives one meaning to every duration and output timestamp. |
| `warp_groups` | Bounds the anonymous execution lanes available to the solve. |
| `resources` | Gives the capacities that allow or forbid simultaneous operations. |
| `types` | Gives shape, element type, and storage space for SSA validation and locality. |
| `inputs` | Declares SSA values not defined in the loop body. |
| `loop` | Gives repetition, carried state, and the schedulable operations. |

All IDs are non-empty ASCII strings; SSA value IDs begin with `%`. Type,
resource, and operation IDs are caller-defined, while `register`, `shared`, and
`global` are the three storage-space values with scheduling meaning. Type
shapes contain positive integers, resource capacities and operation durations
are positive integers, and every operation resource demand is a positive
integer for a resource declared at the top level.

Each type has only `shape`, `dtype`, and `space`. Shape and dtype determine
whether expressions are well typed. Space determines scheduling semantics:
`register` is warpgroup-local, `shared` is cross-warpgroup and iteration-reused,
and `global` is external storage.

The loop has only `index`, `iterations`, `iter_args`, and `ops`. `index` is an
implicit integer scalar used to select each iteration's external data;
implementations need not add it to `types`. `iterations` closes the finite
problem. Each iter arg needs `id`, `init`, and `yield` to express a loop-carried
SSA phi without expanding the loop. `yield` names a body SSA value (or a literal
where the grammar permits one), and its type initializes the corresponding phi
in the next iteration. A scalar `init` is broadcast to the phi's tensor shape.
After the last iteration, the phi values are the loop's final SSA results; no
separate result list is needed.

Each closed operation has only:

| Field | Why it cannot be derived |
| --- | --- |
| `id` | Gives synchronization and timing records a stable reference. |
| `outputs` | Defines the typed SSA computation and therefore its data dependencies. |
| `duration` | Supplies the operation cost in `time_unit`. |
| `resources` | States simultaneous capacity demand during the interval. |

Each output has `id`, `type`, and `expr`. The expression is the computation, not
an implementation catalog. Generic expression operators such as `index`,
`copy`, `cast`, `matmul`, `transpose`, `concat`, `select`, axis-qualified
`reduce`, and elementwise arithmetic (`add`, `sub`, `mul`, `max`, and `exp`)
have typed semantics. `index` removes one leading dimension per index;
`concat` carries its dimension as its first operand; `reduce` carries its
operator and dimension and keeps the reduced dimension with extent one; `copy`
preserves shape and dtype while taking the destination storage space from the
declared output type; `cast` takes its destination dtype from that output type;
ordinary singleton-dimension broadcasting applies to elementwise operators.
JSON numbers are scalar literals and may broadcast to a tensor; `"-inf"` is the
IEEE negative-infinity scalar literal. Multiple outputs of one operation are
atomic at this abstraction level and become available together.

The following are deliberately absent:

- explicit dependencies, because SSA def-use derives them;
- agents, candidate lanes, roles, or warp IDs, because warpgroup placement is a
  solve result and the requested granularity is warpgroup;
- tile IDs, because operation IDs already identify the microtiles;
- phases, implementations, or special cases such as `tma_issuer` and
  `local_or_remote_pv`, because the input already contains executable tile ops;
- constraints and objective envelopes, because type/SSA/storage/resource rules
  define legality and finite makespan defines optimization;
- barrier names, buffer-region names, slots, and generations, because these are
  code-generation choices derived after synchronization is selected;
- profile provenance, because only numeric costs affect this closed solve.

## Output Fields

The output contains exactly four fields:

| Field | Meaning |
| --- | --- |
| `format` | Selects the result grammar. |
| `lanes` | Assigns every operation to one anonymous warpgroup and selects that lane's order. |
| `sync` | Gives the selected cross-lane and cross-iteration happens-before relations. |
| `times` | Gives a concrete interval for every unrolled operation instance. |

`lanes[g]` is the ordered loop-body program for warpgroup `g`. An operation
appears exactly once across all lanes. Adjacent entries imply an order edge, so
the output does not repeat the warpgroup in `times`. Since the body repeats,
the last entry of lane `g` in iteration `i` also precedes its first entry in
iteration `i + 1`; different lanes may overlap across that boundary.

A synchronization record

```json
{"after": "a", "before": "b", "distance": 0}
```

means the source completion happens before the destination start. At loop
instance `i`, its inequality is

```text
end(i, a) <= start(i + distance, b)
```

for in-range instances. `distance` is a non-negative loop distance;
`distance = 0` is an intra-iteration handoff and `distance = 1` relates one
iteration to the next. `after` and `before` name operations in the loop body,
and an edge with an out-of-range destination instance is simply absent at the
finite boundary. Each record is one happens-before edge; the solver may
emit only a transitively reduced set because lane and SSA edges are part of the
same graph. Barrier allocation, arrival counts, phases, and instruction
selection are derived by code generation.

Each time row is `[iteration, op_id, start, end]`. It is a required feasibility
witness, not the canonical control mechanism. `end - start` must equal the
input duration. Times must respect SSA def-use, iter args, lane order,
resource capacities, synchronization records, and shared-allocation reuse. The
makespan is `max(end)` and is therefore not repeated as a field.

## LSY Result

The concrete result selects these lane orders:

```text
lane 0:
  QK0 0..8
  softmax0 -> rescale O-left -> publish m0 -> local P0V0-left
  rescale P0 -> publish P0 -> final rescale O-left -> remote P1V1-left

lane 1:
  QK1 4..8, then 0..3
  softmax1 -> rescale O-right -> publish m1 -> local P1V1-right
  publish P1 -> remote P0V0-right

lane 2:
  copy KV0 0..3
  copy KV1 4..8
  copy KV0 4..8
  copy KV1 0..3
  copy validity
```

The asymmetry is selected order, not an input role. Lane 0 begins QK after the
first KV0 region arrives. Lane 1 begins QK after the KV1 right region arrives,
then consumes the left region. Both overlap with later producer copies.

The distance-zero synchronization records express five facts not represented by
same-lane order:

1. Each consumer waits for the copied region that starts its QK sequence.
2. Both softmax operations wait for the validity data.
3. Lane 1 waits for lane 0's online-softmax maximum.
4. Lane 0 waits for lane 1's updated maximum before rescaling and publishing P0.
5. Lane 1 waits for P0 before it consumes the remote probability. The final
   remote handoff is ordered transitively by the lane and SSA edges. This
   preserves the reference kernel's shared-score handoff.

The witness has distance-one records for the four KV tile groups:

```text
P0V0-left  -> next KV0 tiles 0..3 copy
P1V1-right -> next KV1 tiles 4..8 copy
P0V0-right -> next KV0 tiles 4..8 copy
P1V1-left  -> next KV1 tiles 0..3 copy
```

The validity mask, exchanged maxima, and shared probabilities need no extra
distance-one records in this witness: the four KV edges, lane order, and the
intra-iteration handoffs already put every next definition after its previous
uses. This is what expresses inter-iteration pipelining: each next definition
starts when its own old value is free, rather than after the whole prior
iteration. Packing logical values into physical arrays and choosing hardware
barrier instances are derived lowering steps; the synchronization result
remains valid if that packing changes.

The output introduces no fields or enums for `producer`, `wg0`, `wg1`, buffer
sides, `ready`, `free`, `mbarrier`, or `phase`. Words such as `left` and `right`
remain only inside caller-defined operation IDs from the input; they have no
format semantics. Hardware barrier details are lowering decisions, not parts of
a general scheduling result.

## Validation

A conforming checker must reject a pair unless all of the following hold:

1. IDs and SSA definitions are unique; all expression uses, initial values, and
   yields resolve and type-check independently of JSON array order.
2. `lanes` has exactly `warp_groups` entries. Every operation appears exactly
   once in `lanes`, and each `(iteration, operation)` appears exactly once in
   `times`.
3. Every register SSA def-use component is contained in one lane. Every
   loop-body shared cross-lane use is ordered by the transitive closure of lane
   and sync edges; external shared inputs are covered by the boundary-ready
   assumption above.
4. Each interval has the declared duration; intervals on one lane do not
   overlap; aggregate resource demand never exceeds input capacity.
5. Every in-range synchronization inequality is satisfied; sync edges are
   unique and contain no self edge.
6. Every next-iteration overwrite of a shared SSA allocation follows all
   previous-iteration uses, either through a sync record or an already implied
   path.
7. Iter arg yields complete before their next-iteration uses, and the expanded
   relation graph is acyclic.

For this illustrative unit-cost profile the two-iteration witness ends at cycle
75. The synchronization graph is the reusable schedule; the numeric intervals
show that it is realizable under the supplied costs.
