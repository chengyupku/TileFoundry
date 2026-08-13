---
type: FEAT
component: schedule
target_repo: tilefoundry
---

# [FEAT][schedule] Search finite warpgroup pipelines from typed tile operations

## Description

TileFoundry has two adjacent implementations that do not yet provide the
required workflow:

- `src/tilefoundry/schedule/pipeline` extracts HIR statements and target facts,
  but its current solver chooses one instruction per statement and places every
  statement on one serial clock. It does not assign operations to anonymous
  warpgroups or export synchronization relations.
- `costmodel` has typed tile operations, exact B200 facts, implementation
  lowering, profile storage, and CUDA profiling. Its planned solver boundary
  receives phases whose `warp_ids` were already selected during configuration
  construction, so it cannot discover an operation-to-warpgroup assignment.

The required capability is a narrower warpgroup pipeline search. The caller
describes one explicit finite loop as typed SSA tile operations. A cost library
closes every operation to an integer duration and resource demand. A finite
CP-SAT solve assigns each loop-body operation to one anonymous warpgroup,
selects a stable order on each warpgroup, and minimizes finite makespan. The
result exports that order, the necessary cross-warpgroup and cross-iteration
happens-before relations, and one concrete timed realization.

The design examples are:

- [authored program](../design/lsy-schedule-input.json)
- [closed problem](../design/warpgroup-closed-problem.json)
- [schedule output](../design/lsy-schedule-output.json)
- [minimal scheduling contract](../design/lsy-synchronization-schedule.md)

They model the central loop of FlashMLA sparse prefill and establish the
intended level of abstraction. They are examples until M0 moves their durable
semantics into [schedule](../spec/schedule.md).

### Data Path

```text
WarpgroupProgram
  typed SSA values, one explicit loop, tile operations, warpgroup count
        |
        | build(program, cost_library)
        v
WarpgroupProblem
  integer durations, integer resource demands and capacities, no callback
        |
        | solve(problem)
        v
WarpgroupSchedule
  lanes + sync + times
```

`WarpgroupProgram` is the user-authored semantic input. It does not contain
durations, resource capacities, hardware roles, candidate lanes, explicit
dependencies, or an objective field. `WarpgroupProblem` is the replayable
numeric input to CP-SAT. It contains the fields currently shown in the design
closed problem, including `time_unit`, `duration`, and resource data. A user may inspect
or serialize it, but normally obtains it from `build` rather than writing costs
by hand.

The cost library is outside the solver. It receives canonical operation
signatures and returns exact numeric costs. A signature is derived from the
operation's expression structure and typed operands/results; it never contains
operation IDs, SSA spelling, `left`/`right`, producer/consumer roles, or other
kernel-specific names. `build` must resolve every cost before returning a
problem. `solve` must not import a profile store, CUDA, a target object, or a
cost-provider callback.

### Scope

- One CTA and one explicit loop.
- A finite positive iteration count.
- Anonymous warpgroup lanes; the count is an input and placement is a result.
- SSA-derived data dependencies.
- Register values local to one warpgroup.
- Shared values visible across warpgroups after an exported synchronization
  path, and reused safely across loop iterations.
- Integer operation durations, integer temporal-resource demands, and integer
  capacities.
- One default objective: minimize finite makespan.
- Strict Python construction, strict JSON, deterministic solving, independent
  result verification, and a direct JSON workflow.

### Out Of Scope

- Choosing microtiling, operation fusion, mathematical identities, or a
  different tile-operation graph.
- User-authored warp IDs, producer/consumer roles, phases, implementation
  catalogs, barrier slots, barrier generations, or named barriers.
- Periodic scheduling for unbounded or very long loops.
- Searching pipeline depth or physical shared-memory packing.
- Lowering synchronization edges to PTX/CUDA barriers or generating a kernel.
- Replacing the existing HIR pipeline scheduler before an explicit adapter can
  preserve its public `PipelineSchedulePlan` behavior.

### Ownership

The new implementation is owned by
`src/tilefoundry/schedule/warpgroup/`. It is a scheduling algorithm, not an
extension of the standalone costmodel's fixed-warp configuration solver. The
main package already depends on OR-Tools, so the finite backend belongs in this
package.

The core cost-library protocol is dependency-free and owned by the same
warpgroup package. A frozen numeric catalog is sufficient at runtime. B200
profiling tools may reuse the standalone costmodel's profile store and CUDA
runner to produce catalog entries, but the main TileFoundry import path and a
replayable `WarpgroupProblem` must not depend on `tilefoundry_costmodel`.

### Core Invariants

- The public problem and result contain no OR-Tools objects or callbacks.
- Input array order has no schedule meaning.
- Every loop-body operation is assigned to exactly one lane.
- One operation has one lane and one relative lane order for every iteration;
  different iterations cannot choose different body schedules.
- Operations on one lane do not overlap. Different lanes may overlap when SSA,
  synchronization, lifetime, and resource constraints permit it.
- Register def-use components stay on one lane. Shared def-use may cross lanes.
- A next-iteration shared definition begins only after all uses of the previous
  iteration's logical allocation complete.
- `sync` contains control edges needed to realize cross-lane and cross-iteration
  ordering. It is derived from the solved order rather than accepted from the
  user.
- `times` is a required feasibility witness. It is not the canonical control
  representation.
- Successful output JSON contains exactly `format`, `lanes`, `sync`, and
  `times`; makespan is `max(end)` and is not duplicated.

## Development Dependency Tree

```text
M0 contract and strict typed boundary
`-- M1 cost-library closure
    `-- M2 exact finite CP-SAT
        `-- M3 synchronization export and independent verification
            `-- M4 public API, JSON workflow, and FlashMLA reference solve
                `-- M5 calibrated B200 cost catalog
```

M0-M4 first establish a correct solver with a retained deterministic fixture
catalog. M5 replaces illustrative fixture costs with measured B200 entries; it
does not change the solver problem or result formats.

## Planned Package Layout

```text
src/tilefoundry/schedule/warpgroup/
  __init__.py
  api.py
  cost.py
  errors.py
  expression.py
  model.py
  problem.py
  serialization.py
  solve.py
  sync.py
  verify.py

src/tilefoundry/target/cuda/
  warpgroup_costs.py

schemas/
  warpgroup-program-v1.schema.json
  warpgroup-problem-v1.schema.json
  warpgroup-schedule-v1.schema.json

tests/schedule/
  test_warpgroup_schedule.py

costmodel/benchmarks/
  warpgroup/

costmodel/calibration/
  b200-warpgroup-costs.json
```

Private helpers may be combined when that keeps ownership clearer. The layout
does not require one implementation file per listed name.

## Milestones

### Milestone M0: Freeze The Typed Scheduling Boundary

#### Depends
- None

#### Golden Reference
- Source: [minimal scheduling contract](../design/lsy-synchronization-schedule.md),
  [authored program](../design/lsy-schedule-input.json),
  [closed problem](../design/warpgroup-closed-problem.json),
  [schedule output](../design/lsy-schedule-output.json), and
  [schedule](../spec/schedule.md).
- Functional points: one explicit finite loop; SSA-derived dependencies;
  anonymous warpgroup placement; register locality; shared cross-lane and
  cross-iteration semantics; minimal numeric problem and schedule documents.

#### Related Files
- `docs/spec/schedule.md`
- `src/tilefoundry/schedule/warpgroup/__init__.py`
- `docs/design/lsy-synchronization-schedule.md`
- `docs/design/lsy-schedule-input.json`
- `docs/design/warpgroup-closed-problem.json`
- `docs/design/lsy-schedule-output.json`
- `src/tilefoundry/__init__.py`
- `src/tilefoundry/schedule/__init__.py`
- `src/tilefoundry/schedule/registry.py`
- `src/tilefoundry/schedule/warpgroup/__init__.py`
- `src/tilefoundry/schedule/warpgroup/errors.py`
- `src/tilefoundry/schedule/warpgroup/model.py`
- `src/tilefoundry/schedule/warpgroup/expression.py`
- `src/tilefoundry/schedule/warpgroup/serialization.py`
- `schemas/warpgroup-program-v1.schema.json`
- `schemas/warpgroup-problem-v1.schema.json`
- `schemas/warpgroup-schedule-v1.schema.json`
- `tests/schedule/test_warpgroup_schedule.py`

#### Plan
- [x] step 0.1 Move the durable grammar and semantics from the design document
  into `docs/spec/schedule.md`. Define distinct user-authored
  `WarpgroupProgram`, closed numeric `WarpgroupProblem`, and successful
  `WarpgroupSchedule` contracts without adding roles, explicit dependencies,
  objectives, phases, implementations, or barrier fields.
- [x] step 0.2 Implement deeply immutable records for types, inputs, loop phi
  arguments, operations, outputs, expressions, numeric costs, lanes,
  synchronization edges, and timed instances. Keep authoring and numeric
  problem ownership distinct even where they share records.
- [x] step 0.3 Implement a strict expression decoder and type checker for
  `index`, `copy`, `cast`, `matmul`, `transpose`, `concat`, `select`,
  axis-qualified `reduce`, and singleton-broadcast elementwise arithmetic.
  Derive def-use edges independently of JSON array order.
- [x] step 0.4 Implement strict schemas and canonical codecs. Reject unknown
  fields, unresolved or multiply defined SSA values, invalid yields, invalid
  index bounds, shape/dtype errors, invalid resource data, and non-ASCII IDs.
- [x] step 0.5 Retain the FlashMLA design documents as canonical examples under
  the new codecs, separating the untimed user program from the fully closed
  numeric problem if the final spec uses two serialized documents.

#### Acceptance Criteria
- [x] AC-0-1: Python and JSON construction of the same program, problem, and
  schedule produce byte-identical canonical JSON.
- [x] AC-0-2: The reference input resolves every SSA use and loop yield,
  type-checks every expression, and derives the expected intra-iteration and
  loop-carried dependency graph without an explicit dependency field.
- [x] AC-0-3: Unknown fields/operators, duplicate IDs, unresolved SSA values,
  invalid indices, invalid broadcasting, invalid memory spaces, and malformed
  loop phi values fail before cost resolution or solving.
- [x] AC-0-4: The authoring document has no duration, resource capacity,
  operation resource demand, target role, candidate lane, explicit objective,
  or synchronization field; the closed problem contains only numeric facts the
  solver consumes.
- [x] AC-0-5: Importing the typed boundary does not import OR-Tools, CUDA, the
  standalone costmodel, or target implementations.
<!-- policy_ac:start -->
- [x] Milestone MUST name a `#### Golden Reference` before implementation steps, with the source of truth and the observable functional points it determines. <!-- policy_ac: milestone_review-0 -->
- [x] The gate request MUST show the Golden Reference's functional points exercised through the smallest real workflow, naming the retained evidence; one workflow MAY cover several ACs, and a new test requires a stated reachability gap. <!-- policy_ac: milestone_review-1 -->
- [x] Touched tests and comments MUST be reviewed for redundancy: remove ones superseded by the retained workflow, and do not add source-shape or hypothetical-refactor guards unless that form is a public contract. <!-- policy_ac: milestone_review-2 -->
- [x] A milestone that changes a public contract MUST list the owning `docs/spec/*.md` path in its `#### Related Files`; one that changes none lists no spec path. <!-- policy_ac: spec_impact-0 -->
<!-- policy_ac:end -->

### Milestone M1: Close Every Operation Through A Cost Library

#### Depends
- M0

#### Golden Reference
- Source: [cost-model profile identity](../spec/cost-model.md#81-canonical-profile-identity), [cost-model build boundary](../spec/cost-model.md#9-build-and-search-problem), and the immutable target-facts projection used by `src/tilefoundry/schedule/pipeline`.
- Functional points: semantic operation identity independent of local names;
  exact lookup with no fallback timing; all costs closed before solve; numeric
  replay without the provider that built the problem.

#### Related Files
- `docs/spec/schedule.md`
- `src/tilefoundry/schedule/warpgroup/cost.py`
- `src/tilefoundry/schedule/warpgroup/build.py`
- `src/tilefoundry/schedule/warpgroup/model.py`
- `src/tilefoundry/schedule/warpgroup/expression.py`
- `src/tilefoundry/schedule/warpgroup/serialization.py`
- `tests/schedule/test_warpgroup_schedule.py`

#### Plan
- [x] step 1.1 Define a canonical `OperationSignature` from the complete
  expression forest, ordered external operand types, and output types. Erase
  operation IDs and SSA spelling while preserving operators, constants,
  shapes, dtypes, memory spaces, axes, and multi-output atomicity.
- [x] step 1.2 Define immutable `OperationCost`, resource-capacity, and
  `OperationCostLibrary` boundaries. One exact signature resolves to one
  positive integer duration and positive integer demands in one declared time
  unit; missing or ambiguous entries are errors.
- [x] step 1.3 Implement `build_warpgroup_problem`. Validate the program, resolve
  every unique signature once, copy all numeric results into an immutable
  problem, and discard the library reference before returning.
- [x] step 1.4 Add a deterministic fixture catalog for the retained reference
  solve. Mark it as test/design evidence rather than B200 calibration and keep
  production lookup from manufacturing a default or roofline timing.
- [x] step 1.5 Serialize and reload the closed problem on a process that has no
  cost library. The reloaded value must contain everything needed by M2.

#### Acceptance Criteria
- [x] AC-1-1: Renaming operation IDs, SSA values, or caller-defined type aliases
  without changing typed semantics produces the same operation cost queries.
- [x] AC-1-2: A semantic change to an operator, constant, axis, shape, dtype,
  memory space, or output set produces a different query.
- [x] AC-1-3: One build resolves all unique queries, reports every missing exact
  entry together, and returns no partial problem on failure.
- [x] AC-1-4: The resulting problem round-trips and remains solvable without the
  cost library, target, CUDA, profile database, or standalone costmodel.
- [x] AC-1-5: Neither the authoring input nor lookup behavior depends on names
  such as `qk`, `softmax`, `left`, `right`, `producer`, or `consumer`.
<!-- policy_ac:start -->
- [x] Milestone MUST name a `#### Golden Reference` before implementation steps, with the source of truth and the observable functional points it determines. <!-- policy_ac: milestone_review-0 -->
- [x] The gate request MUST show the Golden Reference's functional points exercised through the smallest real workflow, naming the retained evidence; one workflow MAY cover several ACs, and a new test requires a stated reachability gap. <!-- policy_ac: milestone_review-1 -->
- [x] Touched tests and comments MUST be reviewed for redundancy: remove ones superseded by the retained workflow, and do not add source-shape or hypothetical-refactor guards unless that form is a public contract. <!-- policy_ac: milestone_review-2 -->
- [x] A milestone that changes a public contract MUST list the owning `docs/spec/*.md` path in its `#### Related Files`; one that changes none lists no spec path. <!-- policy_ac: spec_impact-0 -->
<!-- policy_ac:end -->

### Milestone M2: Implement Exact Finite Warpgroup CP-SAT

#### Depends
- M1

#### Golden Reference
- Source: [minimal scheduling contract](../design/lsy-synchronization-schedule.md), [OR-Tools interval scheduling](https://developers.google.com/optimization/scheduling), and the retained CP-SAT modeling conventions in `src/tilefoundry/schedule/partition/solve.py`.
- Functional points: anonymous lane assignment; stable per-lane loop-body
  order; finite expansion; SSA, recurrence, resource, locality, and lifetime
  constraints; minimum makespan with deterministic output.

#### Related Files
- `docs/spec/schedule.md`
- `src/tilefoundry/schedule/warpgroup/solve.py`
- `src/tilefoundry/schedule/warpgroup/model.py`
- `src/tilefoundry/schedule/warpgroup/errors.py`
- `tests/schedule/test_warpgroup_schedule.py`

#### Plan
- [x] step 2.1 Build integer start/end variables for every finite operation
  instance and one shared lane variable per loop-body operation. Create optional
  intervals per possible lane and enforce exactly one placement.
- [x] step 2.2 Search one stable relative order for operations sharing a lane.
  Apply that order to every loop iteration, prevent same-lane overlap, and
  order a lane's final operation in iteration `i` before its first operation in
  iteration `i + 1`.
- [x] step 2.3 Add SSA completion-to-start constraints and loop-phi recurrence.
  Constrain each register def-use component to one lane; allow shared values to
  cross lanes without preassigning either endpoint.
- [x] step 2.4 Add next-iteration shared-allocation reuse constraints from every
  old use to the next definition. Add `NoOverlap` or `Cumulative` constraints
  for each declared temporal resource using the exact numeric demands.
- [x] step 2.5 Minimize the maximum end time. Add lane-symmetry breaking and a
  documented deterministic tie order that does not change the primary
  objective, then map OR-Tools optimal/feasible/infeasible/timeout states to
  typed scheduling outcomes.
- [x] step 2.6 Keep OR-Tools imports local to the solve boundary and export no
  backend variable, status object, or model proto.

#### Acceptance Criteria
- [x] AC-2-1: A hand-computable independent-work problem overlaps operations on
  different lanes, while a one-lane version serializes them and both attain the
  known optimum.
- [x] AC-2-2: Register def-use cannot cross lanes; the equivalent shared
  def-use may cross lanes and is still ordered producer-completion to
  consumer-start.
- [x] AC-2-3: Every iteration uses the same operation-to-lane assignment and
  the same relative body order; no finite expansion can return an
  iteration-specific lane program.
- [x] AC-2-4: Capacity-one resources serialize competing intervals, larger
  capacities permit legal overlap, and no selected time point exceeds any
  capacity.
- [x] AC-2-5: A shared logical allocation is never overwritten in iteration
  `i + 1` before all iteration `i` uses complete. The generic
  `test_m2_m3_loop_carried_shared_init_and_reuse_boundary` retains the
  boundary-ready external init overlap while proving the later carried reuse
  constraint and hand-computed optimum.
- [x] AC-2-6: With deterministic controls, repeated complete solves produce the
  same lane assignment, order, time intervals, objective, and status.
<!-- policy_ac:start -->
- [x] Milestone MUST name a `#### Golden Reference` before implementation steps, with the source of truth and the observable functional points it determines. <!-- policy_ac: milestone_review-0 -->
- [x] The gate request MUST show the Golden Reference's functional points exercised through the smallest real workflow, naming the retained evidence; one workflow MAY cover several ACs, and a new test requires a stated reachability gap. <!-- policy_ac: milestone_review-1 -->
- [x] Touched tests and comments MUST be reviewed for redundancy: remove ones superseded by the retained workflow, and do not add source-shape or hypothetical-refactor guards unless that form is a public contract. <!-- policy_ac: milestone_review-2 -->
- [x] A milestone that changes a public contract MUST list the owning `docs/spec/*.md` path in its `#### Related Files`; one that changes none lists no spec path. <!-- policy_ac: spec_impact-0 -->
<!-- policy_ac:end -->

### Milestone M3: Export Synchronization And Verify Independently

> Status: **CLOSED** after review approval. Closure evidence: 43 focused
> warpgroup tests and all 76 schedule tests passed; Ruff check and format,
> strict mypy, `git diff --check`, verifier/export import isolation, and wheel
> content checks passed. The loop-carried shared boundary regression proves
> iteration-zero external-init overlap, later safe reuse, distance-one
> recurrence synchronization, rejection of the invalid distance-zero carried
> edge, and independent rejection of an early overwrite. M4 and M5 had not
> started when this milestone closed.

#### Depends
- M2

#### Golden Reference
- Source: [schedule output](../design/lsy-schedule-output.json), [minimal scheduling contract](../design/lsy-synchronization-schedule.md), and the synchronization semantics in [TIR](../spec/tir.md#sync).
- Functional points: compact lane programs; explicit happens-before edges;
  cross-iteration distance; concrete time witness; verification without
  re-solving or trusting CP-SAT.

#### Related Files
- `docs/spec/schedule.md`
- `src/tilefoundry/schedule/warpgroup/model.py`
- `src/tilefoundry/schedule/warpgroup/sync.py`
- `src/tilefoundry/schedule/warpgroup/verify.py`
- `src/tilefoundry/schedule/warpgroup/serialization.py`
- `schemas/warpgroup-schedule-v1.schema.json`
- `tests/schedule/test_warpgroup_schedule.py`

#### Plan
- [x] step 3.1 Export `lanes` by sorting each solved lane's body operations by
  its stable order. Export one `[iteration, op_id, start, end]` row for every
  finite operation instance.
- [x] step 3.2 Derive required synchronization candidates from solved
  cross-lane shared def-use, loop-phi handoff where applicable, and
  next-iteration shared reuse. Do not expose solver ordering literals as
  synchronization merely because they existed in the model.
- [x] step 3.3 Remove a synchronization candidate only when the same
  happens-before relation remains reachable through lane-order edges and the
  other emitted synchronization edges. Do not use an un-emitted cross-lane SSA
  edge as proof that a hardware synchronization edge is redundant.
- [x] step 3.4 Implement an independent verifier over only the problem and
  exported schedule. Check total coverage, duration, lane order, recurrence,
  register locality, cross-lane visibility, shared reuse, resources,
  synchronization inequalities, and expanded graph acyclicity.
- [x] step 3.5 Make successful schedule JSON contain exactly `format`, `lanes`,
  `sync`, and `times`, in canonical deterministic order. Keep proof/status in
  the call result rather than adding fields to this schedule document.

#### Acceptance Criteria
- [x] AC-3-1: Removing any synchronization edge that is the only realizable
  cross-lane path makes independent verification fail.
- [x] AC-3-2: Transitively redundant synchronization edges are omitted without
  changing reachability or validity.
- [x] AC-3-3: Every time interval has its declared duration; every operation
  appears once per iteration; lane intervals do not overlap; synchronization
  inequalities and resource capacities hold. The carried-shared boundary test
  also verifies the iteration-zero exception and every later finite lifetime
  edge without changing the compact sync schema.
- [x] AC-3-4: The verifier rejects a modified lane, sync, or time value that
  violates locality, visibility, recurrence, lifetime, capacity, or acyclicity,
  without invoking OR-Tools; its carried-shared lifetime check begins at
  iteration one, while the normal distance-one recurrence remains required.
- [x] AC-3-5: Schedule JSON is byte-identical across repeated deterministic
  solves and contains no lane IDs in `times`, derived makespan, barriers,
  phases, implementation IDs, or profile provenance.
<!-- policy_ac:start -->
- [x] Milestone MUST name a `#### Golden Reference` before implementation steps, with the source of truth and the observable functional points it determines. <!-- policy_ac: milestone_review-0 -->
- [x] The gate request MUST show the Golden Reference's functional points exercised through the smallest real workflow, naming the retained evidence; one workflow MAY cover several ACs, and a new test requires a stated reachability gap. <!-- policy_ac: milestone_review-1 -->
- [x] Touched tests and comments MUST be reviewed for redundancy: remove ones superseded by the retained workflow, and do not add source-shape or hypothetical-refactor guards unless that form is a public contract. <!-- policy_ac: milestone_review-2 -->
- [x] A milestone that changes a public contract MUST list the owning `docs/spec/*.md` path in its `#### Related Files`; one that changes none lists no spec path. <!-- policy_ac: spec_impact-0 -->
<!-- policy_ac:end -->

### Milestone M4: Expose The End-To-End Scheduling Workflow

#### Depends
- M3

#### Golden Reference
- Source: the public scheduling operation in [schedule](../spec/schedule.md), the current `tilefoundry schedule` CLI workflow, and FlashMLA sparse prefill `csrc/sm90/prefill/sparse/phase1.cuh`.
- Functional points: one public build/solve path; strict JSON input/output;
  preserved existing HIR scheduling behavior; reference two-consumer/one-
  producer pipeline with cross-iteration overlap.

#### Related Files
- `docs/spec/schedule.md`
- `src/tilefoundry/schedule/warpgroup/__init__.py`
- `src/tilefoundry/schedule/warpgroup/api.py`
- `src/tilefoundry/schedule/__init__.py`
- `src/tilefoundry/cli/schedule.py`
- `src/tilefoundry/cli/__init__.py`
- `tests/schedule/test_warpgroup_schedule.py`
- `tests/schedule/test_pipeline.py`
- `tests/integration/installed/smoke_schedule.py`
- `docs/design/lsy-schedule-input.json`
- `docs/design/lsy-schedule-output.json`

#### Plan
- [x] step 4.1 Expose typed `build_warpgroup_problem`,
  `solve_warpgroup_problem`, and `schedule_warpgroups` operations. The combined
  operation validates, closes costs, solves, independently verifies, and
  returns both the successful schedule and typed solve status/proof.
- [x] step 4.2 Extend the existing scheduling CLI with an explicit JSON path
  for a warpgroup program or already closed problem. Keep authored HIR source
  behavior and its required topology selection unchanged.
- [x] step 4.3 Run the retained FlashMLA program through the fixture cost
  library and finite CP-SAT. Require two consumer lanes and one copy lane to
  overlap, both shared-probability handoffs to be represented, and next-
  iteration copy to overlap prior-iteration consumer work. Do not require a
  specific anonymous lane number when an equivalent canonical solution exists.
- [x] step 4.4 Render the result as a compact swimlane timeline plus explicit
  synchronization relations, using exactly the same data as JSON.
- [x] step 4.5 Retain current `PipelineSchedulePlan`, HIR thread scheduling,
  partition scheduling, and installed CLI smoke behavior. Do not route existing
  HIR through the new scheduler until a separate adapter can produce the exact
  typed tile-operation program it requires.

#### Acceptance Criteria
- [x] AC-4-1: A user can provide one untimed typed program and one cost library
  and receive a verified `lanes/sync/times` schedule through both Python and
  CLI workflows.
- [x] AC-4-2: A serialized closed problem produces the same schedule without
  loading the cost library that built it.
- [x] AC-4-3: The retained FlashMLA workflow exchanges P0 and P1 through shared
  memory, keeps the two output halves lane-local, overlaps producer and
  consumer work, and begins next-iteration copy before prior-iteration
  consumer completion. The retained LSY smoke derives both register-to-shared
  probability publications from typed SSA, requires each unique cross-lane
  consumer and its distance-zero synchronization edge, and proves that the two
  carried output-half def-use components are individually lane-local on two
  distinct anonymous lanes.
- [x] AC-4-4: JSON and text renderings agree on lane assignment, lane order,
  synchronization, operation intervals, and makespan.
- [x] AC-4-5: Existing authored-HIR pipeline and partition scheduling workflows
  retain their observable plans and CLI behavior. The M4 review gate retains
  strict mypy over the warpgroup and two CLI owning modules, 46 focused
  warpgroup tests, the full schedule suite, 25 CLI/check tests, and 8 installed
  schedule CLI tests.
<!-- policy_ac:start -->
- [x] Milestone MUST name a `#### Golden Reference` before implementation steps, with the source of truth and the observable functional points it determines. <!-- policy_ac: milestone_review-0 -->
- [x] The gate request MUST show the Golden Reference's functional points exercised through the smallest real workflow, naming the retained evidence; one workflow MAY cover several ACs, and a new test requires a stated reachability gap. <!-- policy_ac: milestone_review-1 -->
- [x] Touched tests and comments MUST be reviewed for redundancy: remove ones superseded by the retained workflow, and do not add source-shape or hypothetical-refactor guards unless that form is a public contract. <!-- policy_ac: milestone_review-2 -->
- [x] A milestone that changes a public contract MUST list the owning `docs/spec/*.md` path in its `#### Related Files`; one that changes none lists no spec path. <!-- policy_ac: spec_impact-0 -->
<!-- policy_ac:end -->

### Milestone M5: Populate A Calibrated B200 Cost Catalog

#### Depends
- M4

#### Golden Reference
- Source: [cost-model CUDA benchmark contract](../spec/cost-model.md#83-cuda-benchmark-contract), the retained B200 local CUDA runner, NVIDIA instruction semantics, and FlashMLA sparse prefill `csrc/sm90/prefill/sparse/phase1.cuh`.
- Functional points: correctness-checked B200 measurements; exact canonical
  keys; setup excluded from timing; frozen catalog consumed through the M1
  interface; no solver-contract change.

#### Related Files
- `docs/spec/schedule.md`
- `costmodel/src/tilefoundry_costmodel/profiler/cuda/runner.py`
- `costmodel/src/tilefoundry_costmodel/profiles/store.py`
- `costmodel/src/tilefoundry_costmodel/implementations/b200/`
- `costmodel/benchmarks/warpgroup/`
- `costmodel/calibration/b200-warpgroup-costs.json`
- `src/tilefoundry/target/cuda/warpgroup_costs.py`
- `tests/schedule/test_warpgroup_schedule.py`
- `costmodel/tests/test_cuda_profiler.py`

#### Plan
- [ ] step 5.1 Define the measured coverage matrix required by the retained
  reference: global-to-shared KV copy, shared/shared QK MMA,
  register/shared local PV MMA, shared/shared remote PV MMA, fused online
  softmax, register rescale, and register-to-shared publication for row and
  probability values.
- [ ] step 5.2 Add correctness-checked benchmark providers using the existing
  CUDA runner and profile-store transaction rules. Measure latency and, where
  concurrent issue is modeled, initiation interval under the exact shape,
  dtype, memory-space, and implementation conditions.
- [ ] step 5.3 Export stable measurements into a frozen catalog keyed by the M1
  canonical operation signatures. Record hardware, environment, statistic,
  source digest, and measurement provenance in the artifact without copying
  that provenance into `WarpgroupProblem`.
- [ ] step 5.4 Load the frozen catalog through the main-package cost-library
  protocol and rebuild the FlashMLA problem. Missing measured coverage remains
  an explicit build failure; no synthetic production fallback is permitted.
- [ ] step 5.5 Compare the selected schedule and predicted makespan with the
  handwritten FlashMLA schedule on B200. Record differences in lane order,
  synchronization, overlap, and measured end-to-end timing before claiming the
  catalog useful for search.

#### Acceptance Criteria
- [ ] AC-5-1: Every operation signature reached by the retained FlashMLA
  program resolves from a frozen, correctness-checked B200 measurement with no
  fixture or default timing.
- [ ] AC-5-2: Compilation, allocation, initialization, argument setup, and
  first-use overhead are absent from recorded operation durations.
- [ ] AC-5-3: Renaming the reference program leaves all cost hits unchanged;
  changing a measured semantic/type condition causes an exact miss unless that
  condition was separately calibrated.
- [ ] AC-5-4: The frozen catalog builds a replayable numeric problem on a B200
  host, and that problem solves byte-identically on a host without CUDA or the
  profile database under the same deterministic solver version.
- [ ] AC-5-5: The B200 result is compared against the handwritten reference for
  schedule structure and measured end-to-end performance; unexplained missing
  synchronization or illegal overlap is a gate failure regardless of predicted
  makespan.
<!-- policy_ac:start -->
- [ ] Milestone MUST name a `#### Golden Reference` before implementation steps, with the source of truth and the observable functional points it determines. <!-- policy_ac: milestone_review-0 -->
- [ ] The gate request MUST show the Golden Reference's functional points exercised through the smallest real workflow, naming the retained evidence; one workflow MAY cover several ACs, and a new test requires a stated reachability gap. <!-- policy_ac: milestone_review-1 -->
- [ ] Touched tests and comments MUST be reviewed for redundancy: remove ones superseded by the retained workflow, and do not add source-shape or hypothetical-refactor guards unless that form is a public contract. <!-- policy_ac: milestone_review-2 -->
- [ ] A milestone that changes a public contract MUST list the owning `docs/spec/*.md` path in its `#### Related Files`; one that changes none lists no spec path. <!-- policy_ac: spec_impact-0 -->
<!-- policy_ac:end -->

## Deferred Work

The following work starts only after M5 review:

- a compact periodic solver for large iteration counts;
- search over microtiling, fusion, implementation choice, or pipeline depth;
- physical shared-memory packing and ring-buffer allocation;
- lowering `sync` to barrier participants, slots, phases, fences, and waits;
- emitting or rewriting CUDA/TIR from the selected lane program;
- automatic conversion from arbitrary TileFoundry HIR/TIR into the explicit
  warpgroup tile-operation program.

## Final Gate

<!-- final_gate:start -->
- [ ] Spec section MUST NOT enumerate test names; the pre-commit `spec-rules-lint` and `english-only` hooks already reject forbidden section headers, plan / milestone / task / PR / commit references, agent names, and non-English text. <!-- policy_final: spec_discipline-0 -->
- [ ] No touched C++/CUDA files in this plan — clang-format gate N/A <!-- policy_final: clang_format-na -->
<!-- final_gate:end -->
