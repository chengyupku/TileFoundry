---
type: FEAT
component: costmodel
target_repo: tilefoundry
---

# [FEAT][costmodel] Implement the B200 single-CTA cost model

## Description

The standalone `costmodel` package currently exposes API `(0, 2)`: callers
provide fixed `StageSpec` instances, an opaque `payload`, a `TimingOracle`, and
named resource capacities. Its list scheduler and finite CP-SAT backend can
place already-timed stages, but the package has no typed tile-operation input,
measured timing library, B200 hardware document, pipeline-depth search, periodic
solver, or calibrated workload frontend.

The approved core contract is now owned exclusively by
[cost-model](../spec/cost-model.md). Implementation must follow its exact
`T.* -> TileProgram -> build -> SearchProblem -> solve -> CostModelResult` data
path. The costmodel implementation remains standalone. A later optional
milestone defines a narrow TileFoundry-side dependency adapter, but it does not
integrate the selected result into HIR, TIR, scheduling decisions, or code
generation.

### Frozen Contract Rule

- `docs/spec/cost-model.md` is the only source of public and cross-module
  semantics.
- This development document MUST NOT redefine a public class, field, enum,
  function signature, unit, profile key, SQL row, status, or solver constraint.
- Implementation work MUST conform to the spec even when a shorter local design
  appears convenient.
- If implementation or B200 measurement proves a contract impossible or
  incorrect, work on the affected milestone stops. A separately reviewed spec
  amendment and affected schema/API version change must be approved before code
  continues. The implementation MUST NOT silently reinterpret the contract.
- Private helper names, OR-Tools variable names, CUDA source organization, and
  local algorithms remain implementation details when they preserve the
  contract.

### Scope

- Hardware: NVIDIA B200.
- Scheduling unit: one CTA resident on one SM.
- Timing backend: native CUDA JIT on a local B200.
- Initial logical workloads: GEMM, GQA decode, FlashAttention, and MLP.
- Primary prediction: integer-picosecond end-to-end CTA latency.
- Search dimensions: concrete program/tile variants, per-operation
  implementations, warp configurations, one configuration-wide pipeline depth,
  and implementation layout variants.
- Optimization resources: TMA, Tensor Core, CUDA Core, warp issue, calibrated
  global/shared/tensor/register data paths, in-flight limits, barriers, and
  static CTA capacities.

### Input And Output

The implementation entry is a `CostModelRequest` containing:

- one logical workload record;
- one or more concrete typed `TileProgram` variants built from `T.pipeline`,
  `T.copy`, `T.gemm`, `T.reduce`, and `T.elementwise`;
- an exact B200 hardware/calibration reference;
- finite implementation, warp, depth, and layout choices;
- an exact profile snapshot and require/JIT policy;
- deterministic solver controls.

`build` returns a replayable `SearchProblem` containing resolved integer timing,
constraints, static demand, profile provenance, and no CUDA/SQLite/callback
state. `solve` returns `CostModelResult`; a successful plan includes the selected
program/tile/warp/depth/implementations, source-op-to-phase timeline, per-loop
II, end-to-end time, buffers, utilization, profile provenance, and proof.

### Development Dependency Tree

```text
M0 package/API boundary and legacy isolation
`-- M1 typed programs, relations, and B200 hardware facts
    `-- M2 operation implementation and configuration composition
        |-- M3 SQLite profiles and local CUDA JIT runner
        `-- M4 build boundary and finite/periodic CP-SAT
            `-- M5 calibrated GEMM vertical slice
                `-- M6 GQA, FlashAttention, and MLP frontends
                    `-- M7 replay, packaging, and release validation
                        `-- M8 optional TileFoundry dependency adapter
```

M4 depends on both M2 and M3. The tree shows the dominant ownership path, not a
license to start M4 before timing resolution exists. M8 depends on the typed
program contract from M1 and the released standalone package from M7; it is a
one-way exporter and never makes the standalone package import TileFoundry.

### Planned Package Layout

```text
costmodel/
  pyproject.toml
  README.md
  schemas/
    hardware-v1.schema.json
    plan-v2.schema.json
    profile-snapshot-v1.schema.json
    program-v2.schema.json
    request-v2.schema.json
    result-v2.schema.json
    search-problem-v2.schema.json
  calibration/
    b200-hardware.json
  src/tilefoundry_costmodel/
    __init__.py
    api.py
    build.py
    cli.py
    errors.py
    language.py
    model.py
    program.py
    request.py
    result.py
    search_space.py
    tileop.py
    legacy/
      __init__.py
      model.py
      list_solver.py
      cpsat_solver.py
    hardware/
      model.py
      registry.py
      b200.py
    implementations/
      base.py
      registry.py
      b200/
        copy.py
        gemm.py
        reduce.py
        elementwise.py
        cuda/
          copy.cu
          gemm.cu
          reduce.cu
          elementwise.cu
    profiles/
      model.py
      schema.py
      store.py
      resolver.py
    profiler/
      base.py
      cuda/
        __init__.py
        runner.py
        benchmark_runner.cu
    solver/
      common.py
      finite.py
      model.py
      periodic.py
      search.py
    workloads/
      base.py
      model.py
      gemm.py
      gqa.py
      flash_attention.py
      mlp.py
  tests/
  benchmarks/
```

The optional M8 bridge is owned by the main TileFoundry package and is loaded
only by an explicit caller:

```text
src/tilefoundry/schedule/costmodel_adapter.py
tests/schedule/test_costmodel_adapter.py
```

### Code Standards

The rules in `docs/develop.md`, repository hooks, and the core spec are
mandatory. The package additionally follows these constraints:

- Python support is 3.11 and 3.12.
- Public and cross-module records are immutable and fully typed. `Any` is
  allowed only while decoding an external representation and must be converted
  before entering typed models.
- Quantity names include units: `_ps`, `_ns`, `_bytes`, `_khz`, `_mw`, `_slots`,
  or `_ticks`. A public bare `duration`, `size`, or `bandwidth` field is not
  allowed.
- CP-SAT receives only integer ticks and integer demands. Floating-point timing
  must not cross the `build` boundary.
- Stable IDs and JSON ordering cannot depend on object identity, hash
  randomization, dict insertion order, thread completion order, or OR-Tools
  variable spelling.
- `request.py` and `result.py` own shared immutable API records. `api.py`,
  `build.py`, and `solver/` must not create an import cycle. Type-only cycles are
  handled according to `docs/develop.md`; `TYPE_CHECKING` shims are forbidden.
- Profile-store writes use transactions, parameterized SQL, foreign keys,
  uniqueness constraints, and explicit connection ownership.
- OR-Tools and CUDA dependencies are optional and lazily imported at the call
  that needs them. Importing the package root requires neither dependency, a
  driver, nor a GPU.
- CUDA timing excludes compilation, allocation, initialization, argument setup,
  and first-use overhead. Every timed artifact passes numerical correctness
  before insertion.
- C++/CUDA files follow the repository `.clang-format`; formatting is limited to
  touched files.
- Comments explain local mechanics only. Durable constraints belong in
  `docs/spec/cost-model.md`; a necessary backlink cites the spec, never this
  development document.
- Verification targets observable workflows and hand-checkable optima. It does
  not lock private calls, source text, AST shape, object identity, or helper
  counts.

The standalone package gates are:

```sh
python3 -m ruff check costmodel/src costmodel/tests costmodel/benchmarks
python3 -m ruff format --check costmodel/src costmodel/tests costmodel/benchmarks
python3 -m mypy --strict costmodel/src
python3 -m pytest costmodel/tests --cov=tilefoundry_costmodel --cov-branch
```

The initial optional dependency boundaries are:

```toml
[project.optional-dependencies]
cpsat = ["ortools>=9.15,<10"]
cuda = ["cuda-python>=12.8,<13"]
test = [
    "mypy>=1.15,<2",
    "pytest>=8,<10",
    "pytest-cov>=6,<8",
    "ruff>=0.11,<1",
]
```

The base dependency list stays empty. NVRTC, a CUDA driver, and a B200 are
runtime prerequisites only for local JIT profiling.

### Calibration And Release Gates

- No unavailable B200 fact may become a positive schedulable capacity.
- Profile measurements with relative IQR above 50,000 ppm are rejected.
- The retained exact finite and periodic small-loop cases produce identical
  end-to-end schedules.
- Every workload's retained B200 matrix reaches median absolute percentage error
  at or below 15 percent and p90 absolute percentage error at or below 25
  percent.
- Across the retained matrix, the selected candidate ranks in the measured top
  three for at least 90 percent of cases.
- An exported `SearchProblem` reproduces byte-identical selected configuration
  and predicted timing on a non-GPU host under the same solver package version.

## Milestones

### Milestone M0: Establish The Package Boundary

#### Depends
- None

#### Golden Reference
- Source: [cost-model §1](../spec/cost-model.md#1-boundary-and-invariants),
  [cost-model §12](../spec/cost-model.md#12-public-orchestration-and-serialization),
  the current `costmodel` package root, and the retained finite-stage workflow in
  `costmodel/tests/test_solver.py`.
- Functional points: preserve observable API `(0, 2)` scheduling through the
  legacy namespace; establish the independent `(2, 0)` package surface, strict
  schemas, optional dependency boundaries, and deterministic serialization
  without importing OR-Tools or CUDA at package import.

#### Related Files
- `costmodel/README.md`
- `costmodel/pyproject.toml`
- `costmodel/src/tilefoundry_costmodel/__init__.py`
- `costmodel/src/tilefoundry_costmodel/errors.py`
- `costmodel/src/tilefoundry_costmodel/request.py`
- `costmodel/src/tilefoundry_costmodel/result.py`
- `costmodel/src/tilefoundry_costmodel/legacy/__init__.py`
- `costmodel/src/tilefoundry_costmodel/legacy/model.py`
- `costmodel/src/tilefoundry_costmodel/legacy/list_solver.py`
- `costmodel/src/tilefoundry_costmodel/legacy/cpsat_solver.py`
- `costmodel/schemas/hardware-v1.schema.json`
- `costmodel/schemas/plan-v2.schema.json`
- `costmodel/schemas/profile-snapshot-v1.schema.json`
- `costmodel/schemas/program-v2.schema.json`
- `costmodel/schemas/request-v2.schema.json`
- `costmodel/schemas/result-v2.schema.json`
- `costmodel/schemas/search-problem-v2.schema.json`
- `costmodel/tests/test_solver.py`
- `costmodel/tests/test_api.py`

#### Plan
- [x] step 0.1 Move the current fixed-stage models, list scheduler, and finite
  CP-SAT implementation into `tilefoundry_costmodel.legacy` while preserving
  their observable inputs, statuses, timing, capacity behavior, and API version.
- [x] step 0.2 Add the package exception hierarchy and all schema/API constants;
  reserve the root namespace for the exact primary exports in the core spec.
- [x] step 0.3 Create immutable `SolverOptions`, `CostModelRequest`, result, and
  plan records in their owning modules without importing solver backends.
- [x] step 0.4 Implement strict schema loaders/serializers that reject unknown
  versions and fields and produce canonical JSON ordering.
- [x] step 0.5 Restore and extend package metadata for explicit `src` discovery,
  empty base dependencies, CP-SAT/CUDA/test extras, Ruff, strict mypy, coverage,
  and the CLI entry point.
- [x] step 0.6 Replace README fixed-stage guidance with the profile-build-solve
  workflow while keeping a clearly labeled legacy migration entry.
- [x] step 0.7 Retain the smallest existing fixed-stage solve as legacy evidence;
  remove redundant checks that assert private implementation shape.

#### Acceptance Criteria
- [x] AC-0-1: Existing `(0, 2)` callers can import the legacy namespace and
  obtain the retained finite-stage scheduling behavior.
- [x] AC-0-2: Importing the `(2, 0)` package root succeeds without OR-Tools,
  CUDA Python, a CUDA driver, or a GPU.
- [x] AC-0-3: Every public schema round-trips deterministically and rejects an
  unknown version or field.
- [x] AC-0-4: Base, CP-SAT, CUDA, and test extras install with their documented
  dependency boundaries.
- [x] AC-0-5: Package modules have no runtime import cycle and pass the standalone
  format, lint, strict-type, and retained legacy workflow gates.
<!-- policy_ac:start -->
- [x] Milestone MUST name a `#### Golden Reference` before implementation steps, with the source of truth and the observable functional points it determines. <!-- policy_ac: milestone_review-0 -->
- [x] The gate request MUST show the Golden Reference's functional points exercised through the smallest real workflow, naming the retained evidence; one workflow MAY cover several ACs, and a new test requires a stated reachability gap. <!-- policy_ac: milestone_review-1 -->
- [x] Touched tests and comments MUST be reviewed for redundancy: remove ones superseded by the retained workflow, and do not add source-shape or hypothetical-refactor guards unless that form is a public contract. <!-- policy_ac: milestone_review-2 -->
<!-- policy_ac:end -->

### Milestone M1: Implement Typed Programs And B200 Facts

#### Depends
- M0

#### Golden Reference
- Source: [cost-model §2](../spec/cost-model.md#2-common-values),
  [cost-model §3](../spec/cost-model.md#3-logical-workloads),
  [cost-model §4](../spec/cost-model.md#4-typed-tile-program),
  [cost-model §4.2](../spec/cost-model.md#42-tilefoundry-adapter-boundary),
  [cost-model §5](../spec/cost-model.md#5-search-request),
  [cost-model §6](../spec/cost-model.md#6-b200-hardware), and the NVIDIA CUDA C++
  Programming Guide at `https://docs.nvidia.com/cuda/cuda-c-programming-guide/`.
- Functional points: typed Python and JSON inputs produce the same immutable
  program; all program/value/loop/relation/barrier invariants fail before
  lowering;
  B200 temporal and static resources have stable IDs, units, capacities, and
  auditable provenance.

#### Related Files
- `costmodel/src/tilefoundry_costmodel/model.py`
- `costmodel/src/tilefoundry_costmodel/program.py`
- `costmodel/src/tilefoundry_costmodel/language.py`
- `costmodel/src/tilefoundry_costmodel/search_space.py`
- `costmodel/src/tilefoundry_costmodel/request.py`
- `costmodel/src/tilefoundry_costmodel/workloads/base.py`
- `costmodel/src/tilefoundry_costmodel/workloads/model.py`
- `costmodel/src/tilefoundry_costmodel/hardware/model.py`
- `costmodel/src/tilefoundry_costmodel/hardware/registry.py`
- `costmodel/src/tilefoundry_costmodel/hardware/b200.py`
- `costmodel/calibration/b200-hardware.json`
- `costmodel/tests/test_program.py`
- `costmodel/tests/test_hardware.py`

#### Plan
- [x] step 1.1 Implement shared IDs, enums, named shapes, tensor descriptors,
  logical workload records, warp choices, and search-space validation with the
  exact units and discriminators from the spec.
- [x] step 1.2 Implement `TileValue`, loop/domain records, typed operations,
  `AlignedRelation`, `EndpointRelation`, `LoopBarrier`, explicit value
  dependencies, and `TileProgram` construction-time validation, including
  finite relation expansion, relation-level acyclicity, barrier acyclicity, and
  external-input ownership.
- [x] step 1.3 Implement pure `T.*` helpers and strict program JSON parsing over
  the same constructors; no Python source, callable, HIR, or opaque payload may
  enter the model.
- [x] step 1.4 Add workload-frontend protocols and deterministic catalog
  resolution without implementing workload-specific programs yet.
- [x] step 1.5 Implement hardware model/catalog exact resolution and stable B200
  temporal/static resource IDs.
- [x] step 1.6 Populate `b200-hardware.json` only with vendor, measured, derived,
  or conservative values and explicit conditions; required unavailable facts
  remain unschedulable rather than receiving placeholders.
- [x] step 1.7 Retain one typed program workflow that exercises Python/JSON
  equivalence, program validation, exact hardware lookup, and temporal/static
  type separation.
- [x] step 1.8 Retain hand-checkable relation workflows for aligned recurrence,
  endpoint edges across loop boundaries, explicit loop barriers, and rejection
  of an implicit schedule-tree ordering.

#### Acceptance Criteria
- [x] AC-1-1: Python `T.*` construction and strict JSON parsing produce
  byte-identical canonical `TileProgram` output.
- [x] AC-1-2: Invalid discriminators, IDs, value edges, loop domains, GEMM axes,
  reduction axes, relation shapes, loop barriers, and broadcasting fail before
  implementation selection.
- [x] AC-1-3: Exact B200 lookup exposes every required temporal/static resource
  with unit, provenance, condition, schema, and calibration identity.
- [x] AC-1-4: A nearby hardware name or different calibration never satisfies an
  exact reference.
- [x] AC-1-5: No unavailable or wrong-class resource fact can enter a schedulable
  configuration.
- [x] AC-1-6: Endpoint relations are single edges, loop barriers are explicit
  all-instance control constraints, and neither is silently substituted for the
  other.
- [x] AC-1-7: The typed relation records and schema discriminators round-trip
  deterministically through Python and JSON.
<!-- policy_ac:start -->
- [x] Milestone MUST name a `#### Golden Reference` before implementation steps, with the source of truth and the observable functional points it determines. <!-- policy_ac: milestone_review-0 -->
- [x] The gate request MUST show the Golden Reference's functional points exercised through the smallest real workflow, naming the retained evidence; one workflow MAY cover several ACs, and a new test requires a stated reachability gap. <!-- policy_ac: milestone_review-1 -->
- [x] Touched tests and comments MUST be reviewed for redundancy: remove ones superseded by the retained workflow, and do not add source-shape or hypothetical-refactor guards unless that form is a public contract. <!-- policy_ac: milestone_review-2 -->
<!-- policy_ac:end -->

### Milestone M2: Implement Operation Lowering And Composition

> Status: **CLOSED** after review approval. M3 is the next permitted milestone;
> M4 and later milestones remain unopened.

#### Depends
- M1

#### Golden Reference
- Source: [cost-model §7](../spec/cost-model.md#7-operation-lowering-and-phases)
  and [cost-model §8.1](../spec/cost-model.md#81-canonical-profile-identity).
- Functional points: one typed operation selects a paired lowering/benchmark
  implementation; phases retain source op, component, warp, resource, and
  profile identity; availability bindings compose value edges without assuming
  completion; asynchronous II and latency phases begin together.

#### Related Files
- `costmodel/src/tilefoundry_costmodel/tileop.py`
- `costmodel/src/tilefoundry_costmodel/build.py`
- `costmodel/src/tilefoundry_costmodel/implementations/base.py`
- `costmodel/src/tilefoundry_costmodel/implementations/registry.py`
- `costmodel/src/tilefoundry_costmodel/profiles/model.py`
- `costmodel/src/tilefoundry_costmodel/profiler/base.py`
- `costmodel/tests/test_tileop.py`
- `costmodel/tests/test_configuration_builder.py`

#### Plan
- [x] step 2.1 Implement temporal/static demands, phase domains/templates,
  typed end-to-start relations, explicit loop barriers, start alignments, buffer
  templates, loop templates, and configuration templates with complete
  ID/resource/warp validation.
- [x] step 2.2 Implement canonical `TileOpSignature`, profile queries,
  requirements, benchmark fingerprints, and profile-key hashing from typed
  operation/value records only.
- [x] step 2.3 Implement produced/consumed availability and value-storage
  records; compose each semantic value edge through one exact matching
  availability.
- [x] step 2.4 Implement the lowering/provider pair and catalog. Reject duplicate
  implementation pairs, duplicate provider IDs, unsupported warp-role sharing,
  and a provider that cannot serve every emitted query.
- [x] step 2.5 Implement `ConfigurationBuilder` enumeration over program, warp,
  depth, layout, and per-op implementation choices; canonicalize IDs, deduplicate
  equivalent configurations, and enforce the candidate bound.
- [x] step 2.6 Derive ring/static buffers and static capacity exactly once by
  value ID; canonicalize a configuration with no ring storage to depth 1.
- [x] step 2.7 Retain a synthetic in-memory implementation workflow with no fake
  production timings to demonstrate source-op traceability, explicit warp
  binding, ordered/complete availability selection, and aligned issue/latency
  phases.

#### Acceptance Criteria
- [x] AC-2-1: Every generated phase maps to exactly one typed source operation,
  implementation, component, legal warp set, resource set, and profile query.
- [x] AC-2-2: An issue interval and its latency interval have equal starts;
  modeled readiness is latency rather than `II + latency`.
- [x] AC-2-3: A following GEMM can select ordered availability while an epilogue
  selects completion, with no solver- or frontend-side operation branch.
- [x] AC-2-4: Ring depth changes buffer slots, static bytes, configuration ID,
  and profile keys without double-charging storage.
- [x] AC-2-5: Candidate enumeration and tie identity are invariant to caller
  tuple order and reject missing/ambiguous availability bindings.
- [x] AC-2-6: Lowering preserves aligned and endpoint relations exactly and
  carries explicit loop barriers without inferring barriers from phase order.
<!-- policy_ac:start -->
- [x] Milestone MUST name a `#### Golden Reference` before implementation steps, with the source of truth and the observable functional points it determines. <!-- policy_ac: milestone_review-0 -->
- [x] The gate request MUST show the Golden Reference's functional points exercised through the smallest real workflow, naming the retained evidence; one workflow MAY cover several ACs, and a new test requires a stated reachability gap. <!-- policy_ac: milestone_review-1 -->
- [x] Touched tests and comments MUST be reviewed for redundancy: remove ones superseded by the retained workflow, and do not add source-shape or hypothetical-refactor guards unless that form is a public contract. <!-- policy_ac: milestone_review-2 -->
<!-- policy_ac:end -->

### Milestone M3: Implement Profiles And Local CUDA JIT

> Status: **CLOSED** after host, artifact, and real B200 review gates. M4 and
> later milestones remain unopened.

#### Depends
- M2

#### Golden Reference
- Source: [cost-model §8](../spec/cost-model.md#8-timing-profiles), the CUDA
  Runtime Event API at
  `https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EVENT.html`, and
  the PTX ISA at
  `https://docs.nvidia.com/cuda/parallel-thread-execution/`.
- Functional points: exact keys bind typed op, implementation, component,
  hardware, code, and environment; latency and II are distinct metrics from one
  measurement; setup is excluded; writes are atomic; an exact cache hit needs no
  CUDA installation.

#### Related Files
- `costmodel/src/tilefoundry_costmodel/profiles/model.py`
- `costmodel/src/tilefoundry_costmodel/profiles/schema.py`
- `costmodel/src/tilefoundry_costmodel/profiles/store.py`
- `costmodel/src/tilefoundry_costmodel/profiles/resolver.py`
- `costmodel/src/tilefoundry_costmodel/profiler/base.py`
- `costmodel/src/tilefoundry_costmodel/profiler/cuda/__init__.py`
- `costmodel/src/tilefoundry_costmodel/profiler/cuda/runner.py`
- `costmodel/src/tilefoundry_costmodel/profiler/cuda/benchmark_runner.cu`
- `costmodel/src/tilefoundry_costmodel/implementations/b200/copy.py`
- `costmodel/src/tilefoundry_costmodel/implementations/b200/cuda/copy.cu`
- `costmodel/src/tilefoundry_costmodel/cli.py`
- `costmodel/tests/test_profile_store.py`
- `costmodel/tests/test_profile_resolver.py`
- `costmodel/tests/test_cuda_profiler.py`

#### Plan
- [x] step 3.1 Implement profile environments, measurement policy, aggregate/raw
  measurement records, resolved timings, p50/p90 selection, and stability
  rejection.
- [x] step 3.2 Implement the exact SQLite schema with migrations, foreign keys,
  atomic inserts, one environment per snapshot, immutable frozen revisions,
  deterministic import/export, and typed corruption/conflict errors.
- [x] step 3.3 Implement require-only and JIT-on-miss resolution over canonical
  deduplicated keys; require-only reports all missing keys without mutation.
- [x] step 3.4 Implement the local CUDA runner with lazy CUDA imports, exact B200
  environment capture, NVRTC compilation/cache, CUDA events, warmup, device-side
  repetition, synchronization, and retained raw samples.
- [x] step 3.5 Implement dependency-chain latency and independent-chain
  initiation-interval cases. Verify source/option hashes before compilation and
  validate outputs before summarizing or inserting.
- [x] step 3.6 Implement one minimal correctness-checked B200 copy implementation
  and provider as the real vertical runner workflow; do not add a roofline or
  fake measurement path.
- [x] step 3.7 Add CLI snapshot create/inspect/freeze/export/import and explicit
  profile commands without importing TileFoundry compiler modules.

#### Acceptance Criteria
- [x] AC-3-1: Require-only miss returns every missing key and leaves the database
  byte-for-byte unchanged at the logical snapshot level.
- [x] AC-3-2: The same JIT miss compiles, measures, validates, inserts one atomic
  measurement, and becomes an exact cache hit under require-only mode.
- [x] AC-3-3: Latency and II select different aggregates from one key and
  measurement; neither metric substitutes for a missing other metric.
- [x] AC-3-4: Compilation, allocation, initialization, argument setup, and
  first-use work are absent from recorded CUDA event intervals.
- [x] AC-3-5: Failed compilation, execution, stability, correctness, or
  transaction leaves no usable partial measurement.
- [x] AC-3-6: Export/import preserves canonical keys, aggregates, optional raw
  samples, environment, provenance, membership, and schema deterministically.
- [x] AC-3-7: A host without CUDA can open a frozen snapshot and resolve every
  existing timing without importing CUDA dependencies.
<!-- policy_ac:start -->
- [x] Milestone MUST name a `#### Golden Reference` before implementation steps, with the source of truth and the observable functional points it determines. <!-- policy_ac: milestone_review-0 -->
- [x] The gate request MUST show the Golden Reference's functional points exercised through the smallest real workflow, naming the retained evidence; one workflow MAY cover several ACs, and a new test requires a stated reachability gap. <!-- policy_ac: milestone_review-1 -->
- [x] Touched tests and comments MUST be reviewed for redundancy: remove ones superseded by the retained workflow, and do not add source-shape or hypothetical-refactor guards unless that form is a public contract. <!-- policy_ac: milestone_review-2 -->
<!-- policy_ac:end -->

### Milestone M4: Implement Build And CP-SAT

#### Depends
- M2
- M3

#### Golden Reference
- Source: [cost-model §9](../spec/cost-model.md#9-build-and-search-problem),
  [cost-model §10](../spec/cost-model.md#10-solver-contract), the retained exact
  finite behavior in `costmodel/src/tilefoundry_costmodel/cpsat_solver.py`, and
  OR-Tools interval scheduling at
  `https://developers.google.com/optimization/scheduling`.
- Functional points: build closes every query before solve; finite CP-SAT honors
  aligned/endpoint/barrier/alignment/resource/buffer constraints; periodic
  CP-SAT finds a feasible fixed II without full long-loop expansion; global
  search returns the best proved candidate and complete bounds/status.

#### Related Files
- `costmodel/src/tilefoundry_costmodel/build.py`
- `costmodel/src/tilefoundry_costmodel/api.py`
- `costmodel/src/tilefoundry_costmodel/request.py`
- `costmodel/src/tilefoundry_costmodel/result.py`
- `costmodel/src/tilefoundry_costmodel/solver/common.py`
- `costmodel/src/tilefoundry_costmodel/solver/model.py`
- `costmodel/src/tilefoundry_costmodel/solver/finite.py`
- `costmodel/src/tilefoundry_costmodel/solver/periodic.py`
- `costmodel/src/tilefoundry_costmodel/solver/search.py`
- `costmodel/src/tilefoundry_costmodel/cli.py`
- `costmodel/tests/test_build.py`
- `costmodel/tests/test_finite_solver.py`
- `costmodel/tests/test_periodic_solver.py`
- `costmodel/tests/test_search.py`

#### Plan
- [ ] step 4.1 Implement timed `Phase`, `Configuration`, and `SearchProblem`
  records and strict replay serialization with no protocol, query, float model
  timing, store, CUDA, or OR-Tools value.
- [ ] step 4.2 Implement public `build` in the exact validation/enumeration/
  profile-resolution/quantization order; copy request solver controls into the
  immutable problem and never invoke OR-Tools.
- [ ] step 4.3 Implement finite CP-SAT interval variables, upward-rounded delays,
  aligned and endpoint relation constraints, explicit loop-barrier extrema,
  same-start equalities, `NoOverlap`/`Cumulative`, buffer reuse, static checks,
  makespan objective, and independent sensitivity re-solve.
- [ ] step 4.4 Implement fixed-II periodic lower/upper bounds, integer
  feasibility search, required replication radius, steady and boundary models,
  compact export, and exact finite reconstruction for retained small loops.
- [ ] step 4.5 Compose multiple long loop regions through explicit endpoint
  edges or loop barriers; reject a periodic candidate whose region ordering is
  not representable, and export per-loop II/prologue/epilogue/span plus global
  end-to-end time.
- [ ] step 4.6 Implement `SearchCoordinator` candidate budgets, outer deadline,
  global lower bound/gap, deterministic comparison, rejection diagnostics, and
  optimal/feasible/timeout/infeasible semantics.
- [ ] step 4.7 Implement public `solve`, `evaluate`, problem/result/plan JSON,
  timeline rendering, and CLI build/solve/search commands with exact exit codes.
- [ ] step 4.8 Retain hand-computable workflows for independent TMA/Tensor/CUDA
  overlap, shared-capacity blocking, ordered asynchronous issue, depth reuse,
  periodic equivalence, global candidate selection, and timeout proof state.

#### Acceptance Criteria
- [ ] AC-4-1: `SearchProblem` export/import reproduces the same solve without
  CUDA, SQLite, frontends, implementation catalogs, or typed-op inspection.
- [ ] AC-4-2: TMA, Tensor Core, and CUDA Core intervals overlap exactly when
  dependencies and calibrated shared resources permit.
- [ ] AC-4-3: No selected schedule exceeds temporal capacity, static capacity,
  in-flight limits, explicit loop barriers, warp bindings, or buffer lifetime;
  endpoint dependencies do not impose an unintended whole-region barrier.
- [ ] AC-4-4: Increasing depth allocates corresponding static bytes and never
  overwrites a live slot.
- [ ] AC-4-5: Periodic and exact finite schedules agree on retained small-loop
  end-to-end timing; large loops do not create one phase object per iteration.
- [ ] AC-4-6: Global search selects the minimum legal solved end-to-end result
  under the specified tie order and reports incumbent, bound, gap, and all
  rejection categories.
- [ ] AC-4-7: Repeated deterministic complete solve produces byte-identical
  result JSON.
- [ ] AC-4-8: Exact finite relation expansion and periodic boundary handling
  agree on retained aligned, endpoint, and loop-barrier workflows.
<!-- policy_ac:start -->
- [ ] Milestone MUST name a `#### Golden Reference` before implementation steps, with the source of truth and the observable functional points it determines. <!-- policy_ac: milestone_review-0 -->
- [ ] The gate request MUST show the Golden Reference's functional points exercised through the smallest real workflow, naming the retained evidence; one workflow MAY cover several ACs, and a new test requires a stated reachability gap. <!-- policy_ac: milestone_review-1 -->
- [ ] Touched tests and comments MUST be reviewed for redundancy: remove ones superseded by the retained workflow, and do not add source-shape or hypothetical-refactor guards unless that form is a public contract. <!-- policy_ac: milestone_review-2 -->
<!-- policy_ac:end -->

### Milestone M5: Calibrate The GEMM Vertical Slice

#### Depends
- M4

#### Golden Reference
- Source: [cost-model §3](../spec/cost-model.md#3-logical-workloads),
  [cost-model §4](../spec/cost-model.md#4-typed-tile-program), the B200 timing
  protocol established in M3, and numerical comparison against a high-precision
  host GEMM reference before timing acceptance.
- Functional points: generate concrete `T.copy/T.gemm/epilogue/store` programs,
  lower legal B200 TMA/WGMMA/CUDA implementations, search tile/warp/depth/layout,
  and compare prediction/ranking against complete-CTA CUDA measurements.

#### Related Files
- `costmodel/src/tilefoundry_costmodel/workloads/gemm.py`
- `costmodel/src/tilefoundry_costmodel/implementations/b200/copy.py`
- `costmodel/src/tilefoundry_costmodel/implementations/b200/gemm.py`
- `costmodel/src/tilefoundry_costmodel/implementations/b200/elementwise.py`
- `costmodel/src/tilefoundry_costmodel/implementations/b200/cuda/copy.cu`
- `costmodel/src/tilefoundry_costmodel/implementations/b200/cuda/gemm.cu`
- `costmodel/src/tilefoundry_costmodel/implementations/b200/cuda/elementwise.cu`
- `costmodel/tests/test_gemm_workload.py`
- `costmodel/benchmarks/b200_gemm.py`

#### Plan
- [ ] step 5.1 Implement deterministic GEMM program generation for each explicit
  CTA tile, including K-loop domains, A/B copies, accumulator dependency,
  optional epilogue, output store, values, and buffer lifetimes.
- [ ] step 5.2 Implement B200 TMA copy, ordered WGMMA, completion-bound WGMMA
  where supported, CUDA epilogue, and store lowerings with exact warp roles,
  named availabilities, aligned timing phases, and resource demands.
- [ ] step 5.3 Implement correctness-checked latency/II CUDA providers for every
  emitted component and complete-CTA benchmarks for independent validation.
- [ ] step 5.4 Calibrate shared temporal demands and in-flight capacities against
  measured overlap experiments rather than assuming independent engines.
- [ ] step 5.5 Populate and freeze a B200 GEMM snapshot for the retained
  shape/dtype/tile/warp/depth/layout matrix.
- [ ] step 5.6 Compare predicted end-to-end time, chosen candidate, and top-three
  ranking against full CTA measurements; document calibrated and unsupported
  regions in the package README.

#### Acceptance Criteria
- [ ] AC-5-1: A GEMM request produces stable legal programs/configurations and a
  complete source-op-to-phase/warp/profile timeline without HIR or codegen.
- [ ] AC-5-2: Every timed phase and complete CTA artifact passes numerical
  correctness before its measurement is committed.
- [ ] AC-5-3: Search can choose different tile, warp, implementation, and depth
  as shape or measured data changes.
- [ ] AC-5-4: Exported frozen profiles reproduce the selected configuration and
  prediction on a non-GPU host.
- [ ] AC-5-5: The retained GEMM matrix meets the stated latency-error and
  top-three ranking gates.
<!-- policy_ac:start -->
- [ ] Milestone MUST name a `#### Golden Reference` before implementation steps, with the source of truth and the observable functional points it determines. <!-- policy_ac: milestone_review-0 -->
- [ ] The gate request MUST show the Golden Reference's functional points exercised through the smallest real workflow, naming the retained evidence; one workflow MAY cover several ACs, and a new test requires a stated reachability gap. <!-- policy_ac: milestone_review-1 -->
- [ ] Touched tests and comments MUST be reviewed for redundancy: remove ones superseded by the retained workflow, and do not add source-shape or hypothetical-refactor guards unless that form is a public contract. <!-- policy_ac: milestone_review-2 -->
<!-- policy_ac:end -->

### Milestone M6: Add GQA, FlashAttention, And MLP

#### Depends
- M5

#### Golden Reference
- Source: [cost-model §3](../spec/cost-model.md#3-logical-workloads), typed
  operation semantics in [cost-model §4](../spec/cost-model.md#4-typed-tile-program),
  the calibrated implementation/profile path from M5, and high-precision host
  references for each workload.
- Functional points: each frontend emits only typed operations and explicit
  value/loop edges; existing generic build/solver code remains unchanged; any
  needed B200 reduce/elementwise implementation is paired with a validated
  provider; complete-CTA measurements establish prediction accuracy.

#### Related Files
- `costmodel/src/tilefoundry_costmodel/workloads/gqa.py`
- `costmodel/src/tilefoundry_costmodel/workloads/flash_attention.py`
- `costmodel/src/tilefoundry_costmodel/workloads/mlp.py`
- `costmodel/src/tilefoundry_costmodel/implementations/b200/reduce.py`
- `costmodel/src/tilefoundry_costmodel/implementations/b200/elementwise.py`
- `costmodel/src/tilefoundry_costmodel/implementations/b200/cuda/reduce.cu`
- `costmodel/src/tilefoundry_costmodel/implementations/b200/cuda/elementwise.cu`
- `costmodel/tests/test_gqa_workload.py`
- `costmodel/tests/test_flash_attention_workload.py`
- `costmodel/tests/test_mlp_workload.py`
- `costmodel/benchmarks/b200_gqa.py`
- `costmodel/benchmarks/b200_flash_attention.py`
- `costmodel/benchmarks/b200_mlp.py`

#### Plan
- [ ] step 6.1 Implement GQA decode programs with query setup, full K/V loop,
  explicit one-time tail operations, QK/PV GEMMs, online max/sum/output state,
  normalize, and store.
- [ ] step 6.2 Implement FlashAttention programs with one-time Q load, full K/V
  loop, explicit tail, QK, causal-mask operation, online softmax operations, PV,
  normalize, and store.
- [ ] step 6.3 Implement MLP programs with up/down GEMM loop regions, activation,
  explicit inter-region value lifetime, output epilogue/store, and SWIGLU gate/
  value paths.
- [ ] step 6.4 Implement and calibrate missing B200 reduce/elementwise
  lowerings/providers. Reuse copy/GEMM implementations by typed signature rather
  than duplicating workload-specific timing code.
- [ ] step 6.5 Add correctness-checked phase and complete-CTA B200 workflows,
  populate frozen snapshots, and calibrate shared resource demands for each
  workload.
- [ ] step 6.6 Run retained shape/dtype/tile/depth matrices and publish supported,
  calibrated, and unsupported regions plus prediction/ranking reports.

#### Acceptance Criteria
- [ ] AC-6-1: Each workload produces deterministic typed programs and a complete
  `SearchProblem` without changing generic build or solver behavior.
- [ ] AC-6-2: Full-tile and tail operations have distinct concrete signatures
  and profiles; a tail never reuses full-tile timing.
- [ ] AC-6-3: MLP exports two ordered loop timings and one global end-to-end time
  without fully expanding long loops.
- [ ] AC-6-4: Every selected phase maps to a typed operation, implementation,
  legal warp set, resources, measurement, and placement.
- [ ] AC-6-5: Every timed artifact passes numerical correctness before insertion.
- [ ] AC-6-6: Each retained workload matrix meets the stated latency-error and
  top-three ranking gates.
<!-- policy_ac:start -->
- [ ] Milestone MUST name a `#### Golden Reference` before implementation steps, with the source of truth and the observable functional points it determines. <!-- policy_ac: milestone_review-0 -->
- [ ] The gate request MUST show the Golden Reference's functional points exercised through the smallest real workflow, naming the retained evidence; one workflow MAY cover several ACs, and a new test requires a stated reachability gap. <!-- policy_ac: milestone_review-1 -->
- [ ] Touched tests and comments MUST be reviewed for redundancy: remove ones superseded by the retained workflow, and do not add source-shape or hypothetical-refactor guards unless that form is a public contract. <!-- policy_ac: milestone_review-2 -->
<!-- policy_ac:end -->

### Milestone M7: Validate Replay And Package Delivery

#### Depends
- M6

#### Golden Reference
- Source: [cost-model §11](../spec/cost-model.md#11-result-contract),
  [cost-model §12](../spec/cost-model.md#12-public-orchestration-and-serialization),
  frozen B200 snapshots from M5/M6, and deterministic synthetic solver workflows
  from M4.
- Functional points: one documented GPU workflow profiles/builds/solves every
  calibrated workload; one non-GPU workflow solves exported problems
  byte-identically; package/API/CLI distinguish every typed outcome and install
  cleanly under each optional dependency boundary.

#### Related Files
- `costmodel/README.md`
- `costmodel/pyproject.toml`
- `costmodel/src/tilefoundry_costmodel/__init__.py`
- `costmodel/src/tilefoundry_costmodel/api.py`
- `costmodel/src/tilefoundry_costmodel/cli.py`
- `costmodel/src/tilefoundry_costmodel/result.py`
- `costmodel/tests/test_api.py`
- `costmodel/tests/test_cli.py`
- `costmodel/benchmarks/validate_b200.py`

#### Plan
- [ ] step 7.1 Consolidate B200 prediction, measurement, absolute percentage
  error, ranking, environment, key, and source-fingerprint output into one
  machine-readable validation report.
- [ ] step 7.2 Exercise documented GPU profile/build/search workflows for GEMM,
  GQA decode, FlashAttention, and MLP from clean draft snapshots.
- [ ] step 7.3 Export frozen snapshots and search problems; reproduce
  byte-identical selected configurations and predicted timings in a clean
  non-GPU CP-SAT environment.
- [ ] step 7.4 Verify Python API and CLI mappings for optimal, feasible,
  timeout-with-incumbent, timeout-without-incumbent, infeasible, unsupported,
  missing-profile, profile-failed, and invalid-input outcomes.
- [ ] step 7.5 Verify clean base, CP-SAT, CUDA, and test-extra installs plus
  Python 3.11/3.12 format, lint, strict type, branch coverage, package data,
  SQLite, CUDA/B200, replay, and CLI workflows.
- [ ] step 7.6 Remove obsolete duplicate compatibility checks and private-shape
  assertions after the retained legacy and end-to-end workflows cover their
  observable behavior.

#### Acceptance Criteria
- [ ] AC-7-1: Clean GPU setup produces a complete validated plan for every
  calibrated workload through one documented workflow.
- [ ] AC-7-2: Clean non-GPU setup solves exported problems without CUDA/SQLite
  access and reproduces byte-identical results.
- [ ] AC-7-3: The complete retained matrix meets correctness, latency-error,
  ranking, stability, and reproducibility gates.
- [ ] AC-7-4: Base import and every optional install boundary work without
  undeclared dependencies.
- [ ] AC-7-5: API, JSON, CLI, README, package exports, and generated reports agree
  with the frozen core contract.
- [ ] AC-7-6: Every public failure category is distinguishable through stable
  status, exit code, missing keys, proof state, and diagnostics.
<!-- policy_ac:start -->
- [ ] Milestone MUST name a `#### Golden Reference` before implementation steps, with the source of truth and the observable functional points it determines. <!-- policy_ac: milestone_review-0 -->
- [ ] The gate request MUST show the Golden Reference's functional points exercised through the smallest real workflow, naming the retained evidence; one workflow MAY cover several ACs, and a new test requires a stated reachability gap. <!-- policy_ac: milestone_review-1 -->
- [ ] Touched tests and comments MUST be reviewed for redundancy: remove ones superseded by the retained workflow, and do not add source-shape or hypothetical-refactor guards unless that form is a public contract. <!-- policy_ac: milestone_review-2 -->
<!-- policy_ac:end -->

### Milestone M8: Add The Optional TileFoundry Dependency Adapter

#### Depends
- M1
- M7

#### Golden Reference
- Source: [cost-model §4.2](../spec/cost-model.md#42-tilefoundry-adapter-boundary),
  [cost-model §4](../spec/cost-model.md#4-typed-tile-program), the existing
  `TileGraph` extraction in `src/tilefoundry/analysis/poly.py`, and the current
  pipeline program boundary in `src/tilefoundry/schedule/pipeline/program.py`.
- Functional points: an explicit main-package adapter projects only an exact
  one-dimensional ISL relation into `AlignedRelation` or `EndpointRelation`,
  reports unsupported relations without mutation, and leaves standalone
  costmodel imports independent of TileFoundry.

#### Related Files
- `src/tilefoundry/schedule/costmodel_adapter.py`
- `tests/schedule/test_costmodel_adapter.py`
- `docs/spec/cost-model.md`
- `docs/spec/schedule.md`

#### Plan
- [ ] step 8.1 Implement the adapter-side immutable bindings and
  `CostModelAdapterError` / `UnsupportedDependencyRelationError` records with
  the exact signatures in [cost-model §4.2](../spec/cost-model.md#42-tilefoundry-adapter-boundary).
- [ ] step 8.2 Implement structured ISL projection over an explicitly selected
  statement axis. Compare the finite projected relation against the canonical
  aligned or endpoint relation before constructing any typed dependency.
- [ ] step 8.3 Reject multi-dimensional, skewed, non-uniform, many-to-many,
  empty, missing, duplicate, or ambiguous relations with stable diagnostics;
  never infer a barrier from schedule-tree order.
- [ ] step 8.4 Keep the adapter optional and one-way. The main package may
  import the standalone package only from the explicit adapter entry point; the
  standalone package MUST remain importable without TileFoundry or ISL.
- [ ] step 8.5 Retain finite-domain equivalence workflows for aligned recurrence,
  endpoint loop boundaries, explicit loop barriers, query-order determinism,
  and unsupported relation failures. Verify that existing `schedule(module,
  function)` behavior is unchanged.

#### Acceptance Criteria
- [ ] AC-8-1: Every accepted projection round-trips to the same finite pair set
  as the source ISL relation for the selected axis binding.
- [ ] AC-8-2: An endpoint result is exactly one edge and never silently becomes
  an all-instance barrier.
- [ ] AC-8-3: Unsupported ISL relations identify source statement IDs and a
  canonical relation diagnostic, with no partial typed program emitted.
- [ ] AC-8-4: Importing `tilefoundry_costmodel` does not import TileFoundry,
  ISL, CUDA, or schedule modules; importing the existing TileFoundry package
  does not require the optional adapter dependency.
- [ ] AC-8-5: Existing pipeline and partition schedule tests pass unchanged.
<!-- policy_ac:start -->
- [ ] Milestone MUST name a `#### Golden Reference` before implementation steps, with the source of truth and the observable functional points it determines. <!-- policy_ac: milestone_review-0 -->
- [ ] The gate request MUST show the Golden Reference's functional points exercised through the smallest real workflow, naming the retained evidence; one workflow MAY cover several ACs, and a new test requires a stated reachability gap. <!-- policy_ac: milestone_review-1 -->
- [ ] Touched tests and comments MUST be reviewed for redundancy: remove ones superseded by the retained workflow, and do not add source-shape or hypothetical-refactor guards unless that form is a public contract. <!-- policy_ac: milestone_review-2 -->
- [ ] A milestone that changes a public contract MUST list the owning `docs/spec/*.md` path in its `#### Related Files`; one that changes none lists no spec path. <!-- policy_ac: spec_impact-0 -->
<!-- policy_ac:end -->

## Final Gate

<!-- final_gate:start -->
- [ ] Touched C++/CUDA files (`*.h`/`*.hpp`/`*.cuh`/`*.cu`/`*.cpp`/`*.cc`) MUST be formatted by the pre-commit `clang-format` hook (or an equivalent `clang-format --dry-run -Werror` check). <!-- policy_final: clang_format-0 -->
- [ ] Spec section MUST NOT enumerate test names; the pre-commit `spec-rules-lint` and `english-only` hooks already reject forbidden section headers, plan / milestone / task / PR / commit references, agent names, and non-English text. <!-- policy_final: spec_discipline-0 -->
<!-- final_gate:end -->
