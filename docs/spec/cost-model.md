# TileFoundry Spec - Cost Model

The standalone `tilefoundry-costmodel` package predicts the end-to-end latency
of one CTA on one NVIDIA B200 SM. Its input is a typed tile-operation program,
its measured timing library is independent of compiler IR, and its optimizer
schedules implementation phases on explicit hardware resources.

The core data path is:

```text
TileProgram -> build(CostModelRequest) -> SearchProblem -> solve() -> CostModelResult
```

`build` owns typed-program validation, implementation selection, phase lowering,
profile lookup or explicitly enabled JIT measurement, and time quantization.
`solve` owns only deterministic optimization over the resulting numeric problem.

## 1. Boundary And Invariants

- One request models one CTA resident on one B200 SM.
- Multi-CTA occupancy, SM waves, whole-device scheduling, kernel launch overlap,
  inter-kernel scheduling, energy, and power are outside this package contract.
- The package MUST NOT import TileFoundry HIR, TIR, lowering, or code generation.
- The selected result is a cost-model decision. It MUST NOT be applied to
  executable code by this package.
- The supported logical workload records are GEMM, GQA decode,
  FlashAttention, and MLP. A caller MAY construct their typed programs directly
  or use the built-in workload frontends.
- Search is finite. Concrete one-CTA program variants, implementation IDs, warp
  configurations, pipeline depths, and layout variants MUST be explicit before
  `solve` starts.
- Missing timing data MUST either fail `build` or be measured under an explicit
  JIT policy. `solve` MUST NOT query a database, compile CUDA, run CUDA, invoke a
  callback, or silently substitute an estimate.
- Temporal resources and static capacities are distinct contracts. A temporal
  interval consumes slots over time; a static demand consumes CTA capacity for
  the selected configuration.
- Every scheduled phase MUST retain its source operation ID, selected
  implementation ID, benchmark component ID, bound warp IDs, resource demands,
  and measurement provenance.
- Integer picoseconds are the public timing unit. CP-SAT uses integer ticks and
  every conversion to ticks MUST round upward.

The public schema constants are:

```python
COST_MODEL_API_VERSION: tuple[int, int] = (2, 0)
HARDWARE_SCHEMA_VERSION: int = 1
PLAN_SCHEMA_VERSION: int = 2
PROFILE_SCHEMA_VERSION: int = 1
PROGRAM_SCHEMA_VERSION: int = 2
REQUEST_SCHEMA_VERSION: int = 2
SEARCH_PROBLEM_SCHEMA_VERSION: int = 2
RESULT_SCHEMA_VERSION: int = 2
```

- constraints:
  - A parser MUST reject an unknown schema version and every unknown field.
  - A breaking field, unit, semantic, or ownership change MUST increment the
    affected schema or API version.
  - The legacy `(0, 2)` surface MAY remain importable only from
    `tilefoundry_costmodel.legacy`; the package root MUST expose the contract in
    this document.

The package exception hierarchy is:

```python
class CostModelError(Exception):
    """Base cost-model failure."""


class InvalidRequestError(CostModelError, ValueError):
    """Report malformed caller input."""


class HardwareSpecError(CostModelError, ValueError):
    """Report an invalid hardware document."""


class WorkloadError(CostModelError, ValueError):
    """Report an invalid logical workload or typed program."""


class UnsupportedError(CostModelError):
    """Report a valid request outside installed capability."""


class ProfileError(CostModelError):
    """Base timing-profile failure."""


class MissingProfileError(ProfileError):
    """Report every required profile key absent from a snapshot.

    Attributes:
        key_ids: attribute; Missing canonical profile-key IDs.
    """

    key_ids: tuple[ProfileKeyId, ...]


class ProfileConflictError(ProfileError):
    """Report conflicting immutable profile data."""


class ProfileStoreError(ProfileError):
    """Report profile-store corruption or schema failure."""


class ProfileRunError(ProfileError):
    """Report CUDA compilation, execution, or validation failure."""


class SearchProblemError(CostModelError, ValueError):
    """Report an invalid solver-ready problem."""


class SolverError(CostModelError):
    """Report an internal solver failure."""
```

- constraints:
  - Constructors and JSON parsers MUST raise request, hardware, or workload
    errors for malformed caller data.
  - Profile-store corruption and search-problem invariant failures MUST remain
    exceptions; they MUST NOT be converted into valid performance outcomes.

## 2. Common Values

Stable identifiers and tensor descriptors are defined in
`tilefoundry_costmodel.model`:

```python
ConfigurationId = NewType("ConfigurationId", str)
ProgramId = NewType("ProgramId", str)
OpId = NewType("OpId", str)
ValueId = NewType("ValueId", str)
LoopId = NewType("LoopId", str)
PhaseId = NewType("PhaseId", str)
BufferId = NewType("BufferId", str)
ResourceId = NewType("ResourceId", str)
MeasurementId = NewType("MeasurementId", str)
ProfileKeyId = NewType("ProfileKeyId", str)


class DType(str, Enum):
    """Name a calibrated tensor element type."""

    BF16 = "bf16"
    FP16 = "fp16"
    FP32 = "fp32"
    FP8_E4M3 = "fp8_e4m3"
    FP8_E5M2 = "fp8_e5m2"


class TensorLayout(str, Enum):
    """Name a logical dense matrix layout."""

    ROW_MAJOR = "row_major"
    COLUMN_MAJOR = "column_major"


class AxisExtent:
    """Describe one positive named axis.

    Attributes:
        name: attribute; Non-empty axis name.
        extent: attribute; Positive element count.
    """

    name: str
    extent: int


class NamedShape:
    """Describe a tensor shape with unique axis names.

    Attributes:
        axes: attribute; Ordered named axes.
    """

    axes: tuple[AxisExtent, ...]


class TensorDescriptor:
    """Describe one concrete tensor view.

    Attributes:
        shape: attribute; Concrete named shape.
        dtype: attribute; Element type.
        layout: attribute; Logical dense layout.
        strides_elements: attribute; Explicit element strides, or None.
    """

    shape: NamedShape
    dtype: DType
    layout: TensorLayout
    strides_elements: tuple[int, ...] | None = None
```

- constraints:
  - Every identifier MUST be a non-empty ASCII string.
  - Axis names MUST be non-empty and unique in one shape; extents MUST be
    positive.
  - Explicit strides MUST have the shape rank and contain only non-negative
    integers.
  - Identifiers are canonicalized by their owning model, never by CP-SAT.

## 3. Logical Workloads

Logical workload records state the mathematical problem and remain in requests
and results. They do not generate phases and CP-SAT does not branch on them.

```python
class WorkloadKind(str, Enum):
    """Name one supported logical workload."""

    GEMM = "gemm"
    GQA_DECODE = "gqa_decode"
    FLASH_ATTENTION = "flash_attention"
    MLP = "mlp"


class EpilogueKind(str, Enum):
    """Name a GEMM epilogue."""

    NONE = "none"
    BIAS = "bias"
    RELU = "relu"
    GELU = "gelu"
    SILU = "silu"


class ActivationKind(str, Enum):
    """Name an MLP activation."""

    RELU = "relu"
    GELU = "gelu"
    SILU = "silu"
    SWIGLU = "swiglu"


class GemmSpec:
    """Describe one logical GEMM.

    Attributes:
        kind: attribute; Must be `GEMM`.
        m: attribute; Logical row extent.
        n: attribute; Logical column extent.
        k: attribute; Logical reduction extent.
        dtype_a: attribute; Left operand type.
        dtype_b: attribute; Right operand type.
        dtype_accumulator: attribute; Accumulator type.
        dtype_output: attribute; Output type.
        layout_a: attribute; Left operand layout.
        layout_b: attribute; Right operand layout.
        epilogue: attribute; Requested epilogue.
    """

    kind: WorkloadKind
    m: int
    n: int
    k: int
    dtype_a: DType
    dtype_b: DType
    dtype_accumulator: DType
    dtype_output: DType
    layout_a: TensorLayout
    layout_b: TensorLayout
    epilogue: EpilogueKind = EpilogueKind.NONE


class GqaDecodeSpec:
    """Describe one logical GQA decode operation.

    Attributes:
        kind: attribute; Must be `GQA_DECODE`.
        batch_size: attribute; Batch extent.
        query_heads: attribute; Query-head count.
        kv_heads: attribute; Key/value-head count.
        head_dim: attribute; Per-head reduction extent.
        context_len: attribute; Key/value sequence extent.
        dtype_query: attribute; Query element type.
        dtype_kv: attribute; Key/value element type.
        dtype_accumulator: attribute; Accumulator type.
        dtype_output: attribute; Output element type.
    """

    kind: WorkloadKind
    batch_size: int
    query_heads: int
    kv_heads: int
    head_dim: int
    context_len: int
    dtype_query: DType
    dtype_kv: DType
    dtype_accumulator: DType
    dtype_output: DType


class FlashAttentionSpec:
    """Describe one logical FlashAttention operation.

    Attributes:
        kind: attribute; Must be `FLASH_ATTENTION`.
        batch_size: attribute; Batch extent.
        heads: attribute; Attention-head count.
        query_len: attribute; Query sequence extent.
        key_len: attribute; Key/value sequence extent.
        head_dim: attribute; Per-head reduction extent.
        dtype_query: attribute; Query element type.
        dtype_kv: attribute; Key/value element type.
        dtype_accumulator: attribute; Accumulator type.
        dtype_output: attribute; Output element type.
        causal: attribute; Whether causal masking applies.
    """

    kind: WorkloadKind
    batch_size: int
    heads: int
    query_len: int
    key_len: int
    head_dim: int
    dtype_query: DType
    dtype_kv: DType
    dtype_accumulator: DType
    dtype_output: DType
    causal: bool


class MlpSpec:
    """Describe one logical two-layer MLP.

    Attributes:
        kind: attribute; Must be `MLP`.
        rows: attribute; Input row count.
        input_dim: attribute; Input feature extent.
        hidden_dim: attribute; Hidden feature extent.
        output_dim: attribute; Output feature extent.
        dtype_input: attribute; Input element type.
        dtype_weight: attribute; Weight element type.
        dtype_accumulator: attribute; Accumulator type.
        dtype_output: attribute; Output element type.
        activation: attribute; Intermediate activation.
    """

    kind: WorkloadKind
    rows: int
    input_dim: int
    hidden_dim: int
    output_dim: int
    dtype_input: DType
    dtype_weight: DType
    dtype_accumulator: DType
    dtype_output: DType
    activation: ActivationKind


WorkloadSpec = GemmSpec | GqaDecodeSpec | FlashAttentionSpec | MlpSpec
```

- constraints:
  - Every concrete dimension MUST be positive.
  - Every `kind` field MUST match its concrete record.
  - `GqaDecodeSpec.query_heads` MUST be divisible by `kv_heads`.
  - A workload record describes logical equivalence; executable phase structure
    comes only from its typed program variants and selected implementations.

## 4. Typed Tile Program

`tilefoundry_costmodel.program` is the compiler-independent input IR.

```python
class MemorySpace(str, Enum):
    """Name a B200-visible value storage space."""

    GLOBAL = "global"
    SHARED = "shared"
    TENSOR = "tensor"
    REGISTER = "register"


class TileOpKind(str, Enum):
    """Discriminate one typed tile operation."""

    COPY = "copy"
    GEMM = "gemm"
    REDUCE = "reduce"
    ELEMENTWISE = "elementwise"


class ReductionKind(str, Enum):
    """Name a reduction function."""

    SUM = "sum"
    MAX = "max"


class ElementwiseKind(str, Enum):
    """Name a calibrated elementwise function."""

    ADD = "add"
    MULTIPLY = "multiply"
    DIVIDE = "divide"
    EXP = "exp"
    RELU = "relu"
    GELU = "gelu"
    SILU = "silu"
    CAUSAL_MASK = "causal_mask"


class DependencyRelationKind(str, Enum):
    """Discriminate one supported operation-instance relation."""

    ALIGNED = "aligned"
    ENDPOINT = "endpoint"


class InstanceEndpoint(str, Enum):
    """Select a first or last instance in an operation domain."""

    FIRST = "first"
    LAST = "last"


class AlignedRelation:
    """Pair corresponding instances with one non-negative loop distance.

    Attributes:
        kind: attribute; Must be `ALIGNED`.
        iteration_distance: attribute; Non-negative destination offset.
    """

    kind: DependencyRelationKind
    iteration_distance: int


class EndpointRelation:
    """Pair exactly one source endpoint with one destination endpoint.

    This is a single edge. It does not assert that other source instances have
    completed and it is not a whole-region barrier.

    Attributes:
        kind: attribute; Must be `ENDPOINT`.
        src_endpoint: attribute; Selected source endpoint.
        dst_endpoint: attribute; Selected destination endpoint.
    """

    kind: DependencyRelationKind
    src_endpoint: InstanceEndpoint
    dst_endpoint: InstanceEndpoint


DependencyRelation = AlignedRelation | EndpointRelation


class LoopBarrier:
    """Order every instance of one loop region before another region.

    A loop barrier is a control constraint, not a value dependency and does not
    consume a B200 mbarrier resource. It is explicit because operation order or
    a schedule-tree sequence MUST NOT silently introduce a semantic barrier.

    Attributes:
        barrier_id: attribute; Program-local stable barrier ID.
        src_loop_id: attribute; Source loop whose instances must all complete.
        dst_loop_id: attribute; Destination loop that cannot start before them.
    """

    barrier_id: str
    src_loop_id: LoopId
    dst_loop_id: LoopId


class TileValueType:
    """Describe a typed value in one memory space.

    Attributes:
        tensor: attribute; Concrete tensor descriptor.
        memory_space: attribute; Storage space.
    """

    tensor: TensorDescriptor
    memory_space: MemorySpace


class TileValue:
    """Name one program value.

    Attributes:
        value_id: attribute; Program-local stable ID.
        value_type: attribute; Tensor and memory-space type.
    """

    value_id: ValueId
    value_type: TileValueType


class TileLoop:
    """Describe one repeated pipeline region.

    Attributes:
        loop_id: attribute; Program-local stable ID.
        iterations: attribute; Positive logical iteration count.
    """

    loop_id: LoopId
    iterations: int


class OpIterationDomain:
    """Place an operation over one loop or one-time region.

    Attributes:
        loop_id: attribute; Referenced loop, or None for one-time work.
        first_iteration: attribute; First covered iteration.
        iteration_count: attribute; Positive covered instance count.
    """

    loop_id: LoopId | None
    first_iteration: int
    iteration_count: int


class CopyOp:
    """Copy one concrete tile between values.

    Attributes:
        kind: attribute; Must be `COPY`.
        op_id: attribute; Program-local stable ID.
        source: input; Source value ID.
        destination: input; Destination value ID.
        domain: attribute; Operation iteration domain.
    """

    kind: TileOpKind
    op_id: OpId
    source: ValueId
    destination: ValueId
    domain: OpIterationDomain


class GemmOp:
    """Accumulate one concrete GEMM tile.

    Attributes:
        kind: attribute; Must be `GEMM`.
        op_id: attribute; Program-local stable ID.
        lhs: input; Left value ID.
        rhs: input; Right value ID.
        accumulator: input; Accumulator value ID.
        result: attribute; Result value ID.
        m_axis: attribute; Logical row axis.
        n_axis: attribute; Logical column axis.
        k_axis: attribute; Logical reduction axis.
        domain: attribute; Operation iteration domain.
    """

    kind: TileOpKind
    op_id: OpId
    lhs: ValueId
    rhs: ValueId
    accumulator: ValueId
    result: ValueId
    m_axis: str
    n_axis: str
    k_axis: str
    domain: OpIterationDomain


class ReduceOp:
    """Reduce one concrete tile.

    Attributes:
        kind: attribute; Must be `REDUCE`.
        op_id: attribute; Program-local stable ID.
        source: input; Source value ID.
        result: attribute; Result value ID.
        axes: attribute; Reduced named axes.
        reduction: attribute; Reduction function.
        domain: attribute; Operation iteration domain.
    """

    kind: TileOpKind
    op_id: OpId
    source: ValueId
    result: ValueId
    axes: tuple[str, ...]
    reduction: ReductionKind
    domain: OpIterationDomain


class ElementwiseOp:
    """Apply one concrete elementwise function.

    Attributes:
        kind: attribute; Must be `ELEMENTWISE`.
        op_id: attribute; Program-local stable ID.
        inputs: input; Input value IDs.
        result: attribute; Result value ID.
        function: attribute; Elementwise function.
        domain: attribute; Operation iteration domain.
    """

    kind: TileOpKind
    op_id: OpId
    inputs: tuple[ValueId, ...]
    result: ValueId
    function: ElementwiseKind
    domain: OpIterationDomain


TileOp = CopyOp | GemmOp | ReduceOp | ElementwiseOp


class TileDependency:
    """Connect two operations through one value.

    Attributes:
        value_id: attribute; Value causing the dependency.
        src_op_id: attribute; Producer operation ID.
        dst_op_id: attribute; Consumer operation ID.
        relation: attribute; Typed instance relation.
    """

    value_id: ValueId
    src_op_id: OpId
    dst_op_id: OpId
    relation: DependencyRelation


class TileCandidate:
    """Name one concrete CTA tile.

    Attributes:
        tile_id: attribute; Stable tile ID.
        shape: attribute; Workload-specific named tile shape.
    """

    tile_id: str
    shape: NamedShape


class TileProgram:
    """Describe one concrete one-CTA tile variant.

    Attributes:
        schema_version: attribute; Program schema version.
        program_id: attribute; Stable program ID.
        workload_kind: attribute; Matching logical workload kind.
        tile: attribute; Concrete CTA tile.
        values: attribute; Program-local values.
        loops: attribute; Repeated pipeline regions.
        operations: attribute; Typed operation graph.
        dependencies: attribute; Explicit value dependencies.
        loop_barriers: attribute; Explicit inter-loop control barriers.
        inputs: input; External value IDs.
        outputs: attribute; Logical output value IDs.
    """

    schema_version: int
    program_id: ProgramId
    workload_kind: WorkloadKind
    tile: TileCandidate
    values: tuple[TileValue, ...]
    loops: tuple[TileLoop, ...]
    operations: tuple[TileOp, ...]
    dependencies: tuple[TileDependency, ...]
    loop_barriers: tuple[LoopBarrier, ...]
    inputs: tuple[ValueId, ...]
    outputs: tuple[ValueId, ...]
```

- constraints:
  - `TileProgram.schema_version` MUST equal `PROGRAM_SCHEMA_VERSION`.
  - Program-local value, loop, and operation IDs MUST be unique.
  - Each operation `kind` MUST match its concrete class and acts as the JSON
    union discriminator.
  - Each dependency `relation.kind` MUST match its concrete relation class and
    acts as the JSON union discriminator.
  - A one-time domain MUST equal `(None, 0, 1)`. A loop-bound domain MUST name a
    declared loop and lie within its positive iteration range.
  - An `AlignedRelation` connects `src[i]` to `dst[i +
    iteration_distance]` for every valid pair in the same loop. An
    `EndpointRelation` connects exactly the selected source endpoint to the
    selected destination endpoint, even when the operations use different loops
    or one-time domains.
  - An endpoint relation MUST NOT be interpreted as a barrier. A whole-region
    ordering MUST use an explicit `LoopBarrier`.
  - `AlignedRelation.iteration_distance` MUST be non-negative. A self
    dependency MUST use a positive aligned distance; endpoint self-dependencies
    are invalid.
  - Loop barriers MUST reference distinct declared loops, have unique IDs, and
    form an acyclic loop-order graph. A barrier constrains
    `max(end(src_loop[*])) <= min(start(dst_loop[*]))`.
  - The operation relation graph after expanding finite domains MUST be
    acyclic for zero-distance aligned and endpoint edges.
  - Every non-external consumed value MUST have an explicit producer
    dependency. Input, output, operation, and dependency value IDs MUST exist.
  - Copy source and destination values MUST have compatible element counts and
    dtypes.
  - GEMM axis names MUST resolve unambiguously in referenced value shapes.
  - Reduction axes MUST be non-empty, unique, and present in the source shape.
  - Elementwise inputs use named-axis broadcasting and MUST exactly produce the
    declared result shape and dtype.
  - A concrete program is the source of operation semantics. Phase names,
    implementation names, and profile providers MUST NOT reinterpret it.

### 4.1 Typed Construction Surface

`tilefoundry_costmodel.language` is imported as `T`. It constructs the same
immutable records that strict JSON deserialization constructs.

```python
def value(*, value_id: ValueId, value_type: TileValueType) -> TileValue:
    """Construct a typed tile value.

    Args:
        value_id: Program-local value ID.
        value_type: Tensor and memory-space type.

    Returns:
        The validated value record.
    """
    ...


def pipeline(*, loop_id: LoopId, iterations: int) -> TileLoop:
    """Construct a repeated pipeline region.

    Args:
        loop_id: Program-local loop ID.
        iterations: Positive logical iteration count.

    Returns:
        The validated loop record.
    """
    ...


def copy(
    *,
    op_id: OpId,
    source: ValueId,
    destination: ValueId,
    domain: OpIterationDomain,
) -> CopyOp:
    """Construct a copy operation.

    Args:
        op_id: Program-local operation ID.
        source: Source value ID.
        destination: Destination value ID.
        domain: Operation iteration domain.

    Returns:
        The validated copy operation.
    """
    ...


def gemm(
    *,
    op_id: OpId,
    lhs: ValueId,
    rhs: ValueId,
    accumulator: ValueId,
    result: ValueId,
    m_axis: str,
    n_axis: str,
    k_axis: str,
    domain: OpIterationDomain,
) -> GemmOp:
    """Construct a GEMM operation.

    Args:
        op_id: Program-local operation ID.
        lhs: Left value ID.
        rhs: Right value ID.
        accumulator: Accumulator value ID.
        result: Result value ID.
        m_axis: Logical row axis.
        n_axis: Logical column axis.
        k_axis: Logical reduction axis.
        domain: Operation iteration domain.

    Returns:
        The validated GEMM operation.
    """
    ...


def reduce(
    *,
    op_id: OpId,
    source: ValueId,
    result: ValueId,
    axes: tuple[str, ...],
    reduction: ReductionKind,
    domain: OpIterationDomain,
) -> ReduceOp:
    """Construct a reduction operation.

    Args:
        op_id: Program-local operation ID.
        source: Source value ID.
        result: Result value ID.
        axes: Reduced axis names.
        reduction: Reduction function.
        domain: Operation iteration domain.

    Returns:
        The validated reduction operation.
    """
    ...


def elementwise(
    *,
    op_id: OpId,
    inputs: tuple[ValueId, ...],
    result: ValueId,
    function: ElementwiseKind,
    domain: OpIterationDomain,
) -> ElementwiseOp:
    """Construct an elementwise operation.

    Args:
        op_id: Program-local operation ID.
        inputs: Input value IDs.
        result: Result value ID.
        function: Elementwise function.
        domain: Operation iteration domain.

    Returns:
        The validated elementwise operation.
    """
    ...


def aligned(*, iteration_distance: int = 0) -> AlignedRelation:
    """Construct one corresponding-instance relation.

    Args:
        iteration_distance: Non-negative destination offset.

    Returns:
        The validated aligned relation.
    """
    ...


def endpoint(
    *,
    src_endpoint: InstanceEndpoint,
    dst_endpoint: InstanceEndpoint,
) -> EndpointRelation:
    """Construct one explicit endpoint relation.

    Args:
        src_endpoint: Source operation endpoint.
        dst_endpoint: Destination operation endpoint.

    Returns:
        The validated endpoint relation.
    """
    ...


def depends(
    *,
    value_id: ValueId,
    src_op_id: OpId,
    dst_op_id: OpId,
    relation: DependencyRelation,
) -> TileDependency:
    """Construct one explicit value dependency.

    Args:
        value_id: Value causing the dependency.
        src_op_id: Producer operation ID.
        dst_op_id: Consumer operation ID.
        relation: Explicit aligned or endpoint relation.

    Returns:
        The validated dependency record.
    """
    ...


def program(
    *,
    schema_version: int,
    program_id: ProgramId,
    workload_kind: WorkloadKind,
    tile: TileCandidate,
    values: tuple[TileValue, ...],
    loops: tuple[TileLoop, ...],
    operations: tuple[TileOp, ...],
    dependencies: tuple[TileDependency, ...],
    loop_barriers: tuple[LoopBarrier, ...],
    inputs: tuple[ValueId, ...],
    outputs: tuple[ValueId, ...],
) -> TileProgram:
    """Construct one concrete tile program.

    Args:
        schema_version: Program schema version.
        program_id: Stable program ID.
        workload_kind: Matching logical workload kind.
        tile: Concrete CTA tile.
        values: Program-local values.
        loops: Repeated regions.
        operations: Typed operation graph.
        dependencies: Explicit value dependencies.
        loop_barriers: Explicit inter-loop control barriers.
        inputs: External value IDs.
        outputs: Logical output value IDs.

    Returns:
        The validated tile program.
    """
    ...
```

- constraints:
  - `T.*` MUST set fixed operation discriminators itself.
  - The construction surface MUST NOT accept source text, a callable, an opaque
    payload, or a compiler IR node.
  - `T.pipeline` declares a repeated region and its iteration count. Candidate
    depths remain in `SearchSpace.pipeline_depths`.

Built-in workload frontends are optional program constructors:

```python
class WorkloadFrontend(Protocol):
    """Construct typed programs for one logical workload kind.

    Attributes:
        workload_kind: attribute; Exact supported workload kind.
    """

    workload_kind: WorkloadKind

    def build_programs(
        self,
        workload: WorkloadSpec,
        *,
        tiles: tuple[TileCandidate, ...],
    ) -> tuple[TileProgram, ...]:
        """Construct concrete one-CTA program variants.

        Args:
            workload: Logical workload record.
            tiles: Explicit concrete CTA tiles.

        Returns:
            Canonically ordered typed programs.
        """
        ...


class WorkloadFrontendCatalog:
    """Index one frontend per logical workload kind.

    Attributes:
        frontends: attribute; Unique workload frontends.
    """

    frontends: tuple[WorkloadFrontend, ...]

    def frontend_for(self, kind: WorkloadKind) -> WorkloadFrontend:
        """Resolve one exact workload frontend.

        Args:
            kind: Exact workload kind.

        Returns:
            The unique matching frontend.
        """
        ...


def builtin_workload_frontends() -> WorkloadFrontendCatalog:
    """Construct the built-in workload frontend catalog.

    Returns:
        Frontends for every supported workload kind.
    """
    ...
```

- constraints:
  - A frontend MUST use the typed construction surface.
  - A frontend MUST NOT assign hardware resources, phases, timing keys,
    durations, or CUDA source.
  - Program IDs and output ordering MUST be deterministic and independent of
    caller enumeration order.
  - Direct `TileProgram` construction bypasses only frontend convenience; it
    MUST NOT bypass validation or implementation lowering.

### 4.2 TileFoundry Adapter Boundary

The standalone package does not parse or import TileFoundry HIR, TIR, ISL, or
schedule trees. An optional adapter lives on the TileFoundry side and projects
only a proved one-dimensional subset of an existing `TileGraph` dependency
relation into the typed program. The direction is strictly:

```text
TileFoundry HIR -> TileGraph/ISL -> adapter -> TileProgram dependencies
```

The adapter does not lower HIR operations into `T.copy`/`T.gemm`, choose a tile,
apply a selected result, or infer an order from `build_schedule_tree`. A
workload frontend remains responsible for constructing the complete typed
operation graph and for declaring any internal lowering edges.

The adapter-side records and entry point are:

```python
class CostModelAdapterError(ValueError):
    """Report a malformed or unadaptable TileFoundry projection."""


class UnsupportedDependencyRelationError(CostModelAdapterError):
    """Report an ISL relation outside the typed adapter subset.

    Attributes:
        src_statement_id: attribute; Source TileFoundry statement ID.
        dst_statement_id: attribute; Destination TileFoundry statement ID.
        relation_text: attribute; Canonical diagnostic relation text.
    """

    src_statement_id: str
    dst_statement_id: str
    relation_text: str


class PipelineAxisBinding:
    """Bind one TileFoundry statement dimension to a cost-model iteration.

    Attributes:
        statement_id: attribute; TileFoundry statement tuple name.
        op_id: attribute; Existing typed cost-model operation ID.
        domain_axis: attribute; Zero-based ISL statement dimension, or None for
            a one-time cost-model operation.
    """

    statement_id: str
    op_id: OpId
    domain_axis: int | None


class TileGraphDependencyQuery:
    """Describe one semantic value edge to project from an ISL graph.

    Attributes:
        value_id: attribute; Cost-model value carried by the edge.
        src: attribute; Source statement/operation binding.
        dst: attribute; Destination statement/operation binding.
    """

    value_id: ValueId
    src: PipelineAxisBinding
    dst: PipelineAxisBinding


def project_tile_graph_dependencies(
    graph: TileGraph,
    program: TileProgram,
    *,
    queries: tuple[TileGraphDependencyQuery, ...],
) -> tuple[TileDependency, ...]:
    """Project exactly representable TileGraph edges into typed dependencies.

    Args:
        graph: Existing TileFoundry polyhedral graph.
        program: Typed program whose operation domains are being bound.
        queries: Explicit source/destination/value bindings.

    Returns:
        Canonically ordered typed dependencies.

    Raises:
        CostModelAdapterError: A binding or program ID is malformed.
        UnsupportedDependencyRelationError: A projected relation is not a
            uniform aligned or exact endpoint relation.
    """
    ...
```

- adapter constraints:
  - `statement_id` values MUST match ISL tuple names exactly and `op_id` values
    MUST resolve to operations in `program`.
  - `domain_axis` MUST be explicit for a loop-bound operation. `None` is legal
    only for a one-time operation and represents its sole cost-model instance.
  - The adapter MUST project the requested dimensions with structured ISL APIs
    and compare the resulting finite relation with the exact canonical relation
    before constructing a dependency.
  - A projected relation is supported only when it is one uniform
    `AlignedRelation` or exactly one `EndpointRelation`. Multi-dimensional,
    skewed, non-uniform, many-to-many, empty, or ambiguous relations MUST raise
    `UnsupportedDependencyRelationError`.
  - The adapter MUST NOT drop a relation, manufacture a `LoopBarrier`, or use
    schedule-tree order as a fallback. A caller needing a whole-region order
    must declare a `LoopBarrier` in the typed program.
  - Queries MUST cover each semantic edge they claim to export, have no
    duplicate `(value_id, src_op_id, dst_op_id, relation)` result, and produce
    deterministic output independent of query order.
  - This optional adapter is the only allowed TileFoundry dependency boundary;
    importing `tilefoundry_costmodel` MUST remain optional for the main package,
    and importing the standalone package MUST remain independent of TileFoundry.

## 5. Search Request

```python
class WarpRole(str, Enum):
    """Name one initial B200 warp role."""

    TMA_PRODUCER = "tma_producer"
    TENSOR_CONSUMER = "tensor_consumer"
    CUDA_EPILOGUE = "cuda_epilogue"


class WarpRoleAssignment:
    """Assign warp IDs to one role.

    Attributes:
        role: attribute; Assigned role.
        warp_ids: attribute; Unique zero-based warp IDs.
    """

    role: WarpRole
    warp_ids: tuple[int, ...]


class WarpConfig:
    """Describe one finite CTA warp configuration.

    Attributes:
        config_id: attribute; Stable configuration ID.
        total_warps: attribute; Positive CTA warp count.
        roles: attribute; Role assignments.
    """

    config_id: str
    total_warps: int
    roles: tuple[WarpRoleAssignment, ...]


class SearchSpace:
    """Bound every outer candidate dimension.

    Attributes:
        implementation_ids: attribute; Allowed operation implementation IDs.
        warp_configs: attribute; Allowed CTA warp configurations.
        pipeline_depths: attribute; Allowed ring depths.
        layout_variant_ids: attribute; Allowed implementation layout variants.
        max_candidates: attribute; Maximum canonical configurations.
    """

    implementation_ids: tuple[str, ...]
    warp_configs: tuple[WarpConfig, ...]
    pipeline_depths: tuple[int, ...]
    layout_variant_ids: tuple[str, ...] = ()
    max_candidates: int = 10_000


class ProfileMode(str, Enum):
    """Control behavior on an exact profile miss."""

    REQUIRE = "require"
    JIT_ON_MISS = "jit_on_miss"


class TimingStatistic(str, Enum):
    """Select a measured timing aggregate."""

    P50 = "p50"
    P90 = "p90"


class ProfileSnapshotRef:
    """Identify one immutable or draft profile snapshot revision.

    Attributes:
        snapshot_id: attribute; Stable snapshot family ID.
        revision: attribute; Positive revision number.
    """

    snapshot_id: str
    revision: int


class ProfileSelection:
    """Select timing data for one request.

    Attributes:
        snapshot: attribute; Exact snapshot revision.
        mode: attribute; Miss behavior.
        timing_statistic: attribute; Primary timing statistic.
    """

    snapshot: ProfileSnapshotRef
    mode: ProfileMode = ProfileMode.REQUIRE
    timing_statistic: TimingStatistic = TimingStatistic.P50


class SolverOptions:
    """Configure candidate search and CP-SAT.

    Attributes:
        candidate_timeout_s: attribute; Per-candidate wall-clock limit.
        search_timeout_s: attribute; Optional outer wall-clock limit.
        time_resolution_ps: attribute; Positive tick size in picoseconds.
        ortools_workers: attribute; CP-SAT worker count.
        candidate_workers: attribute; Outer candidate worker count.
        random_seed: attribute; CP-SAT random seed.
        finite_unroll_limit: attribute; Largest exactly expanded loop.
        stop_after_first_solution: attribute; Stop after one feasible result.
        deterministic: attribute; Require deterministic search settings.
    """

    candidate_timeout_s: float | None = 30.0
    search_timeout_s: float | None = None
    time_resolution_ps: int = 10
    ortools_workers: int = 1
    candidate_workers: int = 1
    random_seed: int = 0
    finite_unroll_limit: int = 64
    stop_after_first_solution: bool = False
    deterministic: bool = True


class CostModelRequest:
    """Describe one finite cost-model search.

    Attributes:
        schema_version: attribute; Request schema version.
        request_id: attribute; Stable caller request ID.
        workload: attribute; Logical workload.
        programs: attribute; Concrete equivalent tile-program variants.
        hardware: attribute; Exact calibrated hardware reference.
        search_space: attribute; Finite outer candidate dimensions.
        profiles: attribute; Profile snapshot and miss behavior.
        solver: attribute; Numeric solve controls.
    """

    schema_version: int
    request_id: str
    workload: WorkloadSpec
    programs: tuple[TileProgram, ...]
    hardware: HardwareSpecRef
    search_space: SearchSpace
    profiles: ProfileSelection
    solver: SolverOptions = SolverOptions()
```

- constraints:
  - `schema_version` MUST equal `REQUEST_SCHEMA_VERSION`.
  - Programs MUST be non-empty, have unique IDs, contain no duplicate canonical
    program, and match the logical workload kind.
  - Built-in frontends MUST guarantee logical equivalence of their program
    variants. A direct caller explicitly asserts that equivalence.
  - Warp IDs MUST be unique within a role and less than `total_warps`. Shared
    role IDs are legal only when every selected implementation supports them.
  - Pipeline depths MUST be unique integers in `1..8`. One selected depth
    applies to every pipelined loop in the initial schema.
  - Empty implementation, warp, or depth choices are invalid. An empty layout
    tuple means the single stable layout ID `default`.
  - Deterministic mode requires one OR-Tools worker, one candidate worker, and
    `stop_after_first_solution` false.
  - A JIT request MAY write only to a draft snapshot. A require-only request MAY
    read a draft or frozen snapshot.

The built-in frontend tile-axis contracts are:

```text
GEMM:           m, n, k
GQA decode:     q_heads, kv_tokens, head_dim
FlashAttention: q_tokens, kv_tokens, head_dim
MLP:            up_m, up_n, up_k, down_m, down_n, down_k
```

- constraints:
  - Every required axis MUST occur exactly once and extra axes are invalid.
  - Concrete programs, rather than `SearchSpace`, carry tile choices.

## 6. B200 Hardware

```python
class FactOrigin(str, Enum):
    """Name the provenance class of one hardware fact."""

    VENDOR = "vendor"
    MEASURED = "measured"
    DERIVED = "derived"
    CONSERVATIVE = "conservative"
    UNAVAILABLE = "unavailable"


class StaticUnit(str, Enum):
    """Name a static capacity unit."""

    BYTES = "bytes"
    REGISTERS_32BIT = "registers_32bit"
    WARPS = "warps"
    SLOTS = "slots"


class FactProvenance:
    """Explain one hardware fact.

    Attributes:
        origin: attribute; Fact provenance class.
        source: attribute; Human-auditable source.
        conditions: attribute; Conditions under which the fact applies.
    """

    origin: FactOrigin
    source: str
    conditions: str


class TemporalResourceSpec:
    """Describe one schedulable temporal resource.

    Attributes:
        resource_id: attribute; Stable hardware resource ID.
        capacity_slots: attribute; Positive simultaneous slot capacity.
        description: attribute; Resource meaning.
        provenance: attribute; Capacity provenance.
    """

    resource_id: ResourceId
    capacity_slots: int
    description: str
    provenance: FactProvenance


class StaticResourceSpec:
    """Describe one per-CTA static capacity.

    Attributes:
        resource_id: attribute; Stable hardware resource ID.
        capacity_units: attribute; Positive capacity.
        unit: attribute; Capacity unit.
        description: attribute; Resource meaning.
        provenance: attribute; Capacity provenance.
    """

    resource_id: ResourceId
    capacity_units: int
    unit: StaticUnit
    description: str
    provenance: FactProvenance


class HardwareSpecRef:
    """Identify one exact calibrated hardware document.

    Attributes:
        hardware_id: attribute; Stable hardware ID.
        schema_version: attribute; Hardware schema version.
        calibration_id: attribute; Exact calibration identity.
    """

    hardware_id: str
    schema_version: int
    calibration_id: str


class HardwareSpec:
    """Describe the complete schedulable B200 model.

    Attributes:
        ref: attribute; Exact document identity.
        architecture: attribute; Architecture name.
        temporal_resources: attribute; Time-varying capacities.
        static_resources: attribute; Per-CTA capacities.
        supported_dtypes: attribute; Calibrated element types.
        supported_implementation_ids: attribute; Installed implementation IDs.
    """

    ref: HardwareSpecRef
    architecture: str
    temporal_resources: tuple[TemporalResourceSpec, ...]
    static_resources: tuple[StaticResourceSpec, ...]
    supported_dtypes: tuple[DType, ...]
    supported_implementation_ids: tuple[str, ...]

    def temporal_capacity(self, resource_id: ResourceId) -> int:
        """Return one exact temporal capacity.

        Args:
            resource_id: Exact resource ID.

        Returns:
            Positive slot capacity.
        """
        ...

    def static_capacity(self, resource_id: ResourceId) -> int:
        """Return one exact static capacity.

        Args:
            resource_id: Exact resource ID.

        Returns:
            Positive capacity in the resource unit.
        """
        ...


class HardwareCatalog:
    """Resolve exact hardware references.

    Attributes:
        specs: attribute; Unique installed hardware documents.
    """

    specs: tuple[HardwareSpec, ...]

    def resolve(self, ref: HardwareSpecRef) -> HardwareSpec:
        """Resolve one exact hardware document.

        Args:
            ref: Exact hardware reference.

        Returns:
            The matching hardware document.
        """
        ...
```

- constraints:
  - Catalog resolution MUST require an exact `(hardware_id, schema_version,
    calibration_id)` match and MUST NOT select a nearby architecture or newest
    calibration.
  - Resource IDs MUST be unique within temporal and static namespaces.
  - Positive schedulable capacities MUST have vendor, measured, derived, or
    conservative provenance. An unavailable fact MUST NOT become a capacity.
  - Static and temporal lookups MUST reject the wrong resource class.

The B200 resource IDs are stable:

```python
B200_TMA = ResourceId("b200.tma")
B200_TENSOR_CORE = ResourceId("b200.tensor_core")
B200_CUDA_CORE = ResourceId("b200.cuda_core")
B200_WARP_ISSUE = ResourceId("b200.warp_issue")
B200_GMEM_READ = ResourceId("b200.gmem_read")
B200_GMEM_WRITE = ResourceId("b200.gmem_write")
B200_SMEM_READ = ResourceId("b200.smem_read")
B200_SMEM_WRITE = ResourceId("b200.smem_write")
B200_TMEM_READ = ResourceId("b200.tmem_read")
B200_TMEM_WRITE = ResourceId("b200.tmem_write")
B200_RF_READ = ResourceId("b200.rf_read")
B200_RF_WRITE = ResourceId("b200.rf_write")
B200_TMA_INFLIGHT = ResourceId("b200.tma_inflight")
B200_TENSOR_INFLIGHT = ResourceId("b200.tensor_inflight")
B200_SMEM_BYTES = ResourceId("b200.smem_bytes")
B200_TMEM_BYTES = ResourceId("b200.tmem_bytes")
B200_REGISTERS_32BIT = ResourceId("b200.registers_32bit")
B200_WARPS = ResourceId("b200.warps")
B200_MBARRIER_SLOTS = ResourceId("b200.mbarrier_slots")
```

## 7. Operation Lowering And Phases

```python
class TimingMetric(str, Enum):
    """Select one measured timing dimension."""

    LATENCY = "latency"
    INITIATION_INTERVAL = "initiation_interval"


class TemporalDemand:
    """Reserve temporal resource slots.

    Attributes:
        resource_id: attribute; Exact temporal resource ID.
        slots: attribute; Positive slot demand.
    """

    resource_id: ResourceId
    slots: int


class StaticDemand:
    """Consume one static CTA capacity.

    Attributes:
        resource_id: attribute; Exact static resource ID.
        units: attribute; Positive unit demand.
    """

    resource_id: ResourceId
    units: int


class PhaseIterationDomain:
    """Place a lowered phase over one source operation domain.

    Attributes:
        loop_id: attribute; Referenced loop, or None.
        first_iteration: attribute; First covered iteration.
        iteration_count: attribute; Positive instance count.
    """

    loop_id: LoopId | None
    first_iteration: int
    iteration_count: int


class PhaseTemplate:
    """Describe one implementation phase before timing resolution.

    Attributes:
        phase_id: attribute; Configuration-local phase ID.
        source_op_id: attribute; Originating typed operation ID.
        implementation_id: attribute; Selected operation implementation ID.
        phase_name: attribute; Implementation-local phase label.
        component_id: attribute; Matching benchmark component ID.
        domain: attribute; Preserved operation domain.
        profile: attribute; Exact timing requirement.
        warp_ids: attribute; Bound selected warp IDs.
        temporal_demands: attribute; Reserved temporal resources.
    """

    phase_id: PhaseId
    source_op_id: OpId
    implementation_id: str
    phase_name: str
    component_id: str
    domain: PhaseIterationDomain
    profile: ProfileRequirement
    warp_ids: tuple[int, ...]
    temporal_demands: tuple[TemporalDemand, ...]


class PhaseDependency:
    """Constrain completion before another phase starts.

    Attributes:
        src_phase_id: attribute; Source phase ID.
        dst_phase_id: attribute; Destination phase ID.
        relation: attribute; Typed instance relation.
        delay_ps: attribute; Non-negative end-to-start delay.
    """

    src_phase_id: PhaseId
    dst_phase_id: PhaseId
    relation: DependencyRelation
    delay_ps: int = 0


class PhaseStartAlignment:
    """Constrain corresponding phase starts by an exact offset.

    Attributes:
        src_phase_id: attribute; Source phase ID.
        dst_phase_id: attribute; Destination phase ID.
        offset_ps: attribute; Non-negative start offset.
    """

    src_phase_id: PhaseId
    dst_phase_id: PhaseId
    offset_ps: int = 0


class BufferTemplate:
    """Describe one static or ring value allocation.

    Attributes:
        buffer_id: attribute; Configuration-local buffer ID.
        value_id: attribute; Source program value ID.
        storage_resource_id: attribute; Static storage resource ID.
        bytes_per_slot: attribute; Positive bytes in one slot.
        slot_count: attribute; Positive slot count.
        producer_phase_id: attribute; Allocation-start phase.
        release_phase_ids: attribute; Last-use phases.
    """

    buffer_id: BufferId
    value_id: ValueId
    storage_resource_id: ResourceId
    bytes_per_slot: int
    slot_count: int
    producer_phase_id: PhaseId
    release_phase_ids: tuple[PhaseId, ...]


class LoopTemplate:
    """Describe one lowered repeated region.

    Attributes:
        loop_id: attribute; Source program loop ID.
        iterations: attribute; Positive logical iteration count.
    """

    loop_id: LoopId
    iterations: int


class OpImplementationSelection:
    """Record one operation implementation choice.

    Attributes:
        op_id: attribute; Source operation ID.
        implementation_id: attribute; Selected implementation ID.
    """

    op_id: OpId
    implementation_id: str


class ConfigurationTemplate:
    """Describe one complete candidate before timing resolution.

    Attributes:
        configuration_id: attribute; Canonical configuration digest.
        program_id: attribute; Source program ID.
        workload_kind: attribute; Logical workload kind.
        tile: attribute; Concrete CTA tile.
        implementations: attribute; Per-operation implementation choices.
        warps: attribute; CTA warp configuration.
        pipeline_depth: attribute; Selected ring depth.
        layout_variant_id: attribute; Selected layout implementation variant.
        loops: attribute; Lowered loop regions.
        phases: attribute; Untimed implementation phases.
        dependencies: attribute; End-to-start constraints.
        loop_barriers: attribute; Explicit inter-loop control barriers.
        start_alignments: attribute; Exact start-offset constraints.
        buffers: attribute; Value allocations and reuse points.
        static_demands: attribute; Complete static CTA demand.
    """

    configuration_id: ConfigurationId
    program_id: ProgramId
    workload_kind: WorkloadKind
    tile: TileCandidate
    implementations: tuple[OpImplementationSelection, ...]
    warps: WarpConfig
    pipeline_depth: int
    layout_variant_id: str
    loops: tuple[LoopTemplate, ...]
    phases: tuple[PhaseTemplate, ...]
    dependencies: tuple[PhaseDependency, ...]
    loop_barriers: tuple[LoopBarrier, ...]
    start_alignments: tuple[PhaseStartAlignment, ...]
    buffers: tuple[BufferTemplate, ...]
    static_demands: tuple[StaticDemand, ...]
```

- constraints:
  - Phase IDs MUST be unique in one configuration and every referenced phase
    MUST exist.
  - A phase domain MUST preserve its source operation domain.
  - Phase names are implementation-local strings, not a generic operation enum.
  - A phase demanding `b200.warp_issue` MUST bind at least one unique warp ID
    from the selected warp configuration. A pure in-flight phase MAY bind none.
  - A temporal demand reserves its slots for the phase's full interval.
  - An `AlignedRelation` constrains
    `end(src[i]) + delay <= start(dst[i + relation.iteration_distance])` for
    every valid pair.
  - An `EndpointRelation` constrains exactly the selected source endpoint before
    the selected destination endpoint. It does not constrain any other instance.
  - A `LoopBarrier` constrains every phase instance in the source loop to finish
    before any phase instance in the destination loop starts:
    `max(end(src_loop[*])) <= min(start(dst_loop[*]))`.
  - Loop barriers are control constraints and MUST NOT be represented as value
    dependencies or inferred from phase order.
  - A start alignment constrains
    `start(dst[i]) == start(src[i]) + offset` and requires identical domains.
  - Asynchronous issue and latency phases MUST use zero-offset start alignment.
    They MUST NOT be serialized as initiation interval plus latency.
  - A buffer with `d` slots constrains every
    `end(release[i]) <= start(producer[i + d])`.
  - Buffer bytes are charged exactly once as `bytes_per_slot * slot_count`.

Value availability separates semantic dependencies from implementation timing:

```python
class ValueStoragePolicy(str, Enum):
    """Name one value allocation policy."""

    STATIC = "static"
    PIPELINE_RING = "pipeline_ring"


class ProducedValue:
    """Expose one implementation-defined value availability.

    Attributes:
        value_id: attribute; Produced program value ID.
        availability_id: attribute; Stable implementation availability name.
        ready_phase_id: attribute; Phase whose end exposes the availability.
    """

    value_id: ValueId
    availability_id: str
    ready_phase_id: PhaseId


class ConsumedValue:
    """Require one named value availability.

    Attributes:
        value_id: input; Consumed program value ID.
        required_availability_id: attribute; Required availability name.
        consume_phase_id: attribute; Phase that begins consuming the value.
        release_phase_id: attribute; Phase that ends the value use.
    """

    value_id: ValueId
    required_availability_id: str
    consume_phase_id: PhaseId
    release_phase_id: PhaseId


class ValueStorage:
    """Describe where one produced value lifetime begins.

    Attributes:
        value_id: attribute; Produced program value ID.
        allocation_phase_id: attribute; Allocation-start phase.
        storage_policy: attribute; Static or pipeline-ring policy.
    """

    value_id: ValueId
    allocation_phase_id: PhaseId
    storage_policy: ValueStoragePolicy


class LoweringContext:
    """Carry every explicit operation-lowering choice.

    Attributes:
        program: attribute; Source typed program.
        hardware: attribute; Exact B200 hardware document.
        warps: attribute; Selected warp configuration.
        pipeline_depth: attribute; Selected ring depth.
        layout_variant_id: attribute; Selected layout variant.
    """

    program: TileProgram
    hardware: HardwareSpec
    warps: WarpConfig
    pipeline_depth: int
    layout_variant_id: str


class LoweredTileOp:
    """Carry one operation's complete implementation fragment.

    Attributes:
        source_op_id: attribute; Originating typed operation ID.
        implementation_id: attribute; Selected implementation ID.
        phases: attribute; Untimed phases.
        internal_dependencies: attribute; Intra-operation dependencies.
        internal_start_alignments: attribute; Intra-operation start alignments.
        produced_values: attribute; Named value availabilities.
        consumed_values: input; Required value availabilities and releases.
        value_storage: attribute; Produced value allocations.
        static_demands: attribute; Implementation-private static demand.
    """

    source_op_id: OpId
    implementation_id: str
    phases: tuple[PhaseTemplate, ...]
    internal_dependencies: tuple[PhaseDependency, ...]
    internal_start_alignments: tuple[PhaseStartAlignment, ...]
    produced_values: tuple[ProducedValue, ...]
    consumed_values: tuple[ConsumedValue, ...]
    value_storage: tuple[ValueStorage, ...]
    static_demands: tuple[StaticDemand, ...]


class TileOpLowering(Protocol):
    """Lower one typed operation implementation.

    Attributes:
        op_kind: attribute; Exact supported operation kind.
        implementation_id: attribute; Stable implementation ID.
    """

    op_kind: TileOpKind
    implementation_id: str

    def supports(self, op: TileOp, *, context: LoweringContext) -> bool:
        """Report exact legality for one operation and context.

        Args:
            op: Typed operation.
            context: Explicit hardware and search choices.

        Returns:
            Whether this implementation can lower the operation.
        """
        ...

    def lower(self, op: TileOp, *, context: LoweringContext) -> LoweredTileOp:
        """Lower one legal operation to a complete phase fragment.

        Args:
            op: Typed operation.
            context: Explicit hardware and search choices.

        Returns:
            The complete untimed phase fragment.
        """
        ...


class TileOpImplementation:
    """Pair lowering and measurement for one implementation.

    Attributes:
        lowering: attribute; Typed operation lowerer.
        benchmark_provider: attribute; Matching CUDA benchmark provider.
    """

    lowering: TileOpLowering
    benchmark_provider: CudaBenchmarkProvider


class ImplementationCatalog:
    """Index installed operation implementations.

    Attributes:
        implementations: attribute; Unique lowering/provider pairs.
    """

    implementations: tuple[TileOpImplementation, ...]

    def choices_for(
        self,
        op: TileOp,
        *,
        context: LoweringContext,
        allowed_implementation_ids: tuple[str, ...],
    ) -> tuple[TileOpImplementation, ...]:
        """Resolve every legal allowed implementation.

        Args:
            op: Typed operation.
            context: Explicit lowering context.
            allowed_implementation_ids: Caller allowlist.

        Returns:
            Canonically ordered legal implementations.
        """
        ...


def b200_implementation_catalog() -> ImplementationCatalog:
    """Construct the installed B200 implementation catalog.

    Returns:
        The B200 lowering and benchmark pairs.
    """
    ...
```

- constraints:
  - One implementation owns both lowering and measurement for the same stable
    `(op_kind, implementation_id)` pair.
  - Duplicate implementation pairs and duplicate benchmark provider IDs are
    invalid.
  - Every emitted profile query MUST use the canonical typed operation
    signature without rewriting operation semantics.
  - For each program dependency, the destination requires one availability and
    the source MUST expose exactly one matching `(value_id, availability_id)`.
  - An asynchronous GEMM implementation MAY expose `ordered` availability at
    issue and `complete` availability at completion. Consumers select which
    availability they require; the generic composer and solver MUST NOT guess.
  - Shared value storage is charged once by value ID. Warp count is charged once
    per configuration. Implementation-private static demands are summed.
  - A configuration without pipeline-ring storage has canonical depth 1.
  - Cross-operation fusion is not inferred. A fused semantic operation requires
    an explicit typed operation and corresponding program-schema change.

## 8. Timing Profiles

### 8.1 Canonical Profile Identity

```python
class CanonicalAttribute:
    """Carry one canonical semantic or benchmark attribute.

    Attributes:
        name: attribute; Unique non-empty attribute name.
        value: attribute; Canonical string value.
    """

    name: str
    value: str


class TileOpSignature:
    """Describe typed operation semantics without program identity.

    Attributes:
        op_kind: attribute; Exact typed operation kind.
        operands: input; Operand tensor and memory-space types.
        results: attribute; Result tensor and memory-space types.
        semantic_attributes: attribute; Canonical operation semantics.
    """

    op_kind: TileOpKind
    operands: tuple[TileValueType, ...]
    results: tuple[TileValueType, ...]
    semantic_attributes: tuple[CanonicalAttribute, ...] = ()


def tile_op_signature(
    op: TileOp,
    *,
    program: TileProgram,
) -> TileOpSignature:
    """Derive the only canonical signature for one typed operation.

    Args:
        op: Typed source operation.
        program: Program owning every referenced value.

    Returns:
        Identity-free operation semantics.
    """
    ...


class TileOpProfileQuery:
    """Describe one exact implementation benchmark component.

    Attributes:
        hardware: attribute; Exact calibrated hardware reference.
        operation: attribute; Canonical typed operation signature.
        implementation_id: attribute; Stable implementation ID.
        component_id: attribute; Stable benchmark component ID.
        tile_shape: attribute; Concrete CTA tile shape.
        warp_config_id: attribute; Selected warp configuration ID.
        pipeline_depth: attribute; Selected pipeline depth.
        layout_variant_id: attribute; Selected layout variant.
        conditions: attribute; Canonical benchmark conditions.
    """

    hardware: HardwareSpecRef
    operation: TileOpSignature
    implementation_id: str
    component_id: str
    tile_shape: NamedShape
    warp_config_id: str
    pipeline_depth: int
    layout_variant_id: str
    conditions: tuple[CanonicalAttribute, ...] = ()


class ProfileRequirement:
    """Select one metric from one exact profile query.

    Attributes:
        query: attribute; Exact benchmark component query.
        timing_metric: attribute; Required latency or initiation interval.
    """

    query: TileOpProfileQuery
    timing_metric: TimingMetric


class BenchmarkFingerprint:
    """Identify exact benchmark source and compile behavior.

    Attributes:
        provider_id: attribute; Stable benchmark provider ID.
        provider_version: attribute; Stable provider contract version.
        benchmark_abi_version: attribute; Benchmark runner ABI version.
        source_sha256: attribute; Canonical CUDA source digest.
        compile_options_sha256: attribute; Canonical compile-option digest.
    """

    provider_id: str
    provider_version: str
    benchmark_abi_version: int
    source_sha256: str
    compile_options_sha256: str


class TileOpProfileKey:
    """Identify one exact measured benchmark component.

    Attributes:
        schema_version: attribute; Profile-key schema version.
        query: attribute; Exact typed benchmark query.
        fingerprint: attribute; Exact source and compile fingerprint.
    """

    schema_version: int
    query: TileOpProfileQuery
    fingerprint: BenchmarkFingerprint

    def canonical_json(self) -> str:
        """Serialize the canonical key payload.

        Returns:
            Sorted compact UTF-8 JSON text.
        """
        ...

    def key_id(self) -> ProfileKeyId:
        """Hash the canonical key payload.

        Returns:
            Lowercase SHA-256 key ID.
        """
        ...
```

- constraints:
  - `tile_op_signature` MUST derive shapes, dtypes, layouts, memory spaces,
    GEMM axes, reduction semantics, and elementwise semantics from typed records.
  - A signature MUST NOT contain program, operation, value, or phase IDs.
    Semantically identical operations in different programs share timing data.
  - Providers MUST NOT parse Python source, JSON text, phase names, or operation
    ID spelling to recover semantics.
  - Semantic attributes and conditions MUST each be sorted by `(name, value)`
    with unique names.
  - SHA-256 fields MUST be lowercase 64-character hexadecimal strings.
  - Canonical JSON uses sorted keys, UTF-8, no insignificant whitespace, and no
    floating-point values. `key_id` is the SHA-256 digest of those exact bytes.
  - A global-memory query MUST state canonical memory-residency and cache-policy
    conditions. Different conditions MUST NOT substitute for one another.
  - `component_id` names an implementation benchmark component, not a phase.
    The issue and completion phases for one asynchronous component use one query
    and select different metrics from one measurement.

### 8.2 Measurements And Snapshots

```python
class MeasurementOrigin(str, Enum):
    """Name an accepted timing origin."""

    MEASURED = "measured"


class ProfileEnvironment:
    """Describe the exact CUDA measurement environment.

    Attributes:
        environment_id: attribute; Canonical environment digest.
        device_uuid: attribute; CUDA device UUID.
        hardware: attribute; Exact calibrated hardware reference.
        cuda_arch: attribute; CUDA architecture target.
        driver_version: attribute; Driver version.
        runtime_version: attribute; CUDA runtime version.
        nvrtc_version: attribute; NVRTC version.
        device_clock_khz: attribute; Locked or observed device clock.
        memory_clock_khz: attribute; Locked or observed memory clock.
        power_limit_mw: attribute; Active power limit.
    """

    environment_id: str
    device_uuid: str
    hardware: HardwareSpecRef
    cuda_arch: str
    driver_version: str
    runtime_version: str
    nvrtc_version: str
    device_clock_khz: int | None
    memory_clock_khz: int | None
    power_limit_mw: int | None


class MeasurementPolicy:
    """Configure one stable CUDA measurement.

    Attributes:
        warmup_runs: attribute; Warmup launch count.
        sample_count: attribute; Retained aggregate sample count.
        target_sample_ns: attribute; Target device interval per sample.
        max_repetitions_per_sample: attribute; Device repetition cap.
        max_relative_iqr_ppm: attribute; Maximum accepted relative dispersion.
        retain_raw_samples: attribute; Whether raw samples enter the snapshot.
    """

    warmup_runs: int = 20
    sample_count: int = 100
    target_sample_ns: int = 100_000
    max_repetitions_per_sample: int = 1_000_000
    max_relative_iqr_ppm: int = 50_000
    retain_raw_samples: bool = False


class TileOpMeasurement:
    """Carry validated timing aggregates for one exact key.

    Attributes:
        measurement_id: attribute; Canonical measurement digest.
        key: attribute; Exact benchmark key.
        environment: attribute; Exact measurement environment.
        origin: attribute; Measurement origin.
        latency_p50_ps: attribute; Median latency.
        latency_p90_ps: attribute; Ninetieth-percentile latency.
        initiation_interval_p50_ps: attribute; Median issue interval, or None.
        initiation_interval_p90_ps: attribute; Ninetieth-percentile issue interval, or None.
        warmup_runs: attribute; Warmup launch count.
        sample_count: attribute; Aggregate sample count.
        latency_repetitions_per_sample: attribute; Latency repetitions.
        initiation_interval_repetitions_per_sample: attribute; Issue repetitions, or None.
        target_sample_ns: attribute; Requested sample interval.
        relative_iqr_ppm: attribute; Measured relative dispersion.
        raw_samples_retained: attribute; Whether raw samples are present.
        raw_latency_samples_ps: attribute; Retained latency samples.
        raw_initiation_interval_samples_ps: attribute; Retained issue samples.
        measured_at_utc: attribute; Canonical UTC timestamp.
    """

    measurement_id: MeasurementId
    key: TileOpProfileKey
    environment: ProfileEnvironment
    origin: MeasurementOrigin
    latency_p50_ps: int
    latency_p90_ps: int
    initiation_interval_p50_ps: int | None
    initiation_interval_p90_ps: int | None
    warmup_runs: int
    sample_count: int
    latency_repetitions_per_sample: int
    initiation_interval_repetitions_per_sample: int | None
    target_sample_ns: int
    relative_iqr_ppm: int
    raw_samples_retained: bool
    raw_latency_samples_ps: tuple[int, ...]
    raw_initiation_interval_samples_ps: tuple[int, ...]
    measured_at_utc: str


class ResolvedTiming:
    """Bind one phase requirement to selected measured durations.

    Attributes:
        requirement: attribute; Exact metric requirement.
        measurement_id: attribute; Selected measurement ID.
        profile_key_id: attribute; Selected profile-key ID.
        environment_id: attribute; Selected environment ID.
        selected_duration_ps: attribute; Primary duration.
        statistic: attribute; Primary statistic.
        sensitivity_duration_ps: attribute; Sensitivity duration.
        sensitivity_statistic: attribute; Sensitivity statistic.
    """

    requirement: ProfileRequirement
    measurement_id: MeasurementId
    profile_key_id: ProfileKeyId
    environment_id: str
    selected_duration_ps: int
    statistic: TimingStatistic
    sensitivity_duration_ps: int
    sensitivity_statistic: TimingStatistic


class SnapshotState(str, Enum):
    """Name profile snapshot mutability."""

    DRAFT = "draft"
    FROZEN = "frozen"


class SqliteProfileStore:
    """Own one SQLite profile-store connection."""

    def close(self) -> None:
        """Close the owned database connection."""
        ...

    def create_snapshot(
        self,
        *,
        snapshot_id: str,
        hardware: HardwareSpecRef,
        description: str,
        base: ProfileSnapshotRef | None = None,
    ) -> ProfileSnapshotRef:
        """Create one new draft snapshot revision.

        Args:
            snapshot_id: Snapshot family ID.
            hardware: Exact hardware reference.
            description: Human-readable description.
            base: Optional frozen base revision.

        Returns:
            The new draft revision.
        """
        ...

    def snapshot_state(self, ref: ProfileSnapshotRef) -> SnapshotState:
        """Return one snapshot state.

        Args:
            ref: Exact snapshot revision.

        Returns:
            Draft or frozen state.
        """
        ...

    def lookup(
        self,
        ref: ProfileSnapshotRef,
        key: TileOpProfileKey,
    ) -> TileOpMeasurement | None:
        """Look up one exact snapshot key.

        Args:
            ref: Exact snapshot revision.
            key: Exact profile key.

        Returns:
            The selected measurement, or None.
        """
        ...

    def insert(
        self,
        ref: ProfileSnapshotRef,
        measurement: TileOpMeasurement,
    ) -> None:
        """Insert one complete measurement atomically.

        Args:
            ref: Writable draft revision.
            measurement: Complete validated measurement.
        """
        ...

    def freeze(self, ref: ProfileSnapshotRef) -> ProfileSnapshotRef:
        """Freeze one non-empty draft revision.

        Args:
            ref: Draft revision.

        Returns:
            The same now-frozen revision.
        """
        ...

    def export_snapshot(self, ref: ProfileSnapshotRef, output: Path) -> None:
        """Export one deterministic snapshot document.

        Args:
            ref: Exact snapshot revision.
            output: Destination path.
        """
        ...

    def import_snapshot(self, source: Path) -> ProfileSnapshotRef:
        """Import one deterministic snapshot document.

        Args:
            source: Source document path.

        Returns:
            Imported snapshot revision.
        """
        ...


def open_profile_store(path: Path, *, writable: bool) -> SqliteProfileStore:
    """Open and validate one profile database.

    Args:
        path: SQLite database path.
        writable: Whether writes are permitted.

    Returns:
        An owned store connection.
    """
    ...
```

- constraints:
  - Every aggregate duration and raw sample MUST be positive integer
    picoseconds.
  - A measurement above the policy dispersion threshold MUST NOT become usable.
  - The selected duration is the requested p50 or p90 of the exact metric.
    Sensitivity uses p90; when primary selection is p90, both durations match.
  - The resolver MUST NOT interpolate across operations, shapes, memory spaces,
    metrics, implementations, environments, or conditions.
  - When raw samples are absent, both raw tuples MUST be empty. When present,
    latency has `sample_count` entries and initiation interval has either zero or
    `sample_count` entries.
  - A draft snapshot pins its environment on first insertion and MUST reject a
    later environment or hardware mismatch.
  - A snapshot contains exactly one measurement per profile key. Frozen
    snapshots are immutable.
  - Creation, insertion, freezing, import, and migration MUST be transactional.
    A failed operation MUST leave no usable partial row.
  - Read-only operation MUST use SQLite read-only mode. Writable operation MUST
    enable foreign keys, WAL, and full synchronous durability.

The profile database schema is:

```sql
CREATE TABLE metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE profile_keys (
    key_id TEXT PRIMARY KEY,
    canonical_json TEXT NOT NULL UNIQUE
);

CREATE TABLE environments (
    environment_id TEXT PRIMARY KEY,
    canonical_json TEXT NOT NULL UNIQUE
);

CREATE TABLE measurements (
    measurement_id TEXT PRIMARY KEY,
    key_id TEXT NOT NULL,
    environment_id TEXT NOT NULL,
    aggregate_json TEXT NOT NULL,
    created_at_utc TEXT NOT NULL,
    UNIQUE (measurement_id, key_id),
    FOREIGN KEY (key_id) REFERENCES profile_keys(key_id),
    FOREIGN KEY (environment_id) REFERENCES environments(environment_id)
);

CREATE TABLE samples (
    measurement_id TEXT NOT NULL,
    metric TEXT NOT NULL CHECK (metric IN ('latency', 'initiation_interval')),
    sample_index INTEGER NOT NULL,
    elapsed_ps INTEGER NOT NULL,
    PRIMARY KEY (measurement_id, metric, sample_index),
    FOREIGN KEY (measurement_id) REFERENCES measurements(measurement_id)
);

CREATE TABLE snapshots (
    snapshot_id TEXT NOT NULL,
    revision INTEGER NOT NULL,
    state TEXT NOT NULL CHECK (state IN ('draft', 'frozen')),
    hardware_ref_json TEXT NOT NULL,
    environment_id TEXT,
    description TEXT NOT NULL,
    created_at_utc TEXT NOT NULL,
    PRIMARY KEY (snapshot_id, revision),
    FOREIGN KEY (environment_id) REFERENCES environments(environment_id)
);

CREATE TABLE snapshot_measurements (
    snapshot_id TEXT NOT NULL,
    snapshot_revision INTEGER NOT NULL,
    key_id TEXT NOT NULL,
    measurement_id TEXT NOT NULL,
    PRIMARY KEY (snapshot_id, snapshot_revision, key_id),
    FOREIGN KEY (snapshot_id, snapshot_revision)
        REFERENCES snapshots(snapshot_id, revision),
    FOREIGN KEY (key_id) REFERENCES profile_keys(key_id),
    FOREIGN KEY (measurement_id, key_id)
        REFERENCES measurements(measurement_id, key_id)
);
```

### 8.3 CUDA Benchmark Contract

```python
class CudaBufferRole(str, Enum):
    """Name one CUDA benchmark buffer role."""

    INPUT = "input"
    OUTPUT = "output"
    INOUT = "inout"


class CudaBufferInit(str, Enum):
    """Name one deterministic buffer initialization."""

    ZERO = "zero"
    RANDOM_UNIFORM = "random_uniform"
    SEQUENCE = "sequence"


class CudaScalarDType(str, Enum):
    """Name one CUDA benchmark scalar type."""

    I32 = "i32"
    I64 = "i64"
    U32 = "u32"
    U64 = "u64"
    F32 = "f32"
    F64 = "f64"


class CudaBufferArgument:
    """Describe one CUDA benchmark buffer argument.

    Attributes:
        name: attribute; Kernel argument name.
        nbytes: attribute; Positive allocation bytes.
        role: attribute; Input/output role.
        initialization: attribute; Deterministic initialization.
        seed: attribute; Initialization seed.
    """

    name: str
    nbytes: int
    role: CudaBufferRole
    initialization: CudaBufferInit
    seed: int = 0


class CudaScalarArgument:
    """Describe one CUDA benchmark scalar argument.

    Attributes:
        name: attribute; Kernel argument name.
        dtype: attribute; Scalar type.
        value: attribute; Scalar value.
    """

    name: str
    dtype: CudaScalarDType
    value: int | float


CudaArgument = CudaBufferArgument | CudaScalarArgument


class CudaLaunchSpec:
    """Describe one CUDA launch shape.

    Attributes:
        grid: attribute; Three-dimensional grid extent.
        block: attribute; Three-dimensional block extent.
        dynamic_smem_bytes: attribute; Dynamic shared-memory bytes.
    """

    grid: tuple[int, int, int]
    block: tuple[int, int, int]
    dynamic_smem_bytes: int


class CudaBenchmarkCase:
    """Describe one latency or issue-throughput launch.

    Attributes:
        metric: attribute; Measured timing metric.
        kernel_name: attribute; CUDA entry name.
        launch: attribute; Exact launch shape.
        arguments: attribute; Ordered typed arguments.
        repetition_argument_name: attribute; Repetition scalar argument.
    """

    metric: TimingMetric
    kernel_name: str
    launch: CudaLaunchSpec
    arguments: tuple[CudaArgument, ...]
    repetition_argument_name: str


class CudaBenchmark:
    """Carry one exact JIT benchmark artifact.

    Attributes:
        key: attribute; Exact profile key.
        source_utf8: attribute; Complete CUDA source.
        compile_options: attribute; Canonical NVRTC options.
        latency_case: attribute; Dependency-chain latency case.
        initiation_interval_case: attribute; Independent-chain issue case.
    """

    key: TileOpProfileKey
    source_utf8: str
    compile_options: tuple[str, ...]
    latency_case: CudaBenchmarkCase
    initiation_interval_case: CudaBenchmarkCase | None


class NamedBufferOutput:
    """Carry one correctness output.

    Attributes:
        metric: attribute; Producing benchmark metric.
        name: attribute; Output name.
        data: attribute; Returned raw bytes.
    """

    metric: TimingMetric
    name: str
    data: bytes


class ProfileRun:
    """Carry raw results from one CUDA benchmark.

    Attributes:
        environment: attribute; Exact runtime environment.
        latency_samples_ps: attribute; Positive latency samples.
        initiation_interval_samples_ps: attribute; Positive issue samples.
        latency_repetitions_per_sample: attribute; Latency repetitions.
        initiation_interval_repetitions_per_sample: attribute; Issue repetitions.
        outputs: attribute; Correctness outputs.
    """

    environment: ProfileEnvironment
    latency_samples_ps: tuple[int, ...]
    initiation_interval_samples_ps: tuple[int, ...]
    latency_repetitions_per_sample: int
    initiation_interval_repetitions_per_sample: int | None
    outputs: tuple[NamedBufferOutput, ...]


class CudaBenchmarkProvider(Protocol):
    """Materialize and validate one implementation benchmark."""

    provider_id: str
    provider_version: str

    def supports(self, query: TileOpProfileQuery) -> bool:
        """Report exact support for one query.

        Args:
            query: Exact typed benchmark query.

        Returns:
            Whether this provider owns the query.
        """
        ...

    def fingerprint(
        self,
        query: TileOpProfileQuery,
        hardware: HardwareSpec,
    ) -> BenchmarkFingerprint:
        """Compute source and compile identity without running CUDA.

        Args:
            query: Exact typed benchmark query.
            hardware: Exact hardware document.

        Returns:
            Canonical benchmark fingerprint.
        """
        ...

    def materialize(
        self,
        key: TileOpProfileKey,
        hardware: HardwareSpec,
    ) -> CudaBenchmark:
        """Materialize one exact benchmark artifact.

        Args:
            key: Exact profile key.
            hardware: Exact hardware document.

        Returns:
            Complete CUDA benchmark.
        """
        ...

    def validate(self, benchmark: CudaBenchmark, run: ProfileRun) -> None:
        """Validate benchmark outputs before storage.

        Args:
            benchmark: Materialized benchmark.
            run: Raw CUDA run.
        """
        ...


class ProfileRunner(Protocol):
    """Execute one materialized timing benchmark."""

    def run(
        self,
        benchmark: CudaBenchmark,
        *,
        hardware: HardwareSpec,
        policy: MeasurementPolicy,
    ) -> ProfileRun:
        """Run one benchmark under an exact policy.

        Args:
            benchmark: Complete benchmark artifact.
            hardware: Exact hardware document.
            policy: Measurement policy.

        Returns:
            Raw measured samples and outputs.
        """
        ...


class LocalCudaProfileRunner:
    """Execute benchmarks on one local CUDA device.

    Attributes:
        cache_dir: attribute; Compiled-artifact cache directory.
    """

    cache_dir: Path

    def run(
        self,
        benchmark: CudaBenchmark,
        *,
        hardware: HardwareSpec,
        policy: MeasurementPolicy,
    ) -> ProfileRun:
        """Run one benchmark on the exact local B200.

        Args:
            benchmark: Complete benchmark artifact.
            hardware: Exact hardware document.
            policy: Measurement policy.

        Returns:
            Raw measured samples and outputs.
        """
        ...


def summarize_profile_run(
    key: TileOpProfileKey,
    run: ProfileRun,
    *,
    policy: MeasurementPolicy,
    measured_at_utc: str,
) -> TileOpMeasurement:
    """Validate and summarize one raw profile run.

    Args:
        key: Exact profile key.
        run: Raw measured samples and outputs.
        policy: Measurement policy.
        measured_at_utc: Canonical UTC timestamp.

    Returns:
        Complete immutable measurement.
    """
    ...
```

- constraints:
  - The initial launch grid MUST be `(1, 1, 1)`.
  - CUDA dependencies MUST be imported lazily inside `run`.
  - Compilation, allocation, initialization, argument setup, and first-use
    overhead MUST occur outside CUDA event timing.
  - The latency case MUST form a dependency chain and divide elapsed device time
    by repetitions. The initiation-interval case MUST use enough independent
    chains to saturate issue throughput and divide by issued operations.
  - CUDA events MUST cover device work only. Sub-microsecond operations MUST be
    repeated on device until the target sample interval or repetition cap.
  - The runner MUST verify that source and compile-option hashes match the key
    before compilation.
  - The provider MUST validate complete correctness outputs before a measurement
    is summarized or stored.
  - Summary uses nearest-rank p50 and p90 and integer relative-IQR ppm.
  - `measurement_id` hashes canonical key, environment, aggregates, retained raw
    samples, and timestamp.

### 8.4 Profile Resolution

```python
class BenchmarkProviderCatalog:
    """Resolve one benchmark provider per query.

    Attributes:
        providers: attribute; Unique benchmark providers.
    """

    providers: tuple[CudaBenchmarkProvider, ...]

    def provider_for(self, query: TileOpProfileQuery) -> CudaBenchmarkProvider:
        """Resolve exactly one supporting provider.

        Args:
            query: Exact typed benchmark query.

        Returns:
            The unique supporting provider.
        """
        ...


class ProfileResolver:
    """Resolve exact requirements from snapshots or explicit JIT."""

    def __init__(
        self,
        *,
        store: SqliteProfileStore,
        providers: BenchmarkProviderCatalog,
        runner: ProfileRunner | None,
        measurement_policy: MeasurementPolicy,
    ) -> None:
        """Initialize one explicit resolver.

        Args:
            store: Owned profile store.
            providers: Exact benchmark providers.
            runner: Optional JIT runner.
            measurement_policy: JIT measurement policy.
        """
        ...

    def resolve_many(
        self,
        requirements: tuple[ProfileRequirement, ...],
        *,
        hardware: HardwareSpec,
        selection: ProfileSelection,
    ) -> tuple[ResolvedTiming, ...]:
        """Resolve every exact timing requirement.

        Args:
            requirements: Ordered timing requirements.
            hardware: Exact hardware document.
            selection: Snapshot and miss behavior.

        Returns:
            Timings in first-use requirement order.
        """
        ...
```

- constraints:
  - Provider selection requires exactly one supporting provider.
  - Resolution MUST deduplicate canonical keys and preserve first-use
    requirement order.
  - Require-only mode MUST collect every missing key before raising
    `MissingProfileError` and MUST NOT modify the store.
  - JIT mode without a runner MUST raise `UnsupportedError`.
  - Initial JIT resolution is sequential and each successful complete
    measurement is inserted atomically before it becomes usable.

## 9. Build And Search Problem

```python
class ConfigurationBuilder:
    """Enumerate and compose untimed canonical configurations."""

    def __init__(self, *, implementations: ImplementationCatalog) -> None:
        """Initialize one explicit configuration builder.

        Args:
            implementations: Installed operation implementations.
        """
        ...

    def enumerate_templates(
        self,
        programs: tuple[TileProgram, ...],
        *,
        search_space: SearchSpace,
        hardware: HardwareSpec,
    ) -> tuple[ConfigurationTemplate, ...]:
        """Enumerate every legal canonical untimed configuration.

        Args:
            programs: Concrete tile-program variants.
            search_space: Finite implementation, warp, depth, and layout choices.
            hardware: Exact hardware document.

        Returns:
            Configurations sorted by canonical ID.
        """
        ...
```

- constraints:
  - Enumeration covers the Cartesian product of program, warp configuration,
    pipeline depth, layout variant, and compatible per-operation
    implementations.
  - Hardware-illegal combinations, missing or ambiguous availability bindings,
    invalid phase graphs, invalid loop-barrier graphs, and static-capacity
    violations MUST be rejected before profile resolution.
  - Canonically identical configurations MUST be deduplicated before candidate
    counting or profile lookup.
  - `SearchSpace.max_candidates` applies to canonical legal configurations.
    Exceeding it raises `WorkloadError`; an empty legal set raises
    `UnsupportedError`.
  - A configuration ID is the SHA-256 digest of canonical JSON containing the
    complete program, hardware reference, implementation selections, warp
    configuration, depth, layout, phases, constraints, buffers, and static
    demands. Enumeration order MUST NOT affect it.

After profile resolution, solver-owned immutable records contain only numeric
timings and stable provenance:

```python
class Phase:
    """Carry one fully timed implementation phase.

    Attributes:
        phase_id: attribute; Configuration-local phase ID.
        source_op_id: attribute; Originating typed operation ID.
        implementation_id: attribute; Selected implementation ID.
        phase_name: attribute; Implementation-local phase name.
        component_id: attribute; Benchmark component ID.
        domain: attribute; Preserved source domain.
        duration_ticks: attribute; Primary upward-rounded duration.
        sensitivity_duration_ticks: attribute; Sensitivity duration.
        warp_ids: attribute; Bound selected warp IDs.
        temporal_demands: attribute; Reserved temporal resources.
        measurement_id: attribute; Selected measurement ID.
        profile_key_id: attribute; Selected profile-key ID.
        environment_id: attribute; Selected environment ID.
        timing_metric: attribute; Selected timing metric.
        timing_statistic: attribute; Primary timing statistic.
        sensitivity_timing_statistic: attribute; Sensitivity statistic.
    """

    phase_id: PhaseId
    source_op_id: OpId
    implementation_id: str
    phase_name: str
    component_id: str
    domain: PhaseIterationDomain
    duration_ticks: int
    sensitivity_duration_ticks: int
    warp_ids: tuple[int, ...]
    temporal_demands: tuple[TemporalDemand, ...]
    measurement_id: MeasurementId
    profile_key_id: ProfileKeyId
    environment_id: str
    timing_metric: TimingMetric
    timing_statistic: TimingStatistic
    sensitivity_timing_statistic: TimingStatistic


class Configuration:
    """Carry one complete solver-ready candidate.

    Attributes:
        configuration_id: attribute; Canonical configuration ID.
        program_id: attribute; Source program ID.
        workload_kind: attribute; Logical workload kind.
        tile: attribute; Concrete CTA tile.
        implementations: attribute; Per-operation implementation choices.
        warps: attribute; Selected warp configuration.
        pipeline_depth: attribute; Selected ring depth.
        layout_variant_id: attribute; Selected layout variant.
        loops: attribute; Repeated regions.
        phases: attribute; Fully timed phases.
        dependencies: attribute; End-to-start constraints.
        loop_barriers: attribute; Explicit inter-loop control barriers.
        start_alignments: attribute; Exact start-offset constraints.
        buffers: attribute; Value allocations and reuse constraints.
        static_demands: attribute; Complete per-CTA static demand.
    """

    configuration_id: ConfigurationId
    program_id: ProgramId
    workload_kind: WorkloadKind
    tile: TileCandidate
    implementations: tuple[OpImplementationSelection, ...]
    warps: WarpConfig
    pipeline_depth: int
    layout_variant_id: str
    loops: tuple[LoopTemplate, ...]
    phases: tuple[Phase, ...]
    dependencies: tuple[PhaseDependency, ...]
    loop_barriers: tuple[LoopBarrier, ...]
    start_alignments: tuple[PhaseStartAlignment, ...]
    buffers: tuple[BufferTemplate, ...]
    static_demands: tuple[StaticDemand, ...]


class RejectedCandidate:
    """Explain one candidate rejected before or during solve.

    Attributes:
        configuration_id: attribute; Candidate ID, or None before identity.
        code: attribute; Stable diagnostic code.
        message: attribute; Human-readable explanation.
    """

    configuration_id: ConfigurationId | None
    code: DiagnosticCode
    message: str


class SearchProblem:
    """Carry one complete replayable numeric search problem.

    Attributes:
        schema_version: attribute; Search-problem schema version.
        request_id: attribute; Source request ID.
        hardware: attribute; Complete hardware document.
        workload: attribute; Logical workload.
        programs: attribute; Source typed program variants.
        profile_snapshot: attribute; Selected snapshot revision.
        solver_options: attribute; Frozen solve controls.
        configurations: attribute; Timed canonical candidates.
        rejected_before_solve: attribute; Earlier candidate diagnostics.
    """

    schema_version: int
    request_id: str
    hardware: HardwareSpec
    workload: WorkloadSpec
    programs: tuple[TileProgram, ...]
    profile_snapshot: ProfileSnapshotRef
    solver_options: SolverOptions
    configurations: tuple[Configuration, ...]
    rejected_before_solve: tuple[RejectedCandidate, ...]
```

- constraints:
  - `SearchProblem.schema_version` MUST equal
    `SEARCH_PROBLEM_SCHEMA_VERSION`.
  - `build` converts each selected and sensitivity duration with
    `ticks = ceil(duration_ps / time_resolution_ps)`.
  - A search problem MUST contain no protocol, callback, arbitrary payload,
    unresolved query, floating-point modeled duration, database handle, CUDA
    value, or OR-Tools value.
  - Typed programs remain only for source-operation provenance. Candidate
    schedulers MUST NOT inspect operation semantics.
  - Search-problem JSON plus the recorded solver package version MUST be
    sufficient to replay `solve` without CUDA, SQLite, frontends, or operation
    implementations.

The public build operation is:

```python
def build(
    request: CostModelRequest,
    *,
    hardware_catalog: HardwareCatalog,
    implementation_catalog: ImplementationCatalog,
    profile_store: SqliteProfileStore,
    profile_runner: ProfileRunner | None = None,
    measurement_policy: MeasurementPolicy = MeasurementPolicy(),
) -> SearchProblem:
    """Build one complete replayable search problem.

    Args:
        request: Typed finite search request.
        hardware_catalog: Exact installed hardware documents.
        implementation_catalog: Installed operation implementations.
        profile_store: Timing snapshot store.
        profile_runner: Optional explicit JIT runner.
        measurement_policy: JIT measurement policy.

    Returns:
        A complete solver-ready numeric problem.
    """
    ...
```

- constraints:
  - `build` performs, in order: request validation, exact hardware resolution,
    canonical configuration enumeration, phase and static-capacity validation,
    provider-catalog construction, complete profile resolution, conservative
    quantization, and stable problem construction.
  - `build` MUST NOT import or invoke OR-Tools.
  - Every profile requirement across every legal candidate MUST resolve before
    the search problem is returned.

## 10. Solver Contract

### 10.1 Candidate Interface

```python
class TimelineRegion(str, Enum):
    """Name one compact exported schedule region."""

    FINITE = "finite"
    PROLOGUE = "prologue"
    STEADY = "steady"
    EPILOGUE = "epilogue"


class CompactPlacement:
    """Carry one solver-internal phase placement.

    Attributes:
        phase_id: attribute; Placed phase ID.
        loop_id: attribute; Loop ID, or None for one-time work.
        region: attribute; Compact timeline region.
        iteration: attribute; Region-relative iteration label.
        start_ticks: attribute; Inclusive start tick.
        end_ticks: attribute; Exclusive end tick.
    """

    phase_id: PhaseId
    loop_id: LoopId | None
    region: TimelineRegion
    iteration: int
    start_ticks: int
    end_ticks: int


class CompactLoopTiming:
    """Carry one loop's compact timing.

    Attributes:
        loop_id: attribute; Source loop ID.
        initiation_interval_ticks: attribute; Periodic II, or None.
        prologue_ticks: attribute; Boundary ticks before steady repetition.
        epilogue_ticks: attribute; Boundary ticks after final repetition.
    """

    loop_id: LoopId
    initiation_interval_ticks: int | None
    prologue_ticks: int
    epilogue_ticks: int


class CompactSchedule:
    """Carry one candidate's compact numeric schedule.

    Attributes:
        end_to_end_ticks: attribute; Global schedule span.
        loop_timings: attribute; Per-loop compact timings.
        placements: attribute; Finite or compact phase placements.
    """

    end_to_end_ticks: int
    loop_timings: tuple[CompactLoopTiming, ...]
    placements: tuple[CompactPlacement, ...]


class CandidateSolveResult:
    """Carry one independently solved candidate outcome.

    Attributes:
        status: attribute; Candidate solve status.
        configuration_id: attribute; Solved candidate ID.
        schedule: attribute; Incumbent schedule, or None.
        objective_ticks: attribute; Incumbent objective, or None.
        lower_bound_ticks: attribute; Proven lower bound, or None.
        sensitivity_ticks: attribute; Sensitivity objective, or None.
        diagnostics: attribute; Stable diagnostics.
    """

    status: EvaluationStatus
    configuration_id: ConfigurationId
    schedule: CompactSchedule | None
    objective_ticks: int | None
    lower_bound_ticks: int | None
    sensitivity_ticks: int | None
    diagnostics: tuple[Diagnostic, ...]


class CandidateScheduler(Protocol):
    """Solve one fixed timed configuration."""

    def solve(
        self,
        configuration: Configuration,
        *,
        hardware: HardwareSpec,
        options: SolverOptions,
    ) -> CandidateSolveResult:
        """Solve one independent configuration.

        Args:
            configuration: Fully timed candidate.
            hardware: Exact resource capacities.
            options: Frozen solver controls.

        Returns:
            Candidate schedule, bound, and diagnostics.
        """
        ...
```

- constraints:
  - Candidate schedulers MUST NOT select configurations, read profiles, inspect
    typed operations, or invoke CUDA.
  - Every candidate receives its own `candidate_timeout_s`; unused time MUST NOT
    transfer to another candidate.
  - If every loop has at most `finite_unroll_limit` iterations, exact finite
    scheduling applies. If any loop is longer, periodic scheduling applies.

### 10.2 Exact Finite Formulation

For every phase `p` and covered logical instance `i`, finite scheduling creates:

```text
start[p, i] in [0, horizon]
end[p, i] = start[p, i] + duration[p]
interval[p, i] = [start[p, i], end[p, i])
```

- constraints:
  - `horizon` is the checked integer sum of all phase-instance durations and
    positive dependency delays. Overflow raises `SearchProblemError`.
  - The minimum start is fixed to zero to remove translated equivalent
    schedules.
  - Every aligned dependency adds
    `start[dst, i + relation.iteration_distance] >= end[src, i] +
    ceil(delay_ps / resolution_ps)` for every valid pair.
  - Every endpoint dependency adds one inequality between its selected source
    and destination endpoints. It adds no inequalities for other instances.
  - Every loop barrier adds
    `min(start[dst_loop, *]) >= max(end[src_loop, *]) +
    ceil(delay_ps / resolution_ps)`; an implementation MAY encode this as
    pairwise inequalities when the result is equivalent.
  - Every start alignment adds
    `start[dst, i] == start[src, i] + ceil(offset_ps / resolution_ps)`.
  - For each temporal resource, unit demands at capacity one use `NoOverlap`;
    every other case uses `Cumulative(intervals, demands, capacity_slots)`.
  - The same interval is submitted to every resource a phase demands.
  - For every buffer with `d` slots and every release phase,
    `start[producer, i + d] >= end[release, i]`.
  - Static demand is validated before CP-SAT and MUST NOT be charged again.
  - Makespan is the maximum phase end. The primary solve minimizes makespan
    using primary durations.
  - Sensitivity MUST be independently re-solved with sensitivity durations. A
    p90 duration MUST NOT be substituted into a p50 placement.

### 10.3 Asynchronous Operation Semantics

An asynchronous implementation represents issue throughput and completion
latency as two intervals with the same start:

```text
issue interval:    duration = measured initiation interval
latency interval:  duration = measured start-to-completion latency
start(issue) == start(latency)
consumer starts after end(latency)
```

- constraints:
  - The issue interval reserves warp-issue and primary issue resources.
  - The latency interval reserves calibrated in-flight and data-path resources.
  - Later ordered issue MAY begin after the earlier issue interval while the
    earlier latency interval remains active, subject to in-flight capacity.
  - Readiness comes from the selected named availability. The generic solver
    MUST NOT force completion when an implementation selected ordered issue.

### 10.4 Periodic Formulation

For each long loop, periodic scheduling searches integer fixed-II feasibility:

1. Compute recurrence and temporal-resource lower bounds.
2. Compute a legal serial upper bound for one iteration.
3. Binary-search fixed integer II values and scan every skipped integer tick
   between the final infeasible and feasible bounds.
4. Re-solve the selected configuration with sensitivity durations.

For one fixed II, every complete-loop phase uses:

```text
start[p, i] = offset[p] + i * II
```

The replicated feasibility window radius is:

```text
max_aligned_dependency_distance
+ max_buffer_slot_count
+ ceil(max_phase_duration_ticks / II)
+ 1
```

- constraints:
  - The replicated window includes enough negative and positive iterations to
    enforce every aligned dependency, start alignment, buffer-reuse edge,
    `NoOverlap`, and `Cumulative` constraint around a central period. Endpoint
    dependencies and loop barriers are enforced by the exact boundary model,
    not replicated as steady-state edges.
  - Boundary-domain phases use an exact boundary model connected to first or
    last steady instances by the same constraints.
  - Only one central period is exported as `STEADY`; exact boundaries are
    exported as `PROLOGUE` and `EPILOGUE`.
  - Multiple long loops are solved as separate periodic regions only when their
    ordering is stated by explicit `LoopBarrier` records or their phase
    dependencies prove the same ordering. A schedule-tree sequence MUST NOT be
    treated as proof. Without a representable ordering, the candidate is
    `UNSUPPORTED` for the periodic solver rather than silently serialized.
  - For a loop of `N` iterations,
    `span = prologue + (N - 1) * II + epilogue` MUST hold exactly.
  - A periodic result requires feasible CP-SAT placement and reconstructed
    boundary timing. A resource lower bound alone is never a result.
  - Retained small-loop equivalence cases MUST agree with exact finite
    expansion.

### 10.5 Global Candidate Search

```python
class SearchCoordinator:
    """Solve and compare every canonical configuration."""

    def __init__(
        self,
        *,
        finite_scheduler: CandidateScheduler,
        periodic_scheduler: CandidateScheduler,
    ) -> None:
        """Initialize one explicit candidate coordinator.

        Args:
            finite_scheduler: Exact finite scheduler.
            periodic_scheduler: Fixed-II periodic scheduler.
        """
        ...

    def solve(self, problem: SearchProblem) -> CostModelResult:
        """Solve and compare one complete search problem.

        Args:
            problem: Complete numeric search problem.

        Returns:
            Global incumbent, proof, and diagnostics.
        """
        ...


def solve(problem: SearchProblem) -> CostModelResult:
    """Solve one complete replayable search problem.

    Args:
        problem: Complete numeric search problem.

    Returns:
        Global result and optional selected plan.
    """
    ...
```

- constraints:
  - Candidate order is configuration-ID order.
  - An outer monotonic deadline marks every unstarted legal candidate as timed
    out after `search_timeout_s`.
  - Candidate ordering is primary objective ticks, sensitivity ticks, total
    static bytes, pipeline depth, then configuration ID.
  - The global lower bound is the minimum valid bound over every legal
    candidate. An unstarted legal candidate has lower bound zero.
  - `OPTIMAL` requires an incumbent and proof that every legal candidate is
    infeasible or cannot beat it. Otherwise an incumbent is `FEASIBLE` or
    `TIMEOUT` according to the exhausted budget.
  - Deterministic complete solves over byte-identical search problems MUST emit
    byte-identical result JSON under the same solver package version.

## 11. Result Contract

```python
class EvaluationStatus(str, Enum):
    """Name one public cost-model outcome."""

    OPTIMAL = "optimal"
    FEASIBLE = "feasible"
    TIMEOUT = "timeout"
    INFEASIBLE = "infeasible"
    UNSUPPORTED = "unsupported"
    MISSING_PROFILE = "missing_profile"
    PROFILE_FAILED = "profile_failed"


class DiagnosticCode(str, Enum):
    """Name one stable diagnostic category."""

    INVALID_CANDIDATE = "invalid_candidate"
    STATIC_CAPACITY = "static_capacity"
    MISSING_PROFILE = "missing_profile"
    PROFILE_FAILED = "profile_failed"
    UNSUPPORTED = "unsupported"
    INFEASIBLE = "infeasible"
    TIMEOUT = "timeout"


class Diagnostic:
    """Carry one stable diagnostic.

    Attributes:
        code: attribute; Stable diagnostic category.
        message: attribute; Human-readable explanation.
        subject_id: attribute; Related stable ID, or None.
    """

    code: DiagnosticCode
    message: str
    subject_id: str | None = None


class ResourceReservation:
    """Describe one placed temporal-resource demand.

    Attributes:
        resource_id: attribute; Exact temporal resource ID.
        slots: attribute; Positive reserved slots.
    """

    resource_id: ResourceId
    slots: int


class PhasePlacement:
    """Place one phase instance on the exported timeline.

    Attributes:
        phase_id: attribute; Placed phase ID.
        source_op_id: attribute; Originating typed operation ID.
        loop_id: attribute; Loop ID, or None for one-time work.
        region: attribute; Finite or compact timeline region.
        iteration: attribute; Region-relative instance label.
        start_ps: attribute; Inclusive start picosecond.
        end_ps: attribute; Exclusive end picosecond.
        warp_ids: attribute; Bound warp IDs.
        resources: attribute; Reserved temporal resources.
    """

    phase_id: PhaseId
    source_op_id: OpId
    loop_id: LoopId | None
    region: TimelineRegion
    iteration: int
    start_ps: int
    end_ps: int
    warp_ids: tuple[int, ...]
    resources: tuple[ResourceReservation, ...]


class BufferAllocation:
    """Describe one selected value allocation.

    Attributes:
        buffer_id: attribute; Configuration-local buffer ID.
        value_id: attribute; Source program value ID.
        storage_resource_id: attribute; Static storage resource ID.
        bytes_per_slot: attribute; Bytes per slot.
        slot_count: attribute; Selected slot count.
        total_bytes: attribute; Complete allocation bytes.
    """

    buffer_id: BufferId
    value_id: ValueId
    storage_resource_id: ResourceId
    bytes_per_slot: int
    slot_count: int
    total_bytes: int


class ResourceUtilization:
    """Summarize one resource over the full logical CTA.

    Attributes:
        resource_id: attribute; Exact temporal resource ID.
        capacity_slots: attribute; Resource capacity.
        busy_slot_ps: attribute; Integrated occupied slot-picoseconds.
        horizon_ps: attribute; Global utilization horizon.
    """

    resource_id: ResourceId
    capacity_slots: int
    busy_slot_ps: int
    horizon_ps: int


class LoopTiming:
    """Describe one selected loop schedule.

    Attributes:
        loop_id: attribute; Source loop ID.
        initiation_interval_ps: attribute; Periodic II, or None.
        prologue_ps: attribute; Boundary duration before steady repetition.
        epilogue_ps: attribute; Boundary duration after final repetition.
        span_ps: attribute; Complete loop-region span.
    """

    loop_id: LoopId
    initiation_interval_ps: int | None
    prologue_ps: int
    epilogue_ps: int
    span_ps: int


class ProfileProvenance:
    """Map one selected phase to exact measured timing.

    Attributes:
        phase_id: attribute; Selected phase ID.
        source_op_id: attribute; Originating typed operation ID.
        implementation_id: attribute; Selected implementation ID.
        phase_name: attribute; Implementation-local phase name.
        component_id: attribute; Benchmark component ID.
        measurement_id: attribute; Selected measurement ID.
        profile_key_id: attribute; Selected profile-key ID.
        environment_id: attribute; Selected environment ID.
        timing_metric: attribute; Selected metric.
        statistic: attribute; Primary statistic.
        sensitivity_statistic: attribute; Sensitivity statistic.
    """

    phase_id: PhaseId
    source_op_id: OpId
    implementation_id: str
    phase_name: str
    component_id: str
    measurement_id: MeasurementId
    profile_key_id: ProfileKeyId
    environment_id: str
    timing_metric: TimingMetric
    statistic: TimingStatistic
    sensitivity_statistic: TimingStatistic


class SolveProof:
    """Carry global objective and optimality evidence.

    Attributes:
        status: attribute; Global solve status.
        objective_ps: attribute; Incumbent end-to-end time.
        lower_bound_ps: attribute; Global lower bound.
        sensitivity_ps: attribute; Sensitivity objective.
        optimality_gap_ppm: attribute; Integer relative optimality gap.
        solver_name: attribute; Solver implementation name.
        solver_version: attribute; Solver package version.
        candidate_count: attribute; Legal canonical candidate count.
        solved_candidate_count: attribute; Started candidate count.
        rejected_candidate_count: attribute; Rejected candidate count.
    """

    status: EvaluationStatus
    objective_ps: int
    lower_bound_ps: int
    sensitivity_ps: int
    optimality_gap_ppm: int
    solver_name: str
    solver_version: str
    candidate_count: int
    solved_candidate_count: int
    rejected_candidate_count: int


class SelectedConfiguration:
    """Carry the globally selected outer configuration.

    Attributes:
        configuration_id: attribute; Canonical configuration ID.
        program_id: attribute; Selected program ID.
        tile: attribute; Selected CTA tile.
        implementations: attribute; Per-operation implementation choices.
        warps: attribute; Selected warp configuration.
        pipeline_depth: attribute; Selected ring depth.
        layout_variant_id: attribute; Selected layout variant.
        static_demands: attribute; Complete static CTA demand.
    """

    configuration_id: ConfigurationId
    program_id: ProgramId
    tile: TileCandidate
    implementations: tuple[OpImplementationSelection, ...]
    warps: WarpConfig
    pipeline_depth: int
    layout_variant_id: str
    static_demands: tuple[StaticDemand, ...]


class CostModelPlan:
    """Carry one complete selected cost-model schedule.

    Attributes:
        schema_version: attribute; Result-plan schema version.
        request_id: attribute; Source request ID.
        hardware: attribute; Exact hardware reference.
        profile_snapshot: attribute; Selected profile snapshot.
        workload: attribute; Logical workload.
        program: attribute; Selected typed program.
        selected: attribute; Selected outer configuration.
        end_to_end_ps: attribute; Predicted total CTA latency.
        loop_timings: attribute; Per-loop compact timings.
        placements: attribute; Selected phase timeline.
        buffers: attribute; Selected value allocations.
        utilization: attribute; Full-CTA resource utilization.
        profiles: attribute; Exact phase timing provenance.
        proof: attribute; Global solve proof.
    """

    schema_version: int
    request_id: str
    hardware: HardwareSpecRef
    profile_snapshot: ProfileSnapshotRef
    workload: WorkloadSpec
    program: TileProgram
    selected: SelectedConfiguration
    end_to_end_ps: int
    loop_timings: tuple[LoopTiming, ...]
    placements: tuple[PhasePlacement, ...]
    buffers: tuple[BufferAllocation, ...]
    utilization: tuple[ResourceUtilization, ...]
    profiles: tuple[ProfileProvenance, ...]
    proof: SolveProof

    def to_json(self) -> str:
        """Serialize deterministic plan JSON.

        Returns:
            Strict plan-schema JSON text.
        """
        ...


class CostModelResult:
    """Carry one public evaluation outcome.

    Attributes:
        schema_version: attribute; Result schema version.
        status: attribute; Public outcome status.
        plan: attribute; Complete incumbent plan, or None.
        missing_profiles: attribute; Missing key IDs.
        rejected_candidates: attribute; Candidate diagnostics.
        diagnostics: attribute; Global diagnostics.
    """

    schema_version: int
    status: EvaluationStatus
    plan: CostModelPlan | None
    missing_profiles: tuple[ProfileKeyId, ...]
    rejected_candidates: tuple[RejectedCandidate, ...]
    diagnostics: tuple[Diagnostic, ...]

    def to_json(self) -> str:
        """Serialize deterministic result JSON.

        Returns:
            Strict result-schema JSON text.
        """
        ...
```

- constraints:
  - `CostModelPlan.schema_version` MUST equal `PLAN_SCHEMA_VERSION`.
  - `CostModelResult.schema_version` MUST equal `RESULT_SCHEMA_VERSION`.
  - `OPTIMAL` and `FEASIBLE` require a complete plan. `TIMEOUT` MAY carry an
    incumbent. `INFEASIBLE`, `UNSUPPORTED`, `MISSING_PROFILE`, and
    `PROFILE_FAILED` MUST carry no plan.
  - A result without a plan MUST NOT invent zero latency.
  - `CostModelPlan.program` MUST match the selected program and tile exactly.
    Every selected operation has exactly one implementation.
  - Every placement and profile-provenance entry maps to one operation in the
    selected program through `source_op_id`.
  - Every selected phase has exactly one profile-provenance entry. Issue and
    completion phases MAY share measurement and profile-key IDs while selecting
    different metrics.
  - `BufferAllocation.total_bytes` equals
    `bytes_per_slot * slot_count`, and `value_id` exists in the selected program.
  - Every declared loop has one `LoopTiming`. A finite loop has no II. A
    periodic loop has positive II and satisfies the loop-span equation.
  - `end_to_end_ps` is the global schedule span, not an unconditional sum of loop
    spans.
  - Periodic placements contain compact boundaries and one representative steady
    period. Resource utilization covers the full logical CTA and is expanded
    analytically from loop counts.
  - `busy_slot_ps <= capacity_slots * horizon_ps` for every utilization row.
  - `optimality_gap_ppm` is
    `ceil((objective - lower_bound) * 1_000_000 / objective)` and is zero only
    for a proved optimum.
  - Deterministic ordering is: loop timings by loop ID; placements by
    `(loop_id or "", region, iteration, start_ps, phase_id)`; buffers by buffer
    ID; utilization by resource ID; profiles by phase ID.

## 12. Public Orchestration And Serialization

```python
def evaluate(
    request: CostModelRequest,
    *,
    hardware_catalog: HardwareCatalog,
    implementation_catalog: ImplementationCatalog,
    profile_store: SqliteProfileStore,
    profile_runner: ProfileRunner | None = None,
    measurement_policy: MeasurementPolicy = MeasurementPolicy(),
) -> CostModelResult:
    """Build and solve one request with typed outcome mapping.

    Args:
        request: Typed finite search request.
        hardware_catalog: Exact installed hardware documents.
        implementation_catalog: Installed operation implementations.
        profile_store: Timing snapshot store.
        profile_runner: Optional explicit JIT runner.
        measurement_policy: JIT measurement policy.

    Returns:
        Public outcome with an optional complete plan.
    """
    ...


def program_from_json(text: str) -> TileProgram:
    """Parse strict typed-program JSON.

    Args:
        text: Program JSON text.

    Returns:
        Validated typed program.
    """
    ...


def program_to_json(program: TileProgram) -> str:
    """Serialize strict typed-program JSON.

    Args:
        program: Validated typed program.

    Returns:
        Deterministic program JSON text.
    """
    ...


def request_from_json(text: str) -> CostModelRequest:
    """Parse strict request JSON.

    Args:
        text: Request JSON text.

    Returns:
        Validated request.
    """
    ...


def request_to_json(request: CostModelRequest) -> str:
    """Serialize strict request JSON.

    Args:
        request: Validated request.

    Returns:
        Deterministic request JSON text.
    """
    ...


def problem_from_json(text: str) -> SearchProblem:
    """Parse strict replayable search-problem JSON.

    Args:
        text: Search-problem JSON text.

    Returns:
        Validated numeric search problem.
    """
    ...


def problem_to_json(problem: SearchProblem) -> str:
    """Serialize strict replayable search-problem JSON.

    Args:
        problem: Complete numeric search problem.

    Returns:
        Deterministic search-problem JSON text.
    """
    ...


def plan_to_json(plan: CostModelPlan) -> str:
    """Serialize strict selected-plan JSON.

    Args:
        plan: Complete selected plan.

    Returns:
        Deterministic plan JSON text.
    """
    ...


def result_to_json(result: CostModelResult) -> str:
    """Serialize strict result JSON.

    Args:
        result: Public outcome.

    Returns:
        Deterministic result JSON text.
    """
    ...


def render_timeline(plan: CostModelPlan) -> str:
    """Render one selected phase timeline.

    Args:
        plan: Complete selected plan.

    Returns:
        Deterministic human-readable timeline.
    """
    ...
```

- constraints:
  - `evaluate` is the typed error-mapping convenience path equivalent to
    `solve(build(request, ...))`.
  - Missing profile maps to `MISSING_PROFILE` with key IDs. Unsupported
    capability maps to `UNSUPPORTED`. Profile-run failure maps to
    `PROFILE_FAILED`.
  - No feasible candidate maps to `INFEASIBLE`. A deadline with no incumbent
    maps to `TIMEOUT` with no plan; a deadline with an incumbent maps to
    `TIMEOUT` with a complete plan.
  - Malformed JSON, invalid dimensions or options, profile-store corruption,
    and search-problem invariant failures remain exceptions.
  - Serialization MUST reject unknown versions and fields and MUST be
    deterministic.
  - Floating-point nanoseconds MAY appear only in human presentation. Stored and
    serialized model timings remain integer picoseconds.

The stable command-line surface is:

```text
tilefoundry-costmodel build --request REQUEST.json --profiles PROFILES.db --output PROBLEM.json
tilefoundry-costmodel solve --problem PROBLEM.json --output PLAN.json
tilefoundry-costmodel search --request REQUEST.json --profiles PROFILES.db --output PLAN.json
tilefoundry-costmodel profile --request REQUEST.json --profiles PROFILES.db
tilefoundry-costmodel profiles export --profiles PROFILES.db --snapshot ID@REV --output SNAPSHOT.json
tilefoundry-costmodel profiles import --profiles PROFILES.db SNAPSHOT.json
tilefoundry-costmodel profiles inspect --profiles PROFILES.db --snapshot ID@REV
```

- constraints:
  - `search` is the `solve(build(...))` convenience command and defaults to
    require-only profile lookup.
  - `build` follows the request profile policy. `solve` MUST open neither CUDA
    nor SQLite.
  - `profile` requires JIT-on-miss and a local CUDA-capable B200.
  - Successful build, search, or solve exits zero. Invalid input exits 2;
    missing or unsupported capability exits 3; profile failure exits 4;
    infeasible exits 5; timeout without an incumbent exits 6.

## 13. Package Ownership

The source ownership is:

```text
tilefoundry_costmodel/
  api.py                 public orchestration and serialization
  build.py               configuration composition and profile closure
  errors.py              package exception hierarchy
  language.py            typed T.* construction surface
  model.py               shared IDs, enums, and tensor descriptors
  program.py             typed tile-operation IR
  request.py             request and solver options
  result.py              public result and plan records
  search_space.py        warp and finite search choices
  tileop.py              phase-template records
  hardware/              hardware schema, registry, and B200 facts
  implementations/       operation lowerings paired with benchmarks
  profiles/              profile identity, snapshots, store, and resolver
  profiler/              runner-neutral and local CUDA measurement
  solver/                numeric problem, finite, periodic, and global search
  workloads/             optional typed program frontends
```

- constraints:
  - Importing `tilefoundry_costmodel` MUST NOT require OR-Tools, CUDA Python, a
    CUDA driver, or a GPU.
  - OR-Tools and CUDA dependencies MUST be imported lazily at the operation that
    requires them.
  - `request.py` and `result.py` own immutable records so `api.py`, `build.py`,
    and `solver/` do not form an import cycle.
  - Registries, catalogs, stores, and runners MUST be passed explicitly. Module
    globals MUST NOT affect public behavior.
  - The package root re-exports only version constants, primary request/result
    records, the `T` construction module, B200 catalogs, profile-store/runner
    types, `open_profile_store`, `build`, `solve`, `evaluate`, and
    serialization/rendering functions.
