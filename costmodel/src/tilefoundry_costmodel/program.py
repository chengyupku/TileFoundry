"""Compiler-independent, validated tile-operation programs.

The records in this module are the M1 semantic boundary.  They intentionally
contain no compiler IR, callable, source text, implementation, or timing
state.  JSON decoding and the :mod:`tilefoundry_costmodel.language` helpers
construct these same records.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TypeAlias, TypeVar, cast

from .constants import PROGRAM_SCHEMA_VERSION
from .errors import InvalidRequestError, WorkloadError
from .model import (
    LoopId,
    NamedShape,
    OpId,
    ProgramId,
    TensorDescriptor,
    ValueId,
    WorkloadKind,
    validate_identifier,
)


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
    """Select a first or last operation instance."""

    FIRST = "first"
    LAST = "last"


@dataclass(frozen=True, slots=True)
class AlignedRelation:
    """Pair corresponding instances with a non-negative loop distance."""

    kind: DependencyRelationKind
    iteration_distance: int

    def __post_init__(self) -> None:
        _coerce_enum(self, "kind", DependencyRelationKind)
        if self.kind is not DependencyRelationKind.ALIGNED:
            raise WorkloadError("aligned relation kind must be 'aligned'")
        _non_negative_int(self.iteration_distance, "iteration_distance")


@dataclass(frozen=True, slots=True)
class EndpointRelation:
    """Pair exactly one source endpoint with one destination endpoint."""

    kind: DependencyRelationKind
    src_endpoint: InstanceEndpoint
    dst_endpoint: InstanceEndpoint

    def __post_init__(self) -> None:
        _coerce_enum(self, "kind", DependencyRelationKind)
        _coerce_enum(self, "src_endpoint", InstanceEndpoint)
        _coerce_enum(self, "dst_endpoint", InstanceEndpoint)
        if self.kind is not DependencyRelationKind.ENDPOINT:
            raise WorkloadError("endpoint relation kind must be 'endpoint'")


DependencyRelation: TypeAlias = AlignedRelation | EndpointRelation


@dataclass(frozen=True, slots=True)
class LoopBarrier:
    """Order every instance of one declared loop before another loop."""

    barrier_id: str
    src_loop_id: LoopId
    dst_loop_id: LoopId

    def __post_init__(self) -> None:
        _program_identifier(self.barrier_id, "barrier_id")
        _program_identifier(self.src_loop_id, "src_loop_id")
        _program_identifier(self.dst_loop_id, "dst_loop_id")
        if self.src_loop_id == self.dst_loop_id:
            raise WorkloadError("loop barrier endpoints must be distinct")


@dataclass(frozen=True, slots=True)
class TileValueType:
    """Describe a typed value in one memory space."""

    tensor: TensorDescriptor
    memory_space: MemorySpace

    def __post_init__(self) -> None:
        if type(self.tensor) is not TensorDescriptor:
            raise WorkloadError("value tensor must be TensorDescriptor")
        _coerce_enum(self, "memory_space", MemorySpace)


@dataclass(frozen=True, slots=True)
class TileValue:
    """Name one program value."""

    value_id: ValueId
    value_type: TileValueType

    def __post_init__(self) -> None:
        _program_identifier(self.value_id, "value_id")
        if type(self.value_type) is not TileValueType:
            raise WorkloadError("value_type must be TileValueType")


@dataclass(frozen=True, slots=True)
class TileLoop:
    """Describe one repeated pipeline region."""

    loop_id: LoopId
    iterations: int

    def __post_init__(self) -> None:
        _program_identifier(self.loop_id, "loop_id")
        _positive_int(self.iterations, "loop iterations")


@dataclass(frozen=True, slots=True)
class OpIterationDomain:
    """Place an operation over one loop or one-time region."""

    loop_id: LoopId | None
    first_iteration: int
    iteration_count: int

    def __post_init__(self) -> None:
        _non_negative_int(self.first_iteration, "first_iteration")
        _positive_int(self.iteration_count, "iteration_count")
        if self.loop_id is None:
            if (self.first_iteration, self.iteration_count) != (0, 1):
                raise WorkloadError("a one-time domain must equal (None, 0, 1)")
        else:
            _program_identifier(self.loop_id, "loop_id")

    @property
    def iterations(self) -> tuple[int, ...]:
        """Return the finite iteration labels covered by this domain."""

        return tuple(range(self.first_iteration, self.first_iteration + self.iteration_count))


@dataclass(frozen=True, slots=True)
class CopyOp:
    kind: TileOpKind
    op_id: OpId
    source: ValueId
    destination: ValueId
    domain: OpIterationDomain

    def __post_init__(self) -> None:
        _fixed_kind(self.kind, TileOpKind.COPY)
        _validate_op_id(self.op_id)
        _program_identifier(self.source, "source")
        _program_identifier(self.destination, "destination")
        _require_domain(self.domain)


@dataclass(frozen=True, slots=True)
class GemmOp:
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

    def __post_init__(self) -> None:
        _fixed_kind(self.kind, TileOpKind.GEMM)
        _validate_op_id(self.op_id)
        for name in ("lhs", "rhs", "accumulator", "result"):
            _program_identifier(getattr(self, name), name)
        for name in ("m_axis", "n_axis", "k_axis"):
            _program_identifier(getattr(self, name), name)
        if len({self.m_axis, self.n_axis, self.k_axis}) != 3:
            raise WorkloadError("GEMM axes must be distinct")
        _require_domain(self.domain)


@dataclass(frozen=True, slots=True)
class ReduceOp:
    kind: TileOpKind
    op_id: OpId
    source: ValueId
    result: ValueId
    axes: tuple[str, ...]
    reduction: ReductionKind
    domain: OpIterationDomain

    def __post_init__(self) -> None:
        _fixed_kind(self.kind, TileOpKind.REDUCE)
        _validate_op_id(self.op_id)
        _program_identifier(self.source, "source")
        _program_identifier(self.result, "result")
        if not isinstance(self.axes, (tuple, list)):
            raise WorkloadError("reduction axes must be a sequence")
        axes = tuple(self.axes)
        object.__setattr__(self, "axes", axes)
        if not axes or len(axes) != len(set(axes)):
            raise WorkloadError("reduction axes must be non-empty and unique")
        for axis in axes:
            _program_identifier(axis, "reduction axis")
        _coerce_enum(self, "reduction", ReductionKind)
        _require_domain(self.domain)


@dataclass(frozen=True, slots=True)
class ElementwiseOp:
    kind: TileOpKind
    op_id: OpId
    inputs: tuple[ValueId, ...]
    result: ValueId
    function: ElementwiseKind
    domain: OpIterationDomain

    def __post_init__(self) -> None:
        _fixed_kind(self.kind, TileOpKind.ELEMENTWISE)
        _validate_op_id(self.op_id)
        inputs = _identifier_tuple(self.inputs, "elementwise input")
        object.__setattr__(self, "inputs", inputs)
        if not inputs:
            raise WorkloadError("elementwise operation requires an input")
        _program_identifier(self.result, "result")
        _coerce_enum(self, "function", ElementwiseKind)
        _require_domain(self.domain)


TileOp: TypeAlias = CopyOp | GemmOp | ReduceOp | ElementwiseOp


@dataclass(frozen=True, slots=True)
class TileDependency:
    value_id: ValueId
    src_op_id: OpId
    dst_op_id: OpId
    relation: DependencyRelation

    def __post_init__(self) -> None:
        for value, label in (
            (self.value_id, "value_id"),
            (self.src_op_id, "src_op_id"),
            (self.dst_op_id, "dst_op_id"),
        ):
            _program_identifier(value, label)
        if type(self.relation) not in (AlignedRelation, EndpointRelation):
            raise WorkloadError("dependency relation must be typed")
        if self.src_op_id == self.dst_op_id and isinstance(self.relation, EndpointRelation):
            raise WorkloadError("endpoint self-dependencies are invalid")


@dataclass(frozen=True, slots=True)
class TileCandidate:
    tile_id: str
    shape: NamedShape

    def __post_init__(self) -> None:
        _program_identifier(self.tile_id, "tile_id")
        if type(self.shape) is not NamedShape:
            raise WorkloadError("tile shape must be NamedShape")


@dataclass(frozen=True, slots=True)
class ExpandedDependency:
    """One finite operation-instance edge exposed for validation/debugging."""

    value_id: ValueId
    src_op_id: OpId
    src_iteration: int
    dst_op_id: OpId
    dst_iteration: int


@dataclass(frozen=True, slots=True)
class TileProgram:
    """Describe one concrete one-CTA typed tile variant."""

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

    def __post_init__(self) -> None:
        if (
            isinstance(self.schema_version, bool)
            or not isinstance(self.schema_version, int)
            or self.schema_version != PROGRAM_SCHEMA_VERSION
        ):
            raise WorkloadError(f"unsupported program schema version: {self.schema_version!r}")
        _program_identifier(self.program_id, "program_id")
        _coerce_enum(self, "workload_kind", WorkloadKind)
        if type(self.tile) is not TileCandidate:
            raise WorkloadError("program tile must be TileCandidate")

        values = _typed_tuple(self.values, TileValue, "values")
        loops = _typed_tuple(self.loops, TileLoop, "loops")
        operations = _typed_operation_tuple(self.operations)
        dependencies = _typed_tuple(self.dependencies, TileDependency, "dependencies")
        barriers = _typed_tuple(self.loop_barriers, LoopBarrier, "loop_barriers")
        inputs = _identifier_tuple(self.inputs, "program input")
        outputs = _identifier_tuple(self.outputs, "program output")
        values = tuple(sorted(values, key=lambda item: str(item.value_id)))
        loops = tuple(sorted(loops, key=lambda item: str(item.loop_id)))
        operations = tuple(sorted(operations, key=lambda item: str(item.op_id)))
        dependencies = tuple(sorted(dependencies, key=_dependency_sort_key))
        barriers = tuple(sorted(barriers, key=lambda item: str(item.barrier_id)))
        inputs = tuple(sorted(inputs, key=str))
        outputs = tuple(sorted(outputs, key=str))
        for name, collection in (
            ("values", values),
            ("loops", loops),
            ("operations", operations),
            ("loop_barriers", barriers),
        ):
            _unique_ids(collection, name)
        if len(inputs) != len(set(inputs)) or len(outputs) != len(set(outputs)):
            raise WorkloadError("program inputs and outputs must be unique")
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "loops", loops)
        object.__setattr__(self, "operations", operations)
        object.__setattr__(self, "dependencies", dependencies)
        object.__setattr__(self, "loop_barriers", barriers)
        object.__setattr__(self, "inputs", inputs)
        object.__setattr__(self, "outputs", outputs)

        value_map = {value.value_id: value for value in values}
        loop_map = {loop.loop_id: loop for loop in loops}
        op_map = {op.op_id: op for op in operations}
        for value_id in (*inputs, *outputs):
            if value_id not in value_map:
                raise WorkloadError(f"program references unknown value: {value_id!r}")
        for op in operations:
            if op.domain.loop_id is not None:
                loop = loop_map.get(op.domain.loop_id)
                if loop is None:
                    raise WorkloadError(f"operation references unknown loop: {op.domain.loop_id!r}")
                if op.domain.first_iteration + op.domain.iteration_count > loop.iterations:
                    raise WorkloadError("operation domain exceeds loop iteration range")
        for barrier in barriers:
            if barrier.src_loop_id not in loop_map or barrier.dst_loop_id not in loop_map:
                raise WorkloadError("loop barrier references an unknown loop")
        _validate_barrier_graph(barriers)

        consumed: dict[OpId, tuple[ValueId, ...]] = {
            op.op_id: _consumed_values(op) for op in operations
        }
        produced_by: dict[ValueId, OpId] = {}
        for op in operations:
            produced = _produced_value(op)
            if produced is not None:
                if produced not in value_map:
                    raise WorkloadError(f"operation produces unknown value: {produced!r}")
                if produced in produced_by:
                    raise WorkloadError(f"value has multiple producers: {produced!r}")
                produced_by[produced] = op.op_id
        input_set = set(inputs)
        if input_set.intersection(produced_by):
            raise WorkloadError("external input cannot also have a producer")
        for op in operations:
            for value_id in consumed[op.op_id]:
                if value_id not in value_map:
                    raise WorkloadError(f"operation consumes unknown value: {value_id!r}")

        dependency_keys: set[tuple[ValueId, OpId, OpId]] = set()
        relation_edges: set[tuple[OpId, OpId]] = set()
        expanded: list[ExpandedDependency] = []
        for dependency in dependencies:
            if dependency.value_id not in value_map:
                raise WorkloadError("dependency references an unknown value")
            src = op_map.get(dependency.src_op_id)
            dst = op_map.get(dependency.dst_op_id)
            if src is None or dst is None:
                raise WorkloadError("dependency references an unknown operation")
            if produced_by.get(dependency.value_id) != dependency.src_op_id:
                raise WorkloadError("dependency source is not the value producer")
            if dependency.value_id not in consumed[dependency.dst_op_id]:
                raise WorkloadError("dependency destination does not consume the value")
            key = (dependency.value_id, dependency.src_op_id, dependency.dst_op_id)
            if key in dependency_keys:
                raise WorkloadError("duplicate value dependency")
            dependency_keys.add(key)
            edges = _expand_relation(dependency, src, dst, loop_map)
            if not edges:
                raise WorkloadError("dependency relation has no valid instances")
            if dependency.src_op_id != dependency.dst_op_id:
                relation_edges.add((dependency.src_op_id, dependency.dst_op_id))
            expanded.extend(edges)

        for op_id, values_used in consumed.items():
            for value_id in set(values_used):
                producer = produced_by.get(value_id)
                if producer is None:
                    if value_id not in input_set:
                        raise WorkloadError(
                            f"non-input consumed value lacks a producer: {value_id!r}"
                        )
                elif (value_id, producer, op_id) not in dependency_keys:
                    raise WorkloadError(
                        f"missing explicit dependency for value {value_id!r} to {op_id!r}"
                    )

        _validate_operation_shapes(values, operations, value_map)
        _validate_relation_graph(relation_edges)
        _validate_expanded_graph(expanded)
        _validate_tile_axes(self.workload_kind, self.tile, operations)
        # Expansion is deterministic and cheap for the finite M1 domains; it is
        # intentionally derived on demand so it never becomes part of the JSON
        # ownership surface.

    @property
    def expanded_dependencies(self) -> tuple[ExpandedDependency, ...]:
        """Return canonical finite dependency edges."""

        loop_map = {loop.loop_id: loop for loop in self.loops}
        op_map = {op.op_id: op for op in self.operations}
        expanded: list[ExpandedDependency] = []
        for dependency in self.dependencies:
            src = op_map[dependency.src_op_id]
            dst = op_map[dependency.dst_op_id]
            expanded.extend(_expand_relation(dependency, src, dst, loop_map))
        return tuple(expanded)

    def expanded_dependency_edges(self) -> tuple[ExpandedDependency, ...]:
        """Method alias for callers that prefer an explicit expansion call."""

        return self.expanded_dependencies

    def expand_dependencies(self) -> tuple[ExpandedDependency, ...]:
        """Expand all finite relations into exact operation-instance edges."""

        return self.expanded_dependencies

    def expanded_barrier_edges(self) -> tuple[tuple[LoopId, int, LoopId, int], ...]:
        """Return the all-instance control edges represented by loop barriers."""

        loops = {loop.loop_id: loop for loop in self.loops}
        edges: list[tuple[LoopId, int, LoopId, int]] = []
        for barrier in self.loop_barriers:
            src = loops[barrier.src_loop_id]
            dst = loops[barrier.dst_loop_id]
            for src_iteration in range(src.iterations):
                for dst_iteration in range(dst.iterations):
                    edges.append((src.loop_id, src_iteration, dst.loop_id, dst_iteration))
        return tuple(edges)

    def to_json(self) -> str:
        from ._serialization import program_to_json

        return program_to_json(self)

    @classmethod
    def from_json(cls, text: str) -> "TileProgram":
        from ._serialization import program_from_json

        return program_from_json(text)


_ProgramRecordT = TypeVar("_ProgramRecordT")


def _typed_records(
    value: object, item_type: type[_ProgramRecordT], label: str
) -> tuple[_ProgramRecordT, ...]:
    if not isinstance(value, (tuple, list)):
        raise WorkloadError(f"program {label} must be a sequence")
    items: tuple[object, ...] = tuple(value)
    if not all(type(item) is item_type for item in items):
        raise WorkloadError(f"program {label} must contain typed records")
    return cast(tuple[_ProgramRecordT, ...], items)


def _typed_tuple(
    value: object, item_type: type[_ProgramRecordT], label: str
) -> tuple[_ProgramRecordT, ...]:
    """Copy a sequence while requiring exact typed children."""

    return _typed_records(value, item_type, label)


def _typed_operation_tuple(value: object) -> tuple[TileOp, ...]:
    if not isinstance(value, (tuple, list)):
        raise WorkloadError("program operations must be a sequence")
    items: tuple[object, ...] = tuple(value)
    if not all(type(item) in (CopyOp, GemmOp, ReduceOp, ElementwiseOp) for item in items):
        raise WorkloadError("program operations must contain typed operation records")
    return cast(tuple[TileOp, ...], items)


def _identifier_tuple(value: object, label: str) -> tuple[ValueId, ...]:
    if not isinstance(value, (tuple, list)):
        raise WorkloadError(f"{label}s must be a sequence")
    items: tuple[object, ...] = tuple(value)
    for item in items:
        _program_identifier(item, label)
    return cast(tuple[ValueId, ...], items)


def _unique_ids(values: tuple[object, ...], label: str) -> None:
    field = {
        "values": "value_id",
        "loops": "loop_id",
        "operations": "op_id",
        "loop_barriers": "barrier_id",
    }[label]
    ids = tuple(str(getattr(item, field)) for item in values)
    if len(ids) != len(set(ids)):
        raise WorkloadError(f"program {label[:-1]} IDs must be unique")


def _dependency_sort_key(dependency: TileDependency) -> tuple[str, str, str, str]:
    relation = dependency.relation
    if isinstance(relation, AlignedRelation):
        relation_key = f"aligned:{relation.iteration_distance}"
    else:
        relation_key = f"endpoint:{relation.src_endpoint.value}:{relation.dst_endpoint.value}"
    return (
        str(dependency.value_id),
        str(dependency.src_op_id),
        str(dependency.dst_op_id),
        relation_key,
    )


def _coerce_enum(instance: object, field_name: str, enum_type: type[Enum]) -> None:
    value = getattr(instance, field_name)
    if isinstance(value, enum_type):
        return
    try:
        object.__setattr__(instance, field_name, enum_type(value))
    except (TypeError, ValueError) as exc:
        raise WorkloadError(f"invalid {field_name}") from exc


def _fixed_kind(value: object, expected: TileOpKind) -> None:
    if not isinstance(value, TileOpKind):
        try:
            value = TileOpKind(value)
        except (TypeError, ValueError) as exc:
            raise WorkloadError("unknown tile operation kind") from exc
    if value is not expected:
        raise WorkloadError(f"operation kind must be {expected.value!r}")


def _validate_op_id(value: object) -> None:
    _program_identifier(value, "op_id")


def _program_identifier(value: object, label: str) -> None:
    if not isinstance(value, str):
        raise WorkloadError(f"{label} must be a non-empty ASCII string")
    try:
        validate_identifier(value, label=label)
    except InvalidRequestError as exc:
        raise WorkloadError(str(exc)) from exc


def _positive_int(value: object, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise WorkloadError(f"{label} must be a positive integer")


def _non_negative_int(value: object, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise WorkloadError(f"{label} must be a non-negative integer")


def _require_domain(value: object) -> None:
    if type(value) is not OpIterationDomain:
        raise WorkloadError("operation domain must be OpIterationDomain")


def _consumed_values(op: TileOp) -> tuple[ValueId, ...]:
    if isinstance(op, CopyOp):
        return (op.source,)
    if isinstance(op, GemmOp):
        return (op.lhs, op.rhs, op.accumulator)
    if isinstance(op, ReduceOp):
        return (op.source,)
    return op.inputs


def _produced_value(op: TileOp) -> ValueId | None:
    if isinstance(op, CopyOp):
        return op.destination
    return op.result


def _domain_instances(op: TileOp, loops: dict[LoopId, TileLoop]) -> tuple[int, ...]:
    domain = op.domain
    if domain.loop_id is None:
        return (0,)
    if domain.loop_id not in loops:
        raise WorkloadError(f"operation references unknown loop: {domain.loop_id!r}")
    loop = loops[domain.loop_id]
    if domain.first_iteration + domain.iteration_count > loop.iterations:
        raise WorkloadError("operation domain exceeds loop iteration range")
    return domain.iterations


def _expand_relation(
    dependency: TileDependency,
    src: TileOp,
    dst: TileOp,
    loops: dict[LoopId, TileLoop],
) -> tuple[ExpandedDependency, ...]:
    src_instances = _domain_instances(src, loops)
    dst_instances = _domain_instances(dst, loops)
    relation = dependency.relation
    if isinstance(relation, AlignedRelation):
        if src.domain.loop_id != dst.domain.loop_id or src.domain.loop_id is None:
            raise WorkloadError("aligned relation requires operations in the same loop")
        dst_set = set(dst_instances)
        result = tuple(
            ExpandedDependency(
                dependency.value_id, src.op_id, i, dst.op_id, i + relation.iteration_distance
            )
            for i in src_instances
            if i + relation.iteration_distance in dst_set
        )
        if src.op_id == dst.op_id and relation.iteration_distance == 0:
            raise WorkloadError("zero-distance aligned self-dependency is invalid")
        return result
    src_i = (
        src_instances[0] if relation.src_endpoint is InstanceEndpoint.FIRST else src_instances[-1]
    )
    dst_i = (
        dst_instances[0] if relation.dst_endpoint is InstanceEndpoint.FIRST else dst_instances[-1]
    )
    return (ExpandedDependency(dependency.value_id, src.op_id, src_i, dst.op_id, dst_i),)


def _validate_expanded_graph(edges: list[ExpandedDependency]) -> None:
    # Positive-distance recurrence edges point to later finite instances and are
    # therefore acyclic unless another relation closes a real cycle.
    adjacency: dict[tuple[str, int], list[tuple[str, int]]] = {}
    for edge in edges:
        src = (str(edge.src_op_id), edge.src_iteration)
        dst = (str(edge.dst_op_id), edge.dst_iteration)
        adjacency.setdefault(src, []).append(dst)
    state: dict[tuple[str, int], int] = {}

    def visit(node: tuple[str, int]) -> None:
        state[node] = 1
        for child in adjacency.get(node, ()):
            marker = state.get(child, 0)
            if marker == 1:
                raise WorkloadError("expanded operation relation graph must be acyclic")
            if marker == 0:
                visit(child)
        state[node] = 2

    for node in tuple(adjacency):
        if state.get(node, 0) == 0:
            visit(node)


def _validate_relation_graph(edges: set[tuple[OpId, OpId]]) -> None:
    """Reject cycles in the operation-level relation graph.

    Positive-distance self recurrence is valid and is intentionally omitted
    from this graph; cross-operation relations must remain acyclic even when
    finite endpoint expansion happens to produce no instance-level cycle.
    """

    adjacency: dict[OpId, list[OpId]] = {}
    for src, dst in edges:
        if src != dst:
            adjacency.setdefault(src, []).append(dst)
    state: dict[OpId, int] = {}

    def visit(node: OpId) -> None:
        state[node] = 1
        for child in adjacency.get(node, ()):
            marker = state.get(child, 0)
            if marker == 1:
                raise WorkloadError("operation relation graph must be acyclic")
            if marker == 0:
                visit(child)
        state[node] = 2

    for node in tuple(adjacency):
        if state.get(node, 0) == 0:
            visit(node)


def _validate_barrier_graph(barriers: tuple[LoopBarrier, ...]) -> None:
    adjacency: dict[str, list[str]] = {}
    for barrier in barriers:
        adjacency.setdefault(str(barrier.src_loop_id), []).append(str(barrier.dst_loop_id))
    state: dict[str, int] = {}

    def visit(node: str) -> None:
        state[node] = 1
        for child in adjacency.get(node, ()):
            marker = state.get(child, 0)
            if marker == 1:
                raise WorkloadError("loop barrier graph must be acyclic")
            if marker == 0:
                visit(child)
        state[node] = 2

    for node in tuple(adjacency):
        if state.get(node, 0) == 0:
            visit(node)


def _value_map(values: tuple[TileValue, ...]) -> dict[ValueId, TileValue]:
    return {value.value_id: value for value in values}


def _shape_axes(value: TileValue, label: str) -> dict[str, int]:
    axes = value.value_type.tensor.shape.axes
    result = {axis.name: axis.extent for axis in axes}
    if len(result) != len(axes):
        raise WorkloadError(f"{label} shape axis names must be unique")
    return result


def _element_count(value: TileValue) -> int:
    count = 1
    for axis in value.value_type.tensor.shape.axes:
        count *= axis.extent
    return count


def _validate_operation_shapes(
    values: tuple[TileValue, ...],
    operations: tuple[TileOp, ...],
    value_map: dict[ValueId, TileValue],
) -> None:
    del values
    for op in operations:
        if isinstance(op, CopyOp):
            source = value_map[op.source]
            destination = value_map[op.destination]
            if source.value_type.tensor.dtype is not destination.value_type.tensor.dtype:
                raise WorkloadError("copy source and destination dtypes must match")
            if _element_count(source) != _element_count(destination):
                raise WorkloadError("copy source and destination element counts must match")
        elif isinstance(op, GemmOp):
            lhs = _shape_axes(value_map[op.lhs], "GEMM lhs")
            rhs = _shape_axes(value_map[op.rhs], "GEMM rhs")
            acc = _shape_axes(value_map[op.accumulator], "GEMM accumulator")
            result = _shape_axes(value_map[op.result], "GEMM result")
            _require_axes(lhs, (op.m_axis, op.k_axis), "GEMM lhs")
            _require_axes(rhs, (op.k_axis, op.n_axis), "GEMM rhs")
            _require_axes(acc, (op.m_axis, op.n_axis), "GEMM accumulator")
            _require_axes(result, (op.m_axis, op.n_axis), "GEMM result")
            if lhs[op.m_axis] != acc[op.m_axis] or lhs[op.m_axis] != result[op.m_axis]:
                raise WorkloadError("GEMM m axis extents must agree")
            if rhs[op.n_axis] != acc[op.n_axis] or rhs[op.n_axis] != result[op.n_axis]:
                raise WorkloadError("GEMM n axis extents must agree")
            if lhs[op.k_axis] != rhs[op.k_axis]:
                raise WorkloadError("GEMM k axis extents must agree")
        elif isinstance(op, ReduceOp):
            reduce_source = value_map[op.source]
            source_axes = _shape_axes(reduce_source, "reduction source")
            if any(axis not in source_axes for axis in op.axes):
                raise WorkloadError("reduction axis is absent from source shape")
        else:
            result_value = value_map[op.result]
            result_axes = _shape_axes(result_value, "elementwise result")
            input_values = [value_map[value_id] for value_id in op.inputs]
            input_axes = [_shape_axes(value, "elementwise input") for value in input_values]
            union = {axis for axes in input_axes for axis in axes}
            if set(result_axes) != union:
                raise WorkloadError("elementwise result axes must equal input axis union")
            for axis, result_extent in result_axes.items():
                input_extents = tuple(axes.get(axis, 1) for axes in input_axes)
                if result_extent != max(input_extents):
                    raise WorkloadError(
                        "elementwise broadcasting must exactly produce result extent"
                    )
                for axes in input_axes:
                    extent = axes.get(axis, 1)
                    if extent not in (1, result_extent):
                        raise WorkloadError("elementwise broadcasting is incompatible")
            dtypes = {value.value_type.tensor.dtype for value in (*input_values, result_value)}
            if len(dtypes) != 1:
                raise WorkloadError("elementwise input and result dtypes must match")


def _require_axes(axes: dict[str, int], expected: tuple[str, ...], label: str) -> None:
    if any(axis not in axes for axis in expected):
        raise WorkloadError(f"{label} must contain the named GEMM axes")


def _validate_tile_axes(
    workload_kind: WorkloadKind, tile: TileCandidate, operations: tuple[TileOp, ...]
) -> None:
    # Empty M0 records remain valid for backwards-compatible serialization;
    # executable typed programs must carry the exact frontend axis contract.
    if not operations:
        return
    required = {
        WorkloadKind.GEMM: {"m", "n", "k"},
        WorkloadKind.GQA_DECODE: {"q_heads", "kv_tokens", "head_dim"},
        WorkloadKind.FLASH_ATTENTION: {"q_tokens", "kv_tokens", "head_dim"},
        WorkloadKind.MLP: {"up_m", "up_n", "up_k", "down_m", "down_n", "down_k"},
    }[workload_kind]
    names = {axis.name for axis in tile.shape.axes}
    if names != required:
        raise WorkloadError(
            f"tile shape for {workload_kind.value} must contain exactly {sorted(required)}"
        )


__all__ = [
    "AlignedRelation",
    "CopyOp",
    "DependencyRelation",
    "DependencyRelationKind",
    "ElementwiseKind",
    "ElementwiseOp",
    "EndpointRelation",
    "GemmOp",
    "InstanceEndpoint",
    "LoopBarrier",
    "MemorySpace",
    "OpIterationDomain",
    "ReduceOp",
    "ReductionKind",
    "TileCandidate",
    "TileDependency",
    "TileLoop",
    "TileOp",
    "TileOpKind",
    "TileProgram",
    "TileValue",
    "TileValueType",
]
