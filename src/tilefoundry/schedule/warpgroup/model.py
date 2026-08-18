"""Strict immutable records for warpgroup programs and schedules."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Protocol, TypeVar

from ._identifiers import (
    non_negative_int,
    positive_int,
    validate_id,
    validate_ssa_id,
)
from .errors import WarpgroupValidationError
from .expression import (
    CastExpression,
    ConcatExpression,
    CopyExpression,
    ElementwiseExpression,
    ExpressionValue,
    IndexExpression,
    LoopIndexRef,
    MatmulExpression,
    NegativeInfinity,
    ReduceExpression,
    ScalarLiteral,
    SelectExpression,
    TransposeExpression,
    ValueRef,
    fold_expression,
    value_references,
)

PROGRAM_FORMAT = "tilefoundry.warpgroup_program"
SCHEDULE_FORMAT = "tilefoundry.warpgroup_schedule"


class MemorySpace(str, Enum):
    """The storage semantics visible to warpgroup scheduling."""

    REGISTER = "register"
    SHARED = "shared"
    GLOBAL = "global"


class DType(str, Enum):
    """Scalar element types accepted by the warpgroup expression grammar."""

    I1 = "i1"
    I8 = "i8"
    I16 = "i16"
    I32 = "i32"
    I64 = "i64"
    U8 = "u8"
    U16 = "u16"
    U32 = "u32"
    U64 = "u64"
    FP8_E4M3FN = "fp8_e4m3fn"
    FP8_E5M2 = "fp8_e5m2"
    FP16 = "fp16"
    BF16 = "bf16"
    FP32 = "fp32"
    FP64 = "fp64"


_FLOAT_DTYPES = {
    DType.FP8_E4M3FN,
    DType.FP8_E5M2,
    DType.FP16,
    DType.BF16,
    DType.FP32,
    DType.FP64,
}

_INTEGER_DTYPES = set(DType) - _FLOAT_DTYPES - {DType.I1}


_T = TypeVar("_T")


class _HasId(Protocol):
    @property
    def id(self) -> str: ...


def _typed_tuple(value: object, expected: type[_T], label: str) -> tuple[_T, ...]:
    if not isinstance(value, (tuple, list)):
        raise WarpgroupValidationError(f"{label} must be a sequence of typed records")
    result = tuple(value)
    if not all(type(item) is expected for item in result):
        raise WarpgroupValidationError(
            f"{label} must contain only exact {expected.__name__} records"
        )
    return result


def _unique_ids(values: Sequence[_HasId], label: str) -> None:
    ids = tuple(item.id for item in values)
    if len(ids) != len(set(ids)):
        repeated = sorted({item for item in ids if ids.count(item) > 1})
        raise WarpgroupValidationError(f"duplicate {label} ID(s): {repeated!r}")


@dataclass(frozen=True, slots=True)
class TensorType:
    """One named tensor shape, element type, and scheduling storage space."""

    id: str
    shape: tuple[int, ...]
    dtype: DType
    space: MemorySpace

    def __post_init__(self) -> None:
        validate_id(self.id, "type ID")
        if not isinstance(self.shape, (tuple, list)):
            raise WarpgroupValidationError("type shape must be a sequence")
        shape = tuple(self.shape)
        if not shape:
            raise WarpgroupValidationError(f"type {self.id!r} shape must not be empty")
        for extent in shape:
            positive_int(extent, f"type {self.id!r} shape extent")
        if type(self.dtype) is not DType:
            raise WarpgroupValidationError(f"type {self.id!r} dtype must be DType")
        if type(self.space) is not MemorySpace:
            raise WarpgroupValidationError(f"type {self.id!r} space must be MemorySpace")
        object.__setattr__(self, "shape", shape)


@dataclass(frozen=True, slots=True)
class ProgramInput:
    """One external SSA value available at the loop boundary."""

    id: str
    type_id: str

    def __post_init__(self) -> None:
        validate_ssa_id(self.id, "input ID")
        validate_id(self.type_id, f"input {self.id!r} type ID")


@dataclass(frozen=True, slots=True)
class LoopIterArg:
    """One loop phi relation from init through the body yield."""

    id: str
    init: ValueRef | ScalarLiteral
    yield_value: ValueRef

    def __post_init__(self) -> None:
        validate_ssa_id(self.id, "iter_arg ID")
        if type(self.init) not in (ValueRef, ScalarLiteral):
            raise WarpgroupValidationError(
                f"iter_arg {self.id!r} init must be ValueRef or ScalarLiteral"
            )
        if type(self.yield_value) is not ValueRef:
            raise WarpgroupValidationError(f"iter_arg {self.id!r} yield must be a ValueRef")


@dataclass(frozen=True, slots=True)
class OperationOutput:
    """One typed SSA definition in an atomic loop-body operation."""

    id: str
    type_id: str
    expression: ExpressionValue

    def __post_init__(self) -> None:
        validate_ssa_id(self.id, "operation output ID")
        validate_id(self.type_id, f"output {self.id!r} type ID")
        if not isinstance(
            self.expression,
            (
                ValueRef,
                ScalarLiteral,
                IndexExpression,
                CopyExpression,
                CastExpression,
                MatmulExpression,
                TransposeExpression,
                ConcatExpression,
                SelectExpression,
                ReduceExpression,
                ElementwiseExpression,
            ),
        ):
            raise WarpgroupValidationError(
                f"output {self.id!r} expression is not a typed expression"
            )


def validate_warp_group(value: object) -> None:
    """A lane index, or None for an operation whose lane is not yet decided."""
    if value is None:
        return
    if type(value) is not int:
        raise WarpgroupValidationError("operation warp_group must be an integer or absent")
    if value < 0:
        raise WarpgroupValidationError("operation warp_group must be non-negative")


@dataclass(frozen=True, slots=True)
class RegionOperation:
    """One operation of a region that runs once rather than once per trip.

    A region is the prologue or the epilogue: the transfers, initialisations,
    reductions and write-backs that surround the loop. Two things separate such
    an operation from a body operation, and both follow from it running once.

    **It carries no cost.** ``issue_duration``, ``completion_latency`` and
    ``resource_windows`` price an operation against a steady state, and a region
    has none to be priced against: a 500-cycle transfer before the first trip
    does not lengthen the period, it lengthens the prefix in front of it. So a
    region operation never becomes a ``ProblemOperation``, never enters the
    periodic resource model, and has no row in ``times``. The makespan a solve
    reports is therefore the loop's, and it is short of the kernel's by however
    long the two regions take; pricing them needs a model of a region, which
    this is not.

    **It carries a lane, and the lane is required.** A body operation may leave
    ``warp_group`` absent because there the assignment is what the search
    decides. Nothing searches a region, and the emitter still has to put the
    statement in one warp group's arm of the dispatch, so a declared owner is
    the only answer that is not a placeholder. It is also the load-bearing half:
    a transfer on one lane and the wait for it on another is exactly the
    cross-lane edge a generator needs in order to derive the handshake rather
    than be handed it.

    Order is meaning here. A body operation's place is the schedule's to choose,
    so ``ProgramOperation`` may be canonicalised by ID; a region has nothing but
    the order it was written in, so neither its operations nor their outputs are
    sorted.
    """

    id: str
    warp_group: int
    outputs: tuple[OperationOutput, ...]

    def __post_init__(self) -> None:
        validate_id(self.id, "region operation ID")
        if type(self.warp_group) is not int or self.warp_group < 0:
            raise WarpgroupValidationError(
                f"region operation {self.id!r} warp_group must be a non-negative integer"
            )
        outputs = _typed_tuple(
            self.outputs, OperationOutput, f"region operation {self.id!r} outputs"
        )
        if not outputs:
            raise WarpgroupValidationError(f"region operation {self.id!r} must define an output")
        _unique_ids(outputs, "region operation output")
        object.__setattr__(self, "outputs", outputs)


@dataclass(frozen=True, slots=True)
class ProgramOperation:
    """One user-authored atomic operation without numeric scheduling cost."""

    id: str
    outputs: tuple[OperationOutput, ...]
    #: Which lane owns this operation, or None to leave it to the scheduler.
    warp_group: int | None = None

    def __post_init__(self) -> None:
        validate_id(self.id, "operation ID")
        outputs = _typed_tuple(self.outputs, OperationOutput, f"operation {self.id!r} outputs")
        if not outputs:
            raise WarpgroupValidationError(f"operation {self.id!r} must define an output")
        _unique_ids(outputs, "operation output")
        validate_warp_group(self.warp_group)
        object.__setattr__(self, "outputs", tuple(sorted(outputs, key=lambda item: item.id)))


@dataclass(frozen=True, slots=True)
class ResourceCapacity:
    """One positive integer temporal-resource capacity."""

    id: str
    capacity: int

    def __post_init__(self) -> None:
        validate_id(self.id, "resource ID")
        positive_int(self.capacity, f"resource {self.id!r} capacity")


@dataclass(frozen=True, slots=True, order=True)
class ResourceWindow:
    """One temporal resource window beginning at operation start."""

    resource_id: str
    amount: int
    duration: int

    def __post_init__(self) -> None:
        validate_id(self.resource_id, "resource window ID")
        positive_int(self.amount, f"resource window {self.resource_id!r} amount")
        positive_int(self.duration, f"resource window {self.resource_id!r} duration")


@dataclass(frozen=True, slots=True)
class ProblemOperation:
    """One semantic operation closed to issue/completion timing and windows."""

    id: str
    outputs: tuple[OperationOutput, ...]
    #: Which lane owns this operation, or None to leave it to the scheduler.
    warp_group: int | None
    issue_duration: int
    completion_latency: int
    resource_windows: tuple[ResourceWindow, ...]

    def __post_init__(self) -> None:
        validate_id(self.id, "operation ID")
        outputs = _typed_tuple(self.outputs, OperationOutput, f"operation {self.id!r} outputs")
        if not outputs:
            raise WarpgroupValidationError(f"operation {self.id!r} must define an output")
        _unique_ids(outputs, "operation output")
        validate_warp_group(self.warp_group)
        positive_int(self.issue_duration, f"operation {self.id!r} issue duration")
        positive_int(
            self.completion_latency,
            f"operation {self.id!r} completion latency",
        )
        if self.completion_latency < self.issue_duration:
            raise WarpgroupValidationError(
                f"operation {self.id!r} completion latency must be >= issue duration"
            )
        windows = _typed_tuple(
            self.resource_windows,
            ResourceWindow,
            f"operation {self.id!r} resource windows",
        )
        for window in windows:
            if window.duration > self.completion_latency:
                raise WarpgroupValidationError(
                    f"operation {self.id!r} resource window exceeds completion latency"
                )
        object.__setattr__(self, "outputs", tuple(sorted(outputs, key=lambda item: item.id)))
        object.__setattr__(self, "resource_windows", tuple(sorted(windows)))


@dataclass(frozen=True, slots=True)
class ProgramLoop:
    """One explicit finite loop of user-authored semantic operations."""

    index: str
    iterations: int
    iter_args: tuple[LoopIterArg, ...]
    ops: tuple[ProgramOperation, ...]

    def __post_init__(self) -> None:
        validate_ssa_id(self.index, "loop index")
        positive_int(self.iterations, "loop iterations")
        iter_args = _typed_tuple(self.iter_args, LoopIterArg, "loop iter_args")
        ops = _typed_tuple(self.ops, ProgramOperation, "loop ops")
        if not ops:
            raise WarpgroupValidationError("loop must contain at least one operation")
        _unique_ids(iter_args, "iter_arg")
        _unique_ids(ops, "operation")
        object.__setattr__(self, "iter_args", tuple(sorted(iter_args, key=lambda item: item.id)))
        object.__setattr__(self, "ops", tuple(sorted(ops, key=lambda item: item.id)))


@dataclass(frozen=True, slots=True)
class ProblemLoop:
    """One explicit finite loop whose operations carry closed numeric costs."""

    index: str
    iterations: int
    iter_args: tuple[LoopIterArg, ...]
    ops: tuple[ProblemOperation, ...]

    def __post_init__(self) -> None:
        validate_ssa_id(self.index, "loop index")
        positive_int(self.iterations, "loop iterations")
        iter_args = _typed_tuple(self.iter_args, LoopIterArg, "loop iter_args")
        ops = _typed_tuple(self.ops, ProblemOperation, "loop ops")
        if not ops:
            raise WarpgroupValidationError("loop must contain at least one operation")
        _unique_ids(iter_args, "iter_arg")
        _unique_ids(ops, "operation")
        object.__setattr__(self, "iter_args", tuple(sorted(iter_args, key=lambda item: item.id)))
        object.__setattr__(self, "ops", tuple(sorted(ops, key=lambda item: item.id)))


@dataclass(frozen=True, slots=True, order=True)
class DefUseDependency:
    """One operation dependency derived from SSA, never serialized as input."""

    after: str
    before: str
    distance: int


@dataclass(frozen=True, slots=True)
class WarpgroupProgram:
    """User-authored typed SSA work without any numeric scheduling cost.

    ``prologue`` and ``epilogue`` are the operations that run once, before the
    first trip and after the last: the transfer that makes an input resident,
    the reduction and write-back that turn the loop's last carried value into
    the kernel's result. Both are optional and both default to empty, so a
    program that describes a loop body and nothing else is unchanged. It still
    parses, closes against the same costs and solves to the same schedule,
    because neither region reaches the solver at all.
    """

    format: str
    warp_groups: int
    types: tuple[TensorType, ...]
    inputs: tuple[ProgramInput, ...]
    loop: ProgramLoop
    prologue: tuple[RegionOperation, ...] = ()
    epilogue: tuple[RegionOperation, ...] = ()

    def __post_init__(self) -> None:
        if self.format != PROGRAM_FORMAT:
            raise WarpgroupValidationError(f"program format must be {PROGRAM_FORMAT!r}")
        positive_int(self.warp_groups, "warp_groups")
        types = _typed_tuple(self.types, TensorType, "program types")
        inputs = _typed_tuple(self.inputs, ProgramInput, "program inputs")
        if type(self.loop) is not ProgramLoop:
            raise WarpgroupValidationError("program loop must be ProgramLoop")
        prologue = _typed_tuple(self.prologue, RegionOperation, "program prologue")
        epilogue = _typed_tuple(self.epilogue, RegionOperation, "program epilogue")
        _unique_ids(types, "type")
        _unique_ids(inputs, "input")
        object.__setattr__(self, "types", tuple(sorted(types, key=lambda item: item.id)))
        object.__setattr__(self, "inputs", tuple(sorted(inputs, key=lambda item: item.id)))
        object.__setattr__(self, "prologue", prologue)
        object.__setattr__(self, "epilogue", epilogue)
        _validate_ownership(
            self.loop.ops,
            self.warp_groups,
            label="program",
        )
        _validate_region_ownership(prologue + epilogue, self.warp_groups)
        _validate_semantics(self.types, self.inputs, self.loop, prologue, epilogue)

    def dependencies(self) -> tuple[DefUseDependency, ...]:
        """Derive the canonical operation graph from SSA definitions and uses."""
        return _derive_dependencies(self.inputs, self.loop, self.prologue)


@dataclass(frozen=True, slots=True)
class WarpgroupProblem:
    """The internal pure-integer closure of a program and hardware description."""

    time_unit: str
    warp_groups: int
    resources: tuple[ResourceCapacity, ...]
    types: tuple[TensorType, ...]
    inputs: tuple[ProgramInput, ...]
    loop: ProblemLoop
    #: Carried through from the program unpriced. Closing a program against
    #: hardware prices its loop; a region has no steady state to be priced
    #: against, so the closure copies it rather than costing it.
    prologue: tuple[RegionOperation, ...] = ()
    epilogue: tuple[RegionOperation, ...] = ()

    def __post_init__(self) -> None:
        validate_id(self.time_unit, "time_unit")
        positive_int(self.warp_groups, "warp_groups")
        resources = _typed_tuple(self.resources, ResourceCapacity, "problem resources")
        types = _typed_tuple(self.types, TensorType, "problem types")
        inputs = _typed_tuple(self.inputs, ProgramInput, "problem inputs")
        if type(self.loop) is not ProblemLoop:
            raise WarpgroupValidationError("problem loop must be ProblemLoop")
        prologue = _typed_tuple(self.prologue, RegionOperation, "problem prologue")
        epilogue = _typed_tuple(self.epilogue, RegionOperation, "problem epilogue")
        _unique_ids(resources, "resource")
        _unique_ids(types, "type")
        _unique_ids(inputs, "input")
        object.__setattr__(self, "resources", tuple(sorted(resources, key=lambda item: item.id)))
        object.__setattr__(self, "types", tuple(sorted(types, key=lambda item: item.id)))
        object.__setattr__(self, "inputs", tuple(sorted(inputs, key=lambda item: item.id)))
        object.__setattr__(self, "prologue", prologue)
        object.__setattr__(self, "epilogue", epilogue)
        _validate_ownership(
            self.loop.ops,
            self.warp_groups,
            label="problem",
        )
        _validate_region_ownership(prologue + epilogue, self.warp_groups)
        capacity = {item.id: item.capacity for item in self.resources}
        for operation in self.loop.ops:
            for window in operation.resource_windows:
                if window.resource_id not in capacity:
                    raise WarpgroupValidationError(
                        f"operation {operation.id!r} uses undefined resource {window.resource_id!r}"
                    )
                if window.amount > capacity[window.resource_id]:
                    raise WarpgroupValidationError(
                        f"operation {operation.id!r} window for {window.resource_id!r} "
                        "exceeds capacity"
                    )
        _validate_semantics(self.types, self.inputs, self.loop, prologue, epilogue)

    def dependencies(self) -> tuple[DefUseDependency, ...]:
        """Derive the canonical operation graph from SSA definitions and uses."""
        return _derive_dependencies(self.inputs, self.loop, self.prologue)


@dataclass(frozen=True, slots=True)
class WarpgroupLane:
    """The stable ordered program assigned to one warpgroup.

    ``operations`` is the body, issued once per trip. ``prologue`` and
    ``epilogue`` are the region operations this lane runs before the first trip
    and after the last, in the order the program declared them; an emitter
    places them either side of this lane's trip loop, inside its arm of the
    warp-group dispatch. Both are empty for a program that declares no region.
    """

    operations: tuple[str, ...]
    prologue: tuple[str, ...] = ()
    epilogue: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for region in ("operations", "prologue", "epilogue"):
            value = getattr(self, region)
            if not isinstance(value, (tuple, list)):
                raise WarpgroupValidationError(f"lane {region} must be a sequence")
            entries = tuple(value)
            for operation in entries:
                validate_id(operation, f"lane {region} operation ID")
            if len(entries) != len(set(entries)):
                raise WarpgroupValidationError(f"a lane must not repeat a {region} operation")
            object.__setattr__(self, region, entries)
        everything = self.operations + self.prologue + self.epilogue
        if len(everything) != len(set(everything)):
            raise WarpgroupValidationError(
                "a lane must not name one operation in more than one region"
            )


@dataclass(frozen=True, slots=True, order=True)
class SynchronizationEdge:
    """One selected completion-to-start happens-before relation."""

    after: str
    before: str
    distance: int

    def __post_init__(self) -> None:
        validate_id(self.after, "sync after operation")
        validate_id(self.before, "sync before operation")
        non_negative_int(self.distance, "sync distance")
        if self.after == self.before and self.distance == 0:
            raise WarpgroupValidationError("sync must not contain a distance-zero self edge")


@dataclass(frozen=True, slots=True, order=True)
class TimedOperation:
    """One finite operation witness with issue end and completion timestamps."""

    iteration: int
    operation_id: str
    start: int
    issue_end: int
    completion: int

    def __post_init__(self) -> None:
        non_negative_int(self.iteration, "time iteration")
        validate_id(self.operation_id, "timed operation ID")
        non_negative_int(self.start, "time start")
        positive_int(self.issue_end, "time issue end")
        positive_int(self.completion, "time completion")
        if self.issue_end <= self.start:
            raise WarpgroupValidationError("time issue end must be greater than start")
        if self.completion < self.issue_end:
            raise WarpgroupValidationError("time completion must be >= issue end")


@dataclass(frozen=True, slots=True)
class WarpgroupSchedule:
    """A successful lane, synchronization, and concrete-time result."""

    format: str
    lanes: tuple[WarpgroupLane, ...]
    sync: tuple[SynchronizationEdge, ...]
    times: tuple[TimedOperation, ...]

    def __post_init__(self) -> None:
        if self.format != SCHEDULE_FORMAT:
            raise WarpgroupValidationError(f"schedule format must be {SCHEDULE_FORMAT!r}")
        lanes = _typed_tuple(self.lanes, WarpgroupLane, "schedule lanes")
        sync = _typed_tuple(self.sync, SynchronizationEdge, "schedule sync")
        times = _typed_tuple(self.times, TimedOperation, "schedule times")
        if not lanes:
            raise WarpgroupValidationError("schedule must contain at least one lane")
        lane_ops = tuple(
            operation
            for lane in lanes
            for operation in lane.operations + lane.prologue + lane.epilogue
        )
        if len(lane_ops) != len(set(lane_ops)):
            raise WarpgroupValidationError("an operation must not appear in more than one lane")
        if len(sync) != len(set(sync)):
            raise WarpgroupValidationError("schedule contains duplicate sync edges")
        instances = tuple((item.iteration, item.operation_id) for item in times)
        if len(instances) != len(set(instances)):
            raise WarpgroupValidationError("schedule contains duplicate timed instances")
        object.__setattr__(self, "lanes", lanes)
        object.__setattr__(self, "sync", tuple(sorted(sync)))
        object.__setattr__(self, "times", tuple(sorted(times)))


def _validate_ownership(
    operations: tuple[ProgramOperation, ...] | tuple[ProblemOperation, ...],
    warp_groups: int,
    *,
    label: str,
) -> None:
    for operation in operations:
        if operation.warp_group is None:
            continue
        if operation.warp_group >= warp_groups:
            raise WarpgroupValidationError(
                f"{label} operation {operation.id!r} warp_group is out of range"
            )


def _validate_region_ownership(operations: tuple[RegionOperation, ...], warp_groups: int) -> None:
    for operation in operations:
        if operation.warp_group >= warp_groups:
            raise WarpgroupValidationError(
                f"region operation {operation.id!r} warp_group is out of range"
            )


@dataclass(frozen=True, slots=True)
class _ValueType:
    shape: tuple[int, ...]
    dtype: DType | None
    space: MemorySpace | None


def _validate_semantics(
    types: tuple[TensorType, ...],
    inputs: tuple[ProgramInput, ...],
    loop: ProgramLoop | ProblemLoop,
    prologue: tuple[RegionOperation, ...] = (),
    epilogue: tuple[RegionOperation, ...] = (),
) -> None:
    type_by_id = {item.id: item for item in types}
    for item in inputs:
        if item.type_id not in type_by_id:
            raise WarpgroupValidationError(
                f"input {item.id!r} references undefined type {item.type_id!r}"
            )

    operation_ids = tuple(
        operation.id for operation in (*loop.ops, *prologue, *epilogue)
    )
    if len(operation_ids) != len(set(operation_ids)):
        repeated = sorted({item for item in operation_ids if operation_ids.count(item) > 1})
        raise WarpgroupValidationError(f"operation ID(s) used more than once: {repeated!r}")

    body_outputs = tuple(output for operation in loop.ops for output in operation.outputs)
    prologue_outputs = tuple(output for operation in prologue for output in operation.outputs)
    epilogue_outputs = tuple(output for operation in epilogue for output in operation.outputs)
    output_records = body_outputs + prologue_outputs + epilogue_outputs
    output_owner = {
        output.id: operation.id
        for operation in (*loop.ops, *prologue, *epilogue)
        for output in operation.outputs
    }
    all_ssa = (
        tuple(item.id for item in inputs)
        + tuple(item.id for item in loop.iter_args)
        + tuple(output.id for output in output_records)
    )
    if loop.index in all_ssa:
        raise WarpgroupValidationError(f"loop index {loop.index!r} duplicates an SSA definition")
    if len(all_ssa) != len(set(all_ssa)):
        repeated = sorted({item for item in all_ssa if all_ssa.count(item) > 1})
        raise WarpgroupValidationError(f"duplicate SSA definition(s): {repeated!r}")
    epilogue_ids = {output.id for output in epilogue_outputs}
    for output in output_records:
        declared = type_by_id.get(output.type_id)
        if declared is None:
            raise WarpgroupValidationError(
                f"output {output.id!r} references undefined type {output.type_id!r}"
            )
        # The epilogue is where a kernel's result is written and a result lives
        # in global memory. Anywhere else a global definition would claim that a
        # trip owns storage outside the block, which no operation here does.
        if declared.space is MemorySpace.GLOBAL and output.id not in epilogue_ids:
            raise WarpgroupValidationError(
                f"output {output.id!r} cannot define global external storage"
            )

    body_by_id = {item.id: item for item in body_outputs}
    # A prologue definition has landed by the time the first trip starts, which
    # is what being an input means to the loop, so the two are one class.
    input_ids = {item.id for item in inputs} | {output.id for output in prologue_outputs}
    value_types: dict[str, TensorType] = {item.id: type_by_id[item.type_id] for item in inputs}
    for output in output_records:
        value_types[output.id] = type_by_id[output.type_id]
    for iter_arg in loop.iter_args:
        yielded = body_by_id.get(iter_arg.yield_value.id)
        if yielded is None:
            raise WarpgroupValidationError(
                f"iter_arg {iter_arg.id!r} yield must name a loop-body SSA definition"
            )
        value_types[iter_arg.id] = type_by_id[yielded.type_id]

    for iter_arg in loop.iter_args:
        phi_type = value_types[iter_arg.id]
        if type(iter_arg.init) is ValueRef:
            init_type = value_types.get(iter_arg.init.id)
            if init_type is None or iter_arg.init.id not in input_ids:
                raise WarpgroupValidationError(
                    f"iter_arg {iter_arg.id!r} init must name an external input or be a scalar"
                )
            _require_same_tensor(init_type, phi_type, f"iter_arg {iter_arg.id!r} init")
        elif type(iter_arg.init) is ScalarLiteral:
            _validate_scalar_for_dtype(
                iter_arg.init, phi_type.dtype, f"iter_arg {iter_arg.id!r} init"
            )
        else:
            raise WarpgroupValidationError(
                f"iter_arg {iter_arg.id!r} init is not an exact typed value"
            )

    def check(outputs: tuple[OperationOutput, ...], visible: set[str], region: bool) -> None:
        for output in outputs:
            for use in value_references(output.expression):
                if use not in visible:
                    raise WarpgroupValidationError(
                        f"output {output.id!r} uses undefined SSA value {use!r}"
                    )
            if region and loop.index in _loop_index_references(output.expression):
                raise WarpgroupValidationError(
                    f"output {output.id!r} uses the loop index {loop.index!r} outside the loop"
                )
            _check_expression(
                output.expression,
                type_by_id[output.type_id],
                value_types,
                loop.index,
                loop.iterations,
            )

    # Visibility is positional in a region and global in the body. A region runs
    # once, in the order it is written, so an operation may read only what
    # something ahead of it has already produced. The body's order is the
    # schedule's to choose, so its operations all see each other and the cycle
    # check below is what refuses the arrangements that cannot exist.
    visible = {item.id for item in inputs}
    for operation in prologue:
        visible.update(output.id for output in operation.outputs)
        check(operation.outputs, visible, region=True)
    body_visible = (
        input_ids
        | {item.id for item in loop.iter_args}
        | {output.id for output in body_outputs}
    )
    check(body_outputs, body_visible, region=False)
    visible = set(body_visible)
    for operation in epilogue:
        visible.update(output.id for output in operation.outputs)
        check(operation.outputs, visible, region=True)

    graph = {
        output.id: tuple(ref for ref in value_references(output.expression) if ref in output_owner)
        for output in output_records
    }
    _reject_expression_cycles(graph)


def _loop_index_references(value: ExpressionValue) -> tuple[str, ...]:
    """Every loop-index use in an expression, which a region may not contain."""
    return fold_expression(
        value,
        reference=lambda _item: (),
        scalar=lambda _item: (),
        compose=lambda _operator, attributes, children: tuple(
            item.id for item in attributes if type(item) is LoopIndexRef
        )
        + tuple(reference for child in children for reference in child),
    )


def _validate_scalar_for_dtype(value: ScalarLiteral, dtype: DType, label: str) -> None:
    if value.value is NegativeInfinity.VALUE and dtype not in _FLOAT_DTYPES:
        raise WarpgroupValidationError(f"{label} '-inf' requires a floating dtype")


def _require_same_tensor(actual: TensorType, expected: TensorType, label: str) -> None:
    if (actual.shape, actual.dtype, actual.space) != (
        expected.shape,
        expected.dtype,
        expected.space,
    ):
        raise WarpgroupValidationError(f"{label} type does not match its phi/yield type")


def _value_type(
    value: ExpressionValue,
    value_types: dict[str, TensorType],
    loop_index: str,
    iterations: int,
    matmul_dtype: DType,
) -> _ValueType:
    if type(value) is ValueRef:
        item = value_types[value.id]
        return _ValueType(item.shape, item.dtype, item.space)
    if type(value) is ScalarLiteral:
        return _ValueType((), None, None)
    if type(value) is IndexExpression:
        source = value_types[value.source.id]
        if len(value.indices) > len(source.shape):
            raise WarpgroupValidationError("index expression has more indices than dimensions")
        for position, index in enumerate(value.indices):
            extent = source.shape[position]
            if type(index) is int and index >= extent:
                raise WarpgroupValidationError(
                    f"index {index} is out of bounds for extent {extent}"
                )
            if type(index) is LoopIndexRef:
                if index.id != loop_index:
                    raise WarpgroupValidationError(
                        f"index expression references unknown loop index {index.id!r}"
                    )
                if iterations > extent:
                    raise WarpgroupValidationError(
                        f"loop index range {iterations} exceeds indexed extent {extent}"
                    )
        return _ValueType(source.shape[len(value.indices) :], source.dtype, source.space)
    if type(value) is CopyExpression:
        return _value_type(value.source, value_types, loop_index, iterations, matmul_dtype)
    if type(value) is CastExpression:
        return _value_type(value.source, value_types, loop_index, iterations, matmul_dtype)
    if type(value) is TransposeExpression:
        transposed = _value_type(value.source, value_types, loop_index, iterations, matmul_dtype)
        if len(transposed.shape) != 2:
            raise WarpgroupValidationError("transpose requires a rank-two source")
        return _ValueType(
            (transposed.shape[1], transposed.shape[0]),
            transposed.dtype,
            transposed.space,
        )
    if type(value) is MatmulExpression:
        lhs = _value_type(value.lhs, value_types, loop_index, iterations, matmul_dtype)
        rhs = _value_type(value.rhs, value_types, loop_index, iterations, matmul_dtype)
        if len(lhs.shape) != 2 or len(rhs.shape) != 2:
            raise WarpgroupValidationError("matmul requires two rank-two operands")
        if lhs.shape[1] != rhs.shape[0]:
            raise WarpgroupValidationError("matmul contracting dimensions do not match")
        if lhs.dtype is None or lhs.dtype != rhs.dtype:
            raise WarpgroupValidationError("matmul operand dtypes must match")
        if lhs.dtype in _FLOAT_DTYPES and matmul_dtype not in _FLOAT_DTYPES:
            raise WarpgroupValidationError(
                "floating matmul requires a declared floating result dtype"
            )
        if lhs.dtype in _INTEGER_DTYPES and matmul_dtype not in _INTEGER_DTYPES:
            raise WarpgroupValidationError(
                "integer matmul requires a declared integer result dtype"
            )
        if lhs.dtype is DType.I1:
            raise WarpgroupValidationError("matmul does not accept i1 operands")
        return _ValueType((lhs.shape[0], rhs.shape[1]), matmul_dtype, MemorySpace.REGISTER)
    if type(value) is ConcatExpression:
        items = tuple(
            _value_type(item, value_types, loop_index, iterations, matmul_dtype)
            for item in value.values
        )
        first = items[0]
        if value.axis >= len(first.shape):
            raise WarpgroupValidationError("concat axis is out of bounds")
        if first.dtype is None:
            raise WarpgroupValidationError("concat does not accept scalar literals")
        shape = list(first.shape)
        shape[value.axis] = 0
        for concat_item in items:
            if (
                concat_item.dtype != first.dtype
                or concat_item.space != first.space
                or len(concat_item.shape) != len(first.shape)
            ):
                raise WarpgroupValidationError(
                    "concat operands must have matching rank, dtype, and space"
                )
            if any(
                extent != first.shape[axis]
                for axis, extent in enumerate(concat_item.shape)
                if axis != value.axis
            ):
                raise WarpgroupValidationError("concat non-axis dimensions must match")
            shape[value.axis] += concat_item.shape[value.axis]
        return _ValueType(tuple(shape), first.dtype, first.space)
    if type(value) is SelectExpression:
        condition = _value_type(value.condition, value_types, loop_index, iterations, matmul_dtype)
        when_true = _value_type(value.when_true, value_types, loop_index, iterations, matmul_dtype)
        when_false = _value_type(
            value.when_false, value_types, loop_index, iterations, matmul_dtype
        )
        if condition.dtype is not DType.I1:
            raise WarpgroupValidationError("select condition must have i1 dtype")
        broadcast_shape = _broadcast_shapes(condition.shape, when_true.shape, when_false.shape)
        dtype = _common_dtype((when_true, when_false), "select values")
        return _ValueType(broadcast_shape, dtype, MemorySpace.REGISTER)
    if type(value) is ReduceExpression:
        reduced = _value_type(value.source, value_types, loop_index, iterations, matmul_dtype)
        if value.axis >= len(reduced.shape):
            raise WarpgroupValidationError("reduce axis is out of bounds")
        reduced_shape = list(reduced.shape)
        reduced_shape[value.axis] = 1
        return _ValueType(tuple(reduced_shape), reduced.dtype, MemorySpace.REGISTER)
    if type(value) is ElementwiseExpression:
        operands = tuple(
            _value_type(item, value_types, loop_index, iterations, matmul_dtype)
            for item in value.operands
        )
        broadcast_shape = _broadcast_shapes(*(item.shape for item in operands))
        dtype = _common_dtype(operands, f"{value.operator.value} operands")
        if value.operator.value == "exp" and dtype not in _FLOAT_DTYPES:
            raise WarpgroupValidationError("exp requires a floating dtype")
        return _ValueType(broadcast_shape, dtype, MemorySpace.REGISTER)
    raise WarpgroupValidationError(f"unsupported expression type {type(value).__name__}")


def _check_expression(
    expression: ExpressionValue,
    expected: TensorType,
    value_types: dict[str, TensorType],
    loop_index: str,
    iterations: int,
) -> None:
    if type(expression) is ScalarLiteral:
        _validate_scalar_for_dtype(expression, expected.dtype, "output scalar")
        return
    actual = _value_type(expression, value_types, loop_index, iterations, expected.dtype)
    if actual.shape != expected.shape:
        raise WarpgroupValidationError(
            f"expression shape {actual.shape!r} does not match output shape {expected.shape!r}"
        )
    if type(expression) is CastExpression:
        if actual.space != expected.space:
            raise WarpgroupValidationError("cast must preserve memory space")
        return
    if type(expression) is CopyExpression:
        if actual.dtype != expected.dtype:
            raise WarpgroupValidationError("copy must preserve dtype")
        return
    if actual.dtype is not None and actual.dtype != expected.dtype:
        raise WarpgroupValidationError(
            f"expression dtype {actual.dtype.value!r} does not match output "
            f"dtype {expected.dtype.value!r}"
        )
    if (
        type(expression)
        in (
            MatmulExpression,
            SelectExpression,
            ReduceExpression,
            ElementwiseExpression,
        )
        and expected.space is not MemorySpace.REGISTER
    ):
        raise WarpgroupValidationError("computed expressions must produce register values")
    if type(expression) not in (CastExpression, CopyExpression) and actual.space != expected.space:
        raise WarpgroupValidationError("expression memory space does not match output type")


def _broadcast_shapes(*shapes: tuple[int, ...]) -> tuple[int, ...]:
    rank = max((len(shape) for shape in shapes), default=0)
    result: list[int] = []
    for offset in range(1, rank + 1):
        extents = {
            shape[-offset] for shape in shapes if len(shape) >= offset and shape[-offset] != 1
        }
        if len(extents) > 1:
            raise WarpgroupValidationError(f"invalid singleton broadcast across shapes {shapes!r}")
        result.append(next(iter(extents), 1))
    return tuple(reversed(result))


def _common_dtype(values: tuple[_ValueType, ...], label: str) -> DType | None:
    dtypes = {item.dtype for item in values if item.dtype is not None}
    if len(dtypes) > 1:
        raise WarpgroupValidationError(f"{label} must have one non-literal dtype")
    return next(iter(dtypes), None)


def _reject_expression_cycles(graph: dict[str, tuple[str, ...]]) -> None:
    active: set[str] = set()
    done: set[str] = set()

    def visit(node: str) -> None:
        if node in active:
            raise WarpgroupValidationError(f"SSA expression graph contains a cycle at {node!r}")
        if node in done:
            return
        active.add(node)
        for dependency in graph[node]:
            visit(dependency)
        active.remove(node)
        done.add(node)

    for node in sorted(graph):
        visit(node)


def _derive_dependencies(
    inputs: tuple[ProgramInput, ...],
    loop: ProgramLoop | ProblemLoop,
    prologue: tuple[RegionOperation, ...] = (),
) -> tuple[DefUseDependency, ...]:
    # A prologue definition has completed before the first trip starts, so it
    # constrains no body operation and enters this graph exactly as an input
    # does. That is what keeps a region out of the periodic model entirely.
    external = {item.id for item in inputs} | {
        output.id for operation in prologue for output in operation.outputs
    }
    owner = {output.id: operation.id for operation in loop.ops for output in operation.outputs}
    carried = {item.id: owner[item.yield_value.id] for item in loop.iter_args}
    edges: set[DefUseDependency] = set()
    for operation in loop.ops:
        for output in operation.outputs:
            for use in value_references(output.expression):
                if use in external:
                    continue
                if use in carried:
                    defining_operation = carried[use]
                    edges.add(DefUseDependency(defining_operation, operation.id, 1))
                    continue
                defining_operation = owner[use]
                if defining_operation != operation.id:
                    edges.add(DefUseDependency(defining_operation, operation.id, 0))
    return tuple(sorted(edges))


__all__ = [
    "DType",
    "DefUseDependency",
    "LoopIterArg",
    "MemorySpace",
    "OperationOutput",
    "PROGRAM_FORMAT",
    "ProblemLoop",
    "ProblemOperation",
    "ProgramInput",
    "ProgramLoop",
    "ProgramOperation",
    "RegionOperation",
    "ResourceCapacity",
    "ResourceWindow",
    "SCHEDULE_FORMAT",
    "SynchronizationEdge",
    "TensorType",
    "TimedOperation",
    "WarpgroupLane",
    "WarpgroupProblem",
    "WarpgroupProgram",
    "WarpgroupSchedule",
]
