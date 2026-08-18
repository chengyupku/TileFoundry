"""Immutable expressions for kernel-independent warpgroup programs."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import TypeVar

from ._identifiers import non_negative_int, validate_ssa_id
from .errors import WarpgroupValidationError


class NegativeInfinity(str, Enum):
    """The one non-finite scalar literal admitted by the expression grammar."""

    VALUE = "-inf"


class ReductionOperator(str, Enum):
    """Reduction functions supported by the typed grammar."""

    MAX = "max"
    SUM = "sum"


class ElementwiseOperator(str, Enum):
    """Elementwise functions supported by the typed grammar."""

    ADD = "add"
    SUB = "sub"
    MUL = "mul"
    MAX = "max"
    EXP = "exp"
    #: Base two, and separate from `exp` deliberately. They are different
    #: instructions -- `expf` is `exp2f` and a multiply -- so a program that
    #: cannot name the second cannot ask a backend for the cheaper one, and a
    #: cost model closed over what the program declares would be pricing a
    #: kernel nobody is able to write. Measured on an H200 attention loop,
    #: writing the exponent as `exp2(a*k - b*k)` rather than `exp(a - b)` with
    #: the scale applied in a pass of its own is 0.147 ms against 0.175.
    EXP2 = "exp2"
    #: What an epilogue does to a carried accumulator: an attention output is
    #: rescaled by the denominator the loop summed. There is no way to write
    #: that as a multiply without a reciprocal the grammar also lacks, so a
    #: program that cannot name division cannot describe the statement after
    #: its last trip.
    DIV = "div"


#: Elementwise operators of one operand. Named as a set rather than tested one
#: at a time, so that adding a second unary function is adding it here and not
#: finding every place `exp` was compared against.
UNARY = frozenset({ElementwiseOperator.EXP, ElementwiseOperator.EXP2})
#: Operators of exactly two operands. `add`, `mul` and `max` are associative and
#: read as a fold over any number of them; subtraction and division do not, and
#: a three-operand one would be an expression whose meaning is the reader's
#: guess at an association order.
BINARY = frozenset({ElementwiseOperator.SUB, ElementwiseOperator.DIV})


@dataclass(frozen=True, slots=True)
class Expression:
    """Base class for the closed expression variants."""


@dataclass(frozen=True, slots=True)
class ValueRef:
    """One use of an SSA value."""

    id: str

    def __post_init__(self) -> None:
        validate_ssa_id(self.id, "value reference")


@dataclass(frozen=True, slots=True)
class LoopIndexRef:
    """One use of the enclosing loop index in an index expression."""

    id: str

    def __post_init__(self) -> None:
        validate_ssa_id(self.id, "loop index reference")


@dataclass(frozen=True, slots=True)
class ScalarLiteral:
    """A finite JSON number or the explicit negative-infinity token."""

    value: int | float | NegativeInfinity

    def __post_init__(self) -> None:
        value = self.value
        if isinstance(value, NegativeInfinity):
            return
        if type(value) not in (int, float) or not math.isfinite(value):
            raise WarpgroupValidationError(
                f"scalar literal must be a finite number or '-inf', got {value!r}"
            )


ExpressionValue = Expression | ValueRef | ScalarLiteral
IndexOperand = int | LoopIndexRef
ExpressionAttribute = int | str | LoopIndexRef
_T = TypeVar("_T")


def _expression_value(value: ExpressionValue, label: str) -> None:
    if type(value) not in (
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
    ):
        raise WarpgroupValidationError(f"{label} must be a typed expression value")


@dataclass(frozen=True, slots=True)
class IndexExpression(Expression):
    """Remove one leading source dimension per supplied index."""

    source: ValueRef
    indices: tuple[IndexOperand, ...]

    def __post_init__(self) -> None:
        if type(self.source) is not ValueRef:
            raise WarpgroupValidationError("index source must be a ValueRef")
        indices = tuple(self.indices)
        if not indices:
            raise WarpgroupValidationError("index expression requires at least one index")
        for index in indices:
            if type(index) is int:
                non_negative_int(index, "static index")
            elif type(index) is not LoopIndexRef:
                raise WarpgroupValidationError(
                    "index operands must be non-negative integers or LoopIndexRef records"
                )
        object.__setattr__(self, "indices", indices)


@dataclass(frozen=True, slots=True)
class CopyExpression(Expression):
    """Copy a value while preserving its shape and dtype."""

    source: ExpressionValue

    def __post_init__(self) -> None:
        _expression_value(self.source, "copy source")


@dataclass(frozen=True, slots=True)
class CastExpression(Expression):
    """Cast a value to the dtype declared by its output."""

    source: ExpressionValue

    def __post_init__(self) -> None:
        _expression_value(self.source, "cast source")


@dataclass(frozen=True, slots=True)
class MatmulExpression(Expression):
    """Multiply two rank-two tensors."""

    lhs: ExpressionValue
    rhs: ExpressionValue

    def __post_init__(self) -> None:
        _expression_value(self.lhs, "matmul lhs operand")
        _expression_value(self.rhs, "matmul rhs operand")


@dataclass(frozen=True, slots=True)
class TransposeExpression(Expression):
    """Transpose the two dimensions of a matrix."""

    source: ExpressionValue

    def __post_init__(self) -> None:
        _expression_value(self.source, "transpose source")


@dataclass(frozen=True, slots=True)
class ConcatExpression(Expression):
    """Concatenate equally typed tensors on one axis."""

    axis: int
    values: tuple[ExpressionValue, ...]

    def __post_init__(self) -> None:
        non_negative_int(self.axis, "concat axis")
        values = tuple(self.values)
        if len(values) < 2:
            raise WarpgroupValidationError("concat requires at least two values")
        for value in values:
            _expression_value(value, "concat operand")
        object.__setattr__(self, "values", values)


@dataclass(frozen=True, slots=True)
class SelectExpression(Expression):
    """Select elementwise between two singleton-broadcastable values."""

    condition: ExpressionValue
    when_true: ExpressionValue
    when_false: ExpressionValue

    def __post_init__(self) -> None:
        _expression_value(self.condition, "select condition")
        _expression_value(self.when_true, "select true value")
        _expression_value(self.when_false, "select false value")


@dataclass(frozen=True, slots=True)
class ReduceExpression(Expression):
    """Reduce one axis while retaining it with extent one."""

    operator: ReductionOperator
    axis: int
    source: ExpressionValue

    def __post_init__(self) -> None:
        if type(self.operator) is not ReductionOperator:
            raise WarpgroupValidationError("reduce operator must be ReductionOperator")
        non_negative_int(self.axis, "reduce axis")
        _expression_value(self.source, "reduce source")


@dataclass(frozen=True, slots=True)
class ElementwiseExpression(Expression):
    """Apply one supported scalar function with singleton broadcasting."""

    operator: ElementwiseOperator
    operands: tuple[ExpressionValue, ...]

    def __post_init__(self) -> None:
        if type(self.operator) is not ElementwiseOperator:
            raise WarpgroupValidationError("elementwise operator must be ElementwiseOperator")
        operands = tuple(self.operands)
        unary = self.operator in UNARY
        required = 1 if unary else 2
        if unary and len(operands) != 1:
            raise WarpgroupValidationError(
                f"{self.operator.value} requires exactly one operand")
        if self.operator in BINARY and len(operands) != 2:
            raise WarpgroupValidationError(
                f"{self.operator.value} requires exactly two operands")
        if len(operands) < required:
            raise WarpgroupValidationError(
                f"{self.operator.value} requires at least {required} operand(s)"
            )
        for operand in operands:
            _expression_value(operand, f"{self.operator.value} operand")
        object.__setattr__(self, "operands", operands)


def fold_expression(
    value: ExpressionValue,
    *,
    reference: Callable[[ValueRef], _T],
    scalar: Callable[[ScalarLiteral], _T],
    compose: Callable[[str, tuple[ExpressionAttribute, ...], tuple[_T, ...]], _T],
) -> _T:
    """Fold one expression tree through shared leaf and operator callbacks."""
    if type(value) is ValueRef:
        return reference(value)
    if type(value) is ScalarLiteral:
        return scalar(value)
    if type(value) is IndexExpression:
        return compose("index", value.indices, (reference(value.source),))
    if type(value) is CopyExpression:
        return compose(
            "copy",
            (),
            (fold_expression(value.source, reference=reference, scalar=scalar, compose=compose),),
        )
    if type(value) is CastExpression:
        return compose(
            "cast",
            (),
            (fold_expression(value.source, reference=reference, scalar=scalar, compose=compose),),
        )
    if type(value) is TransposeExpression:
        return compose(
            "transpose",
            (),
            (fold_expression(value.source, reference=reference, scalar=scalar, compose=compose),),
        )
    if type(value) is MatmulExpression:
        return compose(
            "matmul",
            (),
            tuple(
                fold_expression(item, reference=reference, scalar=scalar, compose=compose)
                for item in (value.lhs, value.rhs)
            ),
        )
    if type(value) is ConcatExpression:
        return compose(
            "concat",
            (value.axis,),
            tuple(
                fold_expression(item, reference=reference, scalar=scalar, compose=compose)
                for item in value.values
            ),
        )
    if type(value) is SelectExpression:
        return compose(
            "select",
            (),
            tuple(
                fold_expression(item, reference=reference, scalar=scalar, compose=compose)
                for item in (value.condition, value.when_true, value.when_false)
            ),
        )
    if type(value) is ReduceExpression:
        return compose(
            "reduce",
            (value.operator.value, value.axis),
            (fold_expression(value.source, reference=reference, scalar=scalar, compose=compose),),
        )
    if type(value) is ElementwiseExpression:
        return compose(
            value.operator.value,
            (),
            tuple(
                fold_expression(item, reference=reference, scalar=scalar, compose=compose)
                for item in value.operands
            ),
        )
    raise WarpgroupValidationError(
        f"expression contains unsupported typed value {type(value).__name__}"
    )


def value_references(value: ExpressionValue) -> tuple[str, ...]:
    """Return every SSA use in expression traversal order."""
    return fold_expression(
        value,
        reference=lambda item: (item.id,),
        scalar=lambda _item: (),
        compose=lambda _operator, _attributes, children: tuple(
            reference for child in children for reference in child
        ),
    )


__all__ = [
    "BINARY",
    "CastExpression",
    "ConcatExpression",
    "CopyExpression",
    "ElementwiseExpression",
    "ElementwiseOperator",
    "Expression",
    "ExpressionAttribute",
    "ExpressionValue",
    "IndexExpression",
    "LoopIndexRef",
    "MatmulExpression",
    "NegativeInfinity",
    "ReduceExpression",
    "ReductionOperator",
    "ScalarLiteral",
    "SelectExpression",
    "TransposeExpression",
    "UNARY",
    "ValueRef",
    "fold_expression",
    "value_references",
]
