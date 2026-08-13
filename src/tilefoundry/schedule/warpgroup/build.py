"""Cost closure from an authored program to a replayable numeric problem."""

from __future__ import annotations

import json
from collections.abc import Callable

from .cost import (
    CanonicalExpression,
    CostLibrary,
    OperationCost,
    OperationKind,
    OperationSignature,
    SignatureOutput,
    SignatureValueType,
    WarpgroupCostMissingError,
    WarpgroupMissingSignaturesError,
)
from .errors import WarpgroupValidationError
from .expression import (
    ExpressionAttribute,
    ExpressionValue,
    NegativeInfinity,
    ScalarLiteral,
    fold_expression,
)
from .model import (
    PROBLEM_FORMAT,
    OperationOutput,
    ProblemLoop,
    ProblemOperation,
    ProgramOperation,
    TensorType,
    WarpgroupProblem,
    WarpgroupProgram,
)


def _value_type(value: TensorType) -> SignatureValueType:
    return SignatureValueType(value.shape, value.dtype, value.space)


def _type_table(program: WarpgroupProgram) -> dict[str, TensorType]:
    types = {item.id: item for item in program.types}
    values = {item.id: types[item.type_id] for item in program.inputs}
    outputs = {output.id: output for operation in program.loop.ops for output in operation.outputs}
    for output in outputs.values():
        values[output.id] = types[output.type_id]
    for item in program.loop.iter_args:
        values[item.id] = values[item.yield_value.id]
    return values


def _expression_tree(
    value: ExpressionValue,
    reference: Callable[[str], CanonicalExpression],
) -> CanonicalExpression:
    def scalar(item: ScalarLiteral) -> CanonicalExpression:
        literal = item.value.value if type(item.value) is NegativeInfinity else item.value
        return ("literal", literal)

    def compose(
        operator: str,
        attributes: tuple[ExpressionAttribute, ...],
        children: tuple[CanonicalExpression, ...],
    ) -> CanonicalExpression:
        def encode_attribute(item: ExpressionAttribute) -> str | int:
            if type(item) in (str, int):
                return item  # type: ignore[return-value]
            return "loop_index"

        encoded = tuple(encode_attribute(item) for item in attributes)
        if operator == "index":
            return (operator, *children, *encoded)
        return (operator, *encoded, *children)

    return fold_expression(
        value,
        reference=lambda item: reference(item.id),
        scalar=scalar,
        compose=compose,
    )


def operation_signature(
    program: WarpgroupProgram, operation: ProgramOperation
) -> OperationSignature:
    """Derive one identity-free signature from a validated program operation."""
    if type(program) is not WarpgroupProgram or type(operation) is not ProgramOperation:
        raise WarpgroupValidationError("operation_signature requires exact typed program records")
    values = _type_table(program)
    internal_outputs = {item.id: item for item in operation.outputs}

    def expression_for(
        output: OperationOutput,
        external_reference: Callable[[str], CanonicalExpression],
    ) -> CanonicalExpression:
        def reference(value_id: str) -> CanonicalExpression:
            internal = internal_outputs.get(value_id)
            if internal is None:
                return external_reference(value_id)
            internal_type = _value_type(values[value_id])
            return (
                "output",
                internal_type.shape,
                internal_type.dtype.value,
                internal_type.space.value,
                _expression_tree(internal.expression, reference),
            )

        return _expression_tree(output.expression, reference)

    def output_key(output: OperationOutput) -> str:
        if type(output) is not OperationOutput:
            raise WarpgroupValidationError("operation output must be an exact typed record")
        local_operands: list[SignatureValueType] = []
        local_ids: dict[str, int] = {}

        def local_reference(value_id: str) -> CanonicalExpression:
            if value_id not in local_ids:
                local_ids[value_id] = len(local_operands)
                local_operands.append(_value_type(values[value_id]))
            return ("ref", local_ids[value_id])

        expression = expression_for(output, local_reference)
        output_type = _value_type(values[output.id])
        return json.dumps(
            (
                (output_type.shape, output_type.dtype.value, output_type.space.value),
                tuple((item.shape, item.dtype.value, item.space.value) for item in local_operands),
                expression,
            ),
            separators=(",", ":"),
        )

    outputs = sorted(
        operation.outputs,
        key=output_key,
    )
    operands: list[SignatureValueType] = []
    operand_ids: dict[str, int] = {}

    def reference(value_id: str) -> CanonicalExpression:
        if value_id not in operand_ids:
            operand_ids[value_id] = len(operands)
            operands.append(_value_type(values[value_id]))
        return ("ref", operand_ids[value_id])

    signature_outputs: list[SignatureOutput] = []
    for output in outputs:
        signature_outputs.append(
            SignatureOutput(
                _value_type(values[output.id]),
                expression_for(output, reference),
            )
        )
    expressions = tuple(item.expression for item in signature_outputs)
    if all(expression and expression[0] == "copy" for expression in expressions):
        kind = OperationKind.COPY
    elif all(
        expression and expression[0] in {"concat", "index", "transpose"}
        for expression in expressions
    ):
        kind = OperationKind.VIEW
    else:
        kind = OperationKind.COMPUTE
    return OperationSignature(kind, tuple(operands), tuple(signature_outputs))


def build_warpgroup_problem(
    program: WarpgroupProgram, cost_library: CostLibrary
) -> WarpgroupProblem:
    """Resolve every unique operation signature before returning a problem."""
    if type(program) is not WarpgroupProgram:
        raise WarpgroupValidationError("build requires an exact WarpgroupProgram")
    resolved: dict[OperationSignature, OperationCost | None] = {}
    missing: dict[OperationSignature, None] = {}
    costs: dict[str, OperationCost] = {}
    for operation in program.loop.ops:
        signature = operation_signature(program, operation)
        if signature not in resolved:
            try:
                lookup_cost = cost_library.lookup(signature)
            except WarpgroupCostMissingError:
                missing[signature] = None
                resolved[signature] = None
                continue
            if type(lookup_cost) is not OperationCost:
                raise WarpgroupValidationError(
                    "cost library lookup must return exact OperationCost"
                )
            resolved[signature] = lookup_cost
        resolved_cost = resolved[signature]
        if resolved_cost is not None:
            costs[operation.id] = resolved_cost
    if missing:
        raise WarpgroupMissingSignaturesError(
            tuple(sorted(missing, key=lambda item: item.canonical_key))
        )
    operations = tuple(
        ProblemOperation(
            operation.id,
            operation.outputs,
            costs[operation.id].duration,
            costs[operation.id].resources,
        )
        for operation in program.loop.ops
    )
    loop = ProblemLoop(
        program.loop.index, program.loop.iterations, program.loop.iter_args, operations
    )
    return WarpgroupProblem(
        PROBLEM_FORMAT,
        cost_library.time_unit,
        program.warp_groups,
        cost_library.resources,
        program.types,
        program.inputs,
        loop,
    )


__all__ = ["build_warpgroup_problem", "operation_signature"]
