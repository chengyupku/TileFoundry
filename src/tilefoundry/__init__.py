"""TileFoundry top-level package with lazy public re-exports."""

from __future__ import annotations

from importlib import import_module
from importlib.metadata import version as _distribution_version

__version__ = _distribution_version("tilefoundry")

_PUBLIC = {
    "AnalysisRegistry": ("tilefoundry.ir.core", "AnalysisRegistry"),
    "B": ("tilefoundry.ir.types.shard", "B"),
    "Broadcast": ("tilefoundry.ir.types.shard", "Broadcast"),
    "Call": ("tilefoundry.ir.core", "Call"),
    "CompilerOptions": ("tilefoundry.compile", "CompilerOptions"),
    "ComposedLayout": ("tilefoundry.ir.types.shard", "ComposedLayout"),
    "Constant": ("tilefoundry.ir.core", "Constant"),
    "DType": ("tilefoundry.ir.types", "DType"),
    "DimVar": ("tilefoundry.ir.types.dim", "DimVar"),
    "DimVarRangePat": ("tilefoundry.ir.core.pattern", "DimVarRangePat"),
    "Dynamic": ("tilefoundry.ir.types.shard", "Dynamic"),
    "Expr": ("tilefoundry.ir.core", "Expr"),
    "IntTuple": ("tilefoundry.ir.types.shard", "IntTuple"),
    "Layout": ("tilefoundry.ir.types.shard", "Layout"),
    "LayoutBase": ("tilefoundry.ir.types.shard", "LayoutBase"),
    "Mesh": ("tilefoundry.ir.types.shard", "Mesh"),
    "MeshAxis": ("tilefoundry.ir.types.shard", "MeshAxis"),
    "Op": ("tilefoundry.ir.core", "Op"),
    "P": ("tilefoundry.ir.types.shard", "P"),
    "ParameterInfo": ("tilefoundry.ir.core", "ParameterInfo"),
    "Partial": ("tilefoundry.ir.types.shard", "Partial"),
    "Pattern": ("tilefoundry.ir.core.pattern", "Pattern"),
    "S": ("tilefoundry.ir.types.shard", "S"),
    "ShardAttr": ("tilefoundry.ir.types.shard", "ShardAttr"),
    "ShardLayout": ("tilefoundry.ir.types.shard", "ShardLayout"),
    "Split": ("tilefoundry.ir.types.shard", "Split"),
    "Stmt": ("tilefoundry.ir.tir.stmt", "Stmt"),
    "TensorType": ("tilefoundry.ir.types", "TensorType"),
    "Topology": ("tilefoundry.ir.types.shard", "Topology"),
    "TupleGetItem": (
        "tilefoundry.ir.hir.tensor.tuple_get_item",
        "TupleGetItem",
    ),
    "TupleType": ("tilefoundry.ir.types", "TupleType"),
    "Type": ("tilefoundry.ir.types", "Type"),
    "TypeInferContext": ("tilefoundry.ir.core", "TypeInferContext"),
    "Var": ("tilefoundry.ir.core", "Var"),
    "VerifyError": ("tilefoundry.ir.core", "VerifyError"),
    "build": ("tilefoundry.compile", "build"),
    "compile": ("tilefoundry.compile", "compile"),
    "cost_evaluator_registry": (
        "tilefoundry.ir.core",
        "cost_evaluator_registry",
    ),
    "func": ("tilefoundry.script", "func"),
    "intrinsic": ("tilefoundry.script", "intrinsic"),
    "jit": ("tilefoundry.compile", "jit"),
    "lower": ("tilefoundry.compile", "lower"),
    "module": ("tilefoundry.module", "module"),
    "normalize_to_module": ("tilefoundry.compile", "normalize_to_module"),
    "prim_func": ("tilefoundry.script", "prim_func"),
    "register_cost_evaluator": (
        "tilefoundry.ir.core",
        "register_cost_evaluator",
    ),
    "register_typeinfer": ("tilefoundry.ir.core", "register_typeinfer"),
    "register_verify_stmt": ("tilefoundry.ir.core", "register_verify_stmt"),
    "typeinfer_registry": ("tilefoundry.ir.core", "typeinfer_registry"),
    "verify_stmt_registry": ("tilefoundry.ir.core", "verify_stmt_registry"),
}

_IR_READY = False


def _ensure_ir() -> None:
    """Load the operation registries required by the existing root API."""
    global _IR_READY
    if _IR_READY:
        return
    import_module("tilefoundry.ir.core")
    import_module("tilefoundry.ir.core.pattern")
    import_module("tilefoundry.ir.types")
    import_module("tilefoundry.ir.types.dim")
    import_module("tilefoundry.ir.types.shard")
    import_module("tilefoundry.ir.tir.stmt")
    import_module("tilefoundry.ir.hir")
    import_module("tilefoundry.ir.tir")
    import_module("tilefoundry.visitor_registry.op_cost")
    from tilefoundry.ir.types import _register_dim_typeinfer  # noqa: PLC0415

    _register_dim_typeinfer()
    # Loading HIR may set ``tilefoundry.module`` to the submodule object. Keep
    # the established root-level decorator binding stable for mixed imports.
    globals()["module"] = getattr(import_module("tilefoundry.module"), "module")
    _IR_READY = True


def __getattr__(name: str) -> object:
    """Resolve one established public name on first use."""
    entry = _PUBLIC.get(name)
    if entry is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    _ensure_ir()
    module_name, attribute = entry
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value


def view(root: object, *, port: int = 0, open_browser: bool = True) -> int:
    """Start the interactive HIR viewer for one IR root."""
    _ensure_ir()
    viewer = import_module("tilefoundry.inspection.viewer").Viewer
    return viewer(root).serve(port=port, open_browser=open_browser)


__all__ = ["__version__", *_PUBLIC, "view"]
