# TileFoundry Cost Model M2 Review Gate

Run ID: `20260810T094226Z`

Status: **PASS - M2 REOPENED, AWAITING REVIEW**

This run supersedes the accepted M2 run `20260810T043650Z` for the three
review findings below. It does not close M2 again. M0 and M1 evidence remain
frozen and were not edited. M3, M4, and M5 remain unopened.

## Review fixes

- `build.py` treats a compatibility `TileProgram` with no operations as an
  empty legal candidate set at the executable boundary. Strict M0 JSON still
  decodes and re-encodes it, while configuration enumeration raises
  `UnsupportedError` and never emits a zero-phase template.
- Synthetic elementwise lowering retains every input occurrence in the typed
  `TileOpSignature`, but emits one `ConsumedValue` and one release lifetime per
  distinct `value_id` in stable first-occurrence order. `add(x, x)` is a legal
  configuration.
- `tile_op_signature()` sorts only `ReduceOp.axes` when constructing the
  semantic signature. Query JSON and profile-key JSON/hash are invariant to an
  axis permutation; the source `TileProgram` JSON retains the caller's axis
  order.

## Artifact

- Wheel: `wheel/tilefoundry_costmodel-2.0.0-py3-none-any.whl`
- SHA256:
  `c0402e2e4a850f0f6378834888c41ea66a2a45c8d4cad8126c20fff9933832e3`
- Source/wheel bytes for the M2 modules, seven schemas, and B200 calibration
  match. Draft 2020-12 schema meta-validation and calibration validation pass.
- The M0, M1, and previous M2 wheel hashes are respectively
  `57058acb...e949a`, `fbe0f761...a84a`, and
  `d69725f1...49e3f8`; the frozen-boundary log verifies the first two and the
  old M2 artifact without modifying their manifests.

## Verification matrix

| Environment | Result | Evidence |
| --- | --- | --- |
| Source Python 3.12 | 65 tests, 72% branch coverage | `source-tests.log` |
| Python 3.11 base wheel | `pip check`, CLI, lazy import boundary pass | `py311-base.log` |
| Python 3.12 base wheel | `pip check`, CLI, lazy import boundary pass | `py312-base.log` |
| Python 3.11 `[cpsat,test]` wheel | 65 tests, 72% branch coverage from `site-packages` | `py311-cpsat-test.log` |
| Python 3.12 `[cpsat,test]` wheel | 65 tests, 72% branch coverage from `site-packages` | `py312-cpsat-test.log` |
| Typed/JSON Golden workflow | byte-identical in source and four wheel environments | `golden-parity.log` |
| Static, schema, artifact, and frozen-boundary checks | PASS | the corresponding `*.log` files |

All root-import checks confirm that OR-Tools, CUDA, and `legacy` are not
loaded implicitly. The host is macOS, so no CUDA driver or B200 execution was
attempted by this M2-only review run; CUDA JIT/profiling remains M3 scope.

## Commands

The recorded checks include:

```text
.venv/bin/ruff check costmodel/src costmodel/tests costmodel/benchmarks
.venv/bin/ruff format --check costmodel/src costmodel/tests costmodel/benchmarks
.venv/bin/mypy --strict costmodel/src
.venv/bin/python scripts/forward_references_lint.py <costmodel Python files>
.venv/bin/pytest costmodel/tests --cov=tilefoundry_costmodel --cov-branch
git diff --check
```

The three focused regression tests are recorded in `review-regressions.log`.
`SHA256SUMS.txt` covers every file in this run, including the wheel.
