# TileFoundry Cost Model M2 Closure Gate

Run ID: `20260810T111112Z`

Status: **PASS - M2 CLOSED**

This closure records the review approval of M2. It does not rebuild or copy a
wheel and does not modify the earlier M0, M1, or M2 evidence directories.

## Referenced artifact

The exact accepted M2 wheel is referenced from
`../20260810T094226Z/wheel/tilefoundry_costmodel-2.0.0-py3-none-any.whl`:

`c0402e2e4a850f0f6378834888c41ea66a2a45c8d4cad8126c20fff9933832e3`

The closure checks that hash, source/wheel bytes, seven schemas, and B200
calibration. The typed/JSON Golden digest remains
`1acaec7311fd9a1cb4795bf551693f4eaac5bdd333f06a146ae7007725bb72f7`.

## Review and gate evidence

- The three review regressions pass: empty compatibility programs produce no
  zero-phase candidate; repeated elementwise operands retain duplicate
  signature operands but one value lifetime; reduction axis permutations share
  signature/query/profile-key identity while program JSON order is preserved.
- Source and installed wheel matrix: `65 passed`, `72%` branch coverage.
- Ruff check/format, strict mypy, forward-reference lint, `git diff --check`,
  artifact/schema validation, Golden parity, and frozen-boundary checks pass.
- M0, M1, and the prior M2 wheel/evidence hashes are recorded unchanged.

M3 may now begin. No M3, M4, or M5 implementation is included in this closure
run.
