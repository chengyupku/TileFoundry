# TileFoundry Cost Model M2 Gate Record

Run ID: `20260810T043650Z`

Status: **PASS - M2 CLOSED**

This run supersedes the incomplete local M2 draft run `20260810T035607Z` and
its wheel SHA256
`c0b5bb05d58b67698d6e6e02d4fe4bc08b0a58b42769c9703cd7e3b6dc57e04a`.
M0 and M1 remain frozen and their evidence directories were not modified. M2
implements untimed operation lowering, phase composition, canonical profile
identity, and finite configuration enumeration only. It does not implement
profile persistence or resolution, CUDA execution, public `build()`, or a new
solver.

## Artifact

- Wheel: `wheel/tilefoundry_costmodel-2.0.0-py3-none-any.whl`
- SHA256:
  `d69725f1f0afdea1b08e810faaa98e3a0631eb2846f7a5dcf32cdbd31849e3f8`
- Base runtime dependencies: none.
- Optional requirements: six requirements across `cpsat`, `cuda`, and `test`.
- Seven schemas and `b200-hardware.json` are byte-identical between source and
  wheel and pass Draft 2020-12 / hardware-v1 validation.
- The M2 source modules checked by `verify-artifact.py` are byte-identical to
  their wheel copies.

The package root remains byte-identical across source, M0, M1, and M2 with
SHA256
`e7f7fd0ed8db95273f63ac1f987cdc3d182418f504e3a6c19a19d567c5d09ff7`.
The frozen M0 wheel remains
`57058acb41622f7119da2389d96c9345df0523f5f355a83774a9559d9b2e949a`;
the accepted M1 wheel remains
`fbe0f7619b919ce2442b81051a36f84c9bc091d8d0f874475d53f94b21f4a84a`.
Both prior evidence manifests pass their complete checksum verification. See
`frozen-boundary.log`, `m0-evidence-check.log`, and
`m1-evidence-check.log`.

## Local Matrix

| Environment | Result | Evidence |
| --- | --- | --- |
| Python 3.11 base wheel | PASS; `pip check/freeze`, CLI, isolated root import, Golden workflow | `py311-base.log` |
| Python 3.12 base wheel | PASS; `pip check/freeze`, CLI, isolated root import, Golden workflow | `py312-base.log` |
| Python 3.11 `[cpsat,test]` wheel | PASS; 62 tests from `site-packages`, 71% branch coverage | `py311-cpsat-test.log` |
| Python 3.12 `[cpsat,test]` wheel | PASS; 62 tests from `site-packages`, 71% branch coverage | `py312-cpsat-test.log` |
| Wheel metadata, sources, schemas, calibration | PASS | `artifact-verification.log`, `wheel-metadata.txt`, `wheel-contents.txt` |

Every isolated import records that no OR-Tools, CUDA, or legacy module was
loaded by the version-2 root. The M2 Golden payload is byte-identical across
the source checkout and all four wheel environments. Its SHA256 is
`1acaec7311fd9a1cb4795bf551693f4eaac5bdd333f06a146ae7007725bb72f7`;
see `source-golden.log` and `golden-parity.log`.

## Contract Workflows

The named Golden Reference is `docs/spec/cost-model.md` sections 7 and 8.1.
The retained synthetic workflow has no measured timing and no CUDA execution.
It exercises the following observable points:

- Every phase retains its exact typed source operation, implementation,
  benchmark component, selected warps, temporal resources, hardware reference,
  tile, depth, layout, canonical operation signature, and profile query.
  Rewritten signatures and mismatched provider fingerprints are rejected.
- GEMM issue and latency phases select different metrics from one query and
  have one required zero-offset start alignment. There is no end-to-start edge
  adding II to latency, and complete readiness comes from the latency phase.
- A following GEMM requests `ordered` availability from the producer issue
  phase while an elementwise epilogue requests `complete` from the producer
  latency phase. The generic composer only matches named availability records.
- A hand-checkable loop GEMM at depths 1, 2, and 3 changes ring slots, shared
  bytes, configuration IDs, query depth, and profile-key IDs. Shared storage is
  aggregated once by value ID and CTA warps once per configuration.
- Program, search tuple, catalog, phase, resource, relation, and buffer order is
  canonical. Equivalent no-ring depth requests deduplicate at depth 1; missing
  or ambiguous availability fails at the typed boundary.
- The positive-distance aligned recurrence, cross-loop endpoint edge, and
  explicit loop barrier are preserved as their exact typed relations. The
  composer does not infer a barrier from phase or loop order.

The installed-wheel test matrix also retains M1's Python/JSON canonical
`TileProgram` workflow and the legacy `(0, 2)` scheduling Golden Reference.

## Static Gates

- Ruff check and format: PASS (`ruff-check.log`, `ruff-format.log`).
- Strict mypy: PASS, 41 source files (`mypy-strict.log`).
- Forward-reference lint: PASS (`forward-reference-lint.log`).
- Source tests: PASS, 62 tests and 71% branch-aware total coverage
  (`source-tests.log`).
- Schema, calibration, wheel metadata, and source/wheel bytes: PASS
  (`artifact-verification.log`).
- `git diff --check`: PASS (`git-diff-check.log`).
- Plan-policy, spec, reference, comment, machine-path, and English-only checks:
  PASS (`plan-policy-check.log` and the corresponding lint logs).
- Gate scripts pass Ruff/format and shell syntax checks.
- All retained evidence files are listed in `SHA256SUMS.txt`.

## Review Scope

The M2 tests and comments were reviewed for redundancy. The retained tests
exercise public records, catalogs, builders, canonical JSON, and resulting
templates; none inspect a private helper, AST, source spelling, object identity,
or hypothetical refactor. The additional negative cases close observable
contract gaps for full typed-signature traceability, mandatory asynchronous
alignment, unsupported warp-role configurations, and exact provider version
identity.

M3 remains responsible for SQLite storage, snapshot mutation/resolution,
CUDA/NVRTC/JIT execution, real B200 providers, and measured timings. M4 remains
responsible for public `build()`, timing closure, finite/periodic CP-SAT, and
end-to-end solving. No concrete GEMM, GQA, FlashAttention, or MLP frontend and
no TileFoundry adapter is implemented by this milestone.
