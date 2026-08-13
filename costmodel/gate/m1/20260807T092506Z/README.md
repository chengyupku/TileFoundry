# TileFoundry Cost Model M1 Gate Record

Run ID: `20260807T092506Z`

Status: **PASS - M1 CLOSED**

This run supersedes M1 run `20260807T081907Z` and its wheel SHA256
`f295d99dd75c7b440289bbef6406094504b06384bc083a57caef6212a93d1b0b`.
The superseded artifact predates the relation-DAG, provenance, and canonical
warp-ID corrections recorded below. M0 remains frozen: its evidence and wheel
were not modified, and its wheel SHA256 remains
`57058acb41622f7119da2389d96c9345df0523f5f355a83774a9559d9b2e949a`.

## Review Corrections

- `TileProgram` now rejects cycles in the operation-level relation graph in
  addition to the expanded finite instance graph. Positive-distance self
  recurrences remain legal. The retained regression has two two-iteration
  loops whose endpoint edges are instance-acyclic but form `left <-> right` at
  relation level.
- The fifth-generation TensorCore PTX provenance uses the existing
  `#tensorcore-5th-generation-instructions` anchor. TensorCore and tensor
  memory facts identify their documented `sm_100a/sm_100f` applicability in
  both the Python catalog and packaged calibration JSON.
- `WarpRoleAssignment` canonicalizes `warp_ids` as a sorted set before
  `SearchSpace` duplicate-choice comparison and request serialization. Shared
  physical warps across different roles remain legal.

## Artifact

- Wheel: `wheel/tilefoundry_costmodel-2.0.0-py3-none-any.whl`
- SHA256: `fbe0f7619b919ce2442b81051a36f84c9bc091d8d0f874475d53f94b21f4a84a`
- Base runtime dependencies: none.
- Optional requirements: six requirements across `cpsat`, `cuda`, and `test`.
- The root `tilefoundry_costmodel/__init__.py` is byte-identical to the frozen
  M0 root boundary.
- Seven schemas and `b200-hardware.json` are byte-identical between source and
  wheel and pass Draft 2020-12 / hardware-v1 validation.

See `wheel-build.log`, `wheel.sha256`, `wheel-contents.txt`,
`wheel-metadata.txt`, and `artifact-verification.log`.

## Local Matrix

| Environment | Result | Evidence |
| --- | --- | --- |
| Python 3.11 base wheel | PASS; `pip check`, CLI, isolated root import | `py311-base.log` |
| Python 3.12 base wheel | PASS; `pip check`, CLI, isolated root import | `py312-base.log` |
| Python 3.11 `[cpsat,test]` wheel | PASS; 47 tests from `site-packages` | `py311-cpsat-test.log` |
| Python 3.12 `[cpsat,test]` wheel | PASS; 47 tests from `site-packages` | `py312-cpsat-test.log` |

The installed-wheel tests run from isolated `/private/tmp` work directories.
Importing the root loads no CUDA, OR-Tools, or legacy modules.

## Linux/B200 Matrix

Mutagen session `b200-workspace-sync` delivered the exact wheel to node14.
`linux-b200-node14/host-gpu-environment.txt` records Ubuntu x86_64, driver
`590.48.01`, eight visible B200 devices, and CPython 3.11.15 / 3.12.3.
The remote wheel checksum matches the local artifact.

| Environment | Result | Evidence |
| --- | --- | --- |
| Python 3.11 `[cpsat,test]` | PASS; 47 tests from `site-packages`, `pip check` | `linux-b200-node14/py311-cpsat-test.log` |
| Python 3.12 `[cpsat,test]` | PASS; 47 tests from `site-packages`, `pip check` | `linux-b200-node14/py312-cpsat-test.log` |
| Python 3.11 `[cuda]` | PASS; `cuInit`, eight `NVIDIA B200` devices | `linux-b200-node14/cuda-py311.log` |
| Python 3.12 `[cuda]` | PASS; `cuInit`, eight `NVIDIA B200` devices | `linux-b200-node14/cuda-py312.log` |

The CUDA rows validate dependency installation and driver/device visibility;
they do not claim a CUDA profiler or operation provider.

## Static Gates

- Ruff check and format: PASS (`ruff-check.log`, `ruff-format.log`).
- Strict mypy: PASS, 34 source files (`mypy-strict.log`).
- Forward-reference lint: PASS (`forward-reference-lint.log`).
- Source tests: PASS, 47 tests, 66% branch-aware total coverage
  (`source-tests.log`).
- Schema and calibration validation: PASS (`schema-validator.log`).
- `git diff --check`: PASS (`git-diff-check.log`).
- Evidence checksums: `SHA256SUMS.txt`.

## Scope Boundary

M1 retains typed operations, program graph validation, workload frontend
protocols, exact B200 facts, strict JSON ownership, and legacy `(0, 2)`
behavior. M2 and M3 remain future work: operation lowering and providers,
concrete workload frontends, profile storage and resolution, CUDA/JIT
profiling, `build()`, finite/periodic solving, and end-to-end GEMM were not
implemented in this correction run.
