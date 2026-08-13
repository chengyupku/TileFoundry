# TileFoundry Cost Model M0 Gate Record

Run ID: `20260807T041107Z`

Status: **PASS - M0 CLOSED**

This run is restricted to M0 AC-0-4 and the formal gate. No M1 or later
implementation was added or retained in the package. The source tree contains
only M0 records, schema codecs, legacy scheduling, and explicit future-milestone
placeholders.

## Final Artifact

- Wheel: `wheel/tilefoundry_costmodel-2.0.0-py3-none-any.whl`
- SHA256: `57058acb41622f7119da2389d96c9345df0523f5f355a83774a9559d9b2e949a`
- Wheel contains seven schemas and no B200 calibration data or M1/M3 source
  modules. See `wheel-schema-boundary-final.txt`.

## Results

| Environment | Result | Evidence |
| --- | --- | --- |
| Python 3.11 base | PASS | `final-py311-base.log`, `final-py311-base.pip-check.txt`, `final-py311-base.pip-freeze.txt` |
| Python 3.12 base | PASS | `final-py312-base.log`, `final-py312-base.pip-check.txt`, `final-py312-base.pip-freeze.txt` |
| Python 3.11 CP-SAT | PASS | `final-py311-cpsat.log` |
| Python 3.12 CP-SAT | PASS | `final-py312-cpsat.log` |
| Python 3.11 test | PASS | `final-py311-test.log` |
| Python 3.12 test | PASS | `final-py312-test.log` |
| Python 3.11 `[cpsat,test]` | PASS, 31 tests | `final-py311-cpsat-test.log` |
| Python 3.12 `[cpsat,test]` | PASS, 31 tests | `final-py312-cpsat-test.log` |
| Linux/B200 Python 3.11 `[cuda]` | PASS, install and driver/device smoke | `linux-b200-node14/cuda-extra-py311.log` |
| Linux/B200 Python 3.12 `[cuda]` | PASS, install and driver/device smoke | `linux-b200-node14/cuda-extra-py312.log` |
| Python 3.11/3.12 CUDA resolver on macOS | BLOCKED by platform | `final-cuda-extra-py311.log`, `final-cuda-extra-py312.log` |
| Python 3.11/3.12 manylinux CUDA dependency resolution | PASS (metadata/download only) | `cuda-linux-py311.log`, `cuda-linux-py312.log`, `cuda-linux-py311-dryrun.log`, `cuda-linux-py312-dryrun.log` |

The CUDA target resolution found `cuda-python 12.9.7`, matching
`cuda-bindings 12.9.7` wheels for both CPython versions. The original macOS
result remains retained as negative platform evidence; the supported Linux
installation and runtime result is recorded separately under
`linux-b200-node14/`.

## Linux/B200 CUDA Gate

- Host: Ubuntu 24.04.4 LTS x86_64, NVIDIA driver `590.48.01`, reported CUDA
  `13.1`, and eight visible NVIDIA B200 devices.
- Interpreters: CPython 3.11.15 and CPython 3.12.3.
- Artifact: the exact final wheel above; both remote installs revalidated SHA256
  `57058acb41622f7119da2389d96c9345df0523f5f355a83774a9559d9b2e949a`.
- Both `[cuda]` installs selected `cuda-python 12.9.7`, `cuda-bindings 12.9.7`,
  and `cuda-pathfinder 1.6.0`; native `pip check` reported no broken
  requirements and complete `pip freeze` output is retained.
- `cuInit(0)` returned `CUDA_SUCCESS`; `cuDeviceGetCount()` returned eight; all
  eight driver-reported device names were `NVIDIA B200`.
- `cuda-pathfinder` initializes the top-level `cuda` namespace at interpreter
  startup. A before/after module comparison confirmed that importing
  `tilefoundry_costmodel` loaded no additional CUDA or OR-Tools module and did
  not load the legacy namespace.

See `linux-b200-node14/host-gpu-environment.txt`, the two CUDA logs, per-Python
`pip-check`/`pip-freeze` files, and the nested checksum manifest.

## Static And Formal Gates

- Ruff check and format: PASS.
- strict mypy: PASS, 31 source files.
- Test-extra workflow: 25 passed, 6 expected CP-SAT skips.
- Combined CP-SAT/test workflow: 31 passed on both Python versions.
- Seven Draft 2020-12 schemas: PASS; source and wheel schema bytes match.
- Root import/lazy optional-dependency boundary: PASS on both Python versions.
- CLI `--version`/`--help`: PASS on both Python versions.
- Selected pre-commit hooks and direct formal linters: PASS; no touched C++ or
  CUDA files. See `pre-commit-m0.log` and `formal-gate-checks-final.txt`.
- `pip check` and `pip freeze` are retained for every final environment.

## Golden Reference Review

The milestone's named Golden Reference is exercised by the smallest retained
real workflows:

- `costmodel/tests/test_solver.py` imports only the `(0, 2)` legacy namespace
  and retains observable finite-stage statuses, makespan, precedence, capacity,
  resource-demand, placement, per-group timing, picosecond rounding, and empty
  pipeline behavior. The base and CP-SAT install logs prove both optional
  branches through installed wheels.
- `costmodel/tests/test_api.py` exercises the `(2, 0)` root, strict owned
  serializers, recursive immutability, schema/decoder agreement, shared and
  duplicate warp-role behavior, and strict search-problem reconstruction.
- The clean base, CP-SAT, test, and Linux/B200 CUDA installs exercise the package
  and optional dependency boundaries without relying on the source checkout.

The added regression cases each close a demonstrated reachability gap from M0
review: wrong serializer ownership, nested mutability, request/schema mismatch,
warp-role uniqueness, ASCII identifiers, incomplete nested records, or lost
legacy result behavior.

## Test And Comment Review

Touched tests and comments were reviewed for redundancy. No test inspects a
private helper, AST, source text, or solver-internal name. Assertions that read
schema definitions target the shipped public schemas themselves. The retained
legacy assertions cover public result fields that earlier review found had lost
regression protection. Comments describe only local lazy-import, immutability,
or future-milestone boundary mechanics; no typed lowering, B200 fact,
profiling, or new-solver behavior is implemented.

## Acceptance Decision

- AC-0-4: PASS. Base, CP-SAT, CUDA, and test extras install within their
  documented boundaries on both supported Python versions; CUDA was exercised
  on Linux with visible B200 devices.
- Golden Reference policy: PASS. The source and observable functional points
  are named before the implementation plan.
- Gate-evidence policy: PASS. The installed-wheel and retained workflows above
  name all retained evidence and exercise every Golden Reference point.
- Test/comment policy: PASS. The review outcome and reachability rationale are
  recorded above.

M0 is closed by this acceptance record. M1 and later milestones remain out of
scope.

All command output files and their hashes are listed in `SHA256SUMS.txt`.
