# TileFoundry Cost Model M1 Gate Record

Run ID: `20260807T081907Z`

Status: **PASS - M1 CLOSED**

M0 remains frozen. The M0 evidence directory and its wheel were not modified;
the M1 artifact below is a separate wheel and evidence record. This run covers
typed programs, workload frontend boundaries, exact B200 facts, and their
strict Python/JSON ownership. No operation lowering, profile store, CUDA JIT
runner, build implementation, or new solver was added.

## Artifact

- Wheel: `wheel/tilefoundry_costmodel-2.0.0-py3-none-any.whl`
- SHA256: `f295d99dd75c7b440289bbef6406094504b06384bc083a57caef6212a93d1b0b`
- M0 frozen wheel SHA256 remains
  `57058acb41622f7119da2389d96c9345df0523f5f355a83774a9559d9b2e949a`.
- The M1 root `tilefoundry_costmodel/__init__.py` is byte-identical to the
  frozen M0 root boundary; low-level typed records remain owned by their
  `model`, `program`, `hardware`, and `workloads` modules.

## Local Matrix

| Environment | Result | Evidence |
| --- | --- | --- |
| Python 3.11 base wheel | PASS; `pip check`, CLI, isolated root import | `py311-base.log` |
| Python 3.12 base wheel | PASS; `pip check`, CLI, isolated root import | `py312-base.log` |
| Python 3.11 `[cpsat,test]` wheel | PASS; 45 tests | `py311-cpsat-test.log` |
| Python 3.12 `[cpsat,test]` wheel | PASS; 45 tests | `py312-cpsat-test.log` |
| Wheel metadata and contents | PASS; empty base dependencies, six optional requirements | `artifact-verification.log`, `wheel-metadata.txt`, `wheel-contents.txt` |

All installed-wheel test runs report a `site-packages` module path. Importing
the root loaded no new CUDA, OR-Tools, or legacy modules. The base dependency
list is empty; optional requirements remain `cpsat`, `cuda`, and `test` with
the metadata in the wheel.

## Linux/B200 Matrix

Host evidence is in `linux-b200-node14/host-gpu-environment.txt`: Ubuntu
x86_64, driver `590.48.01`, eight visible B200 devices, and CPython 3.11.15 /
3.12.3. The remote wheel checksum in `linux-b200-node14/wheel.sha256` matches
the local artifact exactly.

| Environment | Result | Evidence |
| --- | --- | --- |
| Linux/B200 Python 3.11 `[cpsat,test]` | PASS; 45 tests and `pip check` | `linux-b200-node14/py311-cpsat-test.log` |
| Linux/B200 Python 3.12 `[cpsat,test]` | PASS; 45 tests and `pip check` | `linux-b200-node14/py312-cpsat-test.log` |
| Linux/B200 Python 3.11 `[cuda]` | PASS; `cuInit`, 8 devices, all `NVIDIA B200` | `linux-b200-node14/cuda-py311-r2.log` |
| Linux/B200 Python 3.12 `[cuda]` | PASS; `cuInit`, 8 devices, all `NVIDIA B200` | `linux-b200-node14/cuda-py312-r2.log` |

The CUDA rows validate installation and driver/device visibility only. They do
not claim that the M3 profiler or a CUDA operation provider exists.

## Contract Workflows

The named M1 Golden Reference is the cost-model specification sections 2--6
and the NVIDIA CUDA Programming Guide. The retained workflows exercise its
observable points:

- `tests/test_program.py` covers `T.*` and strict JSON canonical equality,
  discriminators, IDs and domains, GEMM/reduction axes, broadcasting,
  producer/value dependencies, aligned recurrence, endpoint single edges, and
  explicit loop-barrier DAG validation.
- `tests/test_hardware.py` resolves the exact B200 identity and checks all 14
  temporal and 5 static resources, units, capacities, provenance URLs and
  conditions. The CTA facts are 232448 opt-in shared-memory bytes, 262144
  tensor-memory bytes, 65536 32-bit registers, 32 CTA warps, and a conservative
  16-object mbarrier catalog policy; TMA and tensor inflight are explicit
  conservative one-slot scheduling bounds.
- `tests/test_api.py` covers shared warp IDs across different roles, duplicate
  same-role rejection in both decoder and schema, strict root ownership,
  immutable typed construction, and malformed sequence/domain exceptions.
- `tests/test_solver.py` remains the `(0, 2)` Golden Reference under
  `tilefoundry_costmodel.legacy` and retains its observable scheduling checks.

The calibration document is validated against `hardware-v1.schema.json` and
is byte-for-semantic parity with the installed B200 catalog. All seven source
schemas are Draft 2020-12-valid and byte-identical to their wheel copies; see
`verify-artifact.py` and `artifact-verification.log`.

## Static Gates

- Ruff check and format: PASS (`ruff-check.log`, `ruff-format.log`).
- strict mypy: PASS, 34 source files (`mypy-strict.log`).
- Forward-reference lint: PASS (`forward-reference-lint.log`).
- Source tests: PASS, 45 tests, branch coverage 66% (`source-tests.log`).
- `git diff --check`: PASS (`git-diff-check.log`).

The M1 plan's AC-1-1 through AC-1-7 and its three policy checks are checked in
`docs/plans/b200-cost-model.md`. M2 and M3 remain future work; specifically,
lowering/providers, concrete workload frontends, SQLite/profile resolution,
CUDA/JIT profiling, `build()`, finite/periodic solving, and end-to-end GEMM
are intentionally not implemented here.

The first CUDA smoke attempt is retained as `cuda-py311.log` and
`cuda-py312.log`; it reached `cuInit` and all devices but exposed fixed-width
NUL padding in the device-name API. The final `*-r2.log` records the corrected
normalization and is the acceptance evidence.
