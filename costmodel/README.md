# TileFoundry Cost Model

`tilefoundry-costmodel` is the standalone B200 single-CTA cost-model package.
The public contract is API `(2, 0)` and is defined exclusively by
`docs/spec/cost-model.md`.

## Version 2 workflow

Version 2 carries a typed `TileProgram` in a `CostModelRequest`, resolves an
exact profile snapshot at the build boundary, and then solves a replayable
`SearchProblem`:

```text
TileProgram -> build(request) -> SearchProblem -> solve(problem) -> CostModelResult
```

The stable command names are:

```text
tilefoundry-costmodel build --request REQUEST.json --profiles PROFILES.db --output PROBLEM.json
tilefoundry-costmodel solve --problem PROBLEM.json --output PLAN.json
tilefoundry-costmodel search --request REQUEST.json --profiles PROFILES.db --output PLAN.json
```

The base installation has no runtime dependencies. Optional capabilities are
explicit:

```bash
python -m pip install -e '/path/to/TileFoundry/costmodel[cpsat]'
python -m pip install -e '/path/to/TileFoundry/costmodel[cuda]'
python -m pip install -e '/path/to/TileFoundry/costmodel[test]'
```

Importing `tilefoundry_costmodel` does not import OR-Tools, CUDA Python, a CUDA
driver, or a GPU. Timing and solver capabilities are loaded only by the
milestones that implement them.

## M1 typed boundary

`tilefoundry_costmodel.T` constructs immutable `TileProgram` records. The
program constructor and `program_from_json()` share the same discriminator,
ID, domain, value-edge, relation, broadcasting, and loop-barrier validation.
`b200_hardware_catalog()` resolves only the exact `(hardware_id,
schema_version, calibration_id)` reference and exposes separate temporal and
static resource namespaces.

## M2 lowering boundary

M2 adds compiler-independent phase templates, explicit availability and buffer
lifetimes, canonical `TileOpProfileKey` identity, and the exact
`ConfigurationBuilder(*, implementations=...)` / `enumerate_templates(...)`
workflow. The in-memory `synthetic_implementation_catalog()` is metadata-only:
it never supplies production timings or executes CUDA. The public
`api.build()`/`solve()` orchestration and finite/periodic solvers remain
deferred to M4.

## M3 timing profiles

M3 owns immutable environment/policy/measurement records, the transactional
SQLite snapshot store, exact require/JIT resolution, and a local B200 CUDA
runner. CUDA Python is imported only inside `LocalCudaProfileRunner.run()`;
frozen snapshots remain usable on hosts without CUDA.

Snapshot commands are explicit:

```text
tilefoundry-costmodel profiles create --profiles PROFILES.db --snapshot ID --hardware B200.json
tilefoundry-costmodel profiles inspect --profiles PROFILES.db --snapshot ID@REV
tilefoundry-costmodel profiles freeze --profiles PROFILES.db --snapshot ID@REV
tilefoundry-costmodel profiles export --profiles PROFILES.db --snapshot ID@REV --output SNAPSHOT.json
tilefoundry-costmodel profiles import --profiles PROFILES.db SNAPSHOT.json
```

The installed real M3 provider is the minimal correctness-checked
`b200.copy` vertical workflow. The synthetic providers remain timing-free.

## Legacy migration

The fixed-stage API `(0, 2)` remains available only under
`tilefoundry_costmodel.legacy` while callers migrate:

```python
from tilefoundry_costmodel.legacy import (
    CpSatPipelineSolver,
    ListPipelineSolver,
    PipelineHardware,
    PipelineProblem,
    Precedence,
    ResourceDemand,
    ResourceSpec,
    StageSpec,
)
```

Legacy records and solver statuses retain their old nanosecond/resource
semantics. New integrations must import the version-2 root and check
`COST_MODEL_API_VERSION == (2, 0)`; root imports of `StageSpec` and the legacy
solver classes are intentionally unsupported.
