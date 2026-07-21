# TileFoundry Cost Model

Compiler-independent finite software-pipeline scheduling API.

The caller lowers a local work graph into:

- `StageSpec`: finite stage instances and per-resource capacity demands.
- `ResourceDemand`: named resource slots occupied for the full stage duration.
- `Precedence`: dependency constraints.
- `PipelineHardware`: named resource capacities.
- `TimingOracle`: stage durations.

Both solvers consume the same contract:

- `ListPipelineSolver`: deterministic feasible schedule.
- `CpSatPipelineSolver`: optional OR-Tools makespan minimization.

Install from a TileFoundry checkout:

```bash
python -m pip install '/path/to/TileFoundry/costmodel[cpsat]'
```

Editable development install:

```bash
python -m pip install -e '/path/to/TileFoundry/costmodel[cpsat]'
```

The package exports `COST_MODEL_API_VERSION`. Integrations must check this API
version rather than infer compatibility from solver class names.

`StageSpec(resources=("pipe",))` is the demand-one compatibility form. Use
`StageSpec(resource_demands=(ResourceDemand("pipe", 2),))` when a stage occupies
multiple slots of a resource whose `ResourceSpec.capacity` is greater than one.
