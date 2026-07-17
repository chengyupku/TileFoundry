# TileFoundry Cost Model

Lightweight, compiler-independent software-pipeline scheduling API.

```python
from tilefoundry_costmodel import (
    ListPipelineSolver,
    PipelineHardware,
    PipelineProblem,
    ResourceSpec,
    StageSpec,
)

solution = ListPipelineSolver().solve(problem, timing_oracle, hardware)
```

The caller owns lowering into finite stage instances, precedence constraints,
resource domains, and buffer-reuse constraints. The timing oracle owns stage
durations. The solver does not import TileS2 or TileFoundry compiler IR.

Local editable install:

```bash
python -m pip install -e /path/to/tilefoundry/costmodel
```

`ListPipelineSolver` returns a feasible deterministic schedule; it does not
claim optimality. A later CP-SAT solver can implement the same contract.
