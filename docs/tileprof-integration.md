# TileProf Integration Boundary

TileProf owns benchmark kernels, correctness checks, measurements, provenance,
record storage, and hardware calibration. TileFoundry owns semantic operation
signatures, numeric scheduling problems, CP-SAT search, synchronization export,
and verification.

The integration point is a read-only cost adapter:

```text
TileFoundry OperationSignature
        -> exact TileProf semantic identity
        -> measured issue/completion/resource record
        -> OperationCost
```

Matching must use a versioned canonical semantic payload and its full digest.
It must not infer identity from operation IDs, kernel names, truncated keys,
shapes alone, implementation labels, or residency labels.

One accepted record must provide:

- the exact semantic identity or an immutable manifest mapping to it;
- selected statistic and unit;
- issue duration and completion latency;
- versioned resource-window mapping;
- hardware and software provenance;
- correctness and measurement status.

Absolute timing values are independently rounded up when converted to integer
solver units. Sensitivity variants become separate closed problems and solves;
uncertainty is not hidden inside one duration.

Until the adapter is implemented, TileFoundry accepts authored programs only
with explicitly selected fixture costs and accepts production-style inputs as
already closed `WarpgroupProblem` documents.
