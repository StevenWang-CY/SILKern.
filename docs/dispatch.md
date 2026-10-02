# Choosing a backend and algorithm

[README](../README.md) · [Architecture](architecture.md) · [Apple / MLX](apple-mlx.md) · [Evidence](evidence.md)

Choose the execution framework first, then measure the implementation in its
actual consumer. SILKern does not dispatch CUDA algorithms from context length,
and historical CUDA thresholds are not Apple tuning rules.

## Backend selection

| Environment | Entry point | Starting choice |
|---|---|---|
| Contract debugging / portable reference | `localize_reference` | Python oracle |
| Apple silicon / MLX application | `localize_mlx` | `backend="auto"` |
| CUDA / Triton application | `localize_rowwise` or `localize_hierarchical` | Measure rowwise first, then hierarchical in the consumer |
| Qualified vLLM CUDA integration | `WorkspaceAdapter` | Choose the arm before `prepare()` and capture |

The MLX `auto` choice uses Metal on a Metal GPU stream and composition on a CPU
stream. An explicit `metal` choice requires a Metal GPU; `mlx` selects framework
array operations. See [Apple execution](apple-mlx.md#execution-and-streams).

## CUDA algorithms

### Rowwise

`localize_rowwise` assigns one program to a selection row. It computes mapping
and validity, then derives output positions with a row-wide prefix sum. There
is one kernel launch, no cross-program reservation, and no caller scratch.

Larger rows increase the per-program working set. The maximum supported width
is 4096, and a low converter-only latency does not predict every downstream
interaction.

### Hierarchical

`localize_hierarchical` decomposes compacting work into four launches:

1. Map tiles and compute tile-local positions and counts.
2. Compute deterministic exclusive prefixes over tile counts.
3. Fill the output with `-1`.
4. Scatter each survivor to `tile_offset + local_position`.

Allocate workspace once before capture:

```python
import torch
from silkern import localize_hierarchical, workspace_shapes

# req_ids, block_table, token_indices, out, and counts are existing CUDA tensors.
batch, width = token_indices.shape
shapes = workspace_shapes(batch, width, tile_size=128)
workspace = {
    name: torch.empty(shape, dtype=torch.int32, device=token_indices.device)
    for name, shape in shapes.items()
}
localize_hierarchical(
    req_ids, block_table, token_indices, out, counts,
    workspace["mapped"], workspace["local_positions"],
    workspace["tile_counts"], workspace["tile_offsets"],
    block_size=64, dcp_size=2, dcp_rank=0, tile_size=128,
)
```

Supported tile sizes are 64, 128, and 256; the default is 128. A larger tile
reduces the number of tile counts but increases each program's working set.
The hierarchical path still caps total row width at 4096. It bounds per-program
work; it does not lift the supported width limit.

## Historical measurements

Archived converter-segment graph medians on two H100s, in microseconds for a
**complete 48-layer segment**, rounded from
[`segments.json`](../evidence/05-full-decode-canary/segments.json):

| Implementation | 32K context | 64K context |
|---|---:|---:|
| Rowwise | 119.996 | 121.378 |
| Atomic baseline | 194.393 | 194.423 |
| Hierarchical | 239.727 | 240.170 |

Complete decode-step ratios against atomic, with 98.75% intervals from five
sessions in [`analysis.json`](../evidence/05-full-decode-canary/analysis.json):

| Implementation | 32K context | 64K context |
|---|---|---|
| Rowwise | 1.000031 [0.997373, 1.002696] | 1.014006 [1.010326, 1.017700] |
| Hierarchical | 1.000580 [0.997800, 1.003368] | 1.002178 [0.995750, 1.008648] |

Rowwise wins the converter-only comparison but regresses by about 1.4% in the
64K complete-step experiment. Hierarchical satisfies the study's prespecified
1.01 non-inferiority margin at both contexts. That supports choosing hierarchical
for that 64K workload, not a universal switch at 32K.

The [mechanism decomposition](../evidence/04-mechanism-decomposition/) does not
resolve a consistent order-only cost. Cache state, scheduling, and inter-kernel
resource effects are possible explanations for downstream behavior; the
measurements do not establish a unique cause.

## Reproduce a decision

On separately authorized CUDA hardware, validate and then measure:

```bash
python -m silkern --require-device
python -m bench.bench_converter --with-atomic --width 2048 --dcp-size 2
```

Row width is the selector's top-k; context length is the size of the attended
sequence. They are different axes. Preserve both, along with batch size, page
layout, runtime versions, and consumer implementation, when comparing results.
The standalone benchmark cannot replace a complete-consumer measurement.

On Apple silicon, use `python -m silkern.mlx_verify --require-device` and
`python -m bench.bench_mlx`. Apple timing uses a different memory and execution
model, so cross-platform numbers require independently matched workloads and
timing methodology.
