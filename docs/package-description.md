# SILKern

**Deterministic sparse-index localization for Apple silicon / MLX and CUDA / Triton.**

Sparse attention selects global token positions. A context-parallel rank needs
physical slots in its own paged KV cache. SILKern translates those indices,
filters ownership, and preserves the selector's relative order and duplicates.
It returns exact valid counts with a documented output layout.

[Get started and see examples](https://github.com/StevenWang-CY/SILKern./blob/main/README.md)
· [API contract](https://github.com/StevenWang-CY/SILKern./blob/main/docs/contract.md)
· [Performance evidence](https://github.com/StevenWang-CY/SILKern./blob/main/docs/evidence.md)

| Backend | Entry point | Memory model |
|---|---|---|
| Python 3.11+, no accelerator dependencies | `localize_reference` | New Python lists; readable oracle |
| Apple silicon / MLX | `localize_mlx` | New lazy arrays; Metal kernel or compositional MLX |
| CUDA / Triton | `localize_rowwise`, `localize_hierarchical` | Caller-owned output and workspace buffers |

The base installation has no runtime dependencies. The `mlx` extra installs
MLX on native arm64 macOS; the `gpu` extra installs PyTorch and Triton.
See [Apple setup](https://github.com/StevenWang-CY/SILKern./blob/main/docs/apple-mlx.md)
and [CUDA integration](https://github.com/StevenWang-CY/SILKern./blob/main/docs/integration-vllm.md)
for platform requirements and lifecycle rules.

```python
from silkern import localize_reference

out, counts = localize_reference(
    req_ids=[0],
    block_table=[[11, 2, 7, 5]],
    rows=[[8, 5, 130, 7, -1, 262]],
    block_size=64, dcp_size=2, dcp_rank=0,
)
assert out == [[708, 129, 451, -1, -1, -1]]
assert counts == [3]
```

Accelerator arrays use signed `int32`, with row width at most 4096. Compaction
is bypassed for a single rank, and valid negative page entries are counted;
read the contract before using localized values as cache addresses.

SILKern is a localization primitive. It does not select tokens, manage a KV
cache, install an MLX-LM adapter, or provide a distributed serving runtime.
Measurements and qualification limits are linked above; a primitive speedup
does not establish model-serving throughput.

Apache-2.0 · [Source](https://github.com/StevenWang-CY/SILKern.)
· [Issues](https://github.com/StevenWang-CY/SILKern./issues)
· [Changelog](https://github.com/StevenWang-CY/SILKern./blob/main/CHANGELOG.md)
