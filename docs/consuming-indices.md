# From localized indices to attention

[README](../README.md) · [Contract](contract.md) · [Apple / MLX](apple-mlx.md) · [Architecture](architecture.md)

Localization produces addresses and counts. A consumer still has to respect the
output layout, mask padding before it affects arithmetic, and keep page tables
consistent with the cache they describe. This guide connects those rules to
runnable gathers and selected attention.

![Localization feeds masked K/V gathers and a shared softmax normalization; count prefixes and column masks depend on the output layout](../assets/fig-consumer.svg)

## Choose the right validity mask

| Localization mode | Layout | Consumer mask |
|---|---|---|
| `dcp_size > 1`, `compact_valid_to_front=True` | Valid prefix followed by padding | `column < counts[row]` |
| `dcp_size == 1` | Original columns, even when compaction is requested | Preserve column validity; count alone is insufficient |
| `compact_valid_to_front=False` | Original columns for every rank | Preserve column validity; count alone is insufficient |

A single-rank counterexample runs with the base package:

```python
from silkern import localize_reference

out, counts = localize_reference(
    [0], [[3]], [[2, -1, 0, 99]],
    block_size=4, dcp_size=1, dcp_rank=0,
)
assert out == [[14, -1, 12, -1]]
assert counts == [2]
# out[0][:counts[0]] contains padding and misses the valid address 12.
```

The general contract accepts negative in-range physical page entries and counts
their translations. Such an output can equal `-1` while still being valid;
`out >= 0` is therefore not a general validity test. A real cache gather must
require populated nonnegative physical pages whose translated addresses are
inside that cache. Under that stricter cache precondition, `out >= 0` identifies
valid columns in a non-compacting layout. If an application preserves the wider
negative-page contract, derive its mask from the input ownership and table
bounds instead, and define how that application interprets negative mappings.

SILKern validates array metadata without reading page values back to the host.
Check allocation bounds when constructing or updating the cache and table;
do not rely on the launcher to discover an invalid physical page.

## Mask both addresses and values

For a compacted multi-rank result, `out` has shape `(batch, width)`, `counts` has
shape `(batch,)`, and a nonempty `cache` has shape `(physical_slots, value_dim)`:

```python
columns = mx.arange(out.shape[1], dtype=mx.int32)[None, :]
valid = columns < counts[:, None]
safe_slots = mx.where(valid, out, 0)
gathered = mx.take(cache, safe_slots, axis=0)
gathered = mx.where(valid[..., None], gathered, 0)
```

Substituting address zero makes every padded gather address legal. Masking the
gathered values also matters: the unused slot can contain NaN, and multiplying
NaN by a zero attention weight still yields NaN. Keep the mask with the gathered
tensor so padded keys cannot contribute to logits or normalization either.
Queries and selected cache values used by the attention example must be finite,
and their dot products and weighted sums must remain representable in float32.

The smaller [sparse-gather example](../examples/mlx_sparse_gather.py) demonstrates
this sequence and verifies a recombined sum:

```bash
python -m examples.mlx_sparse_gather
```

## Run a complete selected-attention consumer

The [selected-attention example](../examples/mlx_sparse_attention.py) supplies
fragmented K/V caches, localizes the selection across logical shards, gathers
the selected values, and checks its output against unsharded attention:

```bash
python -m examples.mlx_sparse_attention --backend mlx --device cpu
python -m examples.mlx_sparse_attention --backend metal --device gpu
```

Its educational `selected_attention` helper accepts one float32 query per row,
request IDs, token selections, and a sequence of `(page_table, keys, values)`
shards. It returns the attention result and each shard's counts. The number of
shards defines the ownership degree. `block_size` and `dcp_interleave` must match
the page-table layout and the cache's token ownership. This checkout example is
not an additional public API shipped in the library wheel.

The helper derives single-rank validity from request and token bounds rather
than inferring it from the count. It also masks gathered keys and values before
the attention arithmetic, so an unused placeholder slot can contain NaN safely.

Three consumer rules matter beyond a gather:

1. Replace padded logits with negative infinity before finding maxima and
   computing exponentials. Preserve repeated selected tokens as repeated terms.
2. Normalize across the complete selected set. Recombine unnormalized weighted
   values and denominators using a shared maximum; averaging independently
   normalized shard outputs gives the wrong result when shard weights differ.
3. Define the all-invalid case. The example returns zero output and zero counts,
   avoiding both `-inf - -inf` and division by zero. An empty shard of a nonempty
   selection contributes zero weight while other shards determine the result.

The [consumer tests](../tests/test_mlx_integration.py) exercise this actual helper
against an independent scalar reference, including single-rank holes, empty
selections, duplicated tokens, grouped interleave, and poisoned padding.
All logical shards run on one device. This verifies consumer wiring; it does
not implement distributed collectives, an MLX-LM adapter, a model-serving
runtime, or a serving performance benchmark.

For a bounded timing of this complete consumer, run
`python -m bench.bench_mlx_attention`. Both implementations are compiled and
checked against independent attention before timing. See the
[two recorded cases and measurement scope](apple-mlx.md#complete-selected-attention-consumer).

## Keep the cache and execution lifetimes aligned

The request ID, table, and selection must describe the same cache state. Arrange
cache/page updates before localization and consumer reads. For CUDA, also order
consumer reads before the next overwrite of caller-owned outputs or workspace;
use separate writable buffers for independent concurrent work. Captured graph
replay bypasses Python binding checks. The
[vLLM lifetime rules](integration-vllm.md#buffer-lifetime-and-concurrent-consumers)
explain stream ordering and captured-storage ownership.

MLX returns new lazy arrays. Keep localization and consumption in the intended
stream context and evaluate the resulting attention output when a host result
or a synchronized timing sample is needed. The [Apple guide](apple-mlx.md)
explains streams, compilation, and what the recorded timings include.
