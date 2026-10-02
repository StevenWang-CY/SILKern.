# The localization contract

[README](../README.md) · [Architecture](architecture.md) · [Apple / MLX](apple-mlx.md) · [Determinism](determinism.md)

[`localize_reference`](../silkern/contract.py) defines the result for supported
inputs. Accelerator implementations must match its arrays and counts exactly on
their shared valid-input domain. Memory ownership and malformed-request behavior
are backend-specific and documented below.

## Inputs and outputs

| Input | Shape | Meaning |
|---|---|---|
| `req_ids` | `(batch,)` | Request owning each selection row |
| `block_table` | `(requests, table_width)` | Logical page → physical page for each request |
| `rows` / `token_indices` | `(batch, width)` | Global logical token positions, in selector order |
| `block_size` | Scalar | Positive page size in tokens |
| `dcp_size`, `dcp_rank` | Scalars | Parallel degree and rank, with `0 <= rank < size` |
| `dcp_interleave` | Scalar, default `1` | Positive ownership granularity; divides `block_size` |
| `compact_valid_to_front` | Boolean, default `True` | Enable stable front compaction when `dcp_size > 1` |

| Output | Shape | Meaning |
|---|---|---|
| `out` | `(batch, width)` | Translated physical slots, with `-1` for invalid/padded positions |
| `counts` | `(batch,)` | Exact number of valid mappings in each row |

Sequences and arrays must be nonempty and rectangular. Inputs are integers;
booleans and floating-point values are not accepted as integer data or geometry.
Accelerators use signed `int32` arrays and support widths through
`MAX_ROW_WIDTH = 4096`. The Python oracle uses Python integer arithmetic and
serves as the specification beyond that accelerator storage limit.

## Five stages, one stable result

![Worked localization coordinates, stable destinations, and compacted versus column-preserving output layouts](../assets/fig-contract.svg)

**Figure 2. A worked localization row.** Columns 0, 2, and 5 translate to physical slots 708, 129, and 451. Prefix-derived destinations preserve their input order and leave a padded tail. The right panel gives the general coordinate equations and compares compacted and column-preserving layouts: both have count 3, but only one has a valid three-element prefix.

For each input token, in its original column order:

**1. Route the request.** `req_ids[row]` chooses the page-table row. Different
selection rows may use different, unrelated tables.

**2. Filter ownership.** A negative token is invalid. For a nonnegative token:

```text
owner = (token // dcp_interleave) % dcp_size
```

Keep only tokens whose owner is `dcp_rank`.

**3. Deinterleave.** Convert the global position to the rank's local position:

```text
local = (token // (dcp_size * dcp_interleave)) * dcp_interleave
        + token % dcp_interleave
```

**4. Translate through the page table.**

```text
logical_block, offset = divmod(local, block_size)
physical = block_table[request][logical_block] * block_size + offset
```

A logical block outside the table is invalid. An in-range table entry is used
verbatim, including negative entries. A mapping's validity comes from ownership
and bounds, not from the sign of the resulting physical value.

**5. Preserve order.** When compaction is active, valid mappings occupy the front
prefix in their original relative order. The remainder is `-1`. The count is
the number of valid mappings. With compaction disabled, mapped and invalid
values remain in their original columns.

## Edge cases

| Case | Required behavior |
|---|---|
| `dcp_size == 1` | Bypass compaction even if requested; preserve columns |
| `compact_valid_to_front=False` | Preserve columns and return the exact count |
| Duplicate selected tokens | Preserve each occurrence and its relative position |
| All-invalid row | Fill the row with `-1`, count zero |
| Token beyond table coverage | Map to `-1`; do not clamp it into the last page |
| Negative in-range physical page | Translate verbatim and count the mapping |
| Fragmented/non-monotonic page table | Translate each logical page independently |
| Partial final tile | Preserve the same result as a whole-row scan |

In a non-compacting layout, `out[:count]` is not a valid-prefix view. Even in a
compacting layout, negative page entries mean `out >= 0` is not the validity
mask. Well-formed KV-cache integrations normally provide populated nonnegative
physical page IDs; the oracle preserves the historical converter's wider
semantics.
The [consumer guide](consuming-indices.md) turns these layout rules into safe
KV gathers and selected attention, including a single-rank counterexample.

## Integer and bounds requirements

CUDA and MLX physical-slot arithmetic must fit signed 32-bit storage:
`physical_page * block_size + offset` must be in `[-2**31, 2**31 - 1]` for every
selected in-range entry. Overflow is outside the supported accelerator domain;
Python's unbounded integer result does not make overflowing device arithmetic
valid. Avoid host synchronization in a hot launcher by validating page allocation
bounds when constructing the cache.

Both CUDA and MLX validate scalar geometry against the signed 32-bit domain,
including `dcp_size * dcp_interleave`. Hierarchical workspace sizing also rejects
`batch * width > 2**31 - 1` to keep flattened fill indexing representable.
The CUDA rowwise launcher requires `batch <= 2**31 - 1`, matching the signed
grid-x range; adapter preparation rejects an oversized batch before allocation.
Data-dependent physical-address bounds remain caller responsibilities.

Request IDs have an explicit backend distinction:

| Backend | Out-of-range request ID |
|---|---|
| Python oracle | Raises `LocalizationError` |
| CUDA / Triton | Unsupported caller input; upstream must guarantee `0 <= id < requests` |
| MLX / Metal or composition | Safely produces an all-`-1` row with count zero |

CUDA request IDs reside on the device, so host validation would require
synchronization. The current CUDA kernels require valid IDs and do not provide
the MLX masking behavior. Validate routing upstream; do not silently clamp an
invalid ID to another request's table.

## Execution guarantees

| Property | CUDA | MLX |
|---|---|---|
| Array dtype | Contiguous `torch.int32` on one CUDA device | `mx.int32`; strided inputs supported |
| Output ownership | Caller supplies disjoint output buffers | Function returns new arrays |
| Scratch ownership | Caller supplies hierarchical workspace | MLX manages intermediates |
| Input mutation | Inputs unchanged | Inputs unchanged |
| Allocation policy | Launchers do not allocate device buffers | Allocations permitted; lazy execution |
| Capture model | Designed for fixed-address CUDA graph replay | No equivalent pointer/capture guarantee |

CUDA metadata validation rejects wrong shapes/dtypes/devices, unsupported
geometry, storage aliasing, and incompatible workspace. The integration adapter
also rejects unregistered bindings. MLX validates metadata before constructing
its computation. Neither API silently changes the mathematical layout to handle
an unsupported geometry.

CUDA tensors must also have resolved value metadata. Triton reads raw memory,
so a PyTorch lazy-negation view is rejected even when its dtype and strides
match. Materialize it with
[`tensor.resolve_neg()`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.resolve_neg.html)
before registering buffers or
capturing a graph. Overlapping writable byte ranges are rejected even when
different storage objects, such as separate DLPack imports, refer to the same
physical allocation.

CUDA launches are asynchronous. Order input updates, localization, consumer
reads, and subsequent buffer reuse on the same stream or through CUDA events.
Concurrent work must use disjoint writable output and scratch buffers. Captured
CUDA graphs replay without Python validation: retain their storage and do not
resize or rebind captured tensors until the graph is no longer used.

## Order is observable output

Set equality and count equality are insufficient. The compacted result must
match the input sequence filtered by ownership and table bounds, including
repeated values. CUDA conformance checks compare with an independently derived
order as well as the oracle, and verify guard regions and replay stability.
Apple checks exercise its own backend paths. Consult each report for the checks
actually run; a host-only test does not establish device correctness.
