# Architecture

[README](../README.md) · [Contract](contract.md) · [Apple / MLX](apple-mlx.md) · [Dispatch](dispatch.md)

SILKern localizes indices at the boundary between a sparse selector and a paged
attention consumer. Its architecture keeps the mathematical contract independent
of the execution framework, and keeps each backend's memory model explicit.

<picture>
  <source media="(max-width: 767px)" srcset="../assets/fig-system-narrow.svg">
  <img src="../assets/fig-system.svg" width="100%" alt="A sparse selector turns query q into one row of global positions and sends the same row to two ranks. On each rank SILKern, given the page table P and its rank number, returns physical KV-cache slots in selector order: rank 0 returns 708, 129, 451 with count 3, and rank 1 returns 706, 707 with count 2. Each rank's sparse attention gathers those slots from its own paged KV cache, and the partial results are combined across ranks.">
</picture>

**Where SILKern sits in a decode step.** A sparse selector chooses global positions once per query, and every rank receives the same row. Each rank calls SILKern with its own rank number and the request's page table `P`, and gets back the slots it stores, in selector order, with their count. Rank 0 keeps tokens 8, 130, and 262; rank 1 keeps 5 and 7. In this example both ranks read the page table `[11, 2, 7, 5]`, and each slot addresses the calling rank's own memory, so rank 1's slots 706 and 707 lie in physical block 11 of rank 1's cache. The selector, the attention kernel, and the cross-rank combination are the consumer's; SILKern's stage is highlighted.

## Layers and responsibilities

| Layer | Responsibility | Dependency boundary |
|---|---|---|
| [`contract.py`](../silkern/contract.py) | Geometry validation and readable normative oracle | Python only |
| [`workspace.py`](../silkern/workspace.py) | CUDA workspace shapes without allocation | Python only |
| [`kernels.py`](../silkern/kernels.py) | In-place rowwise and hierarchical CUDA kernels | Optional PyTorch + Triton |
| [`mlx.py`](../silkern/mlx.py) | Functional MLX API; Metal and array-operation implementations | Optional MLX |
| [`verify.py`](../silkern/verify.py) | CUDA oracle, order, guard, and graph checks | CUDA required to execute |
| [`mlx_verify.py`](../silkern/mlx_verify.py) | Apple oracle and repeatability checks | MLX required to execute |
| [`integrations/vllm.py`](../silkern/integrations/vllm.py) | Qualified upstream adapter and fixed bindings | CUDA integration |
| [`bench/`](../bench/) | Developer benchmarks with explicit timing semantics | Checkout-only tools |
| [`evidence/`](../evidence/) | Measurement artifacts and provenance | Data, not runtime dependencies |

Importing the base package works without an accelerator runtime. PyTorch and
MLX are imported when their backend is used. The CUDA module attempts a guarded
Triton import at package import time to define its kernels when Triton is
installed; an absent Triton installation is tolerated. A contract-only
installation therefore remains useful without CUDA or MLX, although installing
Triton can add import-time work.

## Stable destinations

<picture>
  <source media="(max-width: 767px)" srcset="../assets/fig-localization-narrow.svg">
  <img src="../assets/fig-localization.svg" width="100%" alt="Three panels follow one selection row on rank 0: six global positions are filtered to three owned tokens and deinterleaved; their logical pages pass through the page table to physical KV blocks 11, 2, and 7, giving slots 708, 129, and 451; stable prefix compaction then writes 708, 129, 451 followed by padding, with count three.">
</picture>

**From global positions to stable cache addresses.** One selection row on rank 0 of two ranks, with interleave 1, page size 64, and request 0. (a) The owner `o` of each selected global position `t`: rank 0 keeps tokens 8, 130, and 262 and deinterleaves them to local positions 4, 65, and 131. (b) Each local position lies in a logical page at an offset; the request's page table sends pages 0, 1, and 2 to physical KV blocks 11, 2, and 7, giving slots 708, 129, and 451. Crossing lines show that physical order need not follow logical order. (c) Inclusive validity prefixes minus one give destinations 0, 1, and 2, so the output keeps selector order, pads with −1, and reports count 3. Color follows each surviving token. K/V gathering remains a consumer operation.

For a row validity mask `valid`, each surviving element has a deterministic
zero-based destination:

```text
destination[j] = sum(valid[0:j+1]) - 1
```

Invalid elements never write to a survivor's destination. Duplicated selected
tokens remain duplicated: this is a stable filter and translation, not set
construction. The exact count is `sum(valid)`.

The CUDA rowwise implementation scans a whole row in one program. The
hierarchical implementation scans bounded tiles and then scans tile counts;
`tile_offset + local_position` reconstructs the same destination. The Metal
implementation assigns 256 threads to a row and gives each thread a contiguous
chunk. SIMD scans connect each chunk's survivor count, then eight group totals
connect the row prefix. A barrier makes every group total visible before prefix
reads and separates tail initialization from scatter.
The MLX composition uses a validity prefix sum and unique scatter destinations
for the valid prefix and invalid tail; it does not sort the selected values.

<picture>
  <source media="(max-width: 767px)" srcset="../assets/fig-scan-narrow.svg">
  <img src="../assets/fig-scan.svg" width="100%" alt="Three panels number the survivors of one eight-column row with validity 1, 0, 1, 1, 0, 1, 0, 1. Panel a, rowwise and MLX: cumsum of v minus v over the whole row. Panel b, hierarchical: positions within each tile of four, tile counts 3 and 2, then tile offsets 0 and 3 are added. Panel c, Metal: each of four threads counts its two-column chunk, giving 1, 2, 1, 1; a SIMD prefix and group totals give thread offsets 0, 1, 3, 4; each thread writes its chunk in order. All three give destinations 0, 1, 2, 3, 4 at columns 0, 2, 3, 5, and 7.">
</picture>

**Three schedules, one numbering.** Each panel numbers the same row's survivors, and all three give identical destinations `d`. (a) The rowwise kernel and the MLX composition (`backend="mlx"`) take one prefix sum over the whole row. (b) The hierarchical kernel numbers survivors within each tile, here four columns wide (128 by default), counts each tile, and adds each tile's exclusive offset. (c) The Metal kernel gives each thread a contiguous chunk, here two columns (`ceil(width / 256)` in the kernel); a SIMD prefix over the chunk counts and the totals of the earlier SIMD groups give each thread its first destination, here for two groups of two threads (eight groups of 32 in the kernel). No schedule reserves output slots in completion order, so none can reorder survivors between calls.

The non-compacting path and the `dcp_size=1` bypass preserve input column
positions. Those paths still compute the exact count, but the first `count`
columns are not necessarily the valid entries.

## Memory ownership is part of the API

| Property | Python oracle | CUDA / Triton | MLX / Metal |
|---|---|---|---|
| Inputs | Integer sequences | Contiguous CUDA `torch.int32` tensors | `mx.int32` arrays |
| Outputs | Newly created lists | Caller-provided buffers, written in place | Newly created lazy arrays |
| Scratch | Python allocations | Caller-owned hierarchical workspace | Managed by MLX |
| Host evaluation | Immediate | Asynchronous device launch | Lazy; `mx.eval` realizes results |
| Capture guarantee | None | Fixed-buffer CUDA graph design | No CUDA-style allocation or pointer guarantee |
| Invalid request ID | Raises | Caller precondition | Row becomes `-1`, count becomes zero |

A shared mathematical contract does not imply interchangeable call signatures.
An MLX adapter must consume returned arrays and respect MLX evaluation; routing
it through the CUDA workspace adapter would misrepresent ownership.
MLX accepts explicit devices, ordinary streams, and thread-local stream handles.
Independent workers can build and evaluate their own graphs using one
`mx.ThreadLocalStream` handle that resolves separately in each thread; see the
[worker recipe](apple-mlx.md#independent-worker-threads).

## Integration boundaries

SILKern receives a page table and assumes the caller owns its lifecycle. Physical
indices are meaningful only for the KV cache that table describes. The caller
must preserve page-table/selection consistency and guarantee that translated
physical addresses fit signed 32-bit storage.

The vLLM adapter registers a specific tensor binding and qualified geometry
before capture. It rejects drift because reallocation or rebinding would change
the consumer's assumptions. It is a narrowly scoped integration, not a generic
hook for every vLLM version. The Apple API is a localization building block;
this repository does not install an MLX-LM attention or distributed-runtime
adapter.

CUDA output and scratch buffers are reusable mutable storage. A consumer must
finish reading a result before another launch or graph replay overwrites it;
use stream ordering or CUDA events, and separate buffers for independently
concurrent work. Captured graph replay bypasses Python validation, so captured
storage must not be resized or rebound for the graph's entire lifetime. See
[adapter lifetime and concurrency](integration-vllm.md#buffer-lifetime-and-concurrent-consumers).

## Verification follows the boundary

1. The pure-Python oracle defines output arrays, counts, and layout.
2. Backend tests compare whole outputs, including the padded tail.
3. Repeatability checks revisit identical inputs; CUDA additionally checks graph
   replay and guarded storage.
4. Benchmarks verify correctness before timing and name the exact timing unit.
5. Model and serving claims need independent integration evidence.

Historical NVIDIA measurements retain their original scope and implementation
provenance. New host validation or Apple measurements do not requalify modified
CUDA paths. See [evidence](evidence.md).
