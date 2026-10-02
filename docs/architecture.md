# Architecture

[README](../README.md) · [Contract](contract.md) · [Apple / MLX](apple-mlx.md) · [Dispatch](dispatch.md)

SILKern localizes indices at the boundary between a sparse selector and a paged
attention consumer. Its architecture keeps the mathematical contract independent
of the execution framework, and keeps each backend's memory model explicit.

![Tensor flow, stable scatter, page-table routing, and selected KV cache entries](../assets/fig-platforms.svg)

**Figure 1. Localization at the sparse-attention boundary.** (a) Rank 0 keeps three tokens, translates their addresses, and scatters them to destinations derived from the validity prefix. Color tracks each survivor; output retains selector order. (b) Logical pages 0, 1, and 2 map to physical pages 11, 2, and 7. Paired grids represent aligned K/V pages; highlighted offsets 4, 1, and 3 select slots 708, 129, and 451. The example uses two ranks, interleave 1, page size 64, and request 0.

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
