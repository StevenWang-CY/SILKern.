# Integrating with vLLM

[README](../README.md) · [Architecture](architecture.md) · [Contract](contract.md) · [Dispatch](dispatch.md)

This adapter is CUDA-only and targets a pinned upstream integration. The MLX
API has a separate ownership model; it is not a drop-in backend for this adapter.

[`silkern/integrations/vllm.py`](../silkern/integrations/vllm.py) replaces the
allocation-owning `triton_filter_and_convert_dcp_index` used by the sparse MLA
attention backend, without modifying any upstream file.

## Why an adapter is needed at all

The upstream converter owns output allocation. This adapter makes buffer ownership
and tensor bindings explicit before capture, so the consumer can retain stable
addresses and the integration can reject unexpected geometry changes. CUDA graph
support depends on the allocation/capture rules of the surrounding framework;
allocation by itself is not universally forbidden during capture.

So the adapter presents the same callable signature while keeping every output
and scratch buffer caller-owned and fixed-address. It is two-phase:

```python
from silkern.integrations.vllm import WorkspaceAdapter, install_converter

# Phase 1 — once, before warmup or capture.
adapter = WorkspaceAdapter("row_stable")          # or "hierarchical_stable"
adapter.prepare(
    req_id, block_table, token_indices,
    dcp_size=2, dcp_rank=rank,
    block_size=64, num_topk_tokens=2048,
)

# Phase 2 — eager calls and graph capture reuse the bound device buffers.
with install_converter(sparse_backend_module, adapter):
    ...  # warmup, capture, replay

# The original callable is restored on exit, including on exception.
```

`prepare()` is one-shot: construct a new adapter for a different binding rather
than preparing the same instance again. It retains its bound inputs so their
storage stays alive for the adapter lifetime.

`prepare()` allocates the output, counts, and (for the hierarchical arm) the four
workspace buffers, then records the address, shape, stride, dtype, and device of
every input, including its lazy-negation state. Unresolved negation views are
rejected; call `resolve_neg()` before registration and capture. Afterward:

- `adapter.output` / `adapter.counts` — the bound buffers, for wiring downstream
- `adapter.fixed_buffer_signatures` — what your own capture gate should assert
- `adapter.calls` — an invocation counter, useful for confirming the swap took

The selected `arm` and `hierarchical_tile_size` are read-only after construction.

## Buffer lifetime and concurrent consumers

Each adapter owns one reusable output/count pair and, for the hierarchical arm,
one reusable scratch set. Calls and graph replays that share an adapter must be
ordered on one CUDA stream, or connected by explicit CUDA events. Complete the
consumer's reads, or copy the result to separate storage, before the next
invocation overwrites those buffers. A Python lock around the launch alone does
not order asynchronous work on different streams. Give independently concurrent
streams and in-flight graph executions separate adapters and storage.

During a captured graph's entire lifetime, keep every captured input, output,
and scratch allocation alive at the same address, with the same shape, dtype,
device, and strides. Do not resize or rebind those tensors. The adapter's strong
references prevent deletion of its inputs; they cannot prevent a caller from
resizing their storage. A CUDA graph replay runs the recorded GPU operations
without calling Python, so it does not rerun the adapter's binding checks.
Update input values in place with correctly ordered device work.

## It fails closed, on purpose

If the call geometry or any input binding differs from what was registered, the
adapter **raises `AdapterError`**. It does not reallocate, and it does not fall
back to the native converter.

Changed bindings can invalidate the consumer's captured addresses or output
ownership assumptions. Rejecting drift makes the integration error explicit.

Concretely it rejects:

- a `dcp_size` outside `QUALIFIED_DCP_SIZES`
- `cp_kv_cache_interleave_size != 1` (the upstream indexer itself fails closed here)
- a `block_size` outside `QUALIFIED_BLOCK_SIZES`
- a top-k width outside `QUALIFIED_TOPK_WIDTHS`
- a `BLOCK_N` that differs from the wrapper being replaced
- `return_valid_counts` or `compact_valid_to_front` not `True`
- any change to an input tensor's address, shape, stride, dtype, device, or
  lazy-negation state
- changed output or scratch-buffer signatures
- being called at all before `prepare()`
- repeated preparation or preparation while either the caller's current CUDA
  device or the target device is capturing a graph
- empty or oversized hierarchical geometry before workspace allocation

The bound inputs may share storage with one another, since the kernels only read
them. The launchers reject a written buffer that shares or overlaps another
buffer's memory, even through separate storage objects; the adapter's own
outputs and workspace are allocated apart from every input.

## The qualified surface

| Parameter | Qualified values |
|---|---|
| `dcp_size` | 2, 4 |
| `cp_kv_cache_interleave_size` | 1 |
| `block_size` | 32, 64 |
| `num_topk_tokens` | 2048 |
| `BLOCK_N` | 128 |

**This is narrower than what the kernels support.** `silkern.localize_rowwise` and
`silkern.localize_hierarchical` handle grouped interleave and a much wider geometry
matrix. The *integration* has only been exercised on the surface above, and
conflating "the kernel handles it" with "the integration was tested on it" is the
easiest way to ship a wrong index.

To widen it, compare the new upstream signature and semantics with the recorded
revision, run `silkern.conformance()` on the new geometry (from the command
line, `python -m silkern --matrix FILE` with a JSON list of geometry objects),
and verify the actual
integration and graph capture with that consumer. Extend the `QUALIFIED_*`
constants only after all three checks pass, and record the new revision and
qualification evidence. Kernel conformance alone does not qualify an upstream
integration.

## Upstream pinning

The module records the upstream revision its call signature, geometry envelope,
and file digests were checked against:

```python
from silkern.integrations import vllm
vllm.UPSTREAM_COMMIT                  # revision
vllm.UPSTREAM_SPARSE_UTILS_SHA256     # sparse_utils.py
vllm.UPSTREAM_WRAPPER_SHA256          # the wrapper module
vllm.UPSTREAM_INDEXER_SHA256          # the indexer
```

These are informational — nothing enforces them at import. If you are on a
different revision, diff the converter's signature and semantics before
installing the adapter, and re-check the digests. `install_converter` does verify
that the target module actually has a `triton_filter_and_convert_dcp_index`
attribute and raises if it does not, which catches the grossest form of drift.

## A capture checklist

1. `python -m silkern --require-device` passes on the device you will deploy on.
   Its default matrix pairs each qualified `dcp_size` with each qualified
   `block_size` at width 2048, one rank apiece; add your exact deployment
   geometry, every rank included, with `--matrix FILE`.
2. `prepare()` is called once, on the real tensors, before warmup.
3. Every graph bucket gets its own adapter — a different batch size is a
   different binding.
4. Your capture gate asserts `adapter.fixed_buffer_signatures` is unchanged
   before and after capture.
5. Neither the eager calls nor the captured call allocate. After
   `torch.cuda.reset_peak_memory_stats()`, both `torch.cuda.memory_allocated()`
   and `torch.cuda.max_memory_allocated()` stay at their starting level. Measure
   the captured call inside the `torch.cuda.graph` block, around the call alone,
   because entering the block can allocate PyTorch's own bookkeeping. A replay
   never calls the allocator, so measuring around replays alone cannot fail.
6. Each replay computes from the bound inputs. Rewrite them in place with a
   second batch whose correct result differs, overwrite `adapter.output` and
   `adapter.counts` with a value no launch writes, replay, and compare with the
   reference result; then switch back to the first batch and compare again.
   Comparing after replays of unchanged inputs passes even if the graph does
   nothing.
7. `adapter.calls` increased, i.e. the swap actually took effect and you are not
   quietly still running the native converter.
8. Shared-buffer calls, graph replays, and consumer reads have explicit stream
   ordering; concurrent executions use separate storage.

Kernel conformance poisons outputs before every checked launch and replay,
alternates replay inputs between two fixtures, and measures allocation around
every launch, the capture, and every replay. Repeat checks 5 and 6 around the
actual integration: a correct kernel can still be wired to the wrong tensors,
or to a copy that a captured graph never refreshes. `adapter.calls` is a Python
invocation counter; a CUDA graph replay does not execute Python again, so check
it during eager execution/capture rather than expecting it to rise on every
replay.
