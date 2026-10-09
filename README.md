<div align="center">

<img src="assets/logo-text.svg" width="350" alt="SILKern">

Deterministic sparse-index localization for context-parallel attention<br>
on Apple silicon (MLX and Metal) and NVIDIA GPUs (Triton).

[![CI](https://img.shields.io/github/actions/workflow/status/StevenWang-CY/SILKern./ci.yml?branch=main&label=CI)](https://github.com/StevenWang-CY/SILKern./actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/StevenWang-CY/SILKern.?label=release)](https://github.com/StevenWang-CY/SILKern./releases/latest)
[![Python](https://img.shields.io/python/required-version-toml?tomlFilePath=https%3A%2F%2Fraw.githubusercontent.com%2FStevenWang-CY%2FSILKern.%2Fmain%2Fpyproject.toml)](pyproject.toml)
[![License](https://img.shields.io/github/license/StevenWang-CY/SILKern.)](LICENSE)

[Installation](#installation) · [Quickstart](#quickstart) · [Performance](#performance) · [Contract](docs/contract.md) · [Evidence](docs/evidence.md)

</div>

<picture>
  <source media="(max-width: 767px)" srcset="assets/fig-system-narrow.svg">
  <img src="assets/fig-system.svg" width="100%" alt="A sparse selector turns query q into one row of global positions, t = 8, 5, 130, 7, -1, 262, and sends the same row to two ranks. On each rank SILKern, given the page table P and its rank number, returns physical KV-cache slots in selector order: rank 0 returns 708, 129, 451 with count 3, and rank 1 returns 706, 707 with count 2. Each rank's sparse attention gathers those slots from its own paged KV cache, and the partial results are combined across ranks into y = N / Z.">
</picture>

<p align="center"><em>One decode step on two ranks: each rank turns the same selection into slots of its own KV cache, in selector order.</em></p>

## News

- **2026-10-09** · [Version 2.2.0](https://github.com/StevenWang-CY/SILKern./releases/tag/v2.2.0)
  follows a second audit of the repository. Malformed inputs now raise
  `LocalizationError` from every entry point, the MLX verifier checks contiguous
  as well as strided inputs, and the attention benchmark pads its cache with NaN,
  so a consumer that skips masking fails. Figures stay legible when a reader's
  GitHub theme differs from the system theme, and the opening figure shows where
  SILKern sits in a decode step. See the [changelog](CHANGELOG.md).
- **2026-10-07** · [Version 2.1.0](https://github.com/StevenWang-CY/SILKern./releases/tag/v2.1.0)
  follows an independent audit of 2.0.0. It closes two silent MLX failures,
  under shapeless compilation and under concurrent CPU and GPU calls, makes
  the CUDA conformance tool catch graphs that skip work or replay stale inputs,
  and stops the Metal kernel recompiling for each page-table shape. See the
  [changelog](CHANGELOG.md).
- **2026-10-06** · [Version 2.0.0](https://github.com/StevenWang-CY/SILKern./releases/tag/v2.0.0)
  runs natively on Apple silicon: `localize_mlx` takes MLX arrays and executes
  a custom Metal kernel, and a complete selected-attention example shows how to
  consume the result.
- **2026-08-04** · First public release, with the Python oracle, the rowwise
  and hierarchical Triton kernels, a fixed-buffer vLLM adapter, and the NVIDIA
  experiments behind them.

## Overview

Sparse attention chooses, for each query, which earlier tokens to read, and it
names them by global position. Under decode context parallelism the KV cache is
split across ranks and stored in pages. Before attention can gather anything,
each selected position must be filtered to the rank that owns it and translated
through that rank's page table into a physical slot. SILKern performs this step.

<picture>
  <source media="(max-width: 767px)" srcset="assets/fig-localization-narrow.svg">
  <img src="assets/fig-localization.svg" width="100%" alt="Three panels follow one selection row on rank 0: six global positions are filtered to three owned tokens and deinterleaved; their logical pages pass through the page table to physical KV blocks 11, 2, and 7, giving slots 708, 129, and 451; stable prefix compaction then writes 708, 129, 451 followed by padding, with count three.">
</picture>

<p align="center"><em>The quickstart row on rank 0 of two: owned tokens become physical KV-cache slots, in selector order.</em></p>

The step is small, but its output order is observable. A converter that
compacts survivors with atomic reservations can return the same entries in a
different order on every replay, and downstream floating-point reductions see
that order. SILKern assigns each survivor its destination from a prefix sum, so
repeated calls return identical values, counts, and order on every backend.

<picture>
  <source media="(max-width: 767px)" srcset="assets/fig-problem-narrow.svg">
  <img src="assets/fig-problem.svg" width="100%" alt="Three replays of the same four tile groups: atomic reservation emits a different group order on each replay, while offsets from an exclusive scan keep the input order every time.">
</picture>

<p align="center"><em>Atomic reservation emits the same tiles in a new order on every replay; prefix-sum destinations keep input order.</em></p>

- **An exact contract.** A dependency-free Python oracle defines the result, and
  every accelerated path is tested against it for equality. See the
  [contract](docs/contract.md).
- **Apple silicon.** `localize_mlx` runs a custom Metal kernel on a GPU stream
  and composed MLX operations on a CPU stream.
- **CUDA.** Two Triton kernels write into caller-owned buffers without
  allocating, so the same call can be captured in a CUDA graph. A vLLM adapter
  binds those buffers once.
- **Evidence.** Each result below links to its raw, checksummed record and to
  the command that produced it.

## Installation

The base package is pure Python and has no dependencies. Install the 2.2.0
wheel from its GitHub release:

```bash
python -m pip install "silkern @ https://github.com/StevenWang-CY/SILKern./releases/download/v2.2.0/silkern-2.2.0-py3-none-any.whl"
```

For an accelerator backend, write `silkern[mlx]` or `silkern[gpu]` in place of
`silkern`:

| Platform | Extra | Entry points |
|---|---|---|
| <img src="assets/icons/cpu.svg" width="16" height="16" align="absmiddle" alt="">&nbsp; Any Python 3.11+ | none | `localize_reference` |
| <img src="assets/icons/soc.svg" width="16" height="16" align="absmiddle" alt="">&nbsp; Apple silicon with MLX | `mlx` | `localize_mlx` |
| <img src="assets/icons/gpu.svg" width="16" height="16" align="absmiddle" alt="">&nbsp; NVIDIA GPU with PyTorch and Triton | `gpu` | `localize_rowwise`, `localize_hierarchical` |

The examples and benchmarks run from a clone; the trailing dot in `SILKern.` is
part of the repository name.

```bash
git clone https://github.com/StevenWang-CY/SILKern. silkern
cd silkern
python -m pip install -e ".[dev]"
```

## Quickstart

One selection row on rank 0 of two ranks, with 64-token pages. This needs only
the base install:

```python
from silkern import localize_reference

out, counts = localize_reference(
    req_ids=[0],
    block_table=[[11, 2, 7, 5]],
    rows=[[8, 5, 130, 7, -1, 262]],
    block_size=64,
    dcp_size=2,
    dcp_rank=0,
)
assert out == [[708, 129, 451, -1, -1, -1]]
assert counts == [3]
```

<picture>
  <source media="(max-width: 767px)" srcset="assets/fig-contract-narrow.svg">
  <img src="assets/fig-contract.svg" width="100%" alt="A table follows six input columns through owner, validity, local position, page and offset, physical page, and physical slot: columns 0, 2, and 5 survive and map to slots 708, 129, and 451. Beside it are the localization equations and the compacted and column-preserving output layouts, both with count three.">
</picture>

<p align="center"><em>The quickstart row column by column, with the equations and both output layouts.</em></p>

Rank 0 owns tokens 8, 130, and 262. Token 130, for example, is the rank's local
position 65: offset 1 in logical page 1, which the table maps to physical block
2, so its slot is 2 × 64 + 1 = 129. Survivors keep their order and the rest of
the row is padding. Two cases surprise people. With `dcp_size=1`, values stay in
their original columns. A negative page-table entry is a valid, counted
mapping, so find valid entries from `counts` and the layout, never from
`out >= 0`. The [contract](docs/contract.md) specifies every case.

### Apple silicon

```python
import mlx.core as mx
from silkern import localize_mlx

out, counts = localize_mlx(
    mx.array([0], dtype=mx.int32),
    mx.array([[11, 2, 7, 5]], dtype=mx.int32),
    mx.array([[8, 5, 130, 7, -1, 262]], dtype=mx.int32),
    block_size=64, dcp_size=2, dcp_rank=0,
)
mx.eval(out, counts)  # MLX is lazy; evaluate before reading on the host
assert out.tolist() == [[708, 129, 451, -1, -1, -1]]
assert counts.tolist() == [3]
```

The default `backend="auto"` picks the Metal kernel on a GPU stream and
composed MLX operations on a CPU stream; pass `backend="metal"` or
`backend="mlx"` to choose. Each call returns new arrays. The
[Apple guide](docs/apple-mlx.md) covers streams, threads, and `mx.compile`.

### CUDA

```python
import torch
from silkern import localize_rowwise

req_ids = torch.tensor([0], dtype=torch.int32, device="cuda")
block_table = torch.tensor([[11, 2, 7, 5]], dtype=torch.int32, device="cuda")
token_indices = torch.tensor([[8, 5, 130, 7, -1, 262]], dtype=torch.int32, device="cuda")
out = torch.empty_like(token_indices)
counts = torch.empty(1, dtype=torch.int32, device="cuda")

localize_rowwise(
    req_ids, block_table, token_indices, out, counts,
    block_size=64, dcp_size=2, dcp_rank=0,
)
```

The kernel writes into the buffers you pass and never allocates, so this call
can be captured in a CUDA graph and replayed. `localize_hierarchical` adds a
caller-owned workspace for tile-parallel scans; the
[dispatch guide](docs/dispatch.md) compares the two. The
[vLLM adapter](docs/integration-vllm.md) binds buffers once and documents how to
order their reuse across streams and graph replays.

## Performance

### Apple silicon

We timed both MLX backends of the 2.0.0 sources inside `mx.compile` on an
Apple M5 Max with MLX 0.32.3. A timed call includes Python dispatch, output allocation,
execution, and synchronization; compilation and warmup are excluded. Each
value is the median over three fresh processes, and each process ran 12
rotating-order blocks of 50 calls per arm.

<picture>
  <source media="(max-width: 767px)" srcset="assets/fig-apple-performance-narrow.svg">
  <img src="assets/fig-apple-performance.svg" width="100%" alt="Dot plot of median latency per call on Apple M5 Max. For nine localization geometries and two selected-attention cases, a hollow gray dot marks compiled MLX and a filled blue dot marks the compiled Metal kernel, joined by a thin line; a right-hand column lists the ratio, from 1.10 to 1.21 times for localization and 1.12 times for selected attention.">
</picture>

The Metal kernel was **1.10–1.21× faster** than compiled MLX at every batch and
width we tested, and latency changed little across either. The advantage
carried into the complete selected-attention example, at 1.12×. Since then the
compiled MLX path has become about 3% faster while the Metal kernel is
unchanged ([record 10](evidence/10-apple-mlx-post-audit/)), so the current
margin is a few percent smaller. The
[Apple guide](docs/apple-mlx.md#measurement-and-reproduction) lists the eager
results and the exact session commands:

```bash
python -m bench.bench_mlx --compiled
python -m bench.bench_mlx_attention
```

### CUDA (archived)

These measurements come from an earlier revision on two NVIDIA H100 GPUs,
running complete 48-layer decode steps on a reference executor. We have not
repeated NVIDIA experiments since.

<picture>
  <source media="(max-width: 767px)" srcset="assets/fig-cost-narrow.svg">
  <img src="assets/fig-cost.svg" width="100%" alt="Archived two-H100 measurements: converter-segment latency bars for rowwise, atomic, and hierarchical at 32K, and a forest plot of complete-step ratios to atomic with 98.75% intervals and a column of estimates; only the 64K rowwise interval lies beyond the 1.01 margin.">
</picture>

At 32K context the rowwise kernel took **38% less time** than atomic
reservation over the converter segment, and the complete decode step stayed
within the study's prespecified 1% non-inferiority margin. At 64K, rowwise was
**1.4% slower**, a measured regression, while hierarchical stayed within the
margin. Intervals are 98.75% intervals across five sessions. The
[evidence guide](docs/evidence.md#reading-performance-correctly) explains how
to read these records. They describe one workload, so benchmark both kernels in
your own model before choosing:

```bash
python -m bench.bench_converter --with-atomic
```

## Using the output

Localized slots are the input to attention. Before gathering K/V, mask padding
according to the layout: a prefix of length `counts` after compaction, and per
column otherwise. Gather padded entries from a safe address, then mask the
gathered values as well, because an unused slot may hold NaN and a zero weight
times NaN is still NaN. Across logical shards, normalize the softmax over the
whole selection.

<picture>
  <source media="(max-width: 767px)" srcset="assets/fig-consumer-narrow.svg">
  <img src="assets/fig-consumer.svg" width="100%" alt="Three panels show compact versus column-preserving slots and validity masks, masked K/V gathering from a paged cache with a placeholder row, and attention normalization shared across logical shards.">
</picture>

<p align="center"><em>One layout-aware mask selects addresses and gathered values; one softmax spans every logical shard.</em></p>

The [consumer guide](docs/consuming-indices.md) works through these rules, and
its runnable example checks itself against unsharded attention:

```bash
python -m examples.mlx_sparse_attention --backend mlx --device cpu
```

## Verification

The Python oracle is the reference for every backend. Our tests check it
against an independent restatement of the contract and against metamorphic
properties, fuzz the MLX and Metal paths against it, and execute the Python
examples in this README and the guides wherever their backend is available.

| Platform | Recorded result | Record |
|---|---|---|
| <img src="assets/icons/soc.svg" width="16" height="16" align="absmiddle" alt="">&nbsp; Apple M5 Max, MLX 0.32.3 | 90/90 conformance cells across MLX on CPU and GPU and Metal, 16 repeated evaluations each | [10](evidence/10-apple-mlx-post-audit/) |
| <img src="assets/icons/gpu.svg" width="16" height="16" align="absmiddle" alt="">&nbsp; NVIDIA B200 (archived) | 131/131 conformance cells; 10,000 graph replays per arm | [01](evidence/01-oracle-conformance-b200/analysis.json) |
| <img src="assets/icons/gpu.svg" width="16" height="16" align="absmiddle" alt="">&nbsp; 2 × NVIDIA H100 (archived) | Complete-decode canary and converter timing | [05](evidence/05-full-decode-canary/) |
| <img src="assets/icons/gpu.svg" width="16" height="16" align="absmiddle" alt="">&nbsp; NVIDIA SM120 (archived) | Trained layer-0 semantics and mechanism decomposition | [02](evidence/02-trained-layer0-semantics/), [04](evidence/04-mechanism-decomposition/) |

```bash
python -m pytest -q -m "not gpu and not mlx"   # any machine
python -m silkern.mlx_verify --require-device  # Apple silicon: Metal conformance
python -m silkern --require-device             # NVIDIA: CUDA conformance
cd evidence && shasum -a 256 -c SHA256SUMS     # integrity of the records
```

## Scope and limits

- SILKern converts indices. It does not select tokens, manage the KV cache, or
  compute attention; the attention example is a single-device reference
  consumer.
- Device arrays are signed `int32`, accelerator rows hold at most 4096 entries,
  and physical-slot arithmetic must fit in `int32`.
- The CUDA kernels require valid request IDs and page entries that lie inside
  your cache. The MLX backend masks invalid request rows to empty.
- The NVIDIA results predate 2.0.0. Requalify on your own hardware with
  `python -m silkern --require-device` before deploying.

## Documentation

- [Contract](docs/contract.md): exact mapping rules, integer bounds, and edge cases
- [Consuming indices](docs/consuming-indices.md): layout masks, safe gathers, and selected attention
- [Apple silicon and MLX](docs/apple-mlx.md): setup, streams, conformance, and measurements
- [Architecture](docs/architecture.md): layers, execution paths, and memory ownership
- [Choosing a backend](docs/dispatch.md): Apple and CUDA paths and their tradeoffs
- [Determinism](docs/determinism.md): why atomic reservation reorders output
- [vLLM integration](docs/integration-vllm.md): the fixed-buffer adapter and its lifetime rules
- [Evidence](docs/evidence.md) and [release validation](docs/release-validation.md): provenance and claim boundaries
- [Contributing](CONTRIBUTING.md): development workflow, evidence standards, and figure regeneration

## Citation

SILKern is released under the Apache-2.0 license ([LICENSE](LICENSE),
[NOTICE](NOTICE)). If you use it in your work, please cite the release and name
the revision you ran:

```bibtex
@software{silkern2026,
  title   = {SILKern: deterministic sparse-index localization for context-parallel decode},
  author  = {{The SILKern Authors}},
  year    = {2026},
  version = {2.2.0},
  license = {Apache-2.0},
  url     = {https://github.com/StevenWang-CY/SILKern.}
}
```
