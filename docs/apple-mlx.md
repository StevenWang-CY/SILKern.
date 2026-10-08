# Apple silicon and MLX

[README](../README.md) · [Architecture](architecture.md) · [Contract](contract.md) · [Evidence](evidence.md)

`silkern.localize_mlx` implements the localization contract directly on MLX
arrays. It offers a custom Metal kernel and a compositional MLX implementation
behind one functional API. Use it to build and verify sparse-index conversion
on Apple silicon without requiring PyTorch or Triton.

## Installation

The Apple extra currently pins MLX `>=0.32.3,<0.33`. Use Apple silicon, native
arm64 Python 3.11 or newer, and macOS 14 or newer. These combine SILKern's Python
minimum with the [MLX installation requirements](https://ml-explore.github.io/mlx/build/html/install.html).
Install the versioned Apple package:

```bash
python -m pip install "silkern[mlx] @ https://github.com/StevenWang-CY/SILKern./releases/download/v2.0.0/silkern-2.0.0-py3-none-any.whl"
```

For examples, benchmarks, or development, work from a
[clone of the repository](../README.md#installation):

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[mlx,dev]"
python -c 'import mlx.core as mx; print(mx.metal.is_available())'
```

Metal availability should print `True` for the accelerated path. The `mlx`
extra is conditional on arm64 macOS; installing it on another platform leaves
the dependency-free base package installed and does not install MLX.
A Rosetta/x86 Python environment is not an Apple silicon environment. Check
`python -c 'import platform; print(platform.machine())'` if wheel installation
fails or the expected Metal device is unavailable.

## API

```text
out, counts = silkern.localize_mlx(
    req_ids,                   # mx.int32, (batch,)
    block_table,               # mx.int32, (requests, table_width)
    token_indices,             # mx.int32, (batch, width)
    *,
    block_size,
    dcp_size,
    dcp_rank,
    dcp_interleave=1,
    compact_valid_to_front=True,
    backend="auto",            # "auto", "metal", or "mlx"
    stream=None,
)
```

The block above describes the signature. Here is a runnable example:

```python
import mlx.core as mx
from silkern import localize_mlx, localize_reference

requests = [0, 1]
tables = [[11, 2, 7, 5], [3, 9, 1, 8]]
rows = [[8, 5, 130, 7, -1, 262], [2, 128, 3, 258, 128, -1]]
geometry = dict(block_size=64, dcp_size=2, dcp_rank=0)

req_ids = mx.array(requests, dtype=mx.int32)
block_table = mx.array(tables, dtype=mx.int32)
token_indices = mx.array(rows, dtype=mx.int32)
out, counts = localize_mlx(req_ids, block_table, token_indices, **geometry)
mx.eval(out, counts)

expected, expected_counts = localize_reference(requests, tables, rows, **geometry)
assert out.tolist() == expected
assert counts.tolist() == expected_counts
```

Inputs must be nonempty arrays with the shapes shown above and exactly
`mx.int32` dtype. Row width is at most 4096. Strided inputs are supported;
materialization may be required for the Metal path. Inputs remain unchanged.
The output shapes match `token_indices.shape` and `(batch,)`, respectively.

Metadata validation raises `LocalizationError` for malformed shapes, dtypes,
geometry, and backend choices. Geometry scalars must be integers, `block_size`
must be divisible by `dcp_interleave`, and `0 <= dcp_rank < dcp_size`.
Positive geometry values and `dcp_size * dcp_interleave` must fit signed 32-bit
arithmetic. Physical-slot arithmetic must also fit `int32`; callers guarantee
this data-dependent bound because the launcher does not read arrays back to
validate them.

An out-of-range request ID is masked on the device: its output row contains
`-1`, with count zero. This safety behavior differs from the Python oracle,
which raises, and the CUDA path, where valid request IDs remain a caller
precondition. Valid requests obey the same localization contract across all
backends, including duplicate tokens, negative page entries, grouped interleave,
and the `dcp_size=1` compaction bypass.

## Execution and streams

`stream` accepts `mx.Stream`, `mx.ThreadLocalStream`, `mx.Device`, a device type
such as `mx.cpu`/`mx.gpu`, or `None`. A device selects its default stream. With
`None`, `auto` and `mlx` use the current default device; `metal` selects the GPU.
Localization restores the caller's default stream and device after building
its lazy outputs.

| Choice | Execution | Use when |
|---|---|---|
| `backend="auto"` | Metal for a Metal GPU stream; compositional MLX for a CPU stream | Normal application code |
| `backend="metal"` | Custom Metal kernel; requires a Metal GPU | Testing or measuring the specialized implementation |
| `backend="mlx"` | MLX array operations on the selected stream | Comparing implementations or using the CPU stream |

```python
# CPU execution of the compositional implementation.
cpu_out, cpu_counts = localize_mlx(
    req_ids, block_table, token_indices, **geometry, stream=mx.cpu,
)
mx.eval(cpu_out, cpu_counts)

# Explicit GPU stream and specialized Metal implementation.
gpu_stream = mx.new_stream(mx.gpu)
gpu_out, gpu_counts = localize_mlx(
    req_ids, block_table, token_indices, **geometry,
    backend="metal", stream=gpu_stream,
)
mx.eval(gpu_out, gpu_counts)
```

MLX builds work lazily. Returning from `localize_mlx` does not establish that the
kernel has finished. Keep arrays inside your MLX computation and call `mx.eval`
when host results or a timing boundary are needed. Re-evaluating the same already
materialized output is not a new localization invocation. See
[MLX lazy evaluation](https://ml-explore.github.io/mlx/build/html/usage/lazy_evaluation.html).

The API returns newly allocated outputs. The MLX composition may create multiple
intermediates; the Metal path reduces this operation overhead through a
specialized kernel built with
[`mx.fast.metal_kernel`](https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html).
Neither path promises CUDA-style fixed addresses, allocation-free execution, or
CUDA graph capture semantics.

### Independent worker threads

In MLX 0.32.3, an ordinary `mx.new_stream` belongs to the thread that creates
it. For independent work in multiple threads, use
[`mx.new_thread_local_stream`](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.new_thread_local_stream.html):
the same handle resolves to a separate stream in each calling thread. These
semantics are defined by [MLX's stream API](https://github.com/ml-explore/mlx/blob/v0.32.3/python/src/stream.cpp#L115-L143).

Create each worker's arrays and evaluate its graph in that worker. This CPU
example returns plain Python values across the thread boundary:

```python
from concurrent.futures import ThreadPoolExecutor
import mlx.core as mx
from silkern import localize_mlx

worker_stream = mx.new_thread_local_stream(mx.cpu)

def localize_row(row):
    with mx.stream(worker_stream):
        requests = mx.array([0], dtype=mx.int32)
        table = mx.array([[3, 1]], dtype=mx.int32)
        tokens = mx.array([row], dtype=mx.int32)
        out, counts = localize_mlx(
            requests, table, tokens,
            block_size=2, dcp_size=2, dcp_rank=0,
            backend="mlx", stream=worker_stream,
        )
        mx.eval(out, counts)
        return out.tolist(), counts.tolist()

with ThreadPoolExecutor(max_workers=3) as pool:
    results = list(pool.map(localize_row, ([0, 2, -1], [6, 4, 1], [-1, -1, -1])))
assert results == [([[6, 7, -1]], [2]), ([[3, 2, -1]], [2]), ([[-1, -1, -1]], [0])]
```

For Metal workers, create the handle with `mx.gpu` and select `backend="metal"`.
This pattern covers independent per-worker graphs; it does not establish safe
cross-thread sharing of lazy graphs or a worker-throughput improvement.

### Compiled application calls

Pass arrays as arguments to a compiled function so later calls can provide new
values rather than capturing an already evaluated result:

```python
compiled_localize = mx.compile(
    lambda r, table, tokens: localize_mlx(
        r, table, tokens, **geometry, backend="metal",
    )
)
out, counts = compiled_localize(req_ids, block_table, token_indices)
mx.eval(out, counts)
```

Compile and warm up before steady-state timing. Changing shapes or static
geometry can require a new compiled specialization. The recorded comparison
compiles both the native composition and custom Metal call, keeping the
baseline optimized too.

## Measurement and reproduction

Run correctness before interpreting a performance result. `--require-device`
returns exit status 2 when Metal is unavailable; a failed check returns 1, and
a passing run returns 0. Without this flag an unavailable backend is reported as
skipped, which is useful for portable CI but is not qualification:

```bash
python -m silkern.mlx_verify --repeats 16 --require-device
python -m bench.bench_mlx --compiled
```

The benchmark compares the custom Metal implementation and compositional MLX
implementation on identical inputs. `--compiled` additionally measures each
implementation through `mx.compile`, so dispatch and fusion effects are visible.
It checks outputs against the Python oracle and independently changes every
compiled input class before timing, then warms up implementations and includes
output evaluation when measuring each call. Run
`python -m bench.bench_mlx --help` for geometry and sampling options.

### Recorded Apple result

**Apple M5 Max · 48 GiB · macOS 26.6 arm64 · Python 3.12.12 · MLX 0.32.3 ·
SILKern 0.2.0 development checkpoint.** The measured implementation files are
unchanged in 2.0.0; the original records keep their recorded package version.
The [conformance report](../evidence/09-apple-mlx-consumer/conformance.json)
passes 48/48 cells across the custom Metal and compositional MLX paths,
with 16 repeated evaluations per cell. This checkpoint includes the large-batch
Metal grid correction, thread-local streams, and stronger per-session compiled
input checks. Earlier [07](../evidence/07-apple-mlx/) and
[08](../evidence/08-apple-mlx-audit/) records retain their original JSON and source
hashes. Differences between checkpoints are descriptive observations, not a
controlled before/after experiment.

<picture>
  <source media="(max-width: 767px)" srcset="../assets/fig-apple-performance-narrow.svg">
  <img src="../assets/fig-apple-performance.svg" width="100%" alt="Dot plot of median latency per call on Apple M5 Max. For nine localization geometries and two selected-attention cases, a hollow gray dot marks compiled MLX and a filled blue dot marks the compiled Metal kernel, joined by a thin line; a right-hand column lists the speedup, from 1.10 to 1.21 times for localization and 1.12 times for selected attention.">
</picture>

**Compiled MLX and Metal latency on Apple M5 Max.** Each row joins compiled compositional MLX (hollow) and the compiled custom Metal kernel (filled) for one geometry, on one zero-based scale; the right-hand column gives their ratio. The first nine rows are localization by selection width and batch size. The last two are the complete selected-attention consumer with two logical shards, fixed caches, and 64-dimensional keys and values. Each value is the median of three process-session medians and includes dispatch, allocation, execution, and synchronization; warmup and initial compilation are excluded. Compiled Metal was faster in all 33 session pairs; per-session localization ratios range from 1.10× to 1.77× because one of three sessions ran slower for both arms.

All latencies below are microseconds per evaluated functional call. Each value
is the median of three session medians. Speedup divides compiled MLX latency by
compiled Metal latency; these are descriptive ratios, not confidence bounds.

| Width | Batch | Compiled MLX | Compiled Metal | Speedup | Eager MLX | Eager Metal |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 1 | 182.00 | 160.42 | 1.13× | 293.30 | 163.97 |
| 128 | 8 | 182.92 | 158.51 | 1.15× | 299.21 | 165.10 |
| 128 | 32 | 184.82 | 161.08 | 1.15× | 303.83 | 164.61 |
| 2048 | 1 | 188.02 | 164.22 | 1.14× | 298.87 | 163.46 |
| 2048 | 8 | 185.60 | 164.20 | 1.13× | 305.25 | 165.73 |
| 2048 | 32 | 190.97 | 164.44 | 1.16× | 324.14 | 166.21 |
| 4096 | 1 | 183.86 | 167.07 | 1.10× | 295.94 | 168.50 |
| 4096 | 8 | 186.97 | 167.85 | 1.11× | 313.49 | 170.06 |
| 4096 | 32 | 202.97 | 167.92 | 1.21× | 332.63 | 169.56 |

The compiled comparison spans **1.10–1.21×**, while eager spans **1.76–1.96×**.
Session 3 ran slower in every arm for six of the nine geometries: compiled MLX
at width 2048 and batch 1 took 309.7 µs there, against 188.0 µs and 178.7 µs in
sessions 1 and 2. Taking the median of session medians keeps one perturbed
process from setting the result; the raw records keep every session's values.
All four arms passed the original fixture's oracle check before timing. Each
schema-2 session also checked both compiled arms against the oracle after
independently changing request IDs, page tables, and token selections. These
per-run gates verify that compiled calls respond to their dynamic arguments.
Each of three fresh Python processes measured nine geometries and four
arms, with 10 warmups and 12 rotating-order blocks of 50 evaluated calls per
arm. Geometry: page size 64, DCP-2, rank 0, interleave 1, front compaction.

The timed region includes Python dispatch, output allocation, execution, and
evaluation synchronization. It excludes fixture generation, oracle validation,
and first-use compilation. Comparing compiled paths controls for framework
fusion and Python graph-construction overhead more fairly than an eager-only
headline. It still measures a synchronized application call, not isolated Metal
kernel duration or asynchronous steady-state model throughput.

Source: [summary](../evidence/09-apple-mlx-consumer/summary.json),
[session 1](../evidence/09-apple-mlx-consumer/benchmark-session-1.json),
[session 2](../evidence/09-apple-mlx-consumer/benchmark-session-2.json), and
[session 3](../evidence/09-apple-mlx-consumer/benchmark-session-3.json). Runtime metadata and
source SHA-256 digests are recorded in the raw artifacts. Reproduce each session
in a fresh process:

```bash
python -m bench.bench_mlx --compiled \
  --width 128 2048 4096 --batch 1 8 32 \
  --warmup 10 --iterations 50 --blocks 12 --seed 11 \
  --output benchmark-session.json
```


For a useful reproduction, record the device, macOS/Python/MLX versions, package
revision, batch and width, block size, DCP geometry, warmup and sample counts,
and full latency distribution. Keep initial compilation outside steady-state
timing, and label whether Python dispatch and synchronization are included.
A result from one Apple chip is not a performance promise for the entire family.

### Complete selected-attention consumer

The [consumer example](consuming-indices.md#run-a-complete-selected-attention-consumer)
is also measured with each backend inside `mx.compile`. Each call includes
localization, safe paged K/V gathers, masked softmax, recombination across two
logical shards, and one outer evaluation of the result and counts. It uses
float32, a single head, 64-dimensional keys and values, and page size 64.

| Batch × width | Compiled MLX consumer | Compiled Metal consumer | Ratio |
|---:|---:|---:|---:|
| 1 × 128 | 294.82 µs | 263.16 µs | 1.12× |
| 8 × 2048 | 366.67 µs | 326.90 µs | 1.12× |

Each value is the median of three process-session medians, using the same
10-warmup, 12-block, 50-call protocol. Ratios are descriptive and have no
confidence interval. Before timing, each arm must match independent unsharded
float64 attention within `rtol=1e-5, atol=1e-6` and exact per-shard counts, both
for the original fixture and independently changed queries, request IDs, and
token selections. Page tables and K/V caches remain static in these checks and
timings. Unused padding includes NaN to expose incomplete masking.

```bash
python -m bench.bench_mlx_attention \
  --warmup 10 --iterations 50 --blocks 12 --seed 11 \
  --output consumer-session.json
```

Source: [consumer summary](../evidence/09-apple-mlx-consumer/consumer-summary.json)
and [complete reproduction commands and raw sessions](../evidence/09-apple-mlx-consumer/).
The two cases were fixed before measurement. All shards run on one device;
these measurements do not include distributed communication, a whole model, or
a serving runtime.

## Runnable sparse-gather example

```bash
python -m examples.mlx_sparse_gather
```

The [example](../examples/mlx_sparse_gather.py) connects localization to masked
KV gathers using two logical rank caches on one device. It checks that their
combined result matches the original logical selection. This demonstrates
consumer wiring without claiming multi-device attention.

For selected attention with fragmented K/V caches and shared normalization, run
`python -m examples.mlx_sparse_attention --backend mlx --device cpu`, or use
`--backend metal --device gpu` on Apple Metal. The
[consumer guide](consuming-indices.md) covers layout masks, poisoned padding,
single-rank holes, and all-invalid selections. Both examples run from a checkout.

## Integration boundaries

`dcp_size` and `dcp_rank` describe ownership mathematically. Executing the
localizer on one Mac does not establish a multi-Mac collective or distributed
attention implementation. Likewise, returning physical KV indices does not
install this operation into an MLX-LM model. The downstream consumer must use
the same page-table layout and count/layout semantics.

Preserving selection order removes one source of order variation. Attention
reductions, scheduling, batching, and other model operations still determine
whole-model reproducibility. The NVIDIA evidence is historical, uses a different
runtime and executor, and must not be presented as Apple evidence.

## Troubleshooting

| Symptom | Check |
|---|---|
| Missing optional dependency | Install `.[mlx]` in the Python environment running the program |
| Metal unavailable | Confirm native arm64 Python, compatible macOS/MLX, and `mx.metal.is_available()` |
| `LocalizationError` on inputs | Use nonempty `mx.int32` arrays with matching batch dimensions and width ≤ 4096 |
| `backend="metal"` with a CPU stream | Use `auto`/`mlx` for CPU, or supply a GPU stream |
| Implausibly tiny timing | Evaluate newly returned outputs inside every timed iteration |
| Values remain in their original columns | Expected when `dcp_size=1` or `compact_valid_to_front=False` |
| Negative value counted as valid | In-range negative page entries are preserved by the shared contract |
