# Apple MLX audit checkpoint

This record measures the 0.2.0 implementation after the audit converted geometry
values to Python integers before checking them, so NumPy integer scalars cannot
lose low bits, and made the verifier require every check to be exactly `True`.
Its Metal kernel is identical to record 07's, which already synchronized group
totals before prefix reads and computed physical slots in unsigned arithmetic;
supported physical results must still fit signed `int32`. This earlier checkpoint
predates the large-batch grid and thread-local stream updates. Use
[record 10](../10-apple-mlx-post-audit/) for current conformance and
[record 09](../09-apple-mlx-consumer/) for the recorded latencies and stronger
per-session compiled-input gates.
The [initial checkpoint](../07-apple-mlx/)
remains intact with its original source hashes and JSON results. Differences
between checkpoints are not a controlled before/after performance experiment.
No NVIDIA experiments were run.

| Environment | Recorded value |
|---|---|
| Device | Apple M5 Max, 48 GiB unified memory |
| Runtime | macOS 26.6, arm64 Python 3.12.12, MLX 0.32.3 |
| Package | SILKern 0.2.0; implementation SHA-256 hashes in every raw record |
| Correctness | 48 conformance cells, all seven checks passed; 16 repeat evaluations per cell |
| Timing | Three fresh Python processes, nine geometries, four implementations per geometry |

## Records

The [`sources/`](sources/) directory preserves the four source files named in
this checkpoint's metadata, recovered byte-for-byte from a retained source
distribution. Their SHA-256 hashes match the original records. Files end in
`.py.txt` to keep historical code outside current package and test discovery;
to inspect or reconstruct the recorded paths, remove the final `.txt` suffix.
This archive preserves implementation provenance, not a complete environment
snapshot. Original measurement JSON files remain unchanged.

- [`conformance.json`](conformance.json): compositional MLX and custom Metal
  compared with the Python oracle, independently derived order, repeatability
  of arrays and counts, input immutability, output shape/dtype, and invalid-request
  masking. Batch, seed, and repeat count are recorded in its metadata.
- [`benchmark-session-1.json`](benchmark-session-1.json),
  [`benchmark-session-2.json`](benchmark-session-2.json), and
  [`benchmark-session-3.json`](benchmark-session-3.json): raw block timings,
  execution order, geometry, fixture seed, environment, and source hashes.
- [`summary.json`](summary.json): each arm's median of its three session medians;
  ratios divide those aggregate latencies. The summarizer recomputes medians
  from samples and verifies matching source, environment, methodology, geometry,
  correctness gates, and distinct sessions before combining them.

## Recorded invocation

These commands produced this earlier checkpoint on Apple silicon. Current source
measures the current implementation; compare the recorded source hashes before
attributing a reproduction to this checkpoint:

```bash
python -m pip install -e ".[mlx]"
python -m silkern.mlx_verify --backend both --batch 4 --seed 0 --repeats 16 \
  --require-device --json > conformance.json

for session in 1 2 3; do
  python -m bench.bench_mlx --width 128 2048 4096 --batch 1 8 32 \
    --compiled --warmup 10 --iterations 50 --blocks 12 --seed 11 \
    --output "benchmark-session-${session}.json"
done

python -m bench.summarize_mlx benchmark-session-{1,2,3}.json --output summary.json
```

Compiled arms receive request IDs, page tables, and token arrays as dynamic
arguments. Every arm was checked against the original fixture's oracle before
timing. These schema-1 sessions did not record a per-run changing-input gate;
changing-input behavior was covered by separate tests. A passing original
fixture alone does not prove that a compiled call responds to changed arguments.
Every timed iteration creates and evaluates fresh outputs. Each session uses
10 warmups and 12 rotating-order blocks of 50 calls per arm and geometry.

Timings include Python dispatch, output allocation, execution, and evaluation
synchronization. They exclude input generation, correctness checks, and
first-use compilation. Geometry: block size 64, DCP size 2, rank 0, interleave 1,
front compaction; widths 128/2048/4096 and batches 1/8/32. Fixtures include
unsorted selections, duplicates, negative sentinels, and fragmented page tables.

Compiled MLX / compiled Metal ratios span **1.30–1.45×**; eager ratios span
**2.08–2.36×**, rounded from the summary. All nine geometries and all raw samples
are retained. These are descriptive measurements without confidence intervals
from one device and runtime. They measure localization; they do not establish
model-serving throughput, distributed Apple execution, or a performance promise
for other devices.

Verify integrity from the repository root with
`(cd evidence && shasum -a 256 -c SHA256SUMS)`.
