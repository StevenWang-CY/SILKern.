# Apple MLX localization and selected-attention checkpoint

This record measures the 0.2.0 implementation with the large-batch Metal grid
correction and MLX thread-local stream support. It also adds per-session
changing-input gates to compiled localization and measures a complete selected-
attention consumer. Earlier [07](../07-apple-mlx/) and
[08](../08-apple-mlx-audit/) JSON records remain unchanged. Differences between
checkpoints are descriptive observations, not a controlled before/after
performance experiment. No NVIDIA experiments were run.

| Environment | Recorded value |
|---|---|
| Device | Apple M5 Max, 48 GiB unified memory |
| Runtime | macOS 26.6, arm64 Python 3.12.12, MLX 0.32.3 |
| Package | SILKern 0.2.0; implementation SHA-256 hashes in each raw record |
| Correctness | 48 conformance cells, all seven checks passed; 16 repeat evaluations per cell |
| Localization timing | Three fresh Python processes; nine geometries; four arms per geometry |
| Consumer timing | Three further Python processes; two fixed cases; both complete consumers compiled |

## Records

[`sources/`](sources/) preserves all six source files identified across these
records, with SHA-256 hashes matching the recorded metadata. The `.py.txt`
suffix keeps archived code outside current package and test discovery; remove
the final `.txt` when reconstructing the original paths. This preserves the
measured implementation and tools, not a complete environment snapshot.

- [`conformance.json`](conformance.json): compositional MLX and custom Metal
  compared with the Python oracle, independently derived order, repeatability
  of arrays and counts, input immutability, output shape/dtype, and invalid-request
  masking. Batch, seed, and repeat count are recorded in its metadata.
- [`benchmark-session-1.json`](benchmark-session-1.json),
  [`benchmark-session-2.json`](benchmark-session-2.json), and
  [`benchmark-session-3.json`](benchmark-session-3.json): schema-2 localization
  records with original-fixture oracle checks and independent request ID,
  page-table, and token-array perturbation checks for both compiled arms.
- [`summary.json`](summary.json): localization medians and descriptive ratios.
- [`consumer-session-1.json`](consumer-session-1.json),
  [`consumer-session-2.json`](consumer-session-2.json), and
  [`consumer-session-3.json`](consumer-session-3.json): complete selected-attention
  records with independent float64 attention-reference and exact-count checks,
  including independently changed queries, request IDs, and token selections.
- [`consumer-summary.json`](consumer-summary.json): complete-consumer medians
  and descriptive ratios.
- [`dispatch-axis-diagnostic.json`](dispatch-axis-diagnostic.json): a retained
  same-process, rotating-order comparison of one- and two-dimensional Metal
  dispatch for three bounded geometries. It includes the script, source hashes,
  oracle checks, and raw samples. The observed 0.7–2.4 µs median differences did
  not reproduce the larger timing shift between checkpoints, so this diagnostic
  does not establish the cause of that shift.

Both summaries take each arm's median of its three session medians, then divide
those aggregate latencies. The summarizers recompute medians from samples and
validate matching source, environment, methodology, geometry, correctness gates,
and distinct sessions. Raw records include block samples, arm order, fixture
seeds, versions, device metadata, and source hashes.

## Localization measurement

Every timed iteration creates and evaluates fresh outputs. Each session uses
10 warmups and 12 rotating-order blocks of 50 calls per arm and geometry.
Timing includes Python dispatch, output allocation, execution, and evaluation
synchronization; it excludes fixture generation, correctness checks, and first-use
compilation. Geometry: page size 64, DCP size 2, rank 0, interleave 1, front
compaction; widths 128/2048/4096 and batches 1/8/32. Fixtures include unsorted
selections, duplicates, negative sentinels, and fragmented page tables.

Compiled MLX / compiled Metal ratios span **1.10–1.21×**; eager ratios span
**1.76–1.96×**, rounded from the summary. All nine geometries and all raw samples
are retained. The compiled functions accept arrays as dynamic arguments; each
session's separate input perturbation gates check that they respond to each
input class before timing. Earlier schema-1 records lack this per-run gate.

Session 3 ran slower in every arm for six of the nine geometries (width × batch
128 × 32, 2048 × 1, 2048 × 8, 2048 × 32, 4096 × 1, and 4096 × 8). There,
compiled MLX took 34–82% longer than in sessions 1 and 2 and compiled Metal
7–26% longer, so session 3's ratios for those geometries reach 1.44–1.77×.
Compiled Metal was faster in all 27 localization session pairs and all 6
consumer session pairs. The median of session medians keeps the shifted session
from setting either summary; every session's values stay in its record.

## Complete selected-attention measurement

The actual [consumer helper](../../examples/mlx_sparse_attention.py) is compiled
with either localization backend. Each call localizes selected positions,
gathers paged K/V with safe addresses and value masks, computes a shared softmax
normalization, and recombines two logical shards. One outer evaluation realizes
the output and counts. Both cases use float32, one head, 64-dimensional keys and
values, page size 64, interleave 1, and four request tables.

| Batch × width | Compiled MLX consumer | Compiled Metal consumer | MLX / Metal |
|---:|---:|---:|---:|
| 1 × 128 | 294.82 µs | 263.16 µs | 1.12× |
| 8 × 2048 | 366.67 µs | 326.90 µs | 1.12× |

The two cases were fixed before measurement. The protocol uses three fresh
processes, 10 warmups, and 12 rotating-order blocks of 50 evaluated calls per arm.
Before timing, both arms must match independent unsharded float64 attention
within `rtol=1e-5, atol=1e-6`, plus exact per-shard counts. This gate covers the
original fixture and independently changed queries, request IDs, and token
selections. Page tables and K/V caches remain static in these checks and timings;
unused padding contains NaN to expose incomplete masking. Fixture generation,
oracle checks, and initial compilation are outside the timed region.

## Reproduce

From the repository root on compatible Apple silicon:

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

for session in 1 2 3; do
  python -m bench.bench_mlx_attention \
    --warmup 10 --iterations 50 --blocks 12 --seed 11 \
    --output "consumer-session-${session}.json"
done
python -m bench.bench_mlx_attention --summarize consumer-session-{1,2,3}.json \
  --output consumer-summary.json
```

These are descriptive measurements without confidence intervals from one device
and runtime. The consumer executes two logical shards on that one device;
neither benchmark measures distributed communication, full-model decode,
model-serving throughput, or portability to other Apple devices. See the
[consumer guide](../../docs/consuming-indices.md) and
[measurement scope](../../docs/apple-mlx.md#measurement-and-reproduction).

Verify integrity from the repository root with
`(cd evidence && shasum -a 256 -c SHA256SUMS)`.
