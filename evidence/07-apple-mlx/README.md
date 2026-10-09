# Apple MLX initial checkpoint

These Apple-only measurements describe the initial 0.2.0 implementation at the
source hashes recorded here. Its Metal kernel already synchronizes group totals
before prefix reads and computes physical slots in unsigned arithmetic; record 08
changed host-side integer normalization and the verifier's pass criteria. The
original JSON records are retained unchanged. Use
[record 10](../10-apple-mlx-post-audit/) for current conformance and
[record 09](../09-apple-mlx-consumer/) for the recorded latencies. The archived
NVIDIA records were not rerun.

| Environment | Recorded value |
|---|---|
| Device | Apple M5 Max, 48 GiB unified memory |
| Runtime | macOS 26.6, arm64 Python 3.12.12, MLX 0.32.3 |
| Package | SILKern 0.2.0, with implementation SHA-256 hashes in every raw record |
| Correctness | 48 conformance cells; all seven checks passed |
| Timing | Three fresh Python processes, nine geometries, four implementations per geometry |

## Records

The [`sources/`](sources/) directory preserves the four source files named in
this checkpoint's metadata, recovered byte-for-byte from a retained source
distribution. Their SHA-256 hashes match the original records. Files end in
`.py.txt` to keep historical code outside current package and test discovery;
to inspect or reconstruct the recorded paths, remove the final `.txt` suffix.
This archive preserves implementation provenance, not a complete environment
snapshot. Original measurement JSON files remain unchanged.

- [`conformance.json`](conformance.json): native MLX and custom Metal against the
  Python oracle, independently derived order, repeatability of arrays and counts,
  input immutability, output shape/dtype, and invalid-request masking.
- [`benchmark-session-1.json`](benchmark-session-1.json),
  [`benchmark-session-2.json`](benchmark-session-2.json), and
  [`benchmark-session-3.json`](benchmark-session-3.json): raw block timings,
  execution order, geometry, input seed, environment, and source hashes.
- [`summary.json`](summary.json): machine-derived descriptive aggregation. Each
  arm's latency is the median of its three session medians; ratios divide those
  aggregated latencies. These are not confidence intervals.

## Recorded invocation

The following invocation produced this checkpoint on Apple silicon. Running it
against current source measures the current implementation, not this earlier
source identity; compare the recorded SHA-256 hashes when reproducing a result.

```bash
python -m pip install -e ".[mlx]"
python -m silkern.mlx_verify --backend both --batch 4 --repeats 3 \
  --require-device --json > conformance.json

for session in 1 2 3; do
  python -m bench.bench_mlx --width 128 2048 4096 --batch 1 8 32 \
    --compiled --warmup 10 --iterations 50 --blocks 12 --seed 11 \
    --output "benchmark-session-${session}.json"
done

python -m bench.summarize_mlx benchmark-session-{1,2,3}.json --output summary.json
```

The compiled arms receive request IDs, page tables, and token arrays as dynamic
arguments. Each timed iteration creates and evaluates fresh outputs. Warmup
excludes first-use compilation. Execution order rotates between blocks, with
12 blocks of 50 calls per arm and geometry after 10 warmup iterations.
These schema-1 records check the original fixture against the oracle; they do
not record a per-run changing-input gate. Separate changing-input tests do not
add such a gate retroactively to these measurements.

Timings include Python dispatch, output allocation, execution, and
synchronization. They exclude input generation, oracle validation, and first-use
compilation. Block size is 64, DCP size is 2, rank is 0, interleave is 1, and
compaction is enabled. Inputs include unsorted selections, duplicates, negative
sentinels, and non-identity page tables.

The recorded compiled MLX / compiled Metal ratios span **1.34–1.55×**; eager
ratios span **2.27–2.54×**, rounded from the summary. All nine geometries are
retained. This is one device and one runtime configuration. The comparison
measures this localization primitive; it establishes neither full-model decode
performance nor distributed Apple execution.

Verify integrity from the repository root with
`(cd evidence && shasum -a 256 -c SHA256SUMS)`.
