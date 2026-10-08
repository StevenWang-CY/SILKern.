# Apple MLX sources after the post-release audit

This record compares the MLX localization sources measured in
[record 09](../09-apple-mlx-consumer/), the 2.0.0 implementation, with the
sources after the post-release audit, in one process on the same device. It
answers one question: did the audit's changes to `silkern/mlx.py` change
localization latency? It is not a new absolute measurement, and record 09
remains the source of the recorded latencies. No NVIDIA experiments were run.

Between the two sources, the custom Metal kernel began reading the batch, row
width, and block-table shape at run time instead of compiling them in. The
compositional MLX path began naming its stream for every operation instead of
switching MLX's process-wide default device, and divides by multiplication.
Validation now refuses `mx.compile(..., shapeless=True)`.

| Environment | Recorded value |
|---|---|
| Device | Apple M5 Max, the device of record 09 |
| Runtime | macOS 26.6, arm64 Python 3.12.12, MLX 0.32.3 |
| Sources | Record 09 `silkern/mlx.py`, SHA-256 `456a50cc…`, archived in [record 09](../09-apple-mlx-consumer/sources/silkern/mlx.py.txt); current `silkern/mlx.py`, SHA-256 `6cf50b34…`, archived in [`sources/`](sources/silkern/mlx.py.txt) |
| Machine load | One-minute load average 13.9–16.9 during the sessions, with other work running |

## Method

Each of three fresh Python processes loads both implementations: the archived
module under another name, with its Metal kernel renamed so that MLX compiles
and caches it separately. The fixture is the benchmark's own
(`bench.bench_mlx._case`, seed 11) for the nine record-09 geometries: widths
128, 2048, and 4096 at batches 1, 8, and 32, with page size 64, DCP size 2,
rank 0, interleave 1, and front compaction. Every arm is compiled with
`mx.compile`, checked against the Python oracle, warmed up 10 times, and timed
on the GPU stream in 12 rotating-order blocks of 50 synchronized calls, as in
record 09. A session records, per backend and geometry, the median latency of
the current arm divided by that of the record-09 arm.

Interleaving the four arms in one process cancels slow drift in machine load,
so each ratio compares like with like. The absolute latencies in these files are
not latency claims: with the machine loaded, the record-09 arms ran 21–39%
(Metal) and 75–107% (compiled MLX) slower than record 09 measured them on the
same sources. The heavier dispatch work of the compositional path suffers more
from contention, so the Metal-over-MLX ratios in these files overstate the
margin record 09 measured and are not reported.

## Result

Current latency divided by record-09 latency, as the median of three session
ratios; below 1 is faster.

| Width | Batch | Compiled Metal | Compiled MLX |
|---:|---:|---:|---:|
| 128 | 1 | 1.018 | 0.987 |
| 128 | 8 | 1.007 | 0.966 |
| 128 | 32 | 1.010 | 0.960 |
| 2048 | 1 | 1.042 | 1.000 |
| 2048 | 8 | 0.981 | 0.926 |
| 2048 | 32 | 1.002 | 0.968 |
| 4096 | 1 | 1.003 | 0.968 |
| 4096 | 8 | 1.004 | 0.968 |
| 4096 | 32 | 0.990 | 0.941 |

The custom Metal kernel's latency did not change measurably: the median ratio
is 1.004, between 0.981 and 1.042. The compositional MLX path became about 3%
faster: the median ratio is 0.968, between 0.926 and 1.000, and eight of nine
geometries are faster. Record 09's Metal-over-MLX ratios, 1.10–1.21×, describe
the 2.0.0 sources; with the current sources the Metal margin is likely a few
percent smaller. Single sessions vary more under this load, for example 0.864
for Metal at width 128 and batch 1 in session 2, and 0.770 and 0.767 at width
2048 and batch 8 in session 3; the median of three keeps one session from
setting the result.

## Conformance

[`conformance.json`](conformance.json) is the post-audit verifier's report on
the same device: 90/90 cells passed, with 16 repeated evaluations each. The
verifier now runs the compositional path on the CPU stream as well as the GPU,
and the custom Metal kernel on the GPU, over 15 geometries in both layouts with
seeded block tables of 1–9 requests and 1–48 pages at batch 8 (report schema
2). Record 09's 48 cells covered the GPU stream only, with one table shape.

## Records and reproduction

[`ab-session-1.json`](ab-session-1.json), [`ab-session-2.json`](ab-session-2.json),
and [`ab-session-3.json`](ab-session-3.json) hold the environment, source hashes,
load averages, oracle checks, block samples, arm schedules, and the script
itself. Regenerate the conformance report with
`python -m silkern.mlx_verify --backend both --seed 0 --repeats 16 --require-device --json`.
To rerun the comparison from the repository root on Apple silicon:

```bash
python -c 'import json; print(json.load(open("evidence/10-apple-mlx-post-audit/ab-session-1.json"))["script"], end="")' > ab_post_audit.py
for session in 1 2 3; do
  PYTHONPATH=. python ab_post_audit.py \
    evidence/09-apple-mlx-consumer/sources/silkern/mlx.py.txt "ab-session-${session}.json"
done
```

Verify integrity from the repository root with
`(cd evidence && shasum -a 256 -c SHA256SUMS)`.
