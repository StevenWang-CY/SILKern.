# 2.0.0 implementation and release review

[README](../README.md) · [Architecture](architecture.md) · [Changelog](../CHANGELOG.md) · [Evidence](evidence.md)

This update adds native Apple execution and strengthens the existing API,
integration lifecycle, verification, and distribution. It remains a focused
localization library; production serving qualification depends on its consumer.
The implementation was developed under the unreleased `0.2.0` version and ships
as `2.0.0`. Historical measurement records keep their original metadata.

## Release verification · October 6, 2026

The `2.0.0` package passed fresh release checks on Apple M5 Max, macOS 26.6,
Python 3.12.12, and MLX 0.32.3:

| Release check | Result |
|---|---|
| CPU and Apple tests, with warnings treated as errors | 645 passed; two CUDA modules skipped because PyTorch is absent |
| Apple conformance with `--require-device --repeats 16` | 48/48 cells passed across compositional MLX and custom Metal |
| Tests from the extracted source distribution in a clean base environment | 398 passed; seven optional-runtime/NumPy cases skipped and three MLX cases deselected |
| Source archive, wheel, and strict Twine metadata checks | Passed; wheel built through the source archive |
| Wheel installed outside the checkout | Version and oracle verified; no MLX, PyTorch, or Triton installed; both verification commands report explicit skips |
| Evidence and documentation | All 43 evidence checksums and six measured source hashes match; local links, heading anchors, picture sources, and plotted values checked |
| Ruff and whitespace checks | Passed |

The [GitHub release](https://github.com/StevenWang-CY/SILKern./releases/tag/v2.0.0)
provides the wheel, source archive, SHA-256 checksums, fresh Apple conformance
report, and a validation summary identifying its exact commit and CI run.
The source archive includes the guides, examples, figure sources, and recorded
evidence. The wheel contains the library and verification entry points.

Release verification does not replace the recorded performance measurements.
The six implementation and benchmark sources identified in Apple record 09
are byte-identical to the measured checkpoint. No new timing claims or NVIDIA
experiments accompany this release.

## Findings and changes

| Investigation finding | Implemented change | Validation |
|---|---|---|
| CUDA buffer ownership does not transfer to MLX's lazy array model | Separate functional `localize_mlx` API with Metal and native MLX implementations | CPU and real Apple-device oracle comparisons, streams, strided inputs, compiled changing-input calls |
| Device request IDs could index an invalid row | Mask requests before table reads in both new Apple implementations | Negative and oversized IDs produce an invalid row and zero count |
| Integer coercion silently accepted fractional, string, and boolean index data | Strict oracle integer validation | Regressions for each malformed input category |
| Fixed-width host scalar multiplication could overflow during validation | Normalize integral scalars before arithmetic; bound accelerator geometry and workspace indices | Host-only overflow regressions |
| Empty or incomplete check dictionaries could report conformance success | Require every advertised check; preserve skipped/failed states and add machine-readable reports | Report and CLI regression tests |
| CUDA repeatability checks omitted some counts and inputs | Include counts, request IDs, current device, and nonzero memory canaries | Host tests; device behavior still requires CUDA qualification |
| Captured adapters could lose or change registered storage | Retain input references, make preparation one-shot, reject output/workspace drift | Tensor-metadata test doubles exercise lifecycle errors before dispatch |
| Eager-only performance comparison would omit MLX compiler optimization | Benchmark eager and compiled versions of both Apple paths | Three fresh processes; correctness gate before every timing cell |
| Documentation mixed primitive, complete-segment, and model-level claims | Correct units, distinguish historical evidence, retain adverse results | Source-based tables/figures and local link checks |
| Optional backend and packaging regressions could escape device tests | Expand CPU, Apple, wheel, and console-entry-point CI | Local wheel build and clean installation with no accelerator dependencies |

## Independent second-pass audit

A second review read the implementation and its consumers independently of the
first implementation pass. It reproduced and corrected additional issues:

| Finding | Correction | Verification |
|---|---|---|
| Supported NumPy integer geometry produced a non-builtin boolean Metal template argument | Carry normalized Python integers from validation through specialization | CPU, native MLX GPU, and Metal regressions; actual NumPy scalar calls verified |
| Mixed signed/unsigned integer arithmetic could promote to float and accept invalid oracle divisibility | Normalize before every geometry comparison and arithmetic check | Regression for an odd integer above the float64 exact range |
| CUDA pointers could belong to a different device from Triton's current launch context | Select the tensors' CUDA device for the entire launch sequence and restore the caller's context | Two-device host simulations, including failure at every launch stage; no CUDA execution |
| Integral hierarchical tile sizes were normalized only inside the shape helper | Pass a builtin integer to Triton specialization | Exact template-type regression |
| Invalid check values, unknown geometry fields, or oversized diagnostic fixtures could produce misleading reports | Require explicit true checks and exact schemas; reject unsafe or ignored geometry before allocation | Host report, CLI, and fixture regressions |
| A benchmark could emit duplicate geometries that its own aggregator rejected | Reject duplicate axes before loading MLX; validate complete provenance and rotating schedules when aggregating | Round-trip aggregation and malformed-record regressions |
| CPU-only MLX test selection still included GPU fixture variants | Mark every Metal execution variant and place input generation on the selected stream | CPU and Metal selections exercised separately |
| Kernel equality alone did not test attention-consumer masking | Add fragmented K/V-cache gathers and rank-wise softmax recombination against an independent float64 reference | CPU/Metal cases, including empty shards and poisoned padding |
| Buffer lifetime, stream ordering, and graph-replay obligations were underspecified | Document serialized storage reuse, independent concurrent adapters, and graph-lifetime binding rules | Cross-check against the adapter and asynchronous launch implementation |
| CI used a retiring macOS image and outdated action runtimes | Move to macOS 15, Node 24 actions, Python 3.14 coverage, and a mandatory MLX CPU smoke test | Clean local environments, action manifests, YAML parsing, and CPU smoke execution |

The CI maintenance decisions follow GitHub's
[macOS 14 retirement announcement](https://github.blog/changelog/2026-10-01-github-actions-macos-14-runner-image-retirement/)
and [Node 20 removal notice](https://github.blog/changelog/2026-09-23-node-20-is-no-longer-available-in-github-actions/).

## Further integration and compatibility audit

The next review tested malformed metadata, imported memory, MLX worker streams,
compiled consumers, and package-index rendering. It found additional concrete
gaps and added the following corrections:

| Finding | Correction | Verification |
|---|---|---|
| Metal's one-dimensional `batch * 256` dispatch exceeded MLX's signed integer grid type for an otherwise legal batch | Put threads and rows on separate dispatch axes | Reproduced the previous TypeError with lazy broadcast inputs at batch `2**23`; fixed launch construction without evaluating millions of rows, then ran real-device conformance |
| MLX's `ThreadLocalStream` was rejected because it is not a `Stream` subclass | Accept its native type and preserve per-thread resolution | Three concurrent workers with separate data, distinct resolved streams, exact outputs, and restored defaults on CPU, native MLX GPU, and Metal |
| DLPack-imported buffers can overlap despite different storage base pointers | Check contiguous byte spans as well as storage identity | Host simulations cover overlapping input/output/count/workspace regions and legal adjacent buffers |
| PyTorch lazy-negation views expose raw bytes inconsistent with their logical values | Reject unresolved negation metadata and include it in adapter binding signatures | Host regressions for every buffer and binding drift |
| Adapter preparation could miss capture on the caller's device after switching to the input device | Check both caller and target capture before allocating | Two-device host simulations; allocation is never reached in either capturing case |
| A rowwise batch could exceed the signed CUDA launch-axis range | Reject oversized batches in the launcher and before adapter allocation | Boundary metadata tests without allocating those tensors or launching CUDA |
| Original-fixture checks did not prove the measured compiled function consumed changing array arguments | Add independent request/table/token mutations before primitive timing; record them in schema 2 | Deliberately frozen-input regressions fail before timing; schema-1 records remain readable with their original, narrower qualification |
| Retimestamping copied raw samples could produce an apparently independent session | Fingerprint raw measurement payloads independently of timestamps and cached statistics | Duplicate-sample rejection regressions; raw medians are always recomputed |
| Malformed CUDA diagnostic cells could abort a sweep, and integral scalar geometry could break JSON serialization | Record malformed cells as failures and normalize geometry for serialization | Host-only malformed-cell and JSON round-trip regressions |
| Attention-consumer correctness lived only in test code | Extract a runnable, compiled-capable selected-attention example and test that actual helper | Independent float64 oracle, one/two/three logical shards, holes, duplicate tokens, empty rows, invalid requests, and poisoned padding |
| Primitive latency alone did not show its contribution to a surrounding computation | Add a separate benchmark of the complete compiled localization/gather/attention/recombination consumer | Independent float64 output and exact-count gates; query/request/selection membership mutations; one outer evaluation per fresh call; raw samples and a separate summary protocol |
| Repository-relative README links break in package-index metadata | Use a concise package description with absolute links | Built sdist/wheel metadata inspection and strict Twine checks |
| CI could discard the diagnostic artifact on failure; packaging tests depended on collection order | Preserve available conformance reports after failed checks and resolve imports independently | Workflow review and packaging tests run in isolation |

The dispatch and stream decisions were checked against MLX 0.32.3's
[custom-kernel bindings](https://github.com/ml-explore/mlx/blob/v0.32.3/python/src/fast.cpp)
and [stream bindings](https://github.com/ml-explore/mlx/blob/v0.32.3/python/src/stream.cpp).
The new [consumer guide](consuming-indices.md) specifies cache-address and
float32 arithmetic preconditions. Its logical shards run on one device; they
do not simulate transport latency or qualify distributed execution.

The consumer benchmark holds its paged caches fixed during each session and
changes dynamic query, request, and selection inputs during qualification. Its
two representative cells are single-head, float32 computations. It does not
measure a cache allocator, token selector, cache growth, model layers, request
queueing, or transport. Both implementations compile the same consumer, changing
only the localization backend. Timing follows MLX's documented
[fresh-call evaluation pattern](https://ml-explore.github.io/mlx/build/html/usage/compile.html).

## Post-release audit · October 7–8, 2026

An independent audit of the 2.0.0 sources reproduced the defects below on an
Apple M5 Max; CUDA host code was read and exercised through stand-ins, since no
NVIDIA GPU was available. Each correction has a test that fails against the
previous code.

| Finding | Correction | Verification |
|---|---|---|
| CUDA conformance passed a captured graph that did no work, a launcher that wrote only once or leaked memory, and a graph that replayed inputs frozen at capture | Outputs and workspace poisoned before every checked launch and replay; replays alternate two fixtures with different results; allocation measured around launches, capture, and replays | The real harness driven through a PyTorch stand-in: 23 of 28 new tests fail against the previous verifier; not yet run on NVIDIA hardware |
| The CUDA verifier could not check column-preserving layouts, `num_warps`, custom matrices, or three of four vLLM-qualified width-2048 pairings | Both options accepted and passed on; `--matrix FILE`; five default cells added | Schema, CLI, and matrix tests |
| A broken but installed Triton broke `import silkern`; a broken PyTorch or MLX crashed the verifiers and CUDA benchmarks | Any import failure marks the backend unavailable and names the cause | Subprocess tests with stub packages |
| `mx.compile(..., shapeless=True)` replayed stale shapes in the native MLX path: another request's pages, rows wider than 4096, writes past a stale buffer | The native path refuses shapeless compilation on its first call, as Metal already did | 21 sequences across both backends and both streams |
| Native MLX switched the process-wide default device, so concurrent CPU and GPU calls moved each other's work | Every operation names its stream; exact division by multiplication avoids MLX 0.32.3's `floor_divide` stream leak | A streamless default device traps any stray operation; a forced-interleaving race test |
| The Metal kernel compiled anew for each block-table shape, about 43 ms each | Batch, width, and table shape read at run time | Template invariance test; first call on a new shape 0.21 ms |
| The MLX verifier ran on the GPU stream only, with one table shape and batch 4 | CPU stream added; 1–9 requests, 1–48 pages, batch 8, widths 256, 257, and 4095 | 12 broken localizers through the real cell runner |
| The CUDA launchers disagreed on input aliasing | Read-only inputs may alias; outputs and workspace need storage of their own | Host tests; both mutation controls fail |
| Usage errors and an unavailable device both exited 2 | Usage errors exit 64 | CLI tests for both verifiers |
| A source archive built locally included `.claude/` agent worktrees | Explicit source-archive contents | Packaging test against `git ls-files` |

The revised sources passed these checks on October 8, 2026, on the same M5 Max
with macOS 26.6, Python 3.12.12, and MLX 0.32.3:

| Check | Result |
|---|---|
| CPU and Apple tests, warnings as errors | 936 passed; two CUDA modules and one CUDA-only documentation file skipped |
| Base suite without MLX | 540 passed |
| Property tests at 5,000 cases; MLX fuzzing at 600 cases and 15,000 seeded executions | No mismatch with the oracle |
| Apple conformance, `--require-device --repeats 16` | 90/90 cells ([record 10](../evidence/10-apple-mlx-post-audit/conformance.json)) |
| Same-process latency comparison with record 09's sources | Metal unchanged (median ratio 1.004); compiled MLX about 3% faster (0.968) |
| Source archive, wheel, and strict Twine checks | Passed; the archive lists only project files |
| Wheel installed outside the checkout | Base install verified with no optional runtime; with `[mlx]`, 90/90 conformance cells |
| Tests from the extracted source archive in a clean base environment | 535 passed; 11 optional-runtime and checkout-only cases skipped |
| Evidence, figures, and documentation | 48 checksums match; both figure generators reproduce identical bytes; links, anchors, and examples pass |

## Qualification boundary

The implementation-checkpoint checks, recorded before release, passed:

| Check | Result |
|---|---|
| Python 3.12.12, CPU + Apple tests, warnings treated as errors | 645 passed; two CUDA modules skipped because PyTorch is absent |
| Clean Python 3.11.16 base environment | 398 passed; optional accelerator and NumPy-only cases skipped/deselected |
| Clean Python 3.13.15 base environment | 398 passed; optional accelerator and NumPy-only cases skipped/deselected |
| Clean Python 3.14.7 base environment | 398 passed; optional accelerator and NumPy-only cases skipped/deselected |
| Ruff and whitespace checks | Passed |
| Source distribution and wheel build | Passed; strict distribution metadata checks, extracted-source tests, and base wheel/console smoke checks without accelerator dependencies |
| Evidence | Historical checksums retained; new source hashes and raw-sample medians verified; aggregate summary reproduced byte-for-byte |
| Documentation | Local links and heading anchors checked; README and diagrams inspected in light and dark themes |

The [current Apple record](../evidence/09-apple-mlx-consumer/) includes 48 passing
conformance cells with 16 repeatability evaluations per cell and all raw samples
from three fresh primitive benchmark processes and three separate consumer processes.
Current source hashes match those records. Primitive ratios span 1.10–1.21×
over compiled MLX; both complete-consumer geometries round to 1.12×. The maximum
consumer oracle error across qualification cases is below `2.91e-8`. All cells
and raw samples are retained, including sessions with visible latency variation.
The [initial Apple record](../evidence/07-apple-mlx/) remains intact as a prior
implementation checkpoint; its source hashes intentionally describe that earlier
implementation. Both earlier records and the current record retain byte-identical
[source snapshots](../evidence/09-apple-mlx-consumer/sources/) for every source
digest recorded by their benchmarks. These snapshots are stored as `.py.txt`
files, separate from the importable implementation. The [test suite](../tests/) also checks scan
boundaries, partial chunks, duplicate selections, negative page values,
non-compacting layouts, malformed metadata, and input immutability.

All five existing Triton kernel bodies are unchanged. CUDA host validation,
integration lifecycle, verification, and diagnostic code have changed. **No
NVIDIA experiments were run**, so those changes have host regression coverage
and code review, not a new CUDA device qualification. Historical NVIDIA results
retain their original provenance.

Continuous integration now defines Linux Python 3.11–3.14 checks, an isolated
wheel installation, evidence integrity, and an Apple test job. Hosted macOS
runners can lack Metal; their reports explicitly distinguish unavailable-device
skips from passing hardware qualification. Use `--require-device` when a device
qualification must succeed. Publication requires a successful remote workflow
on the release commit.
The Python 3.14 check qualifies the dependency-free surface; Apple runtime
qualification was performed with Python 3.12 and MLX 0.32.3.

Model-serving throughput, queueing, integration with MLX-LM, multi-device Apple
collectives, and whole-model determinism require additional application-specific
evidence. The API and documentation make these limits explicit rather than
promoting a microbenchmark to a deployment guarantee.
