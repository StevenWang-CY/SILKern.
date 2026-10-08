# Changelog

## Unreleased

### Fixed

- Refuse `mx.compile(..., shapeless=True)` around the native MLX path on its
  first call, as the Metal path already did. Shapeless replays kept the first
  trace's table and row sizes and skipped validation, so they could return
  another request's pages, accept rows wider than 4096, and write past a stale
  output buffer.
- Create every native MLX operation on the selected stream instead of switching
  MLX's process-wide default device, which let concurrent CPU and GPU calls
  move each other's work. Integer division is done by exact multiplication,
  because MLX 0.32.3's integer `floor_divide` places part of its work on the
  default stream.
- Make `silkern.conformance()` fail on defects it previously passed: a captured
  graph that does no work, a launcher that writes only once or leaks memory, and
  a graph that replays inputs frozen at capture. Outputs and workspace are
  poisoned before every checked launch and replay, replays alternate their
  inputs between two fixtures with different results, and a new `allocation`
  check measures memory around every launch, the capture, and every replay.
  Verified on CPU with a stand-in for PyTorch; not yet run on NVIDIA hardware.
- Treat an installed but unimportable PyTorch, Triton, or MLX as an unavailable
  backend that names the import error. `import silkern` now survives a broken
  Triton, and neither verifier nor the CUDA benchmarks print a traceback.

### Changed

- The custom Metal kernel reads the batch, row width, and block-table shape at
  run time, so new request or page counts reuse the compiled kernel instead of
  compiling another (about 43 ms each on an M5 Max).
- One aliasing rule for both CUDA launchers: the read-only inputs may share
  storage, while outputs and workspace each need storage of their own.
  `localize_hierarchical` and the vLLM adapter now accept aliased inputs, as
  `localize_rowwise` did.
- Conformance geometries accept `compact_valid_to_front` and `num_warps`; the
  default CUDA matrix adds every vLLM-qualified width-2048 pairing and
  column-preserving cells with several ranks; `python -m silkern --matrix FILE`
  checks a deployment's own geometries.
- The MLX verifier also runs native MLX on the CPU stream, with seeded tables of
  1–9 requests and 1–48 pages, batch 8, and widths 256, 257, and 4095: 90 cells
  by default. Without Metal the CPU cells still run. Reports use schema 2 and
  record the imported module versions.
- Both verifiers exit 64 on a usage error, so exit 2 means only that the
  required device is unavailable.

### Tests

- Property tests check the oracle against an independent restatement of the
  contract and against metamorphic properties; differential fuzzing compares the
  MLX and Metal paths with the oracle at every scan boundary; the documentation's
  links, anchors, and Python examples are tested.
- Negative controls drive the real conformance harnesses with broken localizers
  and broken optional runtimes, and fail against the previous verifiers.

### Documentation and evidence

- Restructure the README: plain sections, a runnable quickstart, and a visible
  citation, with each figure beside the section it explains under a one-line
  caption. Full captions live in the guides.
- Typeset every figure as one system in embedded STIX Two subsets, replacing the
  `docs` extra's Matplotlib dependency with fontTools, and set the figures in
  print conventions: ink text, identity carried by cell fills, ruled operation
  stages, dashed memory outlines, and an Apple dot plot that keeps every
  geometry on one zero-based scale. Export rejects overlapping labels,
  connectors across text, text too light to read, and narrow crops that cut a
  label. The opening figure follows one row from global positions through the
  page table to compacted physical slots.
- Add record `10`: the post-audit verifier's 90-cell report and a same-process
  comparison with record 09's sources, in which the Metal kernel's latency is
  unchanged and compiled MLX is about 3% faster. Document that one of record
  09's three sessions ran slower for both arms.
- List the source distribution's contents explicitly, so local tool folders
  cannot leak into a release built from a working checkout.

## 2.0.0 — 2026-10-06

First tagged software release. This release includes the Apple/MLX and
hardening work developed under the unreleased `0.2.0` version. Recorded
measurements retain their original version labels and source hashes.

### Apple silicon and MLX

- Add `localize_mlx` with a custom Metal implementation and compositional MLX
  path, explicit backend selection, stream support, and lazy returned arrays.
- Accept MLX thread-local stream handles for independent worker graphs and
  preserve the caller's default stream and device.
- Preserve the existing localization semantics: stable survivor order, exact
  counts, grouped interleave, duplicate selections, negative page entries, and
  the single-rank compaction bypass.
- Mask invalid MLX request rows on the device; validate metadata without host
  array reads. Document signed 32-bit arithmetic preconditions.
- Synchronize Metal SIMD-group totals before the cross-group prefix scan and
  use unsigned Metal physical-slot arithmetic to avoid signed-overflow undefined
  behavior; supported physical results must still fit signed `int32`.
- Map Metal rows onto the grid's second dimension so large legal batches do
  not overflow the first dimension during launch construction.
- Add Apple conformance, oracle comparisons, benchmarks, and a runnable
  [sparse-gather example](examples/mlx_sparse_gather.py).
- Add a tested single-head selected-attention example with safe gathers,
  shared softmax normalization, empty selections, and logical-shard recombination.

### Reliability and packaging

- Tighten integer, boolean, shape, geometry, and workspace validation.
- Reject unresolved CUDA lazy-negation views and overlapping byte ranges across
  distinct storage objects; reject adapter preparation during capture on either
  the caller's current device or the target device.
- Bound CUDA row batches before launch or adapter allocation to match the
  device grid's signed 32-bit extent.
- Harden vLLM adapter preparation and retain bound inputs for its fixed-buffer
  lifetime; reject unsafe rebinding and unsupported geometry.
- Keep accelerator dependencies optional and provide separate Apple and CUDA
  verification entry points.
- Expand base-package, isolated-wheel, and Apple CI checks. Metal availability
  is reported explicitly; hosted CI alone does not qualify Apple GPU hardware.
- Require a working MLX CPU runtime in Apple CI, use a supported macOS runner
  and Node 24 actions, and document CUDA stream ordering and captured-storage
  lifetime requirements.
- Preserve Apple conformance artifacts after failed checks and validate package
  metadata, including a package-index description with absolute documentation links.
- Validate CUDA diagnostic inputs and oracle agreement before observations;
  these diagnostic changes have host regression coverage and remain unqualified
  on NVIDIA hardware in this update.

### Documentation and evidence

- Embed the consumer figure in the README and add narrow layouts for every
  figure. Preserve tensor and memory glyphs at readable sizes in side panels
  and on phones; verify GitHub's rendered responsive image sources.
- Add architecture and Apple integration guides with separate memory-ownership
  guarantees for MLX and CUDA.
- Add a practical consumer guide for safe gathers, prefix versus column masks,
  empty selections, and selected-attention normalization across logical shards.
- Rework the README and vector diagrams; correct canonical repository links to
  `StevenWang-CY/SILKern.`.
- Add reproducible tensor, page-table, cache, and consumer diagrams with labeled
  survivor paths. Present Apple latency in grouped bars with zero baselines,
  direct values, and a separate consumer scale; preserve archived CUDA intervals.
- Unify all six figures with shared typography and theme tokens. Use bracketed
  vectors, indexed K/V matrices, explicit mask paths, and consistent mathematical
  typography. Organize localization and consumption into three-panel figures;
  move explanatory detail into captions and present Apple latency in a compact
  two-by-two chart with aligned scales and direct labels.
- Clarify that historical H100 converter timings cover a complete 48-layer
  segment, retain the 64K rowwise regression, and remove unsupported universal
  dispatch and serving-performance claims.
- Separate historical NVIDIA artifacts from current host and Apple validation.
  No NVIDIA experiments were run for this update.
- Retain earlier Apple measurements and add record 09 with 16-repeat conformance,
  per-session compiled changing-input gates, and three fresh sessions for both
  localization and a complete selected-attention consumer. Current tables and
  figures use this checkpoint's source hashes and measured ratios.

## 0.1.0

Initial repository release of the Python localization oracle, deterministic
rowwise and hierarchical CUDA/Triton implementations, fixed-buffer vLLM adapter,
verification tools, and archived experimental evidence.
