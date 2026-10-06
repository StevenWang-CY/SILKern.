# Changelog

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
