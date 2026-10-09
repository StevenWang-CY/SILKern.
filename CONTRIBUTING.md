# Contributing to SILKern

[README](README.md) · [Architecture](docs/architecture.md) · [Contract](docs/contract.md) · [Evidence](docs/evidence.md)

The oracle is the specification. A backend change must reproduce
[`localize_reference`](silkern/contract.py) elementwise on the supported domain,
including counts, survivor order, duplicate selections, and padding. Propose a
contract change explicitly and explain its compatibility implications before
changing an implementation to follow it.

## Development setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
python -m pytest -q -m "not gpu and not mlx"
python -m ruff check .
```

The base suite must run without an accelerator runtime. Optional dependency
imports belong behind their backend boundary. A working Apple installation must
not need PyTorch or Triton; a contract-only installation must not need MLX.

| Change | Additional verification |
|---|---|
| Contract, validation, packaging | Base suite; malformed types/shapes and boundary cases; build/install smoke check |
| Apple / MLX implementation | `.[dev,mlx]`, Apple tests, `python -m silkern.mlx_verify --require-device`, benchmark if performance-relevant |
| CUDA / Triton implementation | On separately authorized CUDA hardware: GPU tests, `python -m silkern --require-device`, graph/guard checks |
| vLLM adapter | Adapter unit tests, binding/capture regressions, integration checks on the pinned upstream surface |
| Documentation or diagrams | `tests/test_docs.py` (links, anchors, runnable examples); source attribution; SVG light/dark readability |

Property and fuzz tests draw from fixed seeds and name the failing case, so a
failure replays exactly. For a deeper local run, raise their case counts:

```bash
SILKERN_PROPERTY_CASES=5000 python -m pytest -q tests/test_properties.py
SILKERN_MLX_FUZZ_CASES=600 python -m pytest -q tests/test_mlx_fuzz.py  # Apple silicon
```

A skipped accelerator suite is not hardware verification. Report which platforms
were actually run. Respect the experiment scope of the task: Apple and CPU work
does not require running NVIDIA experiments.

## Changing a backend

1. Add a meaningful regression case against the oracle or public API contract.
2. Implement the change without broadening unrelated integration claims.
3. Run the relevant host tests and available backend conformance suite.
4. For performance changes, preserve the before/after invocation, environment,
   correctness result, and full measurement output.
5. Document remaining hardware gaps and any changed caller preconditions.

Important cases include partial tiles, sparse survivors across a tile, repeated
tokens, all-invalid rows, fragmented tables, negative page entries, grouped
interleave, every rank, non-compacting mode, and the `dcp_size=1` bypass.
Accelerator arithmetic must remain within the documented signed 32-bit domain.

The CUDA kernel bodies descend from the historical evidence implementation.
Behavior changes require fresh platform qualification before historical results
can be attributed to them. Avoid cosmetic churn that obscures review; preserving
source text alone does not prove current-environment correctness.

MLX has a functional, lazy memory model. Do not claim allocation-free behavior or
CUDA graph compatibility for it. A benchmark must create fresh localization
outputs and evaluate them during each timed iteration.
Compiled calls must accept their arrays as dynamic arguments. Before timing,
change each input class independently and compare the new result with an
independent reference; an unchanged fixture cannot expose a captured constant.
Record these gates in the session artifact, separately from unit-test coverage.

## Changing an integration

The `QUALIFIED_*` constants in
[`integrations/vllm.py`](silkern/integrations/vllm.py) describe a checked
integration envelope. Widening a kernel's mathematical domain does not widen
that envelope automatically. Include the upstream revision, contract comparison,
new geometry conformance, and a real integration/capture result when extending
it. Preserve the adapter's rejection of changed bindings or unsupported geometry.

Apple localization is not yet an MLX-LM or distributed serving adapter. Adding
one requires explicit ownership, count/layout, lifecycle, and consumer tests.

## Measurements and evidence

New timing claims should include a machine-readable artifact, command, device and
runtime versions, geometry, timing definition, correctness gate, and source
revision or digest. Retain unfavorable outcomes and abstentions. Never relabel
an older artifact as a fresh validation run.

Artifacts added to [`evidence/`](evidence/) must be reflected in its checksum
manifest. From the repository root:

```bash
(cd evidence && shasum -a 256 -c SHA256SUMS)
```

Document summaries may round or calculate percentages from raw values, but must
link to the inputs and label the calculation. Keep per-call latency, complete
layer-stack segments, complete selected-attention consumers, and whole-model
latency distinct. A converter or consumer speedup alone does not establish a
serving speedup.

## Regenerating figures

```bash
python -m pip install -e ".[docs]"
python tools/render_figures.py   # charts from evidence/
python tools/render_diagrams.py  # explanatory diagrams, checked against the oracle
python tools/render_brand.py     # logo, wordmark, social card, and README icons
```

The chart generator reads checked-in measurements and runs no hardware
experiments. The Apple figure uses [record 09](evidence/09-apple-mlx-consumer/),
the source of the recorded latencies; its ratios, device, runtime, geometry, and
session labels are read from that record's `summary.json`,
`consumer-summary.json`, and session records. When a new record replaces it,
update the generator's `APPLE_RECORD`, the documentation tables, and their source
links together; retain prior measurement JSON files unchanged. `cover.png`, the
repository's social preview, is a 1280 × 640 raster of `assets/banner.svg`;
re-export it whenever the banner changes.

Both generators share [`tools/figure_style.py`](tools/figure_style.py). They
embed subsets of the fonts in [`tools/fonts`](tools/fonts/), keep text and
accessible descriptions as SVG text, and refuse to export overlapping labels,
connectors that cross labels, or text outside the canvas. Output is
byte-for-byte repeatable for a given fontTools release; run the generators
twice and confirm that `git status` reports no further change. Inspect light and
dark rendering at desktop and phone widths after changes; see
[the figure guide](assets/README.md).

## Checking a distribution

```bash
python -m build
python -m pip install twine
python -m twine check --strict dist/*
```

The source archive lists its contents explicitly under
`[tool.hatch.build.targets.sdist]` in `pyproject.toml`, so local tool folders
never leak into a release built from a working checkout. Add any new top-level
project file there; a packaging test fails until you do.

The full [README](README.md) is the repository guide. Package-index metadata uses
[a concise description](docs/package-description.md) with absolute source links,
so it renders without a checkout and does not duplicate performance numbers.
Keep that description's API and dependency statements consistent with the full
guide. Distribution CI builds a wheel through the source archive, checks its
metadata, and installs it in a fresh environment outside the checkout.

Apple CI requires a usable MLX CPU runtime before testing. Conformance runs and
its JSON artifact are retained after test or conformance failures when the
runtime setup succeeded. A report that says Metal is unavailable remains a skip,
not hardware qualification.

## Reporting an issue

Provide a minimal input or reproducible seed, exact geometry, package revision,
Python/runtime/device versions, and the complete error or conformance report.
Include expected arrays from `localize_reference` and the actual output where
possible. For a timing issue, include warmup, sample count, synchronization, and
whether compilation or host dispatch was measured.

Conformance failures take priority over optimization. Avoid including private
model inputs, prompts, or credentials in public reports.
