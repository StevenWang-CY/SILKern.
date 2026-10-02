"""One call that tries to prove the kernels wrong on *your* stack.

This CUDA verifier checks that the Triton implementations reproduce
:func:`silkern.contract.localize_reference` exactly at fixed addresses without
allocating. Run it on the deployment stack before relying on those guarantees.
The functional Apple API has its own verifier in :mod:`silkern.mlx_verify`.

    >>> import silkern
    >>> report = silkern.conformance()      # doctest: +SKIP
    >>> report.ok                         # doctest: +SKIP
    True

:func:`conformance` sweeps a randomized geometry matrix and runs five checks
per cell:

``oracle``
    Every localized array and count equals the pure-Python oracle, elementwise.
``order``
    The valid prefix equals the input row filtered by rank ownership, in input
    order. Set equality is not enough -- an atomic converter passes set
    equality and still scrambles the order. This is the check that matters.
``determinism``
    Repeated launches on identical input produce a bytewise-identical output
    buffer. A converter whose output depends on tile-reservation race order
    fails here.
``replay``
    The launch is captured in a CUDA graph and replayed; every buffer pointer
    is unchanged and ``torch.cuda.memory_allocated()`` does not grow.
``immutability``
    Inputs are unmodified, and canaries placed around every output buffer
    are unchanged -- i.e. nothing was written out of bounds.

A failing cell is reported, not raised. The report tells you which geometry
failed and which check, so a narrowed geometry is an actionable result rather
than a stack trace.
"""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from numbers import Integral

from silkern.contract import (
    DEFAULT_TILE_SIZE,
    SUPPORTED_TILE_SIZES,
    localize_reference,
)
from silkern.errors import LocalizationError
from silkern.kernels import _validate_launch_config, localize_hierarchical, localize_rowwise
from silkern.workspace import workspace_shapes

CHECKS = ("oracle", "order", "determinism", "replay", "immutability")
ARMS = ("row_stable", "hierarchical_stable")

#: Geometry matrix used when ``conformance()`` is called with no arguments.
#: Chosen to cross the interesting boundaries: width below/at/above a tile,
#: non-power-of-two widths, both qualified page sizes, DCP degrees 1/2/4/8, the
#: last rank of each degree (most likely to expose an off-by-one), and grouped
#: interleave.
DEFAULT_MATRIX: tuple[dict[str, int], ...] = (
    {"width": 1, "block_size": 64, "dcp_size": 1, "dcp_rank": 0, "dcp_interleave": 1},
    {"width": 63, "block_size": 32, "dcp_size": 2, "dcp_rank": 1, "dcp_interleave": 1},
    {"width": 128, "block_size": 64, "dcp_size": 2, "dcp_rank": 0, "dcp_interleave": 1},
    {"width": 129, "block_size": 64, "dcp_size": 4, "dcp_rank": 3, "dcp_interleave": 1},
    {"width": 513, "block_size": 32, "dcp_size": 4, "dcp_rank": 1, "dcp_interleave": 4},
    {"width": 1024, "block_size": 64, "dcp_size": 8, "dcp_rank": 7, "dcp_interleave": 1},
    {"width": 2048, "block_size": 64, "dcp_size": 2, "dcp_rank": 1, "dcp_interleave": 2},
    {"width": 2048, "block_size": 64, "dcp_size": 4, "dcp_rank": 2, "dcp_interleave": 1},
    {"width": 4096, "block_size": 64, "dcp_size": 2, "dcp_rank": 0, "dcp_interleave": 1},
)

GUARD_ELEMENTS = 64
GUARD_VALUE = 0x5A5A5A5A
REPLAY_COUNT = 32
DETERMINISM_REPEATS = 8
MAX_FIXTURE_ELEMENTS = 4_194_304


@dataclass
class CellReport:
    """Outcome of one (arm, geometry) cell."""

    arm: str
    geometry: dict[str, int]
    checks: dict[str, bool] = field(default_factory=dict)
    error: str | None = None

    @property
    def ok(self) -> bool:
        return (
            self.error is None
            and set(self.checks) == set(CHECKS)
            and all(self.checks.get(name) is True for name in CHECKS)
        )

    def failures(self) -> list[str]:
        if self.error is not None:
            return [f"error: {self.error}"]
        missing_or_failed = [name for name in CHECKS if self.checks.get(name) is not True]
        return missing_or_failed + [
            f"unexpected check: {name}" for name in self.checks if name not in CHECKS
        ]


@dataclass
class ConformanceReport:
    """Aggregate outcome. Truthy iff every cell passed every check."""

    cells: list[CellReport] = field(default_factory=list)
    device: str = "unknown"
    skipped: str | None = None

    @property
    def ok(self) -> bool:
        return self.skipped is None and bool(self.cells) and all(c.ok for c in self.cells)

    def __bool__(self) -> bool:
        return self.ok

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable result, including skipped and failed checks."""
        return {
            "backend": "cuda",
            "device": self.device,
            "ok": self.ok,
            "skipped": self.skipped,
            "cells": [
                {
                    "arm": cell.arm,
                    "geometry": dict(cell.geometry),
                    "checks": dict(cell.checks),
                    "error": cell.error,
                    "ok": cell.ok,
                    "failures": cell.failures(),
                }
                for cell in self.cells
            ],
        }

    def summary(self) -> str:
        if self.skipped is not None:
            return f"conformance skipped: {self.skipped}"
        if not self.cells:
            return f"silkern conformance on {self.device}: no cells executed (failed)"
        passed = sum(1 for c in self.cells if c.ok)
        head = (
            f"silkern conformance on {self.device}: "
            f"{passed}/{len(self.cells)} cells passed "
            f"({', '.join(CHECKS)})"
        )
        if passed == len(self.cells):
            return head
        lines = [head, ""]
        for cell in self.cells:
            if cell.ok:
                continue
            geometry = " ".join(f"{k}={v}" for k, v in cell.geometry.items())
            lines.append(f"  FAIL {cell.arm:22s} {geometry}")
            for failure in cell.failures():
                lines.append(f"       - {failure}")
        return "\n".join(lines)

    def __str__(self) -> str:  # pragma: no cover - convenience
        return self.summary()


def _random_case(
    *,
    width: int,
    batch: int,
    block_size: int,
    dcp_size: int,
    seed: int,
) -> tuple[list[int], list[list[int]], list[list[int]]]:
    """Build one adversarial case.

    Deliberately includes negative tokens, tokens past the end of the page
    table, a fragmented (non-identity, non-monotonic) page table, several
    requests sharing one batch, and the two extreme in-range positions.
    """
    rng = random.Random(seed)
    table_width = 17
    requests = 4
    req_ids = [(row * 3 + 1) % requests for row in range(batch)]
    block_table = [
        [rng.randrange(-2, 4096) for _ in range(table_width)] for _ in range(requests)
    ]
    global_limit = block_size * table_width * dcp_size
    rows: list[list[int]] = []
    for _row in range(batch):
        values: list[int] = []
        for column in range(width):
            draw = rng.random()
            if draw < 0.12:
                values.append(rng.choice((-9, -3, -1)))
            elif draw < 0.20:
                values.append(global_limit + rng.randrange(block_size * dcp_size + 1))
            elif column == 0:
                values.append(0)
            elif column == width - 1:
                values.append(max(global_limit - 1, 0))
            else:
                values.append(rng.randrange(max(global_limit, 1)))
        rows.append(values)
    return req_ids, block_table, rows


def _expected_order(
    row: Sequence[int],
    table_row: Sequence[int],
    *,
    block_size: int,
    dcp_size: int,
    dcp_rank: int,
    dcp_interleave: int,
) -> list[int]:
    """The prefix a *stable* converter must produce, derived independently.

    Computed as a filter over the input row rather than reusing the oracle's
    control flow, so agreement is not an artifact of shared code.
    """
    kept: list[int] = []
    for raw in row:
        token = int(raw)
        if token < 0:
            continue
        if (token // dcp_interleave) % dcp_size != dcp_rank:
            continue
        local = (token // (dcp_size * dcp_interleave)) * dcp_interleave + (
            token % dcp_interleave
        )
        logical_block, offset = divmod(local, block_size)
        if logical_block >= len(table_row):
            continue
        kept.append(int(table_row[logical_block]) * block_size + offset)
    return kept


def _guarded(torch, shape, device):
    """An int32 tensor with nonzero guard elements on both sides.

    Returns ``(view, storage)``. Out-of-bounds writes land in the guards, so a
    kernel that overruns its row is caught even when the in-bounds values
    happen to be correct.
    """
    total = 1
    for dim in shape:
        total *= dim
    storage = torch.full(
        (total + 2 * GUARD_ELEMENTS,), GUARD_VALUE, dtype=torch.int32, device=device
    )
    view = storage[GUARD_ELEMENTS : GUARD_ELEMENTS + total].view(*shape)
    return view, storage


def _guards_clean(storage) -> bool:
    return bool(
        (storage[:GUARD_ELEMENTS] == GUARD_VALUE).all()
        and (storage[-GUARD_ELEMENTS:] == GUARD_VALUE).all()
    )


def _run_cell(
    torch,
    arm: str,
    geometry: dict[str, int],
    *,
    batch: int,
    seed: int,
    tile_size: int,
    device: str,
) -> CellReport:
    report = CellReport(arm=arm, geometry={})
    try:
        if not isinstance(geometry, Mapping):
            raise LocalizationError("matrix geometry must be a mapping")
        # Integral scalar subclasses are supported by launch validation. Record
        # them as Python ints so machine-readable reports remain serializable.
        report.geometry = {
            key: int(value) if isinstance(value, Integral) and not isinstance(value, bool) else value
            for key, value in geometry.items()
        }
        geometry = report.geometry
        required = {"width", "block_size", "dcp_size", "dcp_rank", "dcp_interleave"}
        if set(geometry) != required:
            raise LocalizationError(f"matrix geometry must contain {', '.join(sorted(required))}")
        width = geometry["width"]
        block_size = geometry["block_size"]
        dcp_size = geometry["dcp_size"]
        dcp_rank = geometry["dcp_rank"]
        dcp_interleave = geometry["dcp_interleave"]

        # Reject invalid geometry before generating values or allocating device
        # tensors; otherwise a typo in width can allocate an enormous buffer.
        workspace_shapes(batch, width, tile_size=tile_size)
        block_size, dcp_size, dcp_rank, dcp_interleave, _ = _validate_launch_config(
            block_size, dcp_size, dcp_rank, dcp_interleave, True, 4
        )
        width = int(width)
        if batch * width > MAX_FIXTURE_ELEMENTS:
            raise LocalizationError(
                f"diagnostic fixtures must not exceed {MAX_FIXTURE_ELEMENTS} elements"
            )
        # The random fixture includes one ownership period past its 17-page
        # table and physical page IDs up to 4095, all materialized as int32.
        if max(block_size * dcp_size * 18, block_size * 4096) > 2**31 - 1:
            raise LocalizationError("diagnostic indices and physical slots must fit int32")

        req_ids, block_table, rows = _random_case(
            width=width,
            batch=batch,
            block_size=block_size,
            dcp_size=dcp_size,
            seed=seed,
        )

        req = torch.tensor(req_ids, dtype=torch.int32, device=device)
        table = torch.tensor(block_table, dtype=torch.int32, device=device)
        tokens = torch.tensor(rows, dtype=torch.int32, device=device)
        tokens_before = tokens.clone()
        table_before = table.clone()
        req_before = req.clone()

        out, out_storage = _guarded(torch, (batch, width), device)
        counts, counts_storage = _guarded(torch, (batch,), device)

        workspace: dict[str, object] = {}
        guards = [out_storage, counts_storage]
        if arm == "hierarchical_stable":
            for name, shape in workspace_shapes(batch, width, tile_size=tile_size).items():
                view, storage = _guarded(torch, shape, device)
                workspace[name] = view
                guards.append(storage)

        def launch() -> None:
            if arm == "row_stable":
                localize_rowwise(
                    req,
                    table,
                    tokens,
                    out,
                    counts,
                    block_size=block_size,
                    dcp_size=dcp_size,
                    dcp_rank=dcp_rank,
                    dcp_interleave=dcp_interleave,
                )
            else:
                localize_hierarchical(
                    req,
                    table,
                    tokens,
                    out,
                    counts,
                    workspace["mapped"],
                    workspace["local_positions"],
                    workspace["tile_counts"],
                    workspace["tile_offsets"],
                    block_size=block_size,
                    dcp_size=dcp_size,
                    dcp_rank=dcp_rank,
                    dcp_interleave=dcp_interleave,
                    tile_size=tile_size,
                )

        launch()
        torch.cuda.synchronize()
        observed_out = out.cpu().tolist()
        observed_counts = counts.cpu().tolist()

        expected_out, expected_counts = localize_reference(
            req_ids,
            block_table,
            rows,
            block_size=block_size,
            dcp_size=dcp_size,
            dcp_rank=dcp_rank,
            dcp_interleave=dcp_interleave,
        )
        report.checks["oracle"] = (
            observed_out == expected_out and observed_counts == expected_counts
        )

        # dcp_size == 1 bypasses compaction by contract, so there is no prefix
        # to order-check; the oracle check above already covers it.
        if dcp_size == 1:
            report.checks["order"] = observed_out == expected_out
        else:
            order_ok = True
            for row_id, row in enumerate(rows):
                want = _expected_order(
                    row,
                    block_table[req_ids[row_id]],
                    block_size=block_size,
                    dcp_size=dcp_size,
                    dcp_rank=dcp_rank,
                    dcp_interleave=dcp_interleave,
                )
                got = observed_out[row_id][: observed_counts[row_id]]
                tail = observed_out[row_id][observed_counts[row_id] :]
                if got != want or any(value != -1 for value in tail):
                    order_ok = False
                    break
            report.checks["order"] = order_ok

        first = out.clone()
        first_counts = counts.clone()
        stable = True
        for _ in range(DETERMINISM_REPEATS):
            launch()
            torch.cuda.synchronize()
            if not (bool(torch.equal(out, first)) and bool(torch.equal(counts, first_counts))):
                stable = False
                break
        report.checks["determinism"] = stable

        pointers = [t.data_ptr() for t in (req, table, tokens, out, counts)]
        pointers += [t.data_ptr() for t in workspace.values()]
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                launch()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            launch()
        allocated_before = torch.cuda.memory_allocated()
        for _ in range(REPLAY_COUNT):
            graph.replay()
        torch.cuda.synchronize()
        after_pointers = [t.data_ptr() for t in (req, table, tokens, out, counts)]
        after_pointers += [t.data_ptr() for t in workspace.values()]
        report.checks["replay"] = (
            torch.cuda.memory_allocated() == allocated_before
            and after_pointers == pointers
            and out.cpu().tolist() == expected_out
            and counts.cpu().tolist() == expected_counts
        )

        report.checks["immutability"] = (
            bool(torch.equal(tokens, tokens_before))
            and bool(torch.equal(table, table_before))
            and bool(torch.equal(req, req_before))
            and all(_guards_clean(storage) for storage in guards)
        )
    except LocalizationError as exc:
        report.error = f"{type(exc).__name__}: {exc}"
    except Exception as exc:  # pragma: no cover - surfaced, not swallowed
        report.error = f"{type(exc).__name__}: {exc}"

    for name in CHECKS:
        report.checks.setdefault(name, False)
    return report


def conformance(
    matrix: Sequence[dict[str, int]] | None = None,
    *,
    arms: Sequence[str] = ARMS,
    batch: int = 5,
    tile_size: int = DEFAULT_TILE_SIZE,
    seed: int = 20260803,
) -> ConformanceReport:
    """Run the conformance matrix on the local device.

    Args:
        matrix: Geometry dicts with ``width``, ``block_size``, ``dcp_size``,
            ``dcp_rank``, ``dcp_interleave``. Defaults to :data:`DEFAULT_MATRIX`.
            Pass your own deployment geometry before trusting the kernels on it.
        arms: Which implementations to check.
        batch: Selection rows per cell. Rows share a small pool of request ids,
            so request routing is exercised, not bypassed.
        tile_size: Tile size for the hierarchical arm.
        seed: Base seed; each cell derives its own, so runs are reproducible.

    Returns:
        A :class:`ConformanceReport`. It is falsy if anything failed, and
        ``report.summary()`` names the geometry and the check. Missing CUDA or
        Triton yields a skipped (falsy) report rather than an exception, so this
        is safe to call unconditionally in CI.

    Raises:
        LocalizationError: if a sweep option is invalid. A bad
            *geometry* is reported as a failed cell instead, so one unsupported
            entry in a long matrix does not discard the rest of the sweep.
    """
    # Validate the request even on a machine without optional GPU dependencies.
    # A missing backend must not silently accept a misspelled arm or an empty run.
    if isinstance(arms, str):
        raise LocalizationError("arms must be a nonempty sequence of implementation names")
    arms = tuple(arms)
    if not arms:
        raise LocalizationError("arms must be a nonempty sequence of implementation names")
    for arm in arms:
        if arm not in ARMS:
            raise LocalizationError(f"unknown arm: {arm}")
    arms = tuple(dict.fromkeys(arms))
    if not isinstance(batch, Integral) or isinstance(batch, bool) or batch < 1:
        raise LocalizationError("batch must be a positive integer")
    if (
        not isinstance(tile_size, Integral)
        or isinstance(tile_size, bool)
        or tile_size not in SUPPORTED_TILE_SIZES
    ):
        raise LocalizationError(f"tile_size must be one of {SUPPORTED_TILE_SIZES}")
    if not isinstance(seed, Integral) or isinstance(seed, bool):
        raise LocalizationError("seed must be an integer")
    batch, tile_size, seed = int(batch), int(tile_size), int(seed)
    if batch > MAX_FIXTURE_ELEMENTS:
        raise LocalizationError(
            f"diagnostic fixtures must not exceed {MAX_FIXTURE_ELEMENTS} elements"
        )
    cells = tuple(DEFAULT_MATRIX if matrix is None else matrix)
    if not cells:
        raise LocalizationError("matrix must contain at least one geometry")

    report = ConformanceReport()
    try:
        import torch
    except ModuleNotFoundError:
        report.skipped = "torch is not installed"
        return report
    try:
        import triton  # noqa: F401
    except ModuleNotFoundError:
        report.skipped = "triton is not installed"
        return report
    if not torch.cuda.is_available():
        report.skipped = "no CUDA device is available"
        return report

    # All allocations, streams, and counters below use the current CUDA device.
    report.device = torch.cuda.get_device_name(torch.cuda.current_device())
    for arm in arms:
        for index, geometry in enumerate(cells):
            report.cells.append(
                _run_cell(
                    torch,
                    arm,
                    geometry,
                    batch=batch,
                    seed=seed + index,
                    tile_size=tile_size,
                    device="cuda",
                )
            )
    return report


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI
    """``python -m silkern`` -- exits nonzero if any cell fails."""
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Run the silkern conformance matrix.",
        epilog=(
            "Exits 1 if any cell fails and 0 otherwise, including when the sweep "
            "is skipped for want of a CUDA device or Triton. Use --require-device "
            "to treat an unavailable backend as a failure (exit 2)."
        ),
    )
    parser.add_argument(
        "--batch", type=int, default=5, help="selection rows per cell (>= 1)"
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=DEFAULT_TILE_SIZE,
        choices=SUPPORTED_TILE_SIZES,
    )
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--json", action="store_true", help="emit a machine-readable report")
    parser.add_argument(
        "--require-device", action="store_true", help="exit 2 if CUDA verification is skipped"
    )
    parser.add_argument(
        "--arm",
        action="append",
        choices=list(ARMS),
        help="restrict to one arm; repeatable",
    )
    args = parser.parse_args(argv)
    if args.batch < 1:
        parser.error("--batch must be at least 1")

    try:
        report = conformance(
            arms=tuple(dict.fromkeys(args.arm)) if args.arm else ARMS,
            batch=args.batch,
            tile_size=args.tile_size,
            seed=args.seed,
        )
    except LocalizationError as exc:
        parser.error(str(exc))
    print(json.dumps(report.to_dict(), indent=2) if args.json else report.summary())
    if report.skipped is not None:
        return 2 if args.require_device else 0
    return 0 if report.ok else 1


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())
