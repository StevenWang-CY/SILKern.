"""One call that tries to prove the kernels wrong on *your* stack.

This CUDA verifier checks that the Triton implementations reproduce
:func:`silkern.contract.localize_reference` exactly, read their bound input
buffers afresh on every launch and graph replay, and allocate no device memory
while launching, capturing, or replaying. Run it on the deployment stack before
relying on those guarantees. The functional Apple API has its own verifier in
:mod:`silkern.mlx_verify`.

    >>> import silkern
    >>> report = silkern.conformance()      # doctest: +SKIP
    >>> report.ok                         # doctest: +SKIP
    True

:func:`conformance` sweeps a randomized geometry matrix and runs six checks
per cell. Before every checked launch and every replay, each output and
workspace element is overwritten with :data:`POISON`, a value no correct launch
writes, so a launch that skips its work cannot pass on an earlier result.

``oracle``
    Every localized array and count equals the pure-Python oracle, elementwise.
``order``
    Each row matches its layout derived independently from the input row. When
    compacting, the valid prefix is the row filtered by rank ownership and
    table bounds, in input order. Set equality is not enough -- an atomic
    converter passes set equality and still scrambles the order. This is the
    check that matters. Without compaction, every column keeps its own result.
``determinism``
    Repeated launches on identical input, each into poisoned buffers, reproduce
    the first launch's output bytewise. A converter whose output depends on
    tile-reservation race order fails here, as does one that writes only once.
``replay``
    The launch is captured in a CUDA graph and replayed. Before each replay the
    bound input buffers are rewritten in place, alternating between two
    fixtures whose results differ, and each replay must produce its fixture's
    oracle result. A graph that does no work, or that replays inputs it
    captured instead of reading its buffers, fails here.
``allocation``
    ``torch.cuda.memory_allocated()`` and its peak never rise during an eager
    launch, the captured launch, or a replay. The captured launch is measured
    inside the graph context, which excludes PyTorch's own capture bookkeeping.
``immutability``
    Inputs keep the values last written to them, and canaries placed around
    every output and workspace buffer are unchanged -- i.e. nothing was
    written out of bounds.

The allocation check reads device-wide allocator statistics and resets their
peak, so run a sweep while nothing else allocates on the device.

A failing cell is reported, not raised. The report tells you which geometry
failed and which check, so a narrowed geometry is an actionable result rather
than a stack trace.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from numbers import Integral

from silkern.contract import (
    DEFAULT_TILE_SIZE,
    SUPPORTED_TILE_SIZES,
    localize_reference,
)
from silkern.errors import LocalizationError
from silkern.kernels import (
    _import_failure,
    _validate_launch_config,
    localize_hierarchical,
    localize_rowwise,
)
from silkern.workspace import workspace_shapes

CHECKS = ("oracle", "order", "determinism", "replay", "allocation", "immutability")
ARMS = ("row_stable", "hierarchical_stable")

#: Every matrix geometry names these launch parameters ...
GEOMETRY_KEYS = ("width", "block_size", "dcp_size", "dcp_rank", "dcp_interleave")
#: ... and may also set these. An omitted one keeps the launcher's default:
#: compaction on, and ``num_warps`` 8 (rowwise) or 4 (hierarchical).
OPTIONAL_GEOMETRY_KEYS = ("compact_valid_to_front", "num_warps")

#: Geometry matrix used when ``conformance()`` is called with no arguments.
#: Chosen to cross the interesting boundaries: width below/at/above a tile,
#: non-power-of-two widths, both qualified page sizes, DCP degrees 1/2/4/8, the
#: last rank of each degree (most likely to expose an off-by-one), grouped
#: interleave, the column-preserving layout across ranks, and each pairing of
#: ``dcp_size`` and ``block_size`` the vLLM adapter is qualified on (one rank
#: apiece; pass a matrix to check every rank of a deployment).
DEFAULT_MATRIX: tuple[dict[str, int | bool], ...] = (
    {"width": 1, "block_size": 64, "dcp_size": 1, "dcp_rank": 0, "dcp_interleave": 1},
    {"width": 63, "block_size": 32, "dcp_size": 2, "dcp_rank": 1, "dcp_interleave": 1},
    {"width": 128, "block_size": 64, "dcp_size": 2, "dcp_rank": 0, "dcp_interleave": 1},
    {"width": 129, "block_size": 64, "dcp_size": 4, "dcp_rank": 3, "dcp_interleave": 1},
    {"width": 513, "block_size": 32, "dcp_size": 4, "dcp_rank": 1, "dcp_interleave": 4},
    {"width": 1024, "block_size": 64, "dcp_size": 8, "dcp_rank": 7, "dcp_interleave": 1},
    {"width": 2048, "block_size": 64, "dcp_size": 2, "dcp_rank": 1, "dcp_interleave": 2},
    {"width": 2048, "block_size": 64, "dcp_size": 4, "dcp_rank": 2, "dcp_interleave": 1},
    {"width": 4096, "block_size": 64, "dcp_size": 2, "dcp_rank": 0, "dcp_interleave": 1},
    # With the block_size=64, dcp_size=4 entry: the vLLM-qualified pairings.
    {"width": 2048, "block_size": 32, "dcp_size": 2, "dcp_rank": 0, "dcp_interleave": 1},
    {"width": 2048, "block_size": 64, "dcp_size": 2, "dcp_rank": 1, "dcp_interleave": 1},
    {"width": 2048, "block_size": 32, "dcp_size": 4, "dcp_rank": 3, "dcp_interleave": 1},
    # Column-preserving layouts that, unlike dcp_size == 1, drop foreign tokens.
    {"width": 129, "block_size": 64, "dcp_size": 2, "dcp_rank": 1, "dcp_interleave": 1,
     "compact_valid_to_front": False},
    {"width": 513, "block_size": 32, "dcp_size": 4, "dcp_rank": 2, "dcp_interleave": 2,
     "compact_valid_to_front": False},
)

GUARD_ELEMENTS = 64
GUARD_VALUE = 0x5A5A5A5A
#: Written over every output and workspace element before each checked launch
#: and each replay. No correct launch writes it: fixture page entries are at
#: least -2 and the fixture bounds keep ``block_size`` below 2**19, so slots
#: stay above -2**20; counts and workspace entries are at least -1.
POISON = -(2**31)
REPLAY_COUNT = 32
DETERMINISM_REPEATS = 8
MAX_FIXTURE_ELEMENTS = 4_194_304

_Case = tuple[list[int], list[list[int]], list[list[int]]]
_Result = tuple[list[list[int]], list[int]]


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
) -> _Case:
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


def _second_case(
    first: _Case,
    first_result: _Result,
    *,
    seed: int,
    block_size: int,
    dcp_size: int,
    dcp_rank: int,
    dcp_interleave: int,
    compact_valid_to_front: bool,
) -> tuple[_Case, _Result]:
    """A fixture shaped like ``first``, with fresh pages and tokens and a different result.

    Replays rewrite the bound buffers in place, alternating between the two
    fixtures, so a graph that replays inputs it captured instead of reading
    its buffers produces the other fixture's result. Every row's request id is
    rotated to another table row, and row 0 starts with an owned token in
    logical page 0; should the two results still coincide, moving that page
    makes them differ.
    """
    req_ids, block_table, rows = _random_case(
        width=len(first[2][0]),
        batch=len(first[2]),
        block_size=block_size,
        dcp_size=dcp_size,
        seed=seed,
    )
    req_ids = [(request + 1) % len(block_table) for request in req_ids]
    rows[0][0] = dcp_rank * dcp_interleave
    geometry = {
        "block_size": block_size,
        "dcp_size": dcp_size,
        "dcp_rank": dcp_rank,
        "dcp_interleave": dcp_interleave,
        "compact_valid_to_front": compact_valid_to_front,
    }
    result = localize_reference(req_ids, block_table, rows, **geometry)
    if result == first_result:
        page = block_table[req_ids[0]]
        page[0] = page[0] - 1 if page[0] == 4095 else page[0] + 1
        result = localize_reference(req_ids, block_table, rows, **geometry)
    return (req_ids, block_table, rows), result


def _owned_slots(
    row: Sequence[int],
    table_row: Sequence[int],
    *,
    block_size: int,
    dcp_size: int,
    dcp_rank: int,
    dcp_interleave: int,
) -> list[int | None]:
    """Each column's rank-local physical slot, or ``None``, derived independently.

    Computed as a filter over the input row rather than reusing the oracle's
    control flow, so agreement is not an artifact of shared code.
    """
    slots: list[int | None] = []
    for raw in row:
        token = int(raw)
        slot = None
        if token >= 0 and (token // dcp_interleave) % dcp_size == dcp_rank:
            local = (token // (dcp_size * dcp_interleave)) * dcp_interleave + (
                token % dcp_interleave
            )
            logical_block, offset = divmod(local, block_size)
            if logical_block < len(table_row):
                slot = int(table_row[logical_block]) * block_size + offset
        slots.append(slot)
    return slots


def _expected_order(
    row: Sequence[int],
    table_row: Sequence[int],
    *,
    block_size: int,
    dcp_size: int,
    dcp_rank: int,
    dcp_interleave: int,
) -> list[int]:
    """The prefix a *stable* converter must produce, derived independently."""
    slots = _owned_slots(
        row,
        table_row,
        block_size=block_size,
        dcp_size=dcp_size,
        dcp_rank=dcp_rank,
        dcp_interleave=dcp_interleave,
    )
    return [slot for slot in slots if slot is not None]


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


def _allocation_free(torch, run: Callable[[], object]) -> bool:
    """Run ``run``; report whether allocated device memory never rose above its start.

    The peak is read as well as the final level, so a buffer allocated and
    released inside ``run`` still counts. Both are host-side allocator
    statistics, which may be read while a CUDA graph is capturing.
    """
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    run()
    return (
        torch.cuda.memory_allocated() <= before
        and torch.cuda.max_memory_allocated() <= before
    )


def _check_geometry(
    geometry: Mapping[str, int | bool], *, batch: int, tile_size: int
) -> tuple[int, dict[str, int | bool]]:
    """Validate one matrix geometry before generating values or allocating.

    Returns the row width and the launcher's keyword options as Python scalars.
    ``num_warps`` is among them only when the geometry sets it, so each
    launcher otherwise keeps its own default.
    """
    if not isinstance(geometry, Mapping):
        raise LocalizationError("matrix geometry must be a mapping")
    missing = [key for key in GEOMETRY_KEYS if key not in geometry]
    if missing:
        raise LocalizationError(f"matrix geometry is missing {', '.join(missing)}")
    unknown = sorted(str(key) for key in geometry
                     if key not in GEOMETRY_KEYS + OPTIONAL_GEOMETRY_KEYS)
    if unknown:
        raise LocalizationError(
            f"matrix geometry has unknown keys {', '.join(unknown)}; "
            f"beyond {', '.join(GEOMETRY_KEYS)} it may set only "
            f"{', '.join(OPTIONAL_GEOMETRY_KEYS)}"
        )

    # Reject invalid geometry before generating values or allocating device
    # tensors; otherwise a typo in width can allocate an enormous buffer.
    workspace_shapes(batch, geometry["width"], tile_size=tile_size)
    compact = geometry.get("compact_valid_to_front", True)
    # Any supported value validates an omitted num_warps; it is not passed on.
    block_size, dcp_size, dcp_rank, dcp_interleave, num_warps = _validate_launch_config(
        geometry["block_size"],
        geometry["dcp_size"],
        geometry["dcp_rank"],
        geometry["dcp_interleave"],
        compact,
        geometry.get("num_warps", 4),
    )
    width = int(geometry["width"])
    if batch * width > MAX_FIXTURE_ELEMENTS:
        raise LocalizationError(
            f"diagnostic fixtures must not exceed {MAX_FIXTURE_ELEMENTS} elements"
        )
    # The random fixture includes one ownership period past its 17-page
    # table and physical page IDs up to 4095, all materialized as int32.
    if max(block_size * dcp_size * 18, block_size * 4096) > 2**31 - 1:
        raise LocalizationError("diagnostic indices and physical slots must fit int32")
    options: dict[str, int | bool] = {
        "block_size": block_size,
        "dcp_size": dcp_size,
        "dcp_rank": dcp_rank,
        "dcp_interleave": dcp_interleave,
        "compact_valid_to_front": compact,
    }
    if "num_warps" in geometry:
        options["num_warps"] = num_warps
    return width, options


def _run_cell(
    torch,
    arm: str,
    geometry: dict[str, int | bool],
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
        width, options = _check_geometry(report.geometry, batch=batch, tile_size=tile_size)
        reference = {key: value for key, value in options.items() if key != "num_warps"}

        first = _random_case(
            width=width,
            batch=batch,
            block_size=options["block_size"],
            dcp_size=options["dcp_size"],
            seed=seed,
        )
        first_result = localize_reference(*first, **reference)
        # A seed offset no other cell of a sweep uses.
        second, second_result = _second_case(first, first_result, seed=seed + 2**32, **reference)

        def device_ints(values):
            return torch.tensor(values, dtype=torch.int32, device=device)

        # Device copies of each fixture and its oracle result. Replays copy a
        # fixture into the bound buffers in place; the copies are never bound.
        fixtures = [
            (tuple(device_ints(values) for values in case), *map(device_ints, result))
            for case, result in ((first, first_result), (second, second_result))
        ]
        bound = tuple(staged.clone() for staged in fixtures[0][0])
        req, table, tokens = bound

        out, out_storage = _guarded(torch, (batch, width), device)
        counts, counts_storage = _guarded(torch, (batch,), device)

        workspace: dict[str, object] = {}
        written = [out, counts]
        guards = [out_storage, counts_storage]
        if arm == "hierarchical_stable":
            for name, shape in workspace_shapes(batch, width, tile_size=tile_size).items():
                view, storage = _guarded(torch, shape, device)
                workspace[name] = view
                written.append(view)
                guards.append(storage)

        def launch() -> None:
            if arm == "row_stable":
                localize_rowwise(req, table, tokens, out, counts, **options)
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
                    tile_size=tile_size,
                    **options,
                )

        def poison() -> None:
            for buffer in written:
                buffer.fill_(POISON)

        def holds(fixture) -> bool:
            """Whether the bound inputs still hold ``fixture``'s values."""
            return all(
                bool(torch.equal(tensor, staged))
                for tensor, staged in zip(bound, fixture[0], strict=True)
            )

        # One reading per eager launch, the captured launch, and each replay.
        allocation_free: list[bool] = []

        poison()
        allocation_free.append(_allocation_free(torch, launch))
        torch.cuda.synchronize()
        observed_out = out.cpu().tolist()
        observed_counts = counts.cpu().tolist()
        report.checks["oracle"] = (observed_out, observed_counts) == first_result

        req_ids, block_table, rows = first
        compacting = options["compact_valid_to_front"] and options["dcp_size"] > 1
        order_ok = True
        for row_id, row in enumerate(rows):
            slots = _owned_slots(
                row,
                block_table[req_ids[row_id]],
                block_size=options["block_size"],
                dcp_size=options["dcp_size"],
                dcp_rank=options["dcp_rank"],
                dcp_interleave=options["dcp_interleave"],
            )
            kept = [slot for slot in slots if slot is not None]
            if compacting:
                want = kept + [-1] * (width - len(kept))
            else:
                want = [-1 if slot is None else slot for slot in slots]
            if observed_out[row_id] != want or observed_counts[row_id] != len(kept):
                order_ok = False
        report.checks["order"] = order_ok

        first_out = out.clone()
        first_counts = counts.clone()
        stable = True
        for _ in range(DETERMINISM_REPEATS):
            poison()
            allocation_free.append(_allocation_free(torch, launch))
            torch.cuda.synchronize()
            stable = (
                stable
                and bool(torch.equal(out, first_out))
                and bool(torch.equal(counts, first_counts))
            )
        report.checks["determinism"] = stable

        # PyTorch's capture recipe: warm up on a side stream, then capture.
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                allocation_free.append(_allocation_free(torch, launch))
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        inputs_kept = holds(fixtures[0])
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            # Measured inside the context: beginning a capture may allocate
            # PyTorch's own bookkeeping, which no launcher can avoid.
            allocation_free.append(_allocation_free(torch, launch))

        replayed = True
        for replay in range(REPLAY_COUNT):
            # The capture saw the first fixture: start on the second, alternate,
            # and end on the first.
            fixture = fixtures[(replay + 1) % 2]
            staged_inputs, want_out, want_counts = fixture
            for tensor, staged in zip(bound, staged_inputs, strict=True):
                tensor.copy_(staged)
            poison()
            allocation_free.append(_allocation_free(torch, graph.replay))
            torch.cuda.synchronize()
            replayed = (
                replayed
                and bool(torch.equal(out, want_out))
                and bool(torch.equal(counts, want_counts))
            )
            inputs_kept = inputs_kept and holds(fixture)
        report.checks["replay"] = replayed
        report.checks["allocation"] = all(allocation_free)
        report.checks["immutability"] = inputs_kept and all(
            _guards_clean(storage) for storage in guards
        )
    except LocalizationError as exc:
        report.error = f"{type(exc).__name__}: {exc}"
    except Exception as exc:  # pragma: no cover - surfaced, not swallowed
        report.error = f"{type(exc).__name__}: {exc}"

    for name in CHECKS:
        report.checks.setdefault(name, False)
    return report


def conformance(
    matrix: Sequence[dict[str, int | bool]] | None = None,
    *,
    arms: Sequence[str] = ARMS,
    batch: int = 5,
    tile_size: int = DEFAULT_TILE_SIZE,
    seed: int = 20260803,
) -> ConformanceReport:
    """Run the conformance matrix on the local device.

    Args:
        matrix: Geometry dicts with ``width``, ``block_size``, ``dcp_size``,
            ``dcp_rank``, ``dcp_interleave``, optionally adding
            ``compact_valid_to_front`` and ``num_warps``, which apply to every
            arm. Defaults to :data:`DEFAULT_MATRIX`. Pass your own deployment
            geometry before trusting the kernels on it.
        arms: Which implementations to check.
        batch: Selection rows per cell. Rows share a small pool of request ids,
            so request routing is exercised, not bypassed.
        tile_size: Tile size for the hierarchical arm.
        seed: Base seed; each cell derives its own, so runs are reproducible.

    Returns:
        A :class:`ConformanceReport`. It is falsy if anything failed, and
        ``report.summary()`` names the geometry and the check. A missing or
        unimportable torch or Triton, or no CUDA device, yields a skipped
        (falsy) report whose reason names the cause rather than an exception,
        so this is safe to call unconditionally in CI.

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
    # An installed but broken runtime can raise anything at import; either way
    # the backend is unavailable, and the skip reason keeps the exception.
    try:
        import torch
    except Exception as exc:
        report.skipped = _import_failure("torch", exc)
        return report
    try:
        import triton  # noqa: F401
    except Exception as exc:
        report.skipped = _import_failure("triton", exc)
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


def _load_matrix(path: str, *, batch: int, tile_size: int) -> list[dict[str, int | bool]]:
    """Read a ``--matrix`` file: a JSON list of geometry objects.

    Every entry is checked with the schema :func:`conformance` applies, before
    any backend is imported, so a typo fails even where the sweep would skip.
    """
    import json

    try:
        with open(path, encoding="utf-8") as handle:
            matrix = json.load(handle)
    except (OSError, ValueError) as exc:
        raise LocalizationError(f"cannot read --matrix {path}: {exc}") from exc
    if not isinstance(matrix, list) or not matrix:
        raise LocalizationError("--matrix must hold a nonempty JSON list of geometry objects")
    for index, geometry in enumerate(matrix):
        try:
            _check_geometry(geometry, batch=batch, tile_size=tile_size)
        except LocalizationError as exc:
            raise LocalizationError(f"--matrix entry {index}: {exc}") from exc
    return matrix


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI
    """``python -m silkern`` -- exits nonzero if any cell fails."""
    import json

    from silkern._cli import USAGE_ERROR, ArgumentParser

    parser = ArgumentParser(
        description="Run the silkern conformance matrix.",
        epilog=(
            "Exits 1 if any cell fails and 0 otherwise, including when the sweep "
            "is skipped for want of a CUDA device or Triton. Use --require-device "
            f"to treat an unavailable backend as a failure (exit 2). A usage error, "
            f"including an invalid --matrix file, exits {USAGE_ERROR}."
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
    parser.add_argument(
        "--matrix",
        metavar="FILE",
        help=(
            "JSON list of geometry objects to check instead of the built-in matrix; "
            f"keys {', '.join(GEOMETRY_KEYS)}, optionally {', '.join(OPTIONAL_GEOMETRY_KEYS)}"
        ),
    )
    args = parser.parse_args(argv)
    if args.batch < 1:
        parser.error("--batch must be at least 1")

    try:
        matrix = (
            None
            if args.matrix is None
            else _load_matrix(args.matrix, batch=args.batch, tile_size=args.tile_size)
        )
        report = conformance(
            matrix=matrix,
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
