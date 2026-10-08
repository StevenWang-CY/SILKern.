"""Negative controls for the CUDA conformance harness, run without CUDA.

``silkern.verify._run_cell`` takes ``torch`` as an argument, so a pure-Python
stand-in can drive the real harness anywhere. The stand-in models only what
the harness can observe of a device: work issued while a graph is capturing is
recorded rather than run, a replay reruns exactly the recorded work and
allocates nothing, and the allocator reports live bytes and their peak.

Each mutant launcher breaks one property the harness claims to check and must
fail exactly the checks covering it; correct launchers must pass every check.
A check that no mutant can fail would be decoration. These run the harness, not
the Triton kernels: device behavior still needs a CUDA conformance run.
"""

from __future__ import annotations

import json
import math
import sys
import weakref
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace

import pytest

import silkern.verify as verify
from silkern import localize_reference
from silkern.verify import ARMS, CHECKS, DEFAULT_MATRIX


def _flatten(values) -> list[int]:
    if values and isinstance(values[0], list):
        return [element for row in values for element in _flatten(row)]
    return list(values)


def _shape(values) -> tuple[int, ...]:
    inner = _shape(values[0]) if values and isinstance(values[0], list) else ()
    return (len(values), *inner)


class _Device:
    """Allocator statistics, plus the work list of a graph being captured."""

    def __init__(self) -> None:
        self.allocated = 0
        self.peak = 0
        self.capturing: list | None = None

    def charge(self, nbytes: int) -> None:
        self.allocated += nbytes
        self.peak = max(self.peak, self.allocated)

    def issue(self, work) -> None:
        """Run device work now, or record it while a graph is capturing."""
        if self.capturing is None:
            work()
        else:
            self.capturing.append(work)


class _Memory:
    """One allocation: a flat list of int32 values, charged while it lives."""

    def __init__(self, device: _Device, values: list[int]) -> None:
        self.device = device
        self.values = values
        device.charge(4 * len(values))
        weakref.finalize(self, device.charge, -4 * len(values))


class _Tensor:
    """A contiguous view of ``shape`` elements of ``memory``, from ``offset``."""

    def __init__(self, memory: _Memory, offset: int, shape) -> None:
        self.memory, self.offset, self.shape = memory, offset, tuple(shape)

    def _span(self) -> slice:
        return slice(self.offset, self.offset + math.prod(self.shape))

    def _flat(self) -> list[int]:
        return self.memory.values[self._span()]

    def data_ptr(self) -> int:  # read only by the harness as it was before poisoning
        return id(self.memory) + 4 * self.offset

    def tolist(self):
        values = self._flat()
        for extent in reversed(self.shape[1:]):
            values = [values[start : start + extent] for start in range(0, len(values), extent)]
        return values

    def cpu(self):
        return self

    def view(self, *shape):
        assert math.prod(shape) == math.prod(self.shape)
        return _Tensor(self.memory, self.offset, shape)

    def __getitem__(self, index: slice):  # 1-D slices, as the guard checks use
        start, stop, step = index.indices(self.shape[0])
        assert len(self.shape) == 1 and step == 1
        return _Tensor(self.memory, self.offset + start, (stop - start,))

    def __eq__(self, value):  # only ever reduced with .all(), as the harness does
        matches = all(element == value for element in self._flat())
        return SimpleNamespace(all=lambda: matches)

    def clone(self):
        return _Tensor(_Memory(self.memory.device, self._flat()), 0, self.shape)

    def assign(self, nested) -> None:
        """Store a nested list into this view (what a stand-in kernel writes)."""
        values = _flatten(nested)
        assert len(values) == math.prod(self.shape)
        self.memory.values[self._span()] = values

    def fill_(self, value: int) -> None:
        self.memory.device.issue(lambda: self.assign([value] * math.prod(self.shape)))

    def copy_(self, source) -> None:
        assert source.shape == self.shape
        self.memory.device.issue(lambda: self.assign(source._flat()))


def _stand_in_torch():
    gpu = _Device()

    def allocate(values: list[int], shape) -> _Tensor:
        return _Tensor(_Memory(gpu, values), 0, shape)

    class Graph:
        def __init__(self) -> None:
            self.work: list = []

        def replay(self) -> None:
            for work in self.work:
                work()

    @contextmanager
    def capture(graph: Graph):
        gpu.capturing = graph.work
        try:
            yield
        finally:
            gpu.capturing = None

    def reset_peak() -> None:
        gpu.peak = gpu.allocated

    stream = SimpleNamespace(wait_stream=lambda other: None)
    return SimpleNamespace(
        int32="int32",
        gpu=gpu,
        tensor=lambda data, dtype=None, device=None: allocate(_flatten(data), _shape(data)),
        full=lambda shape, value, dtype=None, device=None: allocate(
            [value] * math.prod(shape), shape
        ),
        equal=lambda first, second: (
            first.shape == second.shape and first._flat() == second._flat()
        ),
        cuda=SimpleNamespace(
            is_available=lambda: True,
            current_device=lambda: 0,
            get_device_name=lambda index: "stand-in device",
            synchronize=lambda: None,
            Stream=lambda: stream,
            current_stream=lambda: stream,
            stream=lambda stream: nullcontext(),
            CUDAGraph=Graph,
            graph=capture,
            memory_allocated=lambda: gpu.allocated,
            max_memory_allocated=lambda: gpu.peak,
            reset_peak_memory_stats=reset_peak,
        ),
    )


# Stand-in launchers. Each takes the stand-in torch and returns a function with
# the rowwise launcher's signature; its device work goes through ``issue``.


def _localize(req, table, tokens, out, counts, geometry) -> None:
    """A correct launch's device work: the oracle's result, written in place."""
    rows, totals = localize_reference(req.tolist(), table.tolist(), tokens.tolist(), **geometry)
    out.assign(rows)
    counts.assign(totals)


def correct(torch):
    calls = []

    def launch(req, table, tokens, out, counts, **options):
        calls.append(options)
        geometry = {key: value for key, value in options.items() if key != "num_warps"}
        torch.gpu.issue(lambda: _localize(req, table, tokens, out, counts, geometry))

    launch.calls = calls
    return launch


def noop_graph(torch):
    """Records nothing while capturing, so every replay is a no-op."""

    def launch(req, table, tokens, out, counts, *, num_warps=8, **geometry):
        if torch.gpu.capturing is None:
            _localize(req, table, tokens, out, counts, geometry)

    return launch


def first_call_only(torch):
    """Writes on its first call and never again."""
    calls = []

    def launch(req, table, tokens, out, counts, *, num_warps=8, **geometry):
        calls.append(None)
        if len(calls) == 1:
            torch.gpu.issue(lambda: _localize(req, table, tokens, out, counts, geometry))

    return launch


def leaks_every_call(torch):
    """Allocates a device buffer on every call and keeps it."""
    kept = []

    def launch(req, table, tokens, out, counts, *, num_warps=8, **geometry):
        kept.append(torch.full((1024,), 0))
        torch.gpu.issue(lambda: _localize(req, table, tokens, out, counts, geometry))

    return launch


def stale_inputs(*frozen: str):
    """Freezes the named inputs' values at capture, as a graph replaying copies would."""

    def factory(torch):
        def launch(req, table, tokens, out, counts, *, num_warps=8, **geometry):
            bound = {"req": req, "table": table, "tokens": tokens}
            captured = {}
            if torch.gpu.capturing is not None:
                captured = {name: bound[name].tolist() for name in frozen}

            def work():
                inputs = [
                    captured[name] if name in captured else tensor.tolist()
                    for name, tensor in bound.items()
                ]
                rows, totals = localize_reference(*inputs, **geometry)
                out.assign(rows)
                counts.assign(totals)

            torch.gpu.issue(work)

        return launch

    return factory


def transient_scratch(torch):
    """Allocates scratch on every call and frees it before returning."""

    def launch(req, table, tokens, out, counts, *, num_warps=8, **geometry):
        scratch = torch.full((1024,), 0)
        del scratch
        torch.gpu.issue(lambda: _localize(req, table, tokens, out, counts, geometry))

    return launch


def allocates_while_capturing(torch):
    """Allocates, and keeps, a buffer only while a graph is capturing."""
    kept = []

    def launch(req, table, tokens, out, counts, *, num_warps=8, **geometry):
        if torch.gpu.capturing is not None:
            kept.append(torch.full((1024,), 0))
        torch.gpu.issue(lambda: _localize(req, table, tokens, out, counts, geometry))

    return launch


def reversed_prefix(torch):
    """Writes the right set and count, with the valid prefix reversed."""

    def launch(req, table, tokens, out, counts, *, num_warps=8, **geometry):
        def work():
            _localize(req, table, tokens, out, counts, geometry)
            pairs = zip(out.tolist(), counts.tolist(), strict=True)
            out.assign([row[:count][::-1] + row[count:] for row, count in pairs])

        torch.gpu.issue(work)

    return launch


def order_varies(torch):
    """Rotates the valid prefix on every third execution, as racing atomics would."""
    executions = []

    def launch(req, table, tokens, out, counts, *, num_warps=8, **geometry):
        def work():
            _localize(req, table, tokens, out, counts, geometry)
            executions.append(None)
            if len(executions) % 3 == 0:
                pairs = zip(out.tolist(), counts.tolist(), strict=True)
                out.assign([row[1:count] + row[:1] + row[count:] if count else row
                            for row, count in pairs])

        torch.gpu.issue(work)

    return launch


def writes_past_row_end(torch):
    """Writes the right result, then one element past the end of ``out``."""

    def launch(req, table, tokens, out, counts, *, num_warps=8, **geometry):
        def work():
            _localize(req, table, tokens, out, counts, geometry)
            out.memory.values[out.offset + math.prod(out.shape)] = 7

        torch.gpu.issue(work)

    return launch


def hierarchical(*, stale_workspace: bool = False):
    """A hierarchical stand-in that stages its result through the workspace.

    With ``stale_workspace`` it maps only on its first call, then rebuilds the
    output from whatever the workspace holds.
    """

    def factory(torch):
        mapped_once = []

        def launch(req, table, tokens, out, counts, mapped, local_positions, tile_counts,
                   tile_offsets, *, tile_size=128, num_warps=4, **geometry):
            def map_tiles():
                rows, totals = localize_reference(
                    req.tolist(), table.tolist(), tokens.tolist(), **geometry
                )
                mapped.assign(rows)
                tile_counts.assign([[total] + [0] * (tile_counts.shape[1] - 1)
                                    for total in totals])

            def scatter():
                out.assign(mapped.tolist())
                counts.assign([row[0] for row in tile_counts.tolist()])

            if not (stale_workspace and mapped_once):
                mapped_once.append(None)
                torch.gpu.issue(map_tiles)
            torch.gpu.issue(scatter)

        return launch

    return factory


GEOMETRY = {"width": 129, "block_size": 64, "dcp_size": 4, "dcp_rank": 3, "dcp_interleave": 1}


def _run_cell(monkeypatch, factory, *, arm="row_stable", geometry=GEOMETRY, batch=5):
    torch = _stand_in_torch()
    launcher = factory(torch)
    target = "localize_rowwise" if arm == "row_stable" else "localize_hierarchical"
    monkeypatch.setattr(verify, target, launcher)
    report = verify._run_cell(torch, arm, dict(geometry), batch=batch, seed=1,
                              tile_size=128, device="cuda")
    return report, launcher


@pytest.mark.parametrize(
    "geometry",
    [
        GEOMETRY,
        {"width": 513, "block_size": 32, "dcp_size": 4, "dcp_rank": 2, "dcp_interleave": 2,
         "compact_valid_to_front": False},
        {"width": 1, "block_size": 64, "dcp_size": 1, "dcp_rank": 0, "dcp_interleave": 1},
        # No token of the first fixture maps at rank 1 and width 1.
        {"width": 1, "block_size": 64, "dcp_size": 2, "dcp_rank": 1, "dcp_interleave": 1},
        {**GEOMETRY, "num_warps": 4},
    ],
)
def test_correct_rowwise_launcher_passes_every_check(monkeypatch, geometry) -> None:
    report, launcher = _run_cell(monkeypatch, correct, geometry=geometry)
    assert report.error is None
    assert report.ok, report.failures()
    assert set(report.checks) == set(CHECKS)
    options = {key: value for key, value in geometry.items() if key != "width"}
    options.setdefault("compact_valid_to_front", True)
    assert launcher.calls and all(call == options for call in launcher.calls)


def test_correct_hierarchical_launcher_passes_every_check(monkeypatch) -> None:
    report, _ = _run_cell(monkeypatch, hierarchical(), arm="hierarchical_stable")
    assert report.error is None
    assert report.ok, report.failures()


def _defect(factory, failing, *, arm="row_stable", name=None):
    return pytest.param(factory, arm, failing, id=name or factory.__name__)


@pytest.mark.parametrize(
    "factory, arm, failing",
    [
        # The audit's four blind spots: each passed every check before outputs
        # were poisoned, replay inputs alternated, and allocation was measured
        # around every launch.
        _defect(noop_graph, ["replay"]),
        _defect(first_call_only, ["determinism", "replay"]),
        _defect(leaks_every_call, ["allocation"]),
        _defect(stale_inputs("req", "table", "tokens"), ["replay"], name="stale_inputs"),
        # Narrower defects through the same mechanisms.
        _defect(stale_inputs("req"), ["replay"], name="stale_request_ids"),
        _defect(stale_inputs("table"), ["replay"], name="stale_page_table"),
        _defect(stale_inputs("tokens"), ["replay"], name="stale_tokens"),
        _defect(transient_scratch, ["allocation"]),
        _defect(allocates_while_capturing, ["allocation"]),
        _defect(hierarchical(stale_workspace=True), ["determinism", "replay"],
                arm="hierarchical_stable", name="stale_workspace"),
        # Positive controls the harness caught before; they still fail.
        _defect(reversed_prefix, ["oracle", "order", "replay"]),
        _defect(order_varies, ["determinism", "replay"]),
        _defect(writes_past_row_end, ["immutability"]),
    ],
)
def test_each_defect_fails_the_checks_that_cover_it(monkeypatch, factory, arm, failing) -> None:
    report, _ = _run_cell(monkeypatch, factory, arm=arm)
    assert report.error is None
    assert report.failures() == failing


def test_second_fixture_reroutes_every_row_and_changes_the_result() -> None:
    for seed in range(12):
        for geometry in DEFAULT_MATRIX + (
            # No token of the first fixture maps here: its result is all -1.
            {"width": 1, "block_size": 64, "dcp_size": 4, "dcp_rank": 3, "dcp_interleave": 2},
        ):
            width, options = verify._check_geometry(geometry, batch=2, tile_size=128)
            reference = {key: value for key, value in options.items() if key != "num_warps"}
            first = verify._random_case(width=width, batch=2, block_size=options["block_size"],
                                        dcp_size=options["dcp_size"], seed=seed)
            first_result = localize_reference(*first, **reference)
            second, second_result = verify._second_case(first, first_result, seed=seed + 2**32,
                                                        **reference)
            assert second_result != first_result
            assert all(a != b for a, b in zip(first[0], second[0], strict=True))
            assert [_shape(values) for values in second] == [_shape(values) for values in first]


def test_second_fixture_moves_a_page_when_results_coincide() -> None:
    reference = {"block_size": 64, "dcp_size": 2, "dcp_rank": 1, "dcp_interleave": 1,
                 "compact_valid_to_front": True}
    first = verify._random_case(width=3, batch=1, block_size=64, dcp_size=2, seed=5)
    # Ask for a fixture that differs from what the second fixture would be.
    (_, table, _), unmoved = verify._second_case(first, ([[0, 0, 0]], [9]), seed=6, **reference)
    (_, moved_table, _), moved = verify._second_case(first, unmoved, seed=6, **reference)
    assert moved != unmoved
    changed = [(row, column) for row in range(len(table)) for column in range(len(table[0]))
               if table[row][column] != moved_table[row][column]]
    assert len(changed) == 1 and changed[0][1] == 0


@pytest.fixture
def stand_in_backend(monkeypatch):
    torch = _stand_in_torch()
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "triton", SimpleNamespace())
    monkeypatch.setattr(verify, "localize_rowwise", correct(torch))
    monkeypatch.setattr(verify, "localize_hierarchical", hierarchical()(torch))
    return torch


def test_default_matrix_passes_with_correct_launchers(stand_in_backend) -> None:
    report = verify.conformance(batch=1)
    assert report.device == "stand-in device"
    assert len(report.cells) == len(ARMS) * len(DEFAULT_MATRIX)
    assert report.ok, report.summary()


def test_cli_runs_a_custom_matrix_file(stand_in_backend, tmp_path, capsys) -> None:
    matrix = [
        {"width": 63, "block_size": 32, "dcp_size": 2, "dcp_rank": 1, "dcp_interleave": 1},
        {"width": 257, "block_size": 64, "dcp_size": 4, "dcp_rank": 0, "dcp_interleave": 2,
         "compact_valid_to_front": False, "num_warps": 8},
    ]
    path = tmp_path / "matrix.json"
    path.write_text(json.dumps(matrix))
    assert verify.main(["--matrix", str(path), "--json", "--batch", "2"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["ok"]
    assert [cell["geometry"] for cell in result["cells"]] == matrix * len(ARMS)
    assert all(cell["checks"] == dict.fromkeys(CHECKS, True) for cell in result["cells"])
