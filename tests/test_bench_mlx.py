"""Ensure Apple timing diagnostics validate work and measure fresh evaluations."""

from __future__ import annotations

import json
from functools import partial
from types import SimpleNamespace

import pytest

from bench import bench_mlx
from silkern.contract import localize_reference
from silkern.errors import LocalizationError


@pytest.mark.parametrize(
    "kwargs",
    [
        {"widths": []},
        {"batches": []},
        {"widths": [0]},
        {"widths": [4097]},
        {"batches": [False]},
        {"iterations": 0},
        {"blocks": -1},
        {"warmup": 0},
        {"dcp_rank": 2},
        {"block_size": 63, "dcp_interleave": 2},
        {"seed": True},
        {"compact_valid_to_front": "yes"},
        {"batches": [2**31]},
        {"block_size": 2**30},
        {"dcp_size": 2**30},
        {"include_compiled": 1},
        {"widths": [128, 128]},
        {"batches": [1, 1]},
    ],
)
def test_invalid_benchmark_arguments_fail_before_runtime_load(monkeypatch, kwargs) -> None:
    monkeypatch.setattr(bench_mlx, "_load_mlx", lambda: pytest.fail("loaded runtime"))
    with pytest.raises(LocalizationError):
        bench_mlx.benchmark_mlx(**kwargs)


def test_cli_help_and_no_device(monkeypatch, capsys) -> None:
    with pytest.raises(SystemExit) as caught:
        bench_mlx.main(["--help"])
    assert caught.value.code == 0
    monkeypatch.setattr(bench_mlx, "_load_mlx", lambda: (None, "no Apple Metal device"))
    assert bench_mlx.main(["--width", "1", "--batch", "1"]) == 2
    assert "no Apple Metal device" in capsys.readouterr().err


def test_mixed_numpy_integers_use_exact_divisibility_before_fixture_bounds(monkeypatch):
    np = pytest.importorskip("numpy")
    monkeypatch.setattr(bench_mlx, "_load_mlx", lambda: pytest.fail("loaded runtime"))
    with pytest.raises(LocalizationError, match="divisible"):
        bench_mlx.benchmark_mlx(
            widths=[1], batches=[1], block_size=np.uint64(2**63 + 1),
            dcp_interleave=np.int64(2),
        )


def test_each_timed_iteration_launches_and_evaluates_fresh_outputs(monkeypatch) -> None:
    events = []
    clock = iter((1_000_000, 1_030_000))
    monkeypatch.setattr(bench_mlx.time, "perf_counter_ns", lambda: next(clock))
    mx = SimpleNamespace(
        synchronize=lambda stream: events.append(("sync", stream)),
        eval=lambda out: events.append(("eval", out)),
    )
    serial = iter(range(3))
    result = bench_mlx._time_block(mx, lambda: next(serial), iterations=3, stream="gpu")
    assert result == 10.0
    assert events == [("sync", "gpu"), ("eval", 0), ("eval", 1), ("eval", 2)]


def test_arm_schedule_rotates_and_both_paths_warm_up(monkeypatch) -> None:
    warm = []
    mx = SimpleNamespace(eval=lambda result: warm.append(result))
    measured = []

    def time_block(mx, launch, **kwargs):
        measured.append(launch())
        return float(len(measured))

    monkeypatch.setattr(bench_mlx, "_time_block", time_block)
    samples, schedule = bench_mlx._measure(
        mx,
        {"mlx": lambda: "mlx", "metal": lambda: "metal"},
        warmup=2,
        iterations=3,
        blocks=3,
        stream="gpu",
    )
    assert warm == ["mlx", "metal", "mlx", "metal"]
    assert schedule == [["mlx", "metal"], ["metal", "mlx"], ["mlx", "metal"]]
    assert measured == ["mlx", "metal", "metal", "mlx", "mlx", "metal"]
    assert samples == {"mlx": [1.0, 4.0, 5.0], "metal": [2.0, 3.0, 6.0]}


class _Array:
    def __init__(self, data, dtype=None):
        self.data = data

    def tolist(self):
        return self.data


def _fake_runtime(monkeypatch):
    mx = SimpleNamespace(
        array=_Array,
        int32="int32",
        gpu="gpu",
        eval=lambda *args: None,
        default_stream=lambda device: "stream",
    )
    monkeypatch.setattr(bench_mlx, "_load_mlx", lambda: (mx, None))
    monkeypatch.setattr(bench_mlx, "_metadata", lambda mx: {"device": {"device_name": "Test"}})
    return mx


def test_benchmark_validates_each_arm_and_retains_raw_samples(
    monkeypatch, tmp_path, capsys
) -> None:
    from silkern import mlx as mlx_module

    _fake_runtime(monkeypatch)
    seen = []

    def localize(req, table, rows, *, backend, stream, **kwargs):
        seen.append(backend)
        out, counts = localize_reference(req.data, table.data, rows.data, **kwargs)
        return _Array(out), _Array(counts)

    monkeypatch.setattr(mlx_module, "localize_mlx", localize)
    monkeypatch.setattr(
        bench_mlx,
        "_measure",
        lambda *a, **kw: (
            {"mlx": [20.0, 40.0], "metal": [10.0, 20.0]},
            [["mlx", "metal"], ["metal", "mlx"]],
        ),
    )
    output = tmp_path / "nested" / "benchmark.json"
    assert (
        bench_mlx.main(["--width", "7", "31", "--batch", "1", "--output", str(output), "--json"])
        == 0
    )
    data = json.loads(output.read_text())
    assert json.loads(capsys.readouterr().out) == data
    assert seen == ["mlx", "metal", "mlx", "metal"]
    assert all(cell["oracle_passed"] for cell in data["cells"])
    assert data["cells"][0]["results"]["mlx"]["block_samples_us"] == [20.0, 40.0]
    assert data["cells"][0]["mlx_over_metal"] == 2.0
    assert not data["methodology"]["full_model_decode"]


def test_oracle_mismatch_aborts_before_timing(monkeypatch) -> None:
    from silkern import mlx as mlx_module

    _fake_runtime(monkeypatch)
    monkeypatch.setattr(mlx_module, "localize_mlx", lambda *a, **kw: (_Array([[]]), _Array([-1])))
    monkeypatch.setattr(
        bench_mlx, "_measure", lambda *a, **kw: pytest.fail("timed incorrect output")
    )
    with pytest.raises(LocalizationError, match="oracle mismatch"):
        bench_mlx.benchmark_mlx(widths=[1], batches=[1])


def test_compiled_arms_receive_dynamic_array_arguments(monkeypatch) -> None:
    from silkern import mlx as mlx_module

    mx = _fake_runtime(monkeypatch)
    compiled_arguments = []

    def compile_function(function):
        def compiled(*arrays):
            compiled_arguments.append(arrays)
            return function(*arrays)

        return compiled

    mx.compile = compile_function

    def localize(req, table, rows, *, backend, stream, **kwargs):
        out, counts = localize_reference(req.data, table.data, rows.data, **kwargs)
        return _Array(out), _Array(counts)

    monkeypatch.setattr(mlx_module, "localize_mlx", localize)
    monkeypatch.setattr(
        bench_mlx,
        "_measure",
        lambda mx, arms, **kw: (
            {name: [float(index + 1)] for index, name in enumerate(arms)},
            [list(arms)],
        ),
    )
    report = bench_mlx.benchmark_mlx(widths=[7], batches=[1], include_compiled=True)
    assert len(compiled_arguments) == 10  # Original plus four canonical cases per compiled arm.
    assert all(
        len(arguments) == 3 and all(isinstance(arg, _Array) for arg in arguments)
        for arguments in compiled_arguments
    )
    assert set(report["cells"][0]["results"]) == {"mlx", "metal", "mlx_compiled", "metal_compiled"}
    assert report["cells"][0]["mlx_compiled_over_metal_compiled"] == 3 / 4
    assert report["schema_version"] == 2
    assert report["methodology"]["compiled_input_checks"] == [
        "request_ids", "block_table", "token_indices"
    ]
    assert report["cells"][0]["compiled_changing_inputs_passed"] is True


@pytest.mark.parametrize("frozen_index,name", tuple(enumerate(bench_mlx.COMPILED_INPUT_CHECKS)))
def test_compiled_input_gate_detects_each_independently_captured_array(
    monkeypatch, frozen_index, name
) -> None:
    mx = _fake_runtime(monkeypatch)
    common = dict(block_size=2, dcp_size=2, dcp_rank=1, dcp_interleave=1,
                  compact_valid_to_front=True)
    baseline = None

    def broken_compiled(*arrays):
        nonlocal baseline
        if baseline is None:
            baseline = arrays
        arrays = list(arrays)
        arrays[frozen_index] = baseline[frozen_index]
        out, counts = localize_reference(*(array.data for array in arrays), **common)
        return _Array(out), _Array(counts)

    with pytest.raises(LocalizationError, match=f"changing-input oracle mismatch for {name}"):
        bench_mlx._qualify_compiled_inputs(
            mx, {"broken_compiled": broken_compiled}, width=1, batch=1,
            table_width=17, common=common,
        )


def test_failed_compiled_input_gate_aborts_before_timing(monkeypatch) -> None:
    from silkern import mlx as mlx_module

    mx = _fake_runtime(monkeypatch)

    def localize(req, table, rows, *, backend, stream, **kwargs):
        out, counts = localize_reference(req.data, table.data, rows.data, **kwargs)
        return _Array(out), _Array(counts)

    def capture_inputs(function):
        cached = None

        def compiled(*arrays):
            nonlocal cached
            if cached is None:
                cached = function(*arrays)
            return cached

        return compiled

    mx.compile = capture_inputs
    monkeypatch.setattr(mlx_module, "localize_mlx", localize)
    monkeypatch.setattr(bench_mlx, "_measure", lambda *args, **kwargs: pytest.fail("timed cached inputs"))
    with pytest.raises(LocalizationError, match="changing-input oracle mismatch"):
        bench_mlx.benchmark_mlx(widths=[1], batches=[1], include_compiled=True)


@pytest.mark.mlx
@pytest.mark.metal
@pytest.mark.parametrize("backend", ["mlx", "metal"])
def test_actual_compiled_timing_function_changes_with_input(backend) -> None:
    from silkern.mlx import localize_mlx

    mx = pytest.importorskip("mlx.core")
    if not mx.metal.is_available():
        pytest.skip("Apple Metal unavailable")
    common = dict(block_size=2, dcp_size=2, dcp_rank=0)
    compiled = mx.compile(
        partial(localize_mlx, **common, backend=backend, stream=mx.default_stream(mx.gpu))
    )
    req, table = mx.array([0], dtype=mx.int32), mx.array([[2, 3]], dtype=mx.int32)
    inputs = [[[0, 2, -1, 4, 6]], [[6, -1, 4, 2, 0]]]
    outputs = []
    for rows in inputs:
        out, counts = compiled(req, table, mx.array(rows, dtype=mx.int32))
        mx.eval(out, counts)
        expected, expected_counts = localize_reference([0], [[2, 3]], rows, **common)
        assert out.tolist() == expected
        assert counts.tolist() == expected_counts
        outputs.append(out.tolist())
    assert outputs[0] != outputs[1]
