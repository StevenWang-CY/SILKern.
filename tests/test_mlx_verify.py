"""Conformance-report and adversarial-input tests that never require a GPU.

Tests marked ``mlx`` run the real verifier on the MLX CPU stream, plus the GPU
cells when Metal is available, against deliberately broken localizers.
"""

from __future__ import annotations

import inspect
import json
import os
import pathlib
import subprocess
import sys
import textwrap
import types

import pytest

import silkern
import silkern.mlx as silkern_mlx
from silkern import mlx_verify
from silkern.contract import localize_reference
from silkern.errors import LocalizationError

ROOT = pathlib.Path(__file__).resolve().parent.parent


def test_incomplete_checks_cannot_pass() -> None:
    cell = mlx_verify.MLXCellReport(backend="metal", geometry={"width": 1})
    assert not cell.ok
    cell.checks = dict.fromkeys(mlx_verify.MLX_CHECKS, True)
    assert cell.ok
    cell.checks.pop("request_bounds")
    assert not cell.ok
    assert cell.failures() == ["request_bounds"]
    cell.error = "execution failed"
    assert cell.failures() == ["execution failed"]


def test_empty_skipped_and_failed_reports_are_falsy() -> None:
    assert not mlx_verify.MLXConformanceReport()
    skipped = mlx_verify.MLXConformanceReport(skipped="no Metal")
    assert not skipped
    assert skipped.to_dict()["ok"] is False
    cell = mlx_verify.MLXCellReport(backend="metal", geometry={"width": 33}, error="bad output")
    report = mlx_verify.MLXConformanceReport(cells=[cell])
    assert not report
    assert "width=33" in report.summary()
    assert "bad output" in report.summary()
    assert json.loads(json.dumps(report.to_dict()))["cells"][0]["ok"] is False


@pytest.mark.parametrize("outcome", [1, "failed", [], None])
def test_nonboolean_outcomes_cannot_report_success(outcome) -> None:
    cell = mlx_verify.MLXCellReport(
        backend="metal", geometry={}, checks=dict.fromkeys(mlx_verify.MLX_CHECKS, True)
    )
    cell.checks["oracle"] = outcome
    assert not cell.ok
    assert cell.failures() == ["oracle"]


def test_unknown_checks_cannot_silently_pass() -> None:
    cell = mlx_verify.MLXCellReport(
        backend="metal", geometry={}, checks=dict.fromkeys(mlx_verify.MLX_CHECKS, True)
    )
    cell.checks["orcale"] = True
    assert not cell.ok
    assert cell.failures() == ["unexpected check: orcale"]


@pytest.mark.parametrize("require,code", [(False, 0), (True, 2)])
def test_cli_skip_json_and_required_device(monkeypatch, capsys, require, code) -> None:
    monkeypatch.setattr(mlx_verify, "_import_mlx", lambda: (None, "no MLX"))
    assert mlx_verify.main(["--json", *(["--require-device"] if require else [])]) == code
    data = json.loads(capsys.readouterr().out)
    assert data["skipped"] == "no MLX"
    assert data["ok"] is False
    assert data["cells"] == []


@pytest.mark.parametrize(
    "kwargs",
    [
        {"backend": "cuda"},
        {"batch": 0},
        {"batch": True},
        {"repeats": -1},
        {"seed": 1.5},
        {"matrix": []},
        {"matrix": [{"width": 1}]},
        {"matrix": [{**mlx_verify.DEFAULT_MLX_MATRIX[0], "width": 4097}]},
        {"matrix": [{**mlx_verify.DEFAULT_MLX_MATRIX[0], "dcp_rank": 1}]},
        {"matrix": [{**mlx_verify.DEFAULT_MLX_MATRIX[0], "dcp_interleave": 3}]},
        {"matrix": [{**mlx_verify.DEFAULT_MLX_MATRIX[0], "compact_valid_to_front": 1}]},
        {"batch": mlx_verify.MAX_FIXTURE_ELEMENTS + 1},
        {"matrix": [{**mlx_verify.DEFAULT_MLX_MATRIX[0], "block_size": 2**30}]},
        {"matrix": [{**mlx_verify.DEFAULT_MLX_MATRIX[0], "dcp_size": 2**30}]},
    ],
)
def test_invalid_configuration_rejected_before_loading_mlx(monkeypatch, kwargs) -> None:
    monkeypatch.setattr(mlx_verify, "_import_mlx", lambda: pytest.fail("loaded runtime"))
    with pytest.raises(LocalizationError):
        mlx_verify.conformance_mlx(**kwargs)


def test_mixed_numpy_integers_use_exact_divisibility_before_fixture_bounds(monkeypatch):
    np = pytest.importorskip("numpy")
    monkeypatch.setattr(mlx_verify, "_import_mlx", lambda: pytest.fail("loaded runtime"))
    geometry = {
        **mlx_verify.DEFAULT_MLX_MATRIX[0], "block_size": np.uint64(2**63 + 1),
        "dcp_interleave": np.int64(2),
    }
    with pytest.raises(LocalizationError, match="divisible"):
        mlx_verify.conformance_mlx(matrix=[geometry])


def test_default_matrix_covers_scan_edges_and_compaction() -> None:
    cells = mlx_verify.DEFAULT_MLX_MATRIX
    # 256 -> 257 is where each Metal thread's share grows from one to two.
    assert {cell["width"] for cell in cells} >= {
        1, 31, 32, 33, 127, 128, 129, 256, 257, 4095, 4096,
    }
    assert {cell["dcp_size"] for cell in cells} == {1, 2, 4, 8}
    assert {cell["compact_valid_to_front"] for cell in cells} == {True, False}
    assert all(
        any(cell["dcp_size"] == size and cell["dcp_rank"] == size - 1 for cell in cells)
        for size in (1, 2, 4, 8)
    )


def test_default_run_varies_tables_and_batches_beyond_small_fixtures() -> None:
    shapes = [mlx_verify._table_shape(index) for index in range(len(mlx_verify.DEFAULT_MLX_MATRIX))]
    requests, pages = {shape[0] for shape in shapes}, {shape[1] for shape in shapes}
    assert min(requests) == 1 and max(requests) > 4
    assert min(pages) == 1 and max(pages) > 17
    assert any(count > 4 and width > 17 for count, width in shapes)
    assert mlx_verify.DEFAULT_BATCH > 4
    assert {arm[1] for arm in mlx_verify.MLX_ARMS["both"]} == {"cpu", "gpu"}
    for function in (mlx_verify.conformance_mlx, silkern.conformance_mlx):
        assert inspect.signature(function).parameters["batch"].default == mlx_verify.DEFAULT_BATCH


@pytest.mark.parametrize("seed", range(18))
def test_adversarial_cases_exercise_duplicates_negative_pages_and_empty_prefixes(seed) -> None:
    requests, pages = mlx_verify._table_shape(seed)
    req, table, rows = mlx_verify._random_case(
        width=513, batch=8, block_size=64, dcp_size=4, seed=seed
    )
    assert len(table) == requests and all(len(row) == pages for row in table)
    assert len(set(req)) == min(8, requests)
    assert all(row[:2] == [-1, -2][:pages] for row in table)
    assert all(row[-1] == 0 for row in table) or pages <= 2
    assert rows[0] != sorted(rows[0])
    assert len(set(rows[0])) < len(rows[0])
    assert rows[-1] == [-1] * 513
    assert max(rows[0]) >= 64 * pages * 4


def test_independent_order_matches_oracle_without_dropping_negative_physical_values() -> None:
    for seed in range(4):
        for size, rank, interleave in ((2, 0, 1), (2, 1, 2), (4, 3, 4), (8, 7, 2)):
            req, table, rows = mlx_verify._random_case(
                width=513, batch=4, block_size=64, dcp_size=size, seed=seed
            )
            common = dict(block_size=64, dcp_size=size, dcp_rank=rank, dcp_interleave=interleave)
            expected, counts = localize_reference(req, table, rows, **common)
            for index, row in enumerate(rows):
                order = mlx_verify._expected_order(row, table[req[index]], **common)
                assert order == expected[index][: counts[index]]
    assert mlx_verify._expected_order(
        [0, 2], [-1], block_size=64, dcp_size=2, dcp_rank=0, dcp_interleave=1
    ) == [-64, -63]


def test_both_backends_use_identical_data_seeds_and_preserve_caller_matrix(monkeypatch) -> None:
    matrix = [
        {
            key: value
            for key, value in mlx_verify.DEFAULT_MLX_MATRIX[0].items()
            if key != "compact_valid_to_front"
        }
    ]
    monkeypatch.setattr(mlx_verify, "_import_mlx", lambda: (object(), None))
    monkeypatch.setattr(mlx_verify, "_metal_unavailable", lambda mx: None)
    monkeypatch.setattr(mlx_verify, "_metadata", lambda mx, metal: {})
    seen = []

    def run(mx, backend, geometry, **kwargs):
        seen.append((backend, kwargs["device"], kwargs["seed"]))
        return mlx_verify.MLXCellReport(
            backend, geometry, dict.fromkeys(mlx_verify.MLX_CHECKS, True)
        )

    monkeypatch.setattr(mlx_verify, "_run_cell", run)
    report = mlx_verify.conformance_mlx(matrix=matrix, seed=17)
    assert report.ok
    assert seen == [("mlx", "cpu", 17), ("mlx", "gpu", 17), ("metal", "gpu", 17)]
    assert "compact_valid_to_front" not in matrix[0]


@pytest.mark.parametrize("backend,arms", [
    ("both", [("mlx", "cpu")]), ("mlx", [("mlx", "cpu")]), ("auto", [("auto", "cpu")]),
    ("metal", []),
])
def test_without_metal_cpu_cells_still_run_and_gpu_cells_are_reported_skipped(
    monkeypatch, capsys, backend, arms
) -> None:
    monkeypatch.setattr(mlx_verify, "_import_mlx", lambda: (object(), None))
    monkeypatch.setattr(mlx_verify, "_metal_unavailable", lambda mx: "no Apple Metal device")
    metadata = []
    monkeypatch.setattr(mlx_verify, "_metadata", lambda mx, metal: metadata.append(metal) or {})
    outcome = {"oracle": True}

    def run(mx, backend, geometry, **kwargs):
        cell = mlx_verify.MLXCellReport(
            backend, geometry, dict.fromkeys(mlx_verify.MLX_CHECKS, True), device=kwargs["device"]
        )
        cell.checks["oracle"] = outcome["oracle"]
        return cell

    monkeypatch.setattr(mlx_verify, "_run_cell", run)
    report = mlx_verify.conformance_mlx(backend=backend, matrix=mlx_verify.DEFAULT_MLX_MATRIX[:2])
    assert [(cell.backend, cell.device) for cell in report.cells] == arms * 2
    assert report.skipped.startswith("no Apple Metal device")
    assert not report and metadata == [False]
    # Exit 2 is reserved for --require-device; otherwise CPU failures still fail.
    argv = ["--backend", backend]
    assert mlx_verify.main([*argv, "--require-device"]) == 2
    assert mlx_verify.main(argv) == 0
    outcome["oracle"] = False
    assert mlx_verify.main(argv) == (1 if arms else 0)
    assert mlx_verify.main([*argv, "--require-device"]) == 2
    summary = capsys.readouterr().out
    assert "skipped: no Apple Metal device" in summary
    assert ("FAIL mlx on cpu" in summary or "FAIL auto on cpu" in summary) == bool(arms)


def test_metadata_records_the_imported_modules_not_installed_distributions(monkeypatch):
    # The imported package is the source being verified; a stale installed
    # distribution (an editable install elsewhere, say) must not be reported.
    monkeypatch.setattr(silkern, "__version__", "9.9.9+imported")
    fake = types.SimpleNamespace(
        __version__="0.0.1+imported", cpu="cpu", gpu="gpu",
        device_info=lambda device: {"device_name": f"test-{device}"},
    )
    for metal, device, backend in ((True, "gpu", "apple_metal"), (False, "cpu", "mlx_cpu")):
        metadata = mlx_verify._metadata(fake, metal=metal)
        assert metadata["versions"] == {"silkern": "9.9.9+imported", "mlx": "0.0.1+imported"}
        assert metadata["device"] == {"device_name": f"test-{device}"}
        assert metadata["execution_backend"] == backend
        assert set(metadata["source_sha256"]) == {
            "silkern/contract.py", "silkern/mlx.py", "silkern/mlx_verify.py",
        }


BROKEN_MLX = 'raise {error}("simulated broken MLX installation")\n'


@pytest.mark.parametrize("error", ["OSError", "RuntimeError", "ImportError", "AttributeError"])
def test_broken_mlx_installation_reports_unavailable_instead_of_crashing(tmp_path, error):
    """An installed MLX that fails on import is reported, never a traceback.

    A stub ``mlx`` package on ``sys.path`` shadows any real installation in a
    fresh interpreter, so this runs on every platform.
    """
    package = tmp_path / "mlx"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "core.py").write_text(BROKEN_MLX.format(error=error))
    script = textwrap.dedent(f"""
        import json, pathlib
        import silkern
        from silkern import mlx_verify
        assert pathlib.Path(silkern.__file__).resolve().parent.parent == pathlib.Path({str(ROOT)!r})
        try:
            silkern.localize_mlx(None, None, None, block_size=2, dcp_size=2, dcp_rank=0)
        except silkern.LocalizationError as exc:
            assert "MLX is unavailable" in str(exc) and "{error}" in str(exc), exc
            assert isinstance(exc.__cause__, {error})
        else:
            raise AssertionError("a broken MLX must raise LocalizationError")
        report = silkern.conformance_mlx()
        assert not report and report.cells == [] and "{error}" in report.skipped
        assert mlx_verify._load_mlx()[0] is None
        assert mlx_verify.main(["--json"]) == 0
        assert mlx_verify.main(["--require-device"]) == 2
    """)
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=ROOT, capture_output=True, text=True, check=False,
        env={**os.environ, "PYTHONPATH": str(tmp_path)},
    )
    assert result.returncode == 0, result.stderr + result.stdout
    assert "simulated broken MLX installation" in result.stdout


# Each localizer is wrong in one way; the real verifier must report that check.
# The first six default geometries already reach five requests, 29 pages, the
# default batch of eight rows, and the CPU stream.
VERIFIER_PREFIX = mlx_verify.DEFAULT_MLX_MATRIX[:6]


def _broken_localizers(mx, true):
    calls = {"count": 0}

    def rewrite(transform):
        def localizer(req, table, tokens, **options):
            out, counts = true(req, table, tokens, **options)
            rows, totals = transform(out.tolist(), counts.tolist())
            return mx.array(rows, dtype=mx.int32), mx.array(totals, dtype=mx.int32)
        return localizer

    def wrong_when(condition):
        def localizer(req, table, tokens, **options):
            out, counts = true(req, table, tokens, **options)
            return (mx.add(out, 1), counts) if condition(table, tokens, options) else (out, counts)
        return localizer

    def reversed_prefix(rows, totals):
        return [row[:count][::-1] + row[count:] for row, count in zip(rows, totals, strict=True)], totals

    def every_other_call(rows, totals):
        calls["count"] += 1
        return reversed_prefix(rows, totals) if calls["count"] % 2 == 0 else (rows, totals)

    def clamped_requests(req, table, tokens, **options):
        return true(mx.clip(req, 0, table.shape[0] - 1), table, tokens, **options)

    def wide_outputs(req, table, tokens, **options):
        out, counts = true(req, table, tokens, **options)
        return out.astype(mx.int64), counts.astype(mx.int64)

    def extra_column(req, table, tokens, **options):
        out, counts = true(req, table, tokens, **options)
        return mx.concatenate([out, out[:, :1]], axis=1), counts

    def writes_its_input(req, table, tokens, **options):
        result = true(req, table, tokens, **options)
        tokens[0, 0] = -7
        return result

    def raises(req, table, tokens, **options):
        raise RuntimeError("kernel launch failed")

    def stream_device(options):
        return getattr(options["stream"], "device", options["stream"])

    return {
        "reversed survivor prefix": (rewrite(reversed_prefix), "order"),
        "count off by one": (rewrite(lambda rows, totals: (rows, [n + 1 for n in totals])), "oracle"),
        "order changes between calls": (rewrite(every_other_call), "determinism"),
        "clamps invalid requests": (clamped_requests, "request_bounds"),
        "int64 outputs": (wide_outputs, "dtype"),
        "extra output column": (extra_column, "shape"),
        "writes its input": (writes_its_input, "immutability"),
        "raises": (raises, "RuntimeError: kernel launch failed"),
        "wrong only on the CPU stream": (
            wrong_when(lambda table, tokens, options: stream_device(options) == mx.cpu), "oracle"),
        "wrong above four requests": (
            wrong_when(lambda table, tokens, options: table.shape[0] > 4), "oracle"),
        "wrong above 17 pages": (
            wrong_when(lambda table, tokens, options: table.shape[1] > 17), "oracle"),
        "wrong above batch four": (
            wrong_when(lambda table, tokens, options: tokens.shape[0] > 4), "oracle"),
    }


# Building the table only defines functions; MLX is first used when one runs.
BROKEN_LOCALIZERS = sorted(_broken_localizers(None, None))


@pytest.mark.mlx
@pytest.mark.parametrize("name", BROKEN_LOCALIZERS)
def test_real_verifier_reports_each_broken_localizer(monkeypatch, name) -> None:
    mx = pytest.importorskip("mlx.core")
    localizer, check = _broken_localizers(mx, silkern_mlx.localize_mlx)[name]
    monkeypatch.setattr(silkern_mlx, "localize_mlx", localizer)
    report = mlx_verify.conformance_mlx(matrix=VERIFIER_PREFIX)
    failing = [cell for cell in report.cells if not cell.ok]
    assert report.cells and not report
    assert any(check in cell.failures() for cell in failing), report.summary()
    if name == "wrong only on the CPU stream":
        assert {cell.device for cell in failing} == {"cpu"}


@pytest.mark.mlx
def test_real_verifier_passes_the_true_localizer_on_every_available_device() -> None:
    mx = pytest.importorskip("mlx.core")
    report = mlx_verify.conformance_mlx(matrix=VERIFIER_PREFIX)
    devices = {"cpu", "gpu"} if mx.metal.is_available() else {"cpu"}
    assert {cell.device for cell in report.cells} == devices
    assert len(report.cells) == len(VERIFIER_PREFIX) * (3 if mx.metal.is_available() else 1)
    assert all(cell.ok for cell in report.cells), report.summary()
    assert bool(report) == mx.metal.is_available()
