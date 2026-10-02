"""Conformance-report and adversarial-input tests that never require a GPU."""

from __future__ import annotations

import json

import pytest

from silkern import mlx_verify
from silkern.contract import localize_reference
from silkern.errors import LocalizationError


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
    monkeypatch.setattr(mlx_verify, "_load_mlx", lambda: (None, "no Metal"))
    assert mlx_verify.main(["--json", *(["--require-device"] if require else [])]) == code
    data = json.loads(capsys.readouterr().out)
    assert data["skipped"] == "no Metal"
    assert data["ok"] is False


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
    monkeypatch.setattr(mlx_verify, "_load_mlx", lambda: pytest.fail("loaded runtime"))
    with pytest.raises(LocalizationError):
        mlx_verify.conformance_mlx(**kwargs)


def test_mixed_numpy_integers_use_exact_divisibility_before_fixture_bounds(monkeypatch):
    np = pytest.importorskip("numpy")
    monkeypatch.setattr(mlx_verify, "_load_mlx", lambda: pytest.fail("loaded runtime"))
    geometry = {
        **mlx_verify.DEFAULT_MLX_MATRIX[0], "block_size": np.uint64(2**63 + 1),
        "dcp_interleave": np.int64(2),
    }
    with pytest.raises(LocalizationError, match="divisible"):
        mlx_verify.conformance_mlx(matrix=[geometry])


def test_default_matrix_covers_scan_edges_and_compaction() -> None:
    cells = mlx_verify.DEFAULT_MLX_MATRIX
    assert {cell["width"] for cell in cells} >= {1, 31, 32, 33, 127, 128, 129, 4096}
    assert {cell["dcp_size"] for cell in cells} == {1, 2, 4, 8}
    assert {cell["compact_valid_to_front"] for cell in cells} == {True, False}
    assert all(
        any(cell["dcp_size"] == size and cell["dcp_rank"] == size - 1 for cell in cells)
        for size in (1, 2, 4, 8)
    )


def test_adversarial_cases_exercise_duplicates_negative_pages_and_empty_prefixes() -> None:
    req, table, rows = mlx_verify._random_case(
        width=513, batch=4, block_size=64, dcp_size=4, seed=9
    )
    assert len(set(req)) == 4
    assert all(row[:2] == [-1, -2] for row in table)
    assert rows[0] != sorted(rows[0])
    assert len(set(rows[0])) < len(rows[0])
    assert rows[-1] == [-1] * 513
    assert max(rows[0]) >= 64 * 17 * 4


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
    monkeypatch.setattr(mlx_verify, "_load_mlx", lambda: (object(), None))
    monkeypatch.setattr(mlx_verify, "_metadata", lambda mx: {})
    seen = []

    def run(mx, backend, geometry, **kwargs):
        seen.append((backend, kwargs["seed"]))
        return mlx_verify.MLXCellReport(
            backend, geometry, dict.fromkeys(mlx_verify.MLX_CHECKS, True)
        )

    monkeypatch.setattr(mlx_verify, "_run_cell", run)
    report = mlx_verify.conformance_mlx(matrix=matrix, seed=17)
    assert report.ok
    assert seen == [("mlx", 17), ("metal", 17)]
    assert "compact_valid_to_front" not in matrix[0]
