"""CPU-side tests for the conformance harness.

The harness itself needs a GPU, but its reporting contract, its independent
order derivation, and its skip behavior are all checkable without one -- and
they are the parts most likely to lie about a failure.
"""

from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import pytest

import silkern
import silkern.verify as verify
from silkern.verify import (
    CHECKS,
    DEFAULT_MATRIX,
    CellReport,
    ConformanceReport,
    _expected_order,
    _random_case,
)


def test_conformance_skips_cleanly_without_a_device(monkeypatch) -> None:
    # CPU reporting tests must never start a GPU sweep on a CUDA-equipped host.
    monkeypatch.setitem(sys.modules, "torch", None)
    report = silkern.conformance()
    # In CI there is no CUDA device; a skip must be falsy but not an exception,
    # and must say why.
    if report.skipped is not None:
        assert not report.ok
        assert not report
        assert report.skipped in {
            "torch is not installed",
            "triton is not installed",
            "no CUDA device is available",
        }
        assert "skipped" in report.summary()


def test_empty_report_is_not_ok() -> None:
    assert not ConformanceReport().ok
    assert "no cells executed (failed)" in ConformanceReport().summary()


@pytest.mark.parametrize("checks", [{}, {"oracle": True}])
def test_incomplete_cell_cannot_report_success(checks: dict[str, bool]) -> None:
    cell = CellReport(arm="row_stable", geometry={}, checks=checks)
    assert not cell.ok
    assert cell.failures() == [name for name in CHECKS if name not in checks]


def test_cell_report_ok_requires_every_check() -> None:
    cell = CellReport(arm="row_stable", geometry={"width": 8})
    cell.checks = dict.fromkeys(CHECKS, True)
    assert cell.ok
    assert cell.failures() == []

    cell.checks["order"] = False
    assert not cell.ok
    assert cell.failures() == ["order"]

    cell.error = "boom"
    assert cell.failures() == ["error: boom"]


@pytest.mark.parametrize("outcome", [1, "failed", [], None])
def test_nonboolean_outcomes_cannot_report_success(outcome) -> None:
    cell = CellReport(arm="row_stable", geometry={}, checks=dict.fromkeys(CHECKS, True))
    cell.checks["oracle"] = outcome
    assert not cell.ok
    assert cell.failures() == ["oracle"]


def test_unknown_checks_cannot_silently_pass() -> None:
    cell = CellReport(arm="row_stable", geometry={}, checks=dict.fromkeys(CHECKS, True))
    cell.checks["orcale"] = True
    assert not cell.ok
    assert cell.failures() == ["unexpected check: orcale"]


def test_failing_cell_summary_names_geometry_and_check() -> None:
    cell = CellReport(arm="row_stable", geometry={"width": 513, "dcp_size": 4})
    cell.checks = dict.fromkeys(CHECKS, True)
    cell.checks["determinism"] = False
    report = ConformanceReport(cells=[cell], device="Test Device")
    summary = report.summary()
    assert "FAIL" in summary
    assert "row_stable" in summary
    assert "width=513" in summary
    assert "determinism" in summary


@pytest.mark.parametrize(
    "kwargs",
    [
        {"arms": ()},
        {"arms": "row_stable"},
        {"arms": ("unknown",)},
        {"batch": 0},
        {"batch": True},
        {"batch": 1.5},
        {"batch": verify.MAX_FIXTURE_ELEMENTS + 1},
        {"tile_size": 128.0},
        {"tile_size": 32},
        {"seed": True},
        {"seed": "one"},
        {"matrix": []},
    ],
)
def test_invalid_sweep_options_raise_even_without_backend(monkeypatch, kwargs) -> None:
    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(silkern.LocalizationError):
        verify.conformance(**kwargs)


@pytest.mark.parametrize(
    "change", [
        {"width": 2**40}, {"width": True}, {"dcp_size": 0}, {"block_size": 0},
        {"compact_valid_to_front": False}, {"tile_size": 256},
        {"block_size": 2**20}, {"dcp_size": 2**24},
    ]
)
def test_bad_geometry_fails_before_any_device_allocation(change: dict[str, int]) -> None:
    geometry = {**DEFAULT_MATRIX[0], **change}
    report = verify._run_cell(
        SimpleNamespace(), "row_stable", geometry,
        batch=1, seed=0, tile_size=128, device="cuda",
    )
    assert not report.ok
    assert report.error.startswith("LocalizationError:")
    assert report.checks == dict.fromkeys(CHECKS, False)


def test_large_geometry_rejects_host_fixture_before_allocation() -> None:
    report = verify._run_cell(
        SimpleNamespace(), "row_stable", {**DEFAULT_MATRIX[0], "width": 4096},
        batch=2048, seed=0, tile_size=128, device="cuda",
    )
    assert not report.ok
    assert "diagnostic fixtures" in report.error


def test_malformed_geometry_is_reported_without_aborting_the_sweep() -> None:
    report = verify._run_cell(
        SimpleNamespace(), "row_stable", None,
        batch=1, seed=0, tile_size=128, device="cuda",
    )
    assert not report.ok
    assert report.error == "LocalizationError: matrix geometry must be a mapping"
    assert report.checks == dict.fromkeys(CHECKS, False)


def test_integral_geometry_stays_json_serializable_when_a_check_fails() -> None:
    np = pytest.importorskip("numpy")
    report = verify._run_cell(
        SimpleNamespace(), "row_stable", {**DEFAULT_MATRIX[0], "width": np.int64(2**40)},
        batch=1, seed=0, tile_size=128, device="cuda",
    )
    assert not report.ok
    encoded = json.dumps(ConformanceReport(cells=[report]).to_dict(), allow_nan=False)
    assert json.loads(encoded)["cells"][0]["geometry"]["width"] == 2**40


def test_conformance_uses_current_device_and_deduplicates_arms(monkeypatch) -> None:
    fake_cuda = SimpleNamespace(
        is_available=lambda: True,
        current_device=lambda: 3,
        get_device_name=lambda index: f"Test Device {index}",
    )
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=fake_cuda))
    monkeypatch.setitem(sys.modules, "triton", SimpleNamespace())
    monkeypatch.setattr(
        verify, "_run_cell", lambda _torch, arm, geometry, **kwargs: CellReport(
            arm=arm, geometry=geometry, checks=dict.fromkeys(CHECKS, True)
        ),
    )
    report = verify.conformance(matrix=DEFAULT_MATRIX[:1], arms=("row_stable", "row_stable"))
    assert report.device == "Test Device 3"
    assert len(report.cells) == 1
    assert report.ok


@pytest.mark.parametrize("required, expected", [(False, 0), (True, 2)])
def test_cli_json_reports_skip_and_enforces_required_backend(monkeypatch, capsys, required, expected) -> None:
    monkeypatch.setattr(
        verify, "conformance", lambda **kwargs: ConformanceReport(skipped="test backend missing")
    )
    argv = ["--json"] + (["--require-device"] if required else [])
    assert verify.main(argv) == expected
    result = json.loads(capsys.readouterr().out)
    assert result == {
        "backend": "cuda", "device": "unknown", "ok": False,
        "skipped": "test backend missing", "cells": [],
    }


def test_cli_failed_cell_is_nonzero_and_explained_in_json(monkeypatch, capsys) -> None:
    report = ConformanceReport(cells=[CellReport(arm="row_stable", geometry={"width": 1})])
    monkeypatch.setattr(verify, "conformance", lambda **kwargs: report)
    assert verify.main(["--json"]) == 1
    result = json.loads(capsys.readouterr().out)
    assert result["cells"][0]["failures"] == list(CHECKS)
    assert not result["cells"][0]["ok"]


@pytest.mark.parametrize("batch", ["0", str(verify.MAX_FIXTURE_ELEMENTS + 1)])
def test_cli_rejects_invalid_batch_before_backend(monkeypatch, batch) -> None:
    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(SystemExit, match="2"):
        verify.main(["--batch", batch])


def test_default_matrix_is_well_formed_and_crosses_boundaries() -> None:
    required = {"width", "block_size", "dcp_size", "dcp_rank", "dcp_interleave"}
    for geometry in DEFAULT_MATRIX:
        assert set(geometry) == required
        assert 0 <= geometry["dcp_rank"] < geometry["dcp_size"]
        assert geometry["width"] <= silkern.MAX_ROW_WIDTH
        assert geometry["block_size"] % geometry["dcp_interleave"] == 0
    widths = {g["width"] for g in DEFAULT_MATRIX}
    assert {1, silkern.MAX_ROW_WIDTH} <= widths          # both extremes
    assert any(w % 2 for w in widths)                  # a non-power-of-two width
    assert {g["dcp_size"] for g in DEFAULT_MATRIX} >= {1, 2, 4, 8}
    assert any(g["dcp_interleave"] > 1 for g in DEFAULT_MATRIX)
    assert any(
        g["dcp_rank"] == g["dcp_size"] - 1 for g in DEFAULT_MATRIX if g["dcp_size"] > 1
    )


def test_independent_order_derivation_agrees_with_the_oracle() -> None:
    """The harness's ``order`` check must not be a restatement of the oracle.

    They are written separately on purpose; this test proves they agree, so a
    disagreement in the field is a real signal rather than a harness bug.
    """
    for seed in range(12):
        for dcp_size, dcp_rank, interleave in [
            (2, 0, 1),
            (2, 1, 2),
            (4, 3, 1),
            (4, 1, 4),
            (8, 7, 1),
        ]:
            req_ids, block_table, rows = _random_case(
                width=131,
                batch=3,
                block_size=64,
                dcp_size=dcp_size,
                seed=seed,
            )
            expected_out, expected_counts = silkern.localize_reference(
                req_ids,
                block_table,
                rows,
                block_size=64,
                dcp_size=dcp_size,
                dcp_rank=dcp_rank,
                dcp_interleave=interleave,
            )
            for row_id, row in enumerate(rows):
                want = _expected_order(
                    row,
                    block_table[req_ids[row_id]],
                    block_size=64,
                    dcp_size=dcp_size,
                    dcp_rank=dcp_rank,
                    dcp_interleave=interleave,
                )
                count = expected_counts[row_id]
                assert want == expected_out[row_id][:count]
                assert all(v == -1 for v in expected_out[row_id][count:])


def test_random_case_actually_exercises_the_hard_paths() -> None:
    req_ids, block_table, rows = _random_case(
        width=256, batch=4, block_size=64, dcp_size=2, seed=7
    )
    flat = [v for row in rows for v in row]
    assert any(v < 0 for v in flat), "no negative sentinels generated"
    limit = 64 * 17 * 2
    assert any(v >= limit for v in flat), "no out-of-table tokens generated"
    assert len(set(req_ids)) > 1, "request routing not exercised"
    table_flat = [v for row in block_table for v in row]
    assert table_flat != sorted(table_flat), "page table is not fragmented"
