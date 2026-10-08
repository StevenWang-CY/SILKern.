"""CPU-side tests for the conformance harness.

The harness itself needs a GPU, but its reporting contract, its independent
order derivation, and its skip behavior are all checkable without one -- and
they are the parts most likely to lie about a failure.
"""

from __future__ import annotations

import itertools
import json
import re
import sys
from types import SimpleNamespace

import pytest

import silkern
import silkern.verify as verify
from silkern._cli import USAGE_ERROR
from silkern.contract import DEFAULT_TILE_SIZE
from silkern.integrations import vllm
from silkern.verify import (
    CHECKS,
    DEFAULT_MATRIX,
    GEOMETRY_KEYS,
    OPTIONAL_GEOMETRY_KEYS,
    CellReport,
    ConformanceReport,
    _check_geometry,
    _expected_order,
    _random_case,
)

_NO_DEVICE = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))


@pytest.mark.parametrize(
    "modules, reason",
    [
        ({"torch": None}, "torch is not installed"),
        ({"torch": _NO_DEVICE, "triton": None}, "triton is not installed"),
        ({"torch": _NO_DEVICE, "triton": SimpleNamespace()}, "no CUDA device is available"),
    ],
    ids=["no-torch", "no-triton", "no-device"],
)
def test_conformance_skips_cleanly_without_a_backend(monkeypatch, modules, reason) -> None:
    # The modules are replaced, so these run everywhere: CPU reporting tests
    # must never start a GPU sweep on a CUDA-equipped host. A skip must be
    # falsy, not an exception, and must say why.
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    report = silkern.conformance()
    assert report.skipped == reason
    assert not report.ok
    assert not report
    assert report.summary() == f"conformance skipped: {reason}"


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
        {"tile_size": 256}, {"compact": False}, {"block_size": 2**20}, {"dcp_size": 2**24},
        {"compact_valid_to_front": 0}, {"compact_valid_to_front": None},
        {"num_warps": 16}, {"num_warps": True}, {"num_warps": 8.0},
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


@pytest.mark.parametrize("key", GEOMETRY_KEYS)
def test_geometry_missing_a_launch_parameter_is_named(key: str) -> None:
    geometry = {name: value for name, value in DEFAULT_MATRIX[0].items() if name != key}
    with pytest.raises(silkern.LocalizationError, match=f"missing {key}$"):
        _check_geometry(geometry, batch=1, tile_size=128)


@pytest.mark.parametrize(
    "extra, options",
    [
        ({}, {"compact_valid_to_front": True}),
        ({"compact_valid_to_front": False}, {"compact_valid_to_front": False}),
        ({"num_warps": 4}, {"compact_valid_to_front": True, "num_warps": 4}),
        ({"compact_valid_to_front": True, "num_warps": 8},
         {"compact_valid_to_front": True, "num_warps": 8}),
    ],
)
def test_geometry_accepts_the_optional_launch_keys(extra, options) -> None:
    geometry = {"width": 513, "block_size": 32, "dcp_size": 4, "dcp_rank": 2,
                "dcp_interleave": 2, **extra}
    width, launch = _check_geometry(geometry, batch=3, tile_size=128)
    assert width == 513
    # num_warps is passed on only when set, so each launcher keeps its default.
    assert launch == {"block_size": 32, "dcp_size": 4, "dcp_rank": 2, "dcp_interleave": 2,
                      **options}


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
    with pytest.raises(SystemExit) as exit_info:
        verify.main(["--batch", batch])
    assert exit_info.value.code == USAGE_ERROR


def test_cli_passes_a_matrix_file_to_the_sweep(monkeypatch, tmp_path, capsys) -> None:
    matrix = [
        {"width": 2048, "block_size": 32, "dcp_size": 2, "dcp_rank": 1, "dcp_interleave": 1},
        {"width": 129, "block_size": 64, "dcp_size": 4, "dcp_rank": 0, "dcp_interleave": 1,
         "compact_valid_to_front": False, "num_warps": 4},
    ]
    path = tmp_path / "matrix.json"
    path.write_text(json.dumps(matrix))
    seen = {}

    def sweep(**kwargs):
        seen.update(kwargs)
        return ConformanceReport(skipped="test backend missing")

    monkeypatch.setattr(verify, "conformance", sweep)
    assert verify.main(["--matrix", str(path)]) == 0
    assert seen["matrix"] == matrix
    assert "conformance skipped" in capsys.readouterr().out


def test_cli_without_a_matrix_file_sweeps_the_default_matrix(monkeypatch) -> None:
    seen = {}
    monkeypatch.setattr(verify, "conformance", lambda **kwargs: (
        seen.update(kwargs) or ConformanceReport(skipped="test backend missing")
    ))
    assert verify.main([]) == 0
    assert seen["matrix"] is None


@pytest.mark.parametrize(
    "content, message",
    [
        (None, "cannot read --matrix"),
        ("{not json", "cannot read --matrix"),
        (json.dumps({"width": 1}), "nonempty JSON list"),
        (json.dumps([]), "nonempty JSON list"),
        (json.dumps([DEFAULT_MATRIX[0], 7]), "entry 1: matrix geometry must be a mapping"),
        (json.dumps([{**DEFAULT_MATRIX[0], "tile_size": 64}]), "entry 0: .*unknown keys tile_size"),
        (json.dumps([{**DEFAULT_MATRIX[0], "dcp_rank": 1}]), "entry 0: invalid dcp configuration"),
        (json.dumps([{**DEFAULT_MATRIX[0], "num_warps": 2}]), "entry 0: num_warps must be 4 or 8"),
        (json.dumps([{**DEFAULT_MATRIX[0], "compact_valid_to_front": "no"}]),
         "entry 0: compact_valid_to_front must be a bool"),
    ],
    ids=["missing-file", "not-json", "object", "empty", "non-mapping-entry",
         "unknown-key", "bad-rank", "bad-num-warps", "bad-compaction-flag"],
)
def test_cli_rejects_a_bad_matrix_file_before_any_backend(
    monkeypatch, tmp_path, capsys, content, message
) -> None:
    # A typo must fail on a machine with no backend, where the sweep would skip.
    monkeypatch.setitem(sys.modules, "torch", None)
    path = tmp_path / "matrix.json"
    if content is not None:
        path.write_text(content)
    with pytest.raises(SystemExit) as exit_info:
        verify.main(["--matrix", str(path)])
    assert exit_info.value.code == USAGE_ERROR
    assert re.search(message, capsys.readouterr().err)


@pytest.mark.parametrize("argv", [["--batch", "0"], ["--arm", "atomic"], ["--no-such-flag"]])
def test_cli_usage_errors_are_distinct_from_an_unavailable_device(monkeypatch, argv) -> None:
    # Exit 2 means "no device" under --require-device; a mistyped command must not.
    monkeypatch.setitem(sys.modules, "torch", None)
    assert verify.main(["--require-device"]) == 2
    with pytest.raises(SystemExit) as exit_info:
        verify.main(["--require-device", *argv])
    assert exit_info.value.code == USAGE_ERROR


def test_default_matrix_is_well_formed_and_crosses_boundaries() -> None:
    for geometry in DEFAULT_MATRIX:
        assert set(GEOMETRY_KEYS) <= set(geometry) <= set(GEOMETRY_KEYS + OPTIONAL_GEOMETRY_KEYS)
        _check_geometry(geometry, batch=5, tile_size=DEFAULT_TILE_SIZE)
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
    assert len({json.dumps(g, sort_keys=True) for g in DEFAULT_MATRIX}) == len(DEFAULT_MATRIX)


def test_default_matrix_keeps_columns_across_ranks() -> None:
    """The column-preserving layout must be checked where ownership drops tokens."""
    column_preserving = [
        g for g in DEFAULT_MATRIX if g.get("compact_valid_to_front", True) is False
    ]
    assert any(g["dcp_size"] > 1 for g in column_preserving)
    assert any(g["dcp_size"] > 1 and g["dcp_interleave"] > 1 for g in column_preserving)


def test_default_matrix_covers_the_vllm_qualified_surface() -> None:
    """A bare ``python -m silkern`` sweeps each qualified vLLM geometry (one rank apiece)."""
    # The adapter's BLOCK_N is the hierarchical tile; the sweep's default must match.
    assert DEFAULT_TILE_SIZE == vllm.QUALIFIED_BLOCK_N
    swept = {
        (g["width"], g["block_size"], g["dcp_size"], g["dcp_interleave"])
        for g in DEFAULT_MATRIX
        if g.get("compact_valid_to_front", True) and "num_warps" not in g
    }
    for width, block_size, dcp_size, interleave in itertools.product(
        vllm.QUALIFIED_TOPK_WIDTHS, vllm.QUALIFIED_BLOCK_SIZES,
        vllm.QUALIFIED_DCP_SIZES, vllm.QUALIFIED_INTERLEAVES,
    ):
        assert (width, block_size, dcp_size, interleave) in swept


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
